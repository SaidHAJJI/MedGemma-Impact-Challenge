import streamlit as st
import io
import speech_recognition as sr
from streamlit_mic_recorder import mic_recorder
from datetime import datetime
from google import genai
from google.genai import types
import os
import requests
import json
from pydub import AudioSegment
import shutil
from fpdf import FPDF

# Initialisation de ffmpeg pour pydub (seulement si non présent dans le système)
if not shutil.which("ffmpeg"):
    try:
        import static_ffmpeg
        static_ffmpeg.add_paths()
    except Exception as e:
        print(f"Note: static_ffmpeg n'a pas pu être initialisé ({e}). Assurez-vous que ffmpeg est installé manuellement.")

# --- Configuration ---
# Utilisation du modèle Gemini 2.0 Flash par défaut pour l'API officielle
MODEL_NAME = "gemini-2.0-flash"

# --- PDF Generation Class ---
class MedGemmaPDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 15)
        self.cell(0, 10, 'Rapport de Triage MedGemma', 0, 1, 'C')
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')
        self.cell(0, 10, 'Généré par MedGemma - Prototype IA', 0, 0, 'R')

def create_pdf(report_text, patient_data):
    """Génère un PDF simple avec le rapport."""
    pdf = MedGemmaPDF()
    pdf.add_page()
    
    # Méta-données
    pdf.set_font("Arial", size=10)
    pdf.cell(0, 5, f"Date: {datetime.now().strftime('%d/%m/%Y %H:%M')}", 0, 1)
    pdf.ln(5)
    
    # Info Patient
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 8, "Informations Patient", 0, 1)
    pdf.set_font("Arial", size=11)
    pdf.cell(0, 6, f"Age: {patient_data.get('age')} ans", 0, 1)
    pdf.cell(0, 6, f"Sexe: {patient_data.get('sexe')}", 0, 1)
    
    # Terrain & Antécédents
    history = patient_data.get('history', [])
    if history:
        pdf.multi_cell(0, 6, f"Antecedents: {', '.join(history)}")
    
    meds = patient_data.get('meds', '')
    if meds:
        pdf.multi_cell(0, 6, f"Traitements/Allergies: {meds}")
    
    # Symptomes
    symptoms_list = ", ".join(patient_data.get('symptoms', []))
    pdf.multi_cell(0, 6, f"Symptomes: {symptoms_list}")
    if patient_data.get('description'):
         pdf.multi_cell(0, 6, f"Description: {patient_data.get('description')}")
    pdf.ln(5)
    
    # Rapport LLM
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 8, "Analyse & Recommandations", 0, 1)
    pdf.set_font("Arial", size=11)
    
    # Nettoyage basique des caractères non supportés par latin-1
    safe_text = report_text.encode('latin-1', 'replace').decode('latin-1')
    pdf.multi_cell(0, 6, safe_text)
    
    # Disclaimer
    pdf.ln(10)
    pdf.set_font("Arial", 'I', 9)
    pdf.set_text_color(200, 0, 0)
    pdf.multi_cell(0, 5, "AVERTISSEMENT: Ce rapport est généré par une IA (Gemma 2b / Gemini). Il ne remplace pas un avis médical professionnel. En cas d'urgence, appelez le 15 ou le 112.")
    
    return pdf.output(dest='S').encode('latin-1')

# --- System Prompts (Optimized V3 SafetyFirst) ---
SYSTEM_PROMPT_QUESTIONS = """You are MedGemma, a medical triage expert. 
Based on the symptoms provided, generate 3-4 essential follow-up questions in JSON format.
Each question must have:
1. "text": The question in French.
2. "type": "binary" (for Oui/Non) or "numeric" (for values like temperature, days, score 1-10, etc.).
3. "unit": A short label for the unit (e.g., "°C", "jours", "/10") if numeric, else null.

Focus strictly on differentiating between benign issues and potential emergencies.
Example format:
[
  {"text": "Avez-vous une douleur thoracique ?", "type": "binary", "unit": null},
  {"text": "Quelle est votre température ?", "type": "numeric", "unit": "°C"},
  {"text": "Depuis combien de jours ?", "type": "numeric", "unit": "jours"}
]
Return ONLY the JSON array."""

SYSTEM_PROMPT_FINAL = """You are MedGemma, a helpful medical triage assistant.
CRITICAL: If symptoms suggest a life-threatening emergency (e.g., heart attack signs, stroke, severe bleeding, breathing difficulty), IMMEDIATELY tell the user to call emergency services (15/112) in the first line.

For non-emergencies, analyze the context and provide:
1. Urgency Level (Low, Medium, High).
2. Possible causes (stated with caution as possibilities, not diagnosis).
3. Immediate home care actions.
4. Recommendation on when to see a doctor (e.g., "Within 4 hours", "Tomorrow").

Keep responses concise, structured, and empathetic. Answer in French."""

def get_api_key():
    """Récupère la clé API depuis les secrets ou l'environnement."""
    if "GEMINI_API_KEY" in st.secrets:
        return st.secrets["GEMINI_API_KEY"]
    return os.getenv("GEMINI_API_KEY")

def query_llm(prompt, system_instruction, backend="Gemini API", custom_url=None):
    """Envoie la requête au LLM choisi (Gemini API ou Kaggle/Custom)."""
    
    # --- OPTION 1: KAGGLE / CUSTOM URL ---
    if backend == "Kaggle / Local URL":
        if not custom_url:
            return "Erreur : URL du serveur manquante."
        
        endpoint = f"{custom_url.rstrip('/')}/generate"
        payload = {
            "prompt": prompt,
            "system_instruction": system_instruction
        }
        try:
            response = requests.post(endpoint, json=payload, timeout=60)
            if response.status_code == 200:
                return response.json().get("response", "Erreur: Réponse vide.")
            else:
                return f"Erreur Serveur ({response.status_code}): {response.text}"
        except requests.exceptions.RequestException as e:
            return f"Erreur de connexion au serveur Kaggle : {str(e)}"

    # --- OPTION 2: GOOGLE GEMINI API ---
    else:
        api_key = get_api_key()
        if not api_key:
            return "Erreur : Clé API manquante."

        try:
            client = genai.Client(api_key=api_key)
            config = types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.4,
                max_output_tokens=800,
            )
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt,
                config=config
            )
            # Nettoyage si le modèle entoure le JSON de ```json ... ```
            text = response.text.strip()
            if text.startswith("```json"):
                text = text.split("```json")[1].split("```")[0].strip()
            elif text.startswith("```"):
                text = text.split("```")[1].split("```")[0].strip()
            return text
        except Exception as e:
            return f"Erreur API Gemini : {str(e)}"

def transcribe_audio(audio_bytes, backend="Gemini API", custom_url=None):
    """Convertit l'audio en WAV puis transcrit via Kaggle ou Google."""
    
    try:
        # --- CONVERSION UNIVERSELLE EN WAV ---
        # On utilise pydub pour lire n'importe quel format (WebM, AAC, etc.)
        audio_segment = AudioSegment.from_file(io.BytesIO(audio_bytes))
        
        # On exporte en WAV (format PCM standard)
        wav_io = io.BytesIO()
        audio_segment.export(wav_io, format="wav")
        wav_io.seek(0)
        wav_bytes = wav_io.read()
        
        # --- OPTION KAGGLE (MedASR) ---
        if backend == "Kaggle / Local URL" and custom_url:
            endpoint = f"{custom_url.rstrip('/')}/transcribe"
            files = {'audio': ('audio.wav', wav_bytes, 'audio/wav')}
            response = requests.post(endpoint, files=files, timeout=30)
            if response.status_code == 200:
                return response.json().get("transcription")
            else:
                st.error(f"Erreur MedASR ({response.status_code}): {response.text}")
                return None

        # --- OPTION STANDARD (Google Speech Recognition) ---
        recognizer = sr.Recognizer()
        wav_io.seek(0)
        with sr.AudioFile(wav_io) as source:
            audio_data = recognizer.record(source)
            try:
                return recognizer.recognize_google(audio_data, language="fr-FR")
            except:
                return None
            
    except Exception as e:
        st.error(f"Erreur de traitement audio : {e}")
        return None

# --- Interface Streamlit ---
st.set_page_config(page_title="MedGemma Triage", page_icon="🏥")

# Vérification de la clé au démarrage
api_key_present = get_api_key() is not None

# Sidebar - Context & Privacy
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Choix du Backend
    backend_option = st.radio(
        "Source du Modèle :",
        ("Gemini API", "Kaggle / Local URL")
    )
    
    custom_url = None
    if backend_option == "Kaggle / Local URL":
        custom_url = st.text_input("URL ngrok (Kaggle) :", placeholder="https://xxxx.ngrok-free.app")
        if not custom_url:
            st.warning("⚠️ Collez l'URL ngrok ici")
    
    st.divider() 
    
    st.header("ℹ️ À propos")
    if backend_option == "Gemini API":
        if api_key_present:
            st.success("✅ Clé API Gemini détectée")
        else:
            st.error("❌ Clé API manquante")
        st.info(f"Modèle : `{MODEL_NAME}`")
    else:
        st.info("Mode : Serveur Distant (Kaggle)")
        
    st.warning("**DISCLAIMER MÉDICAL**\nCeci est un prototype. Ne pas utiliser pour de vraies urgences.")

# Init Session State
if 'step' not in st.session_state:
    st.session_state.step = 1
if 'initial_data' not in st.session_state:
    st.session_state.initial_data = {}
if 'followup_questions' not in st.session_state:
    st.session_state.followup_questions = ""
if 'selected_symptoms' not in st.session_state:
    st.session_state.selected_symptoms = set()
if 'symptoms_input' not in st.session_state:
    st.session_state.symptoms_input = ""
if 'medical_history' not in st.session_state:
    st.session_state.medical_history = []

st.title("🏥 MedGemma Triage")
st.markdown("---")

# --- STEP 1: INITIAL INPUT ---
if st.session_state.step == 1:
    st.subheader("1. Profil Patient")
    col_age, col_sex = st.columns(2)
    with col_age:
        age = st.number_input("Âge", min_value=0, max_value=120, value=30)
    with col_sex:
        sexe = st.selectbox("Sexe", ["Masculin", "Féminin", "Autre"])

    st.subheader("2. Antécédents & Terrain")
    col_hist1, col_hist2 = st.columns(2)
    
    with col_hist1:
        diabetes = st.checkbox("Diabète")
        hypertension = st.checkbox("Hypertension")
        heart_disease = st.checkbox("Maladie Cardiaque")
    with col_hist2:
        asthma = st.checkbox("Asthme / BPCO")
        immunosup = st.checkbox("Immunodépression")
        smoker = st.checkbox("Fumeur")
    
    if sexe == "Féminin":
        pregnant = st.checkbox("Grossesse en cours")
    else:
        pregnant = False

    meds_allergies = st.text_input("Traitements actuels ou allergies connues :", placeholder="Ex: Doliprane, allergie à la pénicilline...")

    st.subheader("3. Symptômes")
    
    # Predefined Symptoms Selection
    PREDEFINED_SYMPTOMS = [
        "Fièvre", "Maux de tête", "Toux", "Maux de gorge", "Essoufflement",
        "Fatigue", "Douleurs musculaires", "Nausées", "Vomissements", "Diarrhée",
        "Douleur thoracique", "Vertiges"
    ]

    st.write("Sélectionnez les symptômes présents :")
    cols = st.columns(4)
    for i, symptom in enumerate(PREDEFINED_SYMPTOMS):
        col = cols[i % 4]
        is_selected = symptom in st.session_state.selected_symptoms
        if col.button(f"{'' if is_selected else ''}{symptom}", key=f"btn_{symptom}", use_container_width=True, type="primary" if is_selected else "secondary"):
            if is_selected: st.session_state.selected_symptoms.remove(symptom)
            else: st.session_state.selected_symptoms.add(symptom)
            st.rerun()

    # Voice/Text Input
    audio = mic_recorder(start_prompt="🎤 Parler", stop_prompt="⏹️ Arrêter", key='recorder')
    if audio:
        transcribed = transcribe_audio(audio['bytes'], backend=backend_option, custom_url=custom_url)
        if transcribed: st.session_state.symptoms_input = transcribed

    symptoms_text = st.text_area("Description libre :", value=st.session_state.symptoms_input, height=100)

    if st.button("Suivant ➡️", type="primary"):
        if not symptoms_text.strip() and not st.session_state.selected_symptoms:
            st.error("Précisez vos symptômes.")
        else:
            # Compiler les antécédents
            history = []
            if diabetes: history.append("Diabétique")
            if hypertension: history.append("Hypertendu")
            if heart_disease: history.append("Maladie cardiaque")
            if asthma: history.append("Asthme/BPCO")
            if immunosup: history.append("Immunodéprimé")
            if smoker: history.append("Fumeur")
            if pregnant: history.append("Enceinte")

            st.session_state.initial_data = {
                "age": age, 
                "sexe": sexe, 
                "history": history,
                "meds": meds_allergies,
                "symptoms": list(st.session_state.selected_symptoms),
                "description": symptoms_text
            }
            with st.spinner("Analyse initiale..."):
                prompt = f"""Patient: {age} ans, {sexe}. 
                Terrain: {', '.join(history) if history else 'Aucun antécédent majeur'}.
                Traitements/Allergies: {meds_allergies}.
                Symptômes: {st.session_state.selected_symptoms}. 
                Description: {symptoms_text}"""
                st.session_state.followup_questions = query_llm(prompt, SYSTEM_PROMPT_QUESTIONS, backend=backend_option, custom_url=custom_url)
                st.session_state.step = 2
            st.rerun()

# --- STEP 2: FOLLOW-UP QUESTIONS ---
elif st.session_state.step == 2:
    st.subheader("🔍 Précisions nécessaires")
    st.info("Pour affiner le triage, veuillez répondre à ces questions :")
    
    questions_data = []
    try:
        # Tentative de parsing du JSON
        questions_data = json.loads(st.session_state.followup_questions)
    except:
        # Fallback si le format n'est pas du JSON
        st.markdown(st.session_state.followup_questions)
        answers = st.text_area("Vos réponses :", height=150)
    
    user_responses = {}
    
    if questions_data:
        for i, q in enumerate(questions_data):
            st.write(f"**Q{i+1}: {q['text']}**")
            if q['type'] == 'binary':
                user_responses[q['text']] = st.radio(f"Réponse Q{i+1}", ["Non", "Oui"], key=f"q_{i}", label_visibility="collapsed")
            elif q['type'] == 'numeric':
                unit = q.get('unit', '')
                user_responses[q['text']] = st.number_input(f"Valeur ({unit})", key=f"q_{i}", step=0.5 if unit == "°C" else 1.0)
            st.divider()

    col_back, col_next = st.columns([1, 1])
    with col_back:
        if st.button("⬅️ Retour"):
            st.session_state.step = 1
            st.rerun()
    with col_next:
        if st.button("Obtenir le rapport final 🔍", type="primary"):
            with st.spinner("Génération du rapport de triage..."):
                # Formatage des réponses pour le prompt final
                if questions_data:
                    formatted_answers = "\n".join([f"- {k}: {v}" for k, v in user_responses.items()])
                else:
                    formatted_answers = answers
                
                final_prompt = f"""
                CONTEXTE:
                - Patient: {st.session_state.initial_data['age']} ans, {st.session_state.initial_data['sexe']}
                - Terrain: {', '.join(st.session_state.initial_data['history'])}
                - Traitements/Allergies: {st.session_state.initial_data['meds']}
                - Symptômes: {st.session_state.initial_data['symptoms']}
                - Description: {st.session_state.initial_data['description']}
                
                RÉPONSES AUX QUESTIONS DE PRÉCISION:
                {formatted_answers}
                """
                st.session_state.final_report = query_llm(final_prompt, SYSTEM_PROMPT_FINAL, backend=backend_option, custom_url=custom_url)
                st.session_state.step = 3
            st.rerun()

# --- STEP 3: FINAL REPORT ---
elif st.session_state.step == 3:
    st.success("✅ Analyse de triage terminée")
    st.markdown(st.session_state.final_report)
    
    col_dl, col_new = st.columns([1, 1])
    
    with col_dl:
        # PDF Generation
        try:
            pdf_bytes = create_pdf(st.session_state.final_report, st.session_state.initial_data)
            st.download_button(
                label="📄 Télécharger le Rapport (PDF)",
                data=pdf_bytes,
                file_name=f"Rapport_MedGemma_{datetime.now().strftime('%Y%m%d')}.pdf",
                mime="application/pdf"
            )
        except Exception as e:
            st.error(f"Erreur lors de la génération du PDF: {e}")

    with col_new:
        if st.button("🔄 Nouvelle analyse"):
            st.session_state.step = 1
            st.session_state.selected_symptoms = set()
            st.session_state.symptoms_input = ""
            st.rerun()


# Footer
st.markdown("---")
if not api_key_present:
    st.warning("Configuration requise : Ajoutez votre clé API dans .streamlit/secrets.toml")