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

# --- Configuration ---
# Utilisation du modèle Gemini 2.0 Flash par défaut pour l'API officielle
MODEL_NAME = "gemini-2.0-flash"

# --- System Prompts (Optimized V3 SafetyFirst) ---
SYSTEM_PROMPT_QUESTIONS = """You are MedGemma, a medical triage expert. 
Based on the symptoms provided, generate 3-4 essential follow-up questions to better assess the urgency.
Focus strictly on differentiating between benign issues and potential emergencies.
Format as a simple bulleted list in French."""

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
                max_output_tokens=600,
            )
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt,
                config=config
            )
            return response.text
        except Exception as e:
            return f"Erreur API Gemini : {str(e)}"

def transcribe_audio(audio_bytes):
    """Transcrit les octets audio reçus du navigateur."""
    recognizer = sr.Recognizer()
    audio_file = io.BytesIO(audio_bytes)
    with sr.AudioFile(audio_file) as source:
        audio_data = recognizer.record(source)
        try:
            return recognizer.recognize_google(audio_data, language="fr-FR")
        except:
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

st.title("🏥 MedGemma Triage")
st.markdown("---")

# --- STEP 1: INITIAL INPUT ---
if st.session_state.step == 1:
    st.subheader("1. Informations de base")
    col_age, col_sex = st.columns(2)
    with col_age:
        age = st.number_input("Âge", min_value=0, max_value=120, value=30)
    with col_sex:
        sexe = st.selectbox("Sexe", ["Masculin", "Féminin", "Autre"])

    st.subheader("2. Symptômes")
    
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
        if col.button(f"{'✅ ' if is_selected else ''}{symptom}", key=f"btn_{symptom}", use_container_width=True, type="primary" if is_selected else "secondary"):
            if is_selected: st.session_state.selected_symptoms.remove(symptom)
            else: st.session_state.selected_symptoms.add(symptom)
            st.rerun()

    # Voice/Text Input
    audio = mic_recorder(start_prompt="🎤 Parler", stop_prompt="⏹️ Arrêter", key='recorder')
    if audio:
        transcribed = transcribe_audio(audio['bytes'])
        if transcribed: st.session_state.symptoms_input = transcribed

    symptoms_text = st.text_area("Description libre :", value=st.session_state.symptoms_input, height=100)

    if st.button("Suivant ➡️", type="primary"):
        if not symptoms_text.strip() and not st.session_state.selected_symptoms:
            st.error("Précisez vos symptômes.")
        else:
            st.session_state.initial_data = {
                "age": age, "sexe": sexe, 
                "symptoms": list(st.session_state.selected_symptoms),
                "description": symptoms_text
            }
            with st.spinner("Analyse initiale..."):
                prompt = f"Patient: {age} ans, {sexe}. Symptômes: {st.session_state.selected_symptoms}. Description: {symptoms_text}"
                st.session_state.followup_questions = query_llm(prompt, SYSTEM_PROMPT_QUESTIONS, backend=backend_option, custom_url=custom_url)
                st.session_state.step = 2
            st.rerun()

# --- STEP 2: FOLLOW-UP QUESTIONS ---
elif st.session_state.step == 2:
    st.subheader("🔍 Précisions nécessaires")
    st.info("Pour affiner le triage, veuillez répondre à ces questions :")
    st.markdown(st.session_state.followup_questions)
    
    answers = st.text_area("Vos réponses :", height=150, placeholder="Ex: La douleur dure depuis 2 jours, c'est apparu après manger...")
    
    col_back, col_next = st.columns([1, 1])
    with col_back:
        if st.button("⬅️ Retour"):
            st.session_state.step = 1
            st.rerun()
    with col_next:
        if st.button("Obtenir le rapport final 🔍", type="primary"):
            with st.spinner("Génération du rapport de triage..."):
                final_prompt = f"""
                CONTEXTE:
                - Patient: {st.session_state.initial_data['age']} ans, {st.session_state.initial_data['sexe']}
                - Symptômes: {st.session_state.initial_data['symptoms']}
                - Description: {st.session_state.initial_data['description']}
                
                PRÉCISIONS APPORTÉES:
                {answers}
                """
                st.session_state.final_report = query_llm(final_prompt, SYSTEM_PROMPT_FINAL, backend=backend_option, custom_url=custom_url)
                st.session_state.step = 3
            st.rerun()

# --- STEP 3: FINAL REPORT ---
elif st.session_state.step == 3:
    st.success("✅ Analyse de triage terminée")
    st.markdown(st.session_state.final_report)
    
    if st.button("🔄 Nouvelle analyse"):
        st.session_state.step = 1
        st.session_state.selected_symptoms = set()
        st.session_state.symptoms_input = ""
        st.rerun()


# Footer
st.markdown("---")
if not api_key_present:
    st.warning("Configuration requise : Ajoutez votre clé API dans .streamlit/secrets.toml")
