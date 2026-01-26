import streamlit as st
import io
import speech_recognition as sr
from streamlit_mic_recorder import mic_recorder
from datetime import datetime
from google import genai
from google.genai import types
import os

# --- Configuration ---
# Utilisation du modèle Gemini 2.0 Flash
MODEL_NAME = "gemini-2.0-flash"

# --- System Prompts ---
SYSTEM_PROMPT_QUESTIONS = """You are MedGemma, a medical triage expert. 
Based on the symptoms provided, generate 3-4 essential follow-up questions to better assess the urgency.
Keep questions brief and focused on red flags or duration. 
Format as a simple bulleted list in French."""

SYSTEM_PROMPT_FINAL = """You are MedGemma, a helpful medical triage assistant.
CRITICAL: If symptoms suggest a life-threatening emergency, IMMEDIATELY tell the user to call emergency services (911/112).
Analyze the initial symptoms, patient context, and answers to follow-up questions.
1. Urgency Level (Low, Medium, High).
2. Possible causes (with caution).
3. Immediate home care actions.
4. Always advise seeing a doctor.
Keep responses concise, structured, and empathetic."""

def get_api_key():
    """Récupère la clé API depuis les secrets ou l'environnement."""
    if "GEMINI_API_KEY" in st.secrets:
        return st.secrets["GEMINI_API_KEY"]
    return os.getenv("GEMINI_API_KEY")

def query_gemini(prompt, system_instruction):
    """Envoie la requête à l'API Gemini."""
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
        return f"Erreur API : {str(e)}"

# --- Interface Streamlit ---
st.set_page_config(page_title="MedGemma Triage", page_icon="🏥")

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
        from app import transcribe_audio # Assuming it's in the same file or accessible
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
                st.session_state.followup_questions = query_gemini(prompt, SYSTEM_PROMPT_QUESTIONS)
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
                st.session_state.final_report = query_gemini(final_prompt, SYSTEM_PROMPT_FINAL)
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
