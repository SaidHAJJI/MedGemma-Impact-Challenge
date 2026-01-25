import streamlit as st
import io
import speech_recognition as sr
from streamlit_mic_recorder import mic_recorder
from datetime import datetime
from google import genai
from google.genai import types
import os

# --- Configuration ---
# Utilisation du modèle Gemini 1.5 Flash (Plus stable pour le Free Tier)
MODEL_NAME = "gemini-1.5-flash"

# --- System Prompt (V3 - SafetyFirst) ---
SYSTEM_PROMPT = """You are MedGemma, a helpful medical triage assistant.
CRITICAL: If symptoms suggest a life-threatening emergency (heart attack, stroke, severe bleeding, difficulty breathing), IMMEDIATELY tell the user to call emergency services (911/112).
For non-emergencies:
1. Analyze the symptoms provided.
2. Estimate urgency (Low, Medium, High).
3. Suggest immediate home care actions.
4. Always advise seeing a doctor.
Keep responses concise, structured, and empathetic."""

def get_api_key():
    """Récupère la clé API depuis les secrets ou l'environnement."""
    if "GEMINI_API_KEY" in st.secrets:
        return st.secrets["GEMINI_API_KEY"]
    return os.getenv("GEMINI_API_KEY")

def query_gemini(prompt):
    """Envoie la requête à l'API Gemini via le nouveau SDK google-genai."""
    api_key = get_api_key()
    if not api_key:
        return "Erreur : Clé API manquante. Veuillez configurer .streamlit/secrets.toml."

    try:
        # Initialisation du client avec la nouvelle librairie
        client = genai.Client(api_key=api_key)
        
        # Configuration de la génération
        config = types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0.4,
            max_output_tokens=500,
            candidate_count=1
        )

        # Appel au modèle
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
            config=config
        )
        return response.text
        
    except Exception as e:
        return f"Erreur lors de l'appel à Gemini API : {str(e)}"

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
st.set_page_config(
    page_title="MedGemma Triage (Cloud API)",
    page_icon="🏥",
    layout="centered"
)

# Vérification de la clé au démarrage
api_key_present = get_api_key() is not None

# Header
st.title("🏥 MedGemma Triage Assistant")
st.markdown(f"**Prototype Hackathon - Version Cloud ({MODEL_NAME})**")
st.markdown("---")

# Sidebar - Context & Privacy
with st.sidebar:
    st.header("ℹ️ À propos")
    
    if api_key_present:
        st.success("✅ Clé API détectée")
    else:
        st.error("❌ Clé API manquante")
        st.markdown("""
        Vérifiez `secrets.toml`.
        """)

    st.info(
        f"""
        Lib : `google-genai` (New SDK)
        Modèle : `{MODEL_NAME}`
        Prompt : `V3 SafetyFirst`
        """
    )
    st.warning(
        """
        **DISCLAIMER MÉDICAL**
        Ceci est un prototype.
        Ne pas utiliser pour de vraies urgences. 
        """
    )

# Init Session State
if 'symptoms_input' not in st.session_state:
    st.session_state.symptoms_input = ""

# Main Input
st.subheader("Description des symptômes")

# Voice Input (Browser-based)
st.write("🎤 Enregistrez vos symptômes :")
audio = mic_recorder(
    start_prompt="Démarrer l'enregistrement",
    stop_prompt="Arrêter",
    key='recorder'
)

if audio:
    transcribed = transcribe_audio(audio['bytes'])
    if transcribed:
        st.session_state.symptoms_input = transcribed

symptoms = st.text_area(
    "Décrivez ce que vous ressentez...",
    value=st.session_state.symptoms_input,
    height=150,
    key="text_input_area"
)

col1, col2 = st.columns([1, 4])
with col1:
    analyze_btn = st.button("🔍 Analyser", type="primary", disabled=not api_key_present)

# Analysis Logic
if analyze_btn:
    if not symptoms.strip():
        st.error("Veuillez entrer une description des symptômes.")
    else:
        with st.spinner(f"Analyse avec {MODEL_NAME}..."):
            start_time = datetime.now()
            response = query_gemini(symptoms)
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
        # Display Results
        if "Erreur" in response:
             st.error(response)
        else:
            st.success(f"Analyse terminée en {duration:.2f} secondes.")
            st.markdown("### 📋 Rapport de Triage")
            container = st.container(border=True)
            container.markdown(response)
            st.caption("Généré par Gemini. Vérifiez toujours avec un professionnel.")

# Footer
st.markdown("---")
if not api_key_present:
    st.warning("Configuration requise : Ajoutez votre clé API dans .streamlit/secrets.toml")
