import streamlit as st
import requests
import json
import io
import speech_recognition as sr
from streamlit_mic_recorder import mic_recorder
from datetime import datetime

# --- Configuration ---
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "gemma:2b"

# --- System Prompt (V2 - Structured) ---
SYSTEM_PROMPT = """You are an AI Medical Assistant running locally.
Your goal is to provide preliminary triage advice.
1. Analyze the symptoms provided by the user.
2. Estimate urgency (Low, Medium, High, Emergency).
3. Suggest immediate actions (Home care vs Hospital).
4. Always advise seeing a doctor.
Keep responses concise and structured using Markdown."""

def query_ollama(prompt):
    """Envoie la requête au modèle local Ollama."""
    full_prompt = f"{SYSTEM_PROMPT}\n\nUser Symptoms: {prompt}\nAssistant:"
    
    payload = {
        "model": MODEL_NAME,
        "prompt": full_prompt,
        "stream": False
    }
    
    try:
        response = requests.post(OLLAMA_URL, json=payload)
        response.raise_for_status()
        return response.json()['response']
    except requests.exceptions.RequestException as e:
        return f"Error connecting to Ollama: {e}. Make sure Ollama is running locally and accessible via ngrok or local tunnel if using Streamlit Cloud."

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
    page_title="MedGemma Triage Prototype",
    page_icon="🏥",
    layout="centered"
)

# Header
st.title("🏥 MedGemma Triage Assistant")
st.markdown("**Prototype Hackathon - Privacy-First AI Triage**")
st.markdown("---")

# Sidebar - Context & Privacy
with st.sidebar:
    st.header("ℹ️ À propos")
    st.info(
        """
        Cette application utilise **Google Gemma (2B)** via Ollama.
        """
    )
    st.warning(
        """
        **DISCLAIMER MÉDICAL**
        Ne pas utiliser pour de vraies urgences médicales. 
        """
    )
    
    st.write(f"Modèle actif : `{MODEL_NAME}`")

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

# Note pour l'utilisateur Cloud
if "localhost" in OLLAMA_URL:
    st.warning("⚠️ L'application cloud tente de contacter `localhost`. Pour que cela fonctionne, vous devez exposer votre Ollama local (ex: via `ngrok`) ou héberger le modèle dans le cloud.")

col1, col2 = st.columns([1, 4])
with col1:
    analyze_btn = st.button("🔍 Analyser", type="primary")

# Analysis Logic
if analyze_btn:
    if not symptoms.strip():
        st.error("Veuillez entrer une description des symptômes.")
    else:
        with st.spinner("Analyse locale avec MedGemma en cours..."):
            # Simulation d'un petit délai ou appel réel
            start_time = datetime.now()
            response = query_ollama(symptoms)
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
        # Display Results
        st.success(f"Analyse terminée en {duration:.2f} secondes.")
        
        st.markdown("### 📋 Rapport de Triage")
        st.markdown("---")
        st.markdown(response)
        
        st.markdown("---")
        st.caption("Généré par MedGemma (Local Inference). Vérifiez toujours avec un professionnel.")

# Footer
st.markdown("---")
st.markdown("*The MedGemma Impact Challenge - Prototype v0.1* ")
