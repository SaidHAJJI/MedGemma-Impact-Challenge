import streamlit as st
import io
import speech_recognition as sr
from streamlit_mic_recorder import mic_recorder
from datetime import datetime
import google.generativeai as genai
import os

# --- Configuration ---
# On utilise Gemini 1.5 Flash pour sa vitesse et son faible coût, idéal pour du triage
MODEL_NAME = "gemini-1.5-flash"

# --- System Prompt (V3 - SafetyFirst) ---
# Mise à jour vers la version V3 qui est plus robuste pour la sécurité
SYSTEM_PROMPT = """You are MedGemma, a helpful medical triage assistant.
CRITICAL: If symptoms suggest a life-threatening emergency (heart attack, stroke, severe bleeding, difficulty breathing), IMMEDIATELY tell the user to call emergency services (911/112).
For non-emergencies:
1. Analyze the symptoms provided.
2. Estimate urgency (Low, Medium, High).
3. Suggest immediate home care actions.
4. Always advise seeing a doctor.
Keep responses concise, structured, and empathetic."""

def configure_genai():
    """Configure l'API Google Generative AI via st.secrets ou variables d'env."""
    api_key = None
    
    # 1. Vérifier dans les secrets Streamlit (.streamlit/secrets.toml)
    if "GEMINI_API_KEY" in st.secrets:
        api_key = st.secrets["GEMINI_API_KEY"]
    
    # 2. Vérifier dans les variables d'environnement (backup)
    if not api_key:
        api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        return False
    
    genai.configure(api_key=api_key)
    return True

def query_gemini(prompt):
    """Envoie la requête à l'API Gemini."""
    try:
        model = genai.GenerativeModel(
            model_name=MODEL_NAME,
            system_instruction=SYSTEM_PROMPT
        )
        
        # Configuration de la génération pour être concis
        generation_config = genai.types.GenerationConfig(
            candidate_count=1,
            max_output_tokens=500,
            temperature=0.4, # Assez bas pour rester factuel
        )

        response = model.generate_content(prompt, generation_config=generation_config)
        return response.text
    except Exception as e:
        return f"Erreur lors de l'appel à Gemini API: {str(e)}"

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

# Configuration de l'API au démarrage
api_ready = configure_genai()

# Header
st.title("🏥 MedGemma Triage Assistant")
st.markdown("**Prototype Hackathon - Version Cloud (Gemini API)**")
st.markdown("---")

# Sidebar - Context & Privacy
with st.sidebar:
    st.header("ℹ️ À propos")
    
    if api_ready:
        st.success("✅ API Connectée")
    else:
        st.error("❌ API Key manquante")
        st.markdown("""
        Créez un fichier `.streamlit/secrets.toml` :
        ```toml
        GEMINI_API_KEY = "votre_cle_ici"
        ```
        """)

    st.info(
        f"""
        Modèle actif : **{MODEL_NAME}**
        System Prompt : **V3 SafetyFirst**
        """
    )
    st.warning(
        """
        **DISCLAIMER MÉDICAL**
        Ceci est un prototype de démonstration.
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
    analyze_btn = st.button("🔍 Analyser", type="primary", disabled=not api_ready)

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
        st.success(f"Analyse terminée en {duration:.2f} secondes.")
        
        st.markdown("### 📋 Rapport de Triage")
        container = st.container(border=True)
        container.markdown(response)
        
        st.caption("Généré par Gemini 1.5 Flash. Vérifiez toujours avec un professionnel.")

# Footer
st.markdown("---")
if not api_ready:
    st.warning("Veuillez configurer votre clé API pour utiliser l'application.")