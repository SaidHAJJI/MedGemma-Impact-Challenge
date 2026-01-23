import streamlit as st
import requests
import json
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
        return f"Error connecting to Ollama: {e}. Make sure Ollama is running (`ollama serve`)."

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
        Cette application utilise **Google Gemma (2B)** via Ollama pour analyser les symptômes.
        
        🔒 **Privacy-First** : 
        Aucune donnée ne quitte votre ordinateur. Tout le traitement est local.
        """
    )
    st.warning(
        """
        **DISCLAIMER MÉDICAL**
        Ceci est une démonstration technique. 
        Ne pas utiliser pour de vraies urgences médicales. 
        En cas d'urgence, appelez le 15 ou le 112.
        """
    )
    
    st.write(f"Modèle actif : `{MODEL_NAME}`")

# Main Input
st.subheader("Description des symptômes")
symptoms = st.text_area(
    "Décrivez ce que vous ressentez (ex: 'Douleur poitrine bras gauche', 'Fièvre 39C enfant')...",
    height=150,
    placeholder="Ex: J'ai mal à la tête et la lumière me gêne..."
)

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
