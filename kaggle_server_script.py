# === À COPIER DANS UNE CELLULE DE NOTEBOOK KAGGLE ===
# Assurez-vous d'avoir activé l'accélérateur GPU (T4 x2) dans les options du Notebook.

# 1. Installation des dépendances
!pip install -q -U flask flask-cors pyngrok accelerate transformers bitsandbytes librosa soundfile

# 2. Configuration & Import
import os
import torch
import librosa
import io
from flask import Flask, request, jsonify
from flask_cors import CORS
from pyngrok import ngrok
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoProcessor,
    AutoModelForCTC
)

# --- CONFIGURATION ---
# 1. Votre token ngrok (https://dashboard.ngrok.com/get-started/your-authtoken)
NGROK_AUTH_TOKEN = "VOTRE_TOKEN_NGROK_ICI" 

# 2. Votre token Hugging Face (https://huggingface.co/settings/tokens)
HF_TOKEN = "VOTRE_TOKEN_HF_ICI"

# Modèles à utiliser
LLM_MODEL_ID = "google/gemma-2b-it" 
ASR_MODEL_ID = "google/medasr"

# --- AUTHENTIFICATION & CHARGEMENT ---
from huggingface_hub import login
if HF_TOKEN != "VOTRE_TOKEN_HF_ICI":
    login(token=HF_TOKEN)

print("🔄 Chargement du modèle LLM (Gemma)...")
tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_ID)
llm_model = AutoModelForCausalLM.from_pretrained(
    LLM_MODEL_ID,
    device_map="auto",
    torch_dtype=torch.float16,
)

print("🔄 Chargement du modèle ASR (MedASR)...")
asr_processor = AutoProcessor.from_pretrained(ASR_MODEL_ID)
# Utilisation de AutoModelForCTC pour MedASR
asr_model = AutoModelForCTC.from_pretrained(
    ASR_MODEL_ID,
    device_map="auto",
    torch_dtype=torch.float16,
)
print("✅ Modèles chargés !")

# --- CRÉATION DE L'API ---
app = Flask(__name__)
CORS(app)

@app.route('/generate', methods=['POST'])
def generate():
    try:
        data = request.json
        user_prompt = data.get('prompt', '')
        system_instruction = data.get('system_instruction', '')
        
        full_prompt = f"<start_of_turn>user\n{system_instruction}\n\n{user_prompt}<end_of_turn>\n<start_of_turn>model\n"
        
        inputs = tokenizer(full_prompt, return_tensors="pt").to("cuda")
        
        outputs = llm_model.generate(
            **inputs, 
            max_new_tokens=500,
            temperature=0.4,
            do_sample=True
        )
        
        response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        final_response = response_text.split("model\n")[-1].strip()
        
        return jsonify({"response": final_response})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/transcribe', methods=['POST'])
def transcribe():
    try:
        if 'audio' not in request.files:
            return jsonify({"error": "Fichier audio manquant"}), 400
        
        audio_file = request.files['audio']
        audio_bytes = audio_file.read()
        
        # Charger et ré-échantillonner à 16kHz
        speech, sr = librosa.load(io.BytesIO(audio_bytes), sr=16000)
        
        # Prétraitement CTC
        inputs = asr_processor(speech, sampling_rate=16000, return_tensors="pt")
        input_values = inputs.input_values.to("cuda").to(torch.float16)
        
        # Inférence
        with torch.no_grad():
            logits = asr_model(input_values).logits
        
        # Décodage CTC (argmax)
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = asr_processor.batch_decode(predicted_ids)[0]
        
        return jsonify({"transcription": transcription})
    except Exception as e:
        print(f"Erreur Transcription: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/', methods=['GET'])
def home():
    return "MedGemma & MedASR API is running!"

# --- LANCEMENT DU TUNNEL ---
if NGROK_AUTH_TOKEN != "VOTRE_TOKEN_NGROK_ICI":
    ngrok.set_auth_token(NGROK_AUTH_TOKEN)

try:
    public_url = ngrok.connect(5000).public_url
    print(f"🚀 API ACCESSIBLE À CETTE ADRESSE : {public_url}")
except Exception as e:
    print(f"Erreur ngrok: {e}")

# Lancement du serveur Flask
app.run(port=5000)
