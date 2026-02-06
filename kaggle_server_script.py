# === SCRIPT DE SAUVETAGE ULTIME (KAGGLE) ===
# Ce script corrige TOUTES les erreurs d'import (DryRunError, DryRunFileInfo)
# et lance le serveur avec vos tokens.

# 1. NETTOYAGE ET INSTALLATION (Exécuter cette cellule après un "Restart Session")
import os
import sys
import subprocess
import types

def run_command(command):
    subprocess.check_call(command, shell=True)

print("🧹 Nettoyage des versions conflictuelles...")
try:
    # On désinstalle pour forcer une réinstallation propre
    run_command("pip uninstall -y huggingface_hub transformers")
except:
    pass

print("📦 Installation des versions compatibles (Pinning)...")
# On force des versions stables connues pour fonctionner ensemble sur Kaggle
run_command("pip install -q -U 'huggingface_hub<0.26.0' 'transformers<4.47.0' 'pydantic<2.12' flask flask-cors pyngrok accelerate bitsandbytes librosa soundfile")

# --- 2. PATCHS DE COMPATIBILITÉ (CRUCIAL) ---
# On applique ces patchs AVANT d'importer transformers
print("🔧 Application des correctifs de compatibilité...")

import huggingface_hub
import huggingface_hub.utils
import huggingface_hub.errors
import huggingface_hub.file_download
import huggingface_hub.constants

# Patch 0 : is_tqdm_disabled
if not hasattr(huggingface_hub.utils, 'is_tqdm_disabled'):
    def is_tqdm_disabled(): return False
    huggingface_hub.utils.is_tqdm_disabled = is_tqdm_disabled
    sys.modules['huggingface_hub.utils'].is_tqdm_disabled = is_tqdm_disabled
    print("  ✅ Patch is_tqdm_disabled appliqué.")

# Patch 0.1 : is_offline_mode (Correction de la nouvelle erreur)
def is_offline_mode(): return False
for module in [huggingface_hub, huggingface_hub.utils, huggingface_hub.constants]:
    if not hasattr(module, 'is_offline_mode'):
        setattr(module, 'is_offline_mode', is_offline_mode)
sys.modules['huggingface_hub'].is_offline_mode = is_offline_mode
print("  ✅ Patch is_offline_mode appliqué.")

# Patch 1 : DryRunError
if not hasattr(huggingface_hub.errors, 'DryRunError'):
    class DryRunError(Exception): pass
    huggingface_hub.errors.DryRunError = DryRunError
    sys.modules['huggingface_hub.errors'].DryRunError = DryRunError
    print("  ✅ Patch DryRunError appliqué.")

# Patch 2 : DryRunFileInfo (Nouvelle erreur que vous avez rencontrée)
if not hasattr(huggingface_hub.file_download, 'DryRunFileInfo'):
    class DryRunFileInfo:
        def __init__(self, **kwargs): pass
    huggingface_hub.file_download.DryRunFileInfo = DryRunFileInfo
    # On l'injecte aussi dans le module s'il est importé directement
    sys.modules['huggingface_hub.file_download'].DryRunFileInfo = DryRunFileInfo
    print("  ✅ Patch DryRunFileInfo appliqué.")

# --- 3. LE RESTE DU SCRIPT (VOTRE CODE) ---
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
from huggingface_hub import login

# === VOS TOKENS (SÉCURISÉS VIA KAGGLE SECRETS) ===
try:
    from kaggle_secrets import UserData
    NGROK_AUTH_TOKEN = UserData().get('NGROK_AUTH_TOKEN')
    HF_TOKEN = UserData().get('HF_TOKEN')
    print("✅ Tokens récupérés depuis Kaggle Secrets.")
except Exception:
    # Si pas sur Kaggle ou secrets non configurés, utilisez les placeholders
    NGROK_AUTH_TOKEN = "VOTRE_TOKEN_NGROK_ICI" 
    HF_TOKEN = "VOTRE_TOKEN_HF_ICI"
    print("⚠️ Utilisation des tokens manuels (placeholders).")

# --- AUTHENTIFICATION ---
if HF_TOKEN and HF_TOKEN != "VOTRE_TOKEN_HF_ICI":
    try:
        login(token=HF_TOKEN)
        print("✅ Authentification Hugging Face réussie.")
    except Exception as e:
        print(f"❌ Erreur Authentification HF : {e}")
        print("Vérifiez votre jeton sur hf.co/settings/tokens et acceptez la licence Gemma.")
else:
    print("❌ Erreur : Aucun jeton HF_TOKEN valide trouvé.")
LLM_MODEL_ID = "google/gemma-2b-it" 
ASR_MODEL_ID = "google/medasr"

print("🔄 Chargement Gemma...")
tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_ID)
llm_model = AutoModelForCausalLM.from_pretrained(
    LLM_MODEL_ID,
    device_map="auto",
    torch_dtype=torch.float16,
)

print("🔄 Chargement MedASR...")
try:
    asr_processor = AutoProcessor.from_pretrained(ASR_MODEL_ID)
    asr_model = AutoModelForCTC.from_pretrained(
        ASR_MODEL_ID,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    print("✅ Modèles chargés !")
except Exception as e:
    print(f"⚠️ Erreur MedASR : {e}")

# --- API FLASK ---
app = Flask(__name__)
CORS(app)

@app.route('/generate', methods=['POST'])
def generate():
    try:
        data = request.json
        full_prompt = f"<start_of_turn>user\n{data.get('system_instruction', '')}\n\n{data.get('prompt', '')}<end_of_turn>\n<start_of_turn>model\n"
        inputs = tokenizer(full_prompt, return_tensors="pt").to("cuda")
        outputs = llm_model.generate(**inputs, max_new_tokens=600, temperature=0.4, do_sample=True)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True).split("model\n")[-1].strip()
        return jsonify({"response": response})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/transcribe', methods=['POST'])
def transcribe():
    try:
        if 'audio' not in request.files: return jsonify({"error": "No audio"}), 400
        audio_file = request.files['audio']
        speech, _ = librosa.load(io.BytesIO(audio_file.read()), sr=16000)
        inputs = asr_processor(speech, sampling_rate=16000, return_tensors="pt")
        input_values = inputs.input_values.to("cuda").to(torch.float16)
        with torch.no_grad(): logits = asr_model(input_values).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = asr_processor.batch_decode(predicted_ids)[0]
        return jsonify({"transcription": transcription})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/')
def home(): return "MedGemma API Ready"

# --- LANCEMENT NGROK ---
ngrok.set_auth_token(NGROK_AUTH_TOKEN)
ngrok.kill()
try:
    public_url = ngrok.connect(5000).public_url
    print(f"\n🚀🚀🚀 URL API : {public_url} 🚀🚀🚀\n")
except Exception as e:
    print(f"Erreur Ngrok : {e}")

app.run(port=5000)
