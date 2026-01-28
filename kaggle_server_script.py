# === À COPIER DANS UNE CELLULE DE NOTEBOOK KAGGLE ===
# Assurez-vous d'avoir activé l'accélérateur GPU (T4 x2) dans les options du Notebook.

# 1. Installation des dépendances
!pip install -q -U flask flask-cors pyngrok accelerate transformers bitsandbytes

# 2. Configuration & Import
import os
import torch
from flask import Flask, request, jsonify
from flask_cors import CORS
from pyngrok import ngrok
from transformers import AutoTokenizer, AutoModelForCausalLM

# --- CONFIGURATION ---
# Remplacez par votre token ngrok (https://dashboard.ngrok.com/get-started/your-authtoken)
# Si vous ne mettez pas de token, la session expirera vite.
NGROK_AUTH_TOKEN = "VOTRE_TOKEN_NGROK_ICI" 

# Modèle à utiliser (Gemma 2B IT est léger et performant pour ce test)
# Vous pouvez remplacer par votre propre modèle fine-tuné sur Kaggle si disponible
MODEL_ID = "google/gemma-2b-it" 

# --- CHARGEMENT DU MODÈLE ---
print("🔄 Chargement du modèle...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map="auto",
    torch_dtype=torch.float16, # Utilisation du GPU
)
print("✅ Modèle chargé !")

# --- CRÉATION DE L'API ---
app = Flask(__name__)
CORS(app)

@app.route('/generate', methods=['POST'])
def generate():
    data = request.json
    user_prompt = data.get('prompt', '')
    system_instruction = data.get('system_instruction', '')
    
    # Construction du prompt format Gemma
    # Gemma utilise un format spécifique : <start_of_turn>user ... <end_of_turn><start_of_turn>model
    full_prompt = f"<start_of_turn>user\n{system_instruction}\n\n{user_prompt}<end_of_turn>\n<start_of_turn>model\n"
    
    inputs = tokenizer(full_prompt, return_tensors="pt").to("cuda")
    
    # Génération
    outputs = model.generate(
        **inputs, 
        max_new_tokens=500,
        temperature=0.7,
        do_sample=True
    )
    
    response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Nettoyage pour ne garder que la réponse du modèle (après le prompt)
    # On cherche la dernière occurrence de "model"
    final_response = response_text.split("model\n")[-1].strip()
    
    return jsonify({"response": final_response})

@app.route('/', methods=['GET'])
def home():
    return "MedGemma API is running!"

# --- LANCEMENT DU TUNNEL ---
if NGROK_AUTH_TOKEN != "VOTRE_TOKEN_NGROK_ICI":
    ngrok.set_auth_token(NGROK_AUTH_TOKEN)

# Ouvre le tunnel sur le port 5000
public_url = ngrok.connect(5000).public_url
print(f"🚀 API ACCESSIBLE À CETTE ADRESSE : {public_url}")
print("👉 Copiez cette URL dans votre application Streamlit.")

# Lancement du serveur Flask
app.run(port=5000)
