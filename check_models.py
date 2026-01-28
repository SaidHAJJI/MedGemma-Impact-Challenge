import google.generativeai as genai
import toml
import os

try:
    # Charger la clé depuis secrets.toml
    secrets = toml.load(".streamlit/secrets.toml")
    api_key = secrets.get("GEMINI_API_KEY")
    
    if not api_key:
        print("❌ Clé API introuvable dans .streamlit/secrets.toml")
        exit(1)

    genai.configure(api_key=api_key)
    
    print(f"📚 Version du SDK : {genai.__version__}")
    print("🔍 Recherche des modèles disponibles...")
    
    models = list(genai.list_models())
    found_flash = False
    
    for m in models:
        if "generateContent" in m.supported_generation_methods:
            print(f" - {m.name}")
            if "gemini-1.5-flash" in m.name:
                found_flash = True

    if found_flash:
        print("\n✅ Le modèle 'gemini-1.5-flash' est DISPONIBLE.")
    else:
        print("\n❌ Le modèle 'gemini-1.5-flash' est INTROUVABLE avec cette clé/version.")

except Exception as e:
    print(f"\n❌ Erreur : {e}")
