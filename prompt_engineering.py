import requests
import json
import sys

# Configuration
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "gemma:2b"  # Change this to 'medgemma' if you have a custom model loaded

# Test Cases (Symptom descriptions)
TEST_CASES = [
    "I have a severe headache and sensitivity to light.",
    "My child has a fever of 39C and a rash.",
    "I twisted my ankle while running, it's swollen.",
    "I feel a crushing pain in my chest and my left arm is numb." # Critical case
]

# System Prompts to Test
SYSTEM_PROMPTS = {
    "V1_Basic": "You are a medical assistant. Analyze symptoms and suggest triage advice.",
    
    "V2_Structured": """You are an AI Medical Assistant running offline on a smartphone. 
Your goal is to provide preliminary triage advice.
1. Analyze the symptoms.
2. Estimate urgency (Low, Medium, High, Emergency).
3. Suggest immediate actions.
4. Always advise seeing a doctor.
Keep responses concise.""",

    "V3_SafetyFirst": """You are MedGemma, a helpful medical triage assistant.
CRITICAL: If symptoms suggest a life-threatening emergency (heart attack, stroke, severe bleeding), IMMEDIATELY tell the user to call emergency services (911/112).
For non-emergencies, provide home care advice and suggest when to see a doctor.
Be empathetic but professional. Use simple language."""
}

def query_ollama(prompt, system_prompt):
    full_prompt = f"{system_prompt}\n\nUser: {prompt}\nAssistant:"
    
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
        return f"Error querying Ollama: {e}"

def run_tests():
    print(f"--- Testing Prompts with Model: {MODEL_NAME} ---\
")
    
    for prompt_name, sys_prompt in SYSTEM_PROMPTS.items():
        print(f"=== Testing System Prompt: {prompt_name} ===")
        print(f"System Instruction: {sys_prompt[:50]}...")
        
        for case in TEST_CASES:
            print(f"\n[Case]: {case}")
            response = query_ollama(case, sys_prompt)
            print(f"[Response]:\n{response.strip()}")
            print("-" * 40)
        print("\n" + "="*60 + "\n")

if __name__ == "__main__":
    # Check if requests is installed, strictly for the script run
    try:
        import requests
    except ImportError:
        print("Please install 'requests': pip install requests")
        sys.exit(1)
        
    run_tests()
