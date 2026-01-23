package com.example.medgemma

import android.content.Context
import android.content.Intent
import android.os.Bundle
import android.speech.RecognizerIntent
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Mic
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import com.google.mediapipe.tasks.genai.llminference.LlmInference
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File
import java.util.Locale

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent {
            MaterialTheme {
                Surface(
                    modifier = Modifier.fillMaxSize(),
                    color = MaterialTheme.colorScheme.background
                ) {
                    MedGemmaScreen()
                }
            }
        }
    }
}

@Composable
fun MedGemmaScreen() {
    val context = LocalContext.current
    val scope = rememberCoroutineScope()
    
    var inputText by remember { mutableStateOf("") }
    var outputText by remember { mutableStateOf("") }
    var isLoading by remember { mutableStateOf(false) }
    var isModelLoaded by remember { mutableStateOf(false) }
    var modelError by remember { mutableStateOf<String?>(null) }
    
    // LLM Inference instance (stored in a simplified way for this demo)
    var llmInference by remember { mutableStateOf<LlmInference?>(null) }

    // Load Model on Start
    LaunchedEffect(Unit) {
        scope.launch(Dispatchers.IO) {
            try {
                // MODEL PATH: /data/local/tmp/llm/medgemma.bin 
                // OR pushed to app specific storage. For this demo, we assume a standard path 
                // or that the user has put it in the app's files dir via some mechanism.
                // A robust app would download it. Here we check a fixed path for simplicity.
                val modelPath = File(context.filesDir, "medgemma.bin")
                
                if (!modelPath.exists()) {
                    modelError = "Model file not found at ${modelPath.absolutePath}. \n\nPlease push the model file 'medgemma.bin' to this location."
                } else {
                    val options = LlmInference.LlmInferenceOptions.builder()
                        .setModelPath(modelPath.absolutePath)
                        .setMaxTokens(512)
                        .setTopK(40)
                        .setTemperature(0.8f)
                        .setRandomSeed(101)
                        .build()

                    llmInference = LlmInference.createFromOptions(context, options)
                    isModelLoaded = true
                }
            } catch (e: Exception) {
                modelError = "Error loading model: ${e.message}"
            }
        }
    }

    // Voice Recognition Launcher
    val speechLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.StartActivityForResult()
    ) { result ->
        if (result.resultCode == ComponentActivity.RESULT_OK) {
            val data = result.data
            val results = data?.getStringArrayListExtra(RecognizerIntent.EXTRA_RESULTS)
            if (!results.isNullOrEmpty()) {
                inputText = results[0]
            }
        }
    }

    // Permission Launcher
    val permissionLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.RequestPermission()
    ) { isGranted ->
        if (isGranted) {
            val intent = Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH).apply {
                putExtra(RecognizerIntent.EXTRA_LANGUAGE_MODEL, RecognizerIntent.LANGUAGE_MODEL_FREE_FORM)
                putExtra(RecognizerIntent.EXTRA_LANGUAGE, Locale.getDefault())
                putExtra(RecognizerIntent.EXTRA_PROMPT, "Describe your symptoms...")
            }
            speechLauncher.launch(intent)
        } else {
            Toast.makeText(context, "Microphone permission needed", Toast.LENGTH_SHORT).show()
        }
    }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(16.dp)
    ) {
        Text(
            text = "MedGemma Triage (Local)",
            style = MaterialTheme.typography.headlineMedium
        )
        
        Spacer(modifier = Modifier.height(16.dp))

        if (modelError != null) {
            Card(
                colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.errorContainer)
            ) {
                Text(
                    text = modelError!!,
                    modifier = Modifier.padding(16.dp),
                    color = MaterialTheme.colorScheme.onErrorContainer
                )
            }
        } else if (!isModelLoaded) {
            CircularProgressIndicator()
            Text("Loading MedGemma Model locally...")
        } else {
            Row(modifier = Modifier.fillMaxWidth()) {
                OutlinedTextField(
                    value = inputText,
                    onValueChange = { inputText = it },
                    label = { Text("Describe symptoms...") },
                    modifier = Modifier
                        .weight(1f)
                        .height(150.dp)
                )
                
                IconButton(
                    onClick = { permissionLauncher.launch(android.Manifest.permission.RECORD_AUDIO) },
                    modifier = Modifier.padding(start = 8.dp).align(androidx.compose.ui.Alignment.CenterVertically)
                ) {
                    Icon(Icons.Filled.Mic, contentDescription = "Voice Input")
                }
            }

            Spacer(modifier = Modifier.height(16.dp))

            Button(
                onClick = {
                    if (inputText.isNotBlank() && llmInference != null) {
                        isLoading = true
                        outputText = "" // Reset
                        scope.launch(Dispatchers.IO) {
                            try {
                                val prompt = """You are an AI Medical Assistant running offline on a smartphone. 
Your goal is to provide preliminary triage advice.
1. Analyze the symptoms: $inputText
2. Estimate urgency (Low, Medium, High, Emergency).
3. Suggest immediate actions.
4. Always advise seeing a doctor.
Keep responses concise."""
                                val result = llmInference!!.generateResponse(prompt)
                                withContext(Dispatchers.Main) {
                                    outputText = result
                                    isLoading = false
                                }
                            } catch (e: Exception) {
                                withContext(Dispatchers.Main) {
                                    outputText = "Error: ${e.message}"
                                    isLoading = false
                                }
                            }
                        }
                    }
                },
                enabled = !isLoading && inputText.isNotBlank(),
                modifier = Modifier.fillMaxWidth()
            ) {
                if (isLoading) {
                    CircularProgressIndicator(
                        modifier = Modifier.size(24.dp),
                        color = MaterialTheme.colorScheme.onPrimary
                    )
                } else {
                    Text("Analyze (Offline)")
                }
            }

            Spacer(modifier = Modifier.height(16.dp))

            Text("Recommendation:", style = MaterialTheme.typography.titleMedium)
            
            Box(
                modifier = Modifier
                    .fillMaxWidth()
                    .weight(1f)
                    .verticalScroll(rememberScrollState())
            ) {
                Text(text = outputText)
            }
        }
    }
}
