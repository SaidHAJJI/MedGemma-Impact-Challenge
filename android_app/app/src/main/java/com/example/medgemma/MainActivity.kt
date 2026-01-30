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
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.ArrowBack
import androidx.compose.material.icons.filled.Mic
import androidx.compose.material.icons.filled.Refresh
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.unit.dp
import com.google.mediapipe.tasks.genai.llminference.LlmInference
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File
import java.util.Locale

// --- Enum pour les étapes de triage ---
enum class TriageStep {
    INITIAL, FOLLOW_UP, FINAL
}

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

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun MedGemmaScreen() {
    val context = LocalContext.current
    val scope = rememberCoroutineScope()
    
    // États du Triage
    var currentStep by remember { mutableStateOf(TriageStep.INITIAL) }
    var ageText by remember { mutableStateOf("30") }
    var selectedSex by remember { mutableStateOf("Masculin") }
    var symptomsDescription by remember { mutableStateOf("") }
    var followupQuestions by remember { mutableStateOf("") }
    var followupAnswers by remember { mutableStateOf("") }
    var finalReport by remember { mutableStateOf("") }
    
    // États Techniques
    var isLoading by remember { mutableStateOf(false) }
    var isModelLoaded by remember { mutableStateOf(false) }
    var modelError by remember { mutableStateOf<String?>(null) }
    var llmInference by remember { mutableStateOf<LlmInference?>(null) }

    // --- System Prompts (V3 SafetyFirst) ---
    val promptQuestions = """Tu es MedGemma, expert en triage médical.
Génère 3-4 questions essentielles en français pour évaluer l'urgence des symptômes fournis.
Concentre-toi sur la distinction entre bénin et urgence vitale.
Format: liste à puces simple."""

    val promptFinal = """Tu es MedGemma, assistant de triage médical.
IMPORTANT: Si les symptômes suggèrent une urgence vitale (signes d'AVC, crise cardiaque, hémorragie grave, détresse respiratoire), dis IMMEDIATEMENT d'appeler le 15 ou le 112 sur la première ligne.
Pour les non-urgences, fournis:
1. Niveau d'urgence (Faible, Moyen, Haut).
2. Causes possibles (avec prudence).
3. Actions de soins immédiats.
4. Conseil sur quand consulter.
Réponds en français de façon concise et empathique."""

    // --- Chargement du modèle ---
    LaunchedEffect(Unit) {
        scope.launch(Dispatchers.IO) {
            try {
                val modelPath = File(context.filesDir, "medgemma.bin")
                if (!modelPath.exists()) {
                    modelError = "Modèle 'medgemma.bin' introuvable dans ${modelPath.absolutePath}. \n\nVeuillez copier le fichier via Device File Explorer."
                } else {
                    val options = LlmInference.LlmInferenceOptions.builder()
                        .setModelPath(modelPath.absolutePath)
                        .setMaxTokens(1024)
                        .setTemperature(0.4f)
                        .build()
                    llmInference = LlmInference.createFromOptions(context, options)
                    isModelLoaded = true
                }
            } catch (e: Exception) {
                modelError = "Erreur chargement modèle : ${e.message}"
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
                if (currentStep == TriageStep.INITIAL) symptomsDescription = results[0]
                else followupAnswers = results[0]
            }
        }
    }

    val permissionLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.RequestPermission()
    ) { isGranted ->
        if (isGranted) {
            val intent = Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH).apply {
                putExtra(RecognizerIntent.EXTRA_LANGUAGE_MODEL, RecognizerIntent.LANGUAGE_MODEL_FREE_FORM)
                putExtra(RecognizerIntent.EXTRA_LANGUAGE, "fr-FR")
                putExtra(RecognizerIntent.EXTRA_PROMPT, "Décrivez vos symptômes...")
            }
            speechLauncher.launch(intent)
        }
    }

    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text("MedGemma Triage (Local)") },
                colors = TopAppBarDefaults.topAppBarColors(
                    containerColor = MaterialTheme.colorScheme.primaryContainer,
                    titleContentColor = MaterialTheme.colorScheme.primary
                )
            )
        }
    ) {
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(it)
                .padding(16.dp)
                .verticalScroll(rememberScrollState())
        ) {
            if (modelError != null) {
                ErrorMessage(modelError!!)
            } else if (!isModelLoaded) {
                LoadingState()
            } else {
                when (currentStep) {
                    TriageStep.INITIAL -> {
                        InitialInputStep(
                            age = ageText,
                            onAgeChange = { ageText = it },
                            sex = selectedSex,
                            onSexChange = { selectedSex = it },
                            description = symptomsDescription,
                            onDescriptionChange = { symptomsDescription = it },
                            onVoiceInput = { permissionLauncher.launch(android.Manifest.permission.RECORD_AUDIO) },
                            isLoading = isLoading,
                            onNext = {
                                if (symptomsDescription.isNotBlank()) {
                                    isLoading = true
                                    scope.launch(Dispatchers.IO) {
                                        val prompt = "$promptQuestions\nPatient: $ageText ans, $selectedSex. Symptômes: $symptomsDescription"
                                        val response = llmInference?.generateResponse(prompt) ?: "Erreur interne"
                                        withContext(Dispatchers.Main) {
                                            followupQuestions = response
                                            currentStep = TriageStep.FOLLOW_UP
                                            isLoading = false
                                        }
                                    }
                                }
                            }
                        )
                    }
                    TriageStep.FOLLOW_UP -> {
                        FollowUpStep(
                            questions = followupQuestions,
                            answers = followupAnswers,
                            onAnswersChange = { followupAnswers = it },
                            onVoiceInput = { permissionLauncher.launch(android.Manifest.permission.RECORD_AUDIO) },
                            isLoading = isLoading,
                            onBack = { currentStep = TriageStep.INITIAL },
                            onGenerate = {
                                isLoading = true
                                scope.launch(Dispatchers.IO) {
                                    val prompt = """$promptFinal
                                        CONTEXTE:
                                        - Patient: $ageText ans, $selectedSex
                                        - Symptômes initiaux: $symptomsDescription
                                        - Précisions: $followupAnswers"""
                                    val response = llmInference?.generateResponse(prompt) ?: "Erreur interne"
                                    withContext(Dispatchers.Main) {
                                        finalReport = response
                                        currentStep = TriageStep.FINAL
                                        isLoading = false
                                    }
                                }
                            }
                        )
                    }
                    TriageStep.FINAL -> {
                        FinalReportStep(
                            report = finalReport,
                            onReset = {
                                currentStep = TriageStep.INITIAL
                                symptomsDescription = ""
                                followupAnswers = ""
                            }
                        )
                    }
                }
            }
        }
    }
}

@Composable
fun ErrorMessage(message: String) {
    Card(colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.errorContainer)) {
        Text(text = message, modifier = Modifier.padding(16.dp), color = MaterialTheme.colorScheme.onErrorContainer)
    }
}

@Composable
fun LoadingState() {
    Column(horizontalAlignment = Alignment.CenterHorizontally, modifier = Modifier.fillMaxWidth()) {
        CircularProgressIndicator()
        Spacer(modifier = Modifier.height(8.dp))
        Text("Chargement du modèle MedGemma local...")
    }
}

@Composable
fun InitialInputStep(
    age: String, onAgeChange: (String) -> Unit,
    sex: String, onSexChange: (String) -> Unit,
    description: String, onDescriptionChange: (String) -> Unit,
    onVoiceInput: () -> Unit,
    isLoading: Boolean,
    onNext: () -> Unit
) {
    Text("1. Informations de base", style = MaterialTheme.typography.titleLarge)
    Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
        OutlinedTextField(
            value = age, onValueChange = onAgeChange,
            label = { Text("Âge") },
            modifier = Modifier.weight(1f),
            keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Number)
        )
        var expanded by remember { mutableStateOf(false) }
        Box(modifier = Modifier.weight(1f)) {
            OutlinedButton(onClick = { expanded = true }, modifier = Modifier.fillMaxWidth().height(56.dp)) {
                Text(sex)
            }
            DropdownMenu(expanded = expanded, onDismissRequest = { expanded = false }) {
                listOf("Masculin", "Féminin", "Autre").forEach { option ->
                    DropdownMenuItem(text = { Text(option) }, onClick = { onSexChange(option); expanded = false })
                }
            }
        }
    }
    Spacer(modifier = Modifier.height(16.dp))
    Text("2. Symptômes", style = MaterialTheme.typography.titleLarge)
    OutlinedTextField(
        value = description, onValueChange = onDescriptionChange,
        label = { Text("Décrivez vos symptômes...") },
        modifier = Modifier.fillMaxWidth().height(150.dp),
        trailingIcon = {
            IconButton(onClick = onVoiceInput) { Icon(Icons.Default.Mic, contentDescription = "Vocal") }
        }
    )
    Spacer(modifier = Modifier.height(16.dp))
    Button(onClick = onNext, modifier = Modifier.fillMaxWidth(), enabled = !isLoading && description.isNotBlank()) {
        if (isLoading) CircularProgressIndicator(modifier = Modifier.size(24.dp), color = Color.White) else Text("Analyser ➡️")
    }
}

@Composable
fun FollowUpStep(
    questions: String,
    answers: String, onAnswersChange: (String) -> Unit,
    onVoiceInput: () -> Unit,
    isLoading: Boolean,
    onBack: () -> Unit,
    onGenerate: () -> Unit
) {
    Text("🔍 Questions de précision", style = MaterialTheme.typography.titleLarge)
    Card(modifier = Modifier.fillMaxWidth().padding(vertical = 8.dp)) {
        Text(text = questions, modifier = Modifier.padding(16.dp))
    }
    OutlinedTextField(
        value = answers, onValueChange = onAnswersChange,
        label = { Text("Vos réponses...") },
        modifier = Modifier.fillMaxWidth().height(150.dp),
        trailingIcon = {
            IconButton(onClick = onVoiceInput) { Icon(Icons.Default.Mic, contentDescription = "Vocal") }
        }
    )
    Spacer(modifier = Modifier.height(16.dp))
    Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
        OutlinedButton(onClick = onBack, modifier = Modifier.weight(1f)) { Text("⬅️ Retour") }
        Button(onClick = onGenerate, modifier = Modifier.weight(1f), enabled = !isLoading && answers.isNotBlank()) {
            if (isLoading) CircularProgressIndicator(modifier = Modifier.size(24.dp), color = Color.White) else Text("Rapport Final 🔍")
        }
    }
}

@Composable
fun FinalReportStep(report: String, onReset: () -> Unit) {
    Text("✅ Rapport de Triage", style = MaterialTheme.typography.titleLarge, color = MaterialTheme.colorScheme.primary)
    Card(modifier = Modifier.fillMaxWidth().padding(vertical = 8.dp), colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceVariant)) {
        Text(text = report, modifier = Modifier.padding(16.dp), style = MaterialTheme.typography.bodyLarge)
    }
    Spacer(modifier = Modifier.height(16.dp))
    Button(onClick = onReset, modifier = Modifier.fillMaxWidth()) {
        Icon(Icons.Default.Refresh, contentDescription = null)
        Spacer(modifier = Modifier.width(8.dp))
        Text("Nouvelle analyse")
    }
}