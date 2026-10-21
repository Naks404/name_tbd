package com.example.myapplication

import android.Manifest
import android.annotation.SuppressLint
import android.content.ContentValues
import android.content.pm.PackageManager
import android.content.res.Configuration
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.hardware.camera2.CaptureRequest
import android.hardware.camera2.params.RggbChannelVector
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.os.Build
import android.os.Bundle
import android.provider.MediaStore
import android.util.Log
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.camera2.interop.Camera2CameraControl
import androidx.camera.camera2.interop.CaptureRequestOptions
import androidx.camera.camera2.interop.ExperimentalCamera2Interop
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.video.*
import androidx.camera.video.Recorder
import androidx.camera.video.VideoCapture
import androidx.camera.view.PreviewView
import androidx.compose.animation.*
import androidx.compose.animation.core.*
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.gestures.snapping.rememberSnapFlingBehavior
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.*
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material.icons.filled.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.scale
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.drawscope.rotate
import androidx.compose.ui.hapticfeedback.HapticFeedbackType
import androidx.compose.ui.platform.LocalConfiguration
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalHapticFeedback
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import androidx.compose.ui.platform.LocalLifecycleOwner
import androidx.media3.common.MediaItem
import androidx.media3.exoplayer.ExoPlayer
import androidx.media3.ui.PlayerView
import com.chaquo.python.PyObject
import com.chaquo.python.Python
import com.example.myapplication.ui.theme.MyApplicationTheme
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import org.pytorch.IValue
import org.pytorch.LiteModuleLoader
import org.pytorch.Tensor
import org.json.JSONObject
import org.vosk.Recognizer
import org.vosk.android.StorageService
import java.io.File
import java.io.FileInputStream
import java.io.FileOutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.text.SimpleDateFormat
import java.util.*
import java.util.concurrent.Executors
import kotlin.math.atan2
import kotlin.math.roundToInt

// =========================================================================
// TOP-LEVEL HELPER FUNCTIONS
// =========================================================================

fun assetFilePath(context: android.content.Context, assetName: String): String {
    val file = File(context.filesDir, assetName)
    if (file.exists() && file.length() > 0) return file.absolutePath
    context.assets.open(assetName).use { inputStream ->
        FileOutputStream(file).use { outputStream ->
            val buffer = ByteArray(4 * 1024)
            var read: Int
            while (inputStream.read(buffer).also { read = it } != -1) {
                outputStream.write(buffer, 0, read)
            }
            outputStream.flush()
        }
    }
    return file.absolutePath
}

fun parseEffectPreset(context: android.content.Context, category: String, index: Int): String {
    try {
        val jsonString = context.assets.open("speech_effect_chain_v2.json").bufferedReader().use { it.readText() }
        val jsonObject = JSONObject(jsonString)

        val presetArray = jsonObject.getJSONArray(category)
        val safeIndex = index.coerceIn(0, presetArray.length() - 1)
        val preset = presetArray.getJSONObject(safeIndex)

        if (preset.getBoolean("bypass")) return "Bypassed"

        val effects = preset.getJSONArray("effects")
        val firstEffect = effects.getJSONObject(0)
        val type = firstEffect.getString("type")
        val params = firstEffect.getJSONObject("params")

        return when (type) {
            "NoiseGate" -> "Noise Gate (Thresh: ${params.getDouble("threshold_db")}dB, Ratio: ${params.getDouble("ratio")}:1)"
            "Compressor" -> "Compressor (Ratio: ${params.getDouble("ratio")}:1, Thresh: ${params.getDouble("threshold_db")}dB)"
            "LowShelfFilter" -> "3-Band EQ (Low: ${params.getDouble("gain_db")}dB)"
            "PeakFilter" -> "De-Esser (${params.getDouble("gain_db")}dB at ${params.getDouble("cutoff_frequency_hz")}Hz)"
            "Reverb" -> "Reverb (Room Size: ${params.getDouble("room_size")}, Wet: ${params.getDouble("wet_level")})"
            "Limiter" -> "Limiter (Thresh: ${params.getDouble("threshold_db")}dB)"
            else -> "$type Applied"
        }
    } catch (e: Exception) {
        Log.e("JSON_PARSER", "Failed to parse $category", e)
        return "Parse Error"
    }
}

// =========================================================================
// APP NAVIGATION & CORE SCREENS
// =========================================================================

sealed class Screen {
    object Home : Screen()
    object Video : Screen()
    object Audio : Screen()
    object Gallery : Screen()
    data class Transcription(val filePath: String) : Screen()
}

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContent {
            MyApplicationTheme {
                AppNavigation()
            }
        }
    }
}

@Composable
fun AppNavigation() {
    var currentScreen by remember { mutableStateOf<Screen>(Screen.Home) }

    AnimatedContent(
        targetState = currentScreen,
        transitionSpec = { fadeIn(tween(300)) togetherWith fadeOut(tween(300)) },
        label = "ScreenTransition"
    ) { screen ->
        when (screen) {
            is Screen.Home -> HomeScreen(
                onNavigateToVideo = { currentScreen = Screen.Video },
                onNavigateToAudio = { currentScreen = Screen.Audio },
                onNavigateToGallery = { currentScreen = Screen.Gallery }
            )
            is Screen.Video -> VideoModeScreen(onNavigateBack = { currentScreen = Screen.Home })
            is Screen.Audio -> AudioModeScreen(
                onNavigateBack = { currentScreen = Screen.Home }
            )
            is Screen.Gallery -> GalleryScreen(
                onNavigateBack = { currentScreen = Screen.Home },
                onNavigateToTranscription = { filePath -> currentScreen = Screen.Transcription(filePath) }
            )
            is Screen.Transcription -> TranscriptionScreen(
                filePath = screen.filePath,
                onNavigateBack = { currentScreen = Screen.Gallery }
            )
        }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun HomeScreen(onNavigateToVideo: () -> Unit, onNavigateToAudio: () -> Unit, onNavigateToGallery: () -> Unit) {
    Scaffold(
        topBar = { TopAppBar(title = { Text("Field Assistant", fontWeight = FontWeight.SemiBold) }) }
    ) { paddingValues ->
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(paddingValues)
                .padding(horizontal = 24.dp),
            verticalArrangement = Arrangement.Center,
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            Icon(Icons.Default.Camera, contentDescription = "App Logo", modifier = Modifier.size(80.dp).padding(bottom = 16.dp), tint = MaterialTheme.colorScheme.primary)
            Text("Select Mode", style = MaterialTheme.typography.headlineMedium, fontWeight = FontWeight.Bold)
            Text("Real-time visual and acoustic processing.", style = MaterialTheme.typography.bodyMedium, color = MaterialTheme.colorScheme.onSurfaceVariant, modifier = Modifier.padding(top = 8.dp, bottom = 48.dp))
            ModeCard("Visual Capture", "Live framing & exposure feedback", Icons.Default.Videocam, onNavigateToVideo)
            Spacer(modifier = Modifier.height(16.dp))
            ModeCard("Acoustic Record", "Smart recording & auto-transcription", Icons.Default.Mic, onNavigateToAudio)
            Spacer(modifier = Modifier.height(16.dp))
            ModeCard("Gallery", "View and analyze recordings", Icons.Default.PhotoLibrary, onNavigateToGallery)
        }
    }
}

@Composable
fun ModeCard(title: String, description: String, icon: androidx.compose.ui.graphics.vector.ImageVector, onClick: () -> Unit) {
    ElevatedCard(
        modifier = Modifier.fillMaxWidth().clickable(onClick = onClick),
        elevation = CardDefaults.elevatedCardElevation(0.dp),
        colors = CardDefaults.elevatedCardColors(containerColor = MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.5f))
    ) {
        Row(modifier = Modifier.fillMaxWidth().padding(20.dp), verticalAlignment = Alignment.CenterVertically) {
            Box(modifier = Modifier.size(48.dp).clip(CircleShape).background(MaterialTheme.colorScheme.primaryContainer), contentAlignment = Alignment.Center) {
                Icon(icon, contentDescription = null, tint = MaterialTheme.colorScheme.onPrimaryContainer)
            }
            Spacer(modifier = Modifier.width(16.dp))
            Column {
                Text(title, style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.Bold)
                Text(description, style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
            }
        }
    }
}

@androidx.annotation.OptIn(ExperimentalCamera2Interop::class)
@SuppressLint("MissingPermission")
@Composable
fun VideoModeScreen(onNavigateBack: () -> Unit) {
    val context = LocalContext.current
    val lifecycleOwner = LocalLifecycleOwner.current
    val configuration = LocalConfiguration.current
    val isLandscape = configuration.orientation == Configuration.ORIENTATION_LANDSCAPE

    var hasPermissions by remember {
        mutableStateOf(
            ContextCompat.checkSelfPermission(context, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED &&
                    ContextCompat.checkSelfPermission(context, Manifest.permission.RECORD_AUDIO) == PackageManager.PERMISSION_GRANTED
        )
    }
    val permissionLauncher = rememberLauncherForActivityResult(ActivityResultContracts.RequestMultiplePermissions()) { perms ->
        hasPermissions = perms[Manifest.permission.CAMERA] == true && perms[Manifest.permission.RECORD_AUDIO] == true
    }
    LaunchedEffect(Unit) { if (!hasPermissions) permissionLauncher.launch(arrayOf(Manifest.permission.CAMERA, Manifest.permission.RECORD_AUDIO)) }

    if (!hasPermissions) {
        PermissionDeniedScreen("Camera & Mic", onNavigateBack) { permissionLauncher.launch(arrayOf(Manifest.permission.CAMERA, Manifest.permission.RECORD_AUDIO)) }
        return
    }

    val previewView = remember { PreviewView(context) }
    var camera by remember { mutableStateOf<Camera?>(null) }
    var videoCapture by remember { mutableStateOf<VideoCapture<Recorder>?>(null) }
    var imageCapture by remember { mutableStateOf<ImageCapture?>(null) }
    var previewUseCase by remember { mutableStateOf<Preview?>(null) }
    var imageAnalysisUseCase by remember { mutableStateOf<ImageAnalysis?>(null) }
    var recording by remember { mutableStateOf<Recording?>(null) }
    var isVideoPaused by remember { mutableStateOf(false) }

    var exposureFeedback by remember { mutableStateOf<Map<String, String>>(emptyMap()) }
    var compositionFeedback by remember { mutableStateOf<Map<String, String>>(emptyMap()) }

    var isoValue by remember { mutableIntStateOf(400) }
    var focusDistance by remember { mutableFloatStateOf(0.0f) }
    var exposureTimeNs by remember { mutableLongStateOf(1_000_000_000L / 50) }
    var wbValue by remember { mutableIntStateOf(5000) }
    var activeSetting by remember { mutableStateOf("NONE") }

    val isoOptions = listOf(100, 125, 160, 200, 250, 320, 400, 500, 640, 800, 1000, 1250, 1600, 2000, 2500, 3200).map { it to it.toString() }
    val shutterOptions = listOf<Pair<Long, String>>(
        30_000_000_000L to "30\"", 15_000_000_000L to "15\"", 8_000_000_000L to "8\"", 4_000_000_000L to "4\"", 2_000_000_000L to "2\"", 1_000_000_000L to "1\"",
        1000000000L/2 to "1/2", 1000000000L/4 to "1/4", 1000000000L/8 to "1/8", 1000000000L/15 to "1/15", 1000000000L/24 to "1/24", 1000000000L/25 to "1/25",
        1000000000L/30 to "1/30", 1000000000L/48 to "1/48", 1000000000L/50 to "1/50", 1000000000L/60 to "1/60", 1000000000L/96 to "1/96", 1000000000L/100 to "1/100",
        1000000000L/120 to "1/120", 1000000000L/125 to "1/125", 1000000000L/250 to "1/250", 1000000000L/500 to "1/500", 1000000000L/1000 to "1/1000"
    )
    val wbOptions = (20..80).map { it * 100 }.map { it to "${it}K" }
    val focusOptions = (0..20).map { i -> val v = i * 0.5f; v to if (i == 0) "Inf" else if (i == 20) "Macro" else "|" }

    val listStates = remember {
        mapOf(
            "WB" to LazyListState(firstVisibleItemIndex = wbOptions.indexOfFirst { it.first == wbValue }.coerceAtLeast(0)),
            "FOCUS" to LazyListState(firstVisibleItemIndex = focusOptions.indexOfFirst { it.first == focusDistance }.coerceAtLeast(0)),
            "ISO" to LazyListState(firstVisibleItemIndex = isoOptions.indexOfFirst { it.first == isoValue }.coerceAtLeast(0)),
            "SHUTTER" to LazyListState(firstVisibleItemIndex = shutterOptions.indexOfFirst { it.first == exposureTimeNs }.coerceAtLeast(0))
        )
    }

    val sensorManager = remember { context.getSystemService(android.content.Context.SENSOR_SERVICE) as SensorManager }
    val rotationAngle = remember { mutableFloatStateOf(0f) }
    DisposableEffect(Unit) {
        val sensor = sensorManager.getDefaultSensor(Sensor.TYPE_GRAVITY)
        val listener = object : SensorEventListener {
            override fun onSensorChanged(event: SensorEvent?) { event?.let { rotationAngle.floatValue = Math.toDegrees(atan2(it.values[0].toDouble(), it.values[1].toDouble())).toFloat() } }
            override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {}
        }
        sensor?.let { sensorManager.registerListener(listener, it, SensorManager.SENSOR_DELAY_UI) }
        onDispose { sensorManager.unregisterListener(listener) }
    }

    LaunchedEffect(lifecycleOwner) {
        val cameraProvider = ProcessCameraProvider.getInstance(context).get()
        val preview = Preview.Builder().build().also { it.setSurfaceProvider(previewView.surfaceProvider) }
        previewUseCase = preview
        val recorderBuilder = Recorder.Builder().setQualitySelector(QualitySelector.from(Quality.HIGHEST))

        videoCapture = VideoCapture.Builder(recorderBuilder.build()).build()
        imageCapture = ImageCapture.Builder().build()

        val py = try { Python.getInstance() } catch (e: Exception) { null }
        val expModule = py?.getModule("exposure_analyzer")
        val compModule = py?.getModule("composition_analyzer")

        val exposureAnalyzer = expModule?.callAttr("ExposureAnalyzer")
        var compositionAnalyzer: PyObject? = null
        kotlinx.coroutines.withContext(Dispatchers.IO) {
            try {
                compositionAnalyzer = compModule?.callAttr("CompositionAnalyzer")
            } catch (e: Exception) {
                Log.e("Chaquopy", "Failed to init CompositionAnalyzer", e)
            }
        }

        val analyzerExecutor = Executors.newSingleThreadExecutor()
        val imageAnalysis = ImageAnalysis.Builder()
            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
            .setTargetResolution(android.util.Size(640, 480))
            .build()
            .also { analysisUseCase ->
                imageAnalysisUseCase = analysisUseCase
                analysisUseCase.setAnalyzer(analyzerExecutor) { imageProxy ->
                    try {
                        val nv21Bytes = YuvConverter.imageProxyToNv21(imageProxy)
                        val rotation = imageProxy.imageInfo.rotationDegrees

                        if (exposureAnalyzer != null) {
                            val expResult: PyObject = exposureAnalyzer.callAttr("analyze_frame_nv21", nv21Bytes, imageProxy.width, imageProxy.height)
                            exposureFeedback = expResult.asMap().entries.associate { it.key.toString() to it.value.toString() }
                        }

                        if (compositionAnalyzer != null) {
                            val compResult: PyObject = compositionAnalyzer.callAttr("process_frame_nv21", nv21Bytes, imageProxy.width, imageProxy.height, rotation)
                            compositionFeedback = compResult.asMap().entries.associate { it.key.toString() to it.value.toString() }
                        }
                    } catch (e: Exception) {
                        Log.e("Chaquopy", "Computer Vision Error", e)
                    } finally {
                        imageProxy.close()
                    }
                }
            }

        try {
            cameraProvider.unbindAll()
            camera = cameraProvider.bindToLifecycle(lifecycleOwner, CameraSelector.DEFAULT_BACK_CAMERA, preview, imageCapture, videoCapture, imageAnalysis)
        } catch (exc: Exception) {
            Log.e("CameraX", "Binding failed", exc)
        }
    }

    LaunchedEffect(isoValue, focusDistance, exposureTimeNs, wbValue, camera) {
        delay(150)
        camera?.let { cam ->
            val camera2Control = Camera2CameraControl.from(cam.cameraControl)

            val rGain = (1.0f + (wbValue - 2000) / 6000f * 2.0f).coerceIn(1.0f, 3.0f)
            val bGain = (1.0f + (8000 - wbValue) / 6000f * 2.0f).coerceIn(1.0f, 3.0f)
            val rggb = RggbChannelVector(rGain, 1.0f, 1.0f, bGain)

            val captureRequestOptions = CaptureRequestOptions.Builder()
                .setCaptureRequestOption(CaptureRequest.CONTROL_AE_MODE, CaptureRequest.CONTROL_AE_MODE_OFF)
                .setCaptureRequestOption(CaptureRequest.SENSOR_SENSITIVITY, isoValue)
                .setCaptureRequestOption(CaptureRequest.SENSOR_EXPOSURE_TIME, exposureTimeNs)
                .setCaptureRequestOption(CaptureRequest.CONTROL_AF_MODE, CaptureRequest.CONTROL_AF_MODE_OFF)
                .setCaptureRequestOption(CaptureRequest.LENS_FOCUS_DISTANCE, focusDistance)
                .setCaptureRequestOption(CaptureRequest.CONTROL_AWB_MODE, CaptureRequest.CONTROL_AWB_MODE_OFF)
                .setCaptureRequestOption(CaptureRequest.COLOR_CORRECTION_MODE, CaptureRequest.COLOR_CORRECTION_MODE_TRANSFORM_MATRIX)
                .setCaptureRequestOption(CaptureRequest.COLOR_CORRECTION_GAINS, rggb)
                .build()

            camera2Control.setCaptureRequestOptions(captureRequestOptions)
        }
    }

    Box(modifier = Modifier.fillMaxSize().background(Color.Black)) {
        AndroidView(factory = { previewView }, modifier = Modifier.fillMaxSize())

        Canvas(modifier = Modifier.fillMaxSize()) {
            val w = size.width; val h = size.height
            val linePaint = Color.White.copy(alpha = 0.3f)
            drawLine(linePaint, Offset(w / 3, 0f), Offset(w / 3, h), strokeWidth = 2f)
            drawLine(linePaint, Offset(w * 2 / 3, 0f), Offset(w * 2 / 3, h), strokeWidth = 2f)
            drawLine(linePaint, Offset(0f, h / 3), Offset(w, h / 3), strokeWidth = 2f)
            drawLine(linePaint, Offset(0f, h * 2 / 3), Offset(w, h * 2 / 3), strokeWidth = 2f)

            val display = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) context.display else (context.getSystemService(android.content.Context.WINDOW_SERVICE) as android.view.WindowManager).defaultDisplay
            val angleOffset = when(display?.rotation) { android.view.Surface.ROTATION_90 -> 90f; android.view.Surface.ROTATION_180 -> 180f; android.view.Surface.ROTATION_270 -> -90f; else -> 0f }
            val angle = rotationAngle.floatValue - angleOffset

            val normalized = (angle % 90 + 90) % 90
            val dist = minOf(normalized, 90 - normalized)

            rotate(-angle) {
                val levelColor = if (dist < 2f) Color.Green else Color.White.copy(alpha = 0.7f)
                drawLine(levelColor, Offset(w / 2 - 100f, h / 2), Offset(w / 2 + 100f, h / 2), strokeWidth = 4f)
                drawLine(levelColor, Offset(w / 2, h / 2 - 20f), Offset(w / 2, h / 2 + 20f), strokeWidth = 2f)
            }
        }

        Column(modifier = Modifier.align(Alignment.TopCenter).padding(top = 48.dp), horizontalAlignment = Alignment.CenterHorizontally, verticalArrangement = Arrangement.spacedBy(8.dp)) {
            if (exposureFeedback.isNotEmpty()) {
                Column(modifier = Modifier.background(Color.Black.copy(alpha = 0.6f), RoundedCornerShape(8.dp)).padding(16.dp), horizontalAlignment = Alignment.CenterHorizontally) {
                    Text(text = exposureFeedback["primary_action"] ?: "", color = when(exposureFeedback["primary_color"]) { "RED" -> Color.Red; "AMBER" -> Color(0xFFFFBF00); "GREEN" -> Color.Green; "CYAN" -> Color.Cyan; else -> Color.White }, style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.Bold)
                    val secondary = exposureFeedback["secondary_action"] ?: ""
                    if (secondary.isNotEmpty()) {
                        Spacer(modifier = Modifier.height(4.dp))
                        Text(text = secondary, color = when(exposureFeedback["secondary_color"]) { "CYAN" -> Color.Cyan; else -> Color.White }, style = MaterialTheme.typography.bodyMedium)
                    }
                }
            }

            if (compositionFeedback.isNotEmpty()) {
                val compText = compositionFeedback["feedback_text"] ?: ""
                if (compText.isNotEmpty()) {
                    val isAligned = compositionFeedback["is_aligned"] == "True"
                    Column(modifier = Modifier.background(Color(0xFF1E3A8A).copy(alpha = 0.7f), RoundedCornerShape(8.dp)).padding(12.dp), horizontalAlignment = Alignment.CenterHorizontally) {
                        Text(text = compText, color = if (isAligned) Color.Green else Color.White, style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.SemiBold)
                    }
                }
            }
        }

        Row(modifier = Modifier.fillMaxWidth().padding(top = 48.dp, start = 16.dp, end = 16.dp)) {
            IconButton(onClick = onNavigateBack, modifier = Modifier.background(Color.Black.copy(alpha = 0.4f), CircleShape)) { Icon(Icons.AutoMirrored.Filled.ArrowBack, contentDescription = "Back", tint = Color.White) }
        }

        val controlsModifier = if (isLandscape) Modifier.align(Alignment.CenterEnd).fillMaxHeight().background(Color.Black.copy(alpha = 0.5f)).padding(horizontal = 16.dp, vertical = 24.dp) else Modifier.align(Alignment.BottomCenter).fillMaxWidth().background(Color.Black.copy(alpha = 0.5f)).padding(bottom = 48.dp, top = 16.dp)

        Box(modifier = controlsModifier) {
            if (isLandscape) {
                Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.End, modifier = Modifier.fillMaxHeight()) {
                    AnimatedVisibility(visible = activeSetting != "NONE") {
                        val itemsList = when (activeSetting) { "WB" -> wbOptions; "FOCUS" -> focusOptions; "ISO" -> isoOptions; "SHUTTER" -> shutterOptions; else -> emptyList() }
                        val activeState = listStates[activeSetting]
                        if (activeState != null && itemsList.isNotEmpty()) {
                            key(activeSetting) { ValueWheel(items = itemsList, selectedValue = when (activeSetting) { "WB" -> wbValue; "FOCUS" -> focusDistance; "ISO" -> isoValue; "SHUTTER" -> exposureTimeNs; else -> null }, onValueChange = { when (activeSetting) { "WB" -> wbValue = it as Int; "FOCUS" -> focusDistance = it as Float; "ISO" -> isoValue = it as Int; "SHUTTER" -> exposureTimeNs = it as Long } }, isLandscape = true, listState = activeState) }
                        }
                    }
                    Spacer(modifier = Modifier.width(24.dp))
                    Column(horizontalAlignment = Alignment.CenterHorizontally, verticalArrangement = Arrangement.SpaceEvenly, modifier = Modifier.fillMaxHeight()) {
                        SettingToggle("WB", "${wbValue}K", activeSetting == "WB") { activeSetting = if (activeSetting == "WB") "NONE" else "WB" }
                        SettingToggle("Focus", if (focusDistance == 0f) "Inf" else if (focusDistance == 10f) "Macro" else "MF", activeSetting == "FOCUS") { activeSetting = if (activeSetting == "FOCUS") "NONE" else "FOCUS" }
                        SettingToggle("ISO", "$isoValue", activeSetting == "ISO") { activeSetting = if (activeSetting == "ISO") "NONE" else "ISO" }
                        SettingToggle("S", shutterOptions.find { it.first == exposureTimeNs }?.second ?: "", activeSetting == "SHUTTER") { activeSetting = if (activeSetting == "SHUTTER") "NONE" else "SHUTTER" }
                    }
                    Spacer(modifier = Modifier.width(32.dp))
                    Column(horizontalAlignment = Alignment.CenterHorizontally, verticalArrangement = Arrangement.SpaceEvenly, modifier = Modifier.fillMaxHeight()) { CaptureButtons(context, imageCapture, videoCapture, recording, isVideoPaused, { recording = it }, { isVideoPaused = it }) }
                }
            } else {
                Column(horizontalAlignment = Alignment.CenterHorizontally) {
                    AnimatedVisibility(visible = activeSetting != "NONE") {
                        val itemsList = when (activeSetting) { "WB" -> wbOptions; "FOCUS" -> focusOptions; "ISO" -> isoOptions; "SHUTTER" -> shutterOptions; else -> emptyList() }
                        val activeState = listStates[activeSetting]
                        if (activeState != null && itemsList.isNotEmpty()) {
                            key(activeSetting) { ValueWheel(items = itemsList, selectedValue = when (activeSetting) { "WB" -> wbValue; "FOCUS" -> focusDistance; "ISO" -> isoValue; "SHUTTER" -> exposureTimeNs; else -> null }, onValueChange = { when (activeSetting) { "WB" -> wbValue = it as Int; "FOCUS" -> focusDistance = it as Float; "ISO" -> isoValue = it as Int; "SHUTTER" -> exposureTimeNs = it as Long } }, isLandscape = false, listState = activeState) }
                        }
                    }
                    Spacer(modifier = Modifier.height(16.dp))
                    Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceEvenly, verticalAlignment = Alignment.CenterVertically) {
                        SettingToggle("WB", "${wbValue}K", activeSetting == "WB") { activeSetting = if (activeSetting == "WB") "NONE" else "WB" }
                        SettingToggle("Focus", if (focusDistance == 0f) "Inf" else if (focusDistance == 10f) "Macro" else "MF", activeSetting == "FOCUS") { activeSetting = if (activeSetting == "FOCUS") "NONE" else "FOCUS" }
                        SettingToggle("ISO", "$isoValue", activeSetting == "ISO") { activeSetting = if (activeSetting == "ISO") "NONE" else "ISO" }
                        SettingToggle("S", shutterOptions.find { it.first == exposureTimeNs }?.second ?: "", activeSetting == "SHUTTER") { activeSetting = if (activeSetting == "SHUTTER") "NONE" else "SHUTTER" }
                    }
                    Spacer(modifier = Modifier.height(16.dp))
                    Row(modifier = Modifier.fillMaxWidth().padding(horizontal = 32.dp), horizontalArrangement = Arrangement.SpaceEvenly, verticalAlignment = Alignment.CenterVertically) { CaptureButtons(context, imageCapture, videoCapture, recording, isVideoPaused, { recording = it }, { isVideoPaused = it }) }
                }
            }
        }
    }
}

@Composable
fun SettingToggle(label: String, currentValueStr: String, isSelected: Boolean, onClick: () -> Unit) {
    Column(horizontalAlignment = Alignment.CenterHorizontally, modifier = Modifier.clickable(onClick = onClick).padding(8.dp)) {
        Text(label, color = if (isSelected) Color.Yellow else Color.White, style = MaterialTheme.typography.labelSmall)
        Spacer(modifier = Modifier.height(2.dp))
        Text(currentValueStr, color = if (isSelected) Color.Yellow else Color.White, style = MaterialTheme.typography.titleSmall, fontWeight = if (isSelected) FontWeight.Bold else FontWeight.Normal)
    }
}

@OptIn(androidx.compose.foundation.ExperimentalFoundationApi::class)
@Composable
fun <T> ValueWheel(items: List<Pair<T, String>>, selectedValue: T?, onValueChange: (T) -> Unit, isLandscape: Boolean, listState: LazyListState) {
    val snapBehavior = rememberSnapFlingBehavior(lazyListState = listState)
    val hapticFeedback = LocalHapticFeedback.current
    val itemSizeDp = 48.dp

    val centerItemIndex by remember(listState, isLandscape) {
        derivedStateOf {
            val layoutInfo = listState.layoutInfo
            val visibleItemsInfo = layoutInfo.visibleItemsInfo
            if (visibleItemsInfo.isEmpty()) return@derivedStateOf -1
            val viewportCenter = (layoutInfo.viewportStartOffset + layoutInfo.viewportEndOffset) / 2
            visibleItemsInfo.minByOrNull { Math.abs(it.offset + (it.size / 2) - viewportCenter) }?.index ?: -1
        }
    }

    LaunchedEffect(centerItemIndex) {
        if (listState.isScrollInProgress && centerItemIndex in items.indices) {
            val newValue = items[centerItemIndex].first
            if (newValue != selectedValue) {
                hapticFeedback.performHapticFeedback(HapticFeedbackType.TextHandleMove)
                onValueChange(newValue)
            }
        }
    }

    BoxWithConstraints(modifier = Modifier.run { if (isLandscape) width(80.dp).fillMaxHeight() else fillMaxWidth().height(80.dp) }) {
        val halfSize = if (isLandscape) maxHeight / 2 else maxWidth / 2
        val paddingDp = halfSize - (itemSizeDp / 2)
        val isFocusWheel = items.any { it.second == "Macro" }

        if (isLandscape) {
            LazyColumn(state = listState, flingBehavior = snapBehavior, modifier = Modifier.fillMaxSize(), horizontalAlignment = Alignment.CenterHorizontally, contentPadding = PaddingValues(vertical = paddingDp)) {
                itemsIndexed(items) { index, item ->
                    val isCenter = index == centerItemIndex
                    val isTick = item.second == "|"
                    val color by animateColorAsState(if (isCenter) Color.Yellow else if (isTick) Color.White.copy(alpha = 0.5f) else Color.White, label = "color")
                    val scale by animateFloatAsState(if (isCenter) 1.2f else 1.0f, label = "scale")
                    Box(modifier = Modifier.height(itemSizeDp).fillMaxWidth(), contentAlignment = Alignment.Center) { Text(item.second, color = color, fontWeight = if (isCenter && !isTick) FontWeight.Bold else FontWeight.Normal, style = MaterialTheme.typography.labelMedium, modifier = Modifier.scale(scale)) }
                }
            }
            if (isFocusWheel) {
                Text("Inf", color = Color.White.copy(alpha = 0.3f), style = MaterialTheme.typography.labelSmall, modifier = Modifier.align(Alignment.TopCenter).padding(top = 16.dp))
                Text("Macro", color = Color.White.copy(alpha = 0.3f), style = MaterialTheme.typography.labelSmall, modifier = Modifier.align(Alignment.BottomCenter).padding(bottom = 16.dp))
            }
            Canvas(modifier = Modifier.matchParentSize()) { drawLine(Color.Yellow, Offset(size.width - 15f, size.height / 2), Offset(size.width, size.height / 2), strokeWidth = 8f) }
        } else {
            LazyRow(state = listState, flingBehavior = snapBehavior, modifier = Modifier.fillMaxSize(), verticalAlignment = Alignment.CenterVertically, contentPadding = PaddingValues(horizontal = paddingDp)) {
                itemsIndexed(items) { index, item ->
                    val isCenter = index == centerItemIndex
                    val isTick = item.second == "|"
                    val color by animateColorAsState(if (isCenter) Color.Yellow else if (isTick) Color.White.copy(alpha = 0.5f) else Color.White, label = "color")
                    val scale by animateFloatAsState(if (isCenter) 1.2f else 1.0f, label = "scale")
                    Box(modifier = Modifier.width(itemSizeDp).fillMaxHeight(), contentAlignment = Alignment.Center) { Text(item.second, color = color, fontWeight = if (isCenter && !isTick) FontWeight.Bold else FontWeight.Normal, style = MaterialTheme.typography.labelMedium, modifier = Modifier.scale(scale)) }
                }
            }
            if (isFocusWheel) {
                Text("Inf", color = Color.White.copy(alpha = 0.3f), style = MaterialTheme.typography.labelSmall, modifier = Modifier.align(Alignment.CenterStart).padding(start = 16.dp))
                Text("Macro", color = Color.White.copy(alpha = 0.3f), style = MaterialTheme.typography.labelSmall, modifier = Modifier.align(Alignment.CenterEnd).padding(end = 16.dp))
            }
            Canvas(modifier = Modifier.matchParentSize()) { drawLine(Color.Yellow, Offset(size.width / 2, size.height - 15f), Offset(size.width / 2, size.height), strokeWidth = 8f) }
        }
    }
}

@SuppressLint("MissingPermission")
@Composable
fun CaptureButtons(context: android.content.Context, imageCapture: ImageCapture?, videoCapture: VideoCapture<Recorder>?, recording: Recording?, isVideoPaused: Boolean, onRecordingChange: (Recording?) -> Unit, onPauseChange: (Boolean) -> Unit) {
    Box(
        modifier = Modifier.size(50.dp).clip(CircleShape).background(if (recording != null) Color.Red else Color.White.copy(alpha = 0.2f)).clickable {
            if (recording != null) { recording.stop(); onRecordingChange(null) } else {
                val outputFile = File(context.getExternalFilesDir(null), "VIDEO_${System.currentTimeMillis()}.mp4")
                val options = FileOutputOptions.Builder(outputFile).build()
                val newRecording = videoCapture?.output?.prepareRecording(context, options)?.withAudioEnabled()?.start(ContextCompat.getMainExecutor(context)) { event ->
                    if (event is VideoRecordEvent.Finalize) Toast.makeText(context, "Video Saved", Toast.LENGTH_SHORT).show()
                }
                onRecordingChange(newRecording); onPauseChange(false)
            }
        }, contentAlignment = Alignment.Center
    ) {
        if (recording != null) { Box(modifier = Modifier.size(20.dp).clip(RoundedCornerShape(4.dp)).background(Color.White)) } else { Icon(Icons.Default.Videocam, contentDescription = "Video", tint = Color.White) }
    }

    IconButton(
        onClick = {
            val outputFile = File(context.getExternalFilesDir(null), "PHOTO_${System.currentTimeMillis()}.jpg")
            val outputOptions = ImageCapture.OutputFileOptions.Builder(outputFile).build()
            imageCapture?.takePicture(outputOptions, ContextCompat.getMainExecutor(context), object : ImageCapture.OnImageSavedCallback {
                override fun onImageSaved(output: ImageCapture.OutputFileResults) { Toast.makeText(context, "Photo Saved", Toast.LENGTH_SHORT).show() }
                override fun onError(exc: ImageCaptureException) { Log.e("CameraX", "Photo capture failed", exc) }
            })
        },
        modifier = Modifier.size(72.dp).background(Color.White.copy(alpha = 0.5f), CircleShape)
    ) { Box(modifier = Modifier.size(56.dp).clip(CircleShape).background(Color.White)) }

    IconButton(
        onClick = { if (recording != null) { if (isVideoPaused) { recording.resume(); onPauseChange(false) } else { recording.pause(); onPauseChange(true) } } },
        enabled = recording != null,
        modifier = Modifier.size(50.dp).background(if (recording != null) Color.White.copy(alpha = 0.2f) else Color.Transparent, CircleShape)
    ) { if (recording != null) Icon(if (isVideoPaused) Icons.Default.PlayArrow else Icons.Default.Pause, contentDescription = "Pause", tint = Color.White) }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun AudioModeScreen(onNavigateBack: () -> Unit) {
    val context = LocalContext.current
    var hasMicPermission by remember { mutableStateOf(ContextCompat.checkSelfPermission(context, Manifest.permission.RECORD_AUDIO) == PackageManager.PERMISSION_GRANTED) }
    val permissionLauncher = rememberLauncherForActivityResult(ActivityResultContracts.RequestPermission()) { hasMicPermission = it }

    LaunchedEffect(Unit) { if (!hasMicPermission) permissionLauncher.launch(Manifest.permission.RECORD_AUDIO) }
    if (!hasMicPermission) { PermissionDeniedScreen("Microphone", onNavigateBack) { permissionLauncher.launch(Manifest.permission.RECORD_AUDIO) }; return }

    Scaffold(topBar = { TopAppBar(title = { Text("Acoustic Module") }, navigationIcon = { IconButton(onClick = onNavigateBack) { Icon(Icons.AutoMirrored.Filled.ArrowBack, contentDescription = "Back") } }) }) { paddingValues ->
        Column(modifier = Modifier.padding(paddingValues).fillMaxSize()) { RecordTab(context) }
    }
}

@SuppressLint("MissingPermission")
@Composable
fun RecordTab(context: android.content.Context) {
    var isRecording by remember { mutableStateOf(false) }
    var recordTime by remember { mutableIntStateOf(0) }
    var recordingJob by remember { mutableStateOf<Job?>(null) }
    val scope = rememberCoroutineScope()

    LaunchedEffect(isRecording) { while (isRecording) { delay(1000); recordTime++ } }
    DisposableEffect(Unit) { onDispose { recordingJob?.cancel() } }

    val infiniteTransition = rememberInfiniteTransition(label = "pulse")
    val scale by infiniteTransition.animateFloat(initialValue = 1f, targetValue = if (isRecording) 1.2f else 1f, animationSpec = infiniteRepeatable(animation = tween(800, easing = LinearEasing), repeatMode = RepeatMode.Reverse), label = "pulse_anim")

    Column(modifier = Modifier.fillMaxSize(), verticalArrangement = Arrangement.Center, horizontalAlignment = Alignment.CenterHorizontally) {
        Text(text = String.format(Locale.US, "%02d:%02d", recordTime / 60, recordTime % 60), style = MaterialTheme.typography.displayLarge, fontWeight = FontWeight.Light, color = if (isRecording) MaterialTheme.colorScheme.error else MaterialTheme.colorScheme.onSurface)
        Spacer(modifier = Modifier.height(48.dp))
        Row(horizontalArrangement = Arrangement.spacedBy(24.dp), verticalAlignment = Alignment.CenterVertically) {
            Box(
                modifier = Modifier.size(100.dp).scale(scale).clip(CircleShape).background(if (isRecording) MaterialTheme.colorScheme.error else MaterialTheme.colorScheme.primary).clickable {
                    if (!isRecording) {
                        val outputFile = File(context.getExternalFilesDir(null), "AUDIO_${System.currentTimeMillis()}.wav")
                        recordingJob = scope.launch(Dispatchers.IO) {
                            val sampleRate = 44100
                            val channelConfig = AudioFormat.CHANNEL_IN_MONO
                            val audioFormat = AudioFormat.ENCODING_PCM_16BIT
                            val bufferSize = AudioRecord.getMinBufferSize(sampleRate, channelConfig, audioFormat)
                            val audioRecord = AudioRecord(MediaRecorder.AudioSource.MIC, sampleRate, channelConfig, audioFormat, bufferSize)

                            if (audioRecord.state != AudioRecord.STATE_INITIALIZED) return@launch
                            val out = FileOutputStream(outputFile)
                            out.write(ByteArray(44))
                            val buffer = ByteArray(bufferSize)
                            audioRecord.startRecording()
                            var totalAudioLen = 0L

                            try {
                                while (isActive) {
                                    val read = audioRecord.read(buffer, 0, bufferSize)
                                    if (read > 0) { out.write(buffer, 0, read); totalAudioLen += read }
                                }
                            } finally {
                                audioRecord.stop(); audioRecord.release()
                                val totalDataLen = totalAudioLen + 36
                                val byteRate = 16 * sampleRate * 1 / 8
                                val header = ByteArray(44)
                                header[0] = 'R'.code.toByte(); header[1] = 'I'.code.toByte(); header[2] = 'F'.code.toByte(); header[3] = 'F'.code.toByte()
                                header[4] = (totalDataLen and 0xff).toByte(); header[5] = ((totalDataLen shr 8) and 0xff).toByte(); header[6] = ((totalDataLen shr 16) and 0xff).toByte(); header[7] = ((totalDataLen shr 24) and 0xff).toByte()
                                header[8] = 'W'.code.toByte(); header[9] = 'A'.code.toByte(); header[10] = 'V'.code.toByte(); header[11] = 'E'.code.toByte()
                                header[12] = 'f'.code.toByte(); header[13] = 'm'.code.toByte(); header[14] = 't'.code.toByte(); header[15] = ' '.code.toByte()
                                header[16] = 16; header[17] = 0; header[18] = 0; header[19] = 0
                                header[20] = 1; header[21] = 0; header[22] = 1; header[23] = 0
                                header[24] = (sampleRate and 0xff).toByte(); header[25] = ((sampleRate shr 8) and 0xff).toByte(); header[26] = 0; header[27] = 0
                                header[28] = (byteRate and 0xff).toByte(); header[29] = ((byteRate shr 8) and 0xff).toByte(); header[30] = 0; header[31] = 0
                                header[32] = (1 * 16 / 8).toByte(); header[33] = 0; header[34] = 16; header[35] = 0
                                header[36] = 'd'.code.toByte(); header[37] = 'a'.code.toByte(); header[38] = 't'.code.toByte(); header[39] = 'a'.code.toByte()
                                header[40] = (totalAudioLen and 0xff).toByte(); header[41] = ((totalAudioLen shr 8) and 0xff).toByte(); header[42] = ((totalAudioLen shr 16) and 0xff).toByte(); header[43] = ((totalAudioLen shr 24) and 0xff).toByte()
                                val randomAccessFile = java.io.RandomAccessFile(outputFile, "rw")
                                randomAccessFile.seek(0); randomAccessFile.write(header); randomAccessFile.close(); out.close()
                            }
                        }
                        isRecording = true
                    } else {
                        recordingJob?.cancel(); recordingJob = null; isRecording = false; recordTime = 0; Toast.makeText(context, "Audio Saved (.wav)", Toast.LENGTH_SHORT).show()
                    }
                }, contentAlignment = Alignment.Center
            ) { Icon(imageVector = if (!isRecording) Icons.Default.Mic else Icons.Default.Stop, contentDescription = "Record", tint = Color.White, modifier = Modifier.size(40.dp)) }
        }
        Spacer(modifier = Modifier.height(24.dp))
        Text(text = if (isRecording) "Recording..." else "Tap to start", style = MaterialTheme.typography.bodyMedium, color = MaterialTheme.colorScheme.onSurfaceVariant)
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun GalleryScreen(onNavigateBack: () -> Unit, onNavigateToTranscription: (String) -> Unit) {
    val context = LocalContext.current
    var selectedTab by remember { mutableIntStateOf(0) }
    val tabs = listOf("Images & Videos", "Audio")

    Scaffold(topBar = { TopAppBar(title = { Text("Gallery") }, navigationIcon = { IconButton(onClick = onNavigateBack) { Icon(Icons.AutoMirrored.Filled.ArrowBack, contentDescription = "Back") } }) }) { paddingValues ->
        Column(modifier = Modifier.padding(paddingValues).fillMaxSize()) {
            PrimaryTabRow(selectedTabIndex = selectedTab) { tabs.forEachIndexed { index, title -> Tab(selected = selectedTab == index, onClick = { selectedTab = index }, text = { Text(title, fontWeight = FontWeight.SemiBold) }) } }
            AnimatedContent(targetState = selectedTab, label = "GalleryTabs") { tab ->
                when (tab) {
                    0 -> VisualLibraryTab(context, onNavigateToTranscription)
                    1 -> AudioLibraryTab(context, onNavigateToTranscription)
                }
            }
        }
    }
}

@Composable
fun VisualLibraryTab(context: android.content.Context, onNavigateToTranscription: (String) -> Unit) {
    val mediaDir = context.getExternalFilesDir(null)
    val visuals = mediaDir?.listFiles { file -> file.extension == "jpg" || file.extension == "mp4" }?.sortedByDescending { it.lastModified() }?.map { it.name } ?: emptyList()

    if (visuals.isEmpty()) { Box(modifier = Modifier.fillMaxSize(), contentAlignment = Alignment.Center) { Text("No photos or videos found.") } } else {
        LazyColumn(modifier = Modifier.fillMaxSize().padding(16.dp)) {
            items(visuals) { fileName ->
                val isVideo = fileName.endsWith(".mp4")
                ElevatedCard(modifier = Modifier.fillMaxWidth().padding(bottom = 12.dp), onClick = { onNavigateToTranscription(File(mediaDir, fileName).absolutePath) }, colors = CardDefaults.elevatedCardColors(containerColor = MaterialTheme.colorScheme.surface)) {
                    Row(modifier = Modifier.padding(16.dp), verticalAlignment = Alignment.CenterVertically) {
                        Icon(if (isVideo) Icons.Default.Videocam else Icons.Default.Image, contentDescription = null, tint = MaterialTheme.colorScheme.primary)
                        Spacer(modifier = Modifier.width(16.dp))
                        Column { Text(fileName, style = MaterialTheme.typography.titleMedium) }
                    }
                }
            }
        }
    }
}

@Composable
fun AudioLibraryTab(context: android.content.Context, onNavigateToTranscription: (String) -> Unit) {
    val audioDir = context.getExternalFilesDir(null)
    val recordings = audioDir?.listFiles { file -> file.extension == "wav" }?.sortedByDescending { it.lastModified() }?.map { it.name } ?: emptyList()

    if (recordings.isEmpty()) { Box(modifier = Modifier.fillMaxSize(), contentAlignment = Alignment.Center) { Text("No audio recordings found.") } } else {
        LazyColumn(modifier = Modifier.fillMaxSize().padding(16.dp)) {
            items(recordings) { fileName ->
                ElevatedCard(modifier = Modifier.fillMaxWidth().padding(bottom = 12.dp), onClick = { onNavigateToTranscription(File(audioDir, fileName).absolutePath) }, colors = CardDefaults.elevatedCardColors(containerColor = MaterialTheme.colorScheme.surface)) {
                    Row(modifier = Modifier.padding(16.dp), verticalAlignment = Alignment.CenterVertically) {
                        Icon(Icons.Default.AudioFile, contentDescription = null, tint = MaterialTheme.colorScheme.primary)
                        Spacer(modifier = Modifier.width(16.dp))
                        Column { Text(fileName, style = MaterialTheme.typography.titleMedium) }
                    }
                }
            }
        }
    }
}

// =========================================================================
// ML TRANSCRIPTION & PRESETS SCREEN
// =========================================================================

@androidx.annotation.OptIn(androidx.media3.common.util.UnstableApi::class)
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun TranscriptionScreen(filePath: String, onNavigateBack: () -> Unit) {
    val context = LocalContext.current
    val isAudio = filePath.endsWith(".wav")
    val exoPlayer = remember { ExoPlayer.Builder(context).build().apply { setMediaItem(MediaItem.fromUri(filePath)); prepare() } }

    DisposableEffect(Unit) { onDispose { exoPlayer.release() } }

    var showTranscription by remember { mutableStateOf(true) }
    var showSettings by remember { mutableStateOf(true) }

    // PyTorch States
    var effectPresets by remember { mutableStateOf<Map<String, String>?>(null) }
    var isAnalyzing by remember { mutableStateOf(false) }

    // Vosk States
    var transcriptionText by remember { mutableStateOf("") }
    var isTranscribing by remember { mutableStateOf(false) }

    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text(if (isAudio) "Audio Analytics" else "Video Analytics") },
                navigationIcon = { IconButton(onClick = onNavigateBack) { Icon(Icons.AutoMirrored.Filled.ArrowBack, contentDescription = "Back") } }
            )
        }
    ) { paddingValues ->
        Column(modifier = Modifier.fillMaxSize().padding(paddingValues).padding(16.dp)) {
            Box(modifier = Modifier.fillMaxWidth().height(if (isAudio) 80.dp else 200.dp).clip(RoundedCornerShape(12.dp)).background(Color.Black)) {
                AndroidView(factory = { ctx ->
                    PlayerView(ctx).apply {
                        player = exoPlayer
                        if (isAudio) {
                            setShowBuffering(PlayerView.SHOW_BUFFERING_NEVER)
                            controllerAutoShow = true
                            controllerShowTimeoutMs = -1
                            useArtwork = false
                        }
                    }
                }, modifier = Modifier.fillMaxSize())
            }
            Spacer(modifier = Modifier.height(16.dp))

            LazyColumn(modifier = Modifier.fillMaxSize()) {
                item {
                    Text("File: ${File(filePath).name}", style = MaterialTheme.typography.bodyMedium)
                    Spacer(modifier = Modifier.height(16.dp))
                }

                if (!isAudio) {
                    item {
                        Box(modifier = Modifier.fillMaxWidth().background(MaterialTheme.colorScheme.surfaceVariant, RoundedCornerShape(8.dp)).padding(16.dp)) {
                            Text("Transcription and Acoustic Post-Production Presets are only available for .wav audio recordings.", style = MaterialTheme.typography.bodyMedium, color = MaterialTheme.colorScheme.onSurfaceVariant)
                        }
                    }
                } else {
                    item {
                        ElevatedCard(onClick = { showTranscription = !showTranscription }, modifier = Modifier.fillMaxWidth().padding(bottom = 8.dp)) {
                            Row(modifier = Modifier.padding(16.dp).fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
                                Text("Transcription", style = MaterialTheme.typography.titleMedium, color = MaterialTheme.colorScheme.primary)
                                Icon(if (showTranscription) Icons.Default.ExpandLess else Icons.Default.ExpandMore, contentDescription = null)
                            }
                        }
                    }
                    item {
                        AnimatedVisibility(visible = showTranscription) {
                            Box(modifier = Modifier.fillMaxWidth().background(MaterialTheme.colorScheme.surfaceVariant, RoundedCornerShape(8.dp)).padding(16.dp)) {
                                if (transcriptionText.isNotEmpty()) {
                                    Text(transcriptionText, style = MaterialTheme.typography.bodyMedium, color = MaterialTheme.colorScheme.onSurfaceVariant)
                                } else {
                                    Button(
                                        onClick = {
                                            isTranscribing = true
                                            loadVoskModel(context,
                                                { model ->
                                                    kotlinx.coroutines.CoroutineScope(Dispatchers.IO).launch {
                                                        try {
                                                            val recognizer = Recognizer(model, 44100f)
                                                            val fis = FileInputStream(File(filePath))
                                                            fis.skip(44) // Skip the standard WAV header

                                                            val buffer = ByteArray(4096)
                                                            var bytesRead: Int
                                                            val fullText = java.lang.StringBuilder()

                                                            while (fis.read(buffer).also { bytesRead = it } != -1) {
                                                                if (recognizer.acceptWaveForm(buffer, bytesRead)) {
                                                                    val result = JSONObject(recognizer.result)
                                                                    val text = result.optString("text", "")
                                                                    if (text.isNotEmpty()) fullText.append(text).append(" ")
                                                                }
                                                            }

                                                            val finalResult = JSONObject(recognizer.finalResult)
                                                            val finalText = finalResult.optString("text", "")
                                                            if (finalText.isNotEmpty()) fullText.append(finalText)

                                                            fis.close()

                                                            kotlinx.coroutines.withContext(Dispatchers.Main) {
                                                                transcriptionText = fullText.toString().ifEmpty { "No recognizable speech detected." }
                                                                isTranscribing = false
                                                            }
                                                        } catch (e: Exception) {
                                                            Log.e("VOSK", "Error during transcription", e)
                                                            kotlinx.coroutines.withContext(Dispatchers.Main) {
                                                                transcriptionText = "Error occurred during transcription."
                                                                isTranscribing = false
                                                            }
                                                        }
                                                    }
                                                },
                                                { exception ->
                                                    Log.e("VOSK", "Failed to load model", exception)
                                                    isTranscribing = false
                                                    transcriptionText = "Failed to load Vosk model from assets."
                                                }
                                            )
                                        },
                                        modifier = Modifier.fillMaxWidth(),
                                        enabled = !isTranscribing
                                    ) {
                                        Text(if (isTranscribing) "Transcribing Audio..." else "Generate Transcription")
                                    }
                                }
                            }
                        }
                        Spacer(modifier = Modifier.height(16.dp))
                    }
                    item {
                        ElevatedCard(onClick = { showSettings = !showSettings }, modifier = Modifier.fillMaxWidth().padding(bottom = 8.dp)) {
                            Row(modifier = Modifier.padding(16.dp).fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
                                Text("Acoustic Post-Production Presets", style = MaterialTheme.typography.titleMedium, color = MaterialTheme.colorScheme.primary)
                                Icon(if (showSettings) Icons.Default.ExpandLess else Icons.Default.ExpandMore, contentDescription = null)
                            }
                        }
                    }
                    item {
                        AnimatedVisibility(visible = showSettings) {
                            Box(modifier = Modifier.fillMaxWidth().background(MaterialTheme.colorScheme.surfaceVariant, RoundedCornerShape(8.dp)).padding(16.dp)) {
                                if (effectPresets != null) {
                                    Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                                        Text("Denoise: ${effectPresets!!["denoise"]}", fontWeight = FontWeight.SemiBold)
                                        Text("Dynamic: ${effectPresets!!["compressor"]}")
                                        Text("EQ: ${effectPresets!!["eq"]}")
                                        Text("De-Esser: ${effectPresets!!["deesser"]}")
                                        Text("Space: ${effectPresets!!["reverb"]}")
                                    }
                                } else {
                                    Button(
                                        onClick = {
                                            isAnalyzing = true
                                            kotlinx.coroutines.CoroutineScope(Dispatchers.IO).launch {
                                                try {
                                                    val py = Python.getInstance()
                                                    val audioProcessor = py.getModule("audio_preprocessor").callAttr("AudioPreprocessor")
                                                    val pythonBytes = audioProcessor.callAttr("process_wav_to_tensor", filePath).toJava(ByteArray::class.java)

                                                    val floatBuffer = ByteBuffer.wrap(pythonBytes).order(ByteOrder.nativeOrder()).asFloatBuffer()
                                                    val spectrogramArray = FloatArray(floatBuffer.remaining())
                                                    floatBuffer.get(spectrogramArray)

                                                    val modelPath = assetFilePath(context, "voxeffects_mobile.ptl")
                                                    val module = LiteModuleLoader.load(modelPath)

                                                    val inputTensor = Tensor.fromBlob(spectrogramArray, longArrayOf(1, 1, 513, 1723))
                                                    val outputTuple = module.forward(IValue.from(inputTensor)).toTuple()

                                                    val denoiseIdx = outputTuple[0].toTensor().dataAsFloatArray[0].roundToInt()
                                                    val compIdx = outputTuple[1].toTensor().dataAsFloatArray[0].roundToInt()
                                                    val eqIdx = outputTuple[2].toTensor().dataAsFloatArray[0].roundToInt()
                                                    val deesserIdx = outputTuple[3].toTensor().dataAsFloatArray[0].roundToInt()
                                                    val reverbIdx = outputTuple[4].toTensor().dataAsFloatArray[0].roundToInt()

                                                    val results = mapOf(
                                                        "denoise" to parseEffectPreset(context, "de_noise_presets", denoiseIdx),
                                                        "compressor" to parseEffectPreset(context, "dynamic_presets", compIdx),
                                                        "eq" to parseEffectPreset(context, "eq_presets", eqIdx),
                                                        "deesser" to parseEffectPreset(context, "de_esser_presets", deesserIdx),
                                                        "reverb" to parseEffectPreset(context, "reverb_presets", reverbIdx)
                                                    )

                                                    kotlinx.coroutines.withContext(Dispatchers.Main) {
                                                        effectPresets = results
                                                        isAnalyzing = false
                                                    }
                                                } catch (e: Exception) {
                                                    Log.e("ML_PIPELINE", "Error processing audio", e)
                                                    kotlinx.coroutines.withContext(Dispatchers.Main) { isAnalyzing = false }
                                                }
                                            }
                                        },
                                        modifier = Modifier.fillMaxWidth(),
                                        enabled = !isAnalyzing
                                    ) {
                                        Text(if (isAnalyzing) "Processing Spectrogram & Analyzing..." else "Generate Master Presets")
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun PermissionDeniedScreen(type: String, onNavigateBack: () -> Unit, onRequestPermission: () -> Unit) {
    Scaffold(
        topBar = { TopAppBar(title = { Text("Permission Required") }, navigationIcon = { IconButton(onClick = onNavigateBack) { Icon(Icons.AutoMirrored.Filled.ArrowBack, contentDescription = "Back") } }) }
    ) { paddingValues ->
        Column(modifier = Modifier.fillMaxSize().padding(paddingValues).padding(24.dp), horizontalAlignment = Alignment.CenterHorizontally, verticalArrangement = Arrangement.Center) {
            Icon(Icons.Default.Warning, contentDescription = null, modifier = Modifier.size(64.dp), tint = MaterialTheme.colorScheme.error)
            Spacer(modifier = Modifier.height(16.dp))
            Text("We need $type permission to use this feature.", textAlign = TextAlign.Center)
            Spacer(modifier = Modifier.height(24.dp))
            Button(onClick = onRequestPermission) { Text("Grant Permission") }
        }
    }
}

fun loadVoskModel(context: android.content.Context, onModelReady: (org.vosk.Model) -> Unit, onError: (Exception) -> Unit) {
    kotlinx.coroutines.CoroutineScope(Dispatchers.IO).launch {
        try {
            val modelDir = File(context.filesDir, "model")
            if (!modelDir.exists() || modelDir.listFiles().isNullOrEmpty()) {
                if (modelDir.exists()) modelDir.deleteRecursively()
                modelDir.mkdirs()
                copyAssetRecursive(context, "model-en", modelDir)
            }
            val model = org.vosk.Model(modelDir.absolutePath)
            kotlinx.coroutines.withContext(Dispatchers.Main) {
                onModelReady(model)
            }
        } catch (e: Exception) {
            Log.e("VOSK", "Failed to load model", e)
            kotlinx.coroutines.withContext(Dispatchers.Main) {
                onError(e)
            }
        }
    }
}

fun copyAssetRecursive(context: android.content.Context, assetPath: String, targetDir: File) {
    val assets = context.assets
    val files = assets.list(assetPath)
    if (files == null || files.isEmpty()) {
        assets.open(assetPath).use { input ->
            FileOutputStream(targetDir).use { output ->
                input.copyTo(output)
            }
        }
    } else {
        if (!targetDir.exists()) targetDir.mkdirs()
        for (file in files) {
            val subAssetPath = "$assetPath/$file"
            val subTargetFile = File(targetDir, file)
            copyAssetRecursive(context, subAssetPath, subTargetFile)
        }
    }
}