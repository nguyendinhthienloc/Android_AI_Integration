package mobile_app.android_ai_integration

import android.content.Context
import android.graphics.Bitmap
import com.paddle.ocr.EngineConfig
import com.paddle.ocr.PaddleOCR
import com.paddle.ocr.PaddleOCRConfig
import com.paddle.ocr.util.OpenCVUtils
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.launch
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext

/**
 * Java-friendly lifecycle wrapper around the bundled PaddleOCR Android SDK.
 */
class OfflineOcrEngine(context: Context) {

    data class Result(
        val text: String,
        val lineCount: Int,
        val durationMs: Long,
        val averageConfidence: Float,
    )

    interface Callback {
        fun onSuccess(result: Result)
        fun onFailure(error: Throwable)
    }

    private val appContext = context.applicationContext
    private val supervisor = SupervisorJob()
    private val scope = CoroutineScope(supervisor + Dispatchers.Default)
    private val initializationMutex = Mutex()

    @Volatile
    private var engine: PaddleOCR? = null
    private var activeJob: Job? = null
    private var closed = false

    @Synchronized
    fun recognize(bitmap: Bitmap, callback: Callback) {
        check(!closed) { "OCR engine is closed" }
        activeJob?.cancel()
        activeJob = scope.launch {
            try {
                val ocr = getOrCreateEngine()
                val runResult = ocr.recognize(bitmap)
                val rawLines = runResult.results
                    .filter { it.text.isNotBlank() }
                    .map { it.text }
                val text = OcrTextFormatter.formatLines(rawLines)
                val confidence = if (runResult.results.isEmpty()) {
                    0f
                } else {
                    runResult.results.map { it.confidence }.average().toFloat()
                }
                val result = Result(
                    text = text,
                    lineCount = rawLines.size,
                    durationMs = runResult.totalTimeMs,
                    averageConfidence = confidence,
                )
                withContext(Dispatchers.Main.immediate) {
                    if (!closed) callback.onSuccess(result)
                }
            } catch (_: CancellationException) {
                // A newer recognition request replaced this one.
            } catch (error: Throwable) {
                withContext(Dispatchers.Main.immediate) {
                    if (!closed) callback.onFailure(error)
                }
            }
        }
    }

    @Synchronized
    fun cancel() {
        activeJob?.cancel()
        activeJob = null
    }

    private suspend fun getOrCreateEngine(): PaddleOCR {
        engine?.let { return it }
        return initializationMutex.withLock {
            engine?.let { return@withLock it }
            if (!OpenCVUtils.init(appContext)) {
                error("OpenCV could not be initialized")
            }
            PaddleOCR.create(
                context = appContext,
                config = PaddleOCRConfig(
                    detLimitSideLen = 64,
                    detMaxSideLimit = 1800,
                    detBoxThresh = 0.55f,
                    recScoreThresh = 0.15f,
                    recBatchSize = 1,
                ),
                engineConfig = EngineConfig(numThreads = 4),
                detModelAssetPath = "models/ocr/det/inference.onnx",
                recModelAssetPath = "models/ocr/rec/inference.onnx",
                recConfigAssetPath = "models/ocr/rec/inference.yml",
            ).also { engine = it }
        }
    }

    @Synchronized
    fun close() {
        if (closed) return
        closed = true
        activeJob?.cancel()
        activeJob = null
        val current = engine
        engine = null
        scope.launch {
            try {
                current?.release()
            } finally {
                supervisor.cancel()
            }
        }
    }
}