package mobile_app.android_ai_integration

import android.Manifest
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.BitmapFactory
import androidx.test.core.app.ActivityScenario
import androidx.test.espresso.Espresso.onView
import androidx.test.espresso.action.ViewActions.replaceText
import androidx.test.espresso.action.ViewActions.scrollTo
import androidx.test.espresso.assertion.ViewAssertions.matches
import androidx.test.espresso.matcher.ViewMatchers.isEnabled
import androidx.test.espresso.matcher.ViewMatchers.withId
import androidx.test.espresso.matcher.ViewMatchers.withText
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.hamcrest.Matchers.not
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit

@RunWith(AndroidJUnit4::class)
class InkClearInstrumentedTest {

    private val instrumentation = InstrumentationRegistry.getInstrumentation()
    private val context: Context
        get() = instrumentation.targetContext

    @Test
    fun bundledModelsArePresentAndAppRequestsNoInternetPermission() {
        assertTrue(context.assets.open("models/neural_clean.tflite").use { it.available() > 0 })
        assertTrue(context.assets.open("models/ocr/det/inference.onnx").use { it.available() > 0 })
        assertTrue(context.assets.open("models/ocr/rec/inference.onnx").use { it.available() > 0 })

        val permissions = context.packageManager
            .getPackageInfo(context.packageName, PackageManager.GET_PERMISSIONS)
            .requestedPermissions
            ?.toList()
            .orEmpty()
        assertFalse(permissions.contains(Manifest.permission.INTERNET))
    }

    @Test
    fun recognizeButtonStartsDisabledWithoutAnImage() {
        ActivityScenario.launch(InkClearActivity::class.java).use {
            onView(withId(R.id.recognize_text_button)).check(matches(not(isEnabled())))
        }
    }

    @Test
    fun resultScreenShowsAndAllowsEditingRecognizedText() {
        val intent = Intent(context, OcrResultActivity::class.java)
            .putExtra(OcrResultActivity.EXTRA_TEXT, "hello ink")
            .putExtra(OcrResultActivity.EXTRA_LINE_COUNT, 1)
            .putExtra(OcrResultActivity.EXTRA_DURATION_MS, 420L)
            .putExtra(OcrResultActivity.EXTRA_AVERAGE_CONFIDENCE, 0.9f)

        ActivityScenario.launch<OcrResultActivity>(intent).use {
            onView(withId(R.id.recognized_text)).check(matches(withText("hello ink")))
            onView(withId(R.id.recognized_text))
                .perform(scrollTo(), replaceText("edited text"))
                .check(matches(withText("edited text")))
            onView(withId(R.id.character_count)).check(matches(withText("11 characters")))
        }
    }

    @Test
    fun bundledOcrModelsRunEndToEndOffline() {
        val bitmap = instrumentation.context.assets.open("ocr_reference.png").use {
            BitmapFactory.decodeStream(it)
        }
        val engine = OfflineOcrEngine(context)
        val completed = CountDownLatch(1)
        var result: OfflineOcrEngine.Result? = null
        var failure: Throwable? = null

        engine.recognize(bitmap, object : OfflineOcrEngine.Callback {
            override fun onSuccess(value: OfflineOcrEngine.Result) {
                result = value
                completed.countDown()
            }

            override fun onFailure(error: Throwable) {
                failure = error
                completed.countDown()
            }
        })

        assertTrue("OCR did not finish within 45 seconds", completed.await(45, TimeUnit.SECONDS))
        assertNull("OCR failed: ${failure?.message}", failure)
        assertTrue("OCR should detect at least one line", (result?.lineCount ?: 0) > 0)
        assertTrue("OCR should return non-empty text", result?.text?.isNotBlank() == true)

        engine.close()
        bitmap.recycle()
    }

    @Test
    fun neuralCleanModelRunsEndToEndOffline() {
        val original = instrumentation.context.assets.open("ocr_reference.png").use {
            BitmapFactory.decodeStream(it)
        }
        val source = android.graphics.Bitmap.createScaledBitmap(original, 224, 224, true)
        val enhancer = NeuralDocumentEnhancer(context)
        val output = enhancer.enhance(source, HandwritingEnhancer.Mode.GRAYSCALE, 60)

        assertTrue(output.width == source.width && output.height == source.height)
        assertFalse("Neural output should differ from the source", output.sameAs(source))

        enhancer.close()
        output.recycle()
        if (source !== original) source.recycle()
        original.recycle()
    }}