# InkClear

InkClear is an Android app for privately cleaning and recognizing photographed handwritten notes. It combines a pretrained neural document-cleaning model with a handwriting-aware PP-OCRv5 pipeline. Both models are bundled in the APK and run on-device.

## Features

- Import an image or open the existing one-page document scanner.
- Neural Clean removes uneven paper backgrounds and shadows with a pretrained segmentation network.
- Natural, Grayscale, and Black & White rendering modes.
- Classical enhancement remains an automatic fallback if neural inference cannot start.
- Tap Recognize text to run bundled PP-OCRv5 detection and recognition.
- Dedicated, scroll-safe results screen with editable text, confidence/runtime metadata, Copy, retry, and UTF-8 .txt export.
- Save the cleaned page as JPEG or image-only PDF.
- New InkClear launcher icon and a consistent navy/blue visual system.

## Privacy and offline behavior

The app manifest explicitly removes INTERNET and ACCESS_NETWORK_STATE, including permissions contributed by transitive dependencies. Neural cleanup, OCR, text editing, and exports therefore operate without network access. Images and recognized text are not uploaded or logged.

The optional Google document-scanner UI is supplied by Google Play services and may be unavailable on some devices. Choose image is always the fallback. The app itself has no network permission.

## Requirements

- Android Studio with Android SDK Platform 37 installed
- JDK 17 or 21
- Android 8.0 / API 26 or newer
- Internet access only for the developer machine's first Gradle dependency download

The minimum SDK was raised from 25 to 26 because the official PaddleOCR Android SDK requires API 26.

## Build

On Windows PowerShell:

~~~powershell
$env:JAVA_HOME = "C:\Program Files\Eclipse Adoptium\jdk-21.0.11.10-hotspot"
.\gradlew.bat assembleDebug
~~~

On macOS/Linux, use ./gradlew. The APK is produced at app/build/outputs/apk/debug/app-debug.apk.

## Use

1. Open InkClear and tap Open Neural Clean.
2. Choose a handwritten image or use AI scan.
3. Wait for the Neural Clean badge, then compare Original and Neural Clean.
4. Select Natural, Grayscale, or B&W and adjust Strength if useful.
5. Tap Recognize text.
6. Review and edit the text on the dedicated result screen.
7. Tap Copy, Save as TXT, or Recognize again.

OCR is English-first for this release. The bundled PP-OCRv5 recognizer can decode additional characters, but they are not part of the release acceptance target.

## Architecture

~~~text
Image URI
  -> bounded bitmap decode (max 1800 px)
  -> NeuralDocumentEnhancer / LiteRT (224 px tiles, max 896 px inference canvas)
  -> classical HandwritingEnhancer fallback
  -> enhanced preview and JPEG/PDF export
  -> OfflineOcrEngine
       -> PP-OCRv5 mobile detector (ONNX Runtime)
       -> reading-order crop + recognizer (ONNX Runtime/OpenCV)
       -> OcrTextFormatter
       -> editable OcrResultActivity
       -> clipboard or UTF-8 TXT
~~~

Heavy work runs away from the main thread. Enhancement and OCR each use request-generation/cancellation guards, so a late result cannot replace a newer image. The OCR wrapper owns and releases its native ONNX resources.

Important source files:

- NeuralDocumentEnhancer.java — tiled LiteRT cleanup inference and rendering
- HandwritingEnhancer.java — deterministic fallback enhancer
- OfflineOcrEngine.kt — lifecycle-safe PaddleOCR wrapper
- OcrTextFormatter.java — pure line/whitespace normalization
- InkClearActivity.java — import, enhancement, recognition, and export workflow
- OcrResultActivity.java — editable text, Copy, retry, and TXT export
- ppocr-sdk/ — vendored official PaddleOCR Android SDK source

See [docs/MODELS.md](docs/MODELS.md) for model provenance, checksums, conversion validation, and limitations.

## Tests

~~~powershell
.\gradlew.bat test
.\gradlew.bat :app:assembleDebugAndroidTest
.\gradlew.bat :app:connectedDebugAndroidTest
.\gradlew.bat lint
.\gradlew.bat assembleDebug
~~~

Coverage includes OCR text normalization, neural math, bundled model/privacy checks, result editing, and real on-device inference through both LiteRT Neural Clean and PP-OCRv5.

## Limitations

- Handwriting OCR is probabilistic. Cursive, overlapping strokes, unusual abbreviations, low-resolution pencil, and severe perspective can still produce errors.
- The user must review recognized text before relying on or exporting it.
- Neural cleanup uses a bounded 896 px inference canvas for mobile responsiveness; very tiny writing may lose detail.
- OCR handles one page at a time and exports plain text, not a searchable PDF.
- Model and native runtime assets increase APK size.
- First inference may be slower while native runtimes initialize.

## Licenses

Bundled model and SDK notices are in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md), with complete upstream Apache-2.0 texts in the licenses directory.