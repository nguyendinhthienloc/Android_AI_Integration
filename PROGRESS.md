# InkClear Implementation Progress

This file is the durable checkpoint log requested by the project owner. It is updated after each implementation step.

## Step 1 — Branch and project audit (complete)

- Branch: codex/pretrained-neural-clean
- Confirmed Java/XML Android application and existing classical enhancer.
- Confirmed original minimum SDK 25, target SDK 36, compile SDK 37.1.
- Preserved existing scan, image import, JPEG, and PDF workflows.
- Created the initial implementation plan in TODO.md.
- Created UI/icon concepts in design/.

## Step 2 — Pretrained Neural Clean model (complete)

- Evaluated the larger SBB SavedModel and rejected it because conversion required TensorFlow Flex and remained about 75 MB.
- Selected the Apache-2.0 pretrained sliedes/binarize document-segmentation model.
- Converted it to a native dynamically weight-quantized LiteRT file: 6,906,912 bytes.
- Validated conversion against the original TensorFlow model on DIBCO 2009 handwritten sample dibco09_hand_H01.png.
- Validation: mean absolute logit difference 0.10663; maximum difference 0.67612.
- Bundled asset: app/src/main/assets/models/neural_clean.tflite.
- Added tiled inference, 896 px mobile bound, strength rendering, and classical fallback.
- Added exact license and model card.

## Step 3 — Offline OCR engine (complete)

- Vendored official PaddleOCR Android SDK revision 2661c7c0ef5c613e8f93c6e93b2e052399f0f854.
- Bundled PP-OCRv5 Mobile ONNX detection and recognition models.
- Added ONNX Runtime Android 1.21.1 and official OpenCV Android 4.13.0.
- Raised minimum SDK to API 26 as required by the official SDK.
- Added a lifecycle-safe, cancellable OfflineOcrEngine wrapper.
- Recognition is English-first and starts only after the user taps Recognize text.
- The merged manifest explicitly removes INTERNET and ACCESS_NETWORK_STATE.

## Step 4 — Dedicated OCR result experience (complete)

- Added OcrResultActivity.
- Added editable recognized text, source preview, line/runtime/confidence metadata, character count, Copy, UTF-8 TXT save, and Recognize again.
- Made the screen scroll-safe for compact displays and accessibility scaling.
- Added stale-result and cancellation guards when image/enhancement state changes.

## Step 5 — Visual design (complete)

- Added the navy/blue InkClear visual system.
- Added Neural Clean and Offline status chips.
- Replaced the adaptive launcher foreground/background with the approved paper, handwriting, clean-line, and sparkle concept.
- Polished home, enhancement, and result layouts.

## Step 6 — Automated tests (complete)

Local unit tests:

- OCR horizontal whitespace and paragraph normalization.
- Empty and Unicode text handling.
- Neural sigmoid stability.

Instrumented emulator tests:

- Bundled model assets exist.
- Merged app requests no INTERNET permission.
- Recognize text begins disabled without an image.
- OCR result text is visible, editable, and updates character count.
- Real bundled PP-OCRv5 detection/recognition runs end to end offline.
- Real bundled LiteRT Neural Clean inference runs end to end offline.

Device result: 5/5 tests passed on Resizable_Experimental AVD.

Test fixtures:

- Neural conversion: DIBCO 2009 handwritten sample dibco09_hand_H01.png from the cleanup-model repository.
- Android OCR: official PaddleOCR android_ocr_benchmark_reference.png.
- UI/formatting: synthetic English and Unicode strings.

Important limitation: these fixtures prove runtime integration and conversion fidelity; they are not a statistically meaningful handwriting-accuracy benchmark. IAM-style English handwriting data plus consented phone photos should be used for a future CER/WER evaluation.

## Step 7 — Documentation (complete)

Completed:

- README usage, architecture, privacy, tests, and limitations.
- docs/MODELS.md model provenance, hashes, conversion notes, and known failure cases.
- THIRD_PARTY_NOTICES.md and exact upstream Apache-2.0 license files.

Also completed:

- TODO.md now records agreed decisions, delivered tasks, test data, acceptance criteria, and the next accuracy milestone.
- report/report.tex was replaced with a current six-page technical report and report/report.pdf compiled successfully.
- Final verification results are recorded under Step 8.

## Step 8 — Final verification (completed)

Completed on 5 August 2026:

- `gradlew :app:lintDebug`: BUILD SUCCESSFUL after fixing three AndroidX tint errors; the final report has zero errors and 34 non-blocking maintenance advisories.
- `gradlew test assembleDebug`: BUILD SUCCESSFUL.
- `gradlew :app:connectedDebugAndroidTest`: BUILD SUCCESSFUL; 5/5 tests passed on the API 35 `Resizable_Experimental` emulator, including real PP-OCRv5 and LiteRT inference.
- Packaged and merged manifests contain no `INTERNET` or `ACCESS_NETWORK_STATE` permission. The only generated permission is AndroidX's app-scoped dynamic-receiver permission.
- Debug APK: `app/build/outputs/apk/debug/app-debug.apk`, 329,587,165 bytes (about 314.3 MiB).
- Technical report: `report/report.pdf`, six pages and 222,949 bytes; compiled without overfull boxes and visually checked page by page.
- Final Git audit: `git diff --check` passed; status contains only feature-scoped source, tests, models, vendored SDK, UI assets, and documentation. Generated LaTeX and visual-QA temporary files were removed.