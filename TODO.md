# InkClear Neural Clean and Offline OCR — Implementation Plan

Status: implementation complete; final verification in progress.

## Agreed product decisions

- [x] English-first recognition.
- [x] Fully offline inference after installation.
- [x] No model training required on the user's computer.
- [x] Pretrained models bundled with the application.
- [x] OCR starts only after the user taps Recognize text.
- [x] Dedicated OCR result screen.
- [x] Recognized text is editable.
- [x] Copy action included.
- [x] UTF-8 .txt export is sufficient for this release.
- [x] Existing JPEG and image-only PDF exports remain.
- [x] Minimum SDK may be upgraded when necessary; it is now API 26.
- [x] New favicon/launcher icon and UI polish included.
- [x] Existing deterministic enhancer remains as a failure fallback.
- [x] App manifest contains no INTERNET permission.

## Release goal

Turn InkClear into an on-device handwritten-document assistant that:

1. cleans paper/shadows with a pretrained segmentation network;
2. preserves the existing classical enhancement fallback;
3. detects and recognizes handwriting with bundled PP-OCRv5 models;
4. lets the user review and edit the output;
5. exports plain text without uploading images or text.

## Delivered architecture

### Neural cleanup

- [x] Evaluate candidate pretrained document-cleaning models.
- [x] Reject the SBB model because it requires TensorFlow Flex and remains too large.
- [x] Select the Apache-2.0 sliedes/binarize pretrained network.
- [x] Convert TensorFlow SavedModel to a 6.9 MB dynamic-range-quantized LiteRT model.
- [x] Validate converted logits against the original TensorFlow output.
- [x] Bundle neural_clean.tflite.
- [x] Implement 224 px tiled inference.
- [x] Cap the mobile inference canvas at 896 px.
- [x] Render Natural, Grayscale, and B&W outputs.
- [x] Fall back to HandwritingEnhancer when model loading/inference fails.
- [x] Run inference away from the main thread.
- [x] Release the interpreter with activity lifecycle.

### Offline handwriting OCR

- [x] Vendor official PaddleOCR Android SDK source.
- [x] Bundle PP-OCRv5 Mobile detector.
- [x] Bundle PP-OCRv5 Mobile recognizer and character configuration.
- [x] Use ONNX Runtime Android and official OpenCV Android.
- [x] Initialize models lazily after the recognition button is tapped.
- [x] Run detection, reading-order sorting, crops, and recognition offline.
- [x] Normalize OCR line spacing without changing words.
- [x] Cancel superseded recognition.
- [x] Ignore stale callbacks after image/enhancement changes.
- [x] Release native OCR resources.
- [x] Enforce removal of INTERNET and ACCESS_NETWORK_STATE from the merged manifest.

### OCR result screen

- [x] Dedicated OcrResultActivity.
- [x] Editable multiline text.
- [x] Source image preview when URI access is available.
- [x] Offline status label.
- [x] Line count, elapsed time, and average confidence metadata.
- [x] Character count that updates after editing.
- [x] Copy to clipboard.
- [x] Save UTF-8 .txt through Android's document picker.
- [x] Recognize again action.
- [x] Preserve edits through activity recreation.
- [x] Scroll-safe layout for compact screens and large display scaling.
- [x] Clear empty-result guidance.

### UI and identity

- [x] New adaptive launcher icon based on paper, handwriting, clean lines, and sparkle.
- [x] Navy/blue InkClear color system.
- [x] Neural Clean and Offline chips.
- [x] Updated home screen.
- [x] Updated enhancement controls.
- [x] Dedicated results design.
- [x] Accessible content descriptions and 44–52 dp action targets.
- [x] Generated visual concepts retained in design/ for review.

## Test plan and status

### Local unit tests

- [x] Collapse repeated horizontal whitespace.
- [x] Preserve paragraph breaks.
- [x] Trim leading/trailing blank lines.
- [x] Preserve Unicode text.
- [x] Handle empty line lists.
- [x] Verify numerically stable sigmoid behavior.

### Instrumented tests

- [x] Verify all bundled model assets exist.
- [x] Verify merged package requests no INTERNET permission.
- [x] Verify Recognize text is disabled before an image is available.
- [x] Verify recognized text is displayed and editable.
- [x] Verify edited character count.
- [x] Run real PP-OCRv5 detector/recognizer end to end offline.
- [x] Run real Neural Clean LiteRT inference end to end offline.
- [x] Five instrumented tests pass on the Resizable_Experimental AVD.

### Model fixtures

- Neural conversion fixture: DIBCO 2009 handwritten image dibco09_hand_H01.png from the cleanup-model repository.
- Android OCR fixture: official PaddleOCR android_ocr_benchmark_reference.png.
- Formatter/UI fixtures: synthetic English and Unicode text.

These fixtures validate conversion and runtime integration. They do not constitute a broad OCR accuracy benchmark.

## Acceptance criteria

- [x] A selected page is processed by a real pretrained neural network.
- [x] Classical fallback preserves the old enhancement workflow.
- [x] OCR starts only from an explicit tap.
- [x] OCR model files are bundled and run without network access.
- [x] A dedicated screen allows review and editing.
- [x] Copy and UTF-8 TXT export work.
- [x] New image/settings invalidate stale OCR work.
- [x] JPEG and PDF exports remain available.
- [x] No storage permission is needed.
- [x] No INTERNET permission is present.
- [x] Unit tests pass.
- [x] Instrumented model/UI/privacy tests pass.
- [x] Model provenance, hashes, licenses, and limitations are documented.
- [x] Final lint task passes.
- [x] Final debug APK build passes after all documentation/code cleanup.
- [x] Final diff/status review is complete.

## Documentation deliverables

- [x] README.md
- [x] docs/MODELS.md
- [x] THIRD_PARTY_NOTICES.md
- [x] Exact Apache-2.0 license texts in licenses/
- [x] PROGRESS.md checkpoint log
- [x] This implementation-oriented TODO.md
- [x] Updated report/report.tex
- [x] Final verification results appended to PROGRESS.md

## Known limitations

- Accuracy varies with writing style, blur, glare, perspective, and line overlap.
- English is the release target; incidental support for other characters is not guaranteed.
- Neural cleanup downsizes its inference canvas to 896 px for predictable mobile performance.
- One page is processed at a time.
- TXT is available, but searchable PDF is not included.
- Model/runtime libraries increase APK size.
- Google Play services document scanning may be unavailable; Choose image remains the fallback.

## Recommended next evaluation milestone

- [ ] Obtain IAM Handwriting Database access for English line/word evaluation.
- [ ] Build a consented phone-photo set with shadows, ruled paper, pencil, pen, and cursive.
- [ ] Split evaluation by writer.
- [ ] Record character error rate and word error rate.
- [ ] Compare original, classical, and Neural Clean OCR inputs.
- [ ] Measure latency and peak memory on one lower-end API 26–28 device and one current physical device.
- [ ] Decide whether to add confidence highlighting or an experimental label.

## Definition of done

This feature is done when final lint/build verification passes, PROGRESS.md records the results, and the branch diff contains no unrelated changes.