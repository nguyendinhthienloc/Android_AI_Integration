# InkClear Technical Evaluation Report

## Executive Summary
InkClear embeds two specialized neural pipelines directly into an Android APK for complete on-device document processing:
1. **Neural Clean (LiteRT / TFLite)**: Handles paper background cleanup, shadow suppression, and contrast rendering.
2. **Offline OCR (PP-OCRv5 via ONNX Runtime & OpenCV)**: Performs text region detection and character recognition.

Testing confirms that **Neural Clean operates correctly**, while the **Offline OCR engine requires post-processing index adjustments** to fix character transcription corruption.

---

## 1. Pipeline Component Evaluation

| Pipeline Stage | Model / Engine | Function | Status & Evaluation |
| :--- | :--- | :--- | :--- |
| **Document Cleanup** | `neural_clean.tflite` (LiteRT) | Tile-based paper/ink segmentation (Natural, Grayscale, B&W) | **PASS**: Successfully removes background noise and enhances stroke contrast without network calls. |
| **Fallback Cleanup** | `HandwritingEnhancer.java` | Classical 2D box-blur filtering | **PASS**: Reliable fallback if LiteRT initialization fails. |
| **Text Line Detection** | `inference.onnx` (`_det`) | Bounding box polygon detection for text lines | **PASS**: Accurately bounds text regions across scene and document images. |
| **Text Line Recognition** | `inference.onnx` (`_rec`) | CTC sequence prediction for detected crops | **FAIL (Fixable)**: Produces high confidence scores (90%+), but outputs garbled strings due to CTC dictionary index offset. |

---

## 2. Test Cases & Ground Truth Comparison

### Test Case A: IAM English Handwriting Sample
* **Actual Image Text**:
  * Line 1: `In mid-april Anglesey`
  * Line 2: `moved his family and`
  * Line 3: `entourage from Rome to Naples,`
  * Line 4: `there to await the arrival of`
* **Model Output**: Scrambled string predictions (`ni')[lnep...`, `iknp'deo b[iehu...`).
* **Root Cause**: Attempted text decoding on detection-only output / unaligned character dictionary.

### Test Case B: Chinese Street Sign Sample
* **Actual Image Text**:
  * Line 1: `上海斯格威铂尔大酒店`
  * Line 2: `⬅ 打浦路15号`
  * Line 3: `绿洲仕格维花园公寓`
  * Line 4: `打浦路25-29-35号 ➡`
* **Model Output**:
  * Line 1 [Score: 95.5%]: `大涉壹桥勃钽令扎士涛变`
  * Line 2 [Score: 99.1%]: `递跷-1甲`
  * Line 3 [Score: 96.8%]: `绷洛卟桥续芙呆今慨`
  * Line 4 [Score: 90.6%]: `未递跷.1.5/1甲`
* **Root Cause**: CTC decoding off-by-one index mapping error against `inference.yml`.

---

## 3. Analysis of External Dependencies & Privacy

* **Google ML Kit Document Scanner (`GmsDocumentScanner`)**:
  * Used optionally for camera edge detection and perspective auto-cropping.
  * Runs entirely on-device via Google Play Services; does not upload photos.
  * InkClear enforces zero network usage in `AndroidManifest.xml` (`<uses-permission android:name="android.permission.INTERNET" tools:node="remove" />`).
  * System gracefully falls back to local file selection if Play Services is unavailable.

---

## 4. Remediation Steps for Offline OCR

1. **CTC Dictionary Offset Fix**:
   * Token `0` is reserved as the CTC Blank token in PaddleOCR.
   * Modify the decoding loop or append a dummy `'blank'` token at index `0` of `inference.yml` to shift character index lookups by `+1`.
2. **Handwriting Domain Awareness**:
   * Standard PP-OCR models are optimized for printed text and standard document script.
   * Highly cursive script remains probabilistic and requires manual user verification/editing in `OcrResultActivity`.
