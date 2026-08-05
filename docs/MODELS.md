# InkClear Model Card

## Intended use

InkClear uses two pretrained neural pipelines: Neural Clean estimates clean paper versus ink, while PP-OCRv5 Mobile detects text lines and recognizes characters.

They are intended for personal notes and transcription assistance. Output must be reviewed by a person. The models are not suitable as the sole source for medical, legal, identity, examination-grading, or financial decisions.

## Neural Clean

- Upstream: [sliedes/binarize](https://github.com/sliedes/binarize)
- Upstream revision: f40863ed4f2869add585f8a133af29c37844ffee
- License: Apache-2.0
- Original format: TensorFlow SavedModel
- Android format: dynamically weight-quantized LiteRT/TFLite
- Parameters: 6,501,345
- Android asset size: 6,906,912 bytes
- Input: RGB float32, shape 1 x 224 x 224 x 3, values 0..255
- Output: paper/ink logit map, shape 1 x 224 x 224 x 1
- SHA-256: A4A5E4B8485185FD70AF7867BBD085015C150DEE1200B57F9AD148EAF8C45B4C

The upstream repository describes document images from the Finnish National Archive and includes DIBCO-style handwritten/printed binarization data. InkClear did not train or fine-tune the model.

### Conversion and validation

The SavedModel was loaded with TensorFlow 2.20 legacy Keras compatibility and converted with LiteRT dynamic-range weight optimization. It uses native LiteRT operations and does not require Select TensorFlow Ops/Flex.

A resized DIBCO 2009 handwritten test image was evaluated with both runtimes:

- mean absolute logit difference: 0.10663
- maximum absolute logit difference: 0.67612
- predicted ink ratio at logit threshold zero: 5.893%

These validate conversion similarity, not cleanup quality. The Android instrumented suite separately runs the bundled model end to end.

### Mobile adaptation

The upstream desktop pipeline uses overlapping windows and four rotated predictions. InkClear uses one deterministic pass over non-overlapping 224 px tiles on a canvas capped at 896 px, then upscales the mask. This trades some edge/detail quality for practical mobile latency and memory use. The classical enhancer remains the fallback.

### Rejected alternative

SBB's cleaning SavedModel was evaluated first. Its float16 conversion remained about 75 MB and required the unsupported native tf.ExtractImagePatches operation plus TensorFlow Flex. It was rejected as unsuitable for a lightweight Android release.

## Offline OCR

- SDK: official PaddlePaddle/PaddleOCR Android SDK
- Vendored SDK revision: 2661c7c0ef5c613e8f93c6e93b2e052399f0f854
- Runtime: ONNX Runtime Android 1.21.1 and official OpenCV Android 4.13.0
- License: Apache-2.0
- Detection model: PaddlePaddle/PP-OCRv5_mobile_det_onnx
- Recognition model: PaddlePaddle/PP-OCRv5_mobile_rec_onnx

Detection asset:

- size: 4,826,518 bytes
- SHA-256: A431985659DC921974177A95ADCFBB90FD9E51989A5E04D70D0B75F597B6E61D

Recognition asset:

- size: 16,534,782 bytes
- SHA-256: DA72DC72CA4DC220DF0DFDE68C1DEDC31C58D3E76A25871122E5056227D50092
- character configuration: bundled inference.yml
- release target: English-first handwriting and printed English

InkClear runs detection, reading-order sorting, perspective crops, recognition, and whitespace normalization entirely on-device. Recognition starts only after the user taps the button.

## Privacy boundary

- All model files ship in app/src/main/assets/models/.
- The merged app manifest has no android.permission.INTERNET.
- No image or recognized text is written to logs.
- TXT export uses Android's user-selected document destination.
- The app stores no OCR history.

## Known failure cases

Dense cursive, touching lines, blur, glare, perspective, very small writing, diagrams, tables, equations, uncommon symbols, non-English text, and unusual reading order can reduce accuracy. Users should edit the result before export.