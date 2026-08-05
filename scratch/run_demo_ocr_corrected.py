import os
import cv2
import numpy as np
import onnxruntime as ort


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEMO = os.path.join(ROOT, "report", "demo_results")
IMAGE = os.path.join(DEMO, "3_neural_clean_grayscale.png")
DET = os.path.join(ROOT, "app", "src", "main", "assets", "models", "ocr", "det", "inference.onnx")
REC = os.path.join(ROOT, "app", "src", "main", "assets", "models", "ocr", "rec", "inference.onnx")
YML = os.path.join(ROOT, "app", "src", "main", "assets", "models", "ocr", "rec", "inference.yml")


def character_dict(path):
    chars = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\r\n")
            if line.startswith("  - "):
                value = line[4:]
                if len(value) >= 2 and value[0] == value[-1] == "'":
                    value = value[1:-1]
                chars.append(value)
    return chars


def detect(img):
    h, w = img.shape[:2]
    limit = 960
    scale = limit / max(h, w)
    nh = max(32, int(round(h * scale / 32)) * 32)
    nw = max(32, int(round(w * scale / 32)) * 32)
    resized = cv2.resize(img, (nw, nh))
    data = resized.astype(np.float32) / 255.0
    data = (data - np.array([0.485, 0.456, 0.406], np.float32)) / np.array([0.229, 0.224, 0.225], np.float32)
    data = data.transpose(2, 0, 1)[None, ...]
    session = ort.InferenceSession(DET)
    prob = session.run(None, {session.get_inputs()[0].name: data})[0][0, 0]
    binary = (prob > 0.2).astype(np.uint8) * 255
    binary = cv2.resize(binary, (w, h), interpolation=cv2.INTER_NEAREST)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = [cv2.boundingRect(c) for c in contours if cv2.contourArea(c) > 30]
    return sorted(boxes, key=lambda b: (b[1], b[0]))


def recognize(img, boxes):
    session = ort.InferenceSession(REC)
    chars = character_dict(YML)
    results = []
    for number, (x, y, w, h) in enumerate(boxes, 1):
        crop = img[max(0, y - 2):min(img.shape[0], y + h + 2), max(0, x - 2):min(img.shape[1], x + w + 2)]
        if crop.size == 0:
            continue
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        new_w = max(1, min(3200, int(np.ceil(48 * rgb.shape[1] / rgb.shape[0]))))
        resized = cv2.resize(rgb, (new_w, 48), interpolation=cv2.INTER_LINEAR)
        data = ((resized.astype(np.float32) / 255.0) - 0.5) / 0.5
        data = data.transpose(2, 0, 1)[None, ...]
        output = session.run(None, {session.get_inputs()[0].name: data})[0][0]
        indices = np.argmax(output, axis=-1)
        confidence = np.max(output, axis=-1)
        text = []
        scores = []
        previous = 0
        for index, score in zip(indices, confidence):
            if index != 0 and index != previous and index - 1 < len(chars):
                text.append(chars[index - 1])
                scores.append(float(score))
            previous = index
        value = "".join(text).strip()
        if value:
            results.append((number, value, float(np.mean(scores)) if scores else 0.0))
    return results


def main():
    img = cv2.imread(IMAGE)
    if img is None:
        raise FileNotFoundError(IMAGE)
    boxes = detect(img)
    results = recognize(img, boxes)
    annotated = img.copy()
    for number, (x, y, w, h) in enumerate(boxes, 1):
        cv2.rectangle(annotated, (x, y), (x + w, y + h), (23, 50, 92), 2)
        cv2.putText(annotated, str(number), (x, max(18, y - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (23, 50, 92), 2)
    cv2.imwrite(os.path.join(DEMO, "ocr_detection_boxes.png"), annotated)
    with open(os.path.join(DEMO, "extracted_text.txt"), "w", encoding="utf-8") as f:
        f.write("=== InkClear PP-OCRv5 extraction ===\n\n")
        for number, value, score in results:
            f.write(f"Line {number} [Score: {score * 100:.1f}%]: {value}\n")
        f.write("\n=== End of extraction ===\n")
    print(f"Detected boxes: {len(boxes)}")
    print(f"Recognized lines: {len(results)}")
    print(os.path.join(DEMO, "extracted_text.txt"))


if __name__ == "__main__":
    main()
