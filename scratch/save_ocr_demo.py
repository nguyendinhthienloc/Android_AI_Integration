import os
import cv2
import numpy as np
import onnxruntime as ort

def main():
    img_path = os.path.abspath("app/src/androidTest/assets/ocr_reference.png")
    det_model_path = os.path.abspath("app/src/main/assets/models/ocr/det/inference.onnx")
    out_dir = r"C:\Users\Thien Loc\.gemini\antigravity-cli\brain\633091d0-44ac-4486-a213-48cc101a26ca"
    out_path = os.path.join(out_dir, "demo_ocr_detection_boxes.png")

    session = ort.InferenceSession(det_model_path)
    input_name = session.get_inputs()[0].name

    img = cv2.imread(img_path)
    orig_h, orig_w = img.shape[:2]

    target_size = 960
    scale = target_size / float(max(orig_h, orig_w))
    new_h = max(32, int(round(orig_h * scale / 32.0)) * 32)
    new_w = max(32, int(round(orig_w * scale / 32.0)) * 32)

    resized_img = cv2.resize(img, (new_w, new_h))

    input_data = resized_img.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    input_data = (input_data - mean) / std
    input_data = input_data.transpose((2, 0, 1))
    input_data = np.expand_dims(input_data, axis=0)

    outputs = session.run(None, {input_name: input_data})
    prob_map = outputs[0][0, 0]

    binary_map = (prob_map > 0.3).astype(np.uint8) * 255
    binary_map_resized = cv2.resize(binary_map, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

    contours, _ = cv2.findContours(binary_map_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    annotated = img.copy()
    box_count = 0
    for cnt in contours:
        if cv2.contourArea(cnt) > 100:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(annotated, (x, y), (x + w, y + h), (23, 50, 92), 3)
            box_count += 1

    cv2.imwrite(out_path, annotated)
    print("SAVED_OCR_BOXES:", out_path)

if __name__ == "__main__":
    main()
