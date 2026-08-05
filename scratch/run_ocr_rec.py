import os
import sys
import cv2
import yaml
import numpy as np
import onnxruntime as ort

# Ensure UTF-8 output in Windows PowerShell
sys.stdout.reconfigure(encoding='utf-8')

def load_character_dict(yml_path):
    chars = []
    with open(yml_path, 'r', encoding='utf-8') as f:
        for line in f:
            line_str = line.rstrip('\r\n')
            if line_str.startswith('  - '):
                char = line_str[4:]
                if char.startswith("'") and char.endswith("'"):
                    char = char[1:-1]
                chars.append(char)
            elif line_str.startswith('    - '):
                char = line_str[6:]
                if char.startswith("'") and char.endswith("'"):
                    char = char[1:-1]
                chars.append(char)
    return chars

def run_full_ocr(img_path, det_model_path, rec_model_path, rec_yml_path):
    print(f"\n==========================================")
    print(f"--- Running Full OCR on: {os.path.basename(img_path)} ---")
    print(f"==========================================")
    
    det_session = ort.InferenceSession(det_model_path)
    det_input_name = det_session.get_inputs()[0].name

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

    outputs = det_session.run(None, {det_input_name: input_data})
    prob_map = outputs[0][0, 0]

    binary_map = (prob_map > 0.25).astype(np.uint8) * 255
    binary_map_resized = cv2.resize(binary_map, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

    contours, _ = cv2.findContours(binary_map_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for cnt in contours:
        if cv2.contourArea(cnt) > 50:
            x, y, w, h = cv2.boundingRect(cnt)
            boxes.append((x, y, w, h))

    boxes = sorted(boxes, key=lambda b: b[1])

    rec_session = ort.InferenceSession(rec_model_path)
    rec_input_name = rec_session.get_inputs()[0].name
    char_list = load_character_dict(rec_yml_path)

    results = []
    for idx, (x, y, w, h) in enumerate(boxes):
        crop = img[max(0, y-2):min(orig_h, y+h+2), max(0, x-2):min(orig_w, x+w+2)]
        if crop.size == 0:
            continue
        
        rh, rw = 48, 320
        aspect = crop.shape[1] / float(crop.shape[0])
        nw = int(round(rh * aspect))
        if nw > rw:
            nw = rw
        resized_crop = cv2.resize(crop, (nw, rh))
        padded_crop = np.zeros((rh, rw, 3), dtype=np.uint8)
        padded_crop[:, :nw, :] = resized_crop
        
        rec_data = padded_crop.astype(np.float32) / 255.0
        rec_data = (rec_data - 0.5) / 0.5
        rec_data = rec_data.transpose((2, 0, 1))
        rec_data = np.expand_dims(rec_data, axis=0)

        rec_preds = rec_session.run(None, {rec_input_name: rec_data})[0]
        
        pred_indices = np.argmax(rec_preds[0], axis=-1)
        confidences = np.max(rec_preds[0], axis=-1)
        
        text_chars = []
        scores = []
        last_idx = 0
        for i, idx_val in enumerate(pred_indices):
            if idx_val != 0 and idx_val != last_idx:
                if idx_val - 1 < len(char_list):
                    text_chars.append(char_list[idx_val - 1])
                    scores.append(confidences[i])
            last_idx = idx_val
            
        line_text = "".join(text_chars).strip()
        avg_conf = float(np.mean(scores)) if scores else 0.0
        results.append((line_text, avg_conf))
        print(f"  Line {idx+1}: \"{line_text}\" (Score: {avg_conf*100:.1f}%)")

    return results

if __name__ == "__main__":
    det_model = r"D:\InkClear\app\src\main\assets\models\ocr\det\inference.onnx"
    rec_model = r"D:\InkClear\app\src\main\assets\models\ocr\rec\inference.onnx"
    rec_yml = r"D:\InkClear\app\src\main\assets\models\ocr\rec\inference.yml"

    sample1 = r"D:\InkClear\app\src\androidTest\assets\ocr_reference.png"
    sample2 = r"C:\Users\Thien Loc\.gemini\antigravity-cli\brain\633091d0-44ac-4486-a213-48cc101a26ca\clean_handwritten_note_1785926011150.jpg"
    sample3 = r"D:\InkClear\test_images\img_0.png"

    if os.path.exists(sample1):
        run_full_ocr(sample1, det_model, rec_model, rec_yml)
    if os.path.exists(sample2):
        run_full_ocr(sample2, det_model, rec_model, rec_yml)
    if os.path.exists(sample3):
        run_full_ocr(sample3, det_model, rec_model, rec_yml)
