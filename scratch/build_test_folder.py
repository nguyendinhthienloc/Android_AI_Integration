import os
import sys
import cv2
import yaml
import numpy as np
from PIL import Image, ImageDraw
import tensorflow as tf
import onnxruntime as ort

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

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

def process_and_save(img_path, prefix, test_dir, artifact_dir, clean_model, det_model, rec_model, rec_yml):
    print(f"Processing '{prefix}' from {img_path}")
    source_img = Image.open(img_path).convert("RGB")
    src_w, src_h = source_img.size

    # 1. Original
    orig_name = f"{prefix}_1_original.png"
    source_img.save(os.path.join(test_dir, orig_name))
    source_img.save(os.path.join(artifact_dir, orig_name))

    # 2. Neural Clean
    interpreter = tf.lite.Interpreter(model_path=clean_model)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    max_dim = 896
    scale = min(1.0, max_dim / float(max(src_w, src_h)))
    inf_w = max(1, int(round(src_w * scale)))
    inf_h = max(1, int(round(src_h * scale)))

    inf_img = source_img.resize((inf_w, inf_h), Image.Resampling.BILINEAR)
    inf_pixels = np.array(inf_img, dtype=np.float32)

    mask_pixels = np.zeros((inf_h, inf_w), dtype=np.float32)
    tile_size = 224
    strength = 0.65
    contrast = 1.4 + (1.8 * strength)

    for top in range(0, inf_h, tile_size):
        for left in range(0, inf_w, tile_size):
            tile_input = np.ones((1, tile_size, tile_size, 3), dtype=np.float32) * 255.0
            h_end = min(top + tile_size, inf_h)
            w_end = min(left + tile_size, inf_w)
            slice_h = h_end - top
            slice_w = w_end - left
            tile_input[0, :slice_h, :slice_w, :] = inf_pixels[top:h_end, left:w_end, :]
            interpreter.set_tensor(input_details[0]['index'], tile_input)
            interpreter.invoke()
            tile_output = interpreter.get_tensor(output_details[0]['index'])
            logits = tile_output[0, :slice_h, :slice_w, 0]
            probs = sigmoid(logits)
            cleaned = np.clip(((probs - 0.5) * contrast) + 0.5, 0.0, 1.0)
            mask_pixels[top:h_end, left:w_end] = cleaned

    mask_img = Image.fromarray((mask_pixels * 255.0).astype(np.uint8), mode="L")
    full_mask = mask_img.resize((src_w, src_h), Image.Resampling.BILINEAR)
    full_mask_arr = np.array(full_mask, dtype=np.float32)

    src_arr = np.array(source_img, dtype=np.float32)
    r, g, b = src_arr[:, :, 0], src_arr[:, :, 1], src_arr[:, :, 2]
    original_luma = (r * 77.0 + g * 150.0 + b * 29.0) / 256.0

    blend = 0.55 + (0.4 * strength)
    clean_luma = np.clip(original_luma + ((full_mask_arr - original_luma) * blend), 0.0, 255.0)

    chroma_amount = 0.38
    res_r = np.clip(clean_luma + ((r - original_luma) * chroma_amount), 0.0, 255.0)
    res_g = np.clip(clean_luma + ((g - original_luma) * chroma_amount), 0.0, 255.0)
    res_b = np.clip(clean_luma + ((b - original_luma) * chroma_amount), 0.0, 255.0)
    
    natural_img = Image.fromarray(np.stack([res_r, res_g, res_b], axis=-1).astype(np.uint8))
    gray_img = Image.fromarray(np.stack([clean_luma]*3, axis=-1).astype(np.uint8))
    binary_thresh = round(142.0 + (34.0 * strength))
    bw_arr = np.where(full_mask_arr < binary_thresh, 0, 255).astype(np.uint8)
    bw_img = Image.fromarray(np.stack([bw_arr]*3, axis=-1))

    natural_img.save(os.path.join(test_dir, f"{prefix}_2_enhanced_natural.png"))
    gray_img.save(os.path.join(test_dir, f"{prefix}_3_enhanced_grayscale.png"))
    bw_img.save(os.path.join(test_dir, f"{prefix}_4_enhanced_bw.png"))

    natural_img.save(os.path.join(artifact_dir, f"{prefix}_2_enhanced_natural.png"))
    gray_img.save(os.path.join(artifact_dir, f"{prefix}_3_enhanced_grayscale.png"))
    bw_img.save(os.path.join(artifact_dir, f"{prefix}_4_enhanced_bw.png"))

    # Grid Comparison Image
    grid_w = src_w * 2
    grid_h = src_h * 2
    grid = Image.new("RGB", (grid_w + 30, grid_h + 90), color=(240, 244, 248))
    draw = ImageDraw.Draw(grid)

    grid.paste(source_img, (10, 40))
    grid.paste(natural_img, (src_w + 20, 40))
    grid.paste(gray_img, (10, src_h + 80))
    grid.paste(bw_img, (src_w + 20, src_h + 80))

    draw.text((15, 10), f"1. Original ({prefix})", fill=(23, 50, 92))
    draw.text((src_w + 25, 10), "2. Neural Clean (Natural Mode)", fill=(23, 50, 92))
    draw.text((15, src_h + 55), "3. Neural Clean (Grayscale Mode)", fill=(23, 50, 92))
    draw.text((src_w + 25, src_h + 55), "4. Neural Clean (Black & White Mode)", fill=(23, 50, 92))
    
    grid.save(os.path.join(test_dir, f"{prefix}_5_comparison_grid.png"))
    grid.save(os.path.join(artifact_dir, f"{prefix}_5_comparison_grid.png"))

    # 3. OCR Detection & Recognition
    det_session = ort.InferenceSession(det_model)
    det_input_name = det_session.get_inputs()[0].name

    cv_img = cv2.imread(img_path)
    orig_h, orig_w = cv_img.shape[:2]

    target_size = 960
    scale = target_size / float(max(orig_h, orig_w))
    new_h = max(32, int(round(orig_h * scale / 32.0)) * 32)
    new_w = max(32, int(round(orig_w * scale / 32.0)) * 32)

    resized_img = cv2.resize(cv_img, (new_w, new_h))
    input_data = resized_img.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    input_data = (input_data - mean) / std
    input_data = input_data.transpose((2, 0, 1))
    input_data = np.expand_dims(input_data, axis=0)

    outputs = det_session.run(None, {det_input_name: input_data})
    prob_map = outputs[0][0, 0]

    binary_map = (prob_map > 0.2).astype(np.uint8) * 255
    binary_map_resized = cv2.resize(binary_map, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

    contours, _ = cv2.findContours(binary_map_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for cnt in contours:
        if cv2.contourArea(cnt) > 30:
            x, y, w, h = cv2.boundingRect(cnt)
            boxes.append((x, y, w, h))

    boxes = sorted(boxes, key=lambda b: b[1])

    annotated = cv_img.copy()
    for (x, y, w, h) in boxes:
        cv2.rectangle(annotated, (x, y), (x + w, y + h), (23, 50, 92), 3)
    
    cv2.imwrite(os.path.join(test_dir, f"{prefix}_6_ocr_detection.png"), annotated)
    cv2.imwrite(os.path.join(artifact_dir, f"{prefix}_6_ocr_detection.png"), annotated)

    # Recognition
    rec_session = ort.InferenceSession(rec_model)
    rec_input_name = rec_session.get_inputs()[0].name
    char_list = load_character_dict(rec_yml)

    extracted_lines = []
    for idx, (x, y, w, h) in enumerate(boxes):
        crop = cv_img[max(0, y-2):min(orig_h, y+h+2), max(0, x-2):min(orig_w, x+w+2)]
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
        if line_text:
            extracted_lines.append(f"Line {idx+1} [Confidence Score: {avg_conf*100:.1f}%]: {line_text}")

    txt_content = f"=== InkClear Extracted Text Output for {prefix} ===\n\n"
    txt_content += "\n".join(extracted_lines) if extracted_lines else "No text detected."
    txt_content += "\n\n=== End of Extracted Text ==="

    with open(os.path.join(test_dir, f"{prefix}_7_extracted_text.txt"), "w", encoding="utf-8") as f:
        f.write(txt_content)
    with open(os.path.join(artifact_dir, f"{prefix}_7_extracted_text.txt"), "w", encoding="utf-8") as f:
        f.write(txt_content)

    print(f"DONE: {prefix}")

def main():
    test_dir = r"D:\InkClear\test"
    artifact_dir = r"C:\Users\Thien Loc\.gemini\antigravity-cli\brain\633091d0-44ac-4486-a213-48cc101a26ca"
    os.makedirs(test_dir, exist_ok=True)
    os.makedirs(artifact_dir, exist_ok=True)

    clean_model = r"D:\InkClear\app\src\main\assets\models\neural_clean.tflite"
    det_model = r"D:\InkClear\app\src\main\assets\models\ocr\det\inference.onnx"
    rec_model = r"D:\InkClear\app\src\main\assets\models\ocr\rec\inference.onnx"
    rec_yml = r"D:\InkClear\app\src\main\assets\models\ocr\rec\inference.yml"

    # 1. img_0.png
    img_0_path = r"D:\InkClear\test_images\img_0.png"
    if os.path.exists(img_0_path):
        process_and_save(img_0_path, "img_0", test_dir, artifact_dir, clean_model, det_model, rec_model, rec_yml)

    # 2. clean_note
    clean_sample_path = os.path.join(artifact_dir, "clean_handwritten_note_1785926011150.jpg")
    if os.path.exists(clean_sample_path):
        process_and_save(clean_sample_path, "clean_note", test_dir, artifact_dir, clean_model, det_model, rec_model, rec_yml)

    # 3. ocr_reference
    ref_path = r"D:\InkClear\app\src\androidTest\assets\ocr_reference.png"
    if os.path.exists(ref_path):
        process_and_save(ref_path, "ocr_reference", test_dir, artifact_dir, clean_model, det_model, rec_model, rec_yml)

if __name__ == "__main__":
    main()
