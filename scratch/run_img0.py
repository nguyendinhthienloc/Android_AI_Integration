import os
import cv2
import numpy as np
from PIL import Image, ImageDraw
import tensorflow as tf
import onnxruntime as ort

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def run_img0_demo():
    img_path = r"D:\InkClear\test_images\img_0.png"
    clean_model_path = r"D:\InkClear\app\src\main\assets\models\neural_clean.tflite"
    det_model_path = r"D:\InkClear\app\src\main\assets\models\ocr\det\inference.onnx"
    
    out_dir = r"C:\Users\Thien Loc\.gemini\antigravity-cli\brain\633091d0-44ac-4486-a213-48cc101a26ca"
    out_grid_path = os.path.join(out_dir, "img0_presentation_grid.png")
    out_det_path = os.path.join(out_dir, "img0_ocr_detection.png")
    
    # 1. Neural Clean Inference
    interpreter = tf.lite.Interpreter(model_path=clean_model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    source_img = Image.open(img_path).convert("RGB")
    src_w, src_h = source_img.size

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

    grid_w = src_w * 2
    grid_h = src_h * 2
    grid = Image.new("RGB", (grid_w + 30, grid_h + 90), color=(240, 244, 248))
    draw = ImageDraw.Draw(grid)

    grid.paste(source_img, (10, 40))
    grid.paste(natural_img, (src_w + 20, 40))
    grid.paste(gray_img, (10, src_h + 80))
    grid.paste(bw_img, (src_w + 20, src_h + 80))

    draw.text((15, 10), "1. Original img_0.png", fill=(23, 50, 92))
    draw.text((src_w + 25, 10), "2. Neural Clean (Natural Mode)", fill=(23, 50, 92))
    draw.text((15, src_h + 55), "3. Neural Clean (Grayscale Mode)", fill=(23, 50, 92))
    draw.text((src_w + 25, src_h + 55), "4. Neural Clean (Black & White Mode)", fill=(23, 50, 92))

    grid.save(out_grid_path)
    print("SAVED_IMG0_GRID:", out_grid_path)

    # 2. PP-OCRv5 Line Detection
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

    binary_map = (prob_map > 0.2).astype(np.uint8) * 255
    binary_map_resized = cv2.resize(binary_map, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

    contours, _ = cv2.findContours(binary_map_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    annotated = img.copy()
    box_count = 0
    for cnt in contours:
        if cv2.contourArea(cnt) > 30:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(annotated, (x, y), (x + w, y + h), (23, 50, 92), 3)
            box_count += 1

    cv2.imwrite(out_det_path, annotated)
    print(f"SAVED_IMG0_OCR: {out_det_path} (Detected {box_count} lines)")

if __name__ == "__main__":
    run_img0_demo()
