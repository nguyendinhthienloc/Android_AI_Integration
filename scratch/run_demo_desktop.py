import os
import math
import numpy as np
from PIL import Image, ImageDraw
import tensorflow as tf

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def generate_demo():
    img_path = r"D:\InkClear\app\src\androidTest\assets\ocr_reference.png"
    model_path = r"D:\InkClear\app\src\main\assets\models\neural_clean.tflite"
    out_dir = r"D:\InkClear\design"
    
    os.makedirs(out_dir, exist_ok=True)
    
    interpreter = tf.lite.Interpreter(model_path=model_path)
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
    strength = 0.6
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

    draw.text((15, 10), "1. Original Photographed Note", fill=(23, 50, 92))
    draw.text((src_w + 25, 10), "2. Neural Clean (Natural Mode)", fill=(23, 50, 92))
    draw.text((15, src_h + 55), "3. Neural Clean (Grayscale Mode)", fill=(23, 50, 92))
    draw.text((src_w + 25, src_h + 55), "4. Neural Clean (Black & White Mode)", fill=(23, 50, 92))

    out_path = os.path.join(out_dir, "demo_presentation_grid.png")
    grid.save(out_path)
    print("SUCCESSFULLY_SAVED_PRESENTATION_GRID:", out_path)

if __name__ == "__main__":
    generate_demo()
