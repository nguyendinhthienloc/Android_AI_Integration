import os
import cv2
import numpy as np
from PIL import Image, ImageDraw

def run_ocr_detection(image_path, det_model_path, output_path):
    import onnxruntime as ort
    
    session = ort.InferenceSession(det_model_path)
    input_name = session.get_inputs()[0].name
    
    # Read image
    img = cv2.imread(image_path)
    orig_h, orig_w = img.shape[:2]
    
    # Resize for detection (limit max size to 960 or 1800)
    target_size = 960
    scale = target_size / float(max(orig_h, orig_w))
    new_h = int(round(orig_h * scale / 32.0)) * 32
    new_w = int(round(orig_w * scale / 32.0)) * 32
    new_h = max(32, new_h)
    new_w = max(32, new_w)
    
    resized_img = cv2.resize(img, (new_w, new_h))
    
    # Normalize ImageNet mean/std
    input_data = resized_img.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    input_data = (input_data - mean) / std
    input_data = input_data.transpose((2, 0, 1)) # CHW
    input_data = np.expand_dims(input_data, axis=0) # NCHW
    
    outputs = session.run(None, {input_name: input_data})
    prob_map = outputs[0][0, 0] # 2D map
    
    # Threshold probability map
    binary_map = (prob_map > 0.3).astype(np.uint8) * 255
    binary_map_resized = cv2.resize(binary_map, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
    
    # Draw detected bounding boxes on original image
    contours, _ = cv2.findContours(binary_map_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    annotated = img.copy()
    box_count = 0
    for cnt in contours:
        if cv2.contourArea(cnt) > 100:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(annotated, (x, y), (x + w, y + h), (23, 50, 92), 2)
            box_count += 1
            
    cv2.imwrite(output_path, annotated)
    print(f"Detected {box_count} text line regions. Saved visualization at: {output_path}")

if __name__ == "__main__":
    test_img = "app/src/androidTest/assets/ocr_reference.png"
    if not os.path.exists(test_img):
        test_img = "test_images/image.png"
    model = "app/src/main/assets/models/ocr/det/inference.onnx"
    out = "report/demo_results/ocr_detection_boxes.png"
    run_ocr_detection(test_img, model, out)
