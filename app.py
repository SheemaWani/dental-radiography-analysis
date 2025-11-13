from ultralytics import YOLO
import gradio as gr
import tempfile
from PIL import Image
import numpy as np
import cv2

# Load the YOLO model once
model = YOLO("best.pt")  # uses the copied model

def predict(image):
    # Save uploaded image temporarily
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        image.save(f.name)
        img_path = f.name

    # Run YOLO inference
    results = model.predict(source=img_path, imgsz=640, conf=0.25, save=False)
    r = results[0]

    # Convert PIL image → OpenCV
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # Draw detections
    if hasattr(r, "boxes") and len(r.boxes) > 0:
        for box, conf, cls in zip(r.boxes.xyxy, r.boxes.conf, r.boxes.cls):
            x1, y1, x2, y2 = map(int, box.tolist())
            label = f"{model.names[int(cls)]}: {conf:.2f}"

            cv2.rectangle(img_cv, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(img_cv, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    # Convert back to RGB PIL
    img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    result_img = Image.fromarray(img_rgb)

    return result_img

# Gradio UI
demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload Dental X-ray"),
    outputs=gr.Image(type="pil", label="Detected Anomalies"),
    title="Dental Radiography Analysis (YOLOv8)",
    description="Upload a dental radiograph to detect anomalies using YOLOv8."
)

if __name__ == "__main__":
    demo.launch()
