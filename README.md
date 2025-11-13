# Dental Radiography Analysis (YOLOv8)

This project deploys a YOLOv8 model trained on dental radiographs to detect multiple dental anomalies.

## Files
- `app.py`: Gradio interface + YOLOv8 inference
- `best.pt`: Trained YOLOv8 model from /runs/detect/train2/
- `requirements.txt`: Dependencies for deployment

## Deployment
This project is ready to deploy on **HuggingFace Spaces** using Gradio.

Just upload the entire folder into a new HF Space with SDK = Gradio.
