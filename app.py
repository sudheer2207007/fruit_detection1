from flask import Flask, render_template, request
from ultralytics import YOLO
import os

app = Flask(__name__)
model = YOLO("model/fruit_detection.pt")

UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return render_template('index.html', result="⚠️ No file selected")

    file = request.files['file']
    if file.filename == '':
        return render_template('index.html', result="⚠️ No file selected")

    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    # Run YOLO detection
    results = model(filepath)

    boxes = results[0].boxes
    detected_items = []

    if boxes is not None and len(boxes) > 0:
        for cls_id, conf in zip(boxes.cls.cpu().numpy(), boxes.conf.cpu().numpy()):
            name = results[0].names[int(cls_id)]
            conf_percent = round(float(conf) * 100, 2)
            detected_items.append(f"{name} ({conf_percent}%)")

        detected_str = ', '.join(detected_items)

        # Save image with bounding boxes
        output_path = os.path.join(UPLOAD_FOLDER, "detected_" + file.filename)
        results[0].save(filename=output_path)

        # Just send the filename (not full path)
        detected_image = "uploads/detected_" + file.filename
    else:
        detected_str = "No fruits detected"
        detected_image = None

    return render_template('index.html', result=detected_str, detected_image=detected_image)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
