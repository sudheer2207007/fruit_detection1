from flask import Flask, render_template, request, send_from_directory
from ultralytics import YOLO
import os
import cv2
import uuid

app = Flask(__name__)

# Static and upload folders
UPLOAD_FOLDER = os.path.join(app.root_path, 'static', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load model
model_path = os.path.join(app.root_path, "model", "fruit_detection.pt")
model = YOLO(model_path)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Check file uploaded
    if 'file' not in request.files:
        return "No file uploaded", 400

    file = request.files['file']
    if file.filename == '':
        return "No file selected", 400

    # Save uploaded image
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    # Run YOLO model prediction
    results = model.predict(file_path, save=False)

    # Get first result
    boxes = results[0].boxes
    img = cv2.imread(file_path)

    # Draw boxes and accuracy
    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        label = str(box.cls[0])
        name = results[0].names[int(label)]
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, f"{name} {conf:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Save prediction image
    output_filename = f"predicted_{uuid.uuid4().hex}.jpg"
    output_path = os.path.join(UPLOAD_FOLDER, output_filename)
    cv2.imwrite(output_path, img)

    # Return result
    return render_template('index.html', result_image=output_filename)

@app.route('/static/uploads/<filename>')
def send_uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
