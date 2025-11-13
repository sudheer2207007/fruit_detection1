import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import streamlit as st
from ultralytics import YOLO
import numpy as np
from PIL import Image
import cv2
import os
import base64

# ==============================
# 🌈 Page Setup
# ==============================
st.set_page_config(page_title="🍉 Fruit Detector", page_icon="🍎", layout="centered")

# ==============================
# 🍋 Background Setup
# ==============================
def add_bg_image(image_path):
    with open(image_path, "rb") as f:
        data = f.read()
    encoded = base64.b64encode(data).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
        }}
        h1, h2, h3, p, label {{
            color: #fff !important;
            text-shadow: 0 0 10px rgba(0, 0, 0, 0.8);
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# 👇 background image file
add_bg_image("fruit_bg.jpg")

# ==============================
# 🍊 Title
# ==============================
st.title("🍓 AI Fruit Detector 🍊")
st.write("Upload or capture a fruit image — view bounding boxes, labels & accuracy! 🍇")

# ==============================
# 🍎 Load Model
# ==============================
model_path = os.path.join("model", "fruit_detection.pt")
model = YOLO(model_path)

# ==============================
# 📸 Tabs
# ==============================
tab1, tab2 = st.tabs(["📁 Upload Image", "🎥 Camera Input"])

# ==============================
# 📁 Upload Tab
# ==============================
with tab1:
    uploaded = st.file_uploader("Upload a fruit image", type=["jpg", "jpeg", "png"])
    if uploaded:
        img = Image.open(uploaded)
        img_np = np.array(img)

        st.image(img, caption="📸 Uploaded Image", width=400)

        with st.spinner("Detecting fruits..."):
            # Run detection
            results = model.predict(img_np, conf=0.25)  # set confidence threshold

            # Extract detection results
            result = results[0]

            # Draw bounding boxes manually (for full control)
            annotated = img_np.copy()
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                label = result.names[cls_id]
                text = f"{label} ({conf*100:.1f}%)"

                # Draw rectangle
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 0, 0), 3)
                # Draw label background
                (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                cv2.rectangle(annotated, (x1, y1 - 25), (x1 + w, y1), (255, 0, 0), -1)
                # Put text
                cv2.putText(annotated, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Convert BGR → RGB
            annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            detected_image = Image.fromarray(annotated)

            st.image(detected_image, caption="✅ Detected Fruits (with Bounding Boxes)", width=550)

            if len(result.boxes) > 0:
                st.subheader("🍎 Detection Results")
                for box in result.boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    label = result.names[cls]
                    st.write(f"🔹 **{label}** — {conf*100:.2f}% confidence")
            else:
                st.warning("No fruits detected 😢")

# ==============================
# 🎥 Camera Tab
# ==============================
with tab2:
    camera = st.camera_input("Take a picture")
    if camera:
        img = Image.open(camera)
        img_np = np.array(img)

        st.image(img, caption="📷 Captured Image", width=400)

        with st.spinner("Detecting fruits..."):
            results = model.predict(img_np, conf=0.25)
            result = results[0]

            annotated = img_np.copy()
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                label = result.names[cls_id]
                text = f"{label} ({conf*100:.1f}%)"

                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 3)
                (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                cv2.rectangle(annotated, (x1, y1 - 25), (x1 + w, y1), (0, 255, 0), -1)
                cv2.putText(annotated, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

            annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            detected_image = Image.fromarray(annotated)

            st.image(detected_image, caption="✅ Detected Fruits (with Bounding Boxes)", width=550)

            if len(result.boxes) > 0:
                st.subheader("🍇 Detection Results")
                for box in result.boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    label = result.names[cls]
                    st.write(f"🔸 **{label}** — {conf*100:.2f}% confidence")
            else:
                st.warning("No fruits detected 😢")
