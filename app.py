import streamlit as st
import torch
import torchvision
import cv2
import numpy as np
from PIL import Image
import torchvision.transforms as T
import gdown
import os

# =============================
# PAGE CONFIG
# =============================

st.set_page_config(page_title="Blood Cell Detection", layout="centered")

st.title("🩸 Blood Cell Detection using Faster R-CNN")
st.write("Upload a blood smear image to detect RBC, WBC, and Platelets.")

# =============================
# CLASS LABELS
# =============================

classes = {
    1: "RBC",
    2: "WBC",
    3: "Platelets"
}

# =============================
# DOWNLOAD + LOAD MODEL
# =============================

@st.cache_resource
def load_model():

    model_path = "blood_cell_detector.pth"

    # Google Drive file ID
    file_id = "1AjuSGrARzRMi5x4od8XlHymFYF4riXl2"

    # Download model if not present
    if not os.path.exists(model_path):

        with st.spinner("Downloading trained model..."):
            gdown.download(id=file_id, output=model_path, quiet=False)

    # Load Faster R-CNN
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)

    num_classes = 4

    in_features = model.roi_heads.box_predictor.cls_score.in_features

    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
        in_features,
        num_classes
    )

    # PyTorch 2.6 fix
    state_dict = torch.load(model_path, map_location="cpu", weights_only=False)

    model.load_state_dict(state_dict)

    model.eval()

    return model


model = load_model()

# =============================
# IMAGE TRANSFORM
# =============================

transform = T.Compose([
    T.ToTensor(),
    T.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])

# =============================
# FILE UPLOADER
# =============================

uploaded_file = st.file_uploader(
    "Upload Blood Smear Image",
    type=["jpg", "png", "jpeg"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)

    st.image(image_np, caption="Uploaded Image", use_container_width=True)

    img_tensor = transform(image_np).unsqueeze(0)

    # =============================
    # RUN MODEL
    # =============================

    with st.spinner("Running blood cell detection..."):

        with torch.no_grad():
            prediction = model(img_tensor)

    boxes = prediction[0]["boxes"].numpy()
    scores = prediction[0]["scores"].numpy()
    labels = prediction[0]["labels"].numpy()

    threshold = 0.5

    output = image_np.copy()

    # =============================
    # CELL COUNTERS
    # =============================

    rbc_count = 0
    wbc_count = 0
    platelet_count = 0

    # =============================
    # DRAW BOXES
    # =============================

    for box, score, label in zip(boxes, scores, labels):

        if score < threshold:
            continue

        x1, y1, x2, y2 = box.astype(int)

        class_name = classes[label]

        if class_name == "RBC":
            rbc_count += 1

        elif class_name == "WBC":
            wbc_count += 1

        elif class_name == "Platelets":
            platelet_count += 1

        cv2.rectangle(output, (x1, y1), (x2, y2), (0,255,0), 2)

        cv2.putText(
            output,
            f"{class_name} {score:.2f}",
            (x1, y1-5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0,255,0),
            2
        )

    # =============================
    # SHOW RESULT
    # =============================

    st.image(output, caption="Detection Result", use_container_width=True)

    total_cells = rbc_count + wbc_count + platelet_count

    st.success(f"Total Cells Detected: {total_cells}")

    # =============================
    # SHOW CELL COUNTS
    # =============================

    col1, col2, col3 = st.columns(3)

    col1.metric("RBC", rbc_count)
    col2.metric("WBC", wbc_count)
    col3.metric("Platelets", platelet_count)