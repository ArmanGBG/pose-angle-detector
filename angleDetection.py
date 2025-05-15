import streamlit as st
import numpy as np
from PIL import Image
import mediapipe as mp

st.set_page_config(page_title="تشخیص زاویه مفصل", layout="centered")
st.title("📐 تشخیص زاویه آرنج راست از روی عکس")

# تابع محاسبه زاویه بین سه نقطه
def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)
    return np.degrees(angle)

# آپلود عکس
uploaded_file = st.file_uploader("📷 لطفاً یک عکس واضح از بدن آپلود کنید", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    image_np = np.array(image)
    st.image(image_np, caption='✅ تصویر آپلودشده', use_column_width=True)

    with st.spinner("در حال پردازش تصویر..."):
        mp_pose = mp.solutions.pose
        with mp_pose.Pose(static_image_mode=True) as pose:
            results = pose.process(image_np)

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                h, w, _ = image_np.shape

                # نقاط مفصل آرنج راست
                shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x * w,
                            landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y * h]
                elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x * w,
                         landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y * h]
                wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x * w,
                         landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y * h]

                angle = calculate_angle(shoulder, elbow, wrist)
                st.success(f"📐 زاویه آرنج راست: {angle:.2f} درجه")
            else:
                st.error("❌ نقاط بدن تشخیص داده نشدند. لطفاً یک عکس واضح‌تر آپلود کنید.")
