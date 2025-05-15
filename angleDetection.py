import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from PIL import Image

st.set_page_config(page_title="تشخیص زاویه مفصل", layout="centered")

# محاسبه زاویه بین سه نقطه
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)
    return np.degrees(angle)

# عنوان برنامه
st.title("📐 تشخیص زاویه مفصل با MediaPipe")

# آپلود عکس
uploaded_file = st.file_uploader("🔺 لطفاً یک عکس آپلود کن", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    image_np = np.array(image)

    st.image(image_np, caption='📸 تصویر آپلودشده', use_column_width=True)

    with st.spinner("در حال پردازش تصویر..."):
        mp_drawing = mp.solutions.drawing_utils
        mp_pose = mp.solutions.pose

        with mp_pose.Pose(static_image_mode=True) as pose:
            results = pose.process(image_np)

            if results.pose_landmarks:
                annotated_image = image_np.copy()
                landmarks = results.pose_landmarks.landmark

                # نقاط مفصل آرنج راست
                shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x * image_np.shape[1],
                            landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y * image_np.shape[0]]
                elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x * image_np.shape[1],
                         landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y * image_np.shape[0]]
                wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x * image_np.shape[1],
                         landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y * image_np.shape[0]]

                angle = calculate_angle(shoulder, elbow, wrist)

                # رسم زاویه روی عکس
                mp_drawing.draw_landmarks(annotated_image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                cv2.putText(annotated_image, f'{int(angle)} deg',
                            tuple(np.int32(elbow)),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3, cv2.LINE_AA)

                st.success(f"✅ زاویه آرنج راست: {angle:.2f} درجه")
                st.image(annotated_image, caption="📐 نتیجه نهایی", channels="RGB", use_column_width=True)
            else:
                st.error("❌ نتوانستم نقاط بدن را تشخیص دهم. لطفاً عکس واضح‌تری بده.")
