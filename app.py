import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import requests

# Initialize MediaPipe pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False)

# Flask API endpoint
API_URL = "https://keypointextractor.onrender.com/predict"

st.set_page_config(page_title="Live Gesture Predictor", layout="centered")
st.title("🎥 Live Gesture Pose Prediction")
st.markdown("Uses your webcam to detect gestures and predict using ML model.")

start = st.button("Start Webcam Prediction")

if start:
    cap = cv2.VideoCapture(0)
    stframe = st.empty()

    st.info("⌛ Starting webcam. Press `Stop` to end.")

    stop = st.button("Stop")
    frame_count = 0

    while cap.isOpened() and not stop:
        ret, frame = cap.read()
        if not ret:
            st.error("❌ Failed to access webcam")
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)

        if results.pose_landmarks:
            keypoints = []
            for lm in results.pose_landmarks.landmark:
                keypoints.extend([lm.x, lm.y, lm.z, lm.visibility])

            if len(keypoints) == 132:
                try:
                    res = requests.post(API_URL, json={"keypoints": keypoints})
                    pred = res.json()
                    label = pred.get("prediction", "Unknown")
                    conf = pred.get("confidence_scores", {})

                    cv2.putText(frame, f"Prediction: {label}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                except Exception as e:
                    cv2.putText(frame, "API Error", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            cv2.putText(frame, "No Pose Detected", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        stframe.image(frame, channels="BGR")

    cap.release()
    st.success("✅ Webcam stopped.")
