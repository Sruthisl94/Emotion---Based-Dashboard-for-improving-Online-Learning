# streamlit_app.py

import streamlit as st
import cv2
import numpy as np
from PIL import Image
from datetime import datetime
import os
import pandas as pd
from tensorflow.keras.models import load_model

# Load model and labels
model = load_model("model/model.keras")
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Ensure required folders exist
os.makedirs("data/captured", exist_ok=True)
log_path = "data/emotion_log.csv"

# Face detection model
face_cascade = cv2.CascadeClassifier("haarcascade.xml")

st.title("📊 Student Emotion Monitoring Dashboard")

# Step 1: Input form
with st.form("student_form"):
    student_name = st.text_input("Enter Student Name")
    student_id = st.text_input("Enter Student ID")
    submit = st.form_submit_button("Login & Capture")

if submit:
    camera_image = st.camera_input("Take a picture")

    if camera_image is not None:
        st.success("Image captured successfully!")

        from PIL import Image
        import io

        # Convert captured image to grayscale
        img = Image.open(camera_image).convert("L")
        gray = np.array(img)

        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
        predicted_emotion = "Unknown"

        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            face = cv2.resize(face, (48, 48)) / 255.0
            face = np.expand_dims(face, axis=(0, -1))  # Add channel dim
            face = np.expand_dims(face, axis=0)        # Add batch dim

            prediction = model.predict(face)
            emotion_idx = np.argmax(prediction)
            predicted_emotion = emotion_labels[emotion_idx]
            break  # Just take the first face

        # Save image locally (optional)
        os.makedirs("data/captured", exist_ok=True)
        img_path = f"data/captured/{student_name}_{student_id}.jpg"
        with open(img_path, "wb") as f:
            f.write(camera_image.getbuffer())

        # Log the result
        with open(log_path, "a") as f:
            f.write(f"{datetime.now().isoformat()},{student_name},{student_id},{predicted_emotion}\n")

        # Show result
        st.subheader(f"Prediction: {predicted_emotion}")
        st.image(camera_image, caption=f"{student_name} - {predicted_emotion}", width=300)
        
        if predicted_emotion == "Sad":
            st.warning("The student seems sad. Consider the following improvements:")
            st.markdown("- Offer personal support or mentorship")
            st.markdown("- Encourage breaks or lighter activities")
            st.markdown("- Provide positive feedback or recognition")

       # Optional: Download button
        st.download_button(
            label="Download Captured Image",
            data=camera_image.getbuffer(),
            file_name=f"{student_name}_{student_id}.jpg",
            mime="image/jpeg"
        )


    # Step 6: Plot emotion stats
       if os.path.exists(log_path):
           df = pd.read_csv(log_path, names=["Time", "Name", "ID", "Emotion"], parse_dates=["Time"])
           student_df = df[df["ID"] == student_id]

           st.subheader("📈 Emotion Trend for Student")
           if not student_df.empty:
               st.line_chart(student_df["Emotion"].value_counts())
           else:
               st.info("No previous emotion data found for this student.")

           st.subheader("📊 Overall Emotion Distribution")
           st.bar_chart(df["Emotion"].value_counts())
