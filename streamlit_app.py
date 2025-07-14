# streamlit_app.py
if submit:
    camera_image = st.camera_input("Take a picture")

    if camera_image is not None:
        st.success("Image captured successfully!")

        # Convert to grayscale for OpenCV processing
        img = Image.open(camera_image).convert("L")
        gray = np.array(img)

        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
        predicted_emotion = "Unknown"

        if len(faces) > 0:
            (x, y, w, h) = faces[0]
            face = gray[y:y+h, x:x+w]
            face = cv2.resize(face, (48, 48)) / 255.0
            face = np.expand_dims(face, axis=(0, -1))
            face = np.expand_dims(face, axis=0)

            prediction = model.predict(face)
            emotion_idx = np.argmax(prediction)
            predicted_emotion = emotion_labels[emotion_idx]

        # Save image directly from buffer
        img_path = f"data/captured/{student_name}_{student_id}.jpg"
        with open(img_path, "wb") as f:
            f.write(camera_image.getbuffer())

        # Log result
        with open(log_path, "a") as f:
            f.write(f"{datetime.now().isoformat()},{student_name},{student_id},{predicted_emotion}\n")

        # Display result
        st.subheader(f"Prediction: {predicted_emotion}")
        st.image(camera_image, caption=f"{student_name} - {predicted_emotion}", width=300)

        if predicted_emotion == "Sad":
            st.warning("The student seems sad. Consider the following improvements:")
            st.markdown("- Offer personal support or mentorship")
            st.markdown("- Encourage breaks or lighter activities")
            st.markdown("- Provide positive feedback or recognition")

        # Download option
        st.download_button(
            label="Download Captured Image",
            data=camera_image.getbuffer(),
            file_name=f"{student_name}_{student_id}.jpg",
            mime="image/jpeg"
        )

        # Plot history
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
    else:
        st.warning("Please allow webcam access and capture an image.")
