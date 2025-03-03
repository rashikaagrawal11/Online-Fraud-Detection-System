import cv2
import numpy as np
import os
from tensorflow.keras.models import load_model

# Directory for uploaded videos
UPLOAD_FOLDER = './static/uploaded_videos'

# Load pre-trained model
MODEL_PATH = './models/deepfake_video_model.h5'
model = load_model(MODEL_PATH)

# Process a video frame-by-frame
def process_video(file_path):
    cap = cv2.VideoCapture(file_path)
    frame_count = 0
    fake_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Preprocess frame
        frame = cv2.resize(frame, (128, 128))  # Resize to model input size
        frame = frame / 255.0  # Normalize pixel values
        frame = np.expand_dims(frame, axis=0)

        # Predict using the model
        prediction = model.predict(frame)
        is_fake = np.argmax(prediction)  # 1 if fake, 0 if real

        if is_fake:
            fake_count += 1
        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()

    # Determine result based on fake frame proportion
    if fake_count / frame_count > 0.5:  # If >50% frames are fake
        return "Deepfake Detected"
    else:
        return "Legitimate Video"

# Predict Deepfake for an uploaded video
def predict_deepfake_video(file_path):
    result = process_video(file_path)
    return result
