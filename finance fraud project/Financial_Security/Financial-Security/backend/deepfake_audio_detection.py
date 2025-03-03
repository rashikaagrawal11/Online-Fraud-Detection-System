import os
import numpy as np
import librosa
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle

# Directory containing uploaded audio files
UPLOAD_FOLDER = './static/uploaded_audio'

# Extract MFCC features from audio files
def extract_features(file_path):
    audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfccs_scaled = np.mean(mfccs.T, axis=0)
    return mfccs_scaled

# Prepare Dataset
def prepare_dataset():
    labels = []
    features = []
    for file in os.listdir(UPLOAD_FOLDER):
        if file.endswith(".wav"):
            file_path = os.path.join(UPLOAD_FOLDER, file)
            features.append(extract_features(file_path))
            labels.append(file.split("_")[0])  # Assuming files are named like 'deepfake_01.wav' or 'legit_01.wav'

    # Encode labels (e.g., "deepfake" -> 1, "legit" -> 0)
    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(labels)

    # Save the label encoder
    with open('./models/label_encoder.pkl', 'wb') as file:
        pickle.dump(label_encoder, file)

    return np.array(features), np.array(labels_encoded)

# Train Deepfake Detection Model
def train_deepfake_model():
    # Load the dataset
    features, labels = prepare_dataset()

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

    # Convert labels to categorical
    y_train = to_categorical(y_train, num_classes=2)
    y_test = to_categorical(y_test, num_classes=2)

    # Reshape input data for CNN
    X_train = X_train.reshape(X_train.shape[0], 40, 1, 1)
    X_test = X_test.reshape(X_test.shape[0], 40, 1, 1)

    # Define CNN model
    model = Sequential([
        Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(40, 1, 1)),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(2, activation='softmax')
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(X_train, y_train, epochs=20, batch_size=16, validation_data=(X_test, y_test))

    # Save the model
    model.save('./models/deepfake_model.h5')

# Predict Deepfake
def predict_deepfake(file_path):
    # Load the pre-trained model
    from tensorflow.keras.models import load_model
    model = load_model('./models/deepfake_model.h5')

    # Load the label encoder
    with open('./models/label_encoder.pkl', 'rb') as file:
        label_encoder = pickle.load(file)

    # Extract features from the uploaded audio file
    features = extract_features(file_path)
    features = features.reshape(1, 40, 1, 1)

    # Predict
    prediction = model.predict(features)
    predicted_label = label_encoder.inverse_transform([np.argmax(prediction)])
    return predicted_label[0]

# Uncomment to train the model
# train_deepfake_model()
