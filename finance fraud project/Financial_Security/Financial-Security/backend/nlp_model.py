import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report
import pickle

# Load datasets
fraud_data = pd.read_csv('./datasets/fraud_call.csv')
spam_data = pd.read_csv('./datasets/spam.csv')

# Merge datasets and standardize columns
fraud_data.columns = ['label', 'text']
spam_data.columns = ['label', 'text']

# Combine and clean data
combined_data = pd.concat([fraud_data, spam_data], axis=0)
combined_data['label'] = combined_data['label'].apply(lambda x: 1 if x in ['Fraud', 'spam'] else 0)

# Text preprocessing and vectorization
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(combined_data['text'])
y = combined_data['label']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Isolation Forest model
model = IsolationForest(contamination=0.2, random_state=42)
model.fit(X_train.toarray())

# Evaluate model
y_pred = model.predict(X_test.toarray())
y_pred = [1 if x == -1 else 0 for x in y_pred]  # Convert IsolationForest output
print(classification_report(y_test, y_pred))

# Save the model and vectorizer
with open('./models/nlp_model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('./models/label_encoder.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)
