import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pickle

def train_threat_model():
    df = pd.read_csv('data/attack_dataset.csv')
    
    # Check for NaN values and remove them
    print(f"Dataset shape before cleaning: {df.shape}")
    print(f"NaN values in target: {df['is_threat'].isna().sum()}")
    
    # Drop rows with NaN values in target or features
    df = df.dropna(subset=['is_threat', 'request_rate', 'payload_size', 'suspicious_patterns', 'entropy'])
    print(f"Dataset shape after cleaning: {df.shape}")
    
    X = df[['request_rate', 'payload_size', 'suspicious_patterns', 'entropy']]
    y = df['is_threat']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    print("Model Performance:")
    print(classification_report(y_test, y_pred))
    
    with open('data/threat_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    print("\n[SUCCESS] Model saved to data/threat_model.pkl")
    print(f"Feature importance: {dict(zip(X.columns, model.feature_importances_))}")

if __name__ == '__main__':
    train_threat_model()
