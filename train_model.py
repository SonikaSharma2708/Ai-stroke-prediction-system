# train_model.py
import os
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

def execute_training():
    dataset_path = os.path.join('dataset', 'healthcare-dataset-stroke-data.csv')
    model_dir = 'model'
    
    if not os.path.exists(dataset_path):
        print(f"❌ Error: Missing file targets.")
        return

    df = pd.read_csv(dataset_path)
    if 'id' in df.columns:
        df = df.drop(columns=['id'])
    df['bmi'] = df['bmi'].fillna(df['bmi'].median())
    df = df[df['gender'] != 'Other']
    
    gender_map = {'Male': 1, 'Female': 0}
    married_map = {'Yes': 1, 'No': 0}
    residence_map = {'Urban': 1, 'Rural': 0}
    work_map = {'Private': 0, 'Self-employed': 1, 'Govt_job': 2, 'children': 3, 'Never_worked': 4}
    smoke_map = {'never smoked': 0, 'formerly smoked': 1, 'smokes': 2, 'Unknown': 3}
    
    df['gender'] = df['gender'].map(gender_map)
    df['ever_married'] = df['ever_married'].map(married_map)
    df['Residence_type'] = df['Residence_type'].map(residence_map)
    df['work_type'] = df['work_type'].map(work_map)
    df['smoking_status'] = df['smoking_status'].map(smoke_map)
    
    X = df.drop(columns=['stroke'])
    y = df['stroke']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = RandomForestClassifier(n_estimators=150, class_weight='balanced', random_state=42)
    model.fit(X_scaled, y)
    
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        
    with open(os.path.join(model_dir, 'stroke_model.pkl'), 'wb') as m_file:
        pickle.dump(model, m_file)
    with open(os.path.join(model_dir, 'encoders.pkl'), 'wb') as s_file:
        pickle.dump(scaler, s_file)
        
    print("✅ Pipeline models trained and optimized inside model/ folder.")

if __name__ == "__main__":
    execute_training()