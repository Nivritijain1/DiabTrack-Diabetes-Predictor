# train_and_save.py - Complete training and saving script
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import pickle
import os

from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from xgboost import XGBClassifier

print("🚀 Starting diabetes prediction model training...")

# ==================== DATA LOADING ====================
print("📂 Loading NHANES data...")

# Update paths to point to your data folder
data_folder = "data"

DEMO = pd.read_sas(f"{data_folder}/DEMO.XPT", format="xport")
BMX = pd.read_sas(f"{data_folder}/BMX.XPT", format="xport")
BPX = pd.read_sas(f"{data_folder}/BPX.XPT", format="xport")
GLU = pd.read_sas(f"{data_folder}/GLU.XPT", format="xport")
GHB = pd.read_sas(f"{data_folder}/GHB.XPT", format="xport")
INS = pd.read_sas(f"{data_folder}/INS.XPT", format="xport")
DIQ = pd.read_sas(f"{data_folder}/DIQ.XPT", format="xport")
RHQ = pd.read_sas(f"{data_folder}/RHQ.XPT", format="xport")

print("✅ NHANES data loaded successfully!")

# ==================== DATA MERGING ====================
print("🔄 Merging datasets...")

df_base = DEMO[['SEQN','RIAGENDR','RIDAGEYR','RIDRETH3']]
df_base = df_base.rename(columns={
    'RIAGENDR':'Gender',
    'RIDAGEYR':'Age',
    'RIDRETH3':'Race_Ethnicity'
})

df_base = (
    df_base
    .merge(BMX[['SEQN','BMXBMI','BMXWAIST']], on='SEQN', how='left')
    .merge(BPX[['SEQN','BPXSY1','BPXDI1']], on='SEQN', how='left')
    .merge(GLU[['SEQN','LBXGLU']], on='SEQN', how='left')
    .merge(GHB[['SEQN','LBXGH']], on='SEQN', how='left')
    .merge(INS[['SEQN','LBXIN']], on='SEQN', how='left')
    .merge(DIQ[['SEQN','DIQ010']], on='SEQN', how='left')
    .merge(RHQ[['SEQN','RHQ010','RHQ031']], on='SEQN', how='left')
)

df_base = df_base.rename(columns={
    'BMXBMI':'BMI',
    'BMXWAIST':'Waist_Circumference',
    'BPXSY1':'BP_Systolic',
    'BPXDI1':'BP_Diastolic',
    'LBXGLU':'Glucose_mg_dL',
    'LBXGH':'HbA1c',
    'LBXIN':'Insulin_ml',
    'DIQ010':'Diabetes_SelfReport',
    'RHQ010':'Age_First_Period',
    'RHQ031':'Regular_Periods'
})

# ==================== PIMA DATA ====================
print("📂 Loading PIMA data...")

pima_columns = [
    'Pregnancies','Glucose','BloodPressure','SkinThickness',
    'Insulin','BMI','DiabetesPedigreeFunction','Age','Diabetes_Outcome'
]

df_pima = pd.read_csv(f"{data_folder}/pima_diabetes.csv", header=None, names=pima_columns)

# Replace 0 values with NAN
for i in ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']:
    df_pima[i] = df_pima[i].replace(0, np.nan)

# Generate new columns
df_pima['Data_Source'] = 'PIMA'
df_pima['Gender'] = 'Female'
df_pima['Race_Ethnicity'] = 'Pima_Indian'
df_pima['Participant_ID'] = ["PIMA_" + str(i+1) for i in range(len(df_pima))]

df_pima_clean = df_pima.rename(columns={
    'Glucose': 'Glucose_mg_dL',
    'BloodPressure': 'BP_Systolic',
    'SkinThickness': 'SkinFold_Thickness',
    'Insulin': 'Insulin_ml',
    'DiabetesPedigreeFunction': 'Diabetes_Pedigree',
    'Diabetes_Outcome': 'Diabetes_Target'
})

# Align columns with NHANES
for col in df_base.columns:
    if col not in df_pima_clean.columns:
        df_pima_clean[col] = np.nan

df_pima_aligned = df_pima_clean[df_base.columns]
df_combined = pd.concat([df_base, df_pima_aligned], ignore_index=True)
df = df_combined

print(f"✅ Combined dataset shape: {df.shape}")
print(f"✅ Total samples: {len(df)}")

# ==================== DATA CLEANING ====================
print("🧹 Cleaning and preprocessing data...")

# Convert Diabetes_SelfReport
df['Diabetes_SelfReport'] = df['Diabetes_SelfReport'].replace(3, 0)
df['Diabetes_SelfReport'] = df['Diabetes_SelfReport'].replace([7, 9], 0)

# Convert to numeric and fill missing values
numeric_cols = [
    'Age', 'BMI', 'Waist_Circumference', 'BP_Systolic', 'BP_Diastolic',
    'Glucose_mg_dL', 'HbA1c', 'Insulin_ml', 'Age_First_Period',
    'Gender', 'Race_Ethnicity', 'Diabetes_SelfReport', 'Regular_Periods'
]

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')
    col_skew = df[col].dropna().skew()
    replace = df[col].median() if abs(col_skew) >= 0.5 else df[col].mean()
    df[col] = df[col].fillna(replace)

df['Diabetes_SelfReport'] = df['Diabetes_SelfReport'].map({1: 1, 2: 0})

# Create target variable
glucose_flag = df['Glucose_mg_dL'] >= 126
hba1c_flag = df['HbA1c'] >= 6.5
selfreport_flag = df['Diabetes_SelfReport'] == 1
df['Diabetes_Target_Unified'] = (glucose_flag | hba1c_flag | selfreport_flag).astype(int)

# Handle outliers
key_variables = ['Glucose_mg_dL', 'HbA1c', 'Age', 'BMI', 'Insulin_ml', 
                 'Waist_Circumference', 'BP_Systolic', 'BP_Diastolic']

for var in key_variables:
    if var in df.columns:
        Q1 = df[var].quantile(0.25)
        Q3 = df[var].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        df[var] = df[var].clip(lower, upper)

# Create HOMA-IR feature
df['HOMA_IR'] = (df['Glucose_mg_dL'] * df['Insulin_ml']) / 405

print("✅ Data preprocessing complete!")

# ==================== MODEL TRAINING ====================
print("🤖 Training ensemble model...")

# Select features
features = [
    'Glucose_mg_dL', 'HbA1c', 'Age', 'BMI', 'HOMA_IR',
    'Waist_Circumference', 'BP_Systolic', 'Insulin_ml',
    'Age_First_Period', 'Gender', 'Diabetes_SelfReport','Regular_Periods'
]

X = df[features]
y = df['Diabetes_Target_Unified']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize models
xgb = XGBClassifier(
    n_estimators=300,
    learning_rate=0.01,
    max_depth=6,
    scale_pos_weight=len(y_train[y_train == 0]) / len(y_train[y_train == 1]),
    random_state=42,
    n_jobs=-1,
    eval_metric='auc'
)

rf = RandomForestClassifier(
    n_estimators=400,
    max_depth=15,
    class_weight='balanced_subsample',
    random_state=42,
    n_jobs=-1
)

xgb2 = XGBClassifier(
    n_estimators=200,
    max_depth=8,
    learning_rate=0.05,
    subsample=0.7,
    colsample_bytree=0.7,
    random_state=42,
    n_jobs=-1,
    eval_metric='auc'
)

# Create ensemble
ensemble = VotingClassifier(
    estimators=[('xgb1', xgb), ('rf', rf), ('xgb2', xgb2)],
    voting='soft',
    weights=[1.3, 1.0, 0.7]
)

# Train model
ensemble.fit(X_train_scaled, y_train)
print("✅ Model training complete!")

# ==================== MODEL EVALUATION ====================
print("📊 Evaluating model performance...")

y_pred_proba = ensemble.predict_proba(X_test_scaled)[:, 1]
y_pred = (y_pred_proba >= 0.35).astype(int)  # Using optimized threshold

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)

print("\n" + "="*50)
print("🎯 MODEL PERFORMANCE METRICS:")
print("="*50)
print(f"📈 Accuracy:      {accuracy:.3f}")
print(f"🎯 Precision:     {precision:.3f}")
print(f"🔍 Recall:        {recall:.3f}")
print(f"⚖️  F1 Score:      {f1:.3f}")
print(f"📊 ROC-AUC:       {roc_auc:.3f}")
print("="*50)

# ==================== SAVE MODEL ====================
print("\n💾 Saving model and scaler...")

# Save the model
with open('diabetes_model.pkl', 'wb') as f:
    pickle.dump(ensemble, f)

# Save the scaler
with open('diabetes_scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print("✅ Model saved as 'diabetes_model.pkl'")
print("✅ Scaler saved as 'diabetes_scaler.pkl'")
print("✅ Feature order:", features)

# ==================== SAVE SAMPLE DATA ====================
# Save a small sample for testing
sample_data = {
    'features': features,
    'X_train_sample': X_train_scaled[:5].tolist(),
    'X_test_sample': X_test_scaled[:5].tolist(),
    'feature_means': X_train.mean().to_dict(),
    'feature_stds': X_train.std().to_dict()
}

with open('model_info.json', 'w') as f:
    import json
    json.dump(sample_data, f, indent=2)

print("\n🎉 Training pipeline completed successfully!")
print(f"\n📁 Files created in: {os.getcwd()}")
print("   - diabetes_model.pkl (Trained ensemble model)")
print("   - diabetes_scaler.pkl (Feature scaler)")
print("   - model_info.json (Model metadata)")