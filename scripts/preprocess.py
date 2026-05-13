import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# Load the dataset
file_path = "/Users/adityasrivatsav/Documents/GitHub/End-to-End-Project-on-Medical--Health-Care-Assisstant-with.-Chatbot-/data/Canada_Hosp1_COVID_InpatientData.xlsx"
xls = pd.ExcelFile(file_path)

# Load relevant sheets
df_admission = pd.read_excel(xls, sheet_name='Data-at-admission')
df_days_breakdown = pd.read_excel(xls, sheet_name='Days-breakdown')
df_hospital_los = pd.read_excel(xls, sheet_name='Hospital-length-of-stay')

# Debugging: Print column names to check for patient_id
print("Hospital LOS Columns:", df_hospital_los.columns)

# Rename ID columns for consistency
df_admission.rename(columns={'id': 'patient_id'}, inplace=True)
df_hospital_los.rename(columns={'id': 'patient_id'}, inplace=True)

# Step 1: Handle Missing Values Strategically
df_admission.fillna({
    'ethnicity': "Unknown",
    'smoking_history': "Unknown",
    'year_they_quit': "Never",
    'comorbidities_other': "None",
    'height': df_admission['height'].median(),
    'weight': df_admission['weight'].median(),
}, inplace=True)

df_hospital_los.fillna({
    'hospital_length_of_stay': df_hospital_los['hospital_length_of_stay'].median(),
    'icu_length_of_stay': df_hospital_los['icu_length_of_stay'].median(),
    'time_on_mechanical_ventilation': df_hospital_los['time_on_mechanical_ventilation'].median(),
}, inplace=True)

df_days_breakdown.ffill(inplace=True)  # Forward-fill for time-series consistency

# Step 2: Compute `severity_score` if Missing
if 'severity_score' not in df_hospital_los.columns:
    print("⚠️ 'severity_score' missing. Creating it from ICU stay & hospital stay.")
    df_hospital_los['severity_score'] = (
        df_hospital_los['icu_length_of_stay'].fillna(0) * 2 + df_hospital_los['hospital_length_of_stay'].fillna(0)
    )

# Step 3: Encode Categorical Variables
df_admission['sex'] = df_admission['sex'].map({'Male': 0, 'Female': 1})
df_admission['previous_er_visit_within_14_days'] = df_admission['previous_er_visit_within_14_days'].map({'Yes': 1, 'No': 0})
df_admission['admission_disposition'] = df_admission['admission_disposition'].map({'ICU': 0, 'WARD': 1})
df_admission['intubated'] = df_admission['intubated'].map({'Yes': 1, 'No': 0})

# Convert Yes/No columns to binary
yes_no_cols = ['did_the_patient_expire_in_hospital', 'chest_x_ray', 'chest_ct', 'head_ct', 'antimicrobial', 'anticoagulation', 'steroid']
for col in yes_no_cols:
    if col in df_hospital_los.columns:
        df_hospital_los[col] = df_hospital_los[col].map({'Yes': 1, 'No': 0})

# Convert categorical text-based features explicitly to string before encoding
label_cols = ['ethnicity', 'smoking_history', 'year_they_quit', 'comorbidities_other']
for col in label_cols:
    df_admission[col] = df_admission[col].astype(str)  # Ensure all values are strings
    le = LabelEncoder()
    df_admission[col] = le.fit_transform(df_admission[col])

# Step 4: Normalize Numerical Data for RNN
scaler = MinMaxScaler()

# ✅ Scale numerical values only in their respective DataFrames
admission_num_cols = ['age', 'weight', 'height', 'systolic_blood_pressure', 'diastolic_blood_pressure', 
                      'heart_rate', 'respiratory_rate', 'oxygen_saturation', 'temperature']
df_admission[admission_num_cols] = scaler.fit_transform(df_admission[admission_num_cols])

# ✅ Scale `severity_score` separately in `df_hospital_los`
hospital_num_cols = ['hospital_length_of_stay', 'icu_length_of_stay', 'severity_score']
df_hospital_los[hospital_num_cols] = scaler.fit_transform(df_hospital_los[hospital_num_cols])

# Step 5: Drop Irrelevant Columns
columns_to_drop = ['parent_id', 'reason_for_death', 'days_to_first_covid19_test_negative']
df_hospital_los.drop(columns=columns_to_drop, errors='ignore', inplace=True)

# Step 6: Merge Datasets for RNN Training
df_merged = df_admission.merge(df_days_breakdown, left_on='patient_id', right_on='parent_id', how='left')
df_merged = df_merged.merge(df_hospital_los, on='patient_id', how='left')

# Step 7: Remove Empty or Zero Variance Columns
df_merged = df_merged.dropna(axis=1, how='all')  # Remove empty columns
df_merged = df_merged.loc[:, (df_merged != 0).any(axis=0)]  # Remove all-zero columns

# Step 8: Save Final Processed Data for RNN Model Training
output_file_path = "/Users/adityasrivatsav/Documents/GitHub/End-to-End-Project-on-Medical--Health-Care-Assisstant-with.-Chatbot-/data/Final_Hospital_LoS_RNN.csv"
df_merged.to_csv(output_file_path, index=False)

print("✅ Final Hospital Length of Stay Dataset Ready for RNN Training!")
print(f"✔ Saved as: {output_file_path}")
