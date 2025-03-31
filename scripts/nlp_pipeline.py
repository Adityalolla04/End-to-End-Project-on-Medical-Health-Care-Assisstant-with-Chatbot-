import pandas as pd
import re
import spacy
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import os

# Ensure models directory exists
os.makedirs("models", exist_ok=True)

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Load the dataset from the data folder
file_path = "/Users/adityasrivatsav/Documents/GitHub/End-to-End-Project-on-Medical--Health-Care-Assisstant-with.-Chatbot-/data/Canada_Hosp1_COVID_InpatientData.xlsx"
xls = pd.ExcelFile(file_path)

# Load relevant sheets
df_admission = pd.read_excel(xls, sheet_name='Data-at-admission')
df_days_breakdown = pd.read_excel(xls, sheet_name='Days-breakdown')
df_hospital_los = pd.read_excel(xls, sheet_name='Hospital-length-of-stay')
df_medications = pd.read_excel(xls, sheet_name='Medication-Static-List')

# ✅ Fix: Ensure only text is processed
def preprocess_text(text):
    """Clean and tokenize text data."""
    if pd.isna(text) or isinstance(text, bool):  # Handle NaN & boolean values
        return ""
    
    text = str(text)  # Convert everything to string
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\W+', ' ', text)  # Remove special characters
    tokens = word_tokenize(text)  # Tokenization
    tokens = [word for word in tokens if word not in stopwords.words('english')]  # Remove stopwords
    return " ".join(tokens)

# Step 2: Extract Key Medical Text Fields
df_admission['reason_for_admission'] = df_admission['reason_for_admission'].apply(preprocess_text)
df_admission['comorbidities'] = df_admission['comorbidities'].apply(preprocess_text)
df_admission['medications'] = df_admission['medications'].apply(preprocess_text)

df_hospital_los['reason_for_death'] = df_hospital_los['reason_for_death'].apply(preprocess_text)

df_days_breakdown['cxr_findings'] = df_days_breakdown['cxr_findings'].apply(preprocess_text)
df_days_breakdown['chest_ct_findings'] = df_days_breakdown['chest_ct_findings'].apply(preprocess_text)
df_days_breakdown['head_ct_findings'] = df_days_breakdown['head_ct_findings'].apply(preprocess_text)

df_medications['name'] = df_medications['name'].apply(preprocess_text)  # ✅ Fix applied

# Save preprocessed data for model training
df_admission.to_csv("/Users/adityasrivatsav/Documents/GitHub/End-to-End-Project-on-Medical--Health-Care-Assisstant-with.-Chatbot-/data/Preprocessed_Data-at-Admission.csv", index=False)
df_days_breakdown.to_csv("/Users/adityasrivatsav/Documents/GitHub/End-to-End-Project-on-Medical--Health-Care-Assisstant-with.-Chatbot-/data/Preprocessed_Days-Breakdown.csv", index=False)
df_hospital_los.to_csv("/Users/adityasrivatsav/Documents/GitHub/End-to-End-Project-on-Medical--Health-Care-Assisstant-with.-Chatbot-/data/Preprocessed_Hospital-LoS.csv", index=False)
df_medications.to_csv("/Users/adityasrivatsav/Documents/GitHub/End-to-End-Project-on-Medical--Health-Care-Assisstant-with.-Chatbot-/data/Preprocessed_Medications.csv", index=False)

print("✅ NLP preprocessing completed. Preprocessed data saved in 'data/' folder.")
