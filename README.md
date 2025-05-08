# 🦥 End-to-End Medical Health Care Assistant Chatbot

## 📌 Introduction

This project presents an **AI-powered Medical Health Care Assistant Chatbot** that helps users interact with their electronic health records in an intelligent and conversational way. By leveraging **Natural Language Processing (NLP)** and **Machine Learning (ML)**, the chatbot simplifies access to medical data, enhancing user understanding of symptoms, hospital stays, and treatment summaries.

Developed using **LangChain, LlamaCpp, and Streamlit**, the solution emphasizes **on-device inference** for privacy and speed, without relying on cloud-based APIs.


### ✅ Key Features
- **Medical NLP Pipeline**: Preprocesses admission records, diagnostic summaries, and medications.
- **Structured Data Integration**: Processes hospital records from real-world datasets.
- **LLM-Powered QA**: Uses Llama 2 (7B) to answer health-related queries conversationally.
- **Streamlit Interface**: Clean UI for seamless user interaction with the chatbot.
- **Local Deployment**: Entire pipeline runs offline using optimized and quantized models.

### 🧠 Technologies Used
- **Languages**: Python
- **Libraries**: `scikit-learn`, `spaCy`, `pandas`, `Streamlit`, `LangChain`, `LlamaCpp`, `PyTorch`
- **Model Hosting**: On-premise LLM inference (no external API required)

### 📊 Model Performance

#### 🏥 Hospital Length of Stay (HLOS) Prediction
- **R² Score**: `0.9915`
- **MSE**: `0.002565`
- **MAE**: `0.009588`


### ⚠️ Challenges Encountered
- **LLM Build Failures on Mac M1**: Resolved by upgrading CMake and setting custom flags.
- **Small Intent Classification Dataset**: Model underperformed due to limited training data (currently under development).
- **Complex Medical Text Processing**: Addressed with tokenization, stopword filtering, and spaCy integration.

## 🧹 Conclusion

This chatbot demonstrates how **AI can bridge healthcare data and patient interaction**, offering intelligent insights through structured NLP and ML pipelines. While intent classification and NER components are in early stages, the core system for **hospital length of stay prediction** and **data-driven medical assistance** is highly effective.
