import os
import joblib
import streamlit as st
from langchain.llms import LlamaCpp  # Local LLM like Llama/Mistral
from langchain.schema import SystemMessage, HumanMessage, AIMessage

# ✅ Load trained models
intent_classifier = joblib.load("/Users/adityasrivatsav/Documents/GitHub/End-to-End-Project-on-Medical--Health-Care-Assisstant-with.-Chatbot-/models/intent_classifier.joblib")
ner_model = joblib.load("/Users/adityasrivatsav/Documents/GitHub/End-to-End-Project-on-Medical--Health-Care-Assisstant-with.-Chatbot-/models/ner_model.joblib")

# ✅ Initialize Local LLM (No OpenAI, Fully Free)
llm = LlamaCpp(model_path="../models/llama-2-7b-chat.ggmlv3.q4_0.bin")

# ✅ Define system prompt
system_prompt = SystemMessage(content="You are a medical assistant chatbot. You provide responses based on user queries.")

# ✅ Function to classify intent
def classify_intent(user_query):
    return intent_classifier.predict([user_query])[0]

# ✅ Function to extract medical entities
def extract_medical_entities(user_query):
    doc = ner_model(user_query)
    return [ent.text for ent in doc.ents]

# ✅ Function to generate chatbot response
def generate_response(user_query):
    intent = classify_intent(user_query)
    entities = extract_medical_entities(user_query)
    
    messages = [
        system_prompt,
        HumanMessage(content=f"User asked: {user_query}"),
        AIMessage(content=f"Intent: {intent}, Entities: {entities}")
    ]

    response = llm.invoke(messages).content  # ✅ Using local LLM (Llama/Mistral)
    return response, intent, entities

# ✅ Streamlit UI with Background & Image
st.set_page_config(page_title="Medical Chatbot", layout="wide")

# 🔹 Set background color & chatbot image
st.markdown(
    """
    <style>
        .stApp {
            background: linear-gradient(to right, #a8e063, #56ab2f);
            padding: 20px;
            border-radius: 15px;
        }
        .stChat {
            background: white;
            padding: 15px;
            border-radius: 10px;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Display the chatbot image
st.image("/Users/adityasrivatsav/Documents/GitHub/End-to-End-Project-on-Medical--Health-Care-Assisstant-with.-Chatbot-/images/istockphoto-1401811766-612x612.jpg", width=250)

st.title("🩺 Medical Chatbot")
st.write("Ask any medical-related question.")

# ✅ Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# ✅ Display past messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ✅ User input
user_query = st.chat_input("Ask a question...")

if user_query:
    # Store user message
    st.session_state.messages.append({"role": "user", "content": user_query})

    # Generate response
    response, intent, entities = generate_response(user_query)

    # Store bot response
    bot_response = f"**Intent:** {intent}\n\n**Entities:** {', '.join(entities) if entities else 'None'}\n\n**Response:** {response}"
    st.session_state.messages.append({"role": "assistant", "content": bot_response})

    # ✅ Display response in chat bubble
    with st.chat_message("assistant"):
        st.markdown(bot_response)