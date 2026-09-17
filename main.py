import streamlit as st
from PIL import Image
import random
from streamlit_lottie import st_lottie
import json
import base64
import os
import re
from dotenv import load_dotenv,dotenv_values
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
import io
import speech_recognition as sr
import torch
import clip
from pinecone import Pinecone
import openai
import whisper
import tempfile
# --- Page Config ---
st.set_page_config(
    page_title="MediDet-AI",
    page_icon="🕵️‍♂️",
    layout="wide",
)

INITIAL_ASSISTANT_GREETING = (
    "Hello! MediDet AI is here to help you diagnose symptoms. "
    "How can I assist you today?"
)
IMAGE_RESULT_STATE_KEYS = ("image_data", "image_result")

global uploaded
uploaded=False

def load_lottie_url(path: str):
    with open(path, "r") as file:
        return json.load(file)
    
# Path to your det.json animation
lottie_animation = load_lottie_url("assets/detwalking.json")


# --- Display Lottie Animation Before Title ---
st_lottie(lottie_animation, speed=1, width=600, height=400, key="detective-animation")

# --- Custom Detective CSS with Background ---
st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Special+Elite&display=swap');

    html, body {{
        background-image: url("https://images.unsplash.com/photo-1603126855021-4022415d0d2e?auto=format&fit=crop&w=1650&q=80");
        background-size: cover;
        background-attachment: fixed;
        background-position: center;
        font-family: 'Special Elite', Courier, monospace;
        color: #e0d8c3;
    }}

    .chatbox {{
        background-color: rgba(46, 46, 46, 0.85);
        border-left: 5px solid #f4d35e;
        padding: 15px;
        border-radius: 6px;
        margin-top: 15px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.5);
    }}

    .verdict {{
        font-size: 22px;
        font-weight: bold;
        color: #f4d35e;
    }}

    .centered {{
        display: flex;
        justify-content: center;
        align-items: center;
        flex-direction: column;
        text-align: center;
    }}

    .container {{
        display: flex;
        align-items: flex-start;
        gap: 10px;
        justify-content: center;
        margin: 20px;
    }}

    .emoji {{
        font-size: 60px;
        line-height: 1;
    }}

    .bubble {{
        position: relative;
        background: rgba(61, 61, 61, 0.85);
        border: 2px dashed #f4d35e;
        border-radius: 15px;
        padding: 20px;
        color: #fff;
        font-size: 17px;
        max-width: 400px;
        font-family: 'Special Elite', Courier, monospace;
    }}

    .bubble:hover {{
        cursor: zoom-in;
        box-shadow: 0 0 10px #f4d35e;
    }}

    .stButton button {{
        background-color: #f4d35e !important;
        color: black !important;
        font-weight: bold;
        border-radius: 10px;
        padding: 0.5em 1em;
        border: none;
        transition: background-color 0.3s ease, transform 0.2s ease;
    }}

    .stButton button:hover {{
        background-color: #ffcc00 !important;
        transform: scale(1.1);
    }}

    .stTextInput > div > div > input {{
        font-family: 'Special Elite';
        background-color: rgba(46, 46, 46, 0.85);
        color: #e0d8c3;
        border: 1px solid #f4d35e;
    }}

    .stRadio > div {{
        font-family: 'Special Elite';
        background-color: rgba(46, 46, 46, 0.85);
        border-radius: 10px;
        padding: 10px;
    }}

    .stImage img {{
        border: 3px solid #f4d35e;
        border-radius: 8px;
    }}

    .stSuccess {{
        background-color: rgba(61, 61, 61, 0.85) !important;
        border-left: 5px solid #90ee90 !important;
    }}

    /* Typing animation */
    .typing {{
        font-family: 'Special Elite';
        font-size: 18px;
        color: #ffffff;
        display: inline-block;
        width: 0;
        white-space: nowrap;
        overflow: hidden;
        border-right: 4px solid #f4d35e;
        animation: typing 3s steps(30) 1s 1 normal both, blink 0.75s step-end infinite;
    }}

    @keyframes typing {{
        from {{ width: 0; }}
        to {{ width: 100%; }}
    }}

    @keyframes blink {{
        50% {{ border-color: transparent; }}
    }}

    /* Toggle Button Styling */
    .switch {{
        position: fixed;
        top: 10px;
        right: 10px;
        display: flex;
        align-items: center;
        justify-content: center;
        z-index: 1000;
    }}

    .switch input {{
        opacity: 0;
        position: absolute;
        width: 0;
        height: 0;
    }}

    .slider {{
        cursor: pointer;
        position: relative;
        width: 60px;
        height: 34px;
        border-radius: 34px;
        background-color: #ccc;
        transition: 0.4s;
    }}

    .slider:before {{
        content: "";
        position: absolute;
        height: 26px;
        width: 26px;
        border-radius: 50%;
        left: 4px;
        bottom: 4px;
        background-color: white;
        transition: 0.4s;
    }}

    input:checked + .slider {{
        background-color: #4CAF50;
    }}

    input:checked + .slider:before {{
        transform: translateX(26px);
    }}

    </style>
    """,
    unsafe_allow_html=True
)

# --- Main Panel: Title & Chat ---
st.markdown(
    "<h1 style='font-family: \"Special Elite\"; font-size: 36px; color: #f4d35e; text-align: center;'>MediDet-AI: A Multimodal Health Assistant</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<h2 style='font-family: \"Special Elite\"; font-size: 24px; color: #e0d8c3; text-align: center;'>Along with a medical mystery-solving assistant for your skin</h2>",
    unsafe_allow_html=True
)

config = dotenv_values("keys.env")
os.environ['OPENAI_API_KEY'] = st.secrets["OPENAI_API_KEY"]
os.environ['PINECONE_API_KEY'] = st.secrets["PINECONE_API_KEY"]

index_name = "disease-symptoms-gpt-4"

embed = OpenAIEmbeddings(
model='text-embedding-ada-002',
api_key=os.environ['OPENAI_API_KEY']
)

llm=ChatOpenAI(api_key=os.environ['OPENAI_API_KEY'],
                   model='gpt-4o',
                   temperature=0.0)

vectorstore = PineconeVectorStore(index_name=index_name, embedding=embed)

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": INITIAL_ASSISTANT_GREETING}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# --- Sidebar: Detective Bubble & Upload ---
with st.sidebar:
    st.markdown(
    "<h3 style='font-family: \"Special Elite\"; font-size: 20px; color: #ffffff; text-align: left;'>💬 Interrogate MediDet AI</h3>",
    unsafe_allow_html=True)
    bubble_html = """
    <div class="container">
        <div class="emoji">🕵️‍♂️</div>
        <div class="bubble">
            Got a mystery on your skin? Upload the clues. I'll investigate. 🔍
        </div>
    </div>
    """
    st.markdown(bubble_html, unsafe_allow_html=True)
    st.markdown("<div class='centered'>", unsafe_allow_html=True)
    st.markdown("### 🖼️ Upload or Capture Evidence", unsafe_allow_html=True)
    option = st.radio("Choose method:", ["Upload Image", "Open Camera"])
    st.markdown("</div>", unsafe_allow_html=True)

    image_data = None
    if option == "Upload Image":
        uploaded = st.file_uploader("Drop the evidence (jpg/png)", type=['jpg', 'png'])
        if uploaded:
            image = Image.open(uploaded)
            st.image(image, caption="📁 Exhibit A", use_column_width=True)
            image_data = image
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model, preprocess = clip.load("ViT-B/32", device=device)
            image = image_data.convert("RGB")
            image_tensor = preprocess(image).unsqueeze(0).to(device)
            with torch.no_grad():
                image_features = model.encode_image(image_tensor)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                vector = image_features.cpu().numpy().flatten()
                st.success("✅ Image converted to CLIP vector!")
                st.write("Vector (first 10 values):", vector.shape)
                index_name = "skindisease-symptoms-gpt-4"
                pc = Pinecone(api_key=os.environ['PINECONE_API_KEY'])
                index = pc.Index(index_name)
                rv=vector.reshape(1, -1)
                result= index.query(vector=rv.flatten().tolist(), top_k=1, include_metadata=True)
                prompt=result['matches'][0]['metadata']['Disease']
                st.write(prompt)
                prompt_template='''Accept the user’s skin condition as input and provide probable diagnoses and prescription for only that condition.    
                Text:
                {context}'''
                PROMPT = PromptTemplate(
                template=prompt_template,input_variables=["context"])
                chain = PROMPT | llm
                answer=chain.invoke({"context": prompt}).content
                st.session_state.messages.append({"role": "assistant", "content": answer})

    elif option == "Open Camera":
        
        cam = st.camera_input("Live Surveillance")
        if cam:
            image = Image.open(cam)
            st.image(image, caption="📸 Snapshot captured!", use_column_width=True)
            image_data = image
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model, preprocess = clip.load("ViT-B/32", device=device)
            image = image_data.convert("RGB")
            image_tensor = preprocess(image).unsqueeze(0).to(device)
            with torch.no_grad():
                image_features = model.encode_image(image_tensor)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                vector = image_features.cpu().numpy().flatten()
                st.success("✅ Image converted to CLIP vector!")
                st.write("Vector (first 10 values):", vector.shape)
                index_name = "skindisease-symptoms-gpt-4"
                pc = Pinecone(api_key=os.environ['PINECONE_API_KEY'])
                index = pc.Index(index_name)
                rv=vector.reshape(1, -1)
                result= index.query(vector=rv.flatten().tolist(), top_k=1, include_metadata=True)
                prompt=result['matches'][0]['metadata']['Disease']
                st.write(prompt)
                prompt_template='''Accept the user’s skin condition as input and provide probable diagnoses and prescription for only that condition.    
                Text:
                {context}'''
                PROMPT = PromptTemplate(
                template=prompt_template,input_variables=["context"])
                chain = PROMPT | llm
                answer=chain.invoke({"context": prompt}).content
                st.session_state.messages.append({"role": "assistant", "content": answer})


flag = st.toggle("Audio")
if not flag:
    prompt_template='''If Medical Symptoms type yes else give politely inform the user that the data is insufficient to provide a diagnosis   
    Text:
    {context}'''
    PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context"]
    )

    if prompt := st.chat_input():
        st.markdown('<div class="typing">🕵️‍♂️ Skin Scout is investigating your case...</div>', unsafe_allow_html=True)
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)
        chain = PROMPT | llm
        answer=chain.invoke({"context": prompt}).content
        if re.search(r'\bYes\b', answer):
            prompt_template='''Accept the user’s symptoms as input and provide probable diseases, diagnoses and prescription using only the information stored in the vector database. Politely inform the user that the data is insufficient to provide a diagnosis when the given prompt is not relevant to medical symptoms.
            Retrieved information:
            {context}
            User symptoms:
            {input}'''
            PROMPT = PromptTemplate(
                template=prompt_template, input_variables=["context", "input"]
            )
            document_chain = create_stuff_documents_chain(llm, PROMPT)
            qa_chain = create_retrieval_chain(
                vectorstore.as_retriever(), document_chain
            )

            answer = qa_chain.invoke({"input": prompt})["answer"]
            st.session_state.messages.append({"role": "assistant", "content": answer})
            st.chat_message("assistant").write(answer)
        else:
            prompt_template='''Accept the queries as a customer care and generate an accurate reply.   
                Text:
                {context}'''
            PROMPT = PromptTemplate(
            template=prompt_template, input_variables=["context"])
            chain = PROMPT | llm
            answer = chain.invoke({"context": prompt}).content
            st.session_state.messages.append({"role": "assistant", "content": answer})
            st.chat_message("assistant").write(answer)

else:
    # Set up Streamlit app layout
    st.title("Continuous Speech to Text")
    st.title("Currently still in developing phase")
    
if option == "Open Camera" and cam:
        st.chat_message("assistant").write(answer)
if uploaded:
        st.chat_message("assistant").write(answer)
if st.button("clear"):
    st.session_state["messages"] = [
        {"role": "assistant", "content": INITIAL_ASSISTANT_GREETING}
    ]
    for key in IMAGE_RESULT_STATE_KEYS:
        st.session_state.pop(key, None)
    st.rerun()


