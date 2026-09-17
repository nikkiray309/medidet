import json
import re
from pathlib import Path

import clip
import streamlit as st
import torch
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from PIL import Image
from pinecone import Pinecone
from streamlit_lottie import st_lottie


TEXT_INDEX_NAME = "disease-symptoms-gpt-4"
IMAGE_INDEX_NAME = "skindisease-symptoms-gpt-4"

st.set_page_config(
    page_title="MediDet-AI",
    page_icon="🕵️‍♂️",
    layout="wide",
)


def load_json(path: str):
    with open(path, encoding="utf-8") as file:
        return json.load(file)


def load_css(path: str) -> None:
    css = Path(path).read_text(encoding="utf-8")
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


@st.cache_resource
def load_clip_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    return model, preprocess, device


def analyze_image(image: Image.Image, pinecone_api_key: str, llm: ChatOpenAI) -> str:
    model, preprocess, device = load_clip_model()
    image_tensor = preprocess(image.convert("RGB")).unsqueeze(0).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image_tensor)
        image_features /= image_features.norm(dim=-1, keepdim=True)

    vector = image_features.cpu().numpy().flatten().tolist()
    index = Pinecone(api_key=pinecone_api_key).Index(IMAGE_INDEX_NAME)
    result = index.query(vector=vector, top_k=1, include_metadata=True)
    condition = result["matches"][0]["metadata"]["Disease"]
    st.write("Closest indexed condition:", condition)

    prompt = PromptTemplate.from_template(
        """Accept the user's skin condition as input and provide probable diagnoses
        and prescription for only that condition.
        Text:
        {context}"""
    )
    return (prompt | llm).invoke({"context": condition}).content


load_css("styles/custom.css")
st_lottie(
    load_json("assets/detwalking.json"),
    speed=1,
    width=600,
    height=400,
    key="detective-animation",
)

st.markdown(
    '<h1 class="app-title">MediDet-AI: A Multimodal Health Assistant</h1>',
    unsafe_allow_html=True,
)
st.markdown(
    '<h2 class="app-subtitle">A medical mystery-solving assistant for your skin</h2>',
    unsafe_allow_html=True,
)

openai_api_key = st.secrets["OPENAI_API_KEY"]
pinecone_api_key = st.secrets["PINECONE_API_KEY"]

embeddings = OpenAIEmbeddings(
    model="text-embedding-ada-002",
    api_key=openai_api_key,
)
llm = ChatOpenAI(api_key=openai_api_key, model="gpt-4o", temperature=0.0)
vectorstore = PineconeVectorStore(
    index_name=TEXT_INDEX_NAME,
    embedding=embeddings,
    pinecone_api_key=pinecone_api_key,
)

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "Hello! MediDet AI is here to help you investigate symptoms. "
                "How can I assist you today?"
            ),
        }
    ]

for message in st.session_state.messages:
    st.chat_message(message["role"]).write(message["content"])

image_answer = None
with st.sidebar:
    st.markdown(
        '<h3 class="sidebar-title">💬 Interrogate MediDet AI</h3>',
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="container">
            <div class="emoji">🕵️‍♂️</div>
            <div class="bubble">
                Got a mystery on your skin? Upload the clues. I'll investigate. 🔍
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("### 🖼️ Upload or Capture Evidence")
    image_source = st.radio("Choose method:", ["Upload Image", "Open Camera"])

    image_file = None
    if image_source == "Upload Image":
        image_file = st.file_uploader(
            "Drop the evidence (jpg/png)", type=["jpg", "png"]
        )
    else:
        image_file = st.camera_input("Live Surveillance")

    if image_file:
        image = Image.open(image_file)
        caption = "📁 Exhibit A" if image_source == "Upload Image" else "📸 Snapshot captured!"
        st.image(image, caption=caption, use_container_width=True)
        image_answer = analyze_image(image, pinecone_api_key, llm)
        st.session_state.messages.append(
            {"role": "assistant", "content": image_answer}
        )

audio_enabled = st.toggle("Audio")
if audio_enabled:
    st.info("Audio input is planned but is not implemented yet.")
elif user_prompt := st.chat_input():
    st.markdown(
        '<div class="typing">🕵️‍♂️ MediDet is investigating your case...</div>',
        unsafe_allow_html=True,
    )
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    st.chat_message("user").write(user_prompt)

    classifier_prompt = PromptTemplate.from_template(
        """Reply with Yes if the text describes medical symptoms. Otherwise reply No.
        Text:
        {context}"""
    )
    classification = (classifier_prompt | llm).invoke({"context": user_prompt}).content

    if re.search(r"\byes\b", classification, flags=re.IGNORECASE):
        medical_prompt = PromptTemplate.from_template(
            """Accept the user's symptoms as input and provide probable diseases,
            diagnoses, and prescription using only the retrieved information. Politely
            explain when the information is insufficient for a diagnosis.
            Retrieved information:
            {context}
            User symptoms:
            {input}"""
        )
        document_chain = create_stuff_documents_chain(llm, medical_prompt)
        retrieval_chain = create_retrieval_chain(
            vectorstore.as_retriever(), document_chain
        )
        answer = retrieval_chain.invoke({"input": user_prompt})["answer"]
    else:
        general_prompt = PromptTemplate.from_template(
            """Respond accurately and politely as customer support.
            Text:
            {context}"""
        )
        answer = (general_prompt | llm).invoke({"context": user_prompt}).content

    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.chat_message("assistant").write(answer)

if image_answer:
    st.chat_message("assistant").write(image_answer)
