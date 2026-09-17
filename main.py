import streamlit as st
from PIL import Image
import random
from streamlit_lottie import st_lottie
import json
import base64
import os
import re
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_core.vectorstores import VectorStoreRetriever
from langchain.chains import RetrievalQA
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
import torch
import clip
from pinecone import Pinecone
# --- Page Config ---
st.set_page_config(
    page_title="MediDet-AI",
    page_icon="🕵️‍♂️",
    layout="wide",
)

uploaded = None
cam = None
answer = None


@st.cache_resource
def create_embeddings(openai_api_key: str):
    """Create the shared, immutable OpenAI embedding client."""
    return OpenAIEmbeddings(
        model="text-embedding-ada-002",
        openai_api_key=openai_api_key,
    )


@st.cache_resource
def create_chat_model(openai_api_key: str):
    """Create the shared, immutable chat model client."""
    return ChatOpenAI(
        api_key=openai_api_key,
        model_name="gpt-4o",
        temperature=0.0,
    )


@st.cache_resource
def create_pinecone_client(pinecone_api_key: str):
    """Create a shared Pinecone client without caching query results."""
    return Pinecone(api_key=pinecone_api_key)


@st.cache_resource
def create_vector_store(index_name: str, _embeddings, pinecone_api_key: str):
    """Create the shared vector-store connection."""
    # The API key participates in the cache key; the SDK reads it from the environment.
    return PineconeVectorStore(index_name=index_name, embedding=_embeddings)


@st.cache_resource
def create_clip_model():
    """Load CLIP on first image analysis and reuse only the model resources."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    model.eval()
    return model, preprocess, device


def load_clip_for_analysis():
    """Load CLIP with user feedback, returning None when loading is unsafe."""
    try:
        with st.spinner("Loading the image-analysis model…"):
            return create_clip_model()
    except (MemoryError, RuntimeError):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        st.error(
            "The image-analysis model could not be loaded with the available "
            "memory. You can continue using symptom chat or try the image again."
        )
        return None
    except Exception:
        st.error(
            "Image analysis is temporarily unavailable because the model failed "
            "to load. You can continue using symptom chat and retry later."
        )
        return None


def analyze_image(image_data, llm, pinecone_api_key: str):
    """Analyze one uploaded image; all request-specific values stay local."""
    clip_resources = load_clip_for_analysis()
    if clip_resources is None:
        return None

    model, preprocess, device = clip_resources
    try:
        image_tensor = preprocess(image_data.convert("RGB")).unsqueeze(0).to(device)
        with torch.no_grad():
            image_features = model.encode_image(image_tensor)
            image_features /= image_features.norm(dim=-1, keepdim=True)

        vector = image_features.cpu().numpy().flatten()
        st.success("✅ Image converted to CLIP vector!")
        st.write("Vector shape:", vector.shape)

        index = create_pinecone_client(pinecone_api_key).Index(
            "skindisease-symptoms-gpt-4"
        )
        result = index.query(
            vector=vector.reshape(1, -1).tolist(),
            top_k=1,
            include_metadata=True,
        )
        disease = result["matches"][0]["metadata"]["Disease"]
        st.write(disease)
        prompt = PromptTemplate(
            template=(
                "Accept the user’s skin condition as input and provide probable "
                "diagnoses and prescription for only that condition.\nText:\n{context}"
            ),
            input_variables=["context"],
        )
        answer = LLMChain(llm=llm, prompt=prompt).run(disease)
        st.session_state.messages.append({"role": "assistant", "content": answer})
        return answer
    except (MemoryError, RuntimeError):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        st.error(
            "There was not enough memory to analyze this image. Try a smaller "
            "image, switch to symptom chat, or retry later."
        )
    except Exception:
        st.error("Image analysis failed. You can retry without restarting the app.")
    return None

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

openai_api_key = st.secrets["OPENAI_API_KEY"]
pinecone_api_key = st.secrets["PINECONE_API_KEY"]
os.environ["OPENAI_API_KEY"] = openai_api_key
os.environ["PINECONE_API_KEY"] = pinecone_api_key

index_name = "disease-symptoms-gpt-4"

embeddings = create_embeddings(openai_api_key)
llm = create_chat_model(openai_api_key)
vectorstore = create_vector_store(index_name, embeddings, pinecone_api_key)

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "Hello! MediDet AI is here to help you diagnose symptoms. How can I assist you today?"
}]

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
            answer = analyze_image(image, llm, pinecone_api_key)

    elif option == "Open Camera":
        
        cam = st.camera_input("Live Surveillance")
        if cam:
            image = Image.open(cam)
            st.image(image, caption="📸 Snapshot captured!", use_column_width=True)
            answer = analyze_image(image, llm, pinecone_api_key)


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
        chain = LLMChain(llm=llm, prompt=PROMPT)
        answer=chain.run(prompt)
        if re.search(r'\bYes\b', answer):
            prompt_template='''Accept the user’s symptoms as input and provide probable diseases, diagnoses and prescription using only the information stored in the vector database. politely inform the user that the data is insufficient to provide a diagnosis when the given prompt is not relavent to Medical Symptoms.    
            Text:
            {context}'''
            PROMPT = PromptTemplate(
                template=prompt_template, input_variables=["context"]
            )
            retriever = VectorStoreRetriever(vectorstore=vectorstore)
            qa_chain = RetrievalQA.from_chain_type(llm=llm,
                    chain_type="stuff",
                        retriever=retriever,
                        chain_type_kwargs={"prompt": PROMPT},)

            answer = qa_chain.run(query=prompt)
            st.session_state.messages.append({"role": "assistant", "content": answer})
            st.chat_message("assistant").write(answer)
        else:
            prompt_template='''Accept the queries as a customer care and generate an accurate reply.   
                Text:
                {context}'''
            PROMPT = PromptTemplate(
            template=prompt_template, input_variables=["context"])
            chain = LLMChain(llm=llm, prompt=PROMPT).run(prompt)
            st.session_state.messages.append({"role": "assistant", "content": chain})
            st.chat_message("assistant").write(chain)

else:
    # Set up Streamlit app layout
    st.title("Continuous Speech to Text")
    st.title("Currently still in developing phase")
    
if option == "Open Camera" and cam and answer:
        st.chat_message("assistant").write(answer)
if uploaded and answer:
        st.chat_message("assistant").write(answer)
if st.button('clear'):
    st.session_state.messages = []


