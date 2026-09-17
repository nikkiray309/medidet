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
from langchain_core.vectorstores import VectorStoreRetriever
from langchain.chains import RetrievalQA
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
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

global uploaded
uploaded=False

MEDICAL_DISCLAIMER = """
**Medical disclaimer:** MediDet-AI provides general educational information only. It
does not provide a diagnosis, treatment plan, or prescription and is not a substitute
for care from a qualified health professional. Images and AI-generated text can be
incomplete or wrong. If you may be experiencing a medical emergency, contact your
local emergency services now. Seek professional evaluation for symptoms that are
serious, worsening, or persistent.
"""

RED_FLAG_PATTERNS = {
    "difficulty breathing": r"\b(can'?t breathe|cannot breathe|difficulty breathing|shortness of breath|choking)\b",
    "chest pain": r"\b(chest pain|chest pressure|crushing chest)\b",
    "possible stroke": r"\b(face droop|facial droop|slurred speech|one[- ]sided weakness|sudden weakness)\b",
    "severe allergic reaction": r"\b(anaphylaxis|throat (?:is )?closing|swollen tongue|tongue swelling)\b",
    "loss of consciousness": r"\b(unconscious|passed out|loss of consciousness|not waking)\b",
    "severe bleeding": r"\b(severe bleeding|bleeding (?:won't|will not) stop|hemorrhag)\b",
    "suicide or self-harm risk": r"\b(kill myself|suicid(?:e|al)|self[- ]harm|hurt myself)\b",
}


def detect_emergency_red_flags(text):
    """Return educational red-flag labels found in user-provided text."""
    normalized_text = text or ""
    return [
        label
        for label, pattern in RED_FLAG_PATTERNS.items()
        if re.search(pattern, normalized_text, flags=re.IGNORECASE)
    ]


def source_metadata(metadata):
    """Keep source provenance visible without treating it as generated guidance."""
    return metadata or {"metadata": "No source metadata was supplied."}


CLINICAL_GUIDANCE_PROMPT = """
You are providing cautious, general health education, not medical care.

RETRIEVED EVIDENCE:
{context}

USER'S DESCRIPTION:
{question}

Rules:
- Never state or imply a definitive diagnosis. Present a short list of possible
  explanations using uncertainty language (for example, "could," "may," or
  "one possibility") and explain that an in-person assessment may differ.
- Do not prescribe, recommend, name, dose, start, stop, or change medication.
- Clearly label separate sections "Retrieved evidence" and "Generated educational
  guidance." Do not represent generated statements as retrieved facts.
- Mention important limitations and useful questions a clinician may ask.
- If the description suggests emergency warning signs (including trouble breathing,
  chest pain, stroke signs, anaphylaxis, loss of consciousness, severe bleeding, or
  imminent self-harm), begin with an "Emergency" section instructing the user to
  contact local emergency services now. Do not delay that instruction with analysis.
- Advise evaluation by an appropriately qualified health professional for serious,
  worsening, or persistent symptoms. Do not give false reassurance.
- If evidence is insufficient or unrelated, say so plainly rather than guessing.
- End with: "This is educational information, not a diagnosis or treatment plan."
"""

IMAGE_GUIDANCE_PROMPT = """
You are providing cautious, general educational information about an image match.

RETRIEVED IMAGE-MATCH EVIDENCE AND METADATA:
{context}

Rules:
- An image similarity match is not a diagnosis. Never claim a definitive diagnosis.
- Present possible explanations only as uncertain educational information.
- Do not prescribe, recommend, name, dose, start, stop, or change medication.
- Clearly separate "Retrieved evidence" from "Generated educational guidance," and
  identify the retrieved item as a similarity result rather than a clinical finding.
- Explain image-only limitations and advise professional evaluation, especially for
  serious, rapidly changing, painful, infected-looking, or persistent skin symptoms.
- Tell the user to contact local emergency services now for emergency warning signs
  such as trouble breathing, facial/throat swelling, fainting, or a rapidly spreading
  severe reaction.
- End with: "This is educational information, not a diagnosis or treatment plan."
"""

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

st.warning(MEDICAL_DISCLAIMER)
acknowledged = st.checkbox(
    "I have read and understand the medical disclaimer and want to continue.",
    key="medical_disclaimer_acknowledged",
)
st.caption(
    "Deployment requirement: the final clinical wording and escalation rules must "
    "be reviewed and approved by appropriately qualified medical and legal/compliance "
    "reviewers before this application is made public."
)
if not acknowledged:
    st.info("Please acknowledge the disclaimer before using MediDet-AI.")
    st.stop()

config = dotenv_values("keys.env")
os.environ['OPENAI_API_KEY'] = st.secrets["OPENAI_API_KEY"]
os.environ['PINECONE_API_KEY'] = st.secrets["PINECONE_API_KEY"]

index_name = "disease-symptoms-gpt-4"

embed = OpenAIEmbeddings(
model='text-embedding-ada-002',
openai_api_key=os.environ.get('OPEN_API_KEY')
)

llm=ChatOpenAI(api_key=os.environ['OPENAI_API_KEY'],
                   model_name='gpt-4o',
                   temperature=0.0)

vectorstore = PineconeVectorStore(index_name=index_name, embedding=embed)


def present_image_match(result):
    """Expose image-match evidence, then generate separately labelled guidance."""
    matches = result.get("matches", [])
    if not matches:
        st.warning("No image-match evidence was retrieved; no guidance was generated.")
        return None

    match = matches[0]
    evidence = {
        "match_id": match.get("id"),
        "similarity_score": match.get("score"),
        "metadata": source_metadata(match.get("metadata")),
    }
    st.subheader("Retrieved source metadata")
    st.json(evidence)

    prompt = PromptTemplate(template=IMAGE_GUIDANCE_PROMPT, input_variables=["context"])
    answer = LLMChain(llm=llm, prompt=prompt).run(json.dumps(evidence, default=str))
    return answer

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "Hello! MediDet-AI can provide cautious educational information about symptoms, but cannot diagnose or prescribe. How can I help?"
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
                result= index.query(vector=rv.tolist(), top_k=1, include_metadata=True)
                answer = present_image_match(result)
                if answer:
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
                result= index.query(vector=rv.tolist(), top_k=1, include_metadata=True)
                answer = present_image_match(result)
                if answer:
                    st.session_state.messages.append({"role": "assistant", "content": answer})


flag = st.toggle("Audio")
if not flag:
    prompt_template='''Classify whether the text describes medical symptoms. Reply only Yes or No. This classification is not a diagnosis.
    Text:
    {context}'''
    PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context"]
    )

    if prompt := st.chat_input():
        st.markdown('<div class="typing">🕵️‍♂️ Skin Scout is investigating your case...</div>', unsafe_allow_html=True)
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)
        red_flags = detect_emergency_red_flags(prompt)
        if red_flags:
            answer = (
                "## Emergency\n"
                "Your description contains possible emergency warning signs "
                f"({', '.join(red_flags)}). Contact your local emergency services now. "
                "Do not wait for an online response or use this application as a substitute "
                "for urgent care. If you can do so safely, alert someone nearby.\n\n"
                "This is educational information, not a diagnosis or treatment plan."
            )
            st.session_state.messages.append({"role": "assistant", "content": answer})
            st.chat_message("assistant").error(answer)
        else:
            chain = LLMChain(llm=llm, prompt=PROMPT)
            classification = chain.run(prompt)
        if not red_flags and re.search(r'\bYes\b', classification, flags=re.IGNORECASE):
            PROMPT = PromptTemplate(
                template=CLINICAL_GUIDANCE_PROMPT,
                input_variables=["context", "question"],
            )
            retriever = VectorStoreRetriever(vectorstore=vectorstore)
            qa_chain = RetrievalQA.from_chain_type(llm=llm,
                    chain_type="stuff",
                        retriever=retriever,
                        return_source_documents=True,
                        chain_type_kwargs={"prompt": PROMPT},)

            response = qa_chain.invoke({"query": prompt})
            answer = response["result"]
            source_details = [
                {
                    "source_number": number,
                    "metadata": source_metadata(document.metadata),
                    "retrieved_excerpt": document.page_content[:500],
                }
                for number, document in enumerate(response["source_documents"], start=1)
            ]
            st.subheader("Retrieved evidence and source metadata")
            st.caption(
                "The entries below came from retrieval. The assistant response that "
                "follows is generated guidance, not source evidence."
            )
            st.json(source_details)
            st.subheader("Generated educational guidance")
            st.session_state.messages.append({"role": "assistant", "content": answer})
            st.chat_message("assistant").write(answer)
        elif not red_flags:
            prompt_template='''Respond politely to the non-medical query. Do not provide a diagnosis, treatment plan, prescription, or medication advice.
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

if option == "Open Camera" and cam:
        st.chat_message("assistant").write(answer)
if uploaded:
        st.chat_message("assistant").write(answer)
if st.button('clear'):
    h.update_one({"id": 'krrish'},{"$set": {"text": ""}})


