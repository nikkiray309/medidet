# MediDet-AI

MediDet-AI is a Streamlit proof of concept for retrieval-assisted symptom chat and
image-similarity lookup. It was created for the GradInno Hackathon 2025.

> **Medical disclaimer:** This prototype can produce incorrect or incomplete
> output. It is not a medical device, does not establish a diagnosis, and is not a
> substitute for a qualified clinician or emergency services.

## Implemented behavior

- Accepts typed chat messages and uses an LLM to classify them as symptom-related
  or general inquiries.
- Retrieves context for symptom-related messages from a Pinecone text index and
  gives the retrieved documents to GPT-4o through a LangChain retrieval chain.
- Accepts a JPG or PNG from file upload or Streamlit's camera widget, embeds the
  image with CLIP ViT-B/32, and finds the closest disease record in a separate
  Pinecone index before asking GPT-4o to describe that condition.
- Keeps chat messages in Streamlit session state for the current browser session.

The **Audio** toggle only displays a development notice. Audio capture,
speech-to-text, multilingual support, persistent user accounts, and persistent
chat history are not implemented. The application does not use MongoDB.

## Planned behavior

- Implement audio capture and speech-to-text.
- Add multilingual input and output.
- Improve image data and labels, broaden skin-condition coverage, and reject
  images that are unsuitable for skin-condition matching.
- Add appropriate evaluation, safety controls, and clinician review before any
  use beyond a demonstration.

## Configuration

### Streamlit secrets

Create `.streamlit/secrets.toml` locally (do not commit it):

```toml
OPENAI_API_KEY = "your-openai-api-key"
PINECONE_API_KEY = "your-pinecone-api-key"
```

Both values are required at application startup. A `keys.env` or `.env` file is
not read by the application.

### Pinecone indexes

The configured Pinecone project must contain these indexes before startup:

| Index | Purpose | Embedding model | Dimensions | Required metadata |
| --- | --- | --- | ---: | --- |
| `disease-symptoms-gpt-4` | Symptom-document retrieval | OpenAI `text-embedding-ada-002` | 1536 | Text records compatible with LangChain's Pinecone vector store |
| `skindisease-symptoms-gpt-4` | Skin-image similarity | OpenAI CLIP `ViT-B/32` image encoder | 512 | `Disease` string on every match |

Index dimensions must exactly match their embedding vectors. This repository does
not include index creation or data-ingestion scripts, so operators must provision
and populate both indexes separately. The index metric should be selected to match
the normalization and ingestion process used for the stored vectors.

## Run locally

Python 3.10 or 3.11 is recommended. CLIP and PyTorch make the installation large;
the first image request may also download CLIP model weights.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
mkdir -p .streamlit
# Create .streamlit/secrets.toml as shown above.
streamlit run main.py
```

Open the local URL printed by Streamlit, normally `http://localhost:8501`.

## Deployment limitations

- A deployment needs outbound network access to OpenAI, Pinecone, Google Fonts,
  the background image host, and (on first use) the CLIP model-weight host.
- The pinned PyTorch and Git-installed CLIP dependencies produce a large build and
  may exceed memory, image-size, or build-time limits on small hosting tiers.
- Image inference uses CPU unless CUDA is available and can be slow. The CLIP model
  is cached only within one running application process.
- Streamlit session state is ephemeral and isolated per session; restarting or
  scaling the application loses chat state.
- Secrets must be configured in the hosting provider's Streamlit secrets facility,
  and both Pinecone indexes must be provisioned independently.
- Camera capture depends on browser permission and a secure context when deployed.
- There is no authentication, durable storage, rate limiting, automated index
  provisioning, medical validation, or production safety monitoring.
