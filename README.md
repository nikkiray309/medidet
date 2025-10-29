### GradInno Hackathon 2025 - Final project

# MediDet-AI : A Multi-modal RAG Application
MediDet-AI is an innovative, tech-forward healthcare assistant application designed to empower users to monitor their general health, with a specialized feature for skin disease detection.
## Problem Statement

Despite the growing availability of digital health tools, there remains a significant gap in early detection and personalized care for health conditions, particularly in underserved communities. Skin diseases often go unnoticed due to the lack of dermatological expertise, while other medical symptoms are frequently misunderstood or poorly communicated. Existing platforms often require users to rely on a single mode of input, limiting their accessibility and accuracy.

There is a pressing need for an intelligent, inclusive solution that supports image-based diagnosis for skin conditions and text/audio-based symptom analysis for general health concerns. This would empower users to receive real-time, AI-driven insights, irrespective of their medical literacy or geographic location.

## Proposal

MediDetAI is built to make health support feel simple, smart, and accessible. If someone has a rash, acne, or any visible skin issue, they can just snap a picture—and MediDetAI will help identify what it might be. But it does not stop there. If they are feeling unwell or confused about symptoms that are not visible, they can either type them in or say them aloud. MediDetAI listens, understands, and responds with helpful insights based on medical knowledge.

This uses Agentic AI and Implements multimodal RAG. In the text session one agent classifies whether the given text is related to the medical field or general question. For general question another agent response as the normal customer care prompt. When a medical symptomis asked by the user as a query the RAG agent triggers and responds with semantic search. The same thing goes with audio input but first the whisper agent converts the audio input to text. The image input just uses similarity search and there is a future scope in it where we want add some extra features to it.

We designed it so people do not need to be tech-savvy or medically trained. Whether it is through a photo of a skin condition, a voice note, or a few words typed in, MediDetAI uses AI to break down what might be going on and what to do next—whether it is offering precautions, home remedies, or guidance to seek care. It is like having a friendly health assistant always ready to listen and help, right in your pocket.
Implementation

## Technologies
Streamlit, OpenAI (CLIP, GPT-4, Whisper), LangChain, Pinecone, MongoDB Methods: Retrieval-Augmented Generation (RAG), Speech-to-text (Whisper), Image embeddings (CLIP), LLM-based reasoning (GPT-4) Datasets: Custom disease-symptom metadata in Pinecone, user queries stored in MongoDB


## Results & Demo

· The app supports multimodal interaction – images (via upload or webcam), voice (converted to text), and typed symptoms.

· Uses CLIP for skin condition embedding and Pinecone for fast similarity search.

· Dynamic suggestions powered by GPT-4 using disease-specific prompt templates.

· Integration with MongoDB for storing and updating user data and session history.

· Accurate matches from a vector database of skin disease profiles.



## Impact

Societal Benefits:

· Early diagnosis of common skin conditions, preventing serious complications.

· Voice input and multilingual support improve accessibility for non-tech-savvy users.

· Helpful for remote and underserved populations.

Next Steps:

· Add Support for more languages and disease categories.

· Improve the Image data and labels for RAG.

· Add an agent to detect whether the given image is a Human Face or not.

· And able to address many skin diseases.
