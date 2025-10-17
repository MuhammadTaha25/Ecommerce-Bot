from openai import OpenAI
import streamlit as st
from pathlib import Path
from streamlit_mic_recorder import mic_recorder
import io
import os

from pineconedb import manage_pinecone_store
from creating_chain import create_expert_chain
from llModel import initialize_LLM

# --- Page Config ---
st.set_page_config(page_title="🛍️ ShopEase AI | Your Smart Shopping Assistant", layout="wide")

# --- Secrets & Clients ---
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
GOOGLE_API_KEY = st.secrets["GEMINI_API_KEY"]
client = OpenAI(api_key=OPENAI_API_KEY)

# --- LLM & Pinecone setup ---
llm = initialize_LLM(OPENAI_API_KEY, GOOGLE_API_KEY)
retriever = manage_pinecone_store()
chain = create_expert_chain(llm, retriever)

# --- Session State ---
if "conversation" not in st.session_state:
    st.session_state.conversation = []
if "turn" not in st.session_state:
    st.session_state.turn = 0

# --- Header ---
st.title("🛍️ ShopEase Voice Assistant")
st.markdown("### Welcome! Ask anything about our products, offers, or order updates.")
st.caption("💬 Speak or type your question below — your personal shopping assistant is here to help!")

# --- Voice Recorder ---
st.write("### 🎙️ Ask with your voice:")
audio_bytes = mic_recorder(
    start_prompt="🎤 Start Talking",
    stop_prompt="⏹️ Stop Talking",
    key=f"recorder_{st.session_state.turn}",
    format="wav"
)

# --- Text Input Alternative ---
user_text = st.text_input("Or type your question here 👇", key="typed_query")

# --- Process Audio or Text ---
if audio_bytes or user_text:
    with st.spinner("Processing your query..."):
        if audio_bytes:
            user_audio_path = Path(__file__).parent / f"user_input_{st.session_state.turn}.wav"
            with open(user_audio_path, "wb") as f:
                f.write(audio_bytes['bytes'])

            # Transcribe using Whisper
            audio_file = io.BytesIO(audio_bytes['bytes'])
            audio_file.name = "recording.wav"
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                language="en"
            ).text
        else:
            transcript = user_text
            user_audio_path = None

        # Store user input
        st.session_state.conversation.append({
            "role": "user",
            "audio": str(user_audio_path) if user_audio_path else None,
            "text": transcript
        })

        # --- Generate AI Response ---
        with st.spinner("Fetching the best answer for you..."):
            result = chain.stream({"question": transcript})
            ai_text = "".join(result) if hasattr(result, "__iter__") else str(result)

        # --- TTS (Voice Reply) ---
        bot_audio_path = Path(__file__).parent / f"bot_response_{st.session_state.turn}.mp3"
        with client.audio.speech.with_streaming_response.create(
            model="gpt-4o-mini-tts",
            voice="alloy",
            input=ai_text,
        ) as resp:
            resp.stream_to_file(bot_audio_path)

        # Save bot response
        st.session_state.conversation.append({
            "role": "bot",
            "audio": str(bot_audio_path),
            "text": ai_text
        })

        st.session_state.turn += 1
        st.rerun()

# --- Chat History ---
st.markdown("## 💬 Chat History")
if st.session_state.conversation:
    for msg in st.session_state.conversation:
        if msg["role"] == "bot":
            with st.container():
                col1, col2 = st.columns([2, 2])
                with col1:
                    st.markdown("**🛒 ShopEase:**")
                    if msg["audio"]:
                        st.audio(msg["audio"], format="audio/mp3")
                    with st.expander("🗒️ See Reply Text"):
                        st.write(msg["text"])
                st.markdown("---")
        else:
            with st.container():
                col1, col2 = st.columns([2, 2])
                with col2:
                    st.markdown("**👤 You:**")
                    if msg["audio"]:
                        st.audio(msg["audio"], format="audio/wav")
                    with st.expander("🗒️ See Your Question"):
                        st.write(msg["text"])
                st.markdown("---")
else:
    st.info("No chats yet — try asking about a product or offer!")

# --- Clear Button ---
if st.button("🧹 Clear Chat History"):
    for msg in st.session_state.conversation:
        if msg["audio"] and os.path.exists(msg["audio"]):
            os.remove(msg["audio"])
    st.session_state.conversation = []
    st.session_state.turn = 0
    st.rerun()

# --- Footer ---
st.markdown("<br><center>🛍️ Powered by ShopEase AI — Your 24/7 Shopping Partner</center>", unsafe_allow_html=True)


