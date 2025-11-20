###############################################
# FINAL COMPLETE MULTI-AGENT LANGGRAPH SYSTEM
# USING LOCAL FAISS VECTOR STORE (NO SUPABASE)
###############################################

import os
import pickle
import json
import difflib
import spacy
import networkx as nx
import streamlit as st
from dotenv import load_dotenv
from pyvis.network import Network
import tempfile
import speech_recognition as sr

# LangChain + LangGraph
from langchain_core.documents import Document
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END

# Embeddings + LOCAL VECTOR DB
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# LLM
from langchain_groq import ChatGroq

# Audio recorder
try:
    from audio_recorder_streamlit import audio_recorder
    HAS_AUDIO_RECORDER = True
except ImportError:
    HAS_AUDIO_RECORDER = False


###############################################################################
# ENV + INITIALIZATION
###############################################################################
load_dotenv()


@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )


embeddings = load_embeddings()


@st.cache_resource
def load_local_vector_store():
    """Load FAISS index from local folder 'faiss_index'."""
    try:
        # Newer versions require allow_dangerous_deserialization=True
        vs = FAISS.load_local(
            "faiss_index",
            embeddings,
            allow_dangerous_deserialization=True
        )
    except TypeError:
        # Older versions do not accept that argument
        vs = FAISS.load_local("faiss_index", embeddings)
    except Exception as e:
        st.error(f"❌ Could not load local FAISS index: {e}")
        st.stop()

    st.success("✅ Loaded local FAISS vector store!")
    return vs


vector_store = load_local_vector_store()


@st.cache_resource
def load_kg_and_map():
    nlp_local = spacy.load("en_core_web_sm")

    with open("kg.gpickle_local1", "rb") as f:
        G = pickle.load(f)

    with open("chunk_map_local1.json", "r", encoding="utf-8") as f:
        chunk_map_local = json.load(f)

    return nlp_local, G, chunk_map_local, list(G.nodes)


nlp, KG, chunk_map, node_names = load_kg_and_map()


###############################################################################
# GRAPH RETRIEVAL HELPERS
###############################################################################
def extract_entities_from_text(text):
    doc = nlp(text)
    return [ent.text.strip() for ent in doc.ents if ent.text.strip()]


def find_matching_nodes(entity, nodes, cutoff=0.6):
    lower_entity = entity.lower()
    exact = [n for n in nodes if n.lower() == lower_entity]

    if exact:
        return exact

    return difflib.get_close_matches(entity, nodes, n=3, cutoff=cutoff)


def graph_retrieve(query, depth=1):
    ents = extract_entities_from_text(query)
    found_ids = set()

    for e in ents:
        matches = find_matching_nodes(e, node_names)

        for node in matches:
            found_ids.update(KG.nodes[node].get("chunk_ids", []))

        frontier = set(matches)

        for _ in range(depth):
            new_frontier = set()
            for f in frontier:
                for nb in KG.neighbors(f):
                    new_frontier.add(nb)
                    found_ids.update(KG.nodes[nb].get("chunk_ids", []))

            frontier = new_frontier

    return list(found_ids)


###############################################################################
# LLM
###############################################################################
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0,
    groq_api_key=os.environ["GROQ_API_KEY"],
)


###############################################################################
# SYSTEM PROMPT
###############################################################################
system_prompt = """
[... UNCHANGED — KEEP YOUR ORIGINAL SYSTEM PROMPT HERE ...]
"""


###############################################################################
# RETRIEVAL FUNCTIONS
###############################################################################
def retrieve_context(query: str) -> str:
    text_hits = vector_store.similarity_search(query, k=2)

    graph_ids = graph_retrieve(query, depth=1)

    graph_docs = []
    for cid in graph_ids:
        if cid in chunk_map:
            graph_docs.append(
                Document(
                    page_content=chunk_map[cid]["text"],
                    metadata=chunk_map[cid].get("meta", {}),
                )
            )

    combined = graph_docs + text_hits

    serialized = "\n\n".join(
        f"Source: {d.metadata.get('source', 'unknown')}\nContent: {d.page_content[:2000]}"
        for d in combined[:6]
    )
    return serialized


def summarize_context(context: str) -> str:
    prompt = f"""
Summarize the following context into 5–8 bullet points.
Keep it ≤ 220 words. Keep only medically factual content.
Remove duplicates and irrelevant info.
Context:
{context}
"""
    result = llm.invoke(prompt)
    return result.content


###############################################################################
# VOICE-TO-TEXT FUNCTION
###############################################################################
def transcribe_audio(audio_bytes):
    recognizer = sr.Recognizer()

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            tmp_file.write(audio_bytes)
            tmp_file_path = tmp_file.name

        with sr.AudioFile(tmp_file_path) as source:
            recognizer.adjust_for_ambient_noise(source, duration=0.5)
            audio_data = recognizer.record(source)

        os.remove(tmp_file_path)

        text = recognizer.recognize_google(audio_data)
        return text

    except sr.UnknownValueError:
        return "Sorry, I couldn't understand the audio. Please try speaking more clearly."
    except sr.RequestError as e:
        return f"Could not request results from speech service; {e}"
    except Exception as e:
        return f"Error processing audio: {str(e)}"


###############################################################################
# LANGGRAPH MULTI-AGENT PIPELINE
###############################################################################
from typing import TypedDict

class State(TypedDict):
    query: str
    raw_context: str
    summary: str
    messages: list


def retrieve_node(state: State):
    state["raw_context"] = retrieve_context(state["query"])
    return state


def summarize_node(state: State):
    state["summary"] = summarize_context(state["raw_context"])
    return state


def agent_node(state: State):
    conversation_history = ""

    for msg in state["messages"]:
        if isinstance(msg, HumanMessage):
            conversation_history += f"Patient: {msg.content}\n"
        elif isinstance(msg, AIMessage):
            conversation_history += f"Pam: {msg.content}\n"

    prompt = f"""
{system_prompt}

Here is the verified medical information to use:
{state['summary']}

Conversation so far:
{conversation_history}

Patient: {state['query']}

Pam:
"""

    response = llm.invoke(prompt)
    state["messages"].append(AIMessage(content=response.content))
    return state


# Build workflow
workflow = StateGraph(State)
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("summarize", summarize_node)
workflow.add_node("agent", agent_node)

workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "summarize")
workflow.add_edge("summarize", "agent")
workflow.add_edge("agent", END)

app = workflow.compile()


###############################################################################
# STREAMLIT UI
###############################################################################
st.set_page_config(page_title="Stroke Guidance Assistant", page_icon="🏥", layout="wide")

st.title("🏥 Stroke Patient Guidance Assistant")
st.caption("Powered by Multi-Agent RAG with Hybrid Retrieval | 🎤 Voice Input Available")

if "messages" not in st.session_state:
    st.session_state.messages = [SystemMessage(content=system_prompt)]

if "show_graph" not in st.session_state:
    st.session_state.show_graph = False


# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    show_graph = st.checkbox("Show Knowledge Graph", value=st.session_state.show_graph)
    st.session_state.show_graph = show_graph

    if st.button("Clear Conversation"):
        st.session_state.messages = [SystemMessage(content=system_prompt)]
        st.rerun()

    st.markdown("---")
    st.markdown("""
### About
This AI assistant helps stroke survivors with questions about:
- Returning to driving
- Rehabilitation
- Vision issues
- Fatigue management
- Medical assessments

**Note:** Always consult your doctor for medical decisions.
""")


# Chat history
chat_container = st.container()
with chat_container:
    for msg in st.session_state.messages:
        if isinstance(msg, HumanMessage):
            with st.chat_message("user"):
                st.markdown(msg.content)
        elif isinstance(msg, AIMessage):
            with st.chat_message("assistant", avatar="👩‍⚕️"):
                st.markdown(msg.content)


# Knowledge Graph display
if st.session_state.show_graph:
    st.subheader("🧠 Interactive Knowledge Graph")

    net_full = Network(
        height="500px", width="100%", bgcolor="#222222", font_color="white"
    )
    net_full.barnes_hut()

    nodes_to_show = list(KG.nodes())[:150]

    for node in nodes_to_show:
        net_full.add_node(node, label=node, color="skyblue", title=node)

    for a, b in KG.edges():
        if a in nodes_to_show and b in nodes_to_show:
            net_full.add_edge(a, b)

    try:
        net_full.save_graph("kg_full.html")
        with open("kg_full.html", "r", encoding="utf-8") as f:
            st.components.v1.html(f.read(), height=550)
    except Exception as e:
        st.error(f"Graph display error: {e}")


# Input
st.markdown("---")

col1, col2 = st.columns([5, 1])

user_input = None

with col1:
    text_input = st.chat_input("💬 Type your question here...")
    if text_input:
        user_input = text_input

with col2:
    if HAS_AUDIO_RECORDER:
        st.markdown("#### 🎤 Voice")
        audio_bytes = audio_recorder("", "#e74c3c", "#3498db", "1x")

        if audio_bytes:
            with st.spinner("🎤 Transcribing audio..."):
                transcribed = transcribe_audio(audio_bytes)

            if not transcribed.startswith("Sorry") and not transcribed.startswith("Error"):
                st.success(f"Heard: {transcribed[:50]}...")
                user_input = transcribed
            else:
                st.error(transcribed)
    else:
        st.info("Install audio-recorder-streamlit for voice input")


# Processing
if user_input:
    with st.chat_message("user"):
        st.markdown(user_input)

    st.session_state.messages.append(HumanMessage(content=user_input))

    with st.spinner("🤔 Thinking..."):
        result = app.invoke(
            {"query": user_input, "messages": st.session_state.messages}
        )

    retrieved_ids = graph_retrieve(user_input)

    ai_reply = result["messages"][-1].content

    with st.chat_message("assistant", avatar="👩‍⚕️"):
        st.markdown(ai_reply)

    st.session_state.messages = result["messages"]

    if retrieved_ids:
        with st.expander(f"📚 Retrieved {len(retrieved_ids)} chunks"):
            st.write("The AI used information from these knowledge areas:")
            for i, cid in enumerate(retrieved_ids[:3]):
                if cid in chunk_map:
                    st.markdown(
                        f"**Chunk {i+1}:** {chunk_map[cid]['text'][:200]}..."
                    )

    st.rerun()
