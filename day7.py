import streamlit as st
from google import genai
from dotenv import load_dotenv
import os
import fitz

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# Page config
st.set_page_config(
    page_title="PDF Chat Assistant",
    page_icon="📄",
    layout="centered"
)

# Title
st.title("📄 PDF Chat Assistant")
st.caption("Upload a PDF and ask anything about it")

# Initialize session state
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []
if "pdf_text" not in st.session_state:
    st.session_state.pdf_text = ""
if "pdf_loaded" not in st.session_state:
    st.session_state.pdf_loaded = False
if "messages" not in st.session_state:
    st.session_state.messages = []

def read_pdf(uploaded_file):
    """Extract text from uploaded PDF"""
    try:
        bytes_data = uploaded_file.read()
        doc = fitz.open(stream=bytes_data, filetype="pdf")
        full_text = ""
        num_pages = len(doc)

        for page_num in range(num_pages):
            page = doc[page_num]
            text = page.get_text()
            full_text += f"\n--- Page {page_num + 1} ---\n{text}"

        doc.close()
        return full_text, num_pages

    except Exception as e:
        return None, str(e)

def ask_question(question):
    """Send question to AI with conversation history"""

    if len(st.session_state.conversation_history) == 0:
        first_message = f"""Here is a document for you to analyze:

{st.session_state.pdf_text[:8000]}

--- END OF DOCUMENT ---

Question: {question}"""

        st.session_state.conversation_history.append({
            "role": "user",
            "parts": [{"text": first_message}]
        })
    else:
        st.session_state.conversation_history.append({
            "role": "user",
            "parts": [{"text": question}]
        })

    response = client.models.generate_content(
        model="gemini-3-flash-preview",
        contents=st.session_state.conversation_history,
        config={
            "system_instruction": """You are an expert document analyst.
Answer questions accurately based ONLY on the document content.
If the answer is not in the document, say so clearly.
Be concise and specific. Use bullet points when listing multiple items."""
        }
    )

    answer = response.text

    st.session_state.conversation_history.append({
        "role": "model",
        "parts": [{"text": answer}]
    })

    return answer

# Sidebar for PDF upload
with st.sidebar:
    st.header("📁 Upload Document")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

    if uploaded_file:
        if not st.session_state.pdf_loaded:
            with st.spinner("Reading PDF..."):
                pdf_text, result = read_pdf(uploaded_file)

                if pdf_text:
                    st.session_state.pdf_text = pdf_text
                    st.session_state.pdf_loaded = True
                    st.success(f"✅ Loaded {result} pages")
                else:
                    st.error(f"❌ Error: {result}")

    if st.session_state.pdf_loaded:
        st.divider()
        if st.button("🗑️ Clear & Upload New PDF"):
            st.session_state.conversation_history = []
            st.session_state.pdf_text = ""
            st.session_state.pdf_loaded = False
            st.session_state.messages = []
            st.rerun()

# Main chat area
if not st.session_state.pdf_loaded:
    st.info("👈 Upload a PDF from the sidebar to get started")

    # Show example questions
    st.subheader("What you can do:")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("📋 **Summarize** any document")
        st.markdown("🔍 **Find specific** information")
    with col2:
        st.markdown("📊 **Extract** key data points")
        st.markdown("💬 **Ask follow-up** questions")

else:
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if question := st.chat_input("Ask anything about your PDF..."):
        # Show user message
        st.session_state.messages.append({
            "role": "user",
            "content": question
        })
        with st.chat_message("user"):
            st.markdown(question)

        # Get and show AI response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer = ask_question(question)
            st.markdown(answer)

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer
        })