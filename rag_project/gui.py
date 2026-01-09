import streamlit as st
import base64
from app import RAGPipeline   # your existing RAG code

# ---------------------------
# PAGE CONFIG
# ---------------------------
st.set_page_config(
    page_title="Ancient Inscription RAG",
    page_icon="📜",
    layout="centered"
)

# ---------------------------
# LOAD BACKGROUND IMAGE (BASE64)
# ---------------------------
def load_bg_image(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

bg_image = load_bg_image("2.png")   # your image

# ---------------------------
# CUSTOM CSS
# ---------------------------
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700&family=Source+Serif+4:wght@400;500&display=swap');

/* ---------- GLOBAL ---------- */
html, body {{
    background-color: transparent !important;
}}

.stApp {{
    font-family: 'Source Serif 4', serif;
    color: #2E2A25;
    background-image:
        linear-gradient(rgba(20,15,10,0.35), rgba(20,15,10,0.35)),
        url("data:image/png;base64,{bg_image}");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}}

/* ---------- CENTER MAIN CARD ---------- */
.block-container {{
    max-width: 780px;
    margin: 3rem auto;
    background: rgba(245, 240, 232, 0.90);
    padding: 2.8rem 3rem;
    border-radius: 22px;
    box-shadow: 0 25px 60px rgba(0,0,0,0.45);
}}

/* ---------- TITLES ---------- */
.title {{
    font-family: 'Playfair Display', serif;
    font-size: 42px;
    text-align: center;
    color: #2B2116;
}}

.subtitle {{
    text-align: center;
    font-size: 16px;
    color: #5C4A32;
    margin-bottom: 36px;
}}

/* ---------- LABEL TEXT ---------- */
label {{
    color: #2B2116 !important;
    font-weight: 500;
}}

/* ---------- INPUT ---------- */
textarea {{
    background: #EFE6D8 !important;
    border-radius: 16px !important;
    border: 2px solid #A8905A !important;
    font-size: 16px !important;
    color: #2E2A25 !important;
    transition: box-shadow 0.3s ease, border 0.3s ease;
}}

textarea:focus {{
    border: 2px solid #8C6A3D !important;
    box-shadow: 0 0 0 4px rgba(140,106,61,0.25);
}}

/* ---------- BUTTON ---------- */
button {{
    background: linear-gradient(135deg, #8C6A3D, #6E4F2D) !important;
    color: white !important;
    border-radius: 12px !important;
    font-size: 16px !important;
    padding: 10px 26px !important;
    border: none !important;
    box-shadow: 0 8px 18px rgba(0,0,0,0.35);
}}

button:hover {{
    background: linear-gradient(135deg, #6E4F2D, #4F381F) !important;
}}

/* ---------- ANSWER CARD ---------- */
.answer-box {{
    background: #FBF8F2;
    border-left: 6px solid #8C6A3D;
    padding: 24px;
    border-radius: 16px;
    font-size: 17px;
    color: #2B2116;
    line-height: 1.65;
    box-shadow: 0 14px 35px rgba(0,0,0,0.28);
    animation: fadeUp 0.6s ease-out;
}}

@keyframes fadeUp {{
    from {{
        opacity: 0;
        transform: translateY(15px);
    }}
    to {{
        opacity: 1;
        transform: translateY(0);
    }}
}}

/* ---------- SOURCES ---------- */
.source-box {{
    background: #E8DCC6;
    padding: 14px;
    border-radius: 10px;
    font-size: 14px;
    color: #3A2E1B;
    border: 1px dashed #8C6A3D;
    margin-top: 8px;
}}

/* ---------- FOOTER ---------- */
.footer {{
    text-align: center;
    font-size: 13px;
    color: #6E5A3A;
    margin-top: 60px;
}}
</style>
""", unsafe_allow_html=True)

# ---------------------------
# SIDEBAR
# ---------------------------
st.sidebar.markdown("## 📜 Ancient RAG Explorer")
st.sidebar.markdown("""
**Domain**  
Early Indian inscriptions  

**Tech Stack**  
• Sentence Transformers  
• ChromaDB  
• Gemini LLM  

**Use Case**  
Academic research & historical QA
""")
st.sidebar.markdown("---")
st.sidebar.markdown("👨‍🎓 *Final Year Project*")

# ---------------------------
# MAIN UI
# ---------------------------
st.markdown('<div class="title">Ancient Inscription Question Answering</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Ask questions directly over epigraphic and historical texts</div>', unsafe_allow_html=True)

query = st.text_area(
    "🔎 Enter your question",
    placeholder="Example: Who established the monastery of Vardhamāni?",
    height=90
)

ask_btn = st.button("📖 Search Inscriptions")

# ---------------------------
# PIPELINE CALL
# ---------------------------
if ask_btn and query.strip():
    with st.spinner("📜 Consulting ancient records..."):
        pipeline = RAGPipeline()
        answer, sources = pipeline.run_single_query(query)

    st.markdown("### 🏛 Answer")
    st.markdown(f'<div class="answer-box">{answer}</div>', unsafe_allow_html=True)

    if sources:
        st.markdown("### 📂 Source Inscriptions")
        for src in sources:
            st.markdown(f'<div class="source-box">📄 {src}</div>', unsafe_allow_html=True)

elif ask_btn:
    st.warning("Please enter a question first.")

# ---------------------------
# FOOTER
# ---------------------------
st.markdown(
    '<div class="footer">Digitizing history through AI • Built with ❤️ using RAG</div>',
    unsafe_allow_html=True
)
