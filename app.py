import streamlit as st
import pandas as pd
import google.generativeai as genai
from tenacity import retry, stop_after_attempt, wait_exponential
from streamlit_mic_recorder import mic_recorder

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Foster Finance | Deal Assistant",
    page_icon="🏦",
    layout="wide"
)

# --- 2. CUSTOM CSS ---
st.markdown("""
    <style>
        [data-testid="stSidebar"] { background-color: #0e2f44; }
        [data-testid="stSidebar"] * { color: #ecf0f1 !important; }
        h1 { color: #0e2f44; font-family: 'Helvetica Neue', sans-serif; font-weight: 700; }
        .stTextArea textarea { background-color: #f8f9fa; border: 1px solid #dcdcdc; }
        .stSuccess { background-color: #d4edda; color: #155724; }
        div[data-testid="stVerticalBlock"] > button { width: 100%; }
    </style>
""", unsafe_allow_html=True)

# --- 3. SESSION STATE ---
if 'deal_input_text' not in st.session_state:
    st.session_state.deal_input_text = ""
if 'mic_key' not in st.session_state:
    st.session_state.mic_key = 0

# --- 4. HELPER FUNCTIONS ---

@st.cache_data(ttl=3600)
def get_high_limit_model(api_key):
    """
    Forces the use of Gemini 1.5 Flash which typically has 1,500 requests/day free limit.
    """
    genai.configure(api_key=api_key)
    # We strictly force the older, stable model to avoid the 20/day limit of new models
    return 'models/gemini-1.5-flash'

@st.cache_data
def load_database(file):
    return pd.read_csv(file)

def transcribe_audio(audio_bytes, api_key):
    genai.configure(api_key=api_key)
    # Force 1.5 Flash for transcription to save quota
    model = genai.GenerativeModel('models/gemini-1.5-flash')
    try:
        response = model.generate_content([
            "Transcribe this audio exactly. It is a credit deal summary. Do not summarize.",
            {"mime_type": "audio/wav", "data": audio_bytes}
        ])
        return response.text
    except Exception as e:
        return f"Error: {e}"

# --- 5. SIDEBAR ---
with st.sidebar:
    st.image("https://placehold.co/200x80/0e2f44/ffffff/png?text=Foster+Finance", use_column_width=True)
    st.markdown("---")
    st.header("⚙️ Configuration")
    st.text_input("Google API Key", type="password", key="api_key_input", help="Enter Gemini API Key")
    st.markdown("---")
    st.info("🎙️ **Workflow:** Record -> Check Text -> Click Generate")

# --- 6. MAIN LOGIC ---

st.title("🏦 Foster Finance Deal Assistant")
st.markdown("##### AI-Powered Credit Proposal Generator")

uploaded_file = st.file_uploader("📂 Upload Foundation Database (CSV)", type=['csv'])

if uploaded_file is not None:
    try:
        df = load_database(uploaded_file)
        
        required_columns = ['Client Requirements', 'Client Objectives', 'Product Features', 'Why this Product was Selected']
        missing = [col for col in required_columns if col not in df.columns]
        
        if missing:
            st.error(f"❌ Error: CSV missing headers: {', '.join(missing)}")
        else:
            st.success(f"✅ Database Active: {len(df)} deal scenarios loaded.")
            st.markdown("---")

            col_mic, col_text = st.columns([1, 4])
            
            # --- VOICE LOGIC ---
            with col_mic:
                st.write("🎙️ **Voice Input**")
                mic_container = st.empty()
                with mic_container:
                    audio = mic_recorder(
                        start_prompt="Record", 
                        stop_prompt="Stop", 
                        key=f"recorder_{st.session_state.mic_key}"
                    )
                
                if audio:
                    if st.session_state.api_key_input:
                        with st.spinner("Transcribing..."):
                            new_text = transcribe_audio(
                                audio['bytes'], 
                                st.session_state.api_key_input
                            )
                            if new_text:
                                st.session_state.deal_input_text = new_text
                                st.session_state.mic_key += 1
                                st.rerun()
                    elif not st.session_state.api_key_input:
                        st.warning("⚠️ Enter API Key!")

            # --- TEXT BOX LOGIC ---
            with col_text:
                user_input = st.text_area(
                    "📝 Deal Scenario / Keywords", 
                    key="deal_input_text",
                    height=120, 
                    placeholder="Type here OR use the Voice Button..."
                )
                st.markdown("<br>", unsafe_allow_html=True)
                generate_btn = st.button("✨ Generate Proposal", type="primary", use_container_width=True)

            # --- AI GENERATION ---
            if generate_btn and st.session_state.api_key_input and user_input:
                
                # A. SMART MATCHING
                user_terms = set(user_input.lower().replace(',', '').split())
                def calculate_score(row):
                    row_text = str(row.values).lower()
                    return sum(1 for term in user_terms if term in row_text)

                df['match_score'] = df.apply(calculate_score, axis=1)
                matches = df.sort_values(by='match_score', ascending=False).head(3)
                
                context_type = "Historic Matches" if matches['match_score'].max() > 0 else "General Logic"
                context_data = matches[required_columns].to_markdown(index=False)

                # B. PROMPT ENGINEERING
                genai.configure(api_key=st.session_state.api_key_input)
                
                # FORCE STABLE MODEL (1,500 Requests/Day)
                model = genai.GenerativeModel('models/gemini-1.5-flash')
                
                prompt = f"""
                Role: Senior Credit Analyst at Foster Finance.
                Task: Write a deal summary ADAPTING the style of the Reference Database to the User's new scenario.

                USER INPUT (New Deal Details): 
                "{user_input}"

                REFERENCE DATABASE ({context_type}):
                {context_data}

                INSTRUCTIONS:
                1. **Structure:** Output a numbered list (1, 2, 3) followed by a separate paragraph for the 4th point.
                2. **Tone:** Mimic the sentence structure of the Reference Database exactly.
                3. **Constraint:** Do NOT use bold headers (e.g., NO "**Requirement:**"). Just start the sentence.

                SPECIFIC MAPPING INSTRUCTIONS:
                * **Bullet 1 (Requirements):** Mimic the Reference Database sentence structure, BUT add 10-15% more detail by explicitly stating the likely credit priority (e.g., "prioritising competitive rates" or "maximum borrowing") if not already stated.
                * **Bullet 2 (Objectives):** Strictly mimic the 'Client Objectives' column style.
                * **Bullet 3 (Features):** Strictly mimic the 'Product Features' column style.
                * **Point 4 (Selection):** Strictly mimic the 'Why this Product was Selected' column logic.

                Generate strict Markdown output.
                """
                
                try:
                    with st.spinner("🤖 Dr. Foster is analyzing..."):
                        @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
                        def run_ai():
                            return model.generate_content(prompt).text
                        
                        response = run_ai()
                        st.markdown("### 📄 Draft Proposal")
                        st.markdown("---")
                        st.markdown(response)

                except Exception as e:
                    st.error(f"Analysis Error: {e}")

            elif generate_btn and not st.session_state.api_key_input:
                st.warning("⚠️ Please enter your API Key in the sidebar.")

    except Exception as e:
        st.error(f"File Error: {e}")

else:
    st.info("👆 Please upload your **Database.csv** to begin.")
