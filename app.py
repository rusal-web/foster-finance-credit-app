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
        /* Mic button styling */
        button[kind="secondary"] { 
            border-radius: 20px; 
            border: 1px solid #ccc; 
            width: 100%;
        }
    </style>
""", unsafe_allow_html=True)

# --- 3. SESSION STATE ---
if 'deal_text' not in st.session_state:
    st.session_state.deal_text = ""

# --- 4. HELPER FUNCTIONS ---

@st.cache_data(ttl=3600)
def get_best_model(api_key):
    """
    Finds the correct model name (e.g., gemini-1.5-flash-001) supported by the key.
    """
    genai.configure(api_key=api_key)
    try:
        models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        # Priority list for speed and multimodal support
        priorities = ['models/gemini-1.5-flash-001', 'models/gemini-1.5-flash', 'models/gemini-1.5-pro']
        for p in priorities:
            if p in models: return p
        return models[0] if models else 'models/gemini-1.5-flash'
    except:
        # Fallback if list_models fails
        return 'models/gemini-1.5-flash-001'

@st.cache_data
def load_database(file):
    return pd.read_csv(file)

def transcribe_audio(audio_bytes, api_key, model_name):
    """
    Uses the SPECIFIC valid model name to transcribe audio.
    """
    genai.configure(api_key=api_key)
    # Use the specific model name we found earlier, not a hardcoded string
    model = genai.GenerativeModel(model_name)
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
    st.info("🎙️ **Voice:** Click Record -> Speak -> Click Stop. Text will appear automatically.")

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
            
            # --- PRE-CALCULATE MODEL NAME ---
            # We find the working model ONCE and use it for both Voice and Text
            valid_model_name = 'models/gemini-1.5-flash' # Default
            if st.session_state.api_key_input:
                valid_model_name = get_best_model(st.session_state.api_key_input)

            # --- VOICE LOGIC ---
            with col_mic:
                st.write("🎙️ **Voice Input**")
                audio = mic_recorder(start_prompt="Record", stop_prompt="Stop", key='recorder')
                
                if audio:
                    if st.session_state.api_key_input:
                        with st.spinner("Transcribing..."):
                            # Pass the VALID model name to the transcriber
                            new_text = transcribe_audio(
                                audio['bytes'], 
                                st.session_state.api_key_input, 
                                valid_model_name
                            )
                            
                            if new_text and new_text != st.session_state.deal_text:
                                st.session_state.deal_text = new_text
                                st.rerun()
                    elif not st.session_state.api_key_input:
                        st.warning("⚠️ Enter API Key first!")

            # --- TEXT BOX LOGIC ---
            with col_text:
                user_input = st.text_area(
                    "📝 Deal Scenario / Keywords", 
                    key="deal_text",
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

                # B. PROMPT ENGINEERING (Surgical Fix Preserved)
                genai.configure(api_key=st.session_state.api_key_input)
                # Use the same valid model name here
                model = genai.GenerativeModel(valid_model_name)
                
                prompt = f"""
                Role: Senior Credit Analyst at Foster Finance.
                Task: Write a deal summary ADAPTING the style of the Reference Database to the User's new scenario.

                USER INPUT (New Deal Details): 
                "{user_input}"

                REFERENCE DATABASE ({context_type}):
                {context_data}

                INSTRUCTIONS:
                1. **Structure:** Output a numbered list (1, 2, 3) followed by a separate paragraph for the 4th point.
                2. **Tone:** Mimic the sentence structure of the Reference Database exactly. Do not use generic AI language.
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
