import streamlit as st
import pandas as pd
import google.generativeai as genai
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from google.api_core import exceptions
from streamlit_mic_recorder import mic_recorder
import io

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Foster Finance | Deal Assistant",
    page_icon="🏦",
    layout="wide"
)

# --- 2. CUSTOM CSS (Premium Branding) ---
st.markdown("""
    <style>
        [data-testid="stSidebar"] { background-color: #0e2f44; }
        [data-testid="stSidebar"] * { color: #ecf0f1 !important; }
        h1 { color: #0e2f44; font-family: 'Helvetica Neue', sans-serif; font-weight: 700; }
        .stTextArea textarea { background-color: #f8f9fa; border: 1px solid #dcdcdc; }
        .stSuccess { background-color: #d4edda; color: #155724; }
        /* Style the Mic Button to look professional */
        button[kind="secondary"] { border-radius: 20px; }
    </style>
""", unsafe_allow_html=True)

# --- 3. CACHED HELPER FUNCTIONS ---

@st.cache_data(ttl=3600)
def get_best_model(api_key):
    genai.configure(api_key=api_key)
    try:
        models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        priorities = ['models/gemini-1.5-flash-001', 'models/gemini-1.5-flash']
        for p in priorities:
            if p in models: return p
        return models[0] if models else 'models/gemini-1.5-flash'
    except:
        return 'models/gemini-1.5-flash-001'

@st.cache_data
def load_database(file):
    return pd.read_csv(file)

def transcribe_audio(audio_bytes, api_key):
    """Sends audio directly to Gemini to convert to text."""
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-1.5-flash')
    
    # Simple prompt for accurate transcription
    prompt = "Transcribe this audio exactly. It is a credit deal summary for a finance application. Do not summarize, just transcribe."
    
    try:
        # Create a "blob" for the audio to send to Gemini
        response = model.generate_content([
            prompt,
            {"mime_type": "audio/wav", "data": audio_bytes}
        ])
        return response.text
    except Exception as e:
        return f"Error transcribing: {e}"

# --- 4. SIDEBAR SETUP ---
with st.sidebar:
    st.image("https://placehold.co/200x80/0e2f44/ffffff/png?text=Foster+Finance", use_column_width=True)
    st.markdown("---")
    st.header("⚙️ Configuration")
    api_key = st.text_input("Google API Key", type="password", help="Enter your Gemini API Key")
    
    st.markdown("---")
    st.info("🎙️ **New:** Use the Voice Recorder to dictate your deal!")

# --- 5. MAIN APPLICATION LOGIC ---

st.title("🏦 Foster Finance Deal Assistant")
st.markdown("##### AI-Powered Credit Proposal Generator")

uploaded_file = st.file_uploader("📂 Upload Foundation Database (CSV)", type=['csv'])

# Initialize session state for the text input
if 'deal_text' not in st.session_state:
    st.session_state.deal_text = ""

if uploaded_file is not None:
    try:
        df = load_database(uploaded_file)
        
        # Header Validation
        required_columns = ['Client Requirements', 'Client Objectives', 'Product Features', 'Why this Product was Selected']
        missing = [col for col in required_columns if col not in df.columns]
        
        if missing:
            st.error(f"❌ Error: Your CSV is missing headers: {', '.join(missing)}")
        else:
            st.success(f"✅ Database Active: {len(df)} deal scenarios loaded.")
            st.markdown("---")

            # --- VOICE INPUT SECTION ---
            col_mic, col_text = st.columns([1, 4])
            
            with col_mic:
                st.write("🎙️ **Voice Input**")
                # The Mic Recorder Component
                audio = mic_recorder(start_prompt="Record Deal", stop_prompt="Stop", key='recorder')
                
                if audio and api_key:
                    with st.spinner("Transcribing..."):
                        # Send audio to Gemini
                        transcription = transcribe_audio(audio['bytes'], api_key)
                        st.session_state.deal_text = transcription
                        st.rerun() # Refresh to show text in box

            with col_text:
                # The text area is bound to session_state so voice updates it
                user_input = st.text_area(
                    "📝 Deal Scenario / Keywords", 
                    value=st.session_state.deal_text,
                    height=120, 
                    placeholder="Type here OR use the Voice Button on the left..."
                )
                
                generate_btn = st.button("✨ Generate Proposal", type="primary", use_container_width=True)

            # --- AI LOGIC ---
            if generate_btn and api_key and user_input:
                
                # A. SMART MATCHING
                user_terms = set(user_input.lower().replace(',', '').split())
                def calculate_score(row):
                    row_text = str(row.values).lower()
                    return sum(1 for term in user_terms if term in row_text)

                df['match_score'] = df.apply(calculate_score, axis=1)
                matches = df.sort_values(by='match_score', ascending=False).head(3)
                
                context_type = "General Logic" if matches['match_score'].max() == 0 else "Historic Matches"
                context_data = matches[required_columns].to_markdown(index=False)

                # B. PROMPT ENGINEERING (Updated for Richer Bullet 1)
                model_name = get_best_model(api_key)
                genai.configure(api_key=api_key)
                model = genai.GenerativeModel(model_name)
                
                prompt = f"""
                Role: Senior Credit Analyst at Foster Finance.
                Task: Write a deal summary ADAPTING the style of the Reference Database to the User's new scenario.

                USER INPUT (New Deal Details): 
                "{user_input}"

                REFERENCE DATABASE ({context_type}):
                {context_data}

                INSTRUCTIONS:
                1. **Structure:** Output a numbered list (1, 2, 3) followed by a separate paragraph for the 4th point.
                2. **Tone:** Professional, persuasive, and tailored. 
                
                CRITICAL INSTRUCTION FOR BULLET 1 (Client Requirements):
                - Do NOT just repeat the user input. **EXPAND** this section. 
                - If the input is brief, INFER standard professional requirements relevant to this deal type (e.g., maximizing borrowing capacity, seeking competitive pricing, structure flexibility, or specific policy exceptions).
                - Make Bullet 1 feel personalized and comprehensive, even if the input was short.

                OUTPUT MAPPING:
                1. [Expanded 'Client Requirements']
                2. [Sentence mapping to 'Client Objectives']
                3. [Sentence mapping to 'Product Features']

                4. [Paragraph mapping to 'Why this Product was Selected']

                Generate strict Markdown output.
                """
                
                # C. GENERATION
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

            elif generate_btn and not api_key:
                st.warning("⚠️ Please enter your API Key.")

    except Exception as e:
        st.error(f"File Error: {e}")

else:
    st.info("👆 Please upload your **Database.csv** to begin.")
