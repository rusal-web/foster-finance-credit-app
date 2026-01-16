import streamlit as st
import pandas as pd
import google.generativeai as genai
from tenacity import retry, stop_after_attempt, wait_exponential

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
        div[data-testid="stVerticalBlock"] > button { width: 100%; }
    </style>
""", unsafe_allow_html=True)

# --- 3. HELPER FUNCTIONS ---

@st.cache_data(ttl=3600)
def get_valid_model_name(api_key):
    """
    CRITICAL LOGIC: 
    1. Connects to Google.
    2. Lists ALL available models.
    3. STRICTLY filters for 'gemini-1.5-flash'.
    4. Returns the exact valid ID to prevent 404s and Quota Limits.
    """
    genai.configure(api_key=api_key)
    try:
        # Get all models the key can see
        all_models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        
        # FILTER: Look for the specific '001' stable version first
        for m in all_models:
            if 'gemini-1.5-flash-001' in m: 
                return m
        
        # FILTER: Look for the standard alias
        for m in all_models:
            if 'gemini-1.5-flash' in m: 
                return m
                
        # If we are here, 1.5-Flash is missing. This usually means the Key is wrong/restricted.
        # We return a default as a hail mary, but this is rare.
        return 'models/gemini-1.5-flash'
        
    except Exception as e:
        # If listing fails entirely, return safe default
        return 'models/gemini-1.5-flash'

@st.cache_data
def load_database(file):
    return pd.read_csv(file)

# --- 4. SIDEBAR ---
with st.sidebar:
    st.image("https://placehold.co/200x80/0e2f44/ffffff/png?text=Foster+Finance", use_column_width=True)
    st.markdown("---")
    st.header("⚙️ Configuration")
    
    # Secure API Key Input
    api_key = st.text_input("Google API Key", type="password", key="api_key_input", help="Enter your NEW Project Key here")
    
    st.markdown("---")
    st.success("✅ **System Status:** Stable")
    st.caption("Target: Gemini 1.5 Flash")

# --- 5. MAIN LOGIC ---

st.title("🏦 Foster Finance Deal Assistant")
st.markdown("##### AI-Powered Credit Proposal Generator")

uploaded_file = st.file_uploader("📂 Upload Foundation Database (CSV)", type=['csv'])

if uploaded_file is not None:
    try:
        df = load_database(uploaded_file)
        
        # Validate Headers
        required_columns = ['Client Requirements', 'Client Objectives', 'Product Features', 'Why this Product was Selected']
        missing = [col for col in required_columns if col not in df.columns]
        
        if missing:
            st.error(f"❌ Error: CSV missing headers: {', '.join(missing)}")
        else:
            st.success(f"✅ Database Active: {len(df)} deal scenarios loaded.")
            
            # --- MODEL SELECTION (Runs Once) ---
            active_model_name = None
            if st.session_state.api_key_input:
                active_model_name = get_valid_model_name(st.session_state.api_key_input)

            st.markdown("---")
            
            # --- TEXT INPUT ---
            user_input = st.text_area(
                "📝 Deal Scenario / Keywords", 
                height=150, 
                placeholder="E.g., Federico and Tristan purchasing in Wollstonecraft for $2.1M. They need to refinance existing portfolio..."
            )
            
            st.markdown("<br>", unsafe_allow_html=True)
            generate_btn = st.button("✨ Generate Proposal", type="primary", use_container_width=True)

            # --- AI GENERATION ---
            if generate_btn and active_model_name and user_input:
                
                # A. SMART MATCHING
                user_terms = set(user_input.lower().replace(',', '').split())
                def calculate_score(row):
                    row_text = str(row.values).lower()
                    return sum(1 for term in user_terms if term in row_text)

                df['match_score'] = df.apply(calculate_score, axis=1)
                matches = df.sort_values(by='match_score', ascending=False).head(3)
                
                context_type = "Historic Matches" if matches['match_score'].max() > 0 else "General Logic"
                context_data = matches[required_columns].to_markdown(index=False)

                # B. PROMPT ENGINEERING (Surgical Fix)
                try:
                    genai.configure(api_key=st.session_state.api_key_input)
                    
                    # USE THE VALID NAME FOUND BY THE FILTER
                    model = genai.GenerativeModel(active_model_name)
                    
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
                    
                    with st.spinner("🤖 Dr. Foster is analyzing..."):
                        @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
                        def run_ai():
                            return model.generate_content(prompt).text
                        
                        response = run_ai()
                        st.markdown("### 📄 Draft Proposal")
                        st.markdown("---")
                        st.markdown(response)

                except Exception as e:
                    # Precise Error Handling
                    err_msg = str(e)
                    if "404" in err_msg:
                        st.error(f"🚨 404 Error: The API Key cannot find model '{active_model_name}'. Please ensure you are using a key from a standard Google AI Studio project.")
                    elif "429" in err_msg or "ResourceExhausted" in err_msg:
                        st.error("🚨 Limit Reached: Your Project Quota is full. Please create a NEW Project in Google AI Studio.")
                    else:
                        st.error(f"Analysis Error: {e}")

            elif generate_btn and not st.session_state.api_key_input:
                st.warning("⚠️ Please enter your API Key in the sidebar.")

    except Exception as e:
        st.error(f"File Error: {e}")

else:
    st.info("👆 Please upload your **Database.csv** to begin.")
