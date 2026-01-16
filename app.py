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

# --- 2. CUSTOM CSS (High Contrast & Readability) ---
st.markdown("""
    <style>
        /* Main Sidebar Background */
        [data-testid="stSidebar"] { 
            background-color: #0e2f44; 
        }
        
        /* FORCE all sidebar text to be white and readable */
        [data-testid="stSidebar"] p, 
        [data-testid="stSidebar"] span, 
        [data-testid="stSidebar"] label, 
        [data-testid="stSidebar"] div { 
            color: #ffffff !important; 
        }
        
        /* Input fields in sidebar (Text Input & Select Box) */
        [data-testid="stSidebar"] input {
            color: #0e2f44 !important; /* Text inside box is dark */
            background-color: #ffffff !important; /* Box background is white */
        }
        
        /* Headers */
        h1 { color: #0e2f44; font-family: 'Helvetica Neue', sans-serif; font-weight: 700; }
        
        /* Text Area Styling */
        .stTextArea textarea { background-color: #f8f9fa; border: 1px solid #dcdcdc; }
        
        /* Success Message Styling */
        .stSuccess { background-color: #d4edda; color: #155724; border-color: #c3e6cb; }
        
        /* Button Width */
        div[data-testid="stVerticalBlock"] > button { width: 100%; }
    </style>
""", unsafe_allow_html=True)

# --- 3. HELPER FUNCTIONS ---

def get_available_models(api_key):
    """
    Fetches the real list of models from Google.
    """
    genai.configure(api_key=api_key)
    try:
        models = genai.list_models()
        return [m.name for m in models if 'generateContent' in m.supported_generation_methods]
    except Exception as e:
        return []

@st.cache_data
def load_database(file):
    return pd.read_csv(file)

# --- 4. SIDEBAR ---
with st.sidebar:
    st.image("https://placehold.co/200x80/0e2f44/ffffff/png?text=Foster+Finance", use_column_width=True)
    st.markdown("---")
    st.header("⚙️ Configuration")
    
    # 1. API Key Input
    api_key = st.text_input("Google API Key", type="password", key="api_key_input")
    
    selected_model = None
    
    # 2. Smart Model Selection
    if api_key:
        available_models = get_available_models(api_key)
        
        if available_models:
            st.success(f"✅ Connected! Found {len(available_models)} models.")
            
            # --- AUTO-SELECT LOGIC (Gemini 3 First) ---
            default_ix = 0
            
            # Priority 1: Gemini 3.0 Flash (The 2026 Standard)
            for i, m in enumerate(available_models):
                if 'gemini-3.0-flash' in m or 'gemini-3-flash' in m:
                    default_ix = i
                    break
            else:
                # Priority 2: Gemini 1.5 Flash (The Stable Fallback)
                for i, m in enumerate(available_models):
                    if 'gemini-1.5-flash' in m:
                        default_ix = i
                        break
            
            # 3. Manual Override Dropdown
            selected_model = st.selectbox(
                "Active Model (Override)",
                available_models,
                index=default_ix,
                help="Gemini 3 is selected by default. Use this to switch to 1.5 if needed."
            )
        else:
            st.error("❌ Connection Failed. Check Key.")

# --- 5. MAIN LOGIC ---

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
            
            # --- TEXT INPUT ---
            user_input = st.text_area(
                "📝 Deal Scenario / Keywords", 
                height=150, 
                placeholder="E.g., Federico and Tristan purchasing in Wollstonecraft for $2.1M..."
            )
            
            st.markdown("<br>", unsafe_allow_html=True)
            generate_btn = st.button("✨ Generate Proposal", type="primary", use_container_width=True)

            # --- AI GENERATION ---
            if generate_btn and api_key and user_input:
                if not selected_model:
                    st.error("⚠️ Please select a valid model from the sidebar first.")
                else:
                    
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
                    try:
                        genai.configure(api_key=api_key)
                        model = genai.GenerativeModel(selected_model)
                        
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
                        
                        with st.spinner(f"🤖 Analyzing with {selected_model}..."):
                            @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
                            def run_ai():
                                return model.generate_content(prompt).text
                            
                            response = run_ai()
                            st.markdown("### 📄 Draft Proposal")
                            st.markdown("---")
                            st.markdown(response)

                    except Exception as e:
                        # --- CUSTOM ERROR HANDLING ---
                        err_msg = str(e)
                        st.error("🚨 Generation Failed.")
                        
                        if "404" in err_msg or "NotFound" in err_msg:
                            st.warning(f"ℹ️ **Action Required:** The model '{selected_model}' is not currently available for this key. \n\n👉 **Please go to the Sidebar > 'Active Model' and select 'Gemini 1.5 Flash' to continue.**")
                        elif "429" in err_msg or "ResourceExhausted" in err_msg:
                            st.warning("ℹ️ **Action Required:** Daily Limit Reached. Please create a NEW Project Key or select a different model in the sidebar.")
                        else:
                            st.error(f"Technical Error: {e}")

            elif generate_btn and not api_key:
                st.warning("⚠️ Please enter your API Key in the sidebar.")

    except Exception as e:
        st.error(f"File Error: {e}")

else:
    st.info("👆 Please upload your **Database.csv** to begin.")
