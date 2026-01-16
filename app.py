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

# --- 3. HELPER FUNCTIONS ---

def get_available_models(api_key):
    """
    Directly asks Google for the list of models this key can access.
    Returns a list of names for the user to pick from.
    """
    genai.configure(api_key=api_key)
    try:
        models = genai.list_models()
        # Filter only for models that can generate content (text)
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
    
    # 1. Enter Key
    api_key = st.text_input("Google API Key", type="password", key="api_key_input")
    
    selected_model = None
    
    # 2. Connection Test & Model List
    if api_key:
        available_models = get_available_models(api_key)
        
        if available_models:
            st.success(f"✅ Key Verified! Found {len(available_models)} models.")
            
            # Try to set default to a stable Flash model
            default_ix = 0
            for i, m in enumerate(available_models):
                if 'gemini-1.5-flash' in m:
                    default_ix = i
                    break
            
            # 3. THE MANUAL OVERRIDE (User Picks the Model)
            selected_model = st.selectbox(
                "Select Model (Manual Override)",
                available_models,
                index=default_ix,
                help="If one fails, try another from this list."
            )
        else:
            st.error("❌ Key Error: Connection failed or no models found. Check if 'Generative Language API' is enabled in Google Cloud Console.")

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
                        
                        # USE THE EXACT MODEL THE USER SELECTED
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
                        
                        with st.spinner(f"🤖 Analyzing using {selected_model}..."):
                            @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
                            def run_ai():
                                return model.generate_content(prompt).text
                            
                            response = run_ai()
                            st.markdown("### 📄 Draft Proposal")
                            st.markdown("---")
                            st.markdown(response)

                    except Exception as e:
                        # Error Decoding
                        err_msg = str(e)
                        if "404" in err_msg:
                            st.error(f"🚨 Model Error: The model '{selected_model}' is not accessible. Please pick a different one from the list.")
                        elif "429" in err_msg:
                            st.error("🚨 Limit Reached: Please create a NEW Project Key.")
                        else:
                            st.error(f"Analysis Error: {e}")

            elif generate_btn and not api_key:
                st.warning("⚠️ Please enter your API Key in the sidebar.")

    except Exception as e:
        st.error(f"File Error: {e}")

else:
    st.info("👆 Please upload your **Database.csv** to begin.")
