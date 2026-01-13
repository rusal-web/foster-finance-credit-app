import streamlit as st
import pandas as pd
import google.generativeai as genai
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from google.api_core import exceptions
import time

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Foster Finance | Deal Assistant",
    page_icon="🏦",
    layout="wide"
)

# --- 2. CUSTOM CSS (The "Premium" Branding) ---
# This block forces the specific Navy Blue branding and clean font styles.
st.markdown("""
    <style>
        /* Force Sidebar Background to Investment Navy */
        [data-testid="stSidebar"] {
            background-color: #0e2f44;
        }
        /* Sidebar Text Color to White/Light Grey */
        [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3, [data-testid="stSidebar"] p, [data-testid="stSidebar"] span, [data-testid="stSidebar"] label {
            color: #ecf0f1 !important;
        }
        /* Custom Header Styling */
        h1 {
            color: #0e2f44;
            font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
            font-weight: 700;
        }
        h3 {
            color: #2c3e50;
            padding-top: 10px;
        }
        /* Clean Input Boxes */
        .stTextArea textarea {
            background-color: #f8f9fa;
            border: 1px solid #dcdcdc;
        }
        /* Success Message Box */
        .stSuccess {
            background-color: #d4edda;
            color: #155724;
        }
    </style>
""", unsafe_allow_html=True)

# --- 3. CACHED HELPER FUNCTIONS (Speed Optimization) ---

@st.cache_data(ttl=3600) # Caches the model list for 1 hour
def get_best_model(api_key):
    """
    Finds the best available model. Cached to avoid slow API calls on every click.
    """
    genai.configure(api_key=api_key)
    try:
        models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        priorities = ['models/gemini-1.5-flash-001', 'models/gemini-1.5-flash']
        for p in priorities:
            if p in models: return p
        return models[0] if models else 'models/gemini-1.5-flash'
    except:
        return 'models/gemini-1.5-flash-001'

@st.cache_data # Caches the CSV load so re-runs are instant
def load_database(file):
    return pd.read_csv(file)

# --- 4. SIDEBAR SETUP ---
with st.sidebar:
    # Placeholder for your Logo (White version looks best on Navy)
    st.image("https://placehold.co/200x80/0e2f44/ffffff/png?text=Foster+Finance", use_column_width=True)
    st.markdown("---")
    st.header("⚙️ Configuration")
    api_key = st.text_input("Google API Key", type="password", help="Enter your Gemini API Key")
    
    st.markdown("---")
    st.markdown("**Instructions:**")
    st.info("1. Upload updated **Database.csv**\n2. Enter **Deal Keywords**\n3. Click **Generate**")

# --- 5. MAIN APPLICATION LOGIC ---

st.title("🏦 Foster Finance Deal Assistant")
st.markdown("##### AI-Powered Credit Proposal Generator")

uploaded_file = st.file_uploader("📂 Upload Foundation Database (CSV)", type=['csv'])

if uploaded_file is not None:
    try:
        # Load Data (Cached)
        df = load_database(uploaded_file)
        
        # Header Validation
        required_columns = ['Client Requirements', 'Client Objectives', 'Product Features', 'Why this Product was Selected']
        missing = [col for col in required_columns if col not in df.columns]
        
        if missing:
            st.error(f"❌ Error: Your CSV is missing headers: {', '.join(missing)}")
        else:
            st.success(f"✅ Database Active: {len(df)} deal scenarios loaded.")
            
            # User Input Section
            st.markdown("---")
            col1, col2 = st.columns([3, 1])
            
            with col1:
                user_input = st.text_area("📝 Deal Scenario / Keywords", height=120, 
                                        placeholder="e.g., Federico (Riccy) and Tristan have purchased 2/6 Boronia Street for $2.1M...")
            
            with col2:
                st.markdown("<br><br>", unsafe_allow_html=True) # Spacer for alignment
                generate_btn = st.button("✨ Generate Proposal", type="primary", use_container_width=True)

            # AI Logic
            if generate_btn and api_key and user_input:
                
                # --- A. SMART MATCHING ALGORITHM ---
                # 1. Normalize user input (lowercase, split unique words)
                user_terms = set(user_input.lower().replace(',', '').split())
                
                # 2. Score every row in the DB based on keyword overlap
                def calculate_score(row):
                    # Combine all text in the row to find matches
                    row_text = str(row.values).lower()
                    return sum(1 for term in user_terms if term in row_text)

                df['match_score'] = df.apply(calculate_score, axis=1)
                
                # 3. Pick Top 3 Matches (Using your updated dataset)
                matches = df.sort_values(by='match_score', ascending=False).head(3)
                
                # Check if we actually found something relevant
                if matches['match_score'].max() == 0:
                    st.warning("⚠️ No keywords matched your database. Generating based on general credit logic.")
                    context_type = "General Logic"
                else:
                    st.toast(f"Found {len(matches)} relevant historic deals!", icon="🔍")
                    context_type = "Historic Matches"

                context_data = matches[required_columns].to_markdown(index=False)

                # --- B. PROMPT ENGINEERING (Strict Human Tone) ---
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
                2. **Tone:** Use plain, professional English. Mimic the Reference Database sentence structure exactly.
                3. **Constraint:** Do NOT use bold headers (e.g., NO "**Requirement:**"). Just start the sentence.
                4. **Logic:** - Point 1 matches 'Client Requirements' logic.
                   - Point 2 matches 'Client Objectives' logic.
                   - Point 3 matches 'Product Features' logic.
                   - Point 4 matches 'Why this Product was Selected' logic.

                Generate strict Markdown output.
                """
                
                # --- C. GENERATION & ERROR HANDLING ---
                try:
                    with st.spinner("🤖 Dr. Foster is analyzing..."):
                        
                        # Retry logic for API stability
                        @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
                        def run_ai():
                            return model.generate_content(prompt).text
                        
                        response = run_ai()
                        
                        st.markdown("### 📄 Draft Proposal")
                        st.markdown("---")
                        st.markdown(response)
                        st.caption(f"Generated using {model_name} | Context: {context_type}")

                except Exception as e:
                    st.error(f"Analysis Error: {e}")

            elif generate_btn and not api_key:
                st.warning("⚠️ Please enter your API Key in the sidebar.")

    except Exception as e:
        st.error(f"File Error: {e}")

else:
    st.info("👆 Please upload your updated **Database.csv** to begin.")
