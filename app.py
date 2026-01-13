import streamlit as st
import pandas as pd
import google.generativeai as genai
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from google.api_core import exceptions

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Foster Finance | Credit Summary Generator",
    page_icon="🏦",
    layout="wide"
)

# --- SIDEBAR: BRANDING & SETUP ---
with st.sidebar:
    # REPLACE WITH YOUR LOGO URL
    st.image("https://placehold.co/200x80/png?text=Foster+Finance", use_column_width=True)
    st.markdown("---")
    st.header("⚙️ Setup")
    api_key = st.text_input("Enter Google API Key", type="password", help="Your private Gemini API key.")
    st.markdown("---")
    st.markdown("**How to use:**")
    st.markdown("1. Upload your **Database.csv**.")
    st.markdown("2. Enter deal keywords.")
    st.markdown("3. Click **Generate Summary**.")

# --- HELPER: SMART MODEL SELECTOR (The Fix) ---
def get_best_model(api_key):
    """
    Automatically finds the working model name for this specific API Key.
    Prevents 'NotFound' errors.
    """
    genai.configure(api_key=api_key)
    try:
        # Ask Google what models are available
        models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        
        # Priority List (Try these first)
        priorities = [
            'models/gemini-1.5-flash-001',
            'models/gemini-1.5-flash',
            'models/gemini-1.5-flash-latest',
            'models/gemini-1.5-pro',
            'models/gemini-pro'
        ]
        
        # 1. Check for exact match in priority list
        for p in priorities:
            if p in models:
                return p
        
        # 2. Fallback: Find anything with "flash" in the name
        for m in models:
            if "flash" in m:
                return m
                
        # 3. Last Resort: First available model
        return models[0] if models else 'models/gemini-1.5-flash'
        
    except Exception:
        # If listing fails, default to the safest bet
        return 'models/gemini-1.5-flash-001'

# --- MAIN APP LOGIC ---

st.title("🏦 Credit Summary Proposal Generator")
st.markdown("### AI-Powered Deal Structuring")

# 1. FILE UPLOADER
uploaded_file = st.file_uploader("📂 Step 1: Upload Foundation Database (CSV)", type=['csv'])

if uploaded_file is not None:
    try:
        # Load Data
        df = pd.read_csv(uploaded_file)
        
        # --- STRICT HEADER CHECK ---
        required_columns = [
            'Client Requirements', 
            'Client Objectives', 
            'Product Features', 
            'Why this Product was Selected'
        ]
        
        missing = [col for col in required_columns if col not in df.columns]
        
        if missing:
            st.error(f"❌ Error: The uploaded CSV is missing required headers: {', '.join(missing)}")
        else:
            st.success(f"✅ Database Loaded: {len(df)} records available.")
            
            # 2. USER INPUT
            st.markdown("---")
            st.write("#### 📝 Step 2: Enter Deal Details")
            
            col1, col2 = st.columns([3, 1])
            with col1:
                user_input = st.text_area("Deal Summary / Keywords", height=100, 
                                        placeholder="E.g., Client seeks to refinance investment property portfolio...")
            with col2:
                st.markdown("<br>", unsafe_allow_html=True) # Spacer
                generate_btn = st.button("✨ Generate Summary", type="primary", use_container_width=True)

            # 3. AI GENERATION LOGIC
            if generate_btn and api_key and user_input:
                
                # --- A. SEARCH ---
                mask = pd.Series([False] * len(df))
                for col in df.columns:
                    mask |= df[col].astype(str).str.lower().str.contains(user_input.lower(), na=False)
                
                matches = df[mask]
                
                if len(matches) == 0:
                    st.warning("⚠️ No exact matches found. Generating based on credit logic.")
                    context_data = "No specific database match. Use general credit structuring principles."
                else:
                    st.info(f"🔍 Found {len(matches)} similar past deals. Synthesizing...")
                    context_data = matches[required_columns].head(3).to_markdown(index=False)

                # --- B. PROMPT ENGINEERING ---
                try:
                    # 1. Auto-Select the Best Model
                    model_name = get_best_model(api_key)
                    # st.caption(f"🤖 Using Engine: {model_name}") # Optional: Show user which model picked
                    
                    genai.configure(api_key=api_key)
                    model = genai.GenerativeModel(model_name)
                    
                    prompt = f"""
                    Role: Senior Credit Analyst for Foster Finance.
                    Task: Write a 4-point Credit Proposal.

                    USER INPUT: "{user_input}"
                    HISTORY: {context_data}

                    INSTRUCTIONS:
                    Synthesize exactly 4 bullets into two sections.
                    
                    SECTION 1: REQUIREMENTS & OBJECTIVES
                    * **Bullet 1:** Synthesize 'Client Requirements' (Column F).
                    * **Bullet 2:** Synthesize 'Client Objectives' (Column G).
                    * **Bullet 3:** Synthesize 'Product Features' (Column H).

                    SECTION 2: PRODUCT SELECTION
                    * **Bullet 4:** Synthesize 'Why this Product was Selected' (Column I).

                    Format: Strict Markdown.
                    """
                    
                    # Robust Retry Logic
                    @retry(
                        retry=retry_if_exception_type(exceptions.ResourceExhausted),
                        stop=stop_after_attempt(3), 
                        wait=wait_exponential(multiplier=1, min=2, max=10)
                    )
                    def run_ai():
                        return model.generate_content(prompt).text
                        
                    with st.spinner("🤖 Dr. Foster is analyzing..."):
                        response_text = run_ai()
                        st.markdown("### 📄 Generated Proposal")
                        st.markdown("---")
                        st.markdown(response_text)
                        st.success("Analysis Complete!")
                        
                except Exception as e:
                    st.error(f"Analysis Failed: {e}")
                    st.error("Tip: Check if your API Key has 'Generative Language API' enabled in Google Cloud Console.")

            elif generate_btn and not api_key:
                st.warning("⚠️ Please enter your API Key in the sidebar.")

    except Exception as e:
        st.error(f"File Error: {e}")
