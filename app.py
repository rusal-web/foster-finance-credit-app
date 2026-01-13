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
    st.image("https://placehold.co/200x80/png?text=Foster+Finance", use_column_width=True)
    st.markdown("---")
    st.header("⚙️ Setup")
    api_key = st.text_input("Enter Google API Key", type="password", help="Your private Gemini API key.")
    st.markdown("---")
    st.markdown("**How to use:**")
    st.markdown("1. Upload **Foster Finance Database.csv**")
    st.markdown("2. Enter deal details (e.g., names, amounts, goal).")
    st.markdown("3. Click **Generate**.")

# --- HELPER: SMART MODEL SELECTOR ---
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

# --- MAIN APP LOGIC ---

st.title("🏦 Credit Summary Proposal Generator")
st.markdown("### AI-Powered Deal Structuring")

# 1. FILE UPLOADER
uploaded_file = st.file_uploader("📂 Step 1: Upload Foundation Database (CSV)", type=['csv'])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        
        # --- STRICT HEADER CHECK ---
        required_columns = ['Client Requirements', 'Client Objectives', 'Product Features', 'Why this Product was Selected']
        missing = [col for col in required_columns if col not in df.columns]
        
        if missing:
            st.error(f"❌ Error: CSV missing headers: {', '.join(missing)}")
        else:
            st.success(f"✅ Database Loaded: {len(df)} records.")
            
            # 2. USER INPUT
            st.markdown("---")
            st.write("#### 📝 Step 2: Enter Deal Details")
            
            col1, col2 = st.columns([3, 1])
            with col1:
                user_input = st.text_area("Deal Scenario", height=100, 
                                        placeholder="e.g., Federico (Riccy) and Tristan have purchased 2/6 Boronia Street...")
            with col2:
                st.markdown("<br>", unsafe_allow_html=True)
                generate_btn = st.button("✨ Generate", type="primary", use_container_width=True)

            # 3. AI GENERATION LOGIC
            if generate_btn and api_key and user_input:
                
                # --- A. SMART MATCHING (Keyword Overlap Score) ---
                # We rank rows by how many words they share with the user input
                user_terms = set(user_input.lower().split())
                
                def calculate_score(row):
                    row_text = str(row.values).lower()
                    # Count how many user keywords appear in this row
                    return sum(1 for term in user_terms if term in row_text)

                df['match_score'] = df.apply(calculate_score, axis=1)
                matches = df.sort_values(by='match_score', ascending=False).head(3)
                
                # Format the reference data for the AI
                context_data = matches[required_columns].to_markdown(index=False)

                # --- B. PROMPT ENGINEERING (Human Tone) ---
                try:
                    model_name = get_best_model(api_key)
                    genai.configure(api_key=api_key)
                    model = genai.GenerativeModel(model_name)
                    
                    prompt = f"""
                    Role: You are a Credit Analyst at Foster Finance.
                    Task: Write a deal summary by ADAPTING the style of the Reference Database to the User's new scenario.

                    USER INPUT (New Deal Details): 
                    "{user_input}"

                    REFERENCE DATABASE (Best Matches to mimic):
                    {context_data}

                    INSTRUCTIONS:
                    1. **Format:** Output a numbered list (1, 2, 3) followed by a separate paragraph for the 4th point.
                    2. **Style:** Use plain, professional English. Mimic the sentence structure of the Reference Database exactly.
                    3. **No Headers:** Do NOT use bold labels like **Requirement:** or **Objective:**. Start the sentence immediately.
                    4. **Adaptation:** Use the Reference rows as a template, but swap in the User's specific names (e.g., Federico), addresses, and amounts.

                    OUTPUT STRUCTURE:
                    1. [Sentence mapping to 'Client Requirements']
                    2. [Sentence mapping to 'Client Objectives']
                    3. [Sentence mapping to 'Product Features']

                    [Sentence mapping to 'Why this Product was Selected']
                    """
                    
                    with st.spinner("Writing proposal..."):
                        response = model.generate_content(prompt)
                        st.markdown("### 📄 Proposal")
                        st.markdown("---")
                        st.markdown(response.text)
                        
                except Exception as e:
                    st.error(f"Error: {e}")

            elif generate_btn and not api_key:
                st.warning("⚠️ Enter API Key in sidebar.")

    except Exception as e:
        st.error(f"File Error: {e}")

else:
    st.info("👆 Upload your CSV to begin.")
