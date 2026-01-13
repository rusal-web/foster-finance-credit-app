import streamlit as st
import pandas as pd
import google.generativeai as genai
from tenacity import retry, stop_after_attempt, wait_exponential

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Foster Finance | Credit Summary Generator",
    page_icon="🏦",
    layout="wide"
)

# --- SIDEBAR: BRANDING & SETUP ---
with st.sidebar:
    # REPLACE URL BELOW WITH YOUR COMPANY LOGO URL
    st.image("https://postimg.cc/67S5Ccbj", use_column_width=True)
    st.markdown("---")
    st.header("⚙️ Setup")
    api_key = st.text_input("Enter Google API Key", type="password", help="Your private Gemini API key.")
    st.markdown("---")
    st.markdown("**How to use:**")
    st.markdown("1. Upload your **Database.csv** file.")
    st.markdown("2. Enter deal keywords (e.g., *Refinance, Investment*).")
    st.markdown("3. Click **Generate Summary**.")

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
            st.info("Please verify your CSV file matches the standard template.")
        else:
            st.success(f"✅ Database Loaded Successfully: {len(df)} records available.")
            
            # 2. USER INPUT
            st.markdown("---")
            st.write("#### 📝 Step 2: Enter Deal Details")
            
            col1, col2 = st.columns([3, 1])
            with col1:
                user_input = st.text_area("Deal Summary / Keywords", height=100, 
                                        placeholder="E.g., Client seeks to refinance investment property portfolio, requires interest-only terms...")
            with col2:
                st.markdown("<br>", unsafe_allow_html=True) # Spacer
                generate_btn = st.button("✨ Generate Summary", type="primary", use_container_width=True)

            # 3. AI GENERATION LOGIC
            if generate_btn and api_key and user_input:
                
                # --- A. SEARCH (Find relevant history) ---
                mask = pd.Series([False] * len(df))
                # Broad search to find any relevant context
                for col in df.columns:
                    mask |= df[col].astype(str).str.lower().str.contains(user_input.lower(), na=False)
                
                matches = df[mask]
                
                if len(matches) == 0:
                    st.warning("⚠️ No exact historic matches found. Generating based on general credit principles.")
                    context_data = "No direct database match found. Proceed with general credit structuring logic."
                else:
                    st.info(f"🔍 Found {len(matches)} similar past deals for context. Synthesizing proposal...")
                    # Use top 3 matches as reference context
                    context_data = matches[required_columns].head(3).to_markdown(index=False)

                # --- B. PROMPT ENGINEERING (The Core Logic) ---
                genai.configure(api_key=api_key)
                # Using flash model for speed and efficiency
                model = genai.GenerativeModel('gemini-1.5-flash')
                
                prompt = f"""
                Role: You are a Senior Credit Analyst for Foster Finance.
                Task: Generate a formal 4-point Credit Summary Proposal based on the User Input and Reference Data.

                USER INPUT: "{user_input}"

                REFERENCE DATABASE (History):
                {context_data}

                INSTRUCTIONS:
                Synthesize the output into exactly two sections with 4 bullets total.
                You must strictly adhere to the following structure based on the CSV columns:

                SECTION 1: REQUIREMENTS & OBJECTIVES
                * **Bullet 1:** Synthesize based on 'Client Requirements' (Column F).
                * **Bullet 2:** Synthesize based on 'Client Objectives' (Column G).
                * **Bullet 3:** Synthesize based on 'Product Features' (Column H).

                SECTION 2: PRODUCT SELECTION REASONING
                * **Bullet 4:** Synthesize based on 'Why this Product was Selected' (Column I). Provide the rationale for the solution.

                Output strict Markdown formatting only. Do not add intro/outro text.
                """
                
                with st.spinner("🤖 Dr. Foster is analyzing the deal..."):
                    try:
                        # Add retry logic for robustness
                        @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
                        def call_gemini(p):
                            return model.generate_content(p).text
                            
                        response_text = call_gemini(prompt)
                        
                        st.markdown("### 📄 Generated Proposal")
                        st.markdown("---")
                        st.markdown(response_text)
                        st.success("Done!")
                    except Exception as e:
                        st.error(f"An error occurred during generation: {e}")

            elif generate_btn and not api_key:
                st.warning("⚠️ Please enter your Google API Key in the sidebar to proceed.")

    except Exception as e:
        st.error(f"Error reading the file: {e}")

else:
    st.info("👆 Please begin by uploading your Foundation Database (CSV).")
