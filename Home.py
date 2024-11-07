import streamlit as st
from components import render_sidebar
from utils import init_session_state, load_dataset, get_categorical_columns, clear_session_state

st.set_page_config(
    page_title="Bias Detection and Independence Testing",
    page_icon="📊",
    layout="wide"
)

def main():
    init_session_state()
    render_sidebar()
    
    st.title("🔍 Bias Detection and Independence Testing")
    st.write("""
    Welcome to the Bias Detection and Independence Testing application. 
    This tool helps you analyze categorical data for bias and relationships.
    
    ### Features:
    - 📊 Dataset Information and Column Selection
    - 📈 Bias Detection using Goodness-of-Fit Tests
    - 🔗 Independence Testing between Categories
    - 🤖 AI-powered Data Augmentation
    """)

    # Dataset Upload Section
    st.header("📂 Dataset Upload")
    
    if st.session_state.df is None:
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        if uploaded_file is not None:
            try:
                df = load_dataset(uploaded_file)
                default_cols, optional_cols = get_categorical_columns(df)
                
                st.session_state.df = df
                st.session_state.default_categorical_cols = default_cols
                st.session_state.optional_categorical_cols = optional_cols
                st.session_state.categorical_cols = default_cols.copy()
                st.session_state.previous_file = uploaded_file.name
                
                st.success("✅ Data loaded successfully!")
                st.rerun()
            except Exception as e:
                st.error(f"Error loading file: {str(e)}")
                clear_session_state()
    else:
        st.info(f"Current dataset: {st.session_state.previous_file}")
        if st.button("Remove Dataset"):
            clear_session_state()
            st.rerun()

if __name__ == "__main__":
    main()
