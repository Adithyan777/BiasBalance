import streamlit as st
from components import render_sidebar
from utils import init_session_state
from data_augmentation import process_multiple_columns, openai_function, generate_data
import pandas as pd
import os
from openai import OpenAI,AuthenticationError

def validate_api_key(api_key: str) -> bool:
    """Test if the API key is valid by making a simple API call"""
    client = OpenAI(api_key=api_key)
    try:
        client.models.list()
    except AuthenticationError:
        return False
    else:
        return True

def reset_augmentation_state():
    if 'generated_response' in st.session_state:
        del st.session_state.generated_response
    if 'data_updated' in st.session_state:
        del st.session_state.data_updated
    if 'previous_test_results' in st.session_state:
        del st.session_state.previous_test_results

def main():
    init_session_state()
    render_sidebar()

    # Initialize api_key_input in session state if not present
    if 'openai_api_key' not in st.session_state:
        st.session_state.openai_api_key = None
    
    # Check for OpenAI API key
    api_key = os.getenv("OPENAI_API_KEY") or st.session_state.openai_api_key
    api_key_missing = api_key is None or api_key.strip() == ""
    
    # API Key Section
    if api_key_missing:
        with st.container():
            st.markdown("### 🔑 OpenAI API Key Required")
            st.markdown("""
            To use the data augmentation feature, you need to provide an OpenAI API key.
            The key will be stored only for this session.
            """)
            
            with st.expander("⚙️ API Key Configuration", expanded=True):
                api_key_input = st.text_input(
                    "Enter your OpenAI API key:",
                    type="password",
                    help="The key will only be stored temporarily for this session"
                )
                
                col1, col2 = st.columns([1, 3])
                with col1:
                    if st.button("✅ Validate Key", use_container_width=True):
                        with st.spinner("Validating API key..."):
                            if validate_api_key(api_key_input):
                                st.session_state.openai_api_key = api_key_input
                                with col2:
                                    st.success("API key validated successfully!")
                                    st.rerun()
                            else:
                                with col2:
                                    st.error("Invalid API key. Please check and try again.")
            st.divider()
    
    # Check if test results have changed
    if 'previous_test_results' not in st.session_state:
        st.session_state.previous_test_results = st.session_state.saved_test_results.copy()
    elif st.session_state.previous_test_results != st.session_state.saved_test_results:
        reset_augmentation_state()
        st.session_state.previous_test_results = st.session_state.saved_test_results.copy()
    
    if not st.session_state.saved_test_results:
        st.info("No saved test results found. Please run tests and save the results to do the augmentation.")
        st.stop()
    
    st.title("Final Augmentation")
    # Add Clear Results button in a column layout for better positioning
    col1, col2 = st.columns([1, 4])
    with col1:
        if st.button("Clear Saved Results", type="secondary"):
            st.session_state.saved_test_results = []
            st.rerun()
    
    # Pass it to generate_data function if the messages sent to OpenAI are to be displayed
    show_messages = True
    
    combined_prompt, ContainerModel = process_multiple_columns(st.session_state.saved_test_results)
    st.text_area("Prompt for LLM:", value=combined_prompt, height=300)
    selected_model = st.radio(
        "Select Model:",
        options=[
            ("gpt-4o-2024-08-06", "gpt-4o (Recommended for complex augmentation)"), 
            ("gpt-4o-mini", "gpt-4o-mini (Recommended for simple and faster augmentation)")
        ],
        format_func=lambda x: x[1],
        key="model_selection",
        help="Use gpt-4o for better results."
    )[0]

    if 'generated_response' not in st.session_state:
        st.session_state.generated_response = None

    if st.button("Generate Data", disabled=api_key_missing):
        # Use session state API key if environment variable is not set
        os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY") or st.session_state.openai_api_key
        response = generate_data(combined_prompt, openai_function, ContainerModel, selected_model)
        if not response.empty:
            st.session_state.generated_response = response
    
    # Show the generated data if it exists in session state
    if st.session_state.generated_response is not None:
        st.write("### Generated Data Samples")
        st.dataframe(st.session_state.generated_response)
        
        if hasattr(st.session_state, 'df') and st.session_state.df is not None:
            combined_df = pd.concat([st.session_state.df, st.session_state.generated_response], ignore_index=True)
            csv = combined_df.to_csv(index=False)
            st.download_button(
                label="Download Updated Dataset",
                data=csv,
                file_name="augmented_dataset.csv",
                mime="text/csv"
            )
            
            if st.button("Load Updated Dataset"):
                st.session_state.df = combined_df
                st.session_state.data_updated = True
                st.success(f"Dataset loaded successfully. Updated no.of columns {len(st.session_state.df)}")

if __name__ == "__main__":
    main()