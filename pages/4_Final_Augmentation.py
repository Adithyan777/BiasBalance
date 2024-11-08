import streamlit as st
from components import render_sidebar
from utils import init_session_state
from data_augmentation import process_multiple_columns, openai_function, generate_data
import pandas as pd
import os
from openai import OpenAI,AuthenticationError
import numpy as np

def are_all_test_results_identical(saved_results, previous_results):
    """Compare two lists of test results with more flexible comparison"""
    if saved_results is None or previous_results is None:
        return False
        
    if len(saved_results) != len(previous_results):
        return False
        
    for saved, prev in zip(saved_results, previous_results):
        # Compare only essential elements
        if (saved[0] != prev[0] or  # column name
            not np.array_equal(np.array(saved[2]), np.array(prev[2])) or  # observed
            not np.array_equal(np.array(saved[3]), np.array(prev[3])) or  # expected
            saved[4] != prev[4]):  # categories
            return False
    return True

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
        st.session_state.generated_response = None
    if 'data_updated' in st.session_state:
        st.session_state.data_updated = None
    if 'previous_test_results' in st.session_state:
        st.session_state.previous_test_results = None
    if 'combined_df' in st.session_state:
        st.session_state.combined_df = None
    if 'last_generated_size' in st.session_state:
        st.session_state.last_generated_size = None

def load_updated_dataset():
    """
    Safely loads and updates the dataset while handling state management.
    Returns True if successful, False otherwise.
    """
    try:
        if not hasattr(st.session_state, 'df') or st.session_state.df is None:
            return False
            
        if not hasattr(st.session_state, 'generated_response') or st.session_state.generated_response is None:
            return False
            
        # Store current state
        current_generated_size = len(st.session_state.generated_response)
        current_data = st.session_state.generated_response.copy()
        
        # Verify data integrity
        if current_data.empty:
            return False
            
        # Create combined dataframe
        combined_df = pd.concat(
            [st.session_state.df, current_data], 
            ignore_index=True
        )
        
        # Update all relevant session states atomically
        st.session_state.combined_df = combined_df
        st.session_state.last_generated_size = current_generated_size
        st.session_state.data_updated = True
        st.session_state.df = combined_df.copy()
        
        return True
        
    except Exception as e:
        st.error(f"Error loading dataset: {str(e)}")
        return False

def init_augmentation_state():
    """Initialize or reset augmentation-specific state"""
    if 'previous_test_results' not in st.session_state:
        st.session_state.previous_test_results = None
    if 'generated_response' not in st.session_state:
        st.session_state.generated_response = None
    if 'data_updated' not in st.session_state:
        st.session_state.data_updated = None
    if 'last_generated_size' not in st.session_state:
        st.session_state.last_generated_size = None
    if 'combined_df' not in st.session_state:
        st.session_state.combined_df = None

def main():
    init_session_state()
    init_augmentation_state()
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
    current_test_results = st.session_state.saved_test_results
    previous_results = st.session_state.previous_test_results
    
    if not are_all_test_results_identical(current_test_results, previous_results):
        reset_augmentation_state()
        st.session_state.previous_test_results = current_test_results.copy() if current_test_results else None
    
    if not st.session_state.saved_test_results:
        st.info("No saved test results found. Please run tests and save the results to do the augmentation.")
        st.stop()
    
    st.title("Final Augmentation")
    # Add Clear Results button in a column layout for better positioning
    col1, col2 = st.columns([1, 4])
    with col1:
        if st.button("Clear Saved Results", type="secondary"):
            st.session_state.saved_test_results = []
            reset_augmentation_state()
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
        st.session_state.generated_response = None
        response = generate_data(combined_prompt, openai_function, ContainerModel, selected_model)
        if not response.empty:
            st.session_state.generated_response = response
    
    # Show the generated data if it exists in session state
    if st.session_state.generated_response is not None:
        st.write("### Generated Data Samples")
        st.dataframe(st.session_state.generated_response)
        
        if hasattr(st.session_state, 'df') and st.session_state.df is not None:
            current_generated_size = len(st.session_state.generated_response)
            
            is_already_loaded = (
                'last_generated_size' in st.session_state and 
                st.session_state.last_generated_size == current_generated_size and
                'data_updated' in st.session_state and 
                st.session_state.data_updated
            )
            
            if st.button("Load Updated Dataset", disabled=is_already_loaded):
                with st.spinner("Loading dataset..."):
                    if load_updated_dataset():
                        st.rerun()
                    else:
                        st.error("Failed to load dataset. Please try again.")
            
            if is_already_loaded:
                st.success(f"Dataset loaded successfully. Updated no.of rows: {len(st.session_state.df)}")
                st.info("You can download the updated dataset from the Home page or run more tests to augment further.")


if __name__ == "__main__":
    main()