import streamlit as st
from components import render_sidebar
from utils import init_session_state, is_categorical, test_independence, display_independence_test_report

def main():
    init_session_state()
    render_sidebar()

    if st.session_state.df is None:
        st.warning("No dataset loaded. Please upload a CSV file on the Home page.")
        st.stop()
    
    df = st.session_state.df
    st.header("Independence Testing")
        
    # Use the updated categorical_cols from session state
    col1 = st.selectbox(
        "Select first column:",
        [""] + st.session_state.categorical_cols,
        index=0,
        key="col1"
    )
    
    col2 = st.selectbox(
        "Select second column:",
        [""] + [col for col in st.session_state.categorical_cols if col != col1],
        index=0,
        key="col2"
    )
    
    if col1 and col2:
        if not is_categorical(df[col1]):
            st.error(f"Error: Column '{col1}' is not categorical (too many unique values)")
            return
        if not is_categorical(df[col2]):
            st.error(f"Error: Column '{col2}' is not categorical (too many unique values)")
            return
        
        # Add Apply Test button for Independence Test
        if st.button("Apply Test", key="independence_test"):
            try:
                # Convert both columns to string type for consistency
                df_copy = df.copy()
                df_copy[col1] = df_copy[col1].fillna('Missing').astype(str)
                df_copy[col2] = df_copy[col2].fillna('Missing').astype(str)
                
                chi2, p_value, residuals, observed, expected = test_independence(df_copy, col1, col2)
                display_independence_test_report(
                    col1, col2, chi2, p_value, residuals,
                    observed, expected, st.session_state.alpha
                )
            except Exception as e:
                st.error(f"Error in analysis: {str(e)}")


if __name__ == "__main__":
    main()