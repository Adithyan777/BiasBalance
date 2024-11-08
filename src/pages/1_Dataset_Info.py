import streamlit as st
from components import render_sidebar
from utils import init_session_state
import pandas as pd

def main():
    init_session_state()
    render_sidebar()
    
    if st.session_state.df is None:
        st.warning("No dataset loaded. Please upload a CSV file on the Home page.")
        st.stop()
    
    st.title("Dataset Information")
    df = st.session_state.df
    
    # Basic Dataset Info
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Rows", df.shape[0])
    with col2:
        st.metric("Total Columns", df.shape[1])
    # with col3:
    #     # Update to use the current length of categorical_cols
    #     st.metric("Categorical Columns", len(st.session_state.categorical_cols))
    
    # Column Selection Section
    st.header("Categorical Column Selection")
    all_categorical_options = (
        st.session_state.default_categorical_cols + 
        st.session_state.optional_categorical_cols
    )
    
    # Update categorical columns based on multiselect
    selected_cols = st.multiselect(
        "Select categorical columns for analysis:",
        options=all_categorical_options,
        default=st.session_state.categorical_cols,
        key="categorical_cols_select"
    )
    
    # Important: Update session state with new selection
    st.session_state.categorical_cols = selected_cols
    
    # Show selection criteria
    with st.expander("Column Selection Criteria", expanded=False):
        st.markdown("""
        **Default Selected Columns (marked with ✓):**
        - Categorical/Object type columns with >50% non-null values
        
        **Additional Available Columns:**
        - Other Categorical/Object type columns
        - Numeric columns with unique values < 25% of total rows
        """)
    
    # Data Preview
    st.header("Data Preview")
    st.dataframe(df.head())
    
    # Column Information
    st.header("Column Information")
    col_info = pd.DataFrame({
        'Data Type': df.dtypes,
        'Unique Values': df.nunique(),
        'Non-null Values (%)': (df.notna().mean() * 100).round(2),
        'Is Categorical': df.columns.isin(st.session_state.categorical_cols),
        'Is Default Selection': df.columns.isin(st.session_state.default_categorical_cols)
    })
    st.dataframe(col_info)

if __name__ == "__main__":
    main()
