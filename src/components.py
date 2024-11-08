import streamlit as st

def render_sidebar():
    """Permanent sidebar component for all pages"""
    # st.sidebar.header("Dataset Status")
    if st.session_state.df is not None:
        st.sidebar.success(f"Dataset loaded: {st.session_state.previous_file}")
    else:
        st.sidebar.warning("No dataset loaded")
    
    if st.session_state.df is not None:
        st.sidebar.write("---")
        st.sidebar.header("Analysis Settings")
        st.session_state.alpha = float(st.sidebar.number_input(
            "Significance Level (α) in %", 
            min_value=1, 
            max_value=30,
            value=5,
            help="Enter significance level as percentage (e.g., 5 for 5%)"
        ))
        
        st.session_state.positive_threshold = st.sidebar.slider(
            "Positive Residual Threshold",
            min_value=0.1,
            max_value=2.5,
            value=1.3,
            step=0.1,
            help="Threshold for identifying over-represented categories"
        )
        st.session_state.negative_threshold = st.sidebar.slider(
            "Negative Residual Threshold",
            min_value=-2.5,
            max_value=-0.1,
            value=-1.3,
            step=0.1,
            help="Threshold for identifying under-represented categories"
        )
