import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import chisquare, chi2_contingency
from typing import List, Tuple

# Move all your helper functions here (load_dataset, get_categorical_columns, etc.)
# Copy all the functions from your main.py except the main() function

def load_dataset(file):
    return pd.read_csv(file)

def get_categorical_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """
    Return two lists of categorical columns from dataframe:
    1. Default selected columns: category/object type with >50% non-null values
    2. Optional columns: other categorical/object columns and numeric columns with low cardinality
    
    Returns:
        tuple: (default_selected_cols, optional_cols)
    """
    default_selected_cols = []
    optional_cols = []
    
    for col in df.columns:
        # Calculate metrics
        non_null_ratio = df[col].notna().mean()
        n_unique = df[col].nunique()
        n_total = len(df)
        unique_ratio = n_unique / n_total if n_total > 0 else 0
        
        # Check if column is categorical/object type
        is_cat_or_obj = (df[col].dtype == 'object' or 
                        df[col].dtype.name == 'category')
        
        # Default selected columns: categorical/object with >50% non-null values
        if is_cat_or_obj and non_null_ratio > 0.5:
            default_selected_cols.append(col)
            
        # Optional columns: other categorical/object OR numeric with low cardinality
        elif (is_cat_or_obj or 
              (df[col].dtype in ['int64', 'float64'] and unique_ratio < 0.25)):
            optional_cols.append(col)
            
    return default_selected_cols, optional_cols

def get_column_info(df: pd.DataFrame, col_name: str) -> str:
    """Generate information string about a column's properties."""
    non_null_ratio = df[col_name].notna().mean() * 100
    n_unique = df[col_name].nunique()
    n_total = len(df)
    unique_ratio = (n_unique / n_total * 100) if n_total > 0 else 0
    
    return (f"{col_name} ({df[col_name].dtype}): "
            f"{n_unique} unique values ({unique_ratio:.1f}% of total), "
            f"{non_null_ratio:.1f}% non-null")

def is_categorical(series: pd.Series) -> bool:
    """Check if a series should be treated as categorical."""
    n_unique = series.nunique()
    n_total = len(series)
    unique_ratio = n_unique / n_total if n_total > 0 else 0
    
    return (series.dtype == 'object' or 
            series.dtype.name == 'category' or
            (series.dtype in ['int64', 'float64'] and unique_ratio < 0.25))

def detect_bias(data: pd.Series, categories: List[str], distribution: str = 'equal',
                proportions: dict = None) -> Tuple[float, float, pd.Series, np.ndarray, np.ndarray]:
    """
    Detect bias in categorical data using scipy's chisquare test.
    Tests the null hypothesis that the categorical data has the given frequencies.
    
    Args:
        data (pd.Series): Input data series
        categories (List[str]): List of expected categories
        distribution (str): Type of expected distribution ('equal' or 'proportional')
        proportions (dict): Dictionary of category proportions if distribution is 'proportional'
    
    Returns:
        Tuple containing:
        - chi2_stat: Chi-square test statistic
        - p_value: P-value from the test
        - residuals: Standardized residuals for each category
        - observed: Observed frequencies
        - expected: Expected frequencies
    """
    # Input validation
    if not is_categorical(data):
        raise ValueError("Selected column is not categorical (too many unique values)")
    
    # Convert data to strings for consistent handling
    data = data.fillna('Missing').astype(str)
    categories = [str(cat) for cat in categories]
    
    # Calculate observed frequencies
    observed = data.value_counts().reindex(categories, fill_value=0).values
    total = len(data)
    
    # Calculate expected frequencies based on distribution type
    if distribution == 'equal':
        expected = np.array([total / len(categories)] * len(categories))
    else:
        if not proportions:
            raise ValueError("Proportions must be provided for proportional distribution")
        
        # Validate proportions
        total_proportion = sum(proportions.values())
        if not np.isclose(total_proportion, 100, rtol=1e-5):
            raise ValueError(f"Proportions must sum to 100% (current sum: {total_proportion:.2f}%)")
        
        expected = np.array([proportions[cat] * total / 100 for cat in categories])
    
    # Perform chi-square test using scipy.stats.chisquare
    chi2_stat, p_value = chisquare(observed, expected)
    
    # Calculate residuals
    residuals = (observed - expected) / np.sqrt(expected)
    
    return chi2_stat, p_value, pd.Series(residuals, index=categories), observed, expected

def test_independence(data: pd.DataFrame, col1: str, col2: str) -> Tuple[float, float, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    contingency_table = pd.crosstab(data[col1], data[col2])
    chi2, p_value, _, expected = chi2_contingency(contingency_table)
    expected_df = pd.DataFrame(expected, index=contingency_table.index, columns=contingency_table.columns)
    residuals = (contingency_table - expected_df) / np.sqrt(expected_df)
    return chi2, p_value, residuals, contingency_table, expected_df


# Modify the display_goodness_of_fit_report function to include augmentation UI
def display_goodness_of_fit_report(column: str, categories: List[str], chi2: float, p_value: float, 
                                 residuals: pd.Series, observed: np.ndarray, expected: np.ndarray, 
                                 alpha: float, distribution_type: str):
    st.write("---")
    st.write(f"### Detailed Bias Analysis Report for '{column}'")
    
    st.write("#### Test Information")
    st.write(f"- Test Type: Chi-square Goodness-of-Fit Test")
    st.write(f"- Distribution Type: {distribution_type}")
    st.write(f"- Significance Level (α): {alpha/100:.3f}")
    
    st.write("#### Test Statistics")
    st.write(f"- Chi-square statistic: {chi2:.4f}")
    st.write(f"- p-value: {p_value:.4f}")
    st.write(f"- Degrees of Freedom: {len(categories) - 1}")
    
    comparison_df = pd.DataFrame({
        'Category': categories,
        'Observed': observed,
        'Expected': expected.round(2)
    })
    # Add proportion column to comparison DataFrame
    comparison_df['Observed Proportion (%)'] = (comparison_df['Observed'] / comparison_df['Observed'].sum() * 100).round(2)
    comparison_df['Expected Proportion (%)'] = (comparison_df['Expected'] / comparison_df['Expected'].sum() * 100).round(2)
    comparison_df['Residual'] = residuals.values.round(2)

    st.write("#### Data Comparison")
    st.dataframe(comparison_df)
    
    st.write("#### Interpretation")
    if p_value < alpha/100:
        st.write(f"🚨 **Significant bias detected** (p-value {p_value:.4f} < {alpha/100:.3f})")
        st.write("Significant deviations from expected frequencies:")
        for category, residual in residuals.items():
            if residual > st.session_state.positive_threshold:
                st.write(f"- {category} is significantly over-represented (Residual: {residual:.2f})")
            elif residual < st.session_state.negative_threshold:
                st.write(f"- {category} is significantly under-represented (Residual: {residual:.2f})")
    else:
        st.write(f"✓ **No significant bias detected** (p-value {p_value:.4f} > {alpha/100:.3f})")

def display_independence_test_report(col1: str, col2: str, chi2: float, p_value: float, 
                                  residuals: pd.DataFrame, observed: pd.DataFrame, 
                                  expected: pd.DataFrame, alpha: float):
    st.write("---")
    st.write(f"### Detailed Independence Test Report for '{col1}' and '{col2}'")
    
    st.write("#### Test Information")
    st.write("- Test Type: Chi-square Test of Independence")
    st.write(f"- Significance Level (α): {alpha/100:.3f}")
    
    st.write("#### Test Statistics")
    st.write(f"- Chi-square statistic: {chi2:.4f}")
    st.write(f"- p-value: {p_value:.4f}")
    st.write(f"- Degrees of Freedom: {(observed.shape[0]-1) * (observed.shape[1]-1)}")
    
    st.write("#### Observed Frequencies")
    st.dataframe(observed)
    st.write("#### Expected Frequencies")
    st.dataframe(expected.round(2))
    st.write("#### Standardized Residuals")
    st.dataframe(residuals.round(2))
    
    st.write("#### Interpretation")
    if p_value < alpha/100:
        st.write(f"🚨 **Significant relationship detected** (p-value {p_value:.4f} < {alpha/100:.3f})")
        st.write("Significant associations:")
        for index in residuals.index:
            for column in residuals.columns:
                value = residuals.loc[index, column]
                if abs(value) > 2:
                    direction = "positive" if value > 0 else "negative"
                    st.write(f"- {index} and {column}: {direction} association (Residual: {value:.2f})")
    else:
        st.write(f"✓ **No significant relationship detected** (p-value {p_value:.4f} > {alpha/100:.3f})")

def clear_session_state():
    """Helper function to clear session state variables"""
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    init_session_state()

def init_session_state():
    """Initialize session state variables"""
    if 'df' not in st.session_state:
        st.session_state.df = None
    if 'default_categorical_cols' not in st.session_state:
        st.session_state.default_categorical_cols = []
    if 'optional_categorical_cols' not in st.session_state:
        st.session_state.optional_categorical_cols = []
    if 'categorical_cols' not in st.session_state:
        st.session_state.categorical_cols = []
    if 'previous_file' not in st.session_state:
        st.session_state.previous_file = None
    if 'saved_test_results' not in st.session_state:
        st.session_state.saved_test_results = []
    if 'current_test_results' not in st.session_state:
        st.session_state.current_test_results = None
    if 'positive_threshold' not in st.session_state:
        st.session_state.positive_threshold = 1.3
    if 'negative_threshold' not in st.session_state:
        st.session_state.negative_threshold = -1.3
    if 'alpha' not in st.session_state:
        st.session_state.alpha = 5.0
    if 'latest_saved_test' not in st.session_state:
        st.session_state.latest_saved_test = None
    if 'generated_response' not in st.session_state:
        st.session_state.generated_response = None