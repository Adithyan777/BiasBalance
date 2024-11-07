from typing import Dict, List, Tuple, Any, Optional
from pydantic import BaseModel,create_model,model_validator
from enum import Enum
from collections import Counter
import pandas as pd
import streamlit as st
import numpy as np
from openai import OpenAI
import os
from functools import wraps

system_message = """Create synthetic data to oversample under-represented columns or categories in a dataset based on given parameters.

For each specified column:
- You will receive the categories that need oversampling and the number of specific samples required.
- If multiple columns are being oversampled together, create a combined synthetic dataset that satisfies all given conditions with minimal sample count. Do not create separate datasets for each column; instead, generate data such that new rows meet the requirements of all columns collectively.

# Steps

1. **Input Analysis**: Start by examining which columns need oversampling, the categories for those columns, and the number of additional samples required.
2. **Combination Processing**: If multiple columns need oversampling, prioritize generating samples that can fulfill the needs for multiple columns simultaneously.
3. **Synthetic Data Generation**:
   - Generate samples based on the requests for each category.
   - Ensure that where multiple columns intersect, the generated samples provide overlap such that the synthetic dataset size is minimized.
4. **Quality Check**: Validate that all specified conditions are met for each column and for each category, with careful attention given to minimizing redundant samples.

# Notes

- This prompt assumes that the user will provide clear information about the number of additional samples needed per category.
- Ensure that the synthetic data respects any implicit relationships or dependencies between columns, where possible.
- The goal is to achieve the minimum number of additional rows that fulfill all oversampling requirements across the columns.
"""

messages = [
    {"role": "system", "content": system_message},
]

previous_model_response = None

# Add the new augmentation functions from previous artifact
def calculate_augmentation_needs(
    residuals: pd.Series,
    observed: np.ndarray,
    expected: np.ndarray,
    categories: List[str],
    negative_threshold: float = -1.5 # TODO: set this accordingly
) -> Dict[str, int]:
    """
    Calculate how many samples need to be generated for each underrepresented category.
    """
    augmentation_needs = {}
    
    for cat, residual, obs, exp in zip(categories, residuals, observed, expected):
        if residual < negative_threshold:
            samples_needed = int(np.ceil((exp - obs) * 1.1))
            augmentation_needs[cat] = samples_needed
            
    return augmentation_needs

def generate_prompt(
    column_name: str,
    category: str,
    n_samples: int,
    df: pd.DataFrame,
    context_columns: List[str] = None
) -> str:
    """
    Generate a prompt for the LLM to create synthetic data.
    """
    category_samples = df[df[column_name].astype(str) == str(category)]
    
    if context_columns:
        sample_data = category_samples[context_columns].head(3).to_dict('records')
    else:
        sample_data = category_samples.head(3).to_dict('records')
    
    prompt = f"""Generate {n_samples} synthetic data samples for the category '{category}' in column '{column_name}'.

Context:
- This is for data augmentation to balance an underrepresented category
- The data should be similar to but not exactly like the examples
- Maintain the same general patterns and relationships between fields

Example data points from this category:
{sample_data}

Please generate {n_samples} new, unique data points following the same pattern but with reasonable variations.
Format the output as a list of dictionaries, with each dictionary representing one data point.
"""
    return prompt

def _calculate_augmentation_needs(
    residuals: pd.Series,
    observed: np.ndarray,
    expected: np.ndarray,
    categories: List[str],
    negative_threshold: float = -1.5 # TODO: set this accordingly
) -> Dict[str, int]:
    """
    Calculate how many samples need to be generated for each underrepresented category.
    """
    augmentation_needs = {}
    
    for cat, residual, obs, exp in zip(categories, residuals, observed, expected):
        if residual < negative_threshold:
            samples_needed = int(np.ceil((exp - obs) * 1.1))
            augmentation_needs[cat] = samples_needed
            
    return augmentation_needs

def combine_column_augmentation_needs(
    column_results: List[Tuple[str, Dict[str, int]]]
) -> Tuple[Dict[str, Dict[str, int]], int]:
    """
    Combine augmentation needs across multiple columns and calculate total samples needed.
    
    Args:
        column_results: List of tuples containing (column_name, augmentation_needs)
            where augmentation_needs is the output from _calculate_augmentation_needs()
    
    Returns:
        Tuple containing:
        - Dictionary mapping column names to their augmentation requirements
        - Total number of samples needed
    """
    combined_needs = {}
    total_samples = 0
    
    # First pass to get combined needs
    for column_name, needs in column_results:
        if needs:  # Only include columns that need augmentation
            combined_needs[column_name] = needs
            # Get the maximum number of samples needed for any category in this column
            total_samples = max(total_samples, sum(needs.values()))
            
    return combined_needs, total_samples

def format_augmentation_prompt(
    combined_needs: Dict[str, Dict[str, int]],
    dataset_info: Dict[str, Any],
    category_info: Dict[str, List[str]],
    total_samples: int
) -> str:
    """
    Format the augmentation needs into a clear prompt for an LLM.
    
    Args:
        combined_needs: Dictionary mapping column names to their augmentation requirements
        dataset_info: Dictionary containing information about the dataset
        category_info: Dictionary mapping column names to their unique category values
        total_samples: Total number of samples to generate
    
    Returns:
        Formatted prompt for the LLM
    """
    prompt_parts = [
        f"Please generate {total_samples} synthetic data samples to augment the following dataset:",
        f"\nDataset Information:",
        f"- Total Existing Rows: {dataset_info['num_rows']}",
        f"- Total Columns: {dataset_info['num_cols']}",
        f"- Dataset Tail (for reference):",
        f"{dataset_info['tail']}",
        f"\nGeneration Requirements:"
    ]
    
    # Add overall distribution requirements
    prompt_parts.append("\nEach generated sample must satisfy these distribution requirements:")
    
    for column, needs in combined_needs.items():
        prompt_parts.append(f"\nFor column '{column}':")
        specified_categories = list(needs.keys())
        
        # First, add all positive requirements
        for category, count in needs.items():
            prompt_parts.append(f"- {count} out of {total_samples} should be '{category}'")
        
        # Then add a single "except" statement if needed
        total_specified = sum(needs.values())
        if total_specified < total_samples:
            remaining_count = total_samples - total_specified
            categories_str = "', '".join(specified_categories)
            prompt_parts.append(f"- {remaining_count} samples should be anything except '{categories_str}'")
        
        prompt_parts.append(f"\nValid categories for '{column}':")
        prompt_parts.extend([f"- {cat}" for cat in category_info[column]])
    
    # Add clear instructions
    prompt_parts.extend([
        "\nImportant Instructions:",
        f"1. Generate exactly {total_samples} complete samples",
        "2. Each sample should contain values for ALL columns in the dataset",
        "3. Values must be coherent across columns (maintain realistic relationships)",
        "4. Generated samples should follow the specified distribution requirements",
        "5. Use the dataset tail as a reference for the format and relationships between columns",
        "\nReturn the output as a list of dictionaries in valid JSON format, where each dictionary represents one complete data point."
    ])
    
    return "\n".join(prompt_parts)

def process_multiple_columns(
    test_results: List[Tuple[str, pd.Series, np.ndarray, np.ndarray, List[str]]]
) -> Tuple[str, type[BaseModel]]:
    """
    Process test results from multiple columns and generate an LLM prompt.
    
    Args:
        test_results: List of tuples containing (column_name, residuals, observed, expected, categories)
            for each tested column
    
    Returns:
        prompt : Formatted prompt for the LLM
        RowModel : Pydantic RowModel
        ContainerModel : Pydantic ContainerModel
    """
    column_results = []
    dataset_info = {
        'num_rows': len(st.session_state.df),
        'num_cols': len(st.session_state.df.columns),
        'tail': st.session_state.df.tail().to_string(index=False)
    }
    category_info = {
        col: sorted(st.session_state.df[col].fillna('Missing').astype(str).unique().tolist())
        for col in st.session_state.categorical_cols
    }
    
    for column_name, residuals, observed, expected, categories in test_results:
        needs = _calculate_augmentation_needs(residuals, observed, expected, categories, st.session_state.negative_threshold)
        if needs:  # Only include columns that need augmentation
            column_results.append((column_name, needs))
    
    combined_needs, total_samples = combine_column_augmentation_needs(column_results)
    RowModel, ContainerModel = create_pydantic_model_from_df(st.session_state.df, category_info,combined_needs)
    prompt = format_augmentation_prompt(combined_needs, dataset_info, category_info, total_samples)
    return prompt, ContainerModel

def create_pydantic_model_from_df(
    df: pd.DataFrame, 
    category_info: Dict[str, List[str]], 
    combined_needs: Dict[str, Dict[str, int]],
    model_name: str = "DataModel"
) -> tuple[type[BaseModel], type[BaseModel]]:
    """
    Creates a Pydantic model from a DataFrame and category information with category count validation.
    
    Args:
        df: DataFrame containing the data
        category_info: Dictionary mapping column names to their possible values
        combined_needs: Dictionary specifying required counts for each category in columns
        model_name: Name for the generated model
    
    Returns:
        tuple of (Row model, Container model)
    """
    # Helper function to determine field type
    def get_field_type(column: str) -> tuple[type, Any]:
        # If column is in category_info, create an Enum
        if column.lower() in {k.lower(): k for k in category_info.keys()}:
            original_case = {k.lower(): k for k in category_info.keys()}[column.lower()]
            enum_name = f"{column.title().replace('_', '')}Enum"
            enum_values = category_info[original_case]
            enum_dict = {val: val for val in enum_values}
            enum_type = Enum(enum_name, enum_dict)
            return (enum_type, ...)
        
        # For other columns, infer type from DataFrame
        dtype = str(df[column].dtype)
        if 'int' in dtype:
            return (int, ...)
        elif 'float' in dtype:
            return (float, ...)
        elif 'bool' in dtype:
            return (bool, ...)
        else:
            return (str, ...)

    # Create field definitions for the row model
    field_definitions = {
        col: get_field_type(col) for col in df.columns
    }

    # Create the row model
    RowModel = create_model(
        f"{model_name}Row",
        **field_definitions
    )

    # Create the container model with validation
    class ContainerModelWithValidation(BaseModel):
        rows: List[RowModel]

        @model_validator(mode='after')
        def validate_category_counts(self) -> 'ContainerModelWithValidation':
            global previous_model_response 
            previous_model_response = self.model_dump_json()
            for column, needs in combined_needs.items():
                # Get all values for this column across all rows
                column_values = [getattr(row, column) for row in self.rows]
                
                # Convert enum values to strings if necessary
                column_values = [val.value if isinstance(val, Enum) else val for val in column_values]
                
                # Count occurrences of each category
                value_counts = Counter(column_values)
                
                # Check if each category meets its required count
                for category, required_count in needs.items():
                    actual_count = value_counts.get(category, 0)
                    if actual_count < required_count:
                        raise ValueError(
                            f"Column '{column}' category '{category}' requires {required_count} "
                            f"instances but only has {actual_count}"
                        )
            
            return self

    ContainerModel = ContainerModelWithValidation

    return RowModel, ContainerModel

def container_to_dict(container_model: Any) -> Dict[str, List[Dict[str, Any]]]:
    """
    Converts a ContainerModel instance to a Python dictionary.
    
    Args:
        container_model: Instance of the ContainerModel created by create_pydantic_model_from_df
    
    Returns:
        Dictionary with format: {'rows': [{'column1': value1, 'column2': value2, ...}, ...]}
    """
    def convert_value(value: Any) -> Any:
        """Helper function to convert enum values back to their string representation"""
        if isinstance(value, Enum):
            return value.value
        return value

    return {
        'rows': [
            {
                field_name: convert_value(getattr(row, field_name))
                for field_name in row.model_fields.keys()
            }
            for row in container_model.rows
        ]
    }

def generate_data(prompt: str, llm_function, PydanticModel: type[BaseModel], model_name:str,show_messages: bool = False) -> pd.DataFrame:
    """
    Generate synthetic data using the specified LLM function.
    
    Args:
        prompt: Prompt for the LLM to generate synthetic data
        llm_function: Function that takes a prompt and returns synthetic data
        PydanticModel: Pydantic model for the synthetic data
    
    Returns:
        DataFrame containing the synthetic data
    """
    global messages
    messages.append({"role": "user", "content": prompt})
    try:
        result = llm_function(messages, PydanticModel,model_name)
        if result:
            st.success("Successfully generated and validated synthetic data!")
            messages = messages[:1]
            return pd.DataFrame(container_to_dict(result).get('rows'))
    except ValueError as ve:
        if model_name == 'gpt-4o-mini':
            st.error(f"Failed to generate valid data with the current model. Retry with a more powerful model.")
            messages = messages[:1]
            return pd.DataFrame()
            # return generate_data(prompt, llm_function, PydanticModel, "gpt-4o")
        else:
            st.error(f"Failed to generate valid data after multiple attempts. Last error: {str(ve)}")
            messages = messages[:1]
            return pd.DataFrame()
    except Exception as e:
        st.error(f"Unexpected error: {str(e)}")
        return pd.DataFrame()
    
def create_error_prompt(error_string: str) -> List[Dict[str,str]]:
    """
    Create a prompt that asks the LLM to fix its previous response based on the error message.
    
    Args:
        error_string: Error message from the previous response
        new_chat: Whether to start a new chat or continue the existing one
    
    Returns:
        A new chat message asking for a fixed response
    """
    global messages
    messages = messages[:2]
    messages.append({"role" : "assistant", "content" : previous_model_response})
    messages.append({"role": "user", "content": f"Your previous resposne was not valid. Please use the error message to fix it.\n{error_string}."})
    return messages


def retry_with_prompt(max_retries: int = 3):
    """
    Decorator to handle retries with improved prompts.
    
    Args:
        max_retries: Maximum number of retry attempts
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            retry_count = 0
            last_error = None
            response_str = None
            
            while retry_count < max_retries:
                try:
                    return func(*args, **kwargs)
                except ValueError as ve:
                    retry_count += 1
                    last_error = ve
                    
                    if retry_count < max_retries:
                        st.warning(f"Attempt {retry_count} failed. Retrying...")
                        # st.write("Retrying with improved prompt:")
                        # st.write(retry_prompt)
                        retry_prompt = create_error_prompt(str(ve))
                        # Update the prompt in args for the next try
                        args = (retry_prompt,) + args[1:]
                    else:
                        st.error(f"Failed after {max_retries} attempts. Last error: {str(ve)}")
                        raise ValueError(f"Failed to generate valid data after {max_retries} attempts")
                except Exception as e:
                    st.error(f"Unexpected error: {str(e)}")
                    raise
            
            return None
        return wrapper
    return decorator

@retry_with_prompt(max_retries=3)
def openai_function(messages: List[Dict[str,str]], PydanticModel: type[BaseModel], model_name:str = "gpt-4o-mini",show_messages: bool = False) -> type[BaseModel]:
    """
    Call the OpenAI API to generate synthetic data.
    
    Args:
        prompt: Prompt for the LLM to generate synthetic data
        PydanticModel: Pydantic model for the synthetic data
    
    Returns:
        List of dictionaries containing the synthetic data
    """
    api_key = os.getenv("OPENAI_API_KEY")
    client = OpenAI(api_key=api_key)
    
    # print("Using model: ", model_name)
    st.write("Using model: ", model_name)
    if show_messages:
        st.write(messages)
    response = client.beta.chat.completions.parse(
        model=model_name,
        messages=messages,
        response_format=PydanticModel
    )
    return response.choices[0].message.parsed
    
