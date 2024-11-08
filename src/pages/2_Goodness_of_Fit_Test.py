import streamlit as st
import numpy as np
from components import render_sidebar
from utils import init_session_state, is_categorical, detect_bias, display_goodness_of_fit_report
from data_augmentation import calculate_augmentation_needs

def return_saving_attributes(test_results):
    return (test_results['column'], test_results['residuals'], test_results['observed'], test_results['expected'], test_results['categories'])

def compare_saving_attributes(tuple1, tuple2):
    # Unpack the elements from each tuple for clarity
    column1, residuals1, observed1, expected1, categories1 = tuple1
    column2, residuals2, observed2, expected2, categories2 = tuple2
    
    # Compare each element in the tuples
    same_column = column1 == column2
    same_residuals = np.array_equal(residuals1, residuals2)
    same_observed = np.array_equal(observed1, observed2)
    same_expected = np.array_equal(expected1, expected2)
    same_categories = categories1 == categories2  # Direct comparison for list of strings
    
    # Return True only if all elements match
    return same_column and same_residuals and same_observed and same_expected and same_categories

    
def main():
    init_session_state()
    render_sidebar()
    
    if st.session_state.df is None:
        st.warning("No dataset loaded. Please upload a CSV file on the Home page.")
        st.stop()
    
    st.title("Bias Detection (Goodness-of-Fit Test)")
        
    # Use the updated categorical_cols from session state
    column = st.selectbox(
        "Select column for bias checking:",
        [""] + st.session_state.categorical_cols,
        index=0
    )

    df = st.session_state.df
    
    if column:
        if not is_categorical(df[column]):
            st.error(f"Error: Column '{column}' is not categorical (too many unique values)")
            return
            
        categories = sorted(df[column].fillna('Missing').astype(str).unique().tolist())
        
        distribution_type = st.radio(
            "Select expected distribution type:",
            ['equal', 'proportional'],
            help="Equal: assumes equal distribution across categories \t Proportional: uses custom proportions"
        )
        
        proportions = {}
        valid_proportions = True
        
        if distribution_type == 'proportional':
            # Calculate observed proportions
            value_counts = df[column].value_counts(normalize=True) * 100
            observed_props = {str(cat): value_counts.get(cat, 0) for cat in categories}
            
            # Add button to set expected to observed
            if st.button("Set to Observed Proportions"):
                # Update session state with observed proportions
                for category in categories:
                    st.session_state[f"{column}_{category}"] = round(observed_props[str(category)], 1)
            
            st.write(f"\nSet expected percentages for '{column}' categories:")
            total_percentage = 0
            cols = st.columns(4)
            
            # Initialize proportions dictionary with rounded default values
            default_value = round(100.0/len(categories), 1)
            remaining_percentage = 100.0 - (default_value * (len(categories) - 1))
            
            for i, category in enumerate(categories):
                with cols[i % 4]:
                    # Set the last category to make sum exactly 100
                    if i == len(categories) - 1:
                        value = round(remaining_percentage, 1)
                    else:
                        value = default_value
                        
                    prop = st.number_input(
                        f"Expected % for {category}",
                        min_value=0.0,
                        max_value=100.0,
                        value=value,
                        step=0.1,
                        key=f"{column}_{category}"
                    )
                    proportions[str(category)] = prop
                    total_percentage += prop
            
            # Round total percentage to handle floating point precision
            total_percentage = round(total_percentage, 1)
            st.write(f"Total percentage: {total_percentage:.1f}%")
            
            # Use a more lenient check for equality
            valid_proportions = abs(total_percentage - 100.0) <= 0.1
            
            if not valid_proportions:
                st.warning(f"Total percentage must equal 100% (current: {total_percentage:.1f}%)")
        
        # Add Apply Test button
        if st.button("Apply Test", disabled=distribution_type == 'proportional' and not valid_proportions):
            try:
                # Normalize proportions to ensure they sum to exactly 1.0 if they're close enough
                if distribution_type == 'proportional' and valid_proportions:
                    total = sum(proportions.values())
                    proportions = {k: v/total * 100 for k, v in proportions.items()}
                
                chi2, p_value, residuals, observed, expected = detect_bias(
                    df[column],
                    categories,
                    distribution_type,
                    proportions if distribution_type == 'proportional' else None
                )

                    # Store current test results in session state
                st.session_state.current_test_results = {
                    'column': column,
                    'chi2': chi2,
                    'p_value': p_value,
                    'residuals': residuals,
                    'observed': observed,
                    'expected': expected,
                    'categories': categories,
                    'distribution_type': distribution_type
                }
            except Exception as e:
                st.error(f"Error in analysis: {str(e)}")
                st.session_state.current_test_results = None

            # Display results if they exist in session state
        if st.session_state.current_test_results and st.session_state.current_test_results['column'] == column:
            results = st.session_state.current_test_results
            
            display_goodness_of_fit_report(
                results['column'],
                results['categories'],
                results['chi2'],
                results['p_value'],
                results['residuals'],
                results['observed'],
                results['expected'],
                st.session_state.alpha,
                results['distribution_type']
            )

            if results['p_value'] < st.session_state.alpha/100:
                st.write("### Data Augmentation")
                
                # Calculate augmentation needs
                augmentation_needs = calculate_augmentation_needs(
                    results['residuals'],
                    results['observed'],
                    results['expected'],
                    results['categories'],
                    st.session_state.negative_threshold,
                )
                
                if not augmentation_needs:
                    st.write("✓ No categories need augmentation (all residuals > -1.5)")
                else:
                    st.write("The following categories are underrepresented and need augmentation:")
                    for category, n_samples in augmentation_needs.items():
                        st.write(f"- {category}: need {n_samples} additional samples")
                    
                    # Check if this test is already saved for this column
                    existing_saved = next((
                        (i, saved) for i, saved in enumerate(st.session_state.saved_test_results) 
                        if saved[0] == results['column']
                    ), None)

                    # st.write(compare_saving_attributes(existing_saved[1], return_saving_attributes(results)))

                    if existing_saved and not compare_saving_attributes(existing_saved[1], return_saving_attributes(results)):
                            st.write('---')
                            # Show confirmation dialog using columns
                            st.warning(f"A test for column '{results['column']}' already exists.")
                            st.write("### Existing Saved Test Result")
                            saved_data = existing_saved[1]
                            augmentation_needs = calculate_augmentation_needs(
                                saved_data[1],  # residuals 
                                saved_data[2],  # observed
                                saved_data[3],  # expected
                                saved_data[4],  # categories
                                st.session_state.negative_threshold,
                            )
                            
                            if not augmentation_needs:
                                st.write("✓ No categories need augmentation (all residuals > threshold)")
                            else:
                                st.write("Categories needing augmentation:")
                                for category, n_samples in augmentation_needs.items():
                                    st.write(f"- {category}: need {n_samples} additional samples")
                            
                    if existing_saved:
                        if compare_saving_attributes(existing_saved[1], return_saving_attributes(results)):
                            st.success("Current test results already saved.")
                    
                    button_disabled = False
                    button_text = "Add for Final Augmentation"
                    if existing_saved:
                        if not compare_saving_attributes(existing_saved[1], return_saving_attributes(results)):
                            button_text = "Remove Existing And Save New"
                        else:
                            button_disabled = True        
                    
                    if st.button(button_text,disabled=button_disabled):
                        if existing_saved:
                            # Get the index from the existing_saved tuple
                            index_to_remove = existing_saved[0]
                            
                            # Create new test tuple
                            new_test = return_saving_attributes(results)
                            
                            st.session_state.saved_test_results[index_to_remove] = new_test
                            st.session_state.latest_saved_test = new_test
                            
                            st.toast(f"Removed existing test results and added new test results for {results['column']}! Total saved tests: {len(st.session_state.saved_test_results)}")
                            st.rerun()
                        else:
                            # Save new test results
                            test_result = return_saving_attributes(results)
                            st.session_state.latest_saved_test = test_result
                            st.session_state.saved_test_results.append(test_result)
                            st.success(f"Test results for {results['column']} saved! Total saved tests: {len(st.session_state.saved_test_results)}")

if __name__ == "__main__":
    main()
