"""
UI components for displaying analysis results.
"""
import streamlit as st
import pandas as pd
from typing import Optional

from models.app_data import AnalysisResults, AppDetails

def display_developer_info(app_details: AppDetails) -> None:
    with st.expander("Developer Information & Data Safety Claims", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("App Details")
            st.markdown(f"**Publisher:** {app_details.publisher}")
            st.markdown(f"**Link:** [{app_details.link}]({app_details.link})")
            st.markdown("**Description:**")
            st.markdown(f"_{app_details.description}_")
            
        with col2:
            st.subheader("Data Safety Claims")
            st.markdown(f"**Shared Data:** {app_details.shared_data}")
            st.markdown(f"**Collected Data:** {app_details.collected_data}")
            st.markdown(f"**Security Practices:** {app_details.security_practices}")

def display_app_details_table(app_details: AppDetails) -> None:
    """Displays key app details in a table format with columns."""
    st.subheader("Selected App Details")
    
    # Create a dictionary with app details
    data = {
        "App Name": app_details.name,
        "Publisher": app_details.publisher,
        "Shared Data": app_details.shared_data,
        "Collected Data": app_details.collected_data,
        "Security Practices": app_details.security_practices
    }
    
    # Convert dictionary to DataFrame, transpose it, and reset index
    df = pd.DataFrame.from_dict(data, orient='index', columns=['Value'])
    df.index.name = 'Attribute'
    df = df.reset_index()
    
    # Display the transposed DataFrame
    st.dataframe(df, hide_index=True, use_container_width=True)

    st.markdown(f"**App Link:** [{app_details.link}]({app_details.link})")
    
    # Show full description
    with st.expander("Full Description", expanded=False):
        st.markdown(app_details.description)

def display_review_summary(analysis: str, filtered_count: int, raw_count: int) -> None:
    st.subheader("User Review Summary (Based on AI Analysis)")
    st.markdown(analysis)
    st.caption(f"Analyzed {filtered_count} reviews (out of {raw_count} fetched) with sufficient length.")

def display_difference_analysis(analysis: str) -> None:
    st.subheader("Comparison: User Feedback vs. Developer Claims")
    st.markdown(analysis)

def display_sample_reviews(reviews_sample: Optional[pd.DataFrame]) -> None:
    if reviews_sample is not None and not reviews_sample.empty:
        with st.expander("Sample Filtered Reviews Used for Analysis", expanded=False):
            # Select columns to display
            display_df = reviews_sample[['score', 'content', 'date', 'thumbs_up']].copy()
            
            # Format date if it's a datetime
            if 'date' in display_df.columns and pd.api.types.is_datetime64_any_dtype(display_df['date']):
                display_df['date'] = display_df['date'].dt.strftime('%Y-%m-%d %H:%M')
            
            # Display the DataFrame
            st.dataframe(display_df)

def display_error(error_message: str) -> None:
    st.error(f"Analysis could not be completed: {error_message}")

def display_eu_ai_act_note() -> None:
    st.info("""
        **Note:** This analysis focuses on summarizing user reviews and comparing them to developer descriptions.
        Assessing specific risks or threats according to the EU AI Act criteria would require a separate, dedicated analysis based on the Act's defined risk categories and rules, which is **not** implemented here.
    """)

def display_footer() -> None:
    """Display the footer inside a div for CSS targeting."""
    footer_html = """
    <div class="app-footer">
        <hr>
        <p style="text-align: center; font-size: 0.9em; color: gray;">
            Powered by Streamlit, OpenAI, and google-play-scraper.
        </p>
    </div>
    """
    st.markdown(footer_html, unsafe_allow_html=True)

def display_analysis_results(results: AnalysisResults, app_name: str) -> None:
    st.divider()
    st.subheader(f"Analysis Results for: {app_name}")
    if results.has_error():
        display_error(results.error)
    else:
        # Display review summary analysis
        display_review_summary(results.user_review_analysis, 
                              results.filtered_review_count, 
                              results.raw_review_count)
        
        # Display difference analysis
        display_difference_analysis(results.difference_analysis)
        
        # Display sample reviews
        display_sample_reviews(results.filtered_reviews_sample)
    
    # Display EU AI Act note
    display_eu_ai_act_note()
    
    # Removed footer call from here
    # display_footer() 