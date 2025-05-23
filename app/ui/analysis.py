"""
UI components for displaying analysis results.
"""
import streamlit as st
import pandas as pd
from typing import Optional
import re
import csv
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
    
    # Display the analysis
    st.markdown(analysis)
    
    # Extract review numbers from the analysis
    review_numbers = set(re.findall(r'Review #(\d+)', analysis))
    
    # Display citation information
    if review_numbers:
        st.markdown("**Cited Reviews:**")
        st.markdown(f"This analysis is based on {len(review_numbers)} specific reviews (Reviews #{', #'.join(sorted(review_numbers))})")
    
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

def display_cited_reviews(citations: str, reviews_df: pd.DataFrame) -> None:
    """Display the actual content of cited reviews."""
    if not citations or reviews_df.empty:
        return
        
    # Extract review numbers from citations
    review_numbers = set(re.findall(r'Review #(\d+)', citations))
    
    if not review_numbers:
        return
        
    st.markdown("**Cited Reviews:**")
    
    # Filter reviews to show only the cited ones
    cited_reviews = reviews_df[reviews_df['review_index'].isin([int(num) for num in review_numbers])]
    if not cited_reviews.empty:
        # Select and format columns for display
        display_df = cited_reviews[['review_index', 'score', 'content', 'date', 'thumbs_up']].copy()
        
        # Format date if it's a datetime
        if 'date' in display_df.columns and pd.api.types.is_datetime64_any_dtype(display_df['date']):
            display_df['date'] = display_df['date'].dt.strftime('%Y-%m-%d %H:%M')
        
        # Rename columns for better display
        display_df.columns = ['Review #', 'Rating', 'Content', 'Date', 'Thumbs Up']
        
        # Display the DataFrame
        st.dataframe(display_df, use_container_width=True)

def display_supporting_reviews(supporting_reviews: str, reviews_df: pd.DataFrame) -> None:
    """Display the supporting reviews with their basic details."""
    if not supporting_reviews or reviews_df.empty:
        return

    st.markdown("**Supporting Reviews:**")

    # Split the supporting reviews into individual review sections
    review_sections = supporting_reviews.split('- Review #')
    review_sections = [s for s in review_sections if s.strip()]

    for section_text in review_sections:
        try:
            # Extract review number part and the rest of the section
            num_part, _ = section_text.split(':', 1)
            review_num_str = num_part.strip()
            review_num = int(review_num_str)  # Convert to integer

            # Get the review from the DataFrame using review_index
            review = reviews_df[reviews_df['review_index'] == review_num].iloc[0]

            # Display review information
            st.markdown(f"**Review #{review_num} (Rating: {review['score']}⭐)**")
            st.markdown(f"**Content:** {review['content']}")
            st.caption(f"Date: {review['date']} | Thumbs Up: {review['thumbs_up']}")
            st.divider()

        except IndexError:
            st.warning(f"No review found with index {review_num}. Total reviews in dataset: {len(reviews_df)}")
            continue
        except ValueError:
            st.warning(f"Could not parse review number from '{section_text.split(':')[0].strip()}'.")
            continue
        except Exception as e:
            st.warning(f"Error processing review section: {str(e)}")
            continue

import csv

def display_eu_ai_act_classification(classification_result: dict, prompts_path: str, reviews_df: Optional[pd.DataFrame] = None) -> None:
    """Display the EU AI Act classification results."""
    st.subheader("EU AI Act Classification")
    
    # Define colors for different risk types
    risk_colors = {
        "Unacceptable risk": "red",
        "High risk": "orange",
        "Limited risk": "blue",
        "Minimal Risk": "green"
    }
    
    risk_type = classification_result.get('risk_type', 'Unknown')
    color = risk_colors.get(risk_type, "gray")
    
    # Display the risk classification prominently
    st.markdown(f"<h3 style='color: {color};'>Classification: {risk_type}</h3>", unsafe_allow_html=True)
    
    # Get the list of triggered questions
    triggered_questions = classification_result.get('triggered_questions', [])
    
    # Load prompts and snippets from the prompts_path file
    prompts_data = {}
    try:
        with open(prompts_path, mode='r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                prompts_data[row['Prompt']] = {
                    'Snippet': row['Snippet'],
                    'Type': row['Type']
                }
    except Exception as e:
        st.error(f"Error reading prompts file: {e}")
        return
    
    # Define source links for risk types
    risk_type_links = {
        "Unacceptable risk": "https://artificialintelligenceact.eu/article/5/",
        "High risk": "https://artificialintelligenceact.eu/annex/3/",
        "Limited risk": "https://artificialintelligenceact.eu/article/50/",
        "Minimal Risk": ""
    }
    
    # If there are triggered questions, display them
    if triggered_questions:
        st.markdown("**Triggering Criteria:**")
        for trigger in triggered_questions:
            question = trigger['question']
            snippet = prompts_data.get(question, {}).get('Snippet', 'N/A')
            risk_type = prompts_data.get(question, {}).get('Type', 'Unknown')
            source_link = risk_type_links.get(risk_type, "#")
            
            with st.expander(f"Question: {question}", expanded=True):
                st.markdown(f"**Source:** [Learn more about {risk_type}]({source_link})")
                st.markdown(f"**Section:** {snippet}")
                
                if trigger.get('reasoning'):
                    st.markdown("**Reasoning:**")
                    st.markdown(trigger['reasoning'])
                
                if trigger.get('supporting_reviews') and reviews_df is not None:
                    display_supporting_reviews(trigger['supporting_reviews'], reviews_df)
                
        st.write(" ")  # Add some space
    elif risk_type == "Minimal Risk":
        st.markdown("No specific criteria for higher risk levels were triggered.")
    
    # Show overall confidence score
    confidence = classification_result.get('confidence_score', 0.0)
    st.progress(confidence)
    st.caption(f"Overall Confidence: {confidence:.2f}")
    
    # Add explanation about risk categories
    with st.expander("About EU AI Act Risk Categories", expanded=False):
        st.markdown("""
        **EU AI Act Risk Categories:**
        
        - **Unacceptable risk:** AI systems that pose a clear threat to people's safety, livelihoods, or rights. These are prohibited.
        - **High risk:** AI systems with significant potential for harm to health, safety, fundamental rights, environment, democracy, or rule of law.
        - **Limited risk:** AI systems with specific transparency obligations, like disclosing that content is AI-generated.
        - **Minimal Risk:** AI systems with minimal or no risk that are not specifically regulated.
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

def display_analysis_results(results: AnalysisResults, app_name: str, prompts_path: str) -> None:
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
        
        # Display sample reviews in an expander
        display_sample_reviews(results.filtered_reviews_sample)
        
        # Display EU AI Act classification if available
        if hasattr(results, 'eu_ai_act_classification') and results.eu_ai_act_classification:
            # Use the full filtered reviews dataset if available, otherwise fall back to sample
            reviews_for_display = results.eu_ai_act_classification['filtered_reviews']
            print(f"Reviews for display: {reviews_for_display}")
            display_eu_ai_act_classification(
                results.eu_ai_act_classification,
                prompts_path,
                reviews_for_display,
            )
        else:
            # Display EU AI Act note if no classification is available
            display_eu_ai_act_note()
    
    # Removed footer call from here
    # display_footer() 