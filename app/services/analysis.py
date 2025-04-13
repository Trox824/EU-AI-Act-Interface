"""
Service for interacting with OpenAI for analysis.
"""
import time
from typing import Optional
import pandas as pd
import streamlit as st
from openai import OpenAI

from models.app_data import AppDetails, AnalysisResults
from services.logger import logger, StatusLogger
from config.settings import OPENAI_MODEL
from utils.data_utils import filter_reviews_by_length, prepare_reviews_for_analysis

class AnalysisService:
    def __init__(self, api_key: Optional[str] = None):
        try:
            if api_key:
                self.client = OpenAI(api_key=api_key)
            else:
                self.client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
            
            # Test the connection (optional)
            self.client.models.list()
            logger.info("OpenAI client initialized successfully.")
            
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}", exc_info=True)
            raise
    
    def analyze_reviews(self, app_name: str, reviews_text: str, 
                       status_logger: Optional[StatusLogger] = None) -> str:
        log = status_logger or logger
        
        if not reviews_text:
            log.warning(f"No reviews text provided for analysis of {app_name}.")
            return "No reviews provided for analysis."
        
        prompt = f"""Analyze these user reviews for the app '{app_name}'. Focus on reviews that are detailed and substantive. Extract the following information:

1.  **Key Features Mentioned:** List the main features or functionalities users frequently discuss (positive or negative).
2.  **Common Complaints & Issues:** Summarize the most common problems, bugs, or frustrations reported by users.
3.  **Positive Feedback Themes:** Summarize the main aspects users appreciate or praise.
4.  **Overall User Sentiment Summary:** Provide a brief (1-2 sentence) summary of the general user sentiment towards the app based *only* on these reviews.

Reviews:
---
{reviews_text}
---

Format your response clearly with headings for each point (e.g., **Key Features Mentioned:**). Be concise and focus on recurring themes.
"""
        log.update(label=f"Analyzing reviews for {app_name} with {OPENAI_MODEL}...")
        log.info(f"Sending {len(reviews_text)} characters of review text to OpenAI for {app_name}.")
        
        try:
            start_time = time.time()
            response = self.client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.5,  # Slightly lower temp for more factual summary
                max_tokens=600  # Allow more tokens for potentially detailed analysis
            )
            end_time = time.time()
            
            analysis = response.choices[0].message.content
            log.info(f"Successfully analyzed reviews for {app_name}. Time taken: {end_time - start_time:.2f}s")
            log.write(f"✓ Completed review analysis for {app_name}.")
            
            return analysis
            
        except Exception as e:
            log.error(f"Error in OpenAI API call for {app_name} review analysis: {e}", exc_info=True)
            return f"Error analyzing reviews: {str(e)}"
    
    def analyze_difference(self, app_details: AppDetails, user_review_summary: str,
                          status_logger: Optional[StatusLogger] = None) -> str:
        log = status_logger or logger
        app_name = app_details.name
        
        log.update(label=f"Analyzing differences for {app_name} with {OPENAI_MODEL}...")
        log.info(f"Preparing comparison data for {app_name}.")
        
        # Prepare developer description string
        dev_desc_parts = [
            f"Full Description: {app_details.description}",
            f"Shared Data Claim: {app_details.shared_data}",
            f"Collected Data Claim: {app_details.collected_data}",
            f"Security Practices Claim: {app_details.security_practices}"
        ]
        developer_desc = "\n".join(dev_desc_parts)
        
        prompt = f"""Compare the user review summary with the developer's description and data safety claims for the app '{app_name}'.

User Review Summary (based on actual user feedback):
---
{user_review_summary}
---

Developer's Description & Data Practices (official claims):
---
{developer_desc}
---

Task: Identify and summarize the key differences, discrepancies, or contradictions between the user experiences (from the review summary) and the developer's claims/descriptions. Focus on:
- **Feature Gaps:** Promised/highlighted features vs. actual user experience or common complaints about them.
- **Performance Discrepancies:** Claims of performance/stability vs. user-reported bugs, lag, crashes, or usability issues.
- **Data Practice Concerns:** Stated data privacy/security practices vs. user concerns reflected in reviews (if any mentioned explicitly).
- **Marketing vs. Reality:** Overall tone/marketing language vs. the reality painted by user sentiment and issues.

Output a concise summary highlighting these differences. If the user reviews largely align with the developer's description and claims, state that. Structure the output clearly using markdown (e.g., bullet points).

Difference Analysis:
"""
        try:
            start_time = time.time()
            response = self.client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,  # Lower temp for more objective comparison
                max_tokens=500
            )
            end_time = time.time()
            
            analysis_text = response.choices[0].message.content.strip()
            log.info(f"Successfully analyzed differences for {app_name}. Time taken: {end_time - start_time:.2f}s")
            log.write(f"✓ Completed difference analysis for {app_name}.")
            
            # Remove the "Difference Analysis:" prefix if the model includes it
            if analysis_text.startswith("Difference Analysis:"):
                analysis_text = analysis_text[len("Difference Analysis:"):].strip()
                
            return analysis_text
            
        except Exception as e:
            log.error(f"Error in OpenAI API call for {app_name} difference analysis: {e}", exc_info=True)
            return f"Error analyzing differences: {str(e)}"
    
    def analyze_app(self, app_details: AppDetails, reviews_df: pd.DataFrame, 
                   status_logger: Optional[StatusLogger] = None) -> AnalysisResults:
        log = status_logger or logger
        
        # Initialize results object
        results = AnalysisResults(developer_details=app_details)
        
        # Check if we have reviews
        if reviews_df.empty:
            log.warning(f"No reviews found for {app_details.name}.")
            results.error = "No reviews found."
            return results
        
        # Store raw review count
        results.raw_review_count = len(reviews_df)
        log.write(f"Raw reviews fetched: {results.raw_review_count}")
        
        # Filter reviews
        log.update(label=f"Filtering reviews for {app_details.name}...")
        filtered_df = filter_reviews_by_length(reviews_df)
        results.filtered_review_count = len(filtered_df)
        results.filtered_reviews_sample = filtered_df.head()
        
        log.write(f"✓ Filtered reviews: {results.filtered_review_count} remaining (with sufficient length).")
        
        # Check if we have filtered reviews
        if filtered_df.empty:
            log.warning(f"No reviews met the minimum length requirement for {app_details.name}.")
            results.error = "No sufficiently long reviews found for analysis."
            return results
        
        # Prepare text for analysis
        reviews_text = prepare_reviews_for_analysis(filtered_df)
        
        # Analyze reviews
        results.user_review_analysis = self.analyze_reviews(
            app_details.name, reviews_text, status_logger=log
        )
        
        # Check if review analysis succeeded
        if results.user_review_analysis.startswith("Error analyzing reviews:"):
            results.error = results.user_review_analysis
            return results
        
        # Analyze difference
        results.difference_analysis = self.analyze_difference(
            app_details, results.user_review_analysis, status_logger=log
        )
        
        # Check if difference analysis succeeded
        if results.difference_analysis.startswith("Error analyzing differences:"):
            results.error = results.difference_analysis
            return results
        
        return results 