"""
Service for interacting with OpenAI for analysis.
"""
import time
from typing import Optional, Literal
import pandas as pd
import streamlit as st
from openai import OpenAI
from pydantic import BaseModel

from models.app_data import AppDetails, AnalysisResults
from services.logger import logger, StatusLogger
from config.settings import OPENAI_MODEL
from utils.data_utils import filter_reviews_by_length, prepare_reviews_for_analysis

class EUAIActResponse(BaseModel):
    answer: Literal["Yes", "No"]
    reasoning: str

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
                temperature=0.2,  # Slightly lower temp for more factual summary
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
    
    def perform_eu_ai_act_classification(self, app_details: AppDetails, reviews_analysis: str,
                                        difference_analysis: str,
                                        prompts_df: pd.DataFrame, status_logger: Optional[StatusLogger] = None) -> dict:
        log = status_logger or logger
        log.update(label=f"Classifying {app_details.name} according to EU AI Act...")
        
        # Prepare base description
        base_description = (
            f"App Name: {app_details.name}. "
            f"Publisher: {app_details.publisher}. "
            f"Full Description: {app_details.description}. "
            f"Data shared with third parties: {app_details.shared_data}. "
            f"Data collected by the app: {app_details.collected_data}. "
            f"Security practices: {app_details.security_practices}."
        )
        
        # Add review analysis and difference analysis if available
        input_description = base_description
        if reviews_analysis and not reviews_analysis.startswith("Error"):
            input_description += f"\n\nUser Review Analysis Summary:\n---\n{reviews_analysis}\n---"
        if difference_analysis and not difference_analysis.startswith("Error"):
            input_description += f"\n\nAnalysis of Differences (User Reviews vs. Developer Claims):\n---\n{difference_analysis}\n---"
        
        # Process risk types from highest to lowest risk
        risk_types = ["Unacceptable risk", "High risk", "Limited risk"]
        
        for risk_type in risk_types:
            triggered_questions_info = [] # Store info for questions triggering this risk level
            
            # Filter prompts by risk type
            risk_prompts = prompts_df[prompts_df['Type'] == risk_type]
            
            if risk_prompts.empty:
                continue
                
            log.update(label=f"Evaluating {risk_type} criteria for {app_details.name}...")
            
            # Check ALL prompts for this risk level
            for _, prompt_row in risk_prompts.iterrows():
                trigger_info = None # Reset for each question
                question_text = prompt_row['Prompt']
                
                try:
                    # Prepare the system prompt
                    system_prompt = (
                        "You are an EU AI Act compliance evaluator. "
                        "Answer the question with 'Yes' or 'No' based on the app information provided. "
                        "Provide a brief reasoning for your answer in 1-2 sentences."
                    )
                    
                    # Prepare the user prompt
                    user_prompt = f"Question: {question_text}\n\nApp information: {input_description}"
                    
                    # Call OpenAI API
                    response = self.client.beta.chat.completions.parse(
                        model="gpt-4.1-mini",
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}
                        ],
                        response_format=EUAIActResponse,
                        temperature=0.1
                    )
                    
                    parsed_response = response.choices[0].message.parsed
                    log.write(f"Evaluated: '{question_text}' - Result: {parsed_response.answer}")
                    
                    if parsed_response.answer == "Yes":
                        trigger_info = {
                            'question': question_text,
                            'reasoning': parsed_response.reasoning,
                            'confidence': 1.0
                        }
                        
                except Exception as e:
                    log.error(f"Error evaluating '{question_text}': {e}", exc_info=True)
                    
                    # Simple fallback without parsing
                    try:
                        fallback_prompt = (
                            "Answer with ONLY Yes or No:\n"
                            f"Question: {question_text}\n"
                            f"App information: {input_description}"
                        )
                        
                        fallback_response = self.client.chat.completions.create(
                            model="gpt-4.1-nano",
                            messages=[{"role": "user", "content": fallback_prompt}],
                            temperature=0.1,
                            max_tokens=50
                        )
                        
                        response_text = fallback_response.choices[0].message.content.strip().lower()
                        
                        if "yes" in response_text:
                            log.write(f"✓ [Fallback] Classified as {risk_type} for '{question_text}'")
                            trigger_info = {
                                'question': question_text,
                                'reasoning': response_text, # Store the raw fallback response as reasoning
                                'confidence': 0.8 # Lower confidence for fallback
                            }
                            
                    except Exception as fallback_error:
                        log.error(f"Fallback also failed for '{question_text}': {fallback_error}", exc_info=True)
                
                # If this question triggered a 'Yes', add its info
                if trigger_info:
                    triggered_questions_info.append(trigger_info)

            # If ANY questions triggered this risk level, classify and return
            if triggered_questions_info:
                log.write(f"✓ Classified as {risk_type} based on {len(triggered_questions_info)} trigger(s).")
                # Calculate overall confidence (e.g., average, max, or just use 1.0 if any standard 'Yes')
                overall_confidence = max(info['confidence'] for info in triggered_questions_info) if triggered_questions_info else 0.0
                
                return {
                    'risk_type': risk_type,
                    'confidence_score': overall_confidence,
                    'triggered_questions': triggered_questions_info # Return list of triggers
                }
        
        # If no risk criteria matched in the loops, classify as Minimal Risk
        log.write(f"✓ Classified as Minimal Risk (no higher risk criteria matched)")
        return {
            'risk_type': "Minimal Risk",
            'confidence_score': 1.0,
            'triggered_questions': [] # Empty list for minimal risk
        } 