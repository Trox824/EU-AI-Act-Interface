"""
Service for interacting with OpenAI for analysis.
"""
import time
from typing import Optional, Literal, Dict, Any, List
import pandas as pd
import streamlit as st
from openai import OpenAI
from pydantic import BaseModel
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors

from models.app_data import AppDetails, AnalysisResults
from services.logger import logger, StatusLogger
from services.cache import CacheService
from config.settings import OPENAI_MODEL
from utils.data_utils import filter_reviews_by_length, prepare_reviews_for_analysis
import openai

class EUAIActResponse(BaseModel):
    answer: Literal["Yes", "No"]
    reasoning: str
    supporting_reviews: Optional[str] = None

class EvaluationResult(BaseModel):
    question: str
    reasoning: str
    supporting_reviews: Optional[str] = None
    confidence: float

class ReviewAnalysisResponse(BaseModel):
    content: str

class DifferenceAnalysisResponse(BaseModel):
    content: str

class FallbackResponse(BaseModel):
    answer: Literal["Yes", "No"]

class AnalysisService:
    def __init__(self, api_key: Optional[str] = None):
        try:
            if api_key:
                openai.api_key = api_key
                self.client = OpenAI(api_key=api_key)
            else:
                openai.api_key = st.secrets["OPENAI_API_KEY"]
                self.client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
            
            # Initialize cache service
            self.cache_service = CacheService()
            
            # Test the connection (optional)
            self.client.models.list()
            logger.info("OpenAI client initialized successfully.")
            
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}", exc_info=True)
            raise

    
    def _create_review_vector_db(self, filtered_df: pd.DataFrame, log: StatusLogger) -> dict:
        """Create a vector database from the reviews for semantic search using OpenAI API."""
        log.update(label="Creating vector database from reviews...")
        
        try:
            # Extract review texts and indices
            review_texts = filtered_df['content'].tolist()
            review_indices = filtered_df['review_index'].tolist()
            
            # Generate embeddings using OpenAI API
            response = openai.embeddings.create(
                model="text-embedding-ada-002",
                input=review_texts
            )
            
            # Access the embeddings from the response
            embeddings = [item.embedding for item in response.data]
            
            # Normalize embeddings for cosine similarity
            embeddings = np.array(embeddings)
            embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
            
            # Create a NearestNeighbors index
            nn_model = NearestNeighbors(metric='cosine', algorithm='brute')
            nn_model.fit(embeddings)
            
            log.info(f"Created vector database with {len(review_texts)} reviews")
            
            return {
                'model': "text-embedding-ada-002",  # Store model name
                'nn_model': nn_model,  # Store NearestNeighbors model
                'embeddings': embeddings,  # Keep raw embeddings for reference
                'review_texts': review_texts,
                'review_indices': review_indices
            }
        except Exception as e:
            log.error(f"Error creating review vector database: {e}", exc_info=True)
            # Return an empty DB if there's an error
            return {
                'model': None,
                'nn_model': None,
                'embeddings': None,
                'review_texts': [],
                'review_indices': []
            }

    def _get_relevant_reviews(self, query: str, review_db: dict, top_k: int = 5) -> List[dict]:
        """Retrieve reviews most relevant to the query from the vector database using OpenAI API."""
        # Explicitly check if embeddings are None
        if review_db['embeddings'] is None or review_db['nn_model'] is None:
            logger.warning("Review database is empty or not initialized.")
            return []
        
        try:
            # Generate query embedding using OpenAI API
            response = openai.embeddings.create(
                model="text-embedding-ada-002",
                input=[query]
            )
            
            # Access the query embedding from the response
            query_embedding = np.array(response.data[0].embedding)
            
            # Normalize the query embedding
            query_embedding = query_embedding / np.linalg.norm(query_embedding)
            
            # Use NearestNeighbors to find the top_k most similar reviews
            distances, indices = review_db['nn_model'].kneighbors([query_embedding], n_neighbors=top_k)
            
            # Format the results
            relevant_reviews = []
            for i, idx in enumerate(indices[0]):  # indices is a 2D array
                review_text = review_db['review_texts'][idx]
                review_index = review_db['review_indices'][idx]
                similarity_score = 1 - distances[0][i]  # Convert cosine distance to similarity
                
                relevant_reviews.append({
                    'review_index': review_index,
                    'text': review_text,
                    'similarity': similarity_score
                })
            
            return relevant_reviews
        except Exception as e:
            logger.error(f"Error retrieving relevant reviews: {e}", exc_info=True)
            return []
    
    def analyze_reviews(self, app_name: str, app_id: str, reviews_text: str, 
                       status_logger: Optional[StatusLogger] = None) -> str:
        log = status_logger or logger
        
        # Try to get from cache first
        cached_result = self.cache_service.get_cached_analysis(app_id, "reviews")
        if cached_result:
            log.info(f"Using cached review analysis for {app_name}")
            return cached_result
        
        if not reviews_text:
            log.warning(f"No reviews text provided for analysis of {app_name}.")
            return "No reviews provided for analysis."
        
        prompt = f"""Analyze these user reviews for the app '{app_name}'. Focus on reviews that are detailed and substantive. Extract the following information:

1.  **Key Features Mentioned:** List the main features or functionalities users frequently discuss (positive or negative). Include specific review numbers that mention each feature.
2.  **Common Complaints & Issues:** Summarize the most common problems, bugs, or frustrations reported by users. Cite the specific review numbers that mention each issue.
3.  **Positive Feedback Themes:** Summarize the main aspects users appreciate or praise. Include review numbers that highlight each positive aspect.
4.  **Overall User Sentiment Summary:** Provide a brief (1-2 sentence) summary of the general user sentiment towards the app based *only* on these reviews.

For each point, cite the specific review numbers (e.g., "Review #3, #7") that support your analysis.

Reviews:
---
{reviews_text}
---

Format your response clearly with headings for each point (e.g., **Key Features Mentioned:**). Be concise and focus on recurring themes. Always include review numbers in your citations.
"""
        log.update(label=f"Analyzing reviews for {app_name} with {OPENAI_MODEL}...")
        log.info(f"Sending {len(reviews_text)} characters of review text to OpenAI for {app_name}.")
        
        try:
            start_time = time.time()
            response = self.client.beta.chat.completions.parse(
                model=OPENAI_MODEL,
                messages=[{"role": "user", "content": prompt}],
                response_format=ReviewAnalysisResponse,
                temperature=0.2
            )
            end_time = time.time()
            
            analysis = response.choices[0].message.parsed.content
            log.info(f"Successfully analyzed reviews for {app_name}. Time taken: {end_time - start_time:.2f}s")
            log.write(f"✓ Completed review analysis for {app_name}.")
            
            # Cache the result
            self.cache_service.cache_analysis(app_id, "reviews", analysis)
            
            return analysis
            
        except Exception as e:
            log.error(f"Error in OpenAI API call for {app_name} review analysis: {e}", exc_info=True)
            return f"Error analyzing reviews: {str(e)}"
    
    def analyze_difference(self, app_details: AppDetails, user_review_summary: str,
                          status_logger: Optional[StatusLogger] = None) -> str:
        log = status_logger or logger
        app_name = app_details.name
        app_id = app_details.app_id
        
        # Try to get from cache first
        cached_result = self.cache_service.get_cached_analysis(app_id, "difference")
        if cached_result:
            log.info(f"Using cached difference analysis for {app_name}")
            return cached_result
        
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
            response = self.client.beta.chat.completions.parse(
                model=OPENAI_MODEL,
                messages=[{"role": "user", "content": prompt}],
                response_format=DifferenceAnalysisResponse,
                temperature=0.2
            )
            end_time = time.time()
            
            analysis_text = response.choices[0].message.parsed.content.strip()
            log.info(f"Successfully analyzed differences for {app_name}. Time taken: {end_time - start_time:.2f}s")
            log.write(f"✓ Completed difference analysis for {app_name}.")
            
            # Remove the "Difference Analysis:" prefix if the model includes it
            if analysis_text.startswith("Difference Analysis:"):
                analysis_text = analysis_text[len("Difference Analysis:"):].strip()
            
            # Cache the result
            self.cache_service.cache_analysis(app_id, "difference", analysis_text)
                
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
        
        # # Try to get filtered reviews from cache
        # filtered_df = self.cache_service.get_cached_dataframe(app_details.app_id, "filtered_reviews")
        # if filtered_df is not None:
        #     log.info(f"Using cached filtered reviews for {app_details.name}")
        #     results.filtered_review_count = len(filtered_df)
        #     results.filtered_reviews = filtered_df  # Store complete filtered DataFrame
        #     results.filtered_reviews_sample = filtered_df.head()  # Store sample for display
        #     log.write(f"✓ Using cached filtered reviews: {results.filtered_review_count} reviews.")
        # else:
        #     # Filter reviews
        #     log.update(label=f"Filtering reviews for {app_details.name}...")
        #     # Make sure we have review_index column
        #     if 'review_index' not in reviews_df.columns:
        #         reviews_df['review_index'] = range(1, len(reviews_df) + 1)
            
        #     filtered_df = filter_reviews_by_length(reviews_df)
        #     results.filtered_review_count = len(filtered_df)
        #     results.filtered_reviews = filtered_df  # Store complete filtered DataFrame
        #     results.filtered_reviews_sample = filtered_df.head()  # Store sample for display
            
        #     # Cache the filtered reviews
        #     self.cache_service.cache_dataframe(app_details.app_id, "filtered_reviews", filtered_df)
        #     log.write(f"✓ Filtered reviews: {results.filtered_review_count} remaining (with sufficient length).")
        
        # # Check if we have filtered reviews
        # if filtered_df.empty:
        #     log.warning(f"No reviews met the minimum length requirement for {app_details.name}.")
        #     results.error = "No sufficiently long reviews found for analysis."
        #     return results
        
        # Prepare text for analysis
        reviews_text = prepare_reviews_for_analysis(reviews_df)
        
        # # Analyze reviews
        # results.user_review_analysis = self.analyze_reviews(
        #     app_details.name, app_details.app_id, reviews_text, status_logger=log
        # )
        
        # # Check if review analysis succeeded
        # if results.user_review_analysis.startswith("Error analyzing reviews:"):
        #     results.error = results.user_review_analysis
        #     return results
        
        # Analyze difference
        results.difference_analysis = self.analyze_difference(
            app_details, reviews_text, status_logger=log
        )
        
        # Check if difference analysis succeeded
        if results.difference_analysis.startswith("Error analyzing differences:"):
            results.error = results.difference_analysis
            return results
        
        return results
    
    def _evaluate_single_prompt(self, question_text: str, input_description: str, 
                               relevant_reviews: List[dict], log: StatusLogger) -> Optional[EvaluationResult]:
        """Evaluate a single prompt for EU AI Act classification using relevant reviews."""
        try:
            # Format relevant reviews for the prompt
            reviews_text = ""
            for review in relevant_reviews:
                reviews_text += f"- Review #{review['review_index']}: {review['text']}\n"
            
            prompt = f"""Evaluate the following question about an AI app based on the provided information.
            Answer with ONLY Yes or No, followed by your reasoning.

            Question: {question_text}

            App information:
            {input_description}
            
            Relevant User Reviews:
            {reviews_text}

            If your answer is Yes, you MUST:
            1. Explain your reasoning
            2. Cite specific user reviews that support your answer (use format: "Review #X")
            3. Quote the relevant parts of those reviews
            4. Explain how each cited review supports your answer

            Format your response as:
            Answer: [Yes/No]
            Reasoning: [Your detailed reasoning]
            Supporting Reviews:
            - Review #[number]: [Quote the relevant part]
              How it supports: [Explain how this review supports your answer]
            - Review #[number]: [Quote the relevant part]
              How it supports: [Explain how this review supports your answer]
            """

            response = self.client.beta.chat.completions.parse(
                model=OPENAI_MODEL,
                messages=[{"role": "user", "content": prompt}],
                response_format=EUAIActResponse,
                temperature=0.1
            )
            
            parsed_response = response.choices[0].message.parsed
            log.logger.info(f"Evaluated: '{question_text}' - Result: {parsed_response.answer}")
            
            if parsed_response.answer == "Yes":
                return EvaluationResult(
                    question=question_text,
                    reasoning=parsed_response.reasoning,
                    supporting_reviews=reviews_text,
                    confidence=1.0
                )
                
        except Exception as e:
            log.logger.error(f"Error evaluating '{question_text}': {e}", exc_info=True)
            
            # Simple fallback without parsing
            try:
                fallback_prompt = (
                    "Answer with ONLY Yes or No:\n"
                    f"Question: {question_text}\n"
                    f"App information: {input_description}"
                )
                
                fallback_response = self.client.beta.chat.completions.parse(
                    model="gpt-4.1-mini",
                    messages=[{"role": "user", "content": fallback_prompt}],
                    response_format=FallbackResponse,
                    temperature=0.1
                )
                
                response_text = fallback_response.choices[0].message.parsed.answer.lower()
                
                if response_text == "yes":
                    log.logger.info(f"✓ [Fallback] Evaluated '{question_text}' as Yes")
                    return EvaluationResult(
                        question=question_text,
                        reasoning=response_text,
                        confidence=0.8
                    )
                    
            except Exception as fallback_error:
                log.logger.error(f"Fallback evaluation failed: {fallback_error}", exc_info=True)
        
        return None

    def perform_eu_ai_act_classification(self, app_details: AppDetails,
                                        difference_analysis: str, filtered_reviews_df: pd.DataFrame,
                                        prompts_df: pd.DataFrame, status_logger: Optional[StatusLogger] = None) -> dict:
        log = status_logger or logger
        app_id = app_details.app_id
        
        # Try to get from cache first
        cached_result = self.cache_service.get_cached_analysis(app_id, "eu_ai_act")
        if cached_result:
            log.info(f"Using cached EU AI Act classification for {app_details.name}")
            return cached_result
        
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
        if difference_analysis and not difference_analysis.startswith("Error"):
            input_description += f"\n\nAnalysis of Differences (User Reviews vs. Developer Claims):\n---\n{difference_analysis}\n---"
        
        # Create vector database from filtered reviews
        review_db = self._create_review_vector_db(filtered_reviews_df, log)
        
        # Process risk types from highest to lowest risk
        risk_types = ["Unacceptable risk", "High risk", "Limited risk"]
        
        for risk_type in risk_types:
            # Filter prompts by risk type
            risk_prompts = prompts_df[prompts_df['Type'] == risk_type]
            
            if risk_prompts.empty:
                continue
                
            log.update(label=f"Evaluating {risk_type} criteria for {app_details.name}...")
            
            # Use ThreadPoolExecutor to process prompts in parallel
            triggered_questions_info = []
            with ThreadPoolExecutor(max_workers=5) as executor:
                # Submit all prompts for this risk level
                future_to_prompt = {}
                for _, row in risk_prompts.iterrows():
                    prompt_text = row['Prompt']
                    # Retrieve relevant reviews for this prompt
                    relevant_reviews = self._get_relevant_reviews(prompt_text, review_db, top_k=8)
                    future = executor.submit(
                        self._evaluate_single_prompt,
                        prompt_text,
                        input_description,
                        relevant_reviews,
                        log
                    )
                    future_to_prompt[future] = prompt_text
                
                # Collect results as they complete
                for future in as_completed(future_to_prompt):
                    result = future.result()
                    if result:
                        triggered_questions_info.append(result)

            # If ANY questions triggered this risk level, classify and return
            if triggered_questions_info:
                log.write(f"✓ Classified as {risk_type} based on {len(triggered_questions_info)} trigger(s).")
                overall_confidence = max(info.confidence for info in triggered_questions_info) if triggered_questions_info else 0.0
                
                result = {
                    'filtered_reviews': filtered_reviews_df,
                    'risk_type': risk_type,
                    'confidence_score': overall_confidence,
                    'triggered_questions': [
                        {
                            'question': info.question,
                            'reasoning': info.reasoning,
                            'supporting_reviews': info.supporting_reviews,
                            'confidence': info.confidence
                        }
                        for info in triggered_questions_info
                    ]
                }
                
                # Cache the result
                self.cache_service.cache_analysis(app_id, "eu_ai_act", result)
                return result
        
        # If no risk type was triggered, classify as minimal risk
        result = {
            'risk_type': "Minimal Risk",
            'confidence_score': 1.0,
            'triggered_questions': []
        }
        
        # Cache the result
        self.cache_service.cache_analysis(app_id, "eu_ai_act", result)
        return result