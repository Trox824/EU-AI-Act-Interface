"""
Utility functions for data processing.
"""
import pandas as pd
from typing import List, Dict, Any
from models.app_data import Review
from config.settings import MIN_REVIEW_LENGTH
import tempfile
import json
import os
from openai import vector_stores, files
from services.logger import logger, StatusLogger
from pathlib import Path
import openai
import numpy as np
from sklearn.neighbors import NearestNeighbors

def create_reviews_dataframe(reviews: List[Review]) -> pd.DataFrame:
    if not reviews:
        return pd.DataFrame()
    
    # Convert list of Review objects to list of dicts
    review_dicts = [vars(review) for review in reviews]
    
    # Create DataFrame
    df = pd.DataFrame(review_dicts)
    
    # Add review index
    df['review_index'] = range(0, len(df))
    
    # Convert date to datetime if possible
    if 'date' in df.columns:
        try:
            df['date'] = pd.to_datetime(df['date'])
        except (TypeError, ValueError):
            # Keep as is if conversion fails
            pass
    
    return df

def filter_reviews_by_length(df: pd.DataFrame, min_length: int = MIN_REVIEW_LENGTH) -> pd.DataFrame:
    if df.empty or 'content' not in df.columns:
        return pd.DataFrame()
    
    # Ensure content is string and handle potential None values
    df['content'] = df['content'].astype(str).fillna('')
    
    # Filter by length and return a copy, preserving all original columns including review_index
    filtered_df = df[df['content'].str.len() >= min_length].copy()
    
    # Reindex the reviews to maintain sequential numbering
    filtered_df['review_index'] = range(1, len(filtered_df) + 1)
    
    return filtered_df

def prepare_reviews_for_analysis(df: pd.DataFrame) -> str:
    if df.empty or 'content' not in df.columns:
        return ""
    
    # Add review index if not present
    if 'review_index' not in df.columns:
        df['review_index'] = range(1, len(df) + 1)
    
    # Format each review with its index
    formatted_reviews = []
    for _, row in df.iterrows():
        review_text = f"Review #{row['review_index']}:\n{row['content']}"
        formatted_reviews.append(review_text)
    
    # Join review content with delimiter
    return "\n\n---\n\n".join(formatted_reviews) 

def create_review_vector_db(filtered_df: pd.DataFrame, log: StatusLogger) -> dict:
    """Create a vector database from the reviews for semantic search using OpenAI API."""
    log.update(label="Creating vector database from reviews...")
    
    try:
        # Extract review texts and indices
        review_texts = filtered_df['content'].tolist()
        review_indices = filtered_df['review_index'].tolist()
        
        # Generate embeddings using OpenAI API
        response = openai.embeddings.create(
            model="text-embedding-3-small",
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
            'model': "text-embedding-3-small",  # Store model name
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

def get_relevant_reviews(query: str, review_db: dict, top_k=5) -> List[dict]:
    """Retrieve reviews most relevant to the query from the vector database using OpenAI API."""
    # Explicitly check if embeddings are None
    if review_db['embeddings'] is None or review_db['nn_model'] is None:
        logger.warning("Review database is empty or not initialized.")
        return []
    
    try:
        # Generate query embedding using OpenAI API
        response = openai.embeddings.create(
            model="text-embedding-3-small",
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


