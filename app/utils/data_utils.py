"""
Utility functions for data processing.
"""
import pandas as pd
from typing import List, Dict, Any
from models.app_data import Review
from config.settings import MIN_REVIEW_LENGTH

def create_reviews_dataframe(reviews: List[Review]) -> pd.DataFrame:
    if not reviews:
        return pd.DataFrame()
    
    # Convert list of Review objects to list of dicts
    review_dicts = [vars(review) for review in reviews]
    
    # Create DataFrame
    df = pd.DataFrame(review_dicts)
    
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
    
    # Filter by length and return a copy
    return df[df['content'].str.len() >= min_length].copy()

def prepare_reviews_for_analysis(df: pd.DataFrame) -> str:
    if df.empty or 'content' not in df.columns:
        return ""
    
    # Join review content with delimiter
    return "\n\n---\n\n".join(df['content'].tolist()) 