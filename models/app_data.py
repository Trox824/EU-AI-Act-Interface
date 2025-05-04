"""
Data models for the app.
"""
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import pandas as pd

@dataclass
class AppDetails:
    """App details from the Play Store."""
    name: str
    publisher: str 
    description: str
    link: str
    shared_data: str = "Not specified"
    collected_data: str = "Not specified"
    security_practices: str = "Not specified"

@dataclass
class AnalysisResults:
    """Results of the app analysis."""
    developer_details: AppDetails
    user_review_analysis: str = ""
    difference_analysis: str = ""
    raw_review_count: int = 0
    filtered_review_count: int = 0
    filtered_reviews_sample: Optional[pd.DataFrame] = None
    error: str = ""
    eu_ai_act_classification: Optional[Dict[str, Any]] = None
    
    def has_error(self) -> bool:
        """Check if there's an error in the results."""
        return bool(self.error) 