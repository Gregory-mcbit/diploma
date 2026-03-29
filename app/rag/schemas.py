from pydantic import BaseModel, ConfigDict
from typing import Dict, Any, Optional

class KnowledgeDocument(BaseModel):
    model_config = ConfigDict(extra='ignore')
    id: str
    content: str
    category: str # e.g., "methodology", "constraints", "profile_mapping"

class NewsCacheItem(BaseModel):
    model_config = ConfigDict(extra='ignore')
    ticker: str
    summary_text: str
    scraped_at: float # Unix timestamp
    source: str

class DecisionMemoryRecord(BaseModel):
    model_config = ConfigDict(extra='ignore')
    id: str
    profile_type: str
    regime: str
    critic_verdict: str
    portfolio_summary: str
    full_context_json: str
