from typing import List

from pydantic import BaseModel, ConfigDict, Field

from app.domain.schemas import FreshnessStatus


class KnowledgeDocument(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str
    content: str
    category: str


class KnowledgeMatch(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str
    category: str
    content: str


class KnowledgeRetrievalResult(BaseModel):
    model_config = ConfigDict(extra="ignore")
    retrieval_id: str
    query: str
    matches: List[KnowledgeMatch] = Field(default_factory=list)
    source: str = "knowledge_rag"


class NewsCacheItem(BaseModel):
    model_config = ConfigDict(extra="ignore")
    retrieval_id: str
    ticker: str
    summary_text: str
    scraped_at: float
    source: str


class NewsCacheLookup(BaseModel):
    model_config = ConfigDict(extra="ignore")
    retrieval_id: str
    ticker: str
    hit: bool
    freshness_status: FreshnessStatus
    payload: str | None = None
    scraped_at: float | None = None
    age_hours: float | None = None
    source: str | None = None


class DecisionMemoryRecord(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str
    profile_type: str
    regime: str
    critic_verdict: str
    portfolio_summary: str
    full_context_json: str


class DecisionMemoryMatch(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str
    profile_type: str
    regime: str
    verdict: str
    content: str
    portfolio_json: str | None = None


class DecisionMemoryQueryResult(BaseModel):
    model_config = ConfigDict(extra="ignore")
    retrieval_id: str
    query: str
    matches: List[DecisionMemoryMatch] = Field(default_factory=list)
    source: str = "decision_memory_rag"
