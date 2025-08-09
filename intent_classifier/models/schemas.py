"""
Pydantic models for intent classification system.
Defines schemas for input/output validation.
"""

from typing import List, Dict, Any, Optional, Literal
from datetime import datetime
from pydantic import BaseModel, Field, validator
from enum import Enum


class CompositionMode(str, Enum):
    """Composition modes for multi-intent queries."""
    SEQUENCE = "SEQUENCE"
    PARALLEL = "PARALLEL"
    SINGLE = "SINGLE"


class JoinKey(BaseModel):
    """Join key for connecting intents in a sequence."""
    from_step: int = Field(..., description="Source intent index")
    to_step: int = Field(..., description="Target intent index")
    key: str = Field(..., description="Entity key to pass between intents")


class QueryRequest(BaseModel):
    """Input query request schema."""
    query_id: str = Field(..., description="Unique query identifier")
    user_id: str = Field(..., description="User identifier")
    text: str = Field(..., description="Query text")
    created_at: Optional[datetime] = Field(default_factory=datetime.utcnow)


class Intent(BaseModel):
    """Single intent classification result."""
    coarse_class: str = Field(..., description="Top-level intent class")
    intent_code: str = Field(..., description="Full intent code (CLASS:SUBCLASS)")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Classification confidence")
    entities: Dict[str, Any] = Field(default_factory=dict, description="Extracted entities")
    policies: Dict[str, Any] = Field(default_factory=dict, description="Intent policies")
    routing_hints: Dict[str, Any] = Field(default_factory=dict, description="Routing hints")
    
    @validator('intent_code')
    def validate_intent_code(cls, v, values):
        """Ensure intent_code matches coarse_class."""
        if 'coarse_class' in values:
            if not v.startswith(f"{values['coarse_class']}:"):
                raise ValueError(f"Intent code {v} must start with {values['coarse_class']}:")
        return v


class Composition(BaseModel):
    """Composition information for multi-intent queries."""
    mode: CompositionMode = Field(..., description="Composition mode")
    ordering: List[int] = Field(default_factory=list, description="Intent execution order")
    join_keys: List[JoinKey] = Field(default_factory=list, description="Join keys between intents")
    response_policy: str = Field(default="single_message", description="Response formatting policy")


class IntentClassificationResult(BaseModel):
    """Complete intent classification result matching the output schema."""
    schema_version: str = Field(default="intent-output/1.4.0-multi")
    request: QueryRequest = Field(..., description="Original request information")
    composition: Composition = Field(..., description="Multi-intent composition details")
    intents: List[Intent] = Field(..., description="Classified intents")
    
    @validator('intents')
    def validate_at_least_one_intent(cls, v):
        """Ensure at least one intent is present."""
        if not v:
            raise ValueError("At least one intent must be classified")
        return v


class ClauseSegment(BaseModel):
    """Represents a clause segment from the splitter."""
    text: str = Field(..., description="Clause text")
    start_offset: int = Field(..., description="Start position in original text")
    end_offset: int = Field(..., description="End position in original text")
    dependencies: List[int] = Field(default_factory=list, description="Dependent clause indices")


class LabelingFunctionVote(BaseModel):
    """Vote from a labeling function."""
    function_name: str = Field(..., description="Name of the labeling function")
    intent_code: str = Field(..., description="Voted intent code")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Vote confidence")


class EntityExtractionResult(BaseModel):
    """Result from entity extraction."""
    entities: Dict[str, Any] = Field(..., description="Extracted entities")
    entity_spans: List[Dict[str, Any]] = Field(default_factory=list, description="Entity span information")


class ClassificationConfig(BaseModel):
    """Configuration for the classification pipeline."""
    confidence_threshold: float = Field(default=0.7, description="Minimum confidence threshold")
    ood_threshold: float = Field(default=0.5, description="Out-of-distribution threshold")
    max_intents_per_query: int = Field(default=5, description="Maximum intents to extract")
    enable_knn_backstop: bool = Field(default=True, description="Enable KNN backstop")
    knn_k: int = Field(default=5, description="Number of neighbors for KNN")
    enable_spell_correction: bool = Field(default=True, description="Enable spelling correction")
    
    
class TaxonomyClass(BaseModel):
    """Represents a taxonomy class from taxonomy.json."""
    code: str
    name: str
    description: str
    subclasses: List[Dict[str, Any]]
    
    
class TaxonomyEntity(BaseModel):
    """Represents an entity definition from taxonomy."""
    key: str
    type: str
    pattern: Optional[str] = None
    example: Optional[Any] = None
    enum: Optional[List[str]] = None
