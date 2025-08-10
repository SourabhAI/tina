"""
Basic tests for the intent classification pipeline.
"""

import pytest
from intent_classifier.main import IntentClassificationPipeline
from intent_classifier.models.schemas import ClassificationConfig


class TestPipeline:
    """Test the main pipeline functionality."""
    
    @pytest.fixture
    def pipeline(self):
        """Create a pipeline instance for testing."""
        config = ClassificationConfig(
            confidence_threshold=0.7,
            enable_knn_backstop=True
        )
        return IntentClassificationPipeline(config)
    
    def test_single_intent_query(self, pipeline):
        """Test classification of a single-intent query."""
        query = "What spec section covers fire rated doors?"
        result = pipeline.classify(query)
        
        assert len(result.intents) == 1
        assert result.intents[0].coarse_class == "SPEC"
        assert "SPEC:" in result.intents[0].intent_code
        assert result.composition.mode == "SINGLE"
    
    def test_multi_intent_query(self, pipeline):
        """Test classification of a multi-intent query."""
        query = "Show me the tile submittal and is it approved?"
        result = pipeline.classify(query)
        
        assert len(result.intents) == 2
        assert result.intents[0].coarse_class == "DOC"
        assert result.intents[1].coarse_class == "STAT"
        assert result.composition.mode in ["SEQUENCE", "PARALLEL"]
    
    def test_entity_extraction(self, pipeline):
        """Test entity extraction in queries."""
        query = "What is the status of RFI #1838?"
        result = pipeline.classify(query)
        
        assert len(result.intents) >= 1
        intent = result.intents[0]
        assert "ids.rfi" in intent.entities or "rfi" in intent.entities
        assert intent.entities.get("ids.rfi") == 1838 or intent.entities.get("rfi") == "1838"
    
    def test_unknown_query(self, pipeline):
        """Test handling of unknown/unclear queries."""
        query = "random text that doesn't make sense"
        result = pipeline.classify(query)
        
        # Should still return a result, possibly with low confidence
        assert len(result.intents) >= 1
        # Check if it's marked as unknown or has very low confidence
        intent = result.intents[0]
        assert intent.confidence < 0.5 or intent.intent_code == "UNKNOWN:GENERAL"
