"""
Tests for the KNN backstop module.
"""

import pytest
import numpy as np
from intent_classifier.core.knn_backstop import (
    KNNBackstop, KNNConfig, QueryExample, KNNMatch
)


class TestKNNBackstop:
    """Test the KNNBackstop class."""
    
    @pytest.fixture
    def config(self):
        """Create KNN configuration."""
        return KNNConfig(
            n_neighbors=3,
            min_similarity=0.3,
            use_entity_similarity=True,
            entity_weight=0.3,
            text_weight=0.7
        )
    
    @pytest.fixture
    def sample_examples(self):
        """Create sample query examples."""
        return [
            QueryExample(
                text="What is the status of RFI 1838?",
                preprocessed_text="what is the status of rfi 1838",
                entities={"ids.rfi": 1838},
                intent_code="STAT:RFI_STATUS",
                confidence=0.9,
                source="training"
            ),
            QueryExample(
                text="Show me the tile submittal",
                preprocessed_text="show me the tile submittal",
                entities={"submittal_topic": "tile"},
                intent_code="DOC:SUBMITTAL_RETRIEVE",
                confidence=0.85,
                source="training"
            ),
            QueryExample(
                text="RFI 123 status update",
                preprocessed_text="rfi 123 status update",
                entities={"ids.rfi": 123},
                intent_code="STAT:RFI_STATUS",
                confidence=0.88,
                source="validated"
            ),
            QueryExample(
                text="Find submittal for flooring",
                preprocessed_text="find submittal for flooring",
                entities={"submittal_topic": "flooring"},
                intent_code="DOC:SUBMITTAL_RETRIEVE",
                confidence=0.82,
                source="training"
            ),
            QueryExample(
                text="How many RFIs are open?",
                preprocessed_text="how many rfis are open",
                entities={},
                intent_code="COUNT:RFI_COUNT",
                confidence=0.8,
                source="training"
            )
        ]
    
    def test_initialization(self, config):
        """Test KNN backstop initialization."""
        knn = KNNBackstop(config)
        
        assert knn.config == config
        assert len(knn.examples) == 0
        assert not knn.is_indexed
        assert knn.tfidf_vectorizer is not None
    
    def test_add_examples(self, config, sample_examples):
        """Test adding examples to the index."""
        knn = KNNBackstop(config)
        
        knn.add_examples(sample_examples)
        
        assert len(knn.examples) == len(sample_examples)
        assert len(knn.entity_keys) > 0
        assert not knn.is_indexed  # Not indexed yet
    
    def test_build_index(self, config, sample_examples):
        """Test building the KNN index."""
        knn = KNNBackstop(config)
        knn.add_examples(sample_examples)
        
        knn.build_index()
        
        assert knn.is_indexed
        assert knn.text_vectors is not None
        assert knn.knn_model is not None
    
    def test_find_similar(self, config, sample_examples):
        """Test finding similar queries."""
        knn = KNNBackstop(config)
        knn.add_examples(sample_examples)
        knn.build_index()
        
        # Test with similar RFI query
        matches = knn.find_similar(
            "what is rfi 999 status",
            {"ids.rfi": 999}
        )
        
        assert len(matches) > 0
        assert len(matches) <= config.n_neighbors
        
        # Should find RFI status queries as most similar
        top_match = matches[0]
        assert top_match.example.intent_code == "STAT:RFI_STATUS"
        assert top_match.similarity > config.min_similarity
    
    def test_entity_similarity(self, config):
        """Test entity similarity calculation."""
        knn = KNNBackstop(config)
        
        # Identical entities
        sim1 = knn._calculate_entity_similarity(
            {"ids.rfi": 123, "floor": "Level 5"},
            {"ids.rfi": 123, "floor": "Level 5"}
        )
        assert sim1 == 1.0
        
        # Partial match
        sim2 = knn._calculate_entity_similarity(
            {"ids.rfi": 123, "floor": "Level 5"},
            {"ids.rfi": 123}
        )
        assert 0 < sim2 < 1
        
        # No match
        sim3 = knn._calculate_entity_similarity(
            {"ids.rfi": 123},
            {"ids.submittal": 456}
        )
        assert sim3 == 0.0
        
        # Empty entities
        sim4 = knn._calculate_entity_similarity({}, {})
        assert sim4 == 1.0
    
    def test_get_intent_prediction(self, config, sample_examples):
        """Test intent prediction using KNN."""
        knn = KNNBackstop(config)
        knn.add_examples(sample_examples)
        knn.build_index()
        
        # Test without classifier prediction
        intent, conf, source = knn.get_intent_prediction(
            "rfi 456 current status",
            {"ids.rfi": 456}
        )
        
        assert intent == "STAT:RFI_STATUS"
        assert conf > 0
        assert source == "knn"
        
        # Test with low confidence classifier prediction
        intent, conf, source = knn.get_intent_prediction(
            "rfi 456 current status",
            {"ids.rfi": 456},
            classifier_prediction=("STAT:RFI_STATUS", 0.4)
        )
        
        assert intent == "STAT:RFI_STATUS"
        assert conf > 0.4  # Should be boosted
        assert source == "knn_boosted"
        
        # Test with high confidence classifier prediction
        intent, conf, source = knn.get_intent_prediction(
            "something unrelated",
            {},
            classifier_prediction=("OTHER:INTENT", 0.9)
        )
        
        assert intent == "OTHER:INTENT"
        assert conf == 0.9
        assert source == "classifier"
    
    def test_add_feedback_example(self, config):
        """Test adding feedback examples."""
        knn = KNNBackstop(config)
        
        # Add initial examples
        knn.add_examples([
            QueryExample(
                text="test query",
                preprocessed_text="test query",
                entities={},
                intent_code="TEST:INTENT",
                confidence=0.8,
                source="training"
            )
        ])
        
        initial_count = len(knn.examples)
        
        # Add feedback
        knn.add_feedback_example(
            text="User corrected query",
            preprocessed_text="user corrected query",
            entities={"custom": "entity"},
            intent_code="CORRECTED:INTENT"
        )
        
        assert len(knn.examples) == initial_count + 1
        assert knn.examples[-1].source == "feedback"
        assert knn.examples[-1].intent_code == "CORRECTED:INTENT"
        assert knn.is_indexed  # Should rebuild index
    
    def test_prune_examples(self, config):
        """Test pruning examples."""
        knn = KNNBackstop(config)
        
        # Add many examples for same intent
        examples = []
        for i in range(10):
            examples.append(QueryExample(
                text=f"query {i}",
                preprocessed_text=f"query {i}",
                entities={},
                intent_code="TEST:INTENT",
                confidence=0.5 + i * 0.05,
                source="training" if i < 5 else "feedback"
            ))
        
        knn.add_examples(examples)
        
        # Prune to max 5 per intent
        knn.prune_examples(max_per_intent=5)
        
        assert len(knn.examples) == 5
        # Should prioritize feedback examples
        feedback_count = sum(1 for ex in knn.examples if ex.source == "feedback")
        assert feedback_count == 5
    
    def test_statistics(self, config, sample_examples):
        """Test getting statistics."""
        knn = KNNBackstop(config)
        knn.add_examples(sample_examples)
        knn.build_index()
        
        stats = knn.get_statistics()
        
        assert stats['total_examples'] == len(sample_examples)
        assert stats['indexed'] == True
        assert stats['unique_intents'] == 3  # RFI_STATUS, SUBMITTAL_RETRIEVE, RFI_COUNT
        assert 'source_distribution' in stats
        assert 'intent_distribution' in stats
        assert stats['source_distribution']['training'] == 4
        assert stats['source_distribution']['validated'] == 1
    
    def test_save_load(self, config, sample_examples, tmp_path):
        """Test saving and loading KNN index."""
        knn = KNNBackstop(config)
        knn.add_examples(sample_examples)
        knn.build_index()
        
        # Save
        save_path = tmp_path / "knn_index.pkl"
        knn.save(save_path)
        
        assert save_path.exists()
        
        # Load
        loaded_knn = KNNBackstop.load(save_path)
        
        assert len(loaded_knn.examples) == len(sample_examples)
        assert loaded_knn.is_indexed
        assert loaded_knn.config.n_neighbors == config.n_neighbors
        
        # Test that loaded index works
        matches = loaded_knn.find_similar(
            "rfi status check",
            {"ids.rfi": 789}
        )
        
        assert len(matches) > 0
    
    def test_edge_cases(self, config):
        """Test edge cases."""
        knn = KNNBackstop(config)
        
        # Empty index
        matches = knn.find_similar("test query", {})
        assert len(matches) == 0
        
        intent, conf, source = knn.get_intent_prediction("test", {})
        assert intent == "UNKNOWN"
        assert conf == 0.0
        assert source == "none"
        
        # Single example
        knn.add_examples([QueryExample(
            text="single",
            preprocessed_text="single",
            entities={},
            intent_code="SINGLE",
            confidence=1.0,
            source="training"
        )])
        
        knn.build_index()
        matches = knn.find_similar("test", {})
        assert len(matches) <= 1
    
    def test_min_similarity_filtering(self, config, sample_examples):
        """Test that min_similarity threshold works."""
        config.min_similarity = 0.8  # High threshold
        knn = KNNBackstop(config)
        knn.add_examples(sample_examples)
        knn.build_index()
        
        # Query that's somewhat different
        matches = knn.find_similar(
            "completely different query about something else",
            {"unknown": "entity"}
        )
        
        # Should have few or no matches due to high threshold
        assert all(match.similarity >= config.min_similarity for match in matches)
