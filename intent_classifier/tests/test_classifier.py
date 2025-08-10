"""
Tests for the supervised classifier module.
"""

import pytest
import numpy as np
from intent_classifier.core.classifier import (
    IntentClassifier, ClassifierConfig, 
    EntityFeatureExtractor, LinguisticFeatureExtractor
)
from intent_classifier.core.labeling_functions import LabelingFunctions
from intent_classifier.core.taxonomy_loader import TaxonomyLoader
from intent_classifier.core.entity_extractor import EntityExtractor


class TestFeatureExtractors:
    """Test feature extraction components."""
    
    def test_entity_feature_extractor(self):
        """Test entity feature extraction."""
        entity_keys = ['ids.rfi', 'ids.submittal', 'spec_section', 'floor']
        extractor = EntityFeatureExtractor(entity_keys)
        
        entities_list = [
            {'ids.rfi': 123, 'floor': 'Level 5'},
            {'ids.submittal': 456},
            {},
            {'spec_section': '10 22 33', 'ids.rfi': 789, 'floor': 'Basement'}
        ]
        
        features = extractor.transform(entities_list)
        
        # Check shape
        assert features.shape == (4, 6)  # 4 samples, 4 entity features + 2 count features
        
        # Check first sample
        assert features[0][0] == 1.0  # has ids.rfi
        assert features[0][1] == 0.0  # no ids.submittal
        assert features[0][2] == 0.0  # no spec_section
        assert features[0][3] == 1.0  # has floor
        assert features[0][4] == 2.0  # total entity count
        assert features[0][5] == 1.0  # id count
        
        # Check empty entities
        assert features[2][4] == 0.0  # no entities
    
    def test_linguistic_feature_extractor(self):
        """Test linguistic feature extraction."""
        extractor = LinguisticFeatureExtractor()
        
        texts = [
            "What is the status of RFI 1838?",
            "Show me all submittals",
            "Find the mechanical drawings",
            "Status",
            "How many RFIs are open?"
        ]
        
        features = extractor.transform(texts)
        
        # Check shape
        assert features.shape[0] == 5  # 5 samples
        assert features.shape[1] >= 10  # At least 10 features
        
        # Check question features
        assert features[0][2] == 1.0  # ends with ?
        assert features[0][3] == 1.0  # starts with question word
        assert features[1][4] == 1.0  # starts with imperative
        
        # Check keyword features
        status_idx = 5  # Index for 'status' keyword
        assert features[0][status_idx] == 1.0  # has 'status'
        assert features[3][status_idx] == 1.0  # 'Status' query


class TestIntentClassifier:
    """Test the IntentClassifier class."""
    
    @pytest.fixture
    def mock_labeling_functions(self):
        """Create mock labeling functions."""
        class MockLabelingFunctions:
            def apply_all(self, text, entities):
                # Simple mock that returns predictable labels
                if 'submittal' in text.lower():
                    from intent_classifier.models.schemas import LabelingFunctionVote
                    return [LabelingFunctionVote(
                        function_name='mock',
                        intent_code='DOC:SUBMITTAL_RETRIEVE',
                        confidence=0.9
                    )]
                elif 'rfi' in text.lower():
                    from intent_classifier.models.schemas import LabelingFunctionVote
                    return [LabelingFunctionVote(
                        function_name='mock',
                        intent_code='DOC:RFI_RETRIEVE',
                        confidence=0.8
                    )]
                return []
            
            def get_top_predictions(self, votes, top_k=1):
                if votes:
                    return [(votes[0].intent_code, votes[0].confidence)]
                return []
        
        return MockLabelingFunctions()
    
    @pytest.fixture
    def classifier_config(self):
        """Create classifier configuration."""
        return ClassifierConfig(
            model_type="logistic",
            use_tfidf=True,
            use_entity_features=True,
            use_linguistic_features=True,
            tfidf_max_features=100,
            tfidf_ngram_range=(1, 2),
            confidence_threshold=0.7
        )
    
    @pytest.fixture
    def entity_keys(self):
        """Sample entity keys."""
        return ['ids.rfi', 'ids.submittal', 'spec_section', 'floor']
    
    def test_initialization(self, classifier_config, entity_keys):
        """Test classifier initialization."""
        classifier = IntentClassifier(classifier_config, entity_keys)
        
        assert classifier.config == classifier_config
        assert classifier.entity_keys == entity_keys
        assert not classifier.is_trained
        assert classifier.model is not None
        assert classifier.feature_pipeline is not None
    
    def test_prepare_training_data(self, classifier_config, entity_keys, mock_labeling_functions):
        """Test training data preparation."""
        classifier = IntentClassifier(classifier_config, entity_keys)
        
        queries = [
            {'text': 'show me the tile submittal', 'entities': {'submittal_topic': 'tile'}},
            {'text': 'find RFI 123', 'entities': {'ids.rfi': 123}},
            {'text': 'what is the status', 'entities': {}},  # No votes
            {'text': 'submittal for flooring', 'entities': {'submittal_topic': 'flooring'}},
        ]
        
        texts, entities_list, labels = classifier.prepare_training_data(
            queries, mock_labeling_functions
        )
        
        # Should have 3 examples (one filtered out due to no votes)
        assert len(texts) == 3
        assert len(entities_list) == 3
        assert len(labels) == 3
        
        # Check label encoding
        assert len(classifier.label_encoder) == 2  # Two unique labels
        assert 'DOC:SUBMITTAL_RETRIEVE' in classifier.label_encoder
        assert 'DOC:RFI_RETRIEVE' in classifier.label_encoder
        
        # Check encoded labels
        assert all(isinstance(label, int) for label in labels)
    
    def test_train(self, classifier_config, entity_keys):
        """Test classifier training."""
        classifier = IntentClassifier(classifier_config, entity_keys)
        
        # Prepare simple training data
        texts = [
            'show me the tile submittal',
            'submittal for flooring',
            'find RFI 123',
            'RFI 456 details',
            'mechanical submittal status',
            'electrical RFI response'
        ]
        
        entities_list = [
            {'submittal_topic': 'tile'},
            {'submittal_topic': 'flooring'},
            {'ids.rfi': 123},
            {'ids.rfi': 456},
            {'submittal_topic': 'mechanical'},
            {'ids.rfi': 789}
        ]
        
        # Create labels
        labels = [0, 0, 1, 1, 0, 1]  # 0: submittal, 1: rfi
        
        # Set up label encoding
        classifier.label_encoder = {
            'DOC:SUBMITTAL_RETRIEVE': 0,
            'DOC:RFI_RETRIEVE': 1
        }
        classifier.label_decoder = {
            0: 'DOC:SUBMITTAL_RETRIEVE',
            1: 'DOC:RFI_RETRIEVE'
        }
        
        # Train without validation (too few samples)
        classifier.train(texts, entities_list, labels, validate=False)
        
        assert classifier.is_trained
    
    def test_predict(self, classifier_config, entity_keys):
        """Test prediction on single query."""
        classifier = IntentClassifier(classifier_config, entity_keys)
        
        # Train a simple model first
        texts = ['submittal one', 'submittal two', 'rfi one', 'rfi two']
        entities_list = [{}, {}, {}, {}]
        labels = [0, 0, 1, 1]
        
        classifier.label_encoder = {'SUBMITTAL': 0, 'RFI': 1}
        classifier.label_decoder = {0: 'SUBMITTAL', 1: 'RFI'}
        
        classifier.train(texts, entities_list, labels, validate=False)
        
        # Test prediction
        intent, confidence = classifier.predict('new submittal query', {})
        
        assert intent in ['SUBMITTAL', 'RFI']
        assert 0.0 <= confidence <= 1.0
    
    def test_predict_batch(self, classifier_config, entity_keys):
        """Test batch prediction."""
        classifier = IntentClassifier(classifier_config, entity_keys)
        
        # Train a simple model
        texts = ['submittal' * 10, 'another submittal', 'rfi query', 'rfi document']
        entities_list = [{}, {}, {}, {}]
        labels = [0, 0, 1, 1]
        
        classifier.label_encoder = {'SUBMITTAL': 0, 'RFI': 1}
        classifier.label_decoder = {0: 'SUBMITTAL', 1: 'RFI'}
        
        classifier.train(texts, entities_list, labels, validate=False)
        
        # Test batch prediction
        test_texts = ['submittal test', 'rfi test', 'unknown query']
        test_entities = [{}, {}, {}]
        
        predictions = classifier.predict_batch(test_texts, test_entities)
        
        assert len(predictions) == 3
        assert all(isinstance(pred, tuple) for pred in predictions)
        assert all(pred[0] in ['SUBMITTAL', 'RFI'] for pred in predictions)
        assert all(0.0 <= pred[1] <= 1.0 for pred in predictions)
    
    def test_different_model_types(self, entity_keys):
        """Test different model types."""
        for model_type in ['logistic', 'svm', 'random_forest']:
            config = ClassifierConfig(
                model_type=model_type,
                use_tfidf=True,
                use_entity_features=False,  # Simplify for testing
                use_linguistic_features=False
            )
            
            classifier = IntentClassifier(config, entity_keys)
            
            # Simple training
            texts = ['a', 'b', 'c', 'd']
            entities = [{}, {}, {}, {}]
            labels = [0, 0, 1, 1]
            
            classifier.label_encoder = {'A': 0, 'B': 1}
            classifier.label_decoder = {0: 'A', 1: 'B'}
            
            classifier.train(texts, entities, labels, validate=False)
            
            # Should be able to predict
            intent, conf = classifier.predict('test', {})
            assert intent in ['A', 'B']
    
    def test_feature_importance(self, entity_keys):
        """Test feature importance extraction."""
        config = ClassifierConfig(
            model_type='random_forest',
            use_tfidf=True,
            use_entity_features=True,
            use_linguistic_features=True,
            tfidf_max_features=10
        )
        
        classifier = IntentClassifier(config, entity_keys)
        
        # Before training
        assert classifier.get_feature_importance() is None
        
        # Train
        texts = ['submittal doc', 'rfi doc', 'submittal request', 'rfi query']
        entities = [{'ids.submittal': 1}, {'ids.rfi': 1}, {}, {'ids.rfi': 2}]
        labels = [0, 1, 0, 1]
        
        classifier.label_encoder = {'S': 0, 'R': 1}
        classifier.label_decoder = {0: 'S', 1: 'R'}
        
        classifier.train(texts, entities, labels, validate=False)
        
        # Get importance
        importance = classifier.get_feature_importance()
        
        assert importance is not None
        assert isinstance(importance, dict)
        assert len(importance) > 0
        assert all(isinstance(v, float) for v in importance.values())
    
    def test_save_load(self, classifier_config, entity_keys, tmp_path):
        """Test model saving and loading."""
        classifier = IntentClassifier(classifier_config, entity_keys)
        
        # Train
        texts = ['text1', 'text2', 'text3', 'text4']
        entities = [{}, {}, {}, {}]
        labels = [0, 0, 1, 1]
        
        classifier.label_encoder = {'A': 0, 'B': 1}
        classifier.label_decoder = {0: 'A', 1: 'B'}
        
        classifier.train(texts, entities, labels, validate=False)
        
        # Save
        model_path = tmp_path / "test_model.pkl"
        classifier.save(model_path)
        
        assert model_path.exists()
        
        # Load
        loaded_classifier = IntentClassifier.load(model_path)
        
        assert loaded_classifier.is_trained
        assert loaded_classifier.label_encoder == classifier.label_encoder
        assert loaded_classifier.label_decoder == classifier.label_decoder
        
        # Test prediction with loaded model
        intent, conf = loaded_classifier.predict('test query', {})
        assert intent in ['A', 'B']
    
    def test_evaluation(self, classifier_config, entity_keys):
        """Test model evaluation."""
        classifier = IntentClassifier(classifier_config, entity_keys)
        
        # Train
        train_texts = ['submittal a', 'submittal b', 'rfi x', 'rfi y']
        train_entities = [{}, {}, {}, {}]
        train_labels = [0, 0, 1, 1]
        
        classifier.label_encoder = {'SUBMITTAL': 0, 'RFI': 1}
        classifier.label_decoder = {0: 'SUBMITTAL', 1: 'RFI'}
        
        classifier.train(train_texts, train_entities, train_labels, validate=False)
        
        # Evaluate
        test_texts = ['submittal test', 'rfi test']
        test_entities = [{}, {}]
        true_labels = ['SUBMITTAL', 'RFI']
        
        metrics = classifier.evaluate(test_texts, test_entities, true_labels)
        
        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1_score' in metrics
        assert 'confidence_stats' in metrics
        assert 'classification_report' in metrics
        
        assert 0.0 <= metrics['accuracy'] <= 1.0
        assert metrics['num_samples'] == 2
