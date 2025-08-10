"""
Supervised classifier module for intent classification.
Uses weak supervision labels and features to train classification models.
"""

import logging
import pickle
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass
from collections import Counter, defaultdict
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score, StratifiedKFold
import joblib

from intent_classifier.models.schemas import Intent, LabelingFunctionVote
from intent_classifier.core.entity_extractor import EntityExtractor
from intent_classifier.core.labeling_functions import LabelingFunctions


logger = logging.getLogger(__name__)


@dataclass
class ClassifierConfig:
    """Configuration for the supervised classifier."""
    model_type: str = "logistic"  # "logistic", "svm", "random_forest"
    use_tfidf: bool = True
    use_entity_features: bool = True
    use_linguistic_features: bool = True
    tfidf_max_features: int = 5000
    tfidf_ngram_range: Tuple[int, int] = (1, 3)
    confidence_threshold: float = 0.7
    max_iter: int = 1000
    random_state: int = 42
    cross_validation_folds: int = 5


class EntityFeatureExtractor(BaseEstimator, TransformerMixin):
    """Extract entity-based features for classification."""
    
    def __init__(self, entity_keys: List[str]):
        self.entity_keys = entity_keys
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        """Transform list of entity dicts to feature matrix."""
        features = []
        
        for entities in X:
            feature_vec = []
            # Binary features for entity presence
            for key in self.entity_keys:
                feature_vec.append(1.0 if key in entities else 0.0)
            
            # Count features
            feature_vec.append(len(entities))  # Total entity count
            
            # Specific entity type counts
            id_count = sum(1 for k in entities if k.startswith('ids.'))
            feature_vec.append(id_count)
            
            features.append(feature_vec)
        
        return np.array(features)


class LinguisticFeatureExtractor(BaseEstimator, TransformerMixin):
    """Extract linguistic features from text."""
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        """Extract linguistic features from text."""
        features = []
        
        for text in X:
            feature_vec = []
            
            # Length features
            feature_vec.append(len(text))  # Character count
            feature_vec.append(len(text.split()))  # Word count
            
            # Question features
            feature_vec.append(1.0 if text.strip().endswith('?') else 0.0)
            feature_vec.append(1.0 if text.lower().startswith(('what', 'where', 'when', 'who', 'why', 'how')) else 0.0)
            
            # Imperative features
            feature_vec.append(1.0 if text.lower().startswith(('show', 'find', 'get', 'list', 'display')) else 0.0)
            
            # Keyword presence
            keywords = ['status', 'count', 'submittal', 'rfi', 'spec', 'drawing', 'schedule']
            for keyword in keywords:
                feature_vec.append(1.0 if keyword in text.lower() else 0.0)
            
            features.append(feature_vec)
        
        return np.array(features)


class IntentClassifier:
    """
    Supervised classifier for intent classification.
    Trains on weak supervision labels and uses multiple feature types.
    """
    
    def __init__(self, config: ClassifierConfig, entity_keys: List[str]):
        """
        Initialize the classifier.
        
        Args:
            config: Classifier configuration
            entity_keys: List of possible entity keys
        """
        self.config = config
        self.entity_keys = entity_keys
        self.model = None
        self.label_encoder = {}
        self.label_decoder = {}
        self.feature_pipeline = None
        self.is_trained = False
        
        # Initialize components
        self._build_feature_pipeline()
        self._build_model()
    
    def _build_feature_pipeline(self):
        """Build the feature extraction pipeline."""
        feature_extractors = []
        
        # TF-IDF features
        if self.config.use_tfidf:
            tfidf = TfidfVectorizer(
                max_features=self.config.tfidf_max_features,
                ngram_range=self.config.tfidf_ngram_range,
                min_df=2,
                max_df=0.95,
                sublinear_tf=True,
                use_idf=True
            )
            feature_extractors.append(('tfidf', tfidf))
        
        # Entity features
        if self.config.use_entity_features:
            entity_extractor = Pipeline([
                ('extractor', EntityFeatureExtractor(self.entity_keys)),
                ('scaler', StandardScaler())
            ])
            feature_extractors.append(('entities', entity_extractor))
        
        # Linguistic features
        if self.config.use_linguistic_features:
            linguistic_extractor = Pipeline([
                ('extractor', LinguisticFeatureExtractor()),
                ('scaler', StandardScaler())
            ])
            feature_extractors.append(('linguistic', linguistic_extractor))
        
        # Combine all features
        self.feature_pipeline = FeatureUnion(feature_extractors)
    
    def _build_model(self):
        """Build the classification model."""
        if self.config.model_type == "logistic":
            self.model = LogisticRegression(
                max_iter=self.config.max_iter,
                random_state=self.config.random_state,
                class_weight='balanced',
                multi_class='multinomial',
                solver='lbfgs'
            )
        elif self.config.model_type == "svm":
            self.model = LinearSVC(
                max_iter=self.config.max_iter,
                random_state=self.config.random_state,
                class_weight='balanced',
                dual=False
            )
        elif self.config.model_type == "random_forest":
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=20,
                random_state=self.config.random_state,
                class_weight='balanced',
                n_jobs=-1
            )
        else:
            raise ValueError(f"Unknown model type: {self.config.model_type}")
    
    def prepare_training_data(self, queries: List[Dict[str, Any]], 
                            labeling_functions: LabelingFunctions) -> Tuple[List, List, List]:
        """
        Prepare training data using weak supervision.
        
        Args:
            queries: List of query dictionaries with 'text' and 'entities'
            labeling_functions: Labeling functions instance
            
        Returns:
            Tuple of (texts, entities_list, labels)
        """
        texts = []
        entities_list = []
        labels = []
        
        for query in queries:
            text = query['text']
            entities = query.get('entities', {})
            
            # Get votes from labeling functions
            votes = labeling_functions.apply_all(text, entities)
            
            if votes:
                # Get top prediction
                top_predictions = labeling_functions.get_top_predictions(votes, top_k=1)
                if top_predictions and top_predictions[0][1] >= 0.6:  # Minimum confidence
                    intent_code = top_predictions[0][0]
                    texts.append(text)
                    entities_list.append(entities)
                    labels.append(intent_code)
        
        # Encode labels
        unique_labels = sorted(set(labels))
        self.label_encoder = {label: i for i, label in enumerate(unique_labels)}
        self.label_decoder = {i: label for label, i in self.label_encoder.items()}
        
        encoded_labels = [self.label_encoder[label] for label in labels]
        
        logger.info(f"Prepared {len(texts)} training examples with {len(unique_labels)} unique labels")
        
        return texts, entities_list, encoded_labels
    
    def train(self, texts: List[str], entities_list: List[Dict[str, Any]], 
              labels: List[int], validate: bool = True):
        """
        Train the classifier.
        
        Args:
            texts: List of preprocessed texts
            entities_list: List of entity dictionaries
            labels: List of encoded labels
            validate: Whether to perform cross-validation
        """
        if not texts:
            raise ValueError("No training data provided")
        
        # Prepare features
        logger.info("Extracting features...")
        
        # For TF-IDF
        if self.config.use_tfidf:
            tfidf_features = self.feature_pipeline.transformer_list[0][1].fit_transform(texts)
        
        # For entity features
        if self.config.use_entity_features:
            entity_transformer = self.feature_pipeline.transformer_list[1][1]
            entity_features = entity_transformer.fit_transform(entities_list)
        
        # For linguistic features
        if self.config.use_linguistic_features:
            linguistic_transformer = self.feature_pipeline.transformer_list[2][1]
            linguistic_features = linguistic_transformer.fit_transform(texts)
        
        # Combine features based on configuration
        feature_list = []
        if self.config.use_tfidf:
            feature_list.append(tfidf_features)
        if self.config.use_entity_features:
            feature_list.append(entity_features)
        if self.config.use_linguistic_features:
            feature_list.append(linguistic_features)
        
        # Stack features horizontally
        if len(feature_list) > 1:
            from scipy.sparse import hstack
            X = hstack(feature_list)
        else:
            X = feature_list[0]
        
        y = np.array(labels)
        
        # Cross-validation
        if validate and len(np.unique(y)) > 1:
            # Determine the minimum samples per class
            from collections import Counter
            class_counts = Counter(y)
            min_samples = min(class_counts.values())
            
            # Only do CV if we have enough samples
            if min_samples >= 2:
                logger.info("Performing cross-validation...")
                # Use fewer splits if needed
                n_splits = min(self.config.cross_validation_folds, min_samples)
                cv_scores = cross_val_score(
                    self.model, X, y,
                    cv=StratifiedKFold(n_splits=n_splits),
                    scoring='f1_macro'
                )
                logger.info(f"Cross-validation F1 scores: {cv_scores}")
                logger.info(f"Mean CV F1: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
            else:
                logger.warning("Skipping cross-validation: not enough samples per class")
        
        # Train final model
        logger.info("Training final model...")
        self.model.fit(X, y)
        
        # Training accuracy
        train_pred = self.model.predict(X)
        train_accuracy = (train_pred == y).mean()
        logger.info(f"Training accuracy: {train_accuracy:.3f}")
        
        self.is_trained = True
        
        # Print label distribution
        label_counts = Counter(labels)
        logger.info("Label distribution in training data:")
        for label_id, count in label_counts.most_common():
            intent = self.label_decoder[label_id]
            logger.info(f"  {intent}: {count} examples")
    
    def predict(self, text: str, entities: Dict[str, Any]) -> Tuple[str, float]:
        """
        Predict intent for a single query.
        
        Args:
            text: Preprocessed query text
            entities: Extracted entities
            
        Returns:
            Tuple of (intent_code, confidence)
        """
        if not self.is_trained:
            raise RuntimeError("Classifier must be trained before prediction")
        
        # Prepare features
        features = self._extract_features([text], [entities])
        
        # Get prediction
        if hasattr(self.model, 'predict_proba'):
            proba = self.model.predict_proba(features)[0]
            pred_idx = np.argmax(proba)
            confidence = proba[pred_idx]
        else:
            # For SVM without probability
            pred_idx = self.model.predict(features)[0]
            # Use decision function for confidence
            if hasattr(self.model, 'decision_function'):
                decision = self.model.decision_function(features)[0]
                if len(self.label_decoder) == 2:
                    confidence = 1 / (1 + np.exp(-abs(decision)))  # Sigmoid
                else:
                    confidence = np.max(decision) / np.sum(np.abs(decision))
            else:
                confidence = 1.0
        
        intent_code = self.label_decoder[pred_idx]
        
        return intent_code, confidence
    
    def predict_batch(self, texts: List[str], entities_list: List[Dict[str, Any]]) -> List[Tuple[str, float]]:
        """
        Predict intents for multiple queries.
        
        Args:
            texts: List of preprocessed texts
            entities_list: List of entity dictionaries
            
        Returns:
            List of (intent_code, confidence) tuples
        """
        if not self.is_trained:
            raise RuntimeError("Classifier must be trained before prediction")
        
        # Prepare features
        features = self._extract_features(texts, entities_list)
        
        # Get predictions
        predictions = []
        
        if hasattr(self.model, 'predict_proba'):
            probas = self.model.predict_proba(features)
            for proba in probas:
                pred_idx = np.argmax(proba)
                confidence = proba[pred_idx]
                intent_code = self.label_decoder[pred_idx]
                predictions.append((intent_code, confidence))
        else:
            # For models without probability
            preds = self.model.predict(features)
            for i, pred_idx in enumerate(preds):
                intent_code = self.label_decoder[pred_idx]
                # Simple confidence estimate
                confidence = 0.8 if hasattr(self.model, 'decision_function') else 1.0
                predictions.append((intent_code, confidence))
        
        return predictions
    
    def _extract_features(self, texts: List[str], entities_list: List[Dict[str, Any]]):
        """Extract features for prediction."""
        feature_list = []
        
        # TF-IDF features
        if self.config.use_tfidf:
            tfidf_transformer = self.feature_pipeline.transformer_list[0][1]
            tfidf_features = tfidf_transformer.transform(texts)
            feature_list.append(tfidf_features)
        
        # Entity features
        if self.config.use_entity_features:
            entity_pipeline = self.feature_pipeline.transformer_list[1][1]
            entity_features = entity_pipeline.transform(entities_list)
            feature_list.append(entity_features)
        
        # Linguistic features
        if self.config.use_linguistic_features:
            linguistic_pipeline = self.feature_pipeline.transformer_list[2][1]
            linguistic_features = linguistic_pipeline.transform(texts)
            feature_list.append(linguistic_features)
        
        # Combine features
        if len(feature_list) > 1:
            from scipy.sparse import hstack
            return hstack(feature_list)
        else:
            return feature_list[0]
    
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """
        Get feature importance if available.
        
        Returns:
            Dictionary mapping feature names to importance scores
        """
        if not self.is_trained:
            return None
        
        if self.config.model_type == "random_forest":
            # Get feature names
            feature_names = []
            
            if self.config.use_tfidf:
                tfidf = self.feature_pipeline.transformer_list[0][1]
                feature_names.extend([f"tfidf_{name}" for name in tfidf.get_feature_names_out()])
            
            if self.config.use_entity_features:
                entity_names = [f"entity_{key}" for key in self.entity_keys]
                entity_names.extend(['entity_count', 'id_count'])
                feature_names.extend(entity_names)
            
            if self.config.use_linguistic_features:
                ling_names = ['char_count', 'word_count', 'is_question', 'starts_question', 
                            'starts_imperative', 'has_status', 'has_count', 'has_submittal',
                            'has_rfi', 'has_spec', 'has_drawing', 'has_schedule']
                feature_names.extend([f"ling_{name}" for name in ling_names])
            
            # Get importances
            importances = self.model.feature_importances_
            
            # Create importance dict
            importance_dict = {}
            for name, importance in zip(feature_names, importances):
                importance_dict[name] = float(importance)
            
            # Sort by importance
            sorted_importance = dict(sorted(importance_dict.items(), 
                                         key=lambda x: x[1], reverse=True))
            
            return sorted_importance
        
        return None
    
    def save(self, model_path: Union[str, Path]):
        """
        Save the trained model.
        
        Args:
            model_path: Path to save the model
        """
        if not self.is_trained:
            raise RuntimeError("Cannot save untrained model")
        
        model_path = Path(model_path)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save model components
        model_data = {
            'config': self.config,
            'entity_keys': self.entity_keys,
            'label_encoder': self.label_encoder,
            'label_decoder': self.label_decoder,
            'model': self.model,
            'feature_pipeline': self.feature_pipeline
        }
        
        joblib.dump(model_data, model_path)
        logger.info(f"Model saved to {model_path}")
    
    @classmethod
    def load(cls, model_path: Union[str, Path]) -> 'IntentClassifier':
        """
        Load a saved model.
        
        Args:
            model_path: Path to the saved model
            
        Returns:
            Loaded IntentClassifier instance
        """
        model_path = Path(model_path)
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Load model data
        model_data = joblib.load(model_path)
        
        # Create classifier instance
        classifier = cls(model_data['config'], model_data['entity_keys'])
        
        # Restore components
        classifier.label_encoder = model_data['label_encoder']
        classifier.label_decoder = model_data['label_decoder']
        classifier.model = model_data['model']
        classifier.feature_pipeline = model_data['feature_pipeline']
        classifier.is_trained = True
        
        logger.info(f"Model loaded from {model_path}")
        
        return classifier
    
    def evaluate(self, texts: List[str], entities_list: List[Dict[str, Any]], 
                true_labels: List[str]) -> Dict[str, Any]:
        """
        Evaluate the classifier on test data.
        
        Args:
            texts: List of test texts
            entities_list: List of entity dictionaries
            true_labels: List of true intent codes
            
        Returns:
            Dictionary with evaluation metrics
        """
        if not self.is_trained:
            raise RuntimeError("Classifier must be trained before evaluation")
        
        # Get predictions
        predictions = self.predict_batch(texts, entities_list)
        pred_labels = [pred[0] for pred in predictions]
        confidences = [pred[1] for pred in predictions]
        
        # Calculate metrics
        from sklearn.metrics import accuracy_score, precision_recall_fscore_support
        
        accuracy = accuracy_score(true_labels, pred_labels)
        precision, recall, f1, support = precision_recall_fscore_support(
            true_labels, pred_labels, average='weighted', zero_division=0
        )
        
        # Per-class metrics
        class_report = classification_report(
            true_labels, pred_labels, 
            output_dict=True, zero_division=0
        )
        
        # Confidence statistics
        conf_stats = {
            'mean': np.mean(confidences),
            'std': np.std(confidences),
            'min': np.min(confidences),
            'max': np.max(confidences)
        }
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confidence_stats': conf_stats,
            'classification_report': class_report,
            'num_samples': len(texts)
        }
