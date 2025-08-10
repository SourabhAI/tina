"""
KNN backstop module for handling out-of-distribution queries.
Uses nearest neighbor search to find similar queries when classifier confidence is low.
"""

import logging
import numpy as np
from typing import List, Dict, Any, Tuple, Optional, Union
from dataclasses import dataclass
from pathlib import Path
import json
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
import joblib

from intent_classifier.models.schemas import Intent


logger = logging.getLogger(__name__)


@dataclass
class KNNConfig:
    """Configuration for KNN backstop."""
    n_neighbors: int = 5
    min_similarity: float = 0.3
    use_entity_similarity: bool = True
    entity_weight: float = 0.3
    text_weight: float = 0.7
    max_features: int = 3000
    ngram_range: Tuple[int, int] = (1, 2)
    confidence_boost_threshold: float = 0.5


@dataclass
class QueryExample:
    """Represents a query example in the KNN index."""
    text: str
    preprocessed_text: str
    entities: Dict[str, Any]
    intent_code: str
    confidence: float
    source: str  # 'training', 'validated', 'feedback'


@dataclass
class KNNMatch:
    """Represents a KNN match result."""
    example: QueryExample
    similarity: float
    text_similarity: float
    entity_similarity: float


class KNNBackstop:
    """
    KNN-based backstop for handling queries with low classifier confidence.
    Maintains an index of query examples and finds similar ones.
    """
    
    def __init__(self, config: KNNConfig):
        """
        Initialize the KNN backstop.
        
        Args:
            config: KNN configuration
        """
        self.config = config
        self.examples: List[QueryExample] = []
        self.tfidf_vectorizer = None
        self.text_vectors = None
        self.knn_model = None
        self.entity_keys: List[str] = []
        self.is_indexed = False
        
        self._initialize_vectorizer()
    
    def _initialize_vectorizer(self):
        """Initialize the TF-IDF vectorizer."""
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=self.config.max_features,
            ngram_range=self.config.ngram_range,
            min_df=1,
            max_df=0.95,
            sublinear_tf=True,
            use_idf=True
        )
    
    def add_examples(self, examples: List[QueryExample]):
        """
        Add examples to the KNN index.
        
        Args:
            examples: List of query examples
        """
        self.examples.extend(examples)
        logger.info(f"Added {len(examples)} examples to KNN index")
        
        # Update entity keys
        all_entity_keys = set()
        for example in self.examples:
            all_entity_keys.update(example.entities.keys())
        self.entity_keys = sorted(all_entity_keys)
        
        # Mark as needing reindexing
        self.is_indexed = False
    
    def build_index(self):
        """Build or rebuild the KNN index."""
        if not self.examples:
            logger.warning("No examples to index")
            return
        
        logger.info(f"Building KNN index with {len(self.examples)} examples")
        
        # Extract texts for vectorization
        texts = [ex.preprocessed_text for ex in self.examples]
        
        # Fit and transform texts
        self.text_vectors = self.tfidf_vectorizer.fit_transform(texts)
        
        # Build KNN model
        self.knn_model = NearestNeighbors(
            n_neighbors=min(self.config.n_neighbors, len(self.examples)),
            metric='cosine',
            algorithm='brute'  # For cosine similarity
        )
        
        if self.config.use_entity_similarity:
            # Combine text and entity features
            entity_vectors = self._extract_entity_features(self.examples)
            
            # Normalize and combine
            text_vectors_normalized = self._normalize_vectors(self.text_vectors.toarray())
            entity_vectors_normalized = self._normalize_vectors(entity_vectors)
            
            combined_vectors = np.hstack([
                text_vectors_normalized * self.config.text_weight,
                entity_vectors_normalized * self.config.entity_weight
            ])
            
            self.knn_model.fit(combined_vectors)
        else:
            # Use text vectors only
            self.knn_model.fit(self.text_vectors)
        
        self.is_indexed = True
        logger.info("KNN index built successfully")
    
    def _extract_entity_features(self, examples: List[Union[QueryExample, Dict[str, Any]]]) -> np.ndarray:
        """
        Extract entity features from examples.
        
        Args:
            examples: List of examples or entity dictionaries
            
        Returns:
            Entity feature matrix
        """
        features = []
        
        for example in examples:
            if isinstance(example, QueryExample):
                entities = example.entities
            else:
                entities = example
            
            # Binary features for entity presence
            feature_vec = []
            for key in self.entity_keys:
                feature_vec.append(1.0 if key in entities else 0.0)
            
            # Add entity count
            feature_vec.append(len(entities))
            
            features.append(feature_vec)
        
        return np.array(features)
    
    def _normalize_vectors(self, vectors: np.ndarray) -> np.ndarray:
        """
        Normalize vectors to unit length.
        
        Args:
            vectors: Vector matrix
            
        Returns:
            Normalized vectors
        """
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1  # Avoid division by zero
        return vectors / norms
    
    def find_similar(self, text: str, entities: Dict[str, Any], 
                    k: Optional[int] = None) -> List[KNNMatch]:
        """
        Find similar queries in the index.
        
        Args:
            text: Preprocessed query text
            entities: Extracted entities
            k: Number of neighbors (overrides config)
            
        Returns:
            List of KNN matches
        """
        if not self.is_indexed:
            self.build_index()
        
        if not self.examples:
            return []
        
        k = k or self.config.n_neighbors
        k = min(k, len(self.examples))
        
        # Vectorize query text
        query_text_vector = self.tfidf_vectorizer.transform([text])
        
        if self.config.use_entity_similarity:
            # Extract entity features
            query_entity_features = self._extract_entity_features([entities])
            
            # Normalize and combine
            text_vec_normalized = self._normalize_vectors(query_text_vector.toarray())
            entity_vec_normalized = self._normalize_vectors(query_entity_features)
            
            query_vector = np.hstack([
                text_vec_normalized * self.config.text_weight,
                entity_vec_normalized * self.config.entity_weight
            ])
        else:
            query_vector = query_text_vector
        
        # Find neighbors
        distances, indices = self.knn_model.kneighbors(query_vector, n_neighbors=k)
        
        # Convert to similarity scores (1 - cosine distance)
        similarities = 1 - distances[0]
        
        # Calculate individual similarities for analysis
        text_similarities = cosine_similarity(query_text_vector, self.text_vectors).flatten()
        
        # Create matches
        matches = []
        for idx, similarity in zip(indices[0], similarities):
            if similarity >= self.config.min_similarity:
                # Calculate entity similarity
                entity_sim = 0.0
                if self.config.use_entity_similarity and entities:
                    entity_sim = self._calculate_entity_similarity(
                        entities, self.examples[idx].entities
                    )
                
                match = KNNMatch(
                    example=self.examples[idx],
                    similarity=similarity,
                    text_similarity=text_similarities[idx],
                    entity_similarity=entity_sim
                )
                matches.append(match)
        
        return matches
    
    def _calculate_entity_similarity(self, entities1: Dict[str, Any], 
                                   entities2: Dict[str, Any]) -> float:
        """
        Calculate similarity between two entity dictionaries.
        
        Args:
            entities1: First entity dictionary
            entities2: Second entity dictionary
            
        Returns:
            Similarity score (0-1)
        """
        if not entities1 and not entities2:
            return 1.0
        if not entities1 or not entities2:
            return 0.0
        
        # Count matching keys
        keys1 = set(entities1.keys())
        keys2 = set(entities2.keys())
        
        common_keys = keys1 & keys2
        all_keys = keys1 | keys2
        
        if not all_keys:
            return 0.0
        
        # Jaccard similarity for keys
        key_similarity = len(common_keys) / len(all_keys)
        
        # Value similarity for common keys
        value_similarity = 0.0
        if common_keys:
            matching_values = sum(
                1 for k in common_keys 
                if entities1[k] == entities2[k]
            )
            value_similarity = matching_values / len(common_keys)
        
        # Weighted average
        return 0.5 * key_similarity + 0.5 * value_similarity
    
    def get_intent_prediction(self, text: str, entities: Dict[str, Any],
                            classifier_prediction: Optional[Tuple[str, float]] = None) -> Tuple[str, float, str]:
        """
        Get intent prediction using KNN.
        
        Args:
            text: Preprocessed query text
            entities: Extracted entities
            classifier_prediction: Optional classifier prediction (intent, confidence)
            
        Returns:
            Tuple of (intent_code, confidence, source)
        """
        matches = self.find_similar(text, entities)
        
        if not matches:
            # No similar examples found
            if classifier_prediction:
                return classifier_prediction[0], classifier_prediction[1], 'classifier'
            else:
                return 'UNKNOWN', 0.0, 'none'
        
        # Aggregate predictions from matches
        intent_votes = {}
        for match in matches:
            intent = match.example.intent_code
            weight = match.similarity
            
            if intent in intent_votes:
                intent_votes[intent].append(weight)
            else:
                intent_votes[intent] = [weight]
        
        # Calculate weighted scores
        intent_scores = {}
        for intent, weights in intent_votes.items():
            # Use mean of weights
            intent_scores[intent] = np.mean(weights)
        
        # Get top intent
        best_intent = max(intent_scores, key=intent_scores.get)
        best_score = intent_scores[best_intent]
        
        # Decide whether to use KNN or classifier
        if classifier_prediction:
            classifier_intent, classifier_conf = classifier_prediction
            
            # Use KNN if classifier confidence is low or KNN has high agreement
            if classifier_conf < self.config.confidence_boost_threshold:
                # KNN can boost low confidence predictions
                if best_intent == classifier_intent:
                    # Agreement - boost confidence
                    boosted_conf = (classifier_conf + best_score) / 2
                    return best_intent, boosted_conf, 'knn_boosted'
                elif best_score > 0.7:
                    # Strong KNN prediction overrides weak classifier
                    return best_intent, best_score, 'knn_override'
                else:
                    # Keep classifier prediction
                    return classifier_intent, classifier_conf, 'classifier'
            else:
                # High classifier confidence - keep it
                return classifier_intent, classifier_conf, 'classifier'
        else:
            # No classifier prediction - use KNN
            return best_intent, best_score, 'knn'
    
    def add_feedback_example(self, text: str, preprocessed_text: str,
                           entities: Dict[str, Any], intent_code: str, 
                           confidence: float = 1.0):
        """
        Add a feedback example (e.g., from user corrections).
        
        Args:
            text: Original query text
            preprocessed_text: Preprocessed query text
            entities: Extracted entities
            intent_code: Correct intent code
            confidence: Confidence score
        """
        example = QueryExample(
            text=text,
            preprocessed_text=preprocessed_text,
            entities=entities,
            intent_code=intent_code,
            confidence=confidence,
            source='feedback'
        )
        
        self.add_examples([example])
        
        # Rebuild index to include new example
        self.build_index()
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the KNN index.
        
        Returns:
            Dictionary with statistics
        """
        if not self.examples:
            return {
                'total_examples': 0,
                'indexed': False
            }
        
        # Count by source
        source_counts = {}
        intent_counts = {}
        
        for example in self.examples:
            # Source
            source = example.source
            source_counts[source] = source_counts.get(source, 0) + 1
            
            # Intent
            intent = example.intent_code
            intent_counts[intent] = intent_counts.get(intent, 0) + 1
        
        return {
            'total_examples': len(self.examples),
            'indexed': self.is_indexed,
            'unique_intents': len(intent_counts),
            'entity_keys': len(self.entity_keys),
            'source_distribution': source_counts,
            'intent_distribution': dict(sorted(
                intent_counts.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:10]),  # Top 10 intents
            'config': {
                'n_neighbors': self.config.n_neighbors,
                'min_similarity': self.config.min_similarity,
                'use_entity_similarity': self.config.use_entity_similarity
            }
        }
    
    def prune_examples(self, max_per_intent: int = 50, 
                      prioritize_source: List[str] = ['feedback', 'validated', 'training']):
        """
        Prune examples to limit memory usage.
        
        Args:
            max_per_intent: Maximum examples per intent
            prioritize_source: Source priority order
        """
        # Group by intent
        intent_examples = {}
        for example in self.examples:
            intent = example.intent_code
            if intent not in intent_examples:
                intent_examples[intent] = []
            intent_examples[intent].append(example)
        
        # Prune each intent
        pruned_examples = []
        for intent, examples in intent_examples.items():
            if len(examples) <= max_per_intent:
                pruned_examples.extend(examples)
            else:
                # Sort by source priority and confidence
                def sort_key(ex):
                    source_priority = prioritize_source.index(ex.source) \
                        if ex.source in prioritize_source else len(prioritize_source)
                    return (source_priority, -ex.confidence)
                
                sorted_examples = sorted(examples, key=sort_key)
                pruned_examples.extend(sorted_examples[:max_per_intent])
        
        removed = len(self.examples) - len(pruned_examples)
        if removed > 0:
            logger.info(f"Pruned {removed} examples from KNN index")
            self.examples = pruned_examples
            self.is_indexed = False
    
    def save(self, path: Union[str, Path]):
        """
        Save the KNN index.
        
        Args:
            path: Path to save the index
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save all components
        data = {
            'config': self.config,
            'examples': self.examples,
            'entity_keys': self.entity_keys,
            'tfidf_vectorizer': self.tfidf_vectorizer,
            'text_vectors': self.text_vectors,
            'is_indexed': self.is_indexed
        }
        
        joblib.dump(data, path)
        logger.info(f"KNN index saved to {path}")
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> 'KNNBackstop':
        """
        Load a saved KNN index.
        
        Args:
            path: Path to the saved index
            
        Returns:
            Loaded KNNBackstop instance
        """
        path = Path(path)
        
        if not path.exists():
            raise FileNotFoundError(f"KNN index not found: {path}")
        
        # Load data
        data = joblib.load(path)
        
        # Create instance
        instance = cls(data['config'])
        
        # Restore state
        instance.examples = data['examples']
        instance.entity_keys = data['entity_keys']
        instance.tfidf_vectorizer = data['tfidf_vectorizer']
        instance.text_vectors = data['text_vectors']
        instance.is_indexed = data['is_indexed']
        
        # Rebuild KNN model if indexed
        if instance.is_indexed:
            instance.build_index()
        
        logger.info(f"KNN index loaded from {path}")
        
        return instance
