"""
Main pipeline orchestrator for the intent classification system.
Coordinates all components to process queries end-to-end.
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

from intent_classifier.models.schemas import (
    QueryRequest,
    IntentClassificationResult,
    ClassificationConfig,
    Composition,
    CompositionMode
)
from intent_classifier.core.taxonomy_loader import TaxonomyLoader
from intent_classifier.core.preprocessor import Preprocessor
from intent_classifier.core.clause_splitter import ClauseSplitter
from intent_classifier.core.entity_extractor import EntityExtractor
from intent_classifier.core.labeling_functions import LabelingEngine
from intent_classifier.core.classifier import IntentClassifier
from intent_classifier.core.knn_backstop import KNNBackstop
from intent_classifier.core.confidence import ConfidenceCalibrator
from intent_classifier.core.composition_builder import CompositionBuilder
from intent_classifier.core.router import Router


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class IntentClassificationPipeline:
    """
    Main pipeline for intent classification.
    Orchestrates all components to process queries.
    """
    
    def __init__(self, config: Optional[ClassificationConfig] = None):
        """Initialize the pipeline with all components."""
        self.config = config or ClassificationConfig()
        
        # Initialize components
        logger.info("Initializing intent classification pipeline...")
        
        self.taxonomy = TaxonomyLoader()
        self.preprocessor = Preprocessor()
        self.clause_splitter = ClauseSplitter()
        self.entity_extractor = EntityExtractor(self.taxonomy)
        self.labeling_engine = LabelingEngine(self.taxonomy)
        self.classifier = IntentClassifier(self.taxonomy)
        self.knn_backstop = KNNBackstop(self.taxonomy) if self.config.enable_knn_backstop else None
        self.confidence_calibrator = ConfidenceCalibrator()
        self.composition_builder = CompositionBuilder(self.taxonomy)
        self.router = Router(self.taxonomy)
        
        logger.info("Pipeline initialized successfully")
    
    def classify(self, query: str, user_id: Optional[str] = None, query_id: Optional[str] = None) -> IntentClassificationResult:
        """
        Main classification method.
        Processes a query through the entire pipeline.
        """
        # Create request object
        request = QueryRequest(
            query_id=query_id or self._generate_query_id(),
            user_id=user_id or "anonymous",
            text=query,
            created_at=datetime.utcnow()
        )
        
        logger.info(f"Processing query: {query_id} - '{query[:50]}...'")
        
        try:
            # Step 1: Preprocess the query
            preprocessed_text = self.preprocessor.process(query)
            
            # Step 2: Split into clauses
            clauses = self.clause_splitter.split(preprocessed_text)
            logger.info(f"Split into {len(clauses)} clauses")
            
            # Step 3: Extract entities from each clause
            clause_entities = []
            for clause in clauses:
                entities = self.entity_extractor.extract(clause.text)
                clause_entities.append(entities)
            
            # Step 4: Run labeling functions
            labeling_results = []
            for i, clause in enumerate(clauses):
                votes = self.labeling_engine.get_votes(clause.text, clause_entities[i].entities)
                labeling_results.append(votes)
            
            # Step 5: Run supervised classifier
            intents = []
            for i, clause in enumerate(clauses):
                # Get initial classification
                intent = self.classifier.classify(
                    clause.text, 
                    clause_entities[i].entities,
                    labeling_results[i]
                )
                
                # Step 6: Apply KNN backstop if confidence is low
                if self.knn_backstop and intent.confidence < self.config.confidence_threshold:
                    knn_intent = self.knn_backstop.classify(clause.text, intent.coarse_class)
                    if knn_intent.confidence > intent.confidence:
                        intent = knn_intent
                
                # Step 7: Calibrate confidence
                intent = self.confidence_calibrator.calibrate(intent)
                
                # Step 8: Apply routing hints
                intent = self.router.add_routing_info(intent)
                
                intents.append(intent)
            
            # Step 9: Build composition for multi-intent queries
            composition = self.composition_builder.build(intents, clauses)
            
            # Step 10: Create final result
            result = IntentClassificationResult(
                request=request,
                composition=composition,
                intents=intents
            )
            
            logger.info(f"Classification complete: {len(intents)} intents identified")
            return result
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            # Return a fallback result with UNKNOWN intent
            return self._create_unknown_result(request)
    
    def batch_classify(self, queries: List[str]) -> List[IntentClassificationResult]:
        """Process multiple queries in batch."""
        results = []
        for query in queries:
            result = self.classify(query)
            results.append(result)
        return results
    
    def _generate_query_id(self) -> str:
        """Generate a unique query ID."""
        import uuid
        return str(uuid.uuid4())
    
    def _create_unknown_result(self, request: QueryRequest) -> IntentClassificationResult:
        """Create a result for unknown/failed classification."""
        from intent_classifier.models.schemas import Intent
        
        unknown_intent = Intent(
            coarse_class="UNKNOWN",
            intent_code="UNKNOWN:GENERAL",
            confidence=0.0,
            entities={},
            policies={"freshness": "any"},
            routing_hints={"tools": ["GeneralRAG"], "few_shot_id": "fallback_v1"}
        )
        
        composition = Composition(
            mode=CompositionMode.SINGLE,
            ordering=[0],
            join_keys=[],
            response_policy="single_message"
        )
        
        return IntentClassificationResult(
            request=request,
            composition=composition,
            intents=[unknown_intent]
        )


def main():
    """Example usage of the pipeline."""
    # Initialize pipeline
    pipeline = IntentClassificationPipeline()
    
    # Example queries
    test_queries = [
        "Show me the tile submittal and is it approved?",
        "What spec section covers fire rated doors?",
        "How many RFIs are still open?",
        "Where is AHU-5 located on the mechanical drawings?",
    ]
    
    # Process queries
    for query in test_queries:
        result = pipeline.classify(query)
        print(f"\nQuery: {query}")
        print(f"Intents: {[intent.intent_code for intent in result.intents]}")
        print(f"Composition: {result.composition.mode}")
        print(f"Confidence: {[intent.confidence for intent in result.intents]}")


if __name__ == "__main__":
    main()
