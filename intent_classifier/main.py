"""
Main pipeline orchestrator for the intent classification system.
Coordinates all modules to process queries end-to-end.
"""

import logging
import time
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
import json
from pathlib import Path

from intent_classifier.models.schemas import (
    IntentClassificationResult, QueryRequest, Intent, Composition,
    ClauseSegment, EntityExtractionResult, LabelingFunctionVote,
    ClassificationConfig, CompositionMode
)
from intent_classifier.core.taxonomy_loader import TaxonomyLoader
from intent_classifier.core.preprocessor import Preprocessor
from intent_classifier.core.clause_splitter import ClauseSplitter
from intent_classifier.core.entity_extractor import EntityExtractor
from intent_classifier.core.labeling_functions import LabelingFunctions
from intent_classifier.core.classifier import IntentClassifier
from intent_classifier.core.knn_backstop import KNNBackstop
from intent_classifier.core.confidence import ConfidenceCalibrator
from intent_classifier.core.composition_builder import CompositionBuilder
from intent_classifier.core.router import IntentRouter, RouterBuilder


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Configuration for the intent classification pipeline."""
    # Component enablement
    use_spacy_splitter: bool = True
    use_knn_backstop: bool = True
    use_confidence_calibration: bool = True
    enable_spell_correction: bool = True
    
    # Thresholds
    min_confidence_threshold: float = 0.3
    knn_override_threshold: float = 0.75
    high_confidence_threshold: float = 0.8
    
    # KNN settings
    knn_k_neighbors: int = 5
    knn_text_weight: float = 0.7
    
    # Router settings
    max_parallel_workers: int = 4
    default_timeout: float = 30.0
    
    # Paths
    model_dir: str = "models"
    
    # Debug settings
    debug_mode: bool = False
    return_intermediate_results: bool = False


@dataclass
class PipelineState:
    """State tracking for pipeline execution."""
    query: str
    preprocessed_text: str = ""
    clauses: List[ClauseSegment] = field(default_factory=list)
    entities_per_clause: List[EntityExtractionResult] = field(default_factory=list)
    lf_votes_per_clause: List[List[LabelingFunctionVote]] = field(default_factory=list)
    intents: List[Intent] = field(default_factory=list)
    composition: Optional[Composition] = None
    execution_times: Dict[str, float] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)


class IntentClassificationPipeline:
    """
    Main pipeline orchestrator for intent classification.
    Coordinates all modules to process queries end-to-end.
    """
    
    def __init__(self, config: Optional[PipelineConfig] = None):
        """
        Initialize the pipeline with all components.
        
        Args:
            config: Pipeline configuration
        """
        self.config = config or PipelineConfig()
        
        # Initialize components
        logger.info("Initializing Intent Classification Pipeline...")
        
        # Core components
        self.taxonomy = TaxonomyLoader('taxonomy.json')
        self.preprocessor = Preprocessor(enable_spell_correction=self.config.enable_spell_correction)
        # Initialize clause splitter (disable spacy if not wanted)
        if not self.config.use_spacy_splitter:
            self.clause_splitter = ClauseSplitter(use_simple_fallback=True)
        else:
            self.clause_splitter = ClauseSplitter()
        # Get entity definitions from taxonomy
        entity_definitions = self.taxonomy.get_entity_definitions()
        self.entity_extractor = EntityExtractor(entity_definitions)
        self.weak_labeler = LabelingFunctions(self.taxonomy, self.entity_extractor)
        self.composition_builder = CompositionBuilder(self.taxonomy)
        
        # ML components (loaded lazily)
        self.classifier: Optional[IntentClassifier] = None
        self.knn_backstop: Optional[KNNBackstop] = None
        self.confidence_calibrator: Optional[ConfidenceCalibrator] = None
        
        # Router
        self.router = RouterBuilder.create_default_router(self.taxonomy)
        
        # Model paths
        self.model_dir = Path(self.config.model_dir)
        
        logger.info("Pipeline initialization complete")
    
    def load_models(self) -> bool:
        """
        Load trained models from disk.
        
        Returns:
            True if models loaded successfully
        """
        try:
            # Load classifier
            classifier_path = self.model_dir / "intent_classifier.pkl"
            if classifier_path.exists():
                self.classifier = IntentClassifier.load(str(classifier_path))
                logger.info("Loaded intent classifier")
            else:
                logger.warning(f"Classifier not found at {classifier_path}")
                return False
            
            # Load KNN backstop
            if self.config.use_knn_backstop:
                knn_path = self.model_dir / "knn_backstop.pkl"
                if knn_path.exists():
                    self.knn_backstop = KNNBackstop.load(str(knn_path))
                    logger.info("Loaded KNN backstop")
                else:
                    logger.warning(f"KNN backstop not found at {knn_path}")
                    self.knn_backstop = KNNBackstop(
                        k_neighbors=self.config.knn_k_neighbors,
                        text_weight=self.config.knn_text_weight
                    )
            
            # Load confidence calibrator
            if self.config.use_confidence_calibration:
                calibrator_path = self.model_dir / "confidence_calibrator.pkl"
                if calibrator_path.exists():
                    self.confidence_calibrator = ConfidenceCalibrator.load(str(calibrator_path))
                    logger.info("Loaded confidence calibrator")
                else:
                    logger.warning(f"Calibrator not found at {calibrator_path}")
                    self.confidence_calibrator = ConfidenceCalibrator()
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            return False
    
    def classify(self, query: str) -> IntentClassificationResult:
        """
        Main entry point for intent classification.
        
        Args:
            query: User query to classify
            
        Returns:
            Classification result with intents, entities, and composition
        """
        # Initialize state
        state = PipelineState(query=query)
        start_time = time.time()
        
        try:
            # Step 1: Preprocess
            self._preprocess(state)
            
            # Step 2: Split clauses
            self._split_clauses(state)
            
            # Step 3: Extract entities
            self._extract_entities(state)
            
            # Step 4: Classify intents
            self._classify_intents(state)
            
            # Step 5: Build composition
            self._build_composition(state)
            
            # Step 6: Create result
            result = self._create_result(state)
            
            # Log performance
            total_time = time.time() - start_time
            logger.info(f"Query processed in {total_time:.3f}s")
            
            return result
            
        except Exception as e:
            logger.error(f"Pipeline error: {e}")
            # Return error result with at least one intent
            error_intent = Intent(
                coarse_class="ERROR",
                intent_code="ERROR:CLASSIFICATION_FAILED",
                confidence=0.0,
                entities={},
                policies={},
                routing_hints={}
            )
            
            return IntentClassificationResult(
                request=QueryRequest(
                    query_id=f"q_{int(time.time() * 1000)}",
                    user_id="pipeline_user",
                    text=query
                ),
                intents=[error_intent],
                composition=Composition(
                    mode=CompositionMode.SINGLE,
                    ordering=[0],
                    join_keys=[],
                    response_policy="error"
                )
            )
    
    def _preprocess(self, state: PipelineState):
        """Preprocessing step."""
        start = time.time()
        
        state.preprocessed_text = self.preprocessor.process(state.query)
        
        if self.config.debug_mode:
            logger.debug(f"Original: {state.query}")
            logger.debug(f"Preprocessed: {state.preprocessed_text}")
        
        state.execution_times['preprocess'] = time.time() - start
    
    def _split_clauses(self, state: PipelineState):
        """Clause splitting step."""
        start = time.time()
        
        state.clauses = self.clause_splitter.split(state.preprocessed_text)
        
        if self.config.debug_mode:
            logger.debug(f"Found {len(state.clauses)} clauses:")
            for i, clause in enumerate(state.clauses):
                logger.debug(f"  [{i}] {clause.text} (deps: {clause.dependencies})")
        
        state.execution_times['split_clauses'] = time.time() - start
    
    def _extract_entities(self, state: PipelineState):
        """Entity extraction step."""
        start = time.time()
        
        # Extract entities for each clause
        for clause in state.clauses:
            entities = self.entity_extractor.extract(
                clause.text,
                preprocessed_text=clause.text  # Already preprocessed
            )
            state.entities_per_clause.append(entities)
        
        if self.config.debug_mode:
            for i, (clause, entities) in enumerate(zip(state.clauses, state.entities_per_clause)):
                logger.debug(f"Clause {i} entities: {entities.entities}")
        
        state.execution_times['extract_entities'] = time.time() - start
    
    def _classify_intents(self, state: PipelineState):
        """Intent classification step."""
        start = time.time()
        
        # Process each clause
        for i, (clause, entities) in enumerate(zip(state.clauses, state.entities_per_clause)):
            # Get weak supervision labels
            lf_votes = self.weak_labeler.apply_all(clause.text, entities.entities)
            state.lf_votes_per_clause.append(lf_votes)
            
            # Get initial prediction from weak supervision
            if lf_votes:
                # Use top vote as initial prediction
                top_vote = lf_votes[0]
                initial_intent = top_vote.intent_code
                initial_confidence = top_vote.confidence
            else:
                initial_intent = "UNKNOWN:UNKNOWN"
                initial_confidence = 0.0
            
            # Use classifier if available
            if self.classifier:
                try:
                    # Classify
                    clf_result = self.classifier.predict(
                        [clause.text],
                        [entities.entities]
                    )
                    
                    if clf_result['predictions']:
                        pred = clf_result['predictions'][0]
                        intent_code = pred['intent']
                        confidence = pred['confidence']
                        
                        # Apply confidence calibration
                        if self.confidence_calibrator:
                            confidence = self.confidence_calibrator.calibrate_single(
                                confidence,
                                intent_code
                            )
                    else:
                        intent_code = initial_intent
                        confidence = initial_confidence
                        
                except Exception as e:
                    logger.warning(f"Classifier failed for clause {i}: {e}")
                    intent_code = initial_intent
                    confidence = initial_confidence
            else:
                # No classifier, use weak supervision
                intent_code = initial_intent
                confidence = initial_confidence
            
            # Apply KNN backstop if enabled
            if self.knn_backstop and self.config.use_knn_backstop:
                knn_result = self.knn_backstop.predict(
                    clause.text,
                    entities.entities,
                    intent_code,
                    confidence
                )
                
                # Use KNN result based on mode
                if knn_result.mode in ['knn_only', 'override']:
                    intent_code = knn_result.intent_code
                    confidence = knn_result.confidence
                elif knn_result.mode == 'boost':
                    confidence = knn_result.confidence
            
            # Check minimum confidence
            if confidence < self.config.min_confidence_threshold:
                state.warnings.append(
                    f"Low confidence ({confidence:.2f}) for clause {i}: {clause.text}"
                )
            
            # Get intent info from taxonomy
            intent_info = self.taxonomy.get_intent_info(intent_code)
            if intent_info:
                try:
                    coarse_class = intent_info['coarse_class']
                    policies = intent_info.get('policies', {})
                    routing_hints = self.taxonomy.get_routing_hints(intent_code)
                except KeyError as e:
                    logger.warning(f"Missing key in intent_info for {intent_code}: {e}")
                    coarse_class = intent_code.split(':')[0]
                    policies = {}
                    routing_hints = {}
            else:
                coarse_class = intent_code.split(':')[0]
                policies = {}
                routing_hints = {}
                state.warnings.append(f"Unknown intent code: {intent_code}")
            
            # Create intent object
            intent = Intent(
                coarse_class=coarse_class,
                intent_code=intent_code,
                confidence=confidence,
                entities=entities.entities,
                policies=policies,
                routing_hints=routing_hints
            )
            
            state.intents.append(intent)
        
        state.execution_times['classify_intents'] = time.time() - start
    
    def _build_composition(self, state: PipelineState):
        """Composition building step."""
        start = time.time()
        
        if state.intents:
            state.composition = self.composition_builder.build_composition(
                state.clauses,
                state.intents
            )
        else:
            # Empty composition
            state.composition = Composition(
                mode=CompositionMode.SINGLE,
                ordering=[],
                join_keys=[],
                response_policy="empty"
            )
        
        state.execution_times['build_composition'] = time.time() - start
    
    def _create_result(self, state: PipelineState) -> IntentClassificationResult:
        """Create final classification result."""
        # Calculate overall confidence
        if state.intents:
            overall_confidence = sum(i.confidence for i in state.intents) / len(state.intents)
        else:
            overall_confidence = 0.0
        
        # Create result
        # Ensure at least one intent
        if not state.intents:
            state.intents.append(Intent(
                coarse_class="UNKNOWN",
                intent_code="UNKNOWN:NO_INTENT",
                confidence=0.0,
                entities={},
                policies={},
                routing_hints={}
            ))
        
        result = IntentClassificationResult(
            request=QueryRequest(
                query_id=f"q_{int(time.time() * 1000)}",
                user_id="pipeline_user",
                text=state.query
            ),
            intents=state.intents,
            composition=state.composition
        )
        
        # Store debug info in a way we can access it
        # Note: We can't add arbitrary fields to Pydantic models
        # so we'll need to handle this differently
        
        return result
    
    def process_batch(self, queries: List[str]) -> List[IntentClassificationResult]:
        """
        Process multiple queries in batch.
        
        Args:
            queries: List of queries to process
            
        Returns:
            List of classification results
        """
        results = []
        
        for query in queries:
            result = self.classify(query)
            results.append(result)
        
        return results
    
    def explain_classification(self, query: str) -> Dict[str, Any]:
        """
        Provide detailed explanation of classification decision.
        
        Args:
            query: Query to explain
            
        Returns:
            Detailed explanation
        """
        # Enable debug mode temporarily
        original_debug = self.config.debug_mode
        original_intermediate = self.config.return_intermediate_results
        
        self.config.debug_mode = True
        self.config.return_intermediate_results = True
        
        # Classify with debug info
        result = self.classify(query)
        
        # Restore original settings
        self.config.debug_mode = original_debug
        self.config.return_intermediate_results = original_intermediate
        
        # Build explanation
        explanation = {
            'query': query,
            'final_intents': [
                {
                    'intent': intent.intent_code,
                    'confidence': intent.confidence,
                    'entities': intent.entities
                }
                for intent in result.intents
            ],
            'composition': {
                'mode': result.composition.mode.value,
                'ordering': result.composition.ordering,
                'join_keys': [
                    f"{jk.key}: {jk.from_step}→{jk.to_step}"
                    for jk in result.composition.join_keys
                ]
            },
            'processing_steps': result.debug_info if hasattr(result, 'debug_info') else {},
            'warnings': result.warnings
        }
        
        return explanation
    
    def shutdown(self):
        """Cleanup resources."""
        self.router.shutdown()
        logger.info("Pipeline shutdown complete")


def create_pipeline(config: Optional[PipelineConfig] = None) -> IntentClassificationPipeline:
    """
    Factory function to create a configured pipeline.
    
    Args:
        config: Optional pipeline configuration
        
    Returns:
        Configured pipeline instance
    """
    pipeline = IntentClassificationPipeline(config)
    
    # Try to load existing models
    if pipeline.model_dir.exists():
        logger.info("Loading existing models...")
        if pipeline.load_models():
            logger.info("Models loaded successfully")
        else:
            logger.warning("Some models could not be loaded")
    else:
        logger.info("No existing models found")
    
    return pipeline


def main():
    """Example usage of the pipeline."""
    # Create pipeline
    pipeline = create_pipeline()
    
    # Example queries
    test_queries = [
        "What is the status of RFI 1838?",
        "Find all open RFIs and show me the latest submittal",
        "How many RFIs are related to electrical work in Building A?",
        "Show spec section 03 30 00 and related submittals",
        "Get RFI 123, check its status, and find the response"
    ]
    
    # Process queries
    for query in test_queries:
        print(f"\nQuery: {query}")
        print("-" * 60)
        
        result = pipeline.classify(query)
        
        print(f"Intents found: {len(result.intents)}")
        for i, intent in enumerate(result.intents):
            print(f"  [{i}] {intent.intent_code} (confidence: {intent.confidence:.2f})")
            if intent.entities:
                print(f"      Entities: {intent.entities}")
        
        print(f"Composition: {result.composition.mode.value}")
        print(f"Overall confidence: {result.confidence:.2f}")
        print(f"Processing time: {result.processing_time:.3f}s")
        
        if result.warnings:
            print(f"Warnings: {result.warnings}")
    
    # Cleanup
    pipeline.shutdown()


if __name__ == "__main__":
    main()