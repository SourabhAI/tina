#!/usr/bin/env python3
"""
Testing suite for running intent classification on validation questions.
Processes val_questions.json and stores results as JSON.
"""

import json
import time
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field, asdict
import traceback

from intent_classifier.main import create_pipeline, PipelineConfig


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class TestResult:
    """Result for a single test question."""
    query_id: str
    user_id: str
    question_text: str
    created_at: str
    
    # Classification results
    intents: List[Dict[str, Any]]
    entities: Dict[str, Any]
    composition_mode: str
    execution_order: List[int]
    join_keys: List[str]
    
    # Metadata
    processing_time_ms: float
    success: bool
    error_message: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    
    # Debug info
    preprocessed_text: Optional[str] = None
    clauses: Optional[List[Dict[str, Any]]] = None


@dataclass
class TestSummary:
    """Summary statistics for the test run."""
    total_questions: int
    successful: int
    failed: int
    average_processing_time_ms: float
    
    # Intent distribution
    intent_distribution: Dict[str, int]
    coarse_class_distribution: Dict[str, int]
    
    # Composition analysis
    single_intent_count: int
    multi_intent_count: int
    composition_modes: Dict[str, int]
    
    # Entity statistics
    entity_type_counts: Dict[str, int]
    
    # Timing
    start_time: str
    end_time: str
    total_duration_seconds: float


class ValidationTestSuite:
    """Test suite for running validation questions through the pipeline."""
    
    def __init__(self, pipeline_config: Optional[PipelineConfig] = None):
        """
        Initialize the test suite.
        
        Args:
            pipeline_config: Configuration for the pipeline
        """
        self.config = pipeline_config or PipelineConfig(
            use_spacy_splitter=False,  # Use rule-based for speed
            use_knn_backstop=False,    # Disable for initial testing
            use_confidence_calibration=False,
            enable_spell_correction=False,
            debug_mode=False,
            return_intermediate_results=True  # Get detailed info
        )
        
        self.pipeline = None
        self.results: List[TestResult] = []
        
    def initialize_pipeline(self):
        """Initialize the classification pipeline."""
        logger.info("Initializing intent classification pipeline...")
        self.pipeline = create_pipeline(self.config)
        logger.info("Pipeline initialized successfully")
        
    def load_questions(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Load questions from JSON file.
        
        Args:
            file_path: Path to the questions JSON file
            
        Returns:
            List of question dictionaries
        """
        logger.info(f"Loading questions from {file_path}")
        with open(file_path, 'r') as f:
            questions = json.load(f)
        logger.info(f"Loaded {len(questions)} questions")
        return questions
    
    def process_question(self, question: Dict[str, Any]) -> TestResult:
        """
        Process a single question through the pipeline.
        
        Args:
            question: Question dictionary
            
        Returns:
            Test result
        """
        query_id = question['query_id']
        question_text = question['question_text']
        
        logger.debug(f"Processing {query_id}: {question_text}")
        
        start_time = time.time()
        
        try:
            # Run classification
            result = self.pipeline.classify(question_text)
            
            processing_time_ms = (time.time() - start_time) * 1000
            
            # Extract intents
            intents = []
            all_entities = {}
            
            for intent in result.intents:
                intents.append({
                    'intent_code': intent.intent_code,
                    'coarse_class': intent.coarse_class,
                    'confidence': intent.confidence,
                    'entities': intent.entities
                })
                # Merge entities
                all_entities.update(intent.entities)
            
            # Extract composition info
            composition_mode = result.composition.mode.value
            execution_order = result.composition.ordering
            join_keys = [f"{jk.key}: {jk.from_step}→{jk.to_step}" 
                        for jk in result.composition.join_keys]
            
            test_result = TestResult(
                query_id=query_id,
                user_id=question['user_id'],
                question_text=question_text,
                created_at=question['created_at'],
                intents=intents,
                entities=all_entities,
                composition_mode=composition_mode,
                execution_order=execution_order,
                join_keys=join_keys,
                processing_time_ms=processing_time_ms,
                success=True
            )
            
            # Add warnings if any
            warnings = getattr(result, 'warnings', [])
            if warnings:
                test_result.warnings = warnings
            
        except Exception as e:
            # Handle errors
            processing_time_ms = (time.time() - start_time) * 1000
            
            test_result = TestResult(
                query_id=query_id,
                user_id=question['user_id'],
                question_text=question_text,
                created_at=question['created_at'],
                intents=[],
                entities={},
                composition_mode='ERROR',
                execution_order=[],
                join_keys=[],
                processing_time_ms=processing_time_ms,
                success=False,
                error_message=str(e)
            )
            
            logger.error(f"Error processing {query_id}: {e}")
            logger.debug(traceback.format_exc())
        
        return test_result
    
    def run_tests(self, questions_file: str, 
                  output_file: str = "validation_results.json",
                  summary_file: str = "validation_summary.json"):
        """
        Run tests on all questions and save results.
        
        Args:
            questions_file: Path to questions JSON
            output_file: Path to save detailed results
            summary_file: Path to save summary statistics
        """
        # Initialize
        if not self.pipeline:
            self.initialize_pipeline()
        
        # Load questions
        questions = self.load_questions(questions_file)
        
        # Process questions
        logger.info("Starting validation test run...")
        start_time = datetime.now()
        
        for i, question in enumerate(questions):
            if i % 10 == 0:
                logger.info(f"Progress: {i}/{len(questions)} questions processed")
            
            result = self.process_question(question)
            self.results.append(result)
        
        end_time = datetime.now()
        
        logger.info(f"Completed processing {len(questions)} questions")
        
        # Generate summary
        summary = self.generate_summary(start_time, end_time)
        
        # Save results
        self.save_results(output_file, summary_file, summary)
        
        # Shutdown pipeline
        self.pipeline.shutdown()
        
        return summary
    
    def generate_summary(self, start_time: datetime, end_time: datetime) -> TestSummary:
        """
        Generate summary statistics from test results.
        
        Args:
            start_time: Test start time
            end_time: Test end time
            
        Returns:
            Test summary
        """
        # Basic counts
        total = len(self.results)
        successful = sum(1 for r in self.results if r.success)
        failed = total - successful
        
        # Processing time
        processing_times = [r.processing_time_ms for r in self.results if r.success]
        avg_time = sum(processing_times) / len(processing_times) if processing_times else 0
        
        # Intent distribution
        intent_dist = {}
        coarse_class_dist = {}
        
        for result in self.results:
            if result.success:
                for intent in result.intents:
                    intent_code = intent['intent_code']
                    coarse_class = intent['coarse_class']
                    
                    intent_dist[intent_code] = intent_dist.get(intent_code, 0) + 1
                    coarse_class_dist[coarse_class] = coarse_class_dist.get(coarse_class, 0) + 1
        
        # Composition analysis
        single_intent = sum(1 for r in self.results 
                          if r.success and len(r.intents) == 1)
        multi_intent = sum(1 for r in self.results 
                         if r.success and len(r.intents) > 1)
        
        composition_modes = {}
        for result in self.results:
            if result.success:
                mode = result.composition_mode
                composition_modes[mode] = composition_modes.get(mode, 0) + 1
        
        # Entity statistics
        entity_counts = {}
        for result in self.results:
            if result.success:
                for entity_type in result.entities.keys():
                    entity_counts[entity_type] = entity_counts.get(entity_type, 0) + 1
        
        return TestSummary(
            total_questions=total,
            successful=successful,
            failed=failed,
            average_processing_time_ms=avg_time,
            intent_distribution=intent_dist,
            coarse_class_distribution=coarse_class_dist,
            single_intent_count=single_intent,
            multi_intent_count=multi_intent,
            composition_modes=composition_modes,
            entity_type_counts=entity_counts,
            start_time=start_time.isoformat(),
            end_time=end_time.isoformat(),
            total_duration_seconds=(end_time - start_time).total_seconds()
        )
    
    def save_results(self, results_file: str, summary_file: str, summary: TestSummary):
        """
        Save test results and summary to JSON files.
        
        Args:
            results_file: Path for detailed results
            summary_file: Path for summary
            summary: Test summary object
        """
        # Save detailed results
        results_data = []
        for result in self.results:
            # Convert to dict and remove None values
            result_dict = asdict(result)
            result_dict = {k: v for k, v in result_dict.items() if v is not None}
            results_data.append(result_dict)
        
        with open(results_file, 'w') as f:
            json.dump(results_data, f, indent=2)
        logger.info(f"Saved detailed results to {results_file}")
        
        # Save summary
        summary_dict = asdict(summary)
        with open(summary_file, 'w') as f:
            json.dump(summary_dict, f, indent=2)
        logger.info(f"Saved summary to {summary_file}")
        
        # Print summary to console
        self.print_summary(summary)
    
    def print_summary(self, summary: TestSummary):
        """Print summary to console."""
        print("\n" + "=" * 60)
        print("VALIDATION TEST SUMMARY")
        print("=" * 60)
        
        print(f"\nTotal Questions: {summary.total_questions}")
        print(f"Successful: {summary.successful} ({summary.successful/summary.total_questions*100:.1f}%)")
        print(f"Failed: {summary.failed}")
        print(f"Average Processing Time: {summary.average_processing_time_ms:.1f}ms")
        print(f"Total Duration: {summary.total_duration_seconds:.1f}s")
        
        print(f"\nComposition Analysis:")
        print(f"  Single Intent: {summary.single_intent_count}")
        print(f"  Multi Intent: {summary.multi_intent_count}")
        print(f"  Modes: {summary.composition_modes}")
        
        print(f"\nTop Coarse Classes:")
        sorted_coarse = sorted(summary.coarse_class_distribution.items(), 
                              key=lambda x: x[1], reverse=True)[:5]
        for coarse, count in sorted_coarse:
            print(f"  {coarse}: {count}")
        
        print(f"\nTop Intent Codes:")
        sorted_intents = sorted(summary.intent_distribution.items(), 
                               key=lambda x: x[1], reverse=True)[:5]
        for intent, count in sorted_intents:
            print(f"  {intent}: {count}")
        
        print(f"\nTop Entity Types:")
        sorted_entities = sorted(summary.entity_type_counts.items(), 
                               key=lambda x: x[1], reverse=True)[:5]
        for entity, count in sorted_entities:
            print(f"  {entity}: {count}")
        
        print("\n" + "=" * 60)


def main():
    """Main entry point for the validation test suite."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run validation tests on intent classifier')
    parser.add_argument('--input', '-i', default='val_questions.json',
                       help='Input questions file (default: val_questions.json)')
    parser.add_argument('--output', '-o', default='validation_results.json',
                       help='Output results file (default: validation_results.json)')
    parser.add_argument('--summary', '-s', default='validation_summary.json',
                       help='Output summary file (default: validation_summary.json)')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug mode')
    parser.add_argument('--spacy', action='store_true',
                       help='Use SpaCy clause splitter')
    
    args = parser.parse_args()
    
    # Configure pipeline
    config = PipelineConfig(
        use_spacy_splitter=args.spacy,
        use_knn_backstop=False,
        use_confidence_calibration=False,
        enable_spell_correction=False,
        debug_mode=args.debug,
        return_intermediate_results=True
    )
    
    # Create and run test suite
    suite = ValidationTestSuite(config)
    summary = suite.run_tests(
        questions_file=args.input,
        output_file=args.output,
        summary_file=args.summary
    )
    
    print(f"\nTest run completed successfully!")
    print(f"Results saved to: {args.output}")
    print(f"Summary saved to: {args.summary}")


if __name__ == "__main__":
    main()
