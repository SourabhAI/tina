#!/usr/bin/env python3
"""
Simple script to run validation tests with default settings.
"""

from test_validation_suite import ValidationTestSuite, PipelineConfig
import logging

# Set logging level
logging.basicConfig(level=logging.INFO)

def main():
    """Run validation tests with default settings."""
    print("Starting validation test run...")
    print("=" * 60)
    
    # Create test suite with optimized settings
    config = PipelineConfig(
        use_spacy_splitter=False,      # Faster without SpaCy
        use_knn_backstop=False,         # No KNN for speed
        use_confidence_calibration=False,  # No calibration
        enable_spell_correction=False,  # No spell correction
        debug_mode=False,               # No debug logging
        return_intermediate_results=True  # But we want detailed results
    )
    
    suite = ValidationTestSuite(config)
    
    # Run tests
    summary = suite.run_tests(
        questions_file='val_questions.json',
        output_file='validation_results.json',
        summary_file='validation_summary.json'
    )
    
    print("\n✓ Test run completed!")
    print(f"  - Processed {summary.total_questions} questions")
    print(f"  - Success rate: {summary.successful/summary.total_questions*100:.1f}%")
    print(f"  - Average time: {summary.average_processing_time_ms:.1f}ms per question")
    
    # Show sample results
    print("\nSample Results (first 5):")
    print("-" * 60)
    
    for i, result in enumerate(suite.results[:5]):
        print(f"\nQ{i+1}: {result.question_text}")
        if result.success:
            for intent in result.intents:
                print(f"  → {intent['intent_code']} (conf: {intent['confidence']:.2f})")
            if result.entities:
                print(f"  Entities: {list(result.entities.keys())}")
        else:
            print(f"  ✗ Error: {result.error_message}")
    
    print("\nFull results saved to:")
    print("  - validation_results.json (detailed results)")
    print("  - validation_summary.json (statistics)")


if __name__ == "__main__":
    main()
