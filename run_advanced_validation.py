#!/usr/bin/env python3
"""
Run validation test with all advanced features enabled.
"""

from test_validation_suite import ValidationTestSuite, PipelineConfig
import logging
import json

# Set logging level
logging.basicConfig(level=logging.INFO)

def main():
    """Run validation tests with advanced features enabled."""
    print("Starting validation test with advanced features...")
    print("=" * 60)
    
    # Create test suite with ALL features enabled
    config = PipelineConfig(
        use_spacy_splitter=True,           # Use SpaCy for better clause splitting
        use_knn_backstop=True,             # Enable KNN for similar query matching
        use_confidence_calibration=True,   # Enable confidence calibration
        enable_spell_correction=False,     # Keep disabled for speed
        min_confidence_threshold=0.5,      # Consider low confidence below this
        knn_override_threshold=0.7,        # Override classifier if KNN is very confident
        debug_mode=False,
        return_intermediate_results=True   # Get detailed results
    )
    
    print("\nEnabled features:")
    print("  ✓ SpaCy clause splitter")
    print("  ✓ Supervised classifier") 
    print("  ✓ KNN backstop")
    print("  ✓ Confidence calibration")
    print()
    
    suite = ValidationTestSuite(config)
    
    # Run tests with different output files
    summary = suite.run_tests(
        questions_file='val_questions.json',
        output_file='validation_results_advanced.json',
        summary_file='validation_summary_advanced.json'
    )
    
    print("\n" + "=" * 60)
    print("VALIDATION TEST SUMMARY (WITH ADVANCED FEATURES)")
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
    
    # Top intents
    print(f"\nTop Intent Codes:")
    sorted_intents = sorted(summary.intent_distribution.items(), 
                           key=lambda x: x[1], reverse=True)[:10]
    for intent, count in sorted_intents:
        print(f"  {intent}: {count}")
    
    # Compare with previous results
    try:
        with open("validation_summary.json", 'r') as f:
            prev_summary = json.load(f)
        
        print("\n" + "=" * 60)
        print("COMPARISON WITH BASIC PIPELINE")
        print("=" * 60)
        
        prev_unknown = prev_summary['intent_distribution'].get('UNKNOWN:UNKNOWN', 0)
        curr_unknown = summary.intent_distribution.get('UNKNOWN:UNKNOWN', 0)
        
        print(f"\nUNKNOWN Classifications:")
        print(f"  Basic Pipeline: {prev_unknown}")
        print(f"  Advanced Pipeline: {curr_unknown}")
        if prev_unknown > curr_unknown:
            print(f"  Improvement: {prev_unknown - curr_unknown} fewer unknowns ({(prev_unknown - curr_unknown)/prev_unknown*100:.1f}% reduction)")
        
        print(f"\nProcessing Time:")
        print(f"  Basic: {prev_summary['average_processing_time_ms']:.1f}ms")
        print(f"  Advanced: {summary.average_processing_time_ms:.1f}ms")
        print(f"  Overhead: {summary.average_processing_time_ms - prev_summary['average_processing_time_ms']:.1f}ms")
        
        # Show examples that improved
        if curr_unknown < prev_unknown:
            print("\nExamples of improved classifications:")
            count = 0
            for i, result in enumerate(suite.results):
                if count >= 5:
                    break
                # Check if this was UNKNOWN before
                if result.intents and result.intents[0]['intent_code'] != 'UNKNOWN:UNKNOWN':
                    # Likely was unknown before, now classified
                    if 'knn' in str(result.intents[0].get('source', '')).lower():
                        print(f"\n  Q: {result.question_text}")
                        print(f"  → {result.intents[0]['intent_code']} (KNN match, conf: {result.intents[0]['confidence']:.2f})")
                        count += 1
        
    except FileNotFoundError:
        print("\nNo previous results found for comparison")
    except Exception as e:
        print(f"\nError comparing results: {e}")
    
    print("\n" + "=" * 60)
    
    print("\n✓ Advanced validation test completed!")
    print(f"\nResults saved to:")
    print(f"  - validation_results_advanced.json (detailed results)")
    print(f"  - validation_summary_advanced.json (statistics)")


if __name__ == "__main__":
    main()