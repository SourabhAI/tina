#!/usr/bin/env python3
"""
Simple script to run the intent classifier on command line input.
"""

import sys
import json
from intent_classifier.main import create_pipeline, PipelineConfig

def main():
    """Run classifier on user input."""
    print("TINA Intent Classifier - Interactive Mode")
    print("="*50)
    print("Type your query (or 'quit' to exit)")
    print("="*50)
    
    # Create pipeline with all features
    config = PipelineConfig(
        use_spacy_splitter=True,
        use_knn_backstop=True,
        use_confidence_calibration=True,
        enable_spell_correction=False
    )
    
    try:
        pipeline = create_pipeline(config)
        print("✓ Pipeline loaded successfully\n")
    except Exception as e:
        print(f"Error loading pipeline: {e}")
        print("\nMake sure you have:")
        print("1. Installed requirements: pip install -r requirements.txt")
        print("2. Downloaded NLP models: python setup_nlp.py")
        print("3. Trained models: python train_classifier.py")
        sys.exit(1)
    
    # Interactive loop
    while True:
        try:
            # Get user input
            query = input("\nQuery> ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                print("\nGoodbye!")
                break
            
            if not query:
                continue
            
            # Classify
            result = pipeline.classify(query)
            
            # Display results
            print("\nResults:")
            print("-" * 30)
            
            # Intents
            for i, intent in enumerate(result.intents):
                print(f"Intent {i+1}: {intent.intent_code}")
                print(f"  Confidence: {intent.confidence:.3f}")
                if hasattr(intent, 'parameters') and intent.parameters:
                    print(f"  Parameters: {intent.parameters}")
            
            # Composition
            if len(result.intents) > 1:
                print(f"\nComposition: {result.composition.mode}")
                print(f"Execution Order: {result.composition.execution_order}")
            
            # Entities
            all_entities = {}
            for intent in result.intents:
                if intent.entities:
                    all_entities.update(intent.entities)
            
            if all_entities:
                print(f"\nExtracted Entities:")
                for entity_type, value in all_entities.items():
                    print(f"  {entity_type}: {value}")
            
        except KeyboardInterrupt:
            print("\n\nInterrupted. Goodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")
    
    # Cleanup
    pipeline.shutdown()

if __name__ == "__main__":
    # Check for command line argument
    if len(sys.argv) > 1:
        # Process single query from command line
        query = " ".join(sys.argv[1:])
        
        config = PipelineConfig(
            use_spacy_splitter=True,
            use_knn_backstop=True,
            use_confidence_calibration=True
        )
        
        pipeline = create_pipeline(config)
        result = pipeline.classify(query)
        
        # Output as JSON
        output = {
            "query": query,
            "intents": [
                {
                    "code": intent.intent_code,
                    "confidence": intent.confidence
                }
                for intent in result.intents
            ],
            "composition": result.composition.mode
        }
        
        print(json.dumps(output, indent=2))
        pipeline.shutdown()
    else:
        # Interactive mode
        main()