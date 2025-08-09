#!/usr/bin/env python3
"""
Simple script to run the intent classifier on sample queries.
"""

import json
from intent_classifier.main import IntentClassificationPipeline


def main():
    """Run the classifier on sample queries."""
    print("Initializing Intent Classification Pipeline...")
    pipeline = IntentClassificationPipeline()
    
    # Sample queries to test
    test_queries = [
        "Show me the tile submittal and is it approved?",
        "What spec section covers fire rated doors?",
        "How many RFIs are still open?",
        "Where is AHU-5 located on the mechanical drawings?",
        "What is the status of submittal #232?",
        "Link RFI 1838 to construction bulletin",
        "Who is responsible for concrete testing?",
        "What's the installation sequence for the curtain wall?",
        "Show me shop drawings for structural steel on level 5",
        "Define what a fire damper is",
        "Translate this answer to Spanish",
        "What are the dimensions of door 20110-2?",
        "Which product replaces ACT-9?",
        "Add new user to the project",
        "Is this a question about construction?",
        "Convert 100 square feet to square meters"
    ]
    
    print(f"\nProcessing {len(test_queries)} test queries...\n")
    print("=" * 80)
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n[{i}] Query: {query}")
        print("-" * 40)
        
        try:
            # Classify the query
            result = pipeline.classify(query)
            
            # Display results
            print(f"Composition: {result.composition.mode}")
            print(f"Number of intents: {len(result.intents)}")
            
            for j, intent in enumerate(result.intents):
                print(f"\n  Intent {j+1}:")
                print(f"    Code: {intent.intent_code}")
                print(f"    Confidence: {intent.confidence:.2f}")
                if intent.entities:
                    print(f"    Entities: {intent.entities}")
                if intent.routing_hints.get('tools'):
                    print(f"    Tools: {intent.routing_hints['tools']}")
            
            # Show composition details for multi-intent
            if len(result.intents) > 1 and result.composition.join_keys:
                print(f"\n  Join Keys:")
                for join_key in result.composition.join_keys:
                    print(f"    {join_key.key}: from step {join_key.from_step} to step {join_key.to_step}")
        
        except Exception as e:
            print(f"  ERROR: {e}")
        
        print("=" * 80)
    
    print("\nClassification complete!")
    
    # Save results to file
    print("\nSaving results to 'classification_results.json'...")
    results = []
    for query in test_queries[:5]:  # Save first 5 for brevity
        result = pipeline.classify(query)
        results.append({
            "query": query,
            "result": result.dict()
        })
    
    with open('classification_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("Done!")


if __name__ == "__main__":
    main()
