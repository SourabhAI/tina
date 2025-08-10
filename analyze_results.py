#!/usr/bin/env python3
"""
Analyze validation results to understand classification performance.
"""

import json
from collections import defaultdict, Counter
from typing import Dict, List, Any


def load_results(file_path: str = "validation_results.json") -> List[Dict[str, Any]]:
    """Load validation results from JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)


def analyze_by_intent(results: List[Dict[str, Any]]):
    """Analyze results grouped by intent code."""
    intent_groups = defaultdict(list)
    
    for result in results:
        if result['success']:
            for intent in result['intents']:
                intent_code = intent['intent_code']
                intent_groups[intent_code].append({
                    'question': result['question_text'],
                    'confidence': intent['confidence'],
                    'entities': intent['entities']
                })
    
    return intent_groups


def find_misclassifications(results: List[Dict[str, Any]]):
    """Find potential misclassifications based on keywords."""
    potential_issues = []
    
    keyword_to_intent = {
        'RFI': ['DOC:RFI_RETRIEVE', 'STAT:RFI_STATUS', 'COUNT:RFI_COUNT'],
        'submittal': ['DOC:SUBMITTAL_RETRIEVE', 'STAT:SUBMITTAL_STATUS'],
        'door': ['DOC:DOOR_SCHEDULE', 'PROD:PRODUCT_LOOKUP'],
        'spec': ['SPEC:SECTION_RETRIEVE', 'SPEC:SECTION_MAP'],
        'drawing': ['DRAW:DRAWING_RETRIEVE', 'DRAW:DRAWING_MAP']
    }
    
    for result in results:
        if result['success']:
            question = result['question_text'].lower()
            classified_intents = [i['intent_code'] for i in result['intents']]
            
            for keyword, expected_intents in keyword_to_intent.items():
                if keyword in question:
                    # Check if any expected intent was found
                    if not any(intent in classified_intents for intent in expected_intents):
                        # Check if it's a related intent at least
                        if not any(exp.split(':')[0] == cl.split(':')[0] 
                                 for exp in expected_intents 
                                 for cl in classified_intents):
                            potential_issues.append({
                                'question': result['question_text'],
                                'keyword': keyword,
                                'expected_intents': expected_intents,
                                'actual_intents': classified_intents
                            })
    
    return potential_issues


def analyze_entity_extraction(results: List[Dict[str, Any]]):
    """Analyze entity extraction performance."""
    entity_examples = defaultdict(list)
    
    for result in results:
        if result['success']:
            entities = result['entities']
            for entity_type, value in entities.items():
                entity_examples[entity_type].append({
                    'question': result['question_text'],
                    'value': value
                })
    
    return entity_examples


def print_analysis():
    """Print comprehensive analysis of validation results."""
    results = load_results()
    
    print("=" * 80)
    print("VALIDATION RESULTS ANALYSIS")
    print("=" * 80)
    
    # Group by intent
    intent_groups = analyze_by_intent(results)
    
    print("\nINTENT CLASSIFICATION EXAMPLES:")
    print("-" * 60)
    
    for intent_code, examples in sorted(intent_groups.items()):
        print(f"\n{intent_code} ({len(examples)} examples):")
        # Show up to 3 examples
        for ex in examples[:3]:
            print(f"  Q: {ex['question']}")
            print(f"     Conf: {ex['confidence']:.2f}, Entities: {list(ex['entities'].keys())}")
    
    # Find potential misclassifications
    issues = find_misclassifications(results)
    
    if issues:
        print("\n" + "=" * 80)
        print("POTENTIAL MISCLASSIFICATIONS:")
        print("-" * 60)
        
        for issue in issues[:10]:  # Show up to 10
            print(f"\nQ: {issue['question']}")
            print(f"  Keyword: '{issue['keyword']}'")
            print(f"  Expected: {issue['expected_intents']}")
            print(f"  Actual: {issue['actual_intents']}")
    
    # Analyze entities
    entity_examples = analyze_entity_extraction(results)
    
    print("\n" + "=" * 80)
    print("ENTITY EXTRACTION EXAMPLES:")
    print("-" * 60)
    
    for entity_type, examples in sorted(entity_examples.items()):
        print(f"\n{entity_type} ({len(examples)} examples):")
        # Show unique values
        unique_values = list(set(str(ex['value']) for ex in examples))[:5]
        for value in unique_values:
            print(f"  - {value}")
    
    # Unknown classifications
    unknown_questions = []
    for result in results:
        if result['success']:
            if any(i['intent_code'] == 'UNKNOWN:UNKNOWN' for i in result['intents']):
                unknown_questions.append(result['question_text'])
    
    if unknown_questions:
        print("\n" + "=" * 80)
        print(f"UNKNOWN CLASSIFICATIONS ({len(unknown_questions)} questions):")
        print("-" * 60)
        
        for q in unknown_questions[:10]:  # Show up to 10
            print(f"  - {q}")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    print_analysis()
