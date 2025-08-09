#!/usr/bin/env python3
"""
Label training questions with two-stage intent classification
Stage 1: Coarse class (DOC, STAT, SPEC, etc.)
Stage 2: Fine subclass (specific operation)
"""

import json
import re
from typing import Dict, Tuple, List
import pandas as pd
from collections import defaultdict

# Define intent patterns for classification
INTENT_PATTERNS = {
    # DOC - Document Retrieval
    "DOC:SHOP_DRAWING_RETRIEVE": [
        r"shop draw",
        r"show.*shop",
        r"shop drawings? for",
        r"shop drawings? submitted",
        r"terrazzo.*shop"
    ],
    "DOC:SUBMITTAL_RETRIEVE": [
        r"submittal",
        r"show.*diffuser",
        r"light submittal",
        r"door submittal",
        r"rubber door submittal"
    ],
    "DOC:RFI_RETRIEVE": [
        r"rfi\s*#?\s*\d+",
        r"show.*rfi",
        r"latest rfi",
        r"what.*rfi",
        r"rfi.*on hold"
    ],
    "DOC:CB_RETRIEVE": [
        r"cb\s*\d+",
        r"change bulletin",
        r"pull.*cb"
    ],
    "DOC:DRAWING_PLAN_OR_DETAIL": [
        r"electrical.*diagram",
        r"riser diagram",
        r"floor plan",
        r"detail.*shows",
        r"architectural detail",
        r"window schedule",
        r"panel schedule",
        r"layout"
    ],
    "DOC:SCHEDULE_FILE_RETRIEVE": [
        r"schedule.*excel",
        r"schedule folder"
    ],
    
    # STAT - Record/Process Status
    "STAT:RFI_STATUS": [
        r"status.*rfi",
        r"rfi.*status",
        r"response.*rfi",
        r"rfi.*response"
    ],
    "STAT:SUBMITTAL_STATUS": [
        r"submittal.*approved",
        r"approved.*submittal",
        r"submittal.*status",
        r"submittal.*complete"
    ],
    "STAT:OBSERVATION_STATUS": [
        r"status.*observation",
        r"observation.*status",
        r"open.*observation",
        r"patch drywall.*status"
    ],
    "STAT:CHANGE_DOC_STATUS": [
        r"change document.*open",
        r"open.*change"
    ],
    
    # SPEC - Specifications & Code Lookups
    "SPEC:SECTION_MAP": [
        r"spec section",
        r"specification.*section",
        r"what spec.*references",
        r"spec.*cover",
        r"specification number"
    ],
    "SPEC:REQUIREMENT_RULE": [
        r"requirement",
        r"where can.*used",
        r"need.*caulk",
        r"mounting height",
        r"testing.*requirement",
        r"ada requirement",
        r"clearance",
        r"need to be",
        r"supposed to be"
    ],
    "SPEC:MOCKUP_REQUIREMENT": [
        r"mockup",
        r"mock-up",
        r"field sample"
    ],
    "SPEC:SUBMITTAL_REQUIREMENT": [
        r"shop drawing requirement",
        r"submittal.*requirement",
        r"as-built.*submittal"
    ],
    "SPEC:COMMISSIONING_REQUIREMENT": [
        r"commission",
        r"functional test",
        r"becx",
        r"cx requirement"
    ],
    "SPEC:ADA_ACCESSIBILITY": [
        r"ada",
        r"accessibility",
        r"handrail height",
        r"door.*force",
        r"differential pressure"
    ],
    "SPEC:CROSS_REFERENCE_LIST": [
        r"spec.*mention",
        r"owner training",
        r"maintenance.*spec"
    ],
    
    # ATTR - Attributes/Properties
    "ATTR:FINISH_COLOR": [
        r"color",
        r"paint color",
        r"what color",
        r"finish.*color",
        r"color.*paint"
    ],
    "ATTR:MATERIAL_OR_TYPE": [
        r"what material",
        r"what type",
        r"material.*is",
        r"type of"
    ],
    "ATTR:PRODUCT_ID_OR_PART": [
        r"product\s*#",
        r"product number",
        r"model.*door",
        r"product data"
    ],
    "ATTR:DIMENSION_OR_SIZE": [
        r"thickness",
        r"size",
        r"dimension",
        r"rough opening",
        r"how thick"
    ],
    "ATTR:WARRANTY_OR_DURATION": [
        r"warranty"
    ],
    "ATTR:WEIGHT_OR_RATING": [
        r"weight",
        r"psi",
        r"rating"
    ],
    
    # SYS - System Design
    "SYS:FLOW_RATE": [
        r"flow rate",
        r"water flow",
        r"chw flow",
        r"gpm"
    ],
    "SYS:PRESSURE": [
        r"pressure",
        r"inches.*water",
        r"psi"
    ],
    "SYS:TEMPERATURE_SETPOINT": [
        r"temp",
        r"temperature"
    ],
    "SYS:POWER_REQUIREMENTS": [
        r"power.*requirement",
        r"emergency power",
        r"power.*damper"
    ],
    "SYS:CAPACITY_AIR": [
        r"cfm",
        r"air.*output"
    ],
    
    # COUNT - Quantities
    "COUNT:UNITS_OR_ROOMS": [
        r"how many unit",
        r"number of unit",
        r"units.*floor"
    ],
    "COUNT:FIXTURES_OR_ELEMENTS": [
        r"how many",
        r"number of",
        r"count"
    ],
    "COUNT:ORGANIZATIONS_OR_VENDORS": [
        r"vendor.*project",
        r"how many vendor"
    ],
    
    # SCHED - Schedule/Timeline
    "SCHED:WHEN_START_OR_FINISH": [
        r"when.*start",
        r"when.*complete",
        r"when.*finish",
        r"completion date",
        r"when is"
    ],
    "SCHED:DURATION": [
        r"duration",
        r"how long"
    ],
    "SCHED:PREDECESSOR_OR_SEQUENCE": [
        r"precede",
        r"before",
        r"sequence"
    ],
    
    # RESP - Ownership/Responsibility
    "RESP:WHO_RESPONSIBLE": [
        r"who.*provide",
        r"who.*responsible",
        r"superintendent",
        r"manager"
    ],
    "RESP:SCOPE_OF_WORK": [
        r"scope",
        r"scope.*work"
    ],
    
    # LOC - Location
    "LOC:COMPONENT_LOCATION": [
        r"where.*is",
        r"location",
        r"where.*located"
    ],
    "LOC:APPLICABILITY_ZONE": [
        r"where.*used",
        r"where.*required",
        r"where.*apply"
    ],
    
    # DOOR - Door Hardware
    "DOOR:GROUP_FOR_OPENING": [
        r"door.*hardware group",
        r"what.*group.*door"
    ],
    "DOOR:OPENINGS_FOR_GROUP": [
        r"which.*opening.*group",
        r"opening.*use.*group"
    ],
    "DOOR:HARDWARE_SCHEDULE_RETRIEVE": [
        r"hardware schedule",
        r"show.*hardware.*door"
    ],
    "DOOR:POWERED_OR_ACCESS": [
        r"auto operator",
        r"door.*power",
        r"access control door"
    ],
    "DOOR:FINISH_COLOR": [
        r"door.*hinge.*color",
        r"entry door.*color"
    ],
    
    # PLAN - Unit Plans
    "PLAN:UNIT_LAYOUT": [
        r"unit.*layout",
        r"show.*unit\s+[a-z]\d+"
    ],
    
    # GLOS - Glossary
    "GLOS:ABBREVIATION_OR_CODE": [
        r"what is.*[A-Z]{2,}",
        r"what.*stand for",
        r"what does.*mean"
    ],
    
    # LINK - Lineage
    "LINK:RFI_TO_CB": [
        r"rfi.*cb",
        r"cb.*result.*rfi",
        r"captured.*cb"
    ],
    "LINK:CB_TO_RFI": [
        r"cb.*issued.*rfi"
    ],
    
    # TRAN - Translation
    "TRAN:TRANSLATE_ANSWER": [
        r"translate",
        r"spanish",
        r"arabic"
    ],
    
    # META - Project/App
    "META:APP_HELP": [
        r"application.*personal",
        r"app.*help"
    ],
    "META:PROJECT_FACT": [
        r"project address",
        r"address.*project"
    ],
    "META:AUTH_OR_ACCESS": [
        r"verification code",
        r"access code"
    ],
    
    # MISC - Ambiguous
    "MISC:CHITCHAT": [
        r"tell.*joke",
        r"hello",
        r"hi"
    ],
    "MISC:AMBIGUOUS": [
        r"^[a-z]+$",  # Single word
        r"^windows?$",
        r"^door$"
    ]
}

def classify_question(question: str) -> Tuple[str, str, float]:
    """
    Classify a question into coarse class and fine subclass
    Returns: (coarse_class, fine_subclass, confidence)
    """
    question_lower = question.lower().strip()
    
    # Score each subclass based on pattern matches
    scores = defaultdict(int)
    
    for subclass, patterns in INTENT_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, question_lower, re.IGNORECASE):
                scores[subclass] += 1
    
    if not scores:
        # Default to MISC:AMBIGUOUS if no patterns match
        return "MISC", "MISC:AMBIGUOUS", 0.3
    
    # Get the highest scoring subclass
    best_subclass = max(scores.items(), key=lambda x: x[1])
    subclass_full = best_subclass[0]
    confidence = min(best_subclass[1] / 3.0, 1.0)  # Normalize confidence
    
    # Extract coarse class
    coarse_class = subclass_full.split(':')[0]
    
    return coarse_class, subclass_full, confidence

def label_dataset(input_file: str, output_file: str):
    """
    Label the training dataset with intent classifications
    """
    # Load the data
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    # Label each question
    labeled_data = []
    intent_counts = defaultdict(int)
    coarse_counts = defaultdict(int)
    
    for item in data:
        question = item['question_text']
        coarse, fine, confidence = classify_question(question)
        
        labeled_item = {
            **item,
            'intent_coarse': coarse,
            'intent_fine': fine,
            'intent_confidence': round(confidence, 3)
        }
        
        labeled_data.append(labeled_item)
        intent_counts[fine] += 1
        coarse_counts[coarse] += 1
    
    # Save labeled data
    with open(output_file, 'w') as f:
        json.dump(labeled_data, f, indent=2)
    
    # Print statistics
    print(f"Labeled {len(labeled_data)} questions")
    print("\nCoarse Class Distribution:")
    for coarse, count in sorted(coarse_counts.items(), key=lambda x: x[1], reverse=True):
        pct = count / len(labeled_data) * 100
        print(f"  {coarse}: {count} ({pct:.1f}%)")
    
    print("\nTop 20 Fine Subclasses:")
    sorted_intents = sorted(intent_counts.items(), key=lambda x: x[1], reverse=True)[:20]
    for intent, count in sorted_intents:
        pct = count / len(labeled_data) * 100
        print(f"  {intent}: {count} ({pct:.1f}%)")
    
    # Save distribution summary
    summary = {
        'total_questions': len(labeled_data),
        'coarse_class_distribution': dict(coarse_counts),
        'fine_subclass_distribution': dict(intent_counts),
        'unique_coarse_classes': len(coarse_counts),
        'unique_fine_subclasses': len(intent_counts)
    }
    
    with open('intent_labeling_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    return labeled_data

def create_sample_review_file(labeled_data: List[Dict], sample_size: int = 50):
    """
    Create a sample file for manual review of classifications
    """
    import random
    random.seed(42)
    
    # Sample questions for review
    samples = random.sample(labeled_data, min(sample_size, len(labeled_data)))
    
    # Create review format
    review_data = []
    for item in samples:
        review_data.append({
            'question': item['question_text'],
            'predicted_coarse': item['intent_coarse'],
            'predicted_fine': item['intent_fine'],
            'confidence': item['intent_confidence'],
            'review_correct': None,  # To be filled during review
            'correct_coarse': None,  # To be filled if incorrect
            'correct_fine': None     # To be filled if incorrect
        })
    
    with open('intent_classification_review_sample.json', 'w') as f:
        json.dump(review_data, f, indent=2)
    
    print(f"\nCreated review sample with {len(review_data)} questions")

if __name__ == "__main__":
    # Label the training dataset
    print("Labeling training dataset with two-stage intent classification...")
    labeled_data = label_dataset('train_questions.json', 'train_questions_labeled.json')
    
    # Create sample for manual review
    create_sample_review_file(labeled_data)
    
    print("\nFiles created:")
    print("- train_questions_labeled.json (labeled training data)")
    print("- intent_labeling_summary.json (distribution statistics)")
    print("- intent_classification_review_sample.json (sample for manual review)")
