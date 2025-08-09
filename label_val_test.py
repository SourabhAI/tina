#!/usr/bin/env python3
"""
Label validation and test sets with the same intent classification
"""

import json
from label_intents import classify_question

# Label validation set
print("Labeling validation set...")
with open('val_questions.json', 'r') as f:
    val_data = json.load(f)

val_labeled = []
for item in val_data:
    coarse, fine, confidence = classify_question(item['question_text'])
    val_labeled.append({
        **item,
        'intent_coarse': coarse,
        'intent_fine': fine,
        'intent_confidence': round(confidence, 3)
    })

with open('val_questions_labeled.json', 'w') as f:
    json.dump(val_labeled, f, indent=2)

# Label test set
print("Labeling test set...")
with open('test_questions.json', 'r') as f:
    test_data = json.load(f)

test_labeled = []
for item in test_data:
    coarse, fine, confidence = classify_question(item['question_text'])
    test_labeled.append({
        **item,
        'intent_coarse': coarse,
        'intent_fine': fine,
        'intent_confidence': round(confidence, 3)
    })

with open('test_questions_labeled.json', 'w') as f:
    json.dump(test_labeled, f, indent=2)

print(f'\n✓ Labeled {len(val_labeled)} validation questions')
print(f'✓ Labeled {len(test_labeled)} test questions')
print('\nCreated files:')
print('- val_questions_labeled.json')
print('- test_questions_labeled.json')
