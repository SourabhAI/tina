# Quick Start Guide

## 5-Minute Setup

### 1. Install Dependencies
```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install packages
pip install -r requirements.txt

# Download NLP models
python setup_nlp.py
```

### 2. Train Models (First Time Only)
```bash
# Train classifier (~30 seconds)
python train_classifier.py

# Train advanced features (~1 minute)
python train_advanced_features.py
```

### 3. Test the System
```bash
# Quick test
python run_validation_test.py

# Full test with all features
python run_advanced_validation.py
```

## Example Usage

### Python
```python
from intent_classifier.main import create_pipeline

# Create pipeline
pipeline = create_pipeline()

# Classify a query
result = pipeline.classify("Show me all RFIs about concrete")
print(f"Intent: {result.intents[0].intent_code}")
print(f"Confidence: {result.intents[0].confidence:.2f}")
```

### Results
```
Intent: DOC:RFI_RETRIEVE
Confidence: 0.95
```

## Common Queries & Their Intents

| Query | Intent |
|-------|--------|
| "Show me RFI-123" | DOC:RFI_RETRIEVE |
| "What's the status of submittal S-001?" | STAT:SUBMITTAL_STATUS |
| "How many vendors are there?" | COUNT:VENDOR_COUNT |
| "Who is the general contractor?" | ADMIN:PERSONNEL |
| "Show door schedule for door 101" | SCHED:DOOR_SCHEDULE |
| "Translate to Spanish" | TRAN:TRANSLATE |

## Troubleshooting

- **Import errors**: Make sure virtual environment is activated
- **Model not found**: Run `python setup_nlp.py`
- **Low accuracy**: Train models with `python train_classifier.py`
