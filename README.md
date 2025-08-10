# TINA - Trunk Tools Intent Classification System

A sophisticated intent classification system for construction industry queries using weak supervision, supervised learning, and advanced NLP techniques.

## Overview

This system classifies user queries into specific intents related to construction project management, including:
- Document retrieval (RFIs, submittals, drawings)
- Status tracking
- Specifications lookup
- Schedule queries
- Product information
- Administrative tasks

## Features

- **Weak Supervision**: 31 programmatic labeling functions for initial classification
- **Supervised Learning**: Trained classifier using weak labels
- **Advanced NLP**: SpaCy integration for multi-intent detection
- **KNN Backstop**: Similar query matching for edge cases
- **Confidence Calibration**: Reliable probability scores
- **Multi-Intent Support**: Handles complex queries with multiple intents

## Installation

### Prerequisites

- Python 3.8+
- pip
- ~500MB disk space for models and dependencies

### Setup

1. Clone the repository:
```bash
git clone https://github.com/SourabhAI/tina.git
cd tina
```

2. Create a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download NLP models:
```bash
python setup_nlp.py
```

## Quick Start

### 1. Split Dataset (if you have raw questions.json)
```bash
python split_dataset.py
```
This creates train/validation/test splits with unique query IDs.

### 2. Train Models

#### Option A: Train all models at once
```bash
# Train classifier using weak supervision
python train_classifier.py

# Train advanced features (KNN + calibration)
python train_advanced_features.py
```

#### Option B: Train specific components
```bash
# Train only the classifier
python train_classifier.py --model-type logistic

# Train only KNN backstop
python train_advanced_features.py --feature knn

# Train only confidence calibration
python train_advanced_features.py --feature calibration
```

### 3. Run Validation Tests

#### Basic pipeline (fast, no advanced features):
```bash
python run_validation_test.py
```

#### Advanced pipeline (all features enabled):
```bash
python run_advanced_validation.py
```

### 4. Compare Results
```bash
python compare_results.py
```

## Project Structure

```
tina/
├── intent_classifier/          # Core classification system
│   ├── core/                  # Core modules
│   │   ├── taxonomy_loader.py # Intent taxonomy management
│   │   ├── preprocessor.py    # Text preprocessing
│   │   ├── clause_splitter.py # Multi-intent splitting
│   │   ├── entity_extractor.py# Entity extraction
│   │   ├── labeling_functions.py # Weak supervision
│   │   ├── classifier.py      # Supervised classifier
│   │   ├── knn_backstop.py    # KNN fallback
│   │   ├── confidence.py      # Calibration
│   │   ├── composition_builder.py # Multi-intent composition
│   │   └── router.py          # Intent routing
│   ├── models/               # Data schemas
│   ├── utils/                # Utilities
│   └── main.py              # Pipeline orchestrator
├── models/                   # Trained model files
│   ├── intent_classifier.pkl
│   ├── knn_backstop.pkl
│   └── confidence_calibrator.pkl
├── data files/
│   ├── questions.json        # Raw questions
│   ├── train_questions.json  # Training set
│   ├── val_questions.json    # Validation set
│   ├── test_questions.json   # Test set
│   └── taxonomy.json         # Intent definitions
└── scripts/
    ├── train_classifier.py
    ├── train_advanced_features.py
    ├── run_validation_test.py
    └── compare_results.py
```

## Usage Examples

### Python API

```python
from intent_classifier.main import create_pipeline, PipelineConfig

# Basic usage
pipeline = create_pipeline()
result = pipeline.classify("Show me all RFIs about concrete")
print(f"Intent: {result.intents[0].intent_code}")
print(f"Confidence: {result.intents[0].confidence}")

# Advanced configuration
config = PipelineConfig(
    use_spacy_splitter=True,
    use_knn_backstop=True,
    use_confidence_calibration=True
)
pipeline = create_pipeline(config)

# Batch processing
questions = ["What is RFI-123?", "Show door schedule", "Translate to Spanish"]
for q in questions:
    result = pipeline.classify(q)
    print(f"{q} -> {result.intents[0].intent_code}")
```

### Command Line

```bash
# Process a single query
echo "What is the status of submittal S-001?" | python -m intent_classifier.main

# Process a file
python -m intent_classifier.main --input queries.txt --output results.json
```

## Performance

### Basic Pipeline
- **Speed**: ~2ms per query
- **Accuracy**: 75% coverage
- **Memory**: ~200MB

### Advanced Pipeline
- **Speed**: ~6ms per query
- **Accuracy**: 100% coverage
- **Memory**: ~400MB
- **Multi-intent**: 5x better detection

## Configuration

Edit `intent_classifier/config.py` or pass `PipelineConfig`:

```python
PipelineConfig(
    # Feature toggles
    use_spacy_splitter=True,      # Advanced clause splitting
    use_knn_backstop=True,         # Similar query matching
    use_confidence_calibration=True, # Probability calibration
    enable_spell_correction=False,  # Spell check (slower)
    
    # Thresholds
    min_confidence_threshold=0.3,   # Minimum acceptable confidence
    knn_override_threshold=0.75,    # When KNN overrides classifier
    
    # Performance
    debug_mode=False,              # Detailed logging
    return_intermediate_results=True # Include processing details
)
```

## Training Your Own Models

1. Prepare your data in the same format as `questions.json`
2. Update `taxonomy.json` with your intent definitions
3. Run the training scripts as shown above
4. Models will be saved in the `models/` directory

## Troubleshooting

### Common Issues

1. **ModuleNotFoundError**: Ensure virtual environment is activated
2. **SpaCy model not found**: Run `python setup_nlp.py`
3. **Out of memory**: Disable advanced features or reduce batch size
4. **Slow performance**: Disable spell correction and debug mode

### Logs

Check logs for detailed error information:
```bash
python run_validation_test.py 2>&1 | tee validation.log
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

[Your License Here]

## Acknowledgments

- SpaCy for NLP capabilities
- Scikit-learn for machine learning
- NLTK for additional NLP tools
