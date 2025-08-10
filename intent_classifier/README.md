# Intent Classification System

A multi-intent classification system for construction domain queries, built with a fixed taxonomy and optimized for Mac hardware.

## Architecture Overview

```
Query Input
    ↓
Preprocessor (normalization, spelling)
    ↓
Clause Splitter (multi-intent detection)
    ↓
Entity Extractor (IDs, codes, references)
    ↓
Labeling Functions (weak supervision)
    ↓
Supervised Classifier (SVM/LogReg)
    ↓
KNN Backstop (similarity check)
    ↓
Confidence Calibration
    ↓
Composition Builder (SEQUENCE/PARALLEL)
    ↓
Router (policies & hints)
    ↓
Intent Output (schema-compliant)
```

## Directory Structure

```
intent_classifier/
├── core/                      # Core pipeline components
│   ├── taxonomy_loader.py     # Load and validate taxonomy
│   ├── preprocessor.py        # Text normalization
│   ├── clause_splitter.py     # Multi-intent splitting
│   ├── entity_extractor.py    # Entity recognition
│   ├── labeling_functions.py  # Weak supervision rules
│   ├── classifier.py          # Main classifier
│   ├── knn_backstop.py       # Similarity-based fallback
│   ├── confidence.py         # Confidence calibration
│   ├── composition_builder.py # Multi-intent composition
│   └── router.py             # Routing hints
├── models/                   # Data models
│   └── schemas.py           # Pydantic schemas
├── utils/                   # Utilities
│   └── helpers.py          # Helper functions
├── api/                    # REST API
├── tests/                  # Test suite
├── data/                   # Data storage
├── scripts/                # Utility scripts
├── config.py              # Configuration
└── main.py               # Pipeline orchestrator
```

## Key Features

1. **Fixed Taxonomy**: 16 coarse classes with predefined sub-classes
2. **Multi-Intent Support**: Handles compound queries with SEQUENCE/PARALLEL composition
3. **Entity-First**: Heavy emphasis on extracting IDs, codes, and references
4. **Weak Supervision**: Programmatic labeling functions for rule-based classification
5. **Mac-Optimized**: Lightweight models suitable for CPU-only environments
6. **Schema-Compliant**: Outputs match the defined intent-output schema

## Usage

```python
from intent_classifier.main import IntentClassificationPipeline

# Initialize pipeline
pipeline = IntentClassificationPipeline()

# Classify a query
result = pipeline.classify("Show me the tile submittal and is it approved?")

# Access results
for intent in result.intents:
    print(f"Intent: {intent.intent_code}")
    print(f"Confidence: {intent.confidence}")
    print(f"Entities: {intent.entities}")
```

## Component Details

### Taxonomy Loader
- Loads taxonomy.json as the single source of truth
- Validates intent codes and entity schemas
- Provides allowed intent chains for composition

### Preprocessor
- Case normalization
- Spelling correction for domain terms
- ID canonicalization (RFI #123 → RFI:123)
- CSI section formatting (102233 → 10 22 33)

### Clause Splitter
- SpaCy-based dependency parsing
- Conjunction and punctuation heuristics
- Preserves offsets for reconstruction

### Entity Extractor
- Regex patterns for structured IDs
- SpaCy Matcher for domain entities
- Validates against taxonomy schema

### Labeling Functions
- 30-60 domain-specific rules
- Maps patterns to fixed intent codes
- Snorkel-style vote aggregation

### Classifier
- Two-stage classification:
  1. Coarse class prediction (16 classes)
  2. Sub-class prediction within coarse class
- TF-IDF + char n-grams features
- Linear SVM/Logistic Regression

### KNN Backstop
- Sentence embeddings (all-MiniLM-L6-v2)
- Prototype examples for each intent
- Activated for low-confidence predictions

### Composition Builder
- Determines SEQUENCE vs PARALLEL execution
- Extracts join keys between intents
- Validates against allowed chains

### Router
- Maps intents to tools and policies
- Adds freshness requirements
- Provides few-shot IDs for prompting

## Configuration

See `config.py` for all configurable parameters including:
- Confidence thresholds
- Entity patterns
- Tool mappings
- API settings

## Testing

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest --cov=intent_classifier tests/
```

## Performance Targets

- Latency: <100ms per query
- Accuracy: >85% on coarse classes
- Multi-intent: >80% correct detection
- Memory: <2GB runtime
