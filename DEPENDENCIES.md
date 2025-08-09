# Intent Classification System Dependencies

## Overview
This document explains the dependencies required for the intent classification system and their purposes.

## Core Dependencies

### Data Processing
- **pandas** (≥1.5.0): Data manipulation and analysis
- **numpy** (≥1.23.0): Numerical computations

### Natural Language Processing
- **spacy** (≥3.7.0): Industrial-strength NLP for clause splitting, entity extraction, and dependency parsing
- **nltk** (≥3.8.0): Additional NLP utilities for tokenization and text processing
- **sentence-transformers** (≥2.2.0): Lightweight semantic embeddings for KNN backstop

### Machine Learning
- **scikit-learn** (≥1.2.0): Classic ML algorithms (SVM, LogisticRegression) for lightweight classification
- **scipy** (≥1.9.0): Scientific computing utilities
- **snorkel** (≥0.9.9): Weak supervision framework for programmatic labeling

### Data Validation
- **pydantic** (≥2.0.0): Data validation and schema enforcement

### API Development
- **flask** (≥3.0.0): Lightweight web framework for REST API
- **flask-cors** (≥4.0.0): CORS support for API

### Testing
- **pytest** (≥7.0.0): Testing framework
- **pytest-cov** (≥4.0.0): Code coverage reporting

### Utilities
- **python-dotenv** (≥1.0.0): Environment variable management
- **tqdm** (≥4.65.0): Progress bars for batch processing
- **joblib** (≥1.2.0): Efficient serialization and parallel processing

### Visualization (Optional)
- **matplotlib** (≥3.6.0): Plotting library for analysis
- **seaborn** (≥0.12.0): Statistical data visualization

## Installation Instructions

1. **Create and activate virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On macOS/Linux
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download NLP models:**
   ```bash
   python setup_nlp.py
   ```

## Why These Dependencies?

### Mac-Friendly Choices
- **Lightweight models**: Using spaCy's small English model and sentence-transformers' MiniLM instead of large transformers
- **Classic ML**: SVM and LogisticRegression are fast and efficient on Mac hardware
- **No GPU required**: All dependencies work well on CPU-only environments

### Fixed Taxonomy Constraints
- **Snorkel**: Enables programmatic labeling tied to our fixed taxonomy
- **Pydantic**: Ensures all outputs conform to the intent schema
- **Rule-based components**: Heavy use of regex and spaCy matchers for deterministic entity extraction

### Production Ready
- **Flask**: Simple and reliable for serving the API
- **Comprehensive testing**: pytest suite ensures reliability
- **Well-maintained**: All dependencies are actively maintained and stable

## Memory and Performance Considerations

- **Total disk space**: ~2GB (including models)
- **RAM usage**: ~1-2GB during operation
- **CPU usage**: Optimized for single-core performance on Mac

## Troubleshooting

If you encounter issues:
1. Ensure Python 3.8+ is installed
2. Update pip: `pip install --upgrade pip`
3. For M1/M2 Macs, some packages may need: `pip install --no-binary :all: <package>`
