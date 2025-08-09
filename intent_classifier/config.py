"""
Configuration settings for the intent classification system.
"""

import os
from pathlib import Path
from typing import Dict, Any


# Base paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"
LOG_DIR = PROJECT_ROOT / "logs"

# Ensure directories exist
for dir_path in [DATA_DIR, MODEL_DIR, LOG_DIR]:
    dir_path.mkdir(exist_ok=True)

# File paths
TAXONOMY_PATH = PROJECT_ROOT / "taxonomy.json"
TRAIN_DATA_PATH = PROJECT_ROOT / "train_questions.json"
VAL_DATA_PATH = PROJECT_ROOT / "val_questions.json"
TEST_DATA_PATH = PROJECT_ROOT / "test_questions.json"

# Model settings
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
SPACY_MODEL = "en_core_web_sm"

# Classification settings
DEFAULT_CONFIG = {
    "confidence_threshold": 0.7,
    "ood_threshold": 0.5,
    "max_intents_per_query": 5,
    "enable_knn_backstop": True,
    "knn_k": 5,
    "enable_spell_correction": True,
    "min_clause_length": 3,  # Minimum words in a clause
    "max_clause_length": 100,  # Maximum words in a clause
}

# Labeling function settings
LF_CONFIG = {
    "min_votes_required": 1,
    "abstain_value": -1,
    "conflict_resolution": "majority",  # majority, confidence_weighted, or snorkel
}

# Entity extraction patterns
ENTITY_PATTERNS = {
    "rfi": r'(?:RFI|rfi)\s*[#:]?\s*(\d+)',
    "cb": r'(?:CB|cb|construction bulletin)\s*[#:]?\s*(\d+)',
    "submittal": r'(?:submittal|sub)\s*[#:]?\s*(\d+)',
    "csi_section": r'\b(\d{2})\s*[\-\.]?\s*(\d{2})\s*[\-\.]?\s*(\d{2})\b',
    "door_id": r'\b(\d{4,5})(?:-\d)?\b',
    "product_code": r'\b([A-Z]{1,4}-\d{1,3})\b',
    "floor": r'(?:level|floor|L)\s*(\d+|[A-Z]\d*)',
    "drawing_ref": r'\b([A-Z]+)-?(\d+(?:\.\d+)?)\b',
}

# Routing tool mappings
TOOL_MAPPINGS = {
    "DOC": ["DocumentSearch", "SubmittalSearch"],
    "SPEC": ["SpecIndex", "VectorSearch"],
    "STAT": ["StatusAPI", "SubmittalStatus"],
    "LINK": ["RfiCbGraph", "LinkageSearch"],
    "COUNT": ["CountAPI", "AggregationSearch"],
    "RESP": ["ResponsibilityMatrix", "ContactSearch"],
    "SCHED": ["ScheduleSearch", "SequenceAnalyzer"],
    "DRAW": ["DrawingSearch", "SheetIndex"],
    "LOC": ["LocationSearch", "SpatialQuery"],
    "DEF": ["DefinitionSearch", "GlossaryLookup"],
    "TRAN": ["TranslationService"],
    "PARAM": ["ParameterSearch", "AttributeLookup"],
    "PROD": ["ProductDatabase", "MaterialSearch"],
    "ADMIN": ["AdminPanel", "UserManagement"],
    "QRY": ["QueryAnalyzer", "IntentDetector"],
    "UNIT": ["UnitConverter", "MeasurementTools"],
}

# Logging configuration
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        },
    },
    "handlers": {
        "file": {
            "class": "logging.FileHandler",
            "filename": str(LOG_DIR / "intent_classifier.log"),
            "formatter": "standard",
        },
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "standard",
        },
    },
    "loggers": {
        "intent_classifier": {
            "handlers": ["file", "console"],
            "level": "INFO",
            "propagate": False,
        }
    },
}

# API Configuration
API_CONFIG = {
    "host": "0.0.0.0",
    "port": 5000,
    "debug": os.getenv("FLASK_DEBUG", "False").lower() == "true",
    "cors_origins": ["*"],
    "max_content_length": 1024 * 1024,  # 1MB max request size
    "rate_limit": "100 per minute",
}
