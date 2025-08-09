"""
Utility functions for the intent classification system.
"""

import re
import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_json_file(file_path: str) -> Dict[str, Any]:
    """Load and parse a JSON file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading JSON file {file_path}: {e}")
        raise


def normalize_id(text: str) -> str:
    """
    Normalize various ID formats to a consistent format.
    Examples:
        RFI #1838 -> RFI:1838
        RFI#1838 -> RFI:1838
        CB 309 -> CB:309
    """
    # RFI normalization
    text = re.sub(r'RFI\s*#?\s*(\d+)', r'RFI:\1', text, flags=re.IGNORECASE)
    
    # CB normalization
    text = re.sub(r'CB\s*#?\s*(\d+)', r'CB:\1', text, flags=re.IGNORECASE)
    
    # Submittal normalization
    text = re.sub(r'submittal\s*#?\s*(\d+)', r'submittal:\1', text, flags=re.IGNORECASE)
    
    return text


def normalize_csi_section(text: str) -> str:
    """
    Normalize CSI section numbers to consistent format.
    Examples:
        102233 -> 10 22 33
        10-22-33 -> 10 22 33
        10.22.33 -> 10 22 33
    """
    # Find CSI patterns
    csi_pattern = r'\b(\d{2})[\s\-\.]*(\d{2})[\s\-\.]*(\d{2})\b'
    text = re.sub(csi_pattern, r'\1 \2 \3', text)
    
    # Also handle 6-digit continuous format
    continuous_pattern = r'\b(\d{6})\b'
    def format_continuous(match):
        digits = match.group(1)
        return f"{digits[0:2]} {digits[2:4]} {digits[4:6]}"
    
    text = re.sub(continuous_pattern, format_continuous, text)
    
    return text


def extract_spans(text: str, pattern: str) -> List[Tuple[int, int, str]]:
    """
    Extract spans matching a pattern from text.
    Returns list of (start, end, matched_text) tuples.
    """
    spans = []
    for match in re.finditer(pattern, text):
        spans.append((match.start(), match.end(), match.group()))
    return spans


def clean_text(text: str) -> str:
    """
    Basic text cleaning while preserving important punctuation.
    """
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    # Preserve question marks but remove duplicate punctuation
    text = re.sub(r'([.!?])\1+', r'\1', text)
    
    return text.strip()


def is_question(text: str) -> bool:
    """Check if text is likely a question."""
    question_starters = ['what', 'where', 'when', 'who', 'why', 'how', 'is', 'are', 'can', 'will', 'does', 'do']
    text_lower = text.lower().strip()
    
    # Check for question mark
    if text.strip().endswith('?'):
        return True
    
    # Check for question starters
    first_word = text_lower.split()[0] if text_lower else ""
    return first_word in question_starters


def split_by_conjunctions(text: str) -> List[str]:
    """
    Simple conjunction-based splitting for multi-intent detection.
    This is a fallback when dependency parsing isn't sufficient.
    """
    # Common conjunctions that might indicate multiple intents
    conjunctions = [' and ', ' also ', ' then ', ' plus ', ' additionally ']
    
    segments = [text]
    for conj in conjunctions:
        new_segments = []
        for segment in segments:
            parts = segment.split(conj)
            new_segments.extend(parts)
        segments = new_segments
    
    # Clean up and filter empty segments
    segments = [s.strip() for s in segments if s.strip()]
    
    return segments


def calculate_overlap(span1: Tuple[int, int], span2: Tuple[int, int]) -> float:
    """Calculate overlap ratio between two spans."""
    start1, end1 = span1
    start2, end2 = span2
    
    overlap_start = max(start1, start2)
    overlap_end = min(end1, end2)
    
    if overlap_start >= overlap_end:
        return 0.0
    
    overlap_length = overlap_end - overlap_start
    span1_length = end1 - start1
    
    return overlap_length / span1_length if span1_length > 0 else 0.0


def merge_entities(entities_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Merge multiple entity dictionaries, handling conflicts.
    Later entities override earlier ones for the same key.
    """
    merged = {}
    for entities in entities_list:
        merged.update(entities)
    return merged


def validate_entity_value(entity_key: str, value: Any, entity_schema: Dict[str, Any]) -> bool:
    """
    Validate an entity value against its schema definition.
    """
    if 'type' in entity_schema:
        expected_type = entity_schema['type']
        if expected_type == 'integer' and not isinstance(value, int):
            try:
                int(value)  # Check if convertible
            except:
                return False
        elif expected_type == 'string' and not isinstance(value, str):
            return False
    
    if 'pattern' in entity_schema and isinstance(value, str):
        if not re.match(entity_schema['pattern'], value):
            return False
    
    if 'enum' in entity_schema:
        if value not in entity_schema['enum']:
            return False
    
    return True


def get_project_root() -> Path:
    """Get the project root directory."""
    current_file = Path(__file__)
    # Go up to the tina directory
    return current_file.parent.parent.parent


def load_taxonomy() -> Dict[str, Any]:
    """Load the taxonomy from the project root."""
    root = get_project_root()
    taxonomy_path = root / 'taxonomy.json'
    return load_json_file(str(taxonomy_path))


def format_intent_code(coarse_class: str, subclass: str) -> str:
    """Format a proper intent code from class and subclass."""
    if ':' in subclass:
        return subclass  # Already formatted
    return f"{coarse_class}:{subclass}"


class SpellingCorrector:
    """Simple domain-specific spelling corrector."""
    
    def __init__(self):
        self.corrections = {
            # Common misspellings in construction domain
            'specifcation': 'specification',
            'specificaiton': 'specification',
            'specifiction': 'specification',
            'submital': 'submittal',
            'submittals': 'submittal',
            'rfi': 'RFI',
            'rfis': 'RFI',
            'cb': 'CB',
            'cbs': 'CB',
            'drawigns': 'drawings',
            'drawinsg': 'drawings',
            'shedule': 'schedule',
            'sequnce': 'sequence',
            'sequance': 'sequence',
            'instaled': 'installed',
            'instalation': 'installation',
            'mechnical': 'mechanical',
            'electical': 'electrical',
            'structual': 'structural',
        }
    
    def correct(self, text: str) -> str:
        """Apply spelling corrections to text."""
        for wrong, right in self.corrections.items():
            # Case-insensitive replacement
            pattern = re.compile(re.escape(wrong), re.IGNORECASE)
            text = pattern.sub(right, text)
        return text
