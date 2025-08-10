"""
Entity extractor module for the intent classification system.
Extracts entities from queries based on taxonomy-defined patterns and types.
"""

import re
import logging
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
from datetime import datetime

from intent_classifier.models.schemas import EntityExtractionResult


logger = logging.getLogger(__name__)


@dataclass
class EntityDefinition:
    """Definition of an entity from the taxonomy."""
    key: str
    type: str
    pattern: Optional[str] = None
    example: Optional[Any] = None
    enum: Optional[List[str]] = None
    description: Optional[str] = None


@dataclass 
class ExtractedEntity:
    """Represents an extracted entity with span information."""
    key: str
    value: Any
    span_start: int
    span_end: int
    confidence: float
    raw_text: str


class EntityExtractor:
    """
    Extracts entities from text based on taxonomy-defined patterns and rules.
    Handles IDs, spec sections, dates, locations, and domain-specific entities.
    """
    
    def __init__(self, entity_definitions: List[Dict[str, Any]]):
        """
        Initialize the entity extractor.
        
        Args:
            entity_definitions: List of entity definitions from taxonomy
        """
        self.entity_definitions = self._parse_definitions(entity_definitions)
        self._compile_patterns()
        self._initialize_extractors()
        
    def _parse_definitions(self, definitions: List[Dict[str, Any]]) -> Dict[str, EntityDefinition]:
        """
        Parse entity definitions into structured format.
        
        Args:
            definitions: Raw entity definitions from taxonomy
            
        Returns:
            Dictionary mapping entity keys to EntityDefinition objects
        """
        parsed = {}
        
        for defn in definitions:
            entity_def = EntityDefinition(
                key=defn["key"],
                type=defn["type"],
                pattern=defn.get("pattern"),
                example=defn.get("example"),
                enum=defn.get("enum"),
                description=defn.get("description")
            )
            parsed[entity_def.key] = entity_def
            
        return parsed
    
    def _compile_patterns(self):
        """Compile regex patterns for efficient extraction."""
        self.patterns = {}
        
        # ID patterns with variations
        self.patterns['ids.rfi'] = re.compile(
            r'\b(?:RFI|rfi)[\s:#-]*(\d{1,5})\b',
            re.IGNORECASE
        )
        
        self.patterns['ids.cb'] = re.compile(
            r'\b(?:CB|cb|bulletin|change\s*bulletin)[\s:#-]*(\d{1,5})\b',
            re.IGNORECASE
        )
        
        self.patterns['ids.submittal'] = re.compile(
            r'\b(?:submittal|sub|submission)[\s:#-]*(\d{1,5})\b',
            re.IGNORECASE
        )
        
        # Spec section pattern (handles various formats)
        self.patterns['spec_section'] = re.compile(
            r'\b(\d{2})[\s.-]?(\d{2})[\s.-]?(\d{2})\b'
        )
        
        # Product code pattern
        self.patterns['product_code'] = re.compile(
            r'\b([A-Z]{1,4})[-\s]?(\d{1,3})\b'
        )
        
        # Door ID pattern
        self.patterns['door_id'] = re.compile(
            r'\b(?:door|Door)[\s:#-]*(\d{4,5}(?:-\d)?)\b'
        )
        
        # Floor/level patterns
        self.patterns['floor'] = re.compile(
            r'\b(?:floor|level|lv|lvl|story|storey)[\s:#-]*(\d{1,2}|[A-Z]{1,2}\d?|basement|ground|roof|penthouse)\b',
            re.IGNORECASE
        )
        
        # Date patterns
        self.patterns['date'] = re.compile(
            r'\b(?:'
            r'(?:jan|january|feb|february|mar|march|apr|april|may|jun|june|'
            r'jul|july|aug|august|sep|september|oct|october|nov|november|dec|december)'
            r'[\s,]*\d{1,2}(?:st|nd|rd|th)?[\s,]*\d{2,4}|'
            r'\d{1,2}[-/]\d{1,2}[-/]\d{2,4}|'
            r'\d{4}[-/]\d{1,2}[-/]\d{1,2}|'
            r'(?:last|next|this)\s+(?:week|month|year)|'
            r'(?:yesterday|today|tomorrow)'
            r')\b',
            re.IGNORECASE
        )
        
        # Time range patterns
        self.patterns['date_range'] = re.compile(
            r'\b(?:'
            r'(?:from|between)\s+(.+?)\s+(?:to|and|through)\s+(.+?)|'
            r'(?:last|past|previous)\s+(\d+)\s+(?:days?|weeks?|months?)|'
            r'(?:in|during)\s+(january|february|march|april|may|june|july|august|september|october|november|december)(?:\s+\d{4})?'
            r')\b',
            re.IGNORECASE
        )
        
        # Area/location patterns
        self.patterns['area'] = re.compile(
            r'\b(?:in|at|near|around)\s+(?:the\s+)?'
            r'(lobby|atrium|conference\s*room|cafeteria|parking|garage|'
            r'corridor|hallway|stairwell|elevator|shaft|mechanical\s*room|'
            r'electrical\s*room|office|suite|lab|laboratory|restroom|bathroom|'
            r'kitchen|storage|warehouse|dock|loading\s*dock|roof|penthouse|'
            r'basement|plaza|courtyard|entrance|exit|reception|waiting\s*area)\b',
            re.IGNORECASE
        )
        
        # Discipline patterns
        self.patterns['discipline'] = re.compile(
            r'\b(mechanical|electrical|plumbing|structural|architectural|'
            r'civil|landscape|fire\s*protection|HVAC|MEP|IT|AV|security|'
            r'lighting|low\s*voltage)\b',
            re.IGNORECASE
        )
        
        # Compile custom patterns from entity definitions
        for key, entity_def in self.entity_definitions.items():
            if entity_def.pattern and key not in self.patterns:
                try:
                    self.patterns[key] = re.compile(entity_def.pattern)
                except re.error as e:
                    logger.warning(f"Invalid regex pattern for entity {key}: {e}")
    
    def _initialize_extractors(self):
        """Initialize specialized extractors for complex entities."""
        self.custom_extractors = {
            'topic': self._extract_topic,
            'submittal_topic': self._extract_submittal_topic,
            'answer_text': self._extract_answer_text,
            'target_language': self._extract_target_language
        }
    
    def extract(self, text: str, preprocessed_text: Optional[str] = None) -> EntityExtractionResult:
        """
        Extract entities from text.
        
        Args:
            text: Original query text
            preprocessed_text: Optional preprocessed version of text
            
        Returns:
            EntityExtractionResult with extracted entities
        """
        # Use preprocessed text if available, otherwise original
        working_text = preprocessed_text or text
        
        entities = {}
        entity_spans = []
        
        # Extract pattern-based entities
        for key, pattern in self.patterns.items():
            matches = list(pattern.finditer(working_text))
            
            for match in matches:
                # Extract value based on entity type
                value = self._extract_value(key, match, working_text)
                
                if value is not None:
                    # Check for duplicates
                    if key not in entities or self._is_better_match(entities[key], value):
                        entities[key] = value
                        
                        # Track span information
                        entity_spans.append({
                            'key': key,
                            'value': value,
                            'start': match.start(),
                            'end': match.end(),
                            'text': match.group()
                        })
        
        # Extract custom entities
        for key, extractor in self.custom_extractors.items():
            if key not in entities:  # Don't override pattern matches
                result = extractor(working_text)
                if result:
                    value, span = result
                    entities[key] = value
                    if span:
                        entity_spans.append({
                            'key': key,
                            'value': value,
                            'start': span[0],
                            'end': span[1],
                            'text': working_text[span[0]:span[1]]
                        })
        
        # Post-process entities
        entities = self._post_process_entities(entities, working_text)
        
        return EntityExtractionResult(
            entities=entities,
            entity_spans=entity_spans
        )
    
    def _extract_value(self, key: str, match: re.Match, text: str) -> Any:
        """
        Extract and convert value based on entity type.
        
        Args:
            key: Entity key
            match: Regex match object
            text: Full text
            
        Returns:
            Extracted value in appropriate type
        """
        entity_def = self.entity_definitions.get(key)
        
        if key.startswith('ids.'):
            # Extract numeric ID
            for group in match.groups():
                if group and group.isdigit():
                    return int(group)
            # Fallback to full match
            id_text = match.group().strip()
            id_match = re.search(r'\d+', id_text)
            if id_match:
                return int(id_match.group())
                
        elif key == 'spec_section':
            # Format spec section
            groups = match.groups()
            if len(groups) == 3:
                return f"{groups[0]} {groups[1]} {groups[2]}"
            return match.group().strip()
            
        elif key == 'product_code':
            # Format product code
            groups = match.groups()
            if len(groups) == 2:
                return f"{groups[0]}-{groups[1]}"
            return match.group().strip().upper()
            
        elif key == 'door_id':
            # Extract door ID
            for group in match.groups():
                if group:
                    return group
            return match.group().strip()
            
        elif key == 'floor':
            # Normalize floor
            floor_text = match.group(1) if match.groups() else match.group()
            return self._normalize_floor(floor_text)
            
        elif key == 'area':
            # Extract area name
            return match.group(1).strip()
            
        elif key == 'discipline':
            # Normalize discipline
            disc = match.group(1).strip()
            return self._normalize_discipline(disc)
            
        elif key == 'date' or key == 'date_range':
            # Keep as string for now (could parse to datetime)
            return match.group().strip()
            
        else:
            # Default: return matched text
            return match.group().strip()
    
    def _normalize_floor(self, floor_text: str) -> str:
        """
        Normalize floor/level references.
        
        Args:
            floor_text: Raw floor text
            
        Returns:
            Normalized floor string
        """
        floor_text = floor_text.strip().lower()
        
        # Handle special floors
        special_floors = {
            'basement': 'Basement',
            'ground': 'Ground',
            'roof': 'Roof',
            'penthouse': 'Penthouse',
            'g': 'Ground',
            'b': 'Basement',
            'r': 'Roof',
            'p': 'Penthouse'
        }
        
        if floor_text in special_floors:
            return special_floors[floor_text]
        
        # Handle numeric floors
        if floor_text.isdigit():
            return f"Level {floor_text}"
        
        # Handle alphanumeric (e.g., "2A", "B1")
        if re.match(r'^[A-Z]\d+$', floor_text.upper()):
            return f"Level {floor_text.upper()}"
        if re.match(r'^\d+[A-Z]$', floor_text.upper()):
            return f"Level {floor_text.upper()}"
        
        # Default
        return f"Level {floor_text}"
    
    def _normalize_discipline(self, discipline: str) -> str:
        """
        Normalize discipline names.
        
        Args:
            discipline: Raw discipline text
            
        Returns:
            Normalized discipline
        """
        discipline_map = {
            'mep': 'MEP',
            'hvac': 'HVAC',
            'it': 'IT',
            'av': 'AV',
            'fire protection': 'Fire Protection',
            'low voltage': 'Low Voltage',
            'mechanical': 'Mechanical',
            'electrical': 'Electrical',
            'plumbing': 'Plumbing',
            'structural': 'Structural',
            'architectural': 'Architectural',
            'civil': 'Civil',
            'landscape': 'Landscape',
            'security': 'Security',
            'lighting': 'Lighting'
        }
        
        normalized = discipline.lower().strip()
        return discipline_map.get(normalized, discipline.title())
    
    def _extract_topic(self, text: str) -> Optional[Tuple[str, Optional[Tuple[int, int]]]]:
        """
        Extract general topic from text.
        
        Args:
            text: Query text
            
        Returns:
            Tuple of (topic, span) or None
        """
        # Common topic indicators
        topic_patterns = [
            r'(?:about|regarding|concerning|for|related to)\s+(?:the\s+)?(.+?)(?:\?|$|,|\sand\s)',
            r'(?:submittal|drawing|spec|document|RFI|CB)\s+(?:for|about|regarding)\s+(?:the\s+)?(.+?)(?:\?|$|,|\sand\s)',
            r'(?:what is|what are|show me|find|get)\s+(?:the\s+)?(.+?)(?:\?|$|,|\sand\s)'
        ]
        
        for pattern in topic_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                topic = match.group(1).strip()
                # Clean up common suffixes
                topic = re.sub(r'\s+(?:submittal|drawing|spec|document|RFI|CB)s?$', '', topic, flags=re.IGNORECASE)
                
                if len(topic) > 3:  # Minimum topic length
                    return (topic, (match.start(1), match.end(1)))
        
        return None
    
    def _extract_submittal_topic(self, text: str) -> Optional[Tuple[str, Optional[Tuple[int, int]]]]:
        """
        Extract submittal-specific topic.
        
        Args:
            text: Query text
            
        Returns:
            Tuple of (topic, span) or None
        """
        # Submittal-specific patterns
        patterns = [
            r'submittal\s+(?:for|on|about|regarding)\s+(?:the\s+)?(.+?)(?:\?|$|,|\sand\s)',
            r'(.+?)\s+submittal(?:s)?(?:\?|$|,|\sand\s)',
            r'(?:show|find|get|retrieve)\s+(?:the\s+)?(.+?)\s+submittal',
            r'submittal\s+(?:#\d+\s+)?(?:for\s+)?(.+?)(?:\?|$|,|\sand\s)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                topic = match.group(1).strip()
                
                # Clean up topic
                topic = re.sub(r'^(?:the|a|an)\s+', '', topic)
                topic = re.sub(r'\s+(?:package|document|submission)s?$', '', topic)
                
                # Common submittal topics
                common_topics = [
                    'tile', 'flooring', 'carpet', 'paint', 'wall covering',
                    'doors', 'frames', 'hardware', 'glass', 'glazing',
                    'mechanical', 'electrical', 'plumbing', 'HVAC',
                    'structural steel', 'concrete', 'rebar', 'masonry',
                    'roofing', 'waterproofing', 'insulation', 'fireproofing',
                    'elevators', 'escalators', 'equipment', 'fixtures'
                ]
                
                # Check if it's a valid topic
                topic_lower = topic.lower()
                if any(common in topic_lower for common in common_topics) or len(topic) > 3:
                    return (topic, (match.start(1), match.end(1)))
        
        # Fallback to general topic extraction
        return self._extract_topic(text)
    
    def _extract_answer_text(self, text: str) -> Optional[Tuple[str, Optional[Tuple[int, int]]]]:
        """
        Extract answer text for translation intent.
        
        Args:
            text: Query text
            
        Returns:
            Tuple of (answer_text, span) or None
        """
        # Look for quoted text or text after "translate"
        patterns = [
            r'translate\s+"([^"]+)"',
            r"translate\s+'([^']+)'",
            r'translate\s+(?:the\s+)?(?:following|this):\s*(.+?)(?:\s+to\s+|$)',
            r'translate\s+(.+?)\s+(?:to|into)\s+',
            r'"([^"]+)"\s+(?:to|into)\s+(?:spanish|arabic|english)',
            r"'([^']+)'\s+(?:to|into)\s+(?:spanish|arabic|english)"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                answer = match.group(1).strip()
                if answer:
                    return (answer, (match.start(1), match.end(1)))
        
        return None
    
    def _extract_target_language(self, text: str) -> Optional[Tuple[str, Optional[Tuple[int, int]]]]:
        """
        Extract target language for translation.
        
        Args:
            text: Query text
            
        Returns:
            Tuple of (language_code, span) or None
        """
        # Language patterns
        language_map = {
            'english': 'en',
            'spanish': 'es',
            'arabic': 'ar',
            'en': 'en',
            'es': 'es',
            'ar': 'ar'
        }
        
        pattern = r'(?:to|into|in)\s+(english|spanish|arabic|en|es|ar)\b'
        match = re.search(pattern, text, re.IGNORECASE)
        
        if match:
            lang = match.group(1).lower()
            if lang in language_map:
                return (language_map[lang], (match.start(1), match.end(1)))
        
        return None
    
    def _is_better_match(self, existing: Any, new: Any) -> bool:
        """
        Determine if new match is better than existing.
        
        Args:
            existing: Current entity value
            new: New candidate value
            
        Returns:
            True if new is better
        """
        # For now, just keep the first match
        # Could implement more sophisticated logic
        return False
    
    def _post_process_entities(self, entities: Dict[str, Any], text: str) -> Dict[str, Any]:
        """
        Post-process extracted entities.
        
        Args:
            entities: Extracted entities
            text: Original text
            
        Returns:
            Post-processed entities
        """
        # Apply entity-specific validation
        processed = {}
        
        for key, value in entities.items():
            entity_def = self.entity_definitions.get(key)
            
            # Validate enum values
            if entity_def and entity_def.enum:
                if value not in entity_def.enum:
                    continue
            
            # Type conversion
            if entity_def:
                if entity_def.type == 'integer' and isinstance(value, str):
                    try:
                        value = int(value)
                    except ValueError:
                        continue
                elif entity_def.type == 'float' and isinstance(value, str):
                    try:
                        value = float(value)
                    except ValueError:
                        continue
            
            processed[key] = value
        
        return processed
    
    def get_required_entities(self, intent_code: str, taxonomy_loader) -> List[str]:
        """
        Get required entities for a specific intent.
        
        Args:
            intent_code: Intent code
            taxonomy_loader: TaxonomyLoader instance
            
        Returns:
            List of required entity keys
        """
        intent_info = taxonomy_loader.get_intent_info(intent_code)
        if intent_info:
            return intent_info.get('required_entities', [])
        return []
    
    def get_optional_entities(self, intent_code: str, taxonomy_loader) -> List[str]:
        """
        Get optional entities for a specific intent.
        
        Args:
            intent_code: Intent code
            taxonomy_loader: TaxonomyLoader instance
            
        Returns:
            List of optional entity keys
        """
        intent_info = taxonomy_loader.get_intent_info(intent_code)
        if intent_info:
            return intent_info.get('optional_entities', [])
        return []
    
    def validate_entities_for_intent(self, entities: Dict[str, Any], intent_code: str, 
                                   taxonomy_loader) -> Tuple[bool, List[str]]:
        """
        Validate if extracted entities satisfy intent requirements.
        
        Args:
            entities: Extracted entities
            intent_code: Intent code
            taxonomy_loader: TaxonomyLoader instance
            
        Returns:
            Tuple of (is_valid, missing_entities)
        """
        required = self.get_required_entities(intent_code, taxonomy_loader)
        missing = [entity for entity in required if entity not in entities]
        
        return len(missing) == 0, missing
