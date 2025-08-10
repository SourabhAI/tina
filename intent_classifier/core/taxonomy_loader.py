"""
Taxonomy loader module for the intent classification system.
Loads and manages the taxonomy structure from taxonomy.json.
"""

import logging
from typing import Dict, List, Any, Optional, Set, Tuple
from pathlib import Path
import json

from intent_classifier.utils.helpers import load_json_file
from intent_classifier.models.schemas import TaxonomyClass, TaxonomyEntity
from intent_classifier.config import TAXONOMY_PATH


logger = logging.getLogger(__name__)


class TaxonomyLoader:
    """
    Loads and manages the taxonomy structure.
    Serves as the single source of truth for intent classes and validation.
    """
    
    def __init__(self, taxonomy_path: Optional[str] = None):
        """
        Initialize the taxonomy loader.
        
        Args:
            taxonomy_path: Path to taxonomy.json file. If None, uses default from config.
        """
        self.taxonomy_path = Path(taxonomy_path) if taxonomy_path else TAXONOMY_PATH
        self.taxonomy_data = self._load_taxonomy()
        
        # Cache frequently accessed data
        self._coarse_classes = {}
        self._all_intent_codes = set()
        self._subclass_to_coarse = {}
        self._entity_schemas = {}
        self._allowed_chains = set()
        self._routing_defaults = {}
        
        # Build caches
        self._build_caches()
        
        logger.info(f"Taxonomy loaded: {len(self._coarse_classes)} coarse classes, "
                   f"{len(self._all_intent_codes)} total intent codes")
    
    def _load_taxonomy(self) -> Dict[str, Any]:
        """Load the taxonomy JSON file."""
        try:
            return load_json_file(str(self.taxonomy_path))
        except Exception as e:
            logger.error(f"Failed to load taxonomy from {self.taxonomy_path}: {e}")
            raise ValueError(f"Cannot load taxonomy: {e}")
    
    def _build_caches(self):
        """Build internal caches for efficient access."""
        # Process classes
        for class_data in self.taxonomy_data.get('classes', []):
            coarse_code = class_data['code']
            self._coarse_classes[coarse_code] = TaxonomyClass(**class_data)
            
            # Process subclasses
            for subclass in class_data.get('subclasses', []):
                intent_code = subclass['code']
                self._all_intent_codes.add(intent_code)
                
                # Map subclass to coarse class
                if ':' in intent_code:
                    self._subclass_to_coarse[intent_code] = coarse_code
                else:
                    # Handle case where subclass code doesn't include coarse prefix
                    full_code = f"{coarse_code}:{intent_code}"
                    self._all_intent_codes.add(full_code)
                    self._subclass_to_coarse[full_code] = coarse_code
        
        # Process entities
        for entity_data in self.taxonomy_data.get('entities_schema', []):
            entity = TaxonomyEntity(**entity_data)
            self._entity_schemas[entity.key] = entity
        
        # Process allowed chains
        for chain_rule in self.taxonomy_data.get('allowed_chains', []):
            if isinstance(chain_rule, dict):
                from_intent = chain_rule.get('from')
                to_intent = chain_rule.get('to')
                if from_intent and to_intent:
                    self._allowed_chains.add((from_intent, to_intent))
            elif isinstance(chain_rule, list) and len(chain_rule) == 2:
                self._allowed_chains.add(tuple(chain_rule))
        
        # Process routing defaults
        self._routing_defaults = self.taxonomy_data.get('routing_defaults', {})
    
    def get_labels(self) -> List[str]:
        """Get all valid intent codes."""
        return sorted(list(self._all_intent_codes))
    
    def get_coarse_classes(self) -> List[str]:
        """Get all coarse class codes."""
        return sorted(list(self._coarse_classes.keys()))
    
    def is_valid(self, intent_code: str) -> bool:
        """Check if an intent code is valid."""
        return intent_code in self._all_intent_codes
    
    def is_valid_coarse_class(self, coarse_class: str) -> bool:
        """Check if a coarse class is valid."""
        return coarse_class in self._coarse_classes
    
    def get_coarse_class(self, intent_code: str) -> Optional[str]:
        """Get the coarse class for an intent code."""
        if ':' in intent_code:
            return intent_code.split(':')[0]
        return self._subclass_to_coarse.get(intent_code)
    
    def get_subclasses(self, coarse_class: str) -> List[str]:
        """Get all subclass codes for a coarse class."""
        if coarse_class not in self._coarse_classes:
            return []
        
        subclasses = []
        class_data = self._coarse_classes[coarse_class]
        for subclass in class_data.subclasses:
            subclasses.append(subclass['code'])
        
        return subclasses
    
    def get_intent_info(self, intent_code: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about an intent."""
        coarse_class = self.get_coarse_class(intent_code)
        if not coarse_class or coarse_class not in self._coarse_classes:
            return None
        
        class_data = self._coarse_classes[coarse_class]
        for subclass in class_data.subclasses:
            if subclass['code'] == intent_code:
                # Add coarse_class to the returned info
                result = subclass.copy()
                result['coarse_class'] = coarse_class
                return result
        
        return None
    
    def get_required_entities(self, intent_code: str) -> List[str]:
        """Get required entities for an intent."""
        intent_info = self.get_intent_info(intent_code)
        if intent_info:
            return intent_info.get('required_entities', [])
        return []
    
    def get_optional_entities(self, intent_code: str) -> List[str]:
        """Get optional entities for an intent."""
        intent_info = self.get_intent_info(intent_code)
        if intent_info:
            return intent_info.get('optional_entities', [])
        return []
    
    def get_routing_hints(self, intent_code: str) -> Dict[str, Any]:
        """Get routing hints for an intent."""
        intent_info = self.get_intent_info(intent_code)
        if intent_info and 'routing_hints' in intent_info:
            return intent_info['routing_hints']
        
        # Fall back to coarse class defaults
        coarse_class = self.get_coarse_class(intent_code)
        if coarse_class in self._routing_defaults:
            return {
                'tools': self._get_default_tools(coarse_class),
                'few_shot_id': f"{coarse_class.lower()}_default_v1"
            }
        
        return {}
    
    def get_policies(self, intent_code: str) -> Dict[str, Any]:
        """Get policies for an intent."""
        # Check specific intent first
        intent_info = self.get_intent_info(intent_code)
        if intent_info and 'policies' in intent_info:
            return intent_info['policies']
        
        # Fall back to coarse class defaults
        coarse_class = self.get_coarse_class(intent_code)
        if coarse_class in self._routing_defaults:
            return self._routing_defaults[coarse_class].get('policies', {})
        
        return {'freshness': 'any'}  # Default policy
    
    def is_allowed_chain(self, from_intent: str, to_intent: str) -> bool:
        """Check if a chain from one intent to another is allowed."""
        # Check exact match
        if (from_intent, to_intent) in self._allowed_chains:
            return True
        
        # Check wildcard patterns directly
        if ('*', to_intent) in self._allowed_chains:
            return True
        if (from_intent, '*') in self._allowed_chains:
            return True
        
        # Check coarse class level
        from_coarse = self.get_coarse_class(from_intent)
        to_coarse = self.get_coarse_class(to_intent)
        
        if from_coarse and to_coarse:
            # Check if coarse classes can chain
            if (from_coarse, to_coarse) in self._allowed_chains:
                return True
            
            # Check wildcard patterns at coarse class level
            if (from_coarse, '*') in self._allowed_chains:
                return True
            if ('*', to_coarse) in self._allowed_chains:
                return True
        
        return False
    
    def get_entity_schema(self, entity_key: str) -> Optional[TaxonomyEntity]:
        """Get the schema for an entity."""
        return self._entity_schemas.get(entity_key)
    
    def get_entity_definitions(self) -> List[Dict[str, Any]]:
        """Get all entity definitions for the EntityExtractor."""
        return self.taxonomy_data.get('entities_schema', [])
    
    def validate_entity(self, entity_key: str, value: Any) -> bool:
        """Validate an entity value against its schema."""
        schema = self.get_entity_schema(entity_key)
        if not schema:
            return True  # Allow unknown entities
        
        # Type validation
        if schema.type == 'integer':
            if not isinstance(value, int):
                try:
                    int(value)
                except:
                    return False
        elif schema.type == 'string':
            if not isinstance(value, str):
                return False
        
        # Pattern validation
        if schema.pattern and isinstance(value, str):
            import re
            if not re.match(schema.pattern, value):
                return False
        
        # Enum validation
        if schema.enum and value not in schema.enum:
            return False
        
        return True
    
    def get_example_triggers(self, intent_code: str) -> List[str]:
        """Get example trigger phrases for an intent."""
        intent_info = self.get_intent_info(intent_code)
        if intent_info:
            return intent_info.get('example_triggers', [])
        return []
    
    def _get_default_tools(self, coarse_class: str) -> List[str]:
        """Get default tools for a coarse class."""
        # This would typically be configured in the taxonomy
        default_tools = {
            'DOC': ['DocumentSearch', 'SubmittalSearch'],
            'SPEC': ['SpecIndex', 'VectorSearch'],
            'STAT': ['StatusAPI'],
            'LINK': ['LinkageSearch'],
            'COUNT': ['CountAPI'],
            'RESP': ['ResponsibilityMatrix'],
            'SCHED': ['ScheduleSearch'],
            'DRAW': ['DrawingSearch'],
            'LOC': ['LocationSearch'],
            'DEF': ['GlossaryLookup'],
            'TRAN': ['TranslationService'],
            'PARAM': ['ParameterSearch'],
            'PROD': ['ProductDatabase'],
            'ADMIN': ['AdminPanel'],
            'QRY': ['QueryAnalyzer'],
            'UNIT': ['UnitConverter']
        }
        return default_tools.get(coarse_class, ['GeneralSearch'])
    
    def get_taxonomy_summary(self) -> Dict[str, Any]:
        """Get a summary of the taxonomy structure."""
        return {
            'taxonomy_id': self.taxonomy_data.get('taxonomy_id'),
            'version': self.taxonomy_data.get('taxonomy_version'),
            'description': self.taxonomy_data.get('description'),
            'total_coarse_classes': len(self._coarse_classes),
            'total_intent_codes': len(self._all_intent_codes),
            'total_entities': len(self._entity_schemas),
            'coarse_classes': self.get_coarse_classes(),
            'entity_keys': sorted(list(self._entity_schemas.keys()))
        }
    
    def get_all_chains(self) -> List[Tuple[str, str]]:
        """Get all allowed intent chains."""
        return sorted(list(self._allowed_chains))
    
    def search_intents(self, query: str) -> List[str]:
        """Search for intent codes matching a query string."""
        query_lower = query.lower()
        matches = []
        
        for intent_code in self._all_intent_codes:
            intent_info = self.get_intent_info(intent_code)
            if not intent_info:
                continue
            
            # Check code
            if query_lower in intent_code.lower():
                matches.append(intent_code)
                continue
            
            # Check description
            if 'description' in intent_info:
                if query_lower in intent_info['description'].lower():
                    matches.append(intent_code)
                    continue
            
            # Check example triggers
            for trigger in intent_info.get('example_triggers', []):
                if query_lower in trigger.lower():
                    matches.append(intent_code)
                    break
        
        return sorted(list(set(matches)))
