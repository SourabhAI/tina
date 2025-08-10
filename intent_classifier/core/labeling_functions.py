"""
Programmatic labeling functions for weak supervision.
Uses patterns, keywords, and heuristics to generate initial intent labels.
"""

import re
import logging
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass
from collections import defaultdict

from intent_classifier.models.schemas import LabelingFunctionVote, EntityExtractionResult
from intent_classifier.core.entity_extractor import EntityExtractor
from intent_classifier.core.taxonomy_loader import TaxonomyLoader


logger = logging.getLogger(__name__)


@dataclass
class LabelingFunctionResult:
    """Result from a labeling function."""
    intent_code: Optional[str] = None
    confidence: float = 0.0
    abstain: bool = True
    
    def to_vote(self, function_name: str) -> Optional[LabelingFunctionVote]:
        """Convert to a vote object."""
        if not self.abstain and self.intent_code:
            return LabelingFunctionVote(
                function_name=function_name,
                intent_code=self.intent_code,
                confidence=self.confidence
            )
        return None


class LabelingFunctions:
    """
    Collection of labeling functions for intent classification.
    Each function looks for specific patterns and returns a vote or abstains.
    """
    
    def __init__(self, taxonomy_loader: TaxonomyLoader, entity_extractor: EntityExtractor):
        """
        Initialize labeling functions.
        
        Args:
            taxonomy_loader: Taxonomy loader instance
            entity_extractor: Entity extractor instance
        """
        self.taxonomy = taxonomy_loader
        self.entity_extractor = entity_extractor
        
        # Compile patterns
        self._compile_patterns()
        
        # Initialize keyword mappings
        self._build_keyword_mappings()
        
        # List of all labeling functions
        self.functions = [
            self.lf_submittal_keywords,
            self.lf_rfi_keywords,
            self.lf_cb_keywords,
            self.lf_spec_keywords,
            self.lf_status_keywords,
            self.lf_count_keywords,
            self.lf_schedule_keywords,
            self.lf_drawing_keywords,
            self.lf_location_keywords,
            self.lf_definition_keywords,
            self.lf_translation_keywords,
            self.lf_parameter_keywords,
            self.lf_entity_based,
            self.lf_question_patterns,
            self.lf_imperative_patterns,
            self.lf_linking_patterns,
            self.lf_response_patterns,
            self.lf_product_patterns,
            self.lf_admin_patterns,
            self.lf_query_patterns,
            self.lf_unit_patterns,
            # New enhanced functions
            self.lf_personnel_patterns,
            self.lf_enhanced_drawing_patterns,
            self.lf_color_material_patterns,
            self.lf_project_info_patterns,
            self.lf_vendor_patterns,
            self.lf_requirement_patterns,
            self.lf_activity_schedule_patterns,
            self.lf_equipment_patterns,
            self.lf_door_hardware_patterns,
            self.lf_enhanced_definition_patterns
        ]
    
    def _compile_patterns(self):
        """Compile regex patterns for efficiency."""
        # Document patterns
        self.submittal_pattern = re.compile(
            r'\b(?:submittal|submission|sub)\b.*\b(?:for|about|regarding|on)\b',
            re.IGNORECASE
        )
        
        self.shop_drawing_pattern = re.compile(
            r'\b(?:shop\s*drawing|shop\s*dwg|shops)\b',
            re.IGNORECASE
        )
        
        self.rfi_pattern = re.compile(
            r'\b(?:RFI|request\s*for\s*information)\b',
            re.IGNORECASE
        )
        
        self.cb_pattern = re.compile(
            r'\b(?:CB|change\s*bulletin|bulletin)\b',
            re.IGNORECASE
        )
        
        # Spec patterns
        self.spec_section_pattern = re.compile(
            r'\b(?:spec|specification)\s*(?:section|for)\b',
            re.IGNORECASE
        )
        
        self.requirement_pattern = re.compile(
            r'\b(?:requirement|required|requires|must|shall)\b',
            re.IGNORECASE
        )
        
        # Status patterns
        self.status_pattern = re.compile(
            r'\b(?:status|state|progress|approved|rejected|pending)\b',
            re.IGNORECASE
        )
        
        # Count patterns
        self.count_pattern = re.compile(
            r'\b(?:how\s*many|count|number\s*of|total|quantity)\b',
            re.IGNORECASE
        )
        
        # Schedule patterns
        self.schedule_pattern = re.compile(
            r'\b(?:schedule|door\s*schedule|equipment\s*schedule|finish\s*schedule)\b',
            re.IGNORECASE
        )
        
        # Drawing patterns
        self.drawing_pattern = re.compile(
            r'\b(?:drawing|dwg|plan|detail|section|elevation)\b',
            re.IGNORECASE
        )
        
        # Location patterns
        self.location_pattern = re.compile(
            r'\b(?:where|location|located|position|place)\b',
            re.IGNORECASE
        )
        
        # Definition patterns
        self.definition_pattern = re.compile(
            r'\b(?:what\s*is|what\s*are|define|definition|meaning|explain)\b',
            re.IGNORECASE
        )
        
        # Translation patterns
        self.translation_pattern = re.compile(
            r'\b(?:translate|translation|convert.*language|in\s*(?:spanish|arabic|english))\b',
            re.IGNORECASE
        )
        
        # Response patterns
        self.response_pattern = re.compile(
            r'\b(?:response|answer|reply|feedback)\b',
            re.IGNORECASE
        )
        
        # Link patterns
        self.link_pattern = re.compile(
            r'\b(?:link|related|connected|associated|tied\s*to)\b',
            re.IGNORECASE
        )
        
        # Parameter patterns
        self.parameter_pattern = re.compile(
            r'\b(?:parameter|dimension|size|measurement|specification)\b',
            re.IGNORECASE
        )
        
        # Product patterns
        self.product_pattern = re.compile(
            r'\b(?:product|material|item|equipment|fixture)\b',
            re.IGNORECASE
        )
        
        # Question word patterns
        self.what_pattern = re.compile(r'^\s*what\b', re.IGNORECASE)
        self.where_pattern = re.compile(r'^\s*where\b', re.IGNORECASE)
        self.how_many_pattern = re.compile(r'^\s*how\s*many\b', re.IGNORECASE)
        self.when_pattern = re.compile(r'^\s*when\b', re.IGNORECASE)
        self.is_pattern = re.compile(r'^\s*(?:is|are|was|were)\b', re.IGNORECASE)
        
        # Imperative patterns
        self.show_pattern = re.compile(r'^\s*(?:show|display|list)\b', re.IGNORECASE)
        self.find_pattern = re.compile(r'^\s*(?:find|search|locate|get)\b', re.IGNORECASE)
        self.check_pattern = re.compile(r'^\s*(?:check|verify|confirm)\b', re.IGNORECASE)
        self.calculate_pattern = re.compile(r'^\s*(?:calculate|compute|count)\b', re.IGNORECASE)
    
    def _build_keyword_mappings(self):
        """Build keyword to intent mappings from taxonomy."""
        self.keyword_to_intent = defaultdict(list)
        
        # Manual keyword mappings based on domain knowledge
        keyword_mappings = {
            # DOC intents
            'submittal': ['DOC:SUBMITTAL_RETRIEVE'],
            'shop drawing': ['DOC:SHOP_DRAWING_RETRIEVE'],
            'shops': ['DOC:SHOP_DRAWING_RETRIEVE'],
            'rfi': ['DOC:RFI_RETRIEVE'],
            'change bulletin': ['DOC:CB_RETRIEVE'],
            'cb': ['DOC:CB_RETRIEVE'],
            'bulletin': ['DOC:CB_RETRIEVE'],
            'form': ['DOC:FORM_RETRIEVE'],
            'template': ['DOC:FORM_RETRIEVE'],
            
            # SPEC intents
            'spec section': ['SPEC:SECTION_MAP'],
            'specification': ['SPEC:SECTION_MAP', 'SPEC:REQUIREMENT_RULE'],
            'requirement': ['SPEC:REQUIREMENT_RULE'],
            'required': ['SPEC:REQUIREMENT_RULE'],
            
            # STAT intents
            'status': ['STAT:RFI_STATUS', 'STAT:SUBMITTAL_STATUS', 'STAT:CB_STATUS'],
            'approved': ['STAT:SUBMITTAL_STATUS'],
            'rejected': ['STAT:SUBMITTAL_STATUS'],
            'pending': ['STAT:SUBMITTAL_STATUS'],
            
            # LINK intents
            'linked': ['LINK:RFI_TO_CB', 'LINK:CB_TO_SPEC'],
            'related': ['LINK:RFI_TO_CB', 'LINK:CB_TO_SPEC'],
            'associated': ['LINK:RFI_TO_CB', 'LINK:CB_TO_SPEC'],
            
            # COUNT intents
            'how many': ['COUNT:RFI_COUNT', 'COUNT:SUBMITTAL_COUNT'],
            'count': ['COUNT:RFI_COUNT', 'COUNT:SUBMITTAL_COUNT'],
            'total': ['COUNT:RFI_COUNT', 'COUNT:SUBMITTAL_COUNT'],
            
            # RESP intents
            'response': ['RESP:RFI_RESPONSE'],
            'answer': ['RESP:RFI_RESPONSE'],
            'reply': ['RESP:RFI_RESPONSE'],
            
            # SCHED intents
            'door schedule': ['SCHED:DOOR_SCHEDULE'],
            'equipment schedule': ['SCHED:EQUIPMENT_SCHEDULE'],
            'finish schedule': ['SCHED:FINISH_SCHEDULE'],
            
            # DRAW intents
            'drawing': ['DRAW:FIND_DRAWING'],
            'plan': ['DRAW:FIND_DRAWING'],
            'detail': ['DRAW:FIND_DRAWING'],
            'revision': ['DRAW:LATEST_REVISION'],
            'latest': ['DRAW:LATEST_REVISION'],
            
            # LOC intents
            'where': ['LOC:PHYSICAL_LOCATION', 'LOC:DRAWING_LOCATION'],
            'location': ['LOC:PHYSICAL_LOCATION', 'LOC:DRAWING_LOCATION'],
            'located': ['LOC:PHYSICAL_LOCATION', 'LOC:DRAWING_LOCATION'],
            
            # DEF intents
            'what is': ['DEF:ACRONYM', 'DEF:TERM'],
            'define': ['DEF:ACRONYM', 'DEF:TERM'],
            'meaning': ['DEF:ACRONYM', 'DEF:TERM'],
            
            # TRAN intents
            'translate': ['TRAN:TRANSLATE'],
            'translation': ['TRAN:TRANSLATE'],
            'spanish': ['TRAN:TRANSLATE'],
            'arabic': ['TRAN:TRANSLATE'],
            
            # PARAM intents
            'dimension': ['PARAM:DIMENSIONS'],
            'size': ['PARAM:DIMENSIONS'],
            'measurement': ['PARAM:DIMENSIONS'],
            
            # PROD intents
            'product': ['PROD:PRODUCT_LOOKUP'],
            'material': ['PROD:PRODUCT_LOOKUP'],
            'manufacturer': ['PROD:PRODUCT_LOOKUP'],
            
            # ADMIN intents
            'permission': ['ADMIN:PERMISSIONS'],
            'access': ['ADMIN:PERMISSIONS'],
            'help': ['ADMIN:HELP'],
            
            # QRY intents
            'query': ['QRY:QUERY_BUILDER'],
            'search': ['QRY:QUERY_BUILDER'],
            'filter': ['QRY:QUERY_BUILDER'],
            
            # UNIT intents
            'convert': ['UNIT:CONVERSION'],
            'conversion': ['UNIT:CONVERSION'],
            'unit': ['UNIT:CONVERSION']
        }
        
        # Build the mapping
        for keyword, intents in keyword_mappings.items():
            self.keyword_to_intent[keyword.lower()].extend(intents)
    
    def apply_all(self, text: str, entities: Dict[str, Any]) -> List[LabelingFunctionVote]:
        """
        Apply all labeling functions to text.
        
        Args:
            text: Query text (preprocessed)
            entities: Extracted entities
            
        Returns:
            List of votes from labeling functions
        """
        votes = []
        
        for func in self.functions:
            result = func(text, entities)
            if result and not result.abstain:
                vote = result.to_vote(func.__name__)
                if vote:
                    votes.append(vote)
        
        return votes
    
    def lf_submittal_keywords(self, text: str, entities: Dict[str, Any]) -> LabelingFunctionResult:
        """Label submittal-related queries."""
        if self.submittal_pattern.search(text) or 'submittal_topic' in entities:
            # Check for status keywords
            if self.status_pattern.search(text):
                return LabelingFunctionResult(
                    intent_code="STAT:SUBMITTAL_STATUS",
                    confidence=0.8,
                    abstain=False
                )
            else:
                return LabelingFunctionResult(
                    intent_code="DOC:SUBMITTAL_RETRIEVE",
                    confidence=0.9,
                    abstain=False
                )
        
        return LabelingFunctionResult()
    
    def lf_rfi_keywords(self, text: str, entities: Dict[str, Any]) -> LabelingFunctionResult:
        """Label RFI-related queries."""
        if self.rfi_pattern.search(text) or 'ids.rfi' in entities:
            # Check for status
            if self.status_pattern.search(text):
                return LabelingFunctionResult(
                    intent_code="STAT:RFI_STATUS",
                    confidence=0.9,
                    abstain=False
                )
            # Check for response
            elif self.response_pattern.search(text):
                return LabelingFunctionResult(
                    intent_code="RESP:RFI_RESPONSE",
                    confidence=0.8,
                    abstain=False
                )
            # Check for count
            elif self.count_pattern.search(text):
                return LabelingFunctionResult(
                    intent_code="COUNT:RFI_COUNT",
                    confidence=0.8,
                    abstain=False
                )
            else:
                return LabelingFunctionResult(
                    intent_code="DOC:RFI_RETRIEVE",
                    confidence=0.8,
                    abstain=False
                )
        
        return LabelingFunctionResult()
    
    def lf_cb_keywords(self, text: str, entities: Dict[str, Any]) -> LabelingFunctionResult:
        """Label CB (Change Bulletin) queries."""
        if self.cb_pattern.search(text) or 'ids.cb' in entities:
            # Check for status
            if self.status_pattern.search(text):
                return LabelingFunctionResult(
                    intent_code="STAT:CB_STATUS",
                    confidence=0.8,
                    abstain=False
                )
            else:
                return LabelingFunctionResult(
                    intent_code="DOC:CB_RETRIEVE",
                    confidence=0.9,
                    abstain=False
                )
        
        return LabelingFunctionResult()
    
    def lf_spec_keywords(self, text: str, entities: Dict[str, Any]) -> LabelingFunctionResult:
        """Label specification-related queries."""
        has_spec_section = 'spec_section' in entities
        has_spec_keyword = self.spec_section_pattern.search(text)
        
        if has_spec_section or has_spec_keyword:
            # Check for requirement keywords
            if self.requirement_pattern.search(text):
                return LabelingFunctionResult(
                    intent_code="SPEC:REQUIREMENT_RULE",
                    confidence=0.8,
                    abstain=False
                )
            # Check for "what spec section" pattern
            elif self.what_pattern.search(text) and 'spec' in text.lower():
                return LabelingFunctionResult(
                    intent_code="SPEC:SECTION_MAP",
                    confidence=0.9,
                    abstain=False
                )
            else:
                return LabelingFunctionResult(
                    intent_code="SPEC:SECTION_MAP",
                    confidence=0.7,
                    abstain=False
                )
        
        return LabelingFunctionResult()
    
    def lf_status_keywords(self, text: str, entities: Dict[str, Any]) -> LabelingFunctionResult:
        """Label status-related queries."""
        if self.status_pattern.search(text):
            # Determine which type based on entities
            if 'ids.rfi' in entities:
                return LabelingFunctionResult(
                    intent_code="STAT:RFI_STATUS",
                    confidence=0.9,
                    abstain=False
                )
            elif 'ids.submittal' in entities:
                return LabelingFunctionResult(
                    intent_code="STAT:SUBMITTAL_STATUS",
                    confidence=0.9,
                    abstain=False
                )
            elif 'ids.cb' in entities:
                return LabelingFunctionResult(
                    intent_code="STAT:CB_STATUS",
                    confidence=0.9,
                    abstain=False
                )
        
        return LabelingFunctionResult()
    
    def lf_count_keywords(self, text: str, entities: Dict[str, Any]) -> LabelingFunctionResult:
        """Label count/quantity queries."""
        if self.count_pattern.search(text) or self.how_many_pattern.search(text):
            # Check context
            if 'rfi' in text.lower():
                return LabelingFunctionResult(
                    intent_code="COUNT:RFI_COUNT",
                    confidence=0.8,
                    abstain=False
                )
            elif 'submittal' in text.lower():
                return LabelingFunctionResult(
                    intent_code="COUNT:SUBMITTAL_COUNT",
                    confidence=0.8,
                    abstain=False
                )
        
        return LabelingFunctionResult()
    
    def lf_schedule_keywords(self, text: str, entities: Dict[str, Any]) -> LabelingFunctionResult:
        """Label schedule-related queries."""
        if self.schedule_pattern.search(text):
            # Check for specific schedule types
            if 'door' in text.lower():
                return LabelingFunctionResult(
                    intent_code="SCHED:DOOR_SCHEDULE",
                    confidence=0.9,
                    abstain=False
                )
            elif 'equipment' in text.lower():
                return LabelingFunctionResult(
                    intent_code="SCHED:EQUIPMENT_SCHEDULE",
                    confidence=0.9,
                    abstain=False
                )
            elif 'finish' in text.lower():
                return LabelingFunctionResult(
                    intent_code="SCHED:FINISH_SCHEDULE",
                    confidence=0.9,
                    abstain=False
                )
        
        return LabelingFunctionResult()
    
    def lf_drawing_keywords(self, text: str, entities: Dict[str, Any]) -> LabelingFunctionResult:
        """Label drawing-related queries."""
        if self.drawing_pattern.search(text) or self.shop_drawing_pattern.search(text):
            # Check for revision/latest
            if 'latest' in text.lower() or 'revision' in text.lower():
                return LabelingFunctionResult(
                    intent_code="DRAW:LATEST_REVISION",
                    confidence=0.8,
                    abstain=False
                )
            # Check for shop drawings specifically
            elif self.shop_drawing_pattern.search(text):
                return LabelingFunctionResult(
                    intent_code="DOC:SHOP_DRAWING_RETRIEVE",
                    confidence=0.9,
                    abstain=False
                )
            else:
                return LabelingFunctionResult(
                    intent_code="DRAW:FIND_DRAWING",
                    confidence=0.7,
                    abstain=False
                )
        
        return LabelingFunctionResult()
    
    def lf_location_keywords(self, text: str, entities: Dict[str, Any]) -> LabelingFunctionResult:
        """Label location-related queries."""
        if self.location_pattern.search(text) or self.where_pattern.search(text):
            # Check context for physical vs drawing location
            if 'drawing' in text.lower() or 'sheet' in text.lower():
                return LabelingFunctionResult(
                    intent_code="LOC:DRAWING_LOCATION",
                    confidence=0.8,
                    abstain=False
                )
            else:
                return LabelingFunctionResult(
                    intent_code="LOC:PHYSICAL_LOCATION",
                    confidence=0.7,
                    abstain=False
                )
        
        return LabelingFunctionResult()
    
    def lf_definition_keywords(self, text: str, entities: Dict[str, Any]) -> LabelingFunctionResult:
        """Label definition/explanation queries."""
        if self.definition_pattern.search(text):
            # Check if it's asking about an acronym (all caps word)
            acronym_pattern = re.compile(r'\b[A-Z]{2,}\b')
            if acronym_pattern.search(text):
                return LabelingFunctionResult(
                    intent_code="DEF:ACRONYM",
                    confidence=0.8,
                    abstain=False
                )
            else:
                return LabelingFunctionResult(
                    intent_code="DEF:TERM",
                    confidence=0.7,
                    abstain=False
                )
        
        return LabelingFunctionResult()
    
    def lf_translation_keywords(self, text: str, entities: Dict[str, Any]) -> LabelingFunctionResult:
        """Label translation queries."""
        if self.translation_pattern.search(text) or 'target_language' in entities:
            return LabelingFunctionResult(
                intent_code="TRAN:TRANSLATE",
                confidence=0.95,
                abstain=False
            )
        
        return LabelingFunctionResult()
    
    def lf_parameter_keywords(self, text: str, entities: Dict[str, Any]) -> LabelingFunctionResult:
        """Label parameter/dimension queries."""
        if self.parameter_pattern.search(text):
            return LabelingFunctionResult(
                intent_code="PARAM:DIMENSIONS",
                confidence=0.7,
                abstain=False
            )
        
        return LabelingFunctionResult()
    
    def lf_entity_based(self, text: str, entities: Dict[str, Any]) -> LabelingFunctionResult:
        """Label based on entity presence patterns."""
        # Strong signals from entity combinations
        if 'ids.rfi' in entities and 'ids.cb' in entities:
            return LabelingFunctionResult(
                intent_code="LINK:RFI_TO_CB",
                confidence=0.8,
                abstain=False
            )
        
        if 'product_code' in entities:
            return LabelingFunctionResult(
                intent_code="PROD:PRODUCT_LOOKUP",
                confidence=0.7,
                abstain=False
            )
        
        if 'door_id' in entities and 'schedule' in text.lower():
            return LabelingFunctionResult(
                intent_code="SCHED:DOOR_SCHEDULE",
                confidence=0.9,
                abstain=False
            )
        
        return LabelingFunctionResult()
    
    def lf_question_patterns(self, text: str, entities: Dict[str, Any]) -> LabelingFunctionResult:
        """Label based on question word patterns."""
        # "What is" patterns
        if self.what_pattern.search(text):
            if 'spec' in text.lower() and 'section' in text.lower():
                return LabelingFunctionResult(
                    intent_code="SPEC:SECTION_MAP",
                    confidence=0.8,
                    abstain=False
                )
            elif 'status' in text.lower():
                # Determine type from entities
                if 'ids.rfi' in entities:
                    return LabelingFunctionResult(
                        intent_code="STAT:RFI_STATUS",
                        confidence=0.8,
                        abstain=False
                    )
        
        # "Where" patterns
        if self.where_pattern.search(text):
            return LabelingFunctionResult(
                intent_code="LOC:PHYSICAL_LOCATION",
                confidence=0.6,
                abstain=False
            )
        
        # "How many" patterns
        if self.how_many_pattern.search(text):
            if 'rfi' in text.lower():
                return LabelingFunctionResult(
                    intent_code="COUNT:RFI_COUNT",
                    confidence=0.8,
                    abstain=False
                )
            elif 'submittal' in text.lower():
                return LabelingFunctionResult(
                    intent_code="COUNT:SUBMITTAL_COUNT",
                    confidence=0.8,
                    abstain=False
                )
        
        return LabelingFunctionResult()
    
    def lf_imperative_patterns(self, text: str, entities: Dict[str, Any]) -> LabelingFunctionResult:
        """Label based on imperative verb patterns."""
        # "Show/Display/List" patterns
        if self.show_pattern.search(text):
            if 'submittal' in text.lower():
                return LabelingFunctionResult(
                    intent_code="DOC:SUBMITTAL_RETRIEVE",
                    confidence=0.8,
                    abstain=False
                )
            elif 'drawing' in text.lower():
                return LabelingFunctionResult(
                    intent_code="DRAW:FIND_DRAWING",
                    confidence=0.7,
                    abstain=False
                )
            elif 'schedule' in text.lower():
                if 'door' in text.lower():
                    return LabelingFunctionResult(
                        intent_code="SCHED:DOOR_SCHEDULE",
                        confidence=0.8,
                        abstain=False
                    )
        
        # "Find/Search/Locate" patterns
        if self.find_pattern.search(text):
            if 'rfi' in text.lower():
                return LabelingFunctionResult(
                    intent_code="DOC:RFI_RETRIEVE",
                    confidence=0.7,
                    abstain=False
                )
            elif 'drawing' in text.lower():
                return LabelingFunctionResult(
                    intent_code="DRAW:FIND_DRAWING",
                    confidence=0.7,
                    abstain=False
                )
        
        return LabelingFunctionResult()
    
    def lf_linking_patterns(self, text: str, entities: Dict[str, Any]) -> LabelingFunctionResult:
        """Label linking/relationship queries."""
        if self.link_pattern.search(text):
            # Check what's being linked
            has_rfi = 'ids.rfi' in entities or 'rfi' in text.lower()
            has_cb = 'ids.cb' in entities or 'cb' in text.lower()
            has_spec = 'spec_section' in entities or 'spec' in text.lower()
            
            if has_rfi and has_cb:
                return LabelingFunctionResult(
                    intent_code="LINK:RFI_TO_CB",
                    confidence=0.8,
                    abstain=False
                )
            elif has_cb and has_spec:
                return LabelingFunctionResult(
                    intent_code="LINK:CB_TO_SPEC",
                    confidence=0.8,
                    abstain=False
                )
        
        return LabelingFunctionResult()
    
    def lf_response_patterns(self, text: str, entities: Dict[str, Any]) -> LabelingFunctionResult:
        """Label response/answer queries."""
        if self.response_pattern.search(text) and ('ids.rfi' in entities or 'rfi' in text.lower()):
            return LabelingFunctionResult(
                intent_code="RESP:RFI_RESPONSE",
                confidence=0.8,
                abstain=False
            )
        
        return LabelingFunctionResult()
    
    def lf_product_patterns(self, text: str, entities: Dict[str, Any]) -> LabelingFunctionResult:
        """Label product/material queries."""
        if self.product_pattern.search(text) or 'product_code' in entities:
            return LabelingFunctionResult(
                intent_code="PROD:PRODUCT_LOOKUP",
                confidence=0.7,
                abstain=False
            )
        
        return LabelingFunctionResult()
    
    def lf_admin_patterns(self, text: str, entities: Dict[str, Any]) -> LabelingFunctionResult:
        """Label administrative queries."""
        if 'permission' in text.lower() or 'access' in text.lower():
            return LabelingFunctionResult(
                intent_code="ADMIN:PERMISSIONS",
                confidence=0.7,
                abstain=False
            )
        elif 'help' in text.lower():
            return LabelingFunctionResult(
                intent_code="ADMIN:HELP",
                confidence=0.6,
                abstain=False
            )
        
        return LabelingFunctionResult()
    
    def lf_query_patterns(self, text: str, entities: Dict[str, Any]) -> LabelingFunctionResult:
        """Label query builder patterns."""
        if 'query' in text.lower() or 'filter' in text.lower() or 'search' in text.lower():
            # Only if it seems to be about building queries, not general search
            if 'build' in text.lower() or 'create' in text.lower() or 'filter' in text.lower():
                return LabelingFunctionResult(
                    intent_code="QRY:QUERY_BUILDER",
                    confidence=0.6,
                    abstain=False
                )
        
        return LabelingFunctionResult()
    
    def lf_unit_patterns(self, text: str, entities: Dict[str, Any]) -> LabelingFunctionResult:
        """Label unit conversion queries."""
        if 'convert' in text.lower() and ('unit' in text.lower() or 
                                         re.search(r'\b(?:ft|feet|m|meter|inch|cm)\b', text, re.IGNORECASE)):
            return LabelingFunctionResult(
                intent_code="UNIT:CONVERSION",
                confidence=0.8,
                abstain=False
            )
        
        return LabelingFunctionResult()
    
    def lf_personnel_patterns(self, text: str, entities: Dict[str, Any]) -> LabelingFunctionResult:
        """Label personnel/contact queries."""
        personnel_keywords = ['who is', "who's", 'contact', 'person', 'manager', 'superintendent', 
                            'contractor', 'architect', 'engineer', 'owner', 'pm', 'project manager']
        
        text_lower = text.lower()
        if any(keyword in text_lower for keyword in personnel_keywords):
            return LabelingFunctionResult(
                intent_code="ADMIN:PERSONNEL",
                confidence=0.8,
                abstain=False
            )
        
        return LabelingFunctionResult()
    
    def lf_enhanced_drawing_patterns(self, text: str, entities: Dict[str, Any]) -> LabelingFunctionResult:
        """Label drawing-related queries more accurately."""
        drawing_keywords = ['drawing', 'plan', 'elevation', 'section', 'detail', 'diagram', 'sheet']
        action_keywords = ['show', 'find', 'get', 'pull', 'display', 'view', 'see']
        
        text_lower = text.lower()
        has_drawing_keyword = any(keyword in text_lower for keyword in drawing_keywords)
        has_action = any(action in text_lower for action in action_keywords)
        
        if has_drawing_keyword:
            # Check if it's about finding where something is on drawings
            if 'where' in text_lower or 'which' in text_lower or 'what drawing' in text_lower:
                return LabelingFunctionResult(
                    intent_code="DRAW:DRAWING_MAP",
                    confidence=0.85,
                    abstain=False
                )
            # Otherwise it's likely retrieval
            elif has_action:
                return LabelingFunctionResult(
                    intent_code="DRAW:DRAWING_RETRIEVE",
                    confidence=0.85,
                    abstain=False
                )
        
        return LabelingFunctionResult()
    
    def lf_color_material_patterns(self, text: str, entities: Dict[str, Any]) -> LabelingFunctionResult:
        """Label color and material specification queries."""
        color_keywords = ['color', 'colour', 'paint', 'finish', 'coating']
        material_keywords = ['material', 'type', 'kind', 'thickness', 'size', 'dimension']
        
        text_lower = text.lower()
        has_color = any(keyword in text_lower for keyword in color_keywords)
        has_material = any(keyword in text_lower for keyword in material_keywords)
        
        if has_color or has_material:
            # Check if it's about a specific product
            if 'product_code' in entities or self.product_pattern.search(text):
                return LabelingFunctionResult(
                    intent_code="PROD:MATERIAL_SPEC",
                    confidence=0.75,
                    abstain=False
                )
        
        return LabelingFunctionResult()
    
    def lf_project_info_patterns(self, text: str, entities: Dict[str, Any]) -> LabelingFunctionResult:
        """Label project information queries."""
        info_keywords = ['project address', 'project name', 'project number', 'job number',
                        'project location', 'site address', 'job site']
        
        text_lower = text.lower()
        if any(keyword in text_lower for keyword in info_keywords):
            return LabelingFunctionResult(
                intent_code="ADMIN:PROJECT_INFO",
                confidence=0.85,
                abstain=False
            )
        
        return LabelingFunctionResult()
    
    def lf_vendor_patterns(self, text: str, entities: Dict[str, Any]) -> LabelingFunctionResult:
        """Label vendor/subcontractor queries."""
        vendor_keywords = ['vendor', 'supplier', 'manufacturer', 'subcontractor', 'sub',
                          'company', 'firm', 'provider']
        
        text_lower = text.lower()
        if any(keyword in text_lower for keyword in vendor_keywords):
            # Check if asking for count
            if self.count_pattern.search(text):
                return LabelingFunctionResult(
                    intent_code="COUNT:VENDOR_COUNT",
                    confidence=0.8,
                    abstain=False
                )
            else:
                return LabelingFunctionResult(
                    intent_code="ADMIN:VENDOR_LIST",
                    confidence=0.75,
                    abstain=False
                )
        
        return LabelingFunctionResult()
    
    def lf_requirement_patterns(self, text: str, entities: Dict[str, Any]) -> LabelingFunctionResult:
        """Label requirement/compliance queries."""
        requirement_keywords = ['require', 'requirement', 'need', 'must', 'shall', 'comply',
                               'compliance', 'standard', 'regulation', 'code']
        
        text_lower = text.lower()
        if any(keyword in text_lower for keyword in requirement_keywords):
            # Check if it's about specs
            if 'spec' in text_lower or 'specification' in text_lower:
                return LabelingFunctionResult(
                    intent_code="SPEC:REQUIREMENTS",
                    confidence=0.8,
                    abstain=False
                )
        
        return LabelingFunctionResult()
    
    def lf_activity_schedule_patterns(self, text: str, entities: Dict[str, Any]) -> LabelingFunctionResult:
        """Label activity and schedule-related queries."""
        schedule_keywords = ['activity', 'schedule', 'sequence', 'before', 'after', 'when',
                            'timeline', 'duration', 'complete', 'start', 'finish']
        
        text_lower = text.lower()
        if any(keyword in text_lower for keyword in schedule_keywords):
            return LabelingFunctionResult(
                intent_code="SCHED:ACTIVITY_SEQUENCE",
                confidence=0.75,
                abstain=False
            )
        
        return LabelingFunctionResult()
    
    def lf_equipment_patterns(self, text: str, entities: Dict[str, Any]) -> LabelingFunctionResult:
        """Label equipment-related queries."""
        equipment_keywords = ['equipment', 'unit', 'system', 'hvac', 'mechanical', 'electrical',
                             'plumbing', 'compressor', 'pump', 'fan', 'boiler', 'chiller']
        
        text_lower = text.lower()
        if any(keyword in text_lower for keyword in equipment_keywords):
            # Check if it's about specifications
            if 'spec' in text_lower or 'output' in text_lower or 'capacity' in text_lower:
                return LabelingFunctionResult(
                    intent_code="PROD:EQUIPMENT_SPEC",
                    confidence=0.75,
                    abstain=False
                )
        
        return LabelingFunctionResult()
    
    def lf_door_hardware_patterns(self, text: str, entities: Dict[str, Any]) -> LabelingFunctionResult:
        """Label door and hardware queries more accurately."""
        if 'door_id' in entities or ('door' in text.lower() and 'hardware' in text.lower()):
            # Check for schedule/list request
            if 'schedule' in text.lower() or 'list' in text.lower() or 'show' in text.lower():
                return LabelingFunctionResult(
                    intent_code="DOC:DOOR_SCHEDULE",
                    confidence=0.85,
                    abstain=False
                )
            # Check for hardware group
            elif 'group' in text.lower() or 'hardware' in text.lower():
                return LabelingFunctionResult(
                    intent_code="PROD:HARDWARE_GROUP",
                    confidence=0.8,
                    abstain=False
                )
        
        return LabelingFunctionResult()
    
    def lf_enhanced_definition_patterns(self, text: str, entities: Dict[str, Any]) -> LabelingFunctionResult:
        """Label definition/explanation queries."""
        definition_keywords = ['what is', 'what are', "what's", 'define', 'definition', 
                             'explain', 'mean', 'meaning']
        
        text_lower = text.lower()
        if any(keyword in text_lower for keyword in definition_keywords):
            # Exclude cases that are asking about specific items
            if not any(ent in entities for ent in ['ids.rfi', 'ids.submittal', 'product_code']):
                return LabelingFunctionResult(
                    intent_code="ADMIN:DEFINITION",
                    confidence=0.7,
                    abstain=False
                )
        
        return LabelingFunctionResult()
    
    def aggregate_votes(self, votes: List[LabelingFunctionVote]) -> Dict[str, float]:
        """
        Aggregate votes from multiple labeling functions.
        
        Args:
            votes: List of votes
            
        Returns:
            Dictionary mapping intent codes to aggregated confidence scores
        """
        if not votes:
            return {}
        
        # Group votes by intent
        intent_votes = defaultdict(list)
        for vote in votes:
            intent_votes[vote.intent_code].append(vote.confidence)
        
        # Aggregate confidences (weighted average)
        aggregated = {}
        for intent_code, confidences in intent_votes.items():
            # Use mean confidence, with boost for multiple agreeing functions
            mean_conf = sum(confidences) / len(confidences)
            agreement_boost = min(0.2, 0.05 * (len(confidences) - 1))
            aggregated[intent_code] = min(0.95, mean_conf + agreement_boost)
        
        return aggregated
    
    def get_top_predictions(self, votes: List[LabelingFunctionVote], 
                          top_k: int = 3) -> List[Tuple[str, float]]:
        """
        Get top K predicted intents from votes.
        
        Args:
            votes: List of votes
            top_k: Number of top predictions to return
            
        Returns:
            List of (intent_code, confidence) tuples
        """
        aggregated = self.aggregate_votes(votes)
        
        # Sort by confidence
        sorted_intents = sorted(
            aggregated.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return sorted_intents[:top_k]
