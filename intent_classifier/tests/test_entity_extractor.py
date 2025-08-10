"""
Tests for the entity extractor module.
"""

import pytest
from intent_classifier.core.entity_extractor import EntityExtractor, EntityDefinition


class TestEntityExtractor:
    """Test the EntityExtractor class."""
    
    @pytest.fixture
    def entity_definitions(self):
        """Sample entity definitions matching taxonomy format."""
        return [
            {"key": "ids.rfi", "type": "integer", "example": 1838},
            {"key": "ids.cb", "type": "integer", "example": 309},
            {"key": "ids.submittal", "type": "integer", "example": 232},
            {"key": "spec_section", "type": "string", "pattern": r"\b\d{2}\s?\d{2}\s?\d{2}\b", "example": "10 22 33"},
            {"key": "product_code", "type": "string", "pattern": "^[A-Z]{1,4}-\\d{1,3}$", "example": "ACT-9"},
            {"key": "door_id", "type": "string", "pattern": "^\\d{4,5}(-\\d)?$", "example": "20110-2"},
            {"key": "floor", "type": "string", "example": "Level 5"},
            {"key": "area", "type": "string", "example": "Atrium"},
            {"key": "discipline", "type": "string", "example": "Mechanical"},
            {"key": "topic", "type": "string", "example": "wireless access points"},
            {"key": "submittal_topic", "type": "string", "example": "tile"},
            {"key": "date_range", "type": "string", "example": "2024-09"},
            {"key": "target_language", "type": "string", "enum": ["en", "es", "ar"], "example": "es"},
            {"key": "answer_text", "type": "string", "description": "Used only for translation handoff"}
        ]
    
    @pytest.fixture
    def extractor(self, entity_definitions):
        """Create an entity extractor instance."""
        return EntityExtractor(entity_definitions)
    
    def test_extract_rfi_id(self, extractor):
        """Test RFI ID extraction with various formats."""
        test_cases = [
            ("What is the status of RFI 1838?", {"ids.rfi": 1838}),
            ("Check RFI#1234", {"ids.rfi": 1234}),
            ("rfi 42 needs review", {"ids.rfi": 42}),
            ("RFI:5678 status", {"ids.rfi": 5678}),
            ("Show me RFI-999", {"ids.rfi": 999}),
        ]
        
        for text, expected in test_cases:
            result = extractor.extract(text)
            for key, value in expected.items():
                assert key in result.entities
                assert result.entities[key] == value
    
    def test_extract_cb_id(self, extractor):
        """Test CB (Change Bulletin) ID extraction."""
        test_cases = [
            ("CB 309 was issued", {"ids.cb": 309}),
            ("change bulletin #123", {"ids.cb": 123}),
            ("CB#456 approved", {"ids.cb": 456}),
            ("bulletin 789", {"ids.cb": 789}),
        ]
        
        for text, expected in test_cases:
            result = extractor.extract(text)
            for key, value in expected.items():
                assert key in result.entities
                assert result.entities[key] == value
    
    def test_extract_submittal_id(self, extractor):
        """Test submittal ID extraction."""
        test_cases = [
            ("Review submittal 232", {"ids.submittal": 232}),
            ("submittal#100 status", {"ids.submittal": 100}),
            ("sub #555", {"ids.submittal": 555}),
            ("Submission 888", {"ids.submittal": 888}),
        ]
        
        for text, expected in test_cases:
            result = extractor.extract(text)
            for key, value in expected.items():
                assert key in result.entities
                assert result.entities[key] == value
    
    def test_extract_spec_section(self, extractor):
        """Test spec section extraction and formatting."""
        test_cases = [
            ("Check spec 102233", {"spec_section": "10 22 33"}),
            ("Section 09 51 00", {"spec_section": "09 51 00"}),
            ("10-22-33 requirements", {"spec_section": "10 22 33"}),
            ("Spec 09.91.23", {"spec_section": "09 91 23"}),
        ]
        
        for text, expected in test_cases:
            result = extractor.extract(text)
            for key, value in expected.items():
                assert key in result.entities
                assert result.entities[key] == value
    
    def test_extract_product_code(self, extractor):
        """Test product code extraction."""
        test_cases = [
            ("Product ACT-9", {"product_code": "ACT-9"}),
            ("Use MDF 123", {"product_code": "MDF-123"}),
            ("SPEC-45 details", {"product_code": "SPEC-45"}),
            ("Code XYZ 7", {"product_code": "XYZ-7"}),
        ]
        
        for text, expected in test_cases:
            result = extractor.extract(text)
            for key, value in expected.items():
                assert key in result.entities
                assert result.entities[key] == value
    
    def test_extract_door_id(self, extractor):
        """Test door ID extraction."""
        test_cases = [
            ("Check door 20110", {"door_id": "20110"}),
            ("door 20110-2", {"door_id": "20110-2"}),
            ("Door#12345", {"door_id": "12345"}),
            ("door:9999-1", {"door_id": "9999-1"}),
        ]
        
        for text, expected in test_cases:
            result = extractor.extract(text)
            for key, value in expected.items():
                assert key in result.entities
                assert result.entities[key] == value
    
    def test_extract_floor(self, extractor):
        """Test floor/level extraction and normalization."""
        test_cases = [
            ("on floor 5", {"floor": "Level 5"}),
            ("level 3 drawings", {"floor": "Level 3"}),
            ("basement mechanical", {"floor": "Basement"}),
            ("roof equipment", {"floor": "Roof"}),
            ("ground floor lobby", {"floor": "Ground"}),
            ("penthouse suite", {"floor": "Penthouse"}),
            ("Floor 2A", {"floor": "Level 2A"}),
            ("level B1", {"floor": "Level B1"}),
        ]
        
        for text, expected in test_cases:
            result = extractor.extract(text)
            for key, value in expected.items():
                assert key in result.entities
                assert result.entities[key] == value
    
    def test_extract_area(self, extractor):
        """Test area/location extraction."""
        test_cases = [
            ("in the lobby", {"area": "lobby"}),
            ("at the atrium", {"area": "atrium"}),
            ("near conference room", {"area": "conference room"}),
            ("mechanical room access", {"area": "mechanical room"}),
            ("parking garage level", {"area": "parking"}),
        ]
        
        for text, expected in test_cases:
            result = extractor.extract(text)
            for key, value in expected.items():
                assert key in result.entities
                assert result.entities[key] == value
    
    def test_extract_discipline(self, extractor):
        """Test discipline extraction and normalization."""
        test_cases = [
            ("mechanical drawings", {"discipline": "Mechanical"}),
            ("HVAC system", {"discipline": "HVAC"}),
            ("MEP coordination", {"discipline": "MEP"}),
            ("electrical conduit", {"discipline": "Electrical"}),
            ("fire protection plan", {"discipline": "Fire Protection"}),
        ]
        
        for text, expected in test_cases:
            result = extractor.extract(text)
            for key, value in expected.items():
                assert key in result.entities
                assert result.entities[key] == value
    
    def test_extract_topic(self, extractor):
        """Test general topic extraction."""
        test_cases = [
            ("submittal for tile installation", {"topic": "tile installation"}),
            ("drawing about wireless access points", {"topic": "wireless access points"}),
            ("spec regarding acoustic panels", {"topic": "acoustic panels"}),
            ("RFI concerning waterproofing", {"topic": "waterproofing"}),
        ]
        
        for text, expected in test_cases:
            result = extractor.extract(text)
            for key, value in expected.items():
                assert key in result.entities
                assert result.entities[key] == value
    
    def test_extract_submittal_topic(self, extractor):
        """Test submittal-specific topic extraction."""
        test_cases = [
            ("tile submittal", {"submittal_topic": "tile"}),
            ("submittal for flooring", {"submittal_topic": "flooring"}),
            ("mechanical equipment submittal", {"submittal_topic": "mechanical equipment"}),
            ("Show me the paint submittal", {"submittal_topic": "paint"}),
        ]
        
        for text, expected in test_cases:
            result = extractor.extract(text)
            for key, value in expected.items():
                assert key in result.entities
                assert result.entities[key] == value
    
    def test_extract_target_language(self, extractor):
        """Test target language extraction for translation."""
        test_cases = [
            ("translate to spanish", {"target_language": "es"}),
            ("convert into Arabic", {"target_language": "ar"}),
            ("translation in English", {"target_language": "en"}),
            ("to ES", {"target_language": "es"}),
        ]
        
        for text, expected in test_cases:
            result = extractor.extract(text)
            for key, value in expected.items():
                assert key in result.entities
                assert result.entities[key] == value
    
    def test_extract_answer_text(self, extractor):
        """Test answer text extraction for translation."""
        test_cases = [
            ('translate "approved with conditions" to spanish', {"answer_text": "approved with conditions"}),
            ("translate 'rejected' into arabic", {"answer_text": "rejected"}),
            ('translate the following: pending review', {"answer_text": "pending review"}),
        ]
        
        for text, expected in test_cases:
            result = extractor.extract(text)
            for key, value in expected.items():
                assert key in result.entities
                assert result.entities[key] == value
    
    def test_multiple_entities(self, extractor):
        """Test extraction of multiple entities from one query."""
        text = "Show me the tile submittal #232 for level 5 in the atrium"
        result = extractor.extract(text)
        
        expected = {
            "ids.submittal": 232,
            "submittal_topic": "tile",
            "floor": "Level 5",
            "area": "atrium"
        }
        
        for key, value in expected.items():
            assert key in result.entities
            assert result.entities[key] == value
    
    def test_entity_spans(self, extractor):
        """Test that entity spans are correctly tracked."""
        text = "RFI 1234 on floor 5"
        result = extractor.extract(text)
        
        # Check that spans are recorded
        assert len(result.entity_spans) >= 2
        
        # Verify span for RFI
        rfi_spans = [s for s in result.entity_spans if s['key'] == 'ids.rfi']
        assert len(rfi_spans) == 1
        assert text[rfi_spans[0]['start']:rfi_spans[0]['end']] == rfi_spans[0]['text']
    
    def test_preprocessed_text(self, extractor):
        """Test extraction with preprocessed text."""
        original = "Show RFI #1838 status"
        preprocessed = "Show RFI:1838 status"  # Normalized format
        
        result = extractor.extract(original, preprocessed)
        assert "ids.rfi" in result.entities
        assert result.entities["ids.rfi"] == 1838
    
    def test_validation_methods(self, extractor):
        """Test entity validation methods."""
        # Mock taxonomy loader
        class MockTaxonomyLoader:
            def get_intent_info(self, intent_code):
                if intent_code == "DOC:SUBMITTAL_RETRIEVE":
                    return {
                        "required_entities": ["submittal_topic"],
                        "optional_entities": ["ids.submittal", "discipline"]
                    }
                return None
        
        taxonomy = MockTaxonomyLoader()
        
        # Test with valid entities
        entities = {"submittal_topic": "tile", "discipline": "Architectural"}
        valid, missing = extractor.validate_entities_for_intent(
            entities, "DOC:SUBMITTAL_RETRIEVE", taxonomy
        )
        assert valid
        assert len(missing) == 0
        
        # Test with missing required entities
        entities = {"discipline": "Architectural"}
        valid, missing = extractor.validate_entities_for_intent(
            entities, "DOC:SUBMITTAL_RETRIEVE", taxonomy
        )
        assert not valid
        assert "submittal_topic" in missing
    
    def test_edge_cases(self, extractor):
        """Test edge cases and error handling."""
        # Empty text
        result = extractor.extract("")
        assert len(result.entities) == 0
        assert len(result.entity_spans) == 0
        
        # No entities
        result = extractor.extract("Hello world")
        assert len(result.entities) == 0
        
        # Multiple of same entity (should keep one)
        result = extractor.extract("RFI 123 and RFI 456")
        assert "ids.rfi" in result.entities
        # Should keep first one by default
        assert result.entities["ids.rfi"] == 123
    
    def test_complex_queries(self, extractor):
        """Test complex real-world queries."""
        queries = [
            (
                "What is the spec section for acoustic ceiling tiles in conference rooms on level 3?",
                {"topic": "acoustic ceiling tiles", "area": "conference rooms", "floor": "Level 3"}
            ),
            (
                "Show me all mechanical submittals from last month",
                {"discipline": "Mechanical", "date_range": "last month"}
            ),
            (
                "Find RFI 1838 response and translate it to Spanish",
                {"ids.rfi": 1838, "target_language": "es"}
            ),
        ]
        
        for text, expected_subset in queries:
            result = extractor.extract(text)
            for key, value in expected_subset.items():
                assert key in result.entities
                if isinstance(value, str):
                    assert value.lower() in result.entities[key].lower()
                else:
                    assert result.entities[key] == value
