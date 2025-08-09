"""
Tests for the taxonomy loader module.
"""

import pytest
from pathlib import Path
import json
import tempfile

from intent_classifier.core.taxonomy_loader import TaxonomyLoader


class TestTaxonomyLoader:
    """Test the TaxonomyLoader class."""
    
    @pytest.fixture
    def sample_taxonomy(self):
        """Create a sample taxonomy for testing."""
        return {
            "taxonomy_id": "test-taxonomy",
            "taxonomy_version": "1.0.0",
            "description": "Test taxonomy",
            "entities_schema": [
                {
                    "key": "ids.rfi",
                    "type": "integer",
                    "example": 123
                },
                {
                    "key": "spec_section",
                    "type": "string",
                    "pattern": r"\b\d{2}\s?\d{2}\s?\d{2}\b",
                    "example": "10 22 33"
                },
                {
                    "key": "language",
                    "type": "string",
                    "enum": ["en", "es", "ar"],
                    "example": "en"
                }
            ],
            "routing_defaults": {
                "DOC": {
                    "policies": {
                        "freshness": "latest_only"
                    }
                },
                "SPEC": {
                    "policies": {
                        "freshness": "any"
                    }
                }
            },
            "classes": [
                {
                    "code": "DOC",
                    "name": "Document Retrieval",
                    "description": "Fetch documents",
                    "subclasses": [
                        {
                            "code": "DOC:SUBMITTAL_RETRIEVE",
                            "description": "Retrieve submittals",
                            "required_entities": ["submittal_topic"],
                            "optional_entities": ["ids.submittal"],
                            "routing_hints": {
                                "tools": ["SubmittalSearch"],
                                "few_shot_id": "doc_v1"
                            },
                            "example_triggers": ["show me the tile submittal"]
                        },
                        {
                            "code": "DOC:RFI_RETRIEVE",
                            "description": "Retrieve RFIs",
                            "required_entities": ["ids.rfi"],
                            "optional_entities": [],
                            "routing_hints": {
                                "tools": ["RFISearch"],
                                "few_shot_id": "rfi_v1"
                            }
                        }
                    ]
                },
                {
                    "code": "SPEC",
                    "name": "Specification",
                    "description": "Specification queries",
                    "subclasses": [
                        {
                            "code": "SPEC:SECTION_MAP",
                            "description": "Find spec sections",
                            "required_entities": [],
                            "optional_entities": ["spec_section"],
                            "example_triggers": ["what spec section covers"]
                        }
                    ]
                }
            ],
            "composition_rules": {
                "allowed_chains": [
                    {"from": "DOC:RFI_RETRIEVE", "to": "DOC:SUBMITTAL_RETRIEVE"},
                    {"from": "DOC", "to": "SPEC"},
                    ["SPEC", "*"]
                ]
            }
        }
    
    @pytest.fixture
    def taxonomy_loader(self, sample_taxonomy):
        """Create a TaxonomyLoader with sample data."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(sample_taxonomy, f)
            temp_path = f.name
        
        loader = TaxonomyLoader(temp_path)
        
        # Clean up
        Path(temp_path).unlink()
        
        return loader
    
    def test_initialization(self, taxonomy_loader):
        """Test that the loader initializes correctly."""
        assert taxonomy_loader is not None
        assert len(taxonomy_loader.get_coarse_classes()) == 2
        assert len(taxonomy_loader.get_labels()) == 3
    
    def test_get_labels(self, taxonomy_loader):
        """Test getting all intent labels."""
        labels = taxonomy_loader.get_labels()
        assert "DOC:SUBMITTAL_RETRIEVE" in labels
        assert "DOC:RFI_RETRIEVE" in labels
        assert "SPEC:SECTION_MAP" in labels
        assert len(labels) == 3
    
    def test_get_coarse_classes(self, taxonomy_loader):
        """Test getting coarse classes."""
        classes = taxonomy_loader.get_coarse_classes()
        assert classes == ["DOC", "SPEC"]
    
    def test_is_valid(self, taxonomy_loader):
        """Test intent code validation."""
        assert taxonomy_loader.is_valid("DOC:SUBMITTAL_RETRIEVE")
        assert taxonomy_loader.is_valid("SPEC:SECTION_MAP")
        assert not taxonomy_loader.is_valid("INVALID:CODE")
        assert not taxonomy_loader.is_valid("DOC:INVALID")
    
    def test_get_coarse_class(self, taxonomy_loader):
        """Test getting coarse class from intent code."""
        assert taxonomy_loader.get_coarse_class("DOC:SUBMITTAL_RETRIEVE") == "DOC"
        assert taxonomy_loader.get_coarse_class("SPEC:SECTION_MAP") == "SPEC"
        assert taxonomy_loader.get_coarse_class("INVALID:CODE") == "INVALID"
    
    def test_get_subclasses(self, taxonomy_loader):
        """Test getting subclasses for a coarse class."""
        doc_subclasses = taxonomy_loader.get_subclasses("DOC")
        assert len(doc_subclasses) == 2
        assert "DOC:SUBMITTAL_RETRIEVE" in doc_subclasses
        assert "DOC:RFI_RETRIEVE" in doc_subclasses
        
        spec_subclasses = taxonomy_loader.get_subclasses("SPEC")
        assert len(spec_subclasses) == 1
        assert "SPEC:SECTION_MAP" in spec_subclasses
    
    def test_get_intent_info(self, taxonomy_loader):
        """Test getting detailed intent information."""
        info = taxonomy_loader.get_intent_info("DOC:SUBMITTAL_RETRIEVE")
        assert info is not None
        assert info['description'] == "Retrieve submittals"
        assert info['required_entities'] == ["submittal_topic"]
        assert info['optional_entities'] == ["ids.submittal"]
    
    def test_get_required_entities(self, taxonomy_loader):
        """Test getting required entities."""
        entities = taxonomy_loader.get_required_entities("DOC:SUBMITTAL_RETRIEVE")
        assert entities == ["submittal_topic"]
        
        entities = taxonomy_loader.get_required_entities("SPEC:SECTION_MAP")
        assert entities == []
    
    def test_get_routing_hints(self, taxonomy_loader):
        """Test getting routing hints."""
        hints = taxonomy_loader.get_routing_hints("DOC:SUBMITTAL_RETRIEVE")
        assert hints['tools'] == ["SubmittalSearch"]
        assert hints['few_shot_id'] == "doc_v1"
        
        # Test fallback for missing routing hints
        hints = taxonomy_loader.get_routing_hints("SPEC:SECTION_MAP")
        assert 'tools' in hints  # Should have default tools
    
    def test_get_policies(self, taxonomy_loader):
        """Test getting policies."""
        policies = taxonomy_loader.get_policies("DOC:SUBMITTAL_RETRIEVE")
        assert policies['freshness'] == "latest_only"
        
        policies = taxonomy_loader.get_policies("SPEC:SECTION_MAP")
        assert policies['freshness'] == "any"
    
    def test_is_allowed_chain(self, taxonomy_loader):
        """Test chain validation."""
        # Specific chain
        assert taxonomy_loader.is_allowed_chain("DOC:RFI_RETRIEVE", "DOC:SUBMITTAL_RETRIEVE")
        
        # Coarse class chain
        assert taxonomy_loader.is_allowed_chain("DOC:SUBMITTAL_RETRIEVE", "SPEC:SECTION_MAP")
        
        # Wildcard chain
        assert taxonomy_loader.is_allowed_chain("SPEC:SECTION_MAP", "DOC:RFI_RETRIEVE")
        
        # Invalid chain
        assert not taxonomy_loader.is_allowed_chain("SPEC:SECTION_MAP", "DOC:SUBMITTAL_RETRIEVE")
    
    def test_entity_validation(self, taxonomy_loader):
        """Test entity validation."""
        # Integer validation
        assert taxonomy_loader.validate_entity("ids.rfi", 123)
        assert taxonomy_loader.validate_entity("ids.rfi", "123")  # String that can be int
        assert not taxonomy_loader.validate_entity("ids.rfi", "abc")
        
        # Pattern validation
        assert taxonomy_loader.validate_entity("spec_section", "10 22 33")
        assert not taxonomy_loader.validate_entity("spec_section", "invalid")
        
        # Enum validation
        assert taxonomy_loader.validate_entity("language", "en")
        assert not taxonomy_loader.validate_entity("language", "fr")
        
        # Unknown entity (should pass)
        assert taxonomy_loader.validate_entity("unknown_entity", "anything")
    
    def test_get_example_triggers(self, taxonomy_loader):
        """Test getting example triggers."""
        triggers = taxonomy_loader.get_example_triggers("DOC:SUBMITTAL_RETRIEVE")
        assert triggers == ["show me the tile submittal"]
        
        triggers = taxonomy_loader.get_example_triggers("DOC:RFI_RETRIEVE")
        assert triggers == []
    
    def test_search_intents(self, taxonomy_loader):
        """Test intent search functionality."""
        # Search by code
        results = taxonomy_loader.search_intents("submittal")
        assert "DOC:SUBMITTAL_RETRIEVE" in results
        
        # Search by description
        results = taxonomy_loader.search_intents("retrieve")
        assert len(results) == 2
        assert "DOC:SUBMITTAL_RETRIEVE" in results
        assert "DOC:RFI_RETRIEVE" in results
        
        # Search by trigger
        results = taxonomy_loader.search_intents("tile")
        assert "DOC:SUBMITTAL_RETRIEVE" in results
    
    def test_get_taxonomy_summary(self, taxonomy_loader):
        """Test getting taxonomy summary."""
        summary = taxonomy_loader.get_taxonomy_summary()
        assert summary['taxonomy_id'] == "test-taxonomy"
        assert summary['version'] == "1.0.0"
        assert summary['total_coarse_classes'] == 2
        assert summary['total_intent_codes'] == 3
        assert summary['total_entities'] == 3
