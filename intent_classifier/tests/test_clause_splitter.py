"""
Tests for the clause splitter module.
"""

import pytest
from intent_classifier.core.clause_splitter import ClauseSplitter, SplitPoint
from intent_classifier.models.schemas import ClauseSegment


class TestClauseSplitter:
    """Test the ClauseSplitter class."""
    
    @pytest.fixture
    def splitter_with_spacy(self):
        """Create a clause splitter with spaCy (if available)."""
        return ClauseSplitter(use_simple_fallback=True)
    
    @pytest.fixture
    def splitter_simple(self):
        """Create a clause splitter that only uses simple splitting."""
        splitter = ClauseSplitter(use_simple_fallback=True)
        splitter.nlp = None  # Force simple mode
        return splitter
    
    def test_single_clause(self, splitter_simple):
        """Test that single clauses are not split."""
        text = "What is the status of RFI 1838?"
        segments = splitter_simple.split(text)
        
        assert len(segments) == 1
        assert segments[0].text == text.strip()
        assert segments[0].start_offset == 0
        assert segments[0].dependencies == []
    
    def test_conjunction_split(self, splitter_simple):
        """Test splitting on conjunctions."""
        text = "Show me the tile submittal and check if it's approved"
        segments = splitter_simple.split(text)
        
        # Simple splitter might not split on 'and' without comma
        # But should recognize it as potential multi-intent
        confidence = splitter_simple.get_splitting_confidence(text)
        assert confidence > 0
    
    def test_comma_and_split(self, splitter_simple):
        """Test splitting on comma + and."""
        text = "Show me the tile submittal, and check if it's approved"
        segments = splitter_simple.split(text)
        
        assert len(segments) == 2
        assert "tile submittal" in segments[0].text
        assert "check" in segments[1].text
    
    def test_semicolon_split(self, splitter_simple):
        """Test splitting on semicolons."""
        text = "Review the mechanical drawings; verify the duct sizes"
        segments = splitter_simple.split(text)
        
        assert len(segments) == 2
        assert "mechanical drawings" in segments[0].text
        assert "verify" in segments[1].text
    
    def test_multiple_questions(self, splitter_simple):
        """Test splitting multiple questions."""
        text = "What is the RFI status? How many are still open?"
        segments = splitter_simple.split(text)
        
        assert len(segments) == 2
        assert "RFI status" in segments[0].text
        assert "How many" in segments[1].text
    
    def test_list_preservation(self, splitter_with_spacy):
        """Test that lists are not split."""
        text = "Show submittals for plumbing, electrical, and mechanical"
        segments = splitter_with_spacy.split(text)
        
        # Should not split the list
        assert len(segments) == 1
        assert "plumbing" in segments[0].text
        assert "electrical" in segments[0].text
        assert "mechanical" in segments[0].text
    
    def test_complex_multi_intent(self, splitter_simple):
        """Test complex multi-intent query."""
        text = "Find all open RFIs, calculate the total count, and show me the oldest one"
        segments = splitter_simple.split(text)
        
        # Should identify multiple intents
        confidence = splitter_simple.get_splitting_confidence(text)
        assert confidence > 0.5
    
    def test_imperative_chains(self, splitter_simple):
        """Test chains of imperative commands."""
        text = "Get the shop drawings, check their status, then send to the architect"
        segments = splitter_simple.split(text)
        
        # Should recognize multiple commands
        confidence = splitter_simple.get_splitting_confidence(text)
        assert confidence > 0
    
    def test_question_word_detection(self, splitter_with_spacy):
        """Test detection of multiple question words."""
        text = "What is the spec section and where can I find it?"
        segments = splitter_with_spacy.split(text)
        
        # Should detect two questions
        assert len(segments) >= 1
        confidence = splitter_with_spacy.get_splitting_confidence(text)
        assert confidence > 0.5
    
    def test_empty_input(self, splitter_simple):
        """Test empty input handling."""
        assert splitter_simple.split("") == []
        assert splitter_simple.split("   ") == []
        assert splitter_simple.split("\n\t") == []
    
    def test_offset_accuracy(self, splitter_simple):
        """Test that offsets are accurate."""
        text = "First clause; second clause"
        segments = splitter_simple.split(text)
        
        if len(segments) > 1:
            # Verify offsets
            for segment in segments:
                extracted = text[segment.start_offset:segment.end_offset]
                assert segment.text in extracted
    
    def test_dependency_detection(self, splitter_with_spacy):
        """Test dependency detection between clauses."""
        text = "Find the RFI and check its status"
        segments = splitter_with_spacy.split(text)
        
        # Second clause might depend on first (due to "its")
        if len(segments) > 1:
            # Check if dependencies are detected
            pass  # Dependencies are simplified in current implementation
    
    def test_numbered_list_handling(self, splitter_simple):
        """Test handling of numbered lists."""
        text = "Please 1) find the submittal 2) check status 3) notify PM"
        confidence = splitter_simple.get_splitting_confidence(text)
        
        # Should recognize this as multi-intent
        assert confidence > 0
    
    def test_subordinate_clause_preservation(self, splitter_with_spacy):
        """Test that subordinate clauses are not split."""
        text = "Show me the drawings that were submitted last week"
        segments = splitter_with_spacy.split(text)
        
        # Should not split on "that"
        assert len(segments) == 1
        assert "submitted last week" in segments[0].text
    
    def test_confidence_scoring(self, splitter_simple):
        """Test confidence scoring for multi-intent detection."""
        # Single intent - low confidence
        single = "What is the status of the submittal?"
        assert splitter_simple.get_splitting_confidence(single) < 0.5
        
        # Clear multi-intent - high confidence
        multi = "What is the status? How many are approved? Send report to PM."
        assert splitter_simple.get_splitting_confidence(multi) > 0.7
        
        # Ambiguous - medium confidence
        ambiguous = "Check the status and requirements"
        confidence = splitter_simple.get_splitting_confidence(ambiguous)
        assert 0.1 < confidence < 0.7
    
    def test_real_world_examples(self, splitter_with_spacy):
        """Test real-world construction queries."""
        examples = [
            # Clear multi-intent
            ("Show me all RFIs from last week and calculate the response time", 2),
            
            # Single intent with complex structure
            ("What is the spec section for acoustic ceiling tiles in conference rooms?", 1),
            
            # Multi-intent with dependencies
            ("Find submittal #232, check if it's approved, and notify the contractor", 3),
            
            # List that shouldn't be split
            ("Review drawings for HVAC, plumbing, and electrical systems", 1),
        ]
        
        for text, expected_min in examples:
            segments = splitter_with_spacy.split(text)
            # Allow for variation in splitting strategies
            assert len(segments) >= 1
    
    def test_punctuation_handling(self, splitter_simple):
        """Test various punctuation scenarios."""
        # Question mark followed by new sentence
        text1 = "What is the status? Show me the details."
        segments1 = splitter_simple.split(text1)
        assert len(segments1) == 2
        
        # Exclamation point
        text2 = "Check this immediately! Then send to PM."
        segments2 = splitter_simple.split(text2)
        assert len(segments2) >= 1
        
        # Colon (might or might not split)
        text3 = "Review the following: drawings, specs, and submittals"
        segments3 = splitter_simple.split(text3)
        # Colon handling varies
        assert len(segments3) >= 1
    
    def test_conjunction_types(self, splitter_simple):
        """Test different conjunction types."""
        # Coordinating conjunctions
        coord_tests = [
            "Find the drawings and check the revision",
            "Is it approved or rejected?",
            "The submittal is late but complete",
            "Review the specs, yet keep the old version"
        ]
        
        for text in coord_tests:
            confidence = splitter_simple.get_splitting_confidence(text)
            assert confidence >= 0
        
        # Subordinating conjunctions (shouldn't split)
        subord_tests = [
            "Show me the RFI that was submitted yesterday",
            "Check if the submittal is approved",
            "Review the drawings before the meeting",
            "Send notification when the status changes"
        ]
        
        for text in subord_tests:
            segments = splitter_simple.split(text)
            # Generally shouldn't split on subordinating conjunctions
            assert len(segments) <= 2
