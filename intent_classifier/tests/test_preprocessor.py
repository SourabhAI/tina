"""
Tests for the preprocessor module.
"""

import pytest
from intent_classifier.core.preprocessor import Preprocessor


class TestPreprocessor:
    """Test the Preprocessor class."""
    
    @pytest.fixture
    def preprocessor(self):
        """Create a preprocessor instance for testing."""
        return Preprocessor(enable_spell_correction=True)
    
    def test_basic_cleaning(self, preprocessor):
        """Test basic text cleaning."""
        # Test multiple spaces
        assert preprocessor.process("What  is   the    status?") == "What is the status?"
        
        # Test tabs and newlines
        assert preprocessor.process("What\tis\nthe status?") == "What is the status?"
        
        # Test duplicate punctuation
        assert preprocessor.process("What is the status???") == "What is the status?"
        assert preprocessor.process("Really!!!") == "Really!"
    
    def test_abbreviation_expansion(self, preprocessor):
        """Test abbreviation expansion."""
        # Construction abbreviations
        assert preprocessor.process("Show me the mech dwgs") == "Show me the mechanical drawings"
        assert preprocessor.process("Check the elec specs") == "Check the electrical specifications"
        assert preprocessor.process("Where is the ahu located?") == "Where is the AHU located?"
        
        # ID abbreviations (should be uppercase)
        assert preprocessor.process("What is the status of rfi 123?") == "What is the status of RFI:123?"
        assert preprocessor.process("Show cb for this rfi") == "Show CB for this RFI"
        
        # Unit abbreviations
        assert preprocessor.process("Room is 100 sf") == "Room is 100 square feet"
        assert preprocessor.process("Pressure is 50 psi") == "Pressure is 50 PSI"
    
    def test_id_normalization(self, preprocessor):
        """Test ID normalization."""
        # RFI normalization
        assert preprocessor.process("RFI #1838") == "RFI:1838"
        assert preprocessor.process("RFI#1838") == "RFI:1838"
        assert preprocessor.process("rfi 1838") == "RFI:1838"
        assert preprocessor.process("Show me RFI # 1838") == "Show me RFI:1838"
        
        # CB normalization
        assert preprocessor.process("CB #309") == "CB:309"
        assert preprocessor.process("cb309") == "cb309"  # Without space, not normalized
        assert preprocessor.process("CB 309") == "CB:309"
        
        # Submittal normalization
        assert preprocessor.process("submittal #232") == "submittal:232"
        assert preprocessor.process("Submittal# 232") == "submittal:232"
        
        # Door normalization (only in context)
        assert preprocessor.process("door 20110-2") == "door 20110-2"
        assert preprocessor.process("door 20110 - 2") == "door 20110-2"
        assert preprocessor.process("door #20110") == "door 20110"
    
    def test_csi_section_normalization(self, preprocessor):
        """Test CSI section normalization."""
        assert preprocessor.process("Section 102233") == "Section 10 22 33"
        assert preprocessor.process("See 10-22-33") == "See 10 22 33"
        assert preprocessor.process("Ref 10.22.33") == "Ref 10 22 33"
        assert preprocessor.process("CSI 10 22 33") == "CSI 10 22 33"  # Already normalized
    
    def test_product_code_normalization(self, preprocessor):
        """Test product code normalization."""
        assert preprocessor.process("Replace ACT 9") == "Replace ACT-9"
        assert preprocessor.process("Install VAV-10") == "Install VAV-10"  # Already normalized
        assert preprocessor.process("Check AP 2") == "Check AP-2"
        assert preprocessor.process("DF4 specifications") == "DF4 specifications"  # No space, not normalized
    
    def test_unit_normalization(self, preprocessor):
        """Test unit and measurement normalization."""
        # Area units
        assert preprocessor.process("Room is 500 sf") == "Room is 500 square feet"
        assert preprocessor.process("Total area: 1000 sy") == "Total area: 1000 square yards"
        
        # Dimensions
        assert preprocessor.process("Size is 4x8") == "Size is 4 x 8"
        assert preprocessor.process("Door is 3' x 7'") == "Door is 3 x 7"
        
        # Percentages
        assert preprocessor.process("Complete 75%") == "Complete 75 percent"
        assert preprocessor.process("50 percent done") == "50 percent done"  # Already normalized
        
        # Temperature
        assert preprocessor.process("Set to 72°F") == "Set to 72 degrees F"
        assert preprocessor.process("Maintain 20 deg C") == "Maintain 20 degrees C"
        
        # Ranges
        assert preprocessor.process("Pages 10-15") == "Pages 10 to 15"
        assert preprocessor.process("Floors 3 thru 7") == "Floors 3 to 7"
    
    def test_spelling_correction(self, preprocessor):
        """Test spelling correction."""
        assert preprocessor.process("What is the specificaiton?") == "What is the specification?"
        assert preprocessor.process("Show submital status") == "Show submittal status"
        assert preprocessor.process("Check mechnical drawings") == "Check mechanical drawings"
        assert preprocessor.process("Review electical plans") == "Review electrical plans"
    
    def test_unicode_normalization(self, preprocessor):
        """Test unicode character normalization."""
        assert preprocessor.process("What's the status?") == "What's the status?"
        assert preprocessor.process(""Smart quotes"") == '"Smart quotes"'
        assert preprocessor.process("Temperature: 72°F") == "Temperature: 72 degrees F"
        assert preprocessor.process("Size: 4×8") == "Size: 4 x 8"
        assert preprocessor.process("Plus–minus ±5") == "Plus-minus +/-5"
    
    def test_question_preservation(self, preprocessor):
        """Test that question marks are preserved."""
        assert preprocessor.process("What is the status?") == "What is the status?"
        assert preprocessor.process("What is the status") == "What is the status"  # No question mark added
        assert preprocessor.process("What is the status ?") == "What is the status?"
        assert preprocessor.process("Is it complete? ") == "Is it complete?"
    
    def test_complex_queries(self, preprocessor):
        """Test complex real-world queries."""
        # Complex query 1
        input1 = "Show me the mech submital for AHU-5 on flr 3"
        expected1 = "Show me the mechanical submittal for AHU-5 on floor 3"
        assert preprocessor.process(input1) == expected1
        
        # Complex query 2
        input2 = "What's the status of RFI #1838 and CB# 309?"
        expected2 = "What's the status of RFI:1838 and CB:309?"
        assert preprocessor.process(input2) == expected2
        
        # Complex query 3
        input3 = "Review specs section 102233 for 500 sf room"
        expected3 = "Review specifications section 10 22 33 for 500 square feet room"
        assert preprocessor.process(input3) == expected3
        
        # Complex query 4
        input4 = "Is door 20110-2 rated for 90 min @ 1000°F?"
        expected4 = "Is door 20110-2 rated for 90 minimum @ 1000 degrees F?"
        assert preprocessor.process(input4) == expected4
    
    def test_batch_processing(self, preprocessor):
        """Test batch processing."""
        texts = [
            "What is RFI #123?",
            "Show cb 456",
            "Check mech specs"
        ]
        
        expected = [
            "What is RFI:123?",
            "Show CB:456",
            "Check mechanical specifications"
        ]
        
        results = preprocessor.batch_process(texts)
        assert results == expected
    
    def test_preprocessing_stats(self, preprocessor):
        """Test preprocessing statistics."""
        text = "Show me RFI #123 and the mech specs for 100 sf"
        stats = preprocessor.get_preprocessing_stats(text)
        
        assert stats['original_length'] == len(text)
        assert stats['abbreviations_expanded'] >= 2  # 'mech' and 'sf'
        assert stats['ids_found'] >= 1  # RFI #123
        assert stats['units_normalized'] >= 1  # 100 sf
        assert stats['has_question_mark'] == False
        assert 'processed_text' in stats
        assert 'original_text' in stats
    
    def test_edge_cases(self, preprocessor):
        """Test edge cases."""
        # Empty string
        assert preprocessor.process("") == ""
        
        # Only whitespace
        assert preprocessor.process("   \t\n   ") == ""
        
        # Only punctuation
        assert preprocessor.process("???!!!...") == "?!..."
        
        # Very long input
        long_text = "What " * 100 + "?"
        result = preprocessor.process(long_text)
        assert result.endswith("?")
        assert "What" in result
        
        # Mixed case preservation
        assert preprocessor.process("Check HVAC system") == "Check HVAC system"
        assert preprocessor.process("Review IBC code") == "Review IBC code"
    
    def test_no_over_normalization(self, preprocessor):
        """Test that we don't over-normalize."""
        # Don't normalize numbers that aren't IDs
        assert preprocessor.process("Year 2023") == "Year 2023"
        assert preprocessor.process("Page 123") == "Page 123"
        
        # Don't normalize all numbers as doors
        assert preprocessor.process("Room 1234") == "Room 1234"
        assert preprocessor.process("Code 20110") == "Code 20110"
        
        # Preserve intentional formatting
        assert preprocessor.process("A/C unit") == "A/C unit"
        assert preprocessor.process("24/7 support") == "24/7 support"
