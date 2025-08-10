"""
Tests for the composition builder module.
"""

import pytest
import networkx as nx
from intent_classifier.core.composition_builder import (
    CompositionBuilder, IntentNode, CompositionAnalysis
)
from intent_classifier.models.schemas import (
    Intent, Composition, CompositionMode, JoinKey, ClauseSegment
)
from intent_classifier.core.taxonomy_loader import TaxonomyLoader


class TestCompositionBuilder:
    """Test the CompositionBuilder class."""
    
    @pytest.fixture
    def mock_taxonomy(self):
        """Create a mock taxonomy loader."""
        class MockTaxonomy:
            def __init__(self):
                self.taxonomy_data = {
                    'allowed_chains': [
                        {'from_intent': 'DOC:RFI_RETRIEVE', 'to_intent': 'STAT:RFI_STATUS'},
                        {'from_intent': 'COUNT:*', 'to_intent': 'DOC:*'},
                        {'from_intent': 'STAT:*', 'to_intent': 'RESP:*'}
                    ]
                }
            
            def get_intent_info(self, intent_code):
                intent_map = {
                    'DOC:RFI_RETRIEVE': {
                        'required_entities': ['ids.rfi'],
                        'optional_entities': []
                    },
                    'STAT:RFI_STATUS': {
                        'required_entities': ['ids.rfi'],
                        'optional_entities': []
                    },
                    'DOC:SUBMITTAL_RETRIEVE': {
                        'required_entities': ['submittal_topic'],
                        'optional_entities': ['ids.submittal']
                    },
                    'COUNT:RFI_COUNT': {
                        'required_entities': [],
                        'optional_entities': ['discipline', 'date_range']
                    },
                    'RESP:RFI_RESPONSE': {
                        'required_entities': ['ids.rfi'],
                        'optional_entities': []
                    }
                }
                return intent_map.get(intent_code, {})
            
            def is_allowed_chain(self, chain):
                if len(chain) != 2:
                    return False
                
                from_intent, to_intent = chain
                
                # Check exact matches
                for allowed in self.taxonomy_data['allowed_chains']:
                    if (allowed['from_intent'] == from_intent and 
                        allowed['to_intent'] == to_intent):
                        return True
                
                # Check wildcards
                from_prefix = from_intent.split(':')[0]
                to_prefix = to_intent.split(':')[0]
                
                for allowed in self.taxonomy_data['allowed_chains']:
                    allowed_from = allowed['from_intent']
                    allowed_to = allowed['to_intent']
                    
                    if allowed_from.endswith('*') and allowed_to.endswith('*'):
                        if (from_prefix == allowed_from[:-2] and 
                            to_prefix == allowed_to[:-2]):
                            return True
                
                return False
        
        return MockTaxonomy()
    
    @pytest.fixture
    def composition_builder(self, mock_taxonomy):
        """Create composition builder instance."""
        return CompositionBuilder(mock_taxonomy)
    
    def test_single_intent(self, composition_builder):
        """Test composition for single intent."""
        clauses = [ClauseSegment(
            text="What is the status of RFI 1838?",
            start_offset=0,
            end_offset=30,
            dependencies=[]
        )]
        
        intents = [Intent(
            coarse_class="STAT",
            intent_code="STAT:RFI_STATUS",
            confidence=0.9,
            entities={"ids.rfi": 1838}
        )]
        
        analysis = composition_builder.analyze_composition(clauses, intents)
        
        assert analysis.mode == CompositionMode.SINGLE
        assert analysis.execution_order == [0]
        assert len(analysis.join_keys) == 0
        assert analysis.response_policy == "single_message"
    
    def test_sequential_intents(self, composition_builder):
        """Test sequential intent composition."""
        clauses = [
            ClauseSegment(
                text="Find RFI 123",
                start_offset=0,
                end_offset=12,
                dependencies=[]
            ),
            ClauseSegment(
                text="check its status",
                start_offset=17,
                end_offset=33,
                dependencies=[0]  # Depends on first clause
            )
        ]
        
        intents = [
            Intent(
                coarse_class="DOC",
                intent_code="DOC:RFI_RETRIEVE",
                confidence=0.85,
                entities={"ids.rfi": 123}
            ),
            Intent(
                coarse_class="STAT",
                intent_code="STAT:RFI_STATUS",
                confidence=0.8,
                entities={"ids.rfi": 123}
            )
        ]
        
        analysis = composition_builder.analyze_composition(clauses, intents)
        
        assert analysis.mode == CompositionMode.SEQUENCE
        assert analysis.execution_order == [0, 1]
        assert len(analysis.join_keys) > 0
        
        # Check join key
        join_key = analysis.join_keys[0]
        assert join_key.from_step == 0
        assert join_key.to_step == 1
        assert join_key.key == "ids.rfi"
    
    def test_parallel_intents(self, composition_builder):
        """Test parallel intent composition."""
        clauses = [
            ClauseSegment(
                text="Show all submittals",
                start_offset=0,
                end_offset=19,
                dependencies=[]
            ),
            ClauseSegment(
                text="count open RFIs",
                start_offset=24,
                end_offset=39,
                dependencies=[]
            )
        ]
        
        intents = [
            Intent(
                coarse_class="DOC",
                intent_code="DOC:SUBMITTAL_RETRIEVE",
                confidence=0.9,
                entities={"submittal_topic": "all"}
            ),
            Intent(
                coarse_class="COUNT",
                intent_code="COUNT:RFI_COUNT",
                confidence=0.85,
                entities={}
            )
        ]
        
        analysis = composition_builder.analyze_composition(clauses, intents)
        
        assert analysis.mode == CompositionMode.PARALLEL
        # Should prioritize DOC over COUNT
        assert analysis.execution_order[0] == 0
        assert len(analysis.join_keys) == 0  # No shared entities
    
    def test_entity_flow(self, composition_builder):
        """Test entity flow detection."""
        nodes = [
            IntentNode(
                clause_idx=0,
                intent=Intent(
                    coarse_class="DOC",
                    intent_code="DOC:RFI_RETRIEVE",
                    confidence=0.9,
                    entities={"ids.rfi": 456}
                ),
                clause_text="Find RFI 456",
                dependencies=[]
            ),
            IntentNode(
                clause_idx=1,
                intent=Intent(
                    coarse_class="RESP",
                    intent_code="RESP:RFI_RESPONSE",
                    confidence=0.85,
                    entities={}  # Needs ids.rfi
                ),
                clause_text="show response",
                dependencies=[]
            )
        ]
        
        # Test entity flow detection
        can_flow = composition_builder._can_flow_entities(nodes[0], nodes[1])
        assert can_flow == True
        
        # Build graph and check edges
        graph = composition_builder._build_intent_graph(nodes)
        assert graph.has_edge(0, 1)
    
    def test_chain_validation(self, composition_builder, mock_taxonomy):
        """Test intent chain validation."""
        # Valid chain
        assert mock_taxonomy.is_allowed_chain([
            'DOC:RFI_RETRIEVE',
            'STAT:RFI_STATUS'
        ]) == True
        
        # Valid wildcard chain
        assert mock_taxonomy.is_allowed_chain([
            'COUNT:RFI_COUNT',
            'DOC:RFI_RETRIEVE'
        ]) == True
        
        # Invalid chain
        assert mock_taxonomy.is_allowed_chain([
            'DOC:RFI_RETRIEVE',
            'COUNT:SUBMITTAL_COUNT'
        ]) == False
    
    def test_response_policy_determination(self, composition_builder):
        """Test response policy determination."""
        # All document retrieval - should combine
        doc_nodes = [
            IntentNode(0, Intent("DOC", "DOC:RFI_RETRIEVE", 0.9, {}), "", []),
            IntentNode(1, Intent("DOC", "DOC:SUBMITTAL_RETRIEVE", 0.85, {}), "", [])
        ]
        
        policy = composition_builder._determine_response_policy(
            doc_nodes, CompositionMode.PARALLEL
        )
        assert policy == "combined_results"
        
        # Mix of status and count - separate
        mixed_nodes = [
            IntentNode(0, Intent("STAT", "STAT:RFI_STATUS", 0.9, {}), "", []),
            IntentNode(1, Intent("COUNT", "COUNT:RFI_COUNT", 0.85, {}), "", [])
        ]
        
        policy = composition_builder._determine_response_policy(
            mixed_nodes, CompositionMode.PARALLEL
        )
        assert policy == "separate_messages"
    
    def test_validation_warnings(self, composition_builder):
        """Test composition validation."""
        nodes = [
            IntentNode(
                clause_idx=0,
                intent=Intent(
                    coarse_class="STAT",
                    intent_code="STAT:RFI_STATUS",
                    confidence=0.9,
                    entities={}  # Missing required ids.rfi
                ),
                clause_text="check status",
                dependencies=[]
            ),
            IntentNode(
                clause_idx=1,
                intent=Intent(
                    coarse_class="STAT",
                    intent_code="STAT:SUBMITTAL_STATUS",
                    confidence=0.85,
                    entities={}
                ),
                clause_text="submittal status",
                dependencies=[]
            ),
            IntentNode(
                clause_idx=2,
                intent=Intent(
                    coarse_class="STAT",
                    intent_code="STAT:CB_STATUS",
                    confidence=0.8,
                    entities={}
                ),
                clause_text="CB status",
                dependencies=[]
            )
        ]
        
        graph = nx.DiGraph()
        warnings = composition_builder._validate_composition(nodes, graph, [0, 1, 2])
        
        # Should warn about missing entities
        assert any("missing required entities" in w for w in warnings)
        
        # Should warn about multiple status checks
        assert any("Multiple status checks" in w for w in warnings)
    
    def test_build_composition(self, composition_builder):
        """Test building final composition object."""
        clauses = [
            ClauseSegment("Find RFI 789", 0, 12, []),
            ClauseSegment("and its status", 13, 27, [0])
        ]
        
        intents = [
            Intent("DOC", "DOC:RFI_RETRIEVE", 0.9, {"ids.rfi": 789}),
            Intent("STAT", "STAT:RFI_STATUS", 0.85, {"ids.rfi": 789})
        ]
        
        composition = composition_builder.build_composition(clauses, intents)
        
        assert isinstance(composition, Composition)
        assert composition.mode == CompositionMode.SEQUENCE
        assert len(composition.ordering) == 2
        assert len(composition.join_keys) > 0
    
    def test_explain_composition(self, composition_builder):
        """Test composition explanation generation."""
        intents = [
            Intent("DOC", "DOC:RFI_RETRIEVE", 0.9, {"ids.rfi": 123}),
            Intent("STAT", "STAT:RFI_STATUS", 0.85, {"ids.rfi": 123})
        ]
        
        composition = Composition(
            mode=CompositionMode.SEQUENCE,
            ordering=[0, 1],
            join_keys=[JoinKey(from_step=0, to_step=1, key="ids.rfi")],
            response_policy="step_by_step"
        )
        
        explanation = composition_builder.explain_composition(composition, intents)
        
        assert "Sequential execution" in explanation
        assert "DOC:RFI_RETRIEVE → STAT:RFI_STATUS" in explanation
        assert "Pass ids.rfi" in explanation
        assert "step-by-step" in explanation
    
    def test_complex_multi_intent(self, composition_builder):
        """Test complex multi-intent scenario."""
        clauses = [
            ClauseSegment("How many RFIs are open", 0, 22, []),
            ClauseSegment("show me the latest one", 23, 45, [0]),
            ClauseSegment("and check its response", 46, 68, [1])
        ]
        
        intents = [
            Intent("COUNT", "COUNT:RFI_COUNT", 0.9, {}),
            Intent("DOC", "DOC:RFI_RETRIEVE", 0.85, {"ids.rfi": 999}),
            Intent("RESP", "RESP:RFI_RESPONSE", 0.8, {"ids.rfi": 999})
        ]
        
        analysis = composition_builder.analyze_composition(clauses, intents)
        
        # Should be sequential due to dependencies
        assert analysis.mode == CompositionMode.SEQUENCE
        assert analysis.execution_order == [0, 1, 2]
        
        # Should have join keys for RFI ID
        rfi_joins = [jk for jk in analysis.join_keys if jk.key == "ids.rfi"]
        assert len(rfi_joins) > 0
