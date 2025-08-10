"""
Composition builder module for handling multi-intent queries.
Determines how multiple intents should be composed and executed.
"""

import logging
from typing import List, Dict, Any, Tuple, Optional, Set
from dataclasses import dataclass
from collections import defaultdict
import networkx as nx

from intent_classifier.models.schemas import (
    Intent, Composition, CompositionMode, JoinKey, ClauseSegment
)
from intent_classifier.core.taxonomy_loader import TaxonomyLoader


logger = logging.getLogger(__name__)


@dataclass
class IntentNode:
    """Represents an intent in the composition graph."""
    clause_idx: int
    intent: Intent
    clause_text: str
    dependencies: List[int]  # Indices of clauses this depends on


@dataclass
class CompositionAnalysis:
    """Analysis result for multi-intent composition."""
    mode: CompositionMode
    intent_graph: nx.DiGraph
    execution_order: List[int]
    join_keys: List[JoinKey]
    response_policy: str
    warnings: List[str]


class CompositionBuilder:
    """
    Builds composition plans for multi-intent queries.
    Analyzes intent relationships and determines execution order.
    """
    
    def __init__(self, taxonomy_loader: TaxonomyLoader):
        """
        Initialize the composition builder.
        
        Args:
            taxonomy_loader: Taxonomy loader instance
        """
        self.taxonomy = taxonomy_loader
        
        # Common entity types that can be passed between intents
        self.shareable_entities = {
            'ids.rfi', 'ids.cb', 'ids.submittal',
            'spec_section', 'product_code', 'door_id',
            'floor', 'area', 'discipline', 'topic',
            'submittal_topic', 'date_range'
        }
        
        # Intent pairs that commonly chain together
        self.common_chains = self._load_common_chains()
    
    def _load_common_chains(self) -> Dict[str, List[str]]:
        """Load common intent chains from taxonomy."""
        chains = defaultdict(list)
        
        # Get allowed chains from taxonomy
        allowed_chains = self.taxonomy.taxonomy_data.get('allowed_chains', [])
        
        for chain in allowed_chains:
            from_intent = chain.get('from_intent', '')
            to_intent = chain.get('to_intent', '')
            
            if from_intent and to_intent:
                # Handle wildcards
                if '*' not in from_intent and '*' not in to_intent:
                    chains[from_intent].append(to_intent)
        
        # Add some common patterns if not in taxonomy
        default_chains = {
            'DOC:RFI_RETRIEVE': ['STAT:RFI_STATUS', 'RESP:RFI_RESPONSE'],
            'DOC:SUBMITTAL_RETRIEVE': ['STAT:SUBMITTAL_STATUS'],
            'COUNT:RFI_COUNT': ['DOC:RFI_RETRIEVE'],
            'COUNT:SUBMITTAL_COUNT': ['DOC:SUBMITTAL_RETRIEVE'],
            'STAT:RFI_STATUS': ['RESP:RFI_RESPONSE'],
            'LINK:RFI_TO_CB': ['DOC:CB_RETRIEVE', 'STAT:CB_STATUS'],
        }
        
        for from_intent, to_intents in default_chains.items():
            if from_intent not in chains:
                chains[from_intent] = to_intents
            else:
                # Merge lists
                chains[from_intent].extend(
                    [t for t in to_intents if t not in chains[from_intent]]
                )
        
        return dict(chains)
    
    def analyze_composition(self, clauses: List[ClauseSegment], 
                          intents: List[Intent]) -> CompositionAnalysis:
        """
        Analyze how multiple intents should be composed.
        
        Args:
            clauses: List of clause segments
            intents: List of classified intents (one per clause)
            
        Returns:
            CompositionAnalysis with execution plan
        """
        if len(intents) == 0:
            raise ValueError("No intents to compose")
        
        if len(intents) == 1:
            # Single intent - simple case
            return CompositionAnalysis(
                mode=CompositionMode.SINGLE,
                intent_graph=nx.DiGraph(),
                execution_order=[0],
                join_keys=[],
                response_policy="single_message",
                warnings=[]
            )
        
        # Create intent nodes
        nodes = []
        for i, (clause, intent) in enumerate(zip(clauses, intents)):
            node = IntentNode(
                clause_idx=i,
                intent=intent,
                clause_text=clause.text,
                dependencies=clause.dependencies
            )
            nodes.append(node)
        
        # Build intent graph
        graph = self._build_intent_graph(nodes)
        
        # Determine composition mode
        mode = self._determine_composition_mode(graph, nodes)
        
        # Determine execution order
        execution_order = self._determine_execution_order(graph, nodes, mode)
        
        # Identify join keys
        join_keys = self._identify_join_keys(nodes, execution_order)
        
        # Determine response policy
        response_policy = self._determine_response_policy(nodes, mode)
        
        # Validate composition
        warnings = self._validate_composition(nodes, graph, execution_order)
        
        return CompositionAnalysis(
            mode=mode,
            intent_graph=graph,
            execution_order=execution_order,
            join_keys=join_keys,
            response_policy=response_policy,
            warnings=warnings
        )
    
    def _build_intent_graph(self, nodes: List[IntentNode]) -> nx.DiGraph:
        """
        Build a directed graph of intent relationships.
        
        Args:
            nodes: List of intent nodes
            
        Returns:
            NetworkX directed graph
        """
        graph = nx.DiGraph()
        
        # Add nodes
        for i, node in enumerate(nodes):
            graph.add_node(i, intent=node.intent.intent_code, clause=node.clause_text)
        
        # Add edges based on dependencies
        for i, node in enumerate(nodes):
            for dep_idx in node.dependencies:
                if 0 <= dep_idx < len(nodes):
                    graph.add_edge(dep_idx, i, type='dependency')
        
        # Add edges based on entity flow
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                if self._can_flow_entities(nodes[i], nodes[j]):
                    graph.add_edge(i, j, type='entity_flow')
        
        # Add edges based on common chains
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                from_intent = nodes[i].intent.intent_code
                to_intent = nodes[j].intent.intent_code
                
                if to_intent in self.common_chains.get(from_intent, []):
                    graph.add_edge(i, j, type='common_chain')
                elif self.taxonomy.is_allowed_chain(from_intent, to_intent):
                    graph.add_edge(i, j, type='allowed_chain')
        
        return graph
    
    def _can_flow_entities(self, from_node: IntentNode, to_node: IntentNode) -> bool:
        """
        Check if entities can flow from one intent to another.
        
        Args:
            from_node: Source intent node
            to_node: Target intent node
            
        Returns:
            True if entity flow is possible
        """
        # Get entities from source intent
        from_entities = set(from_node.intent.entities.keys())
        
        # Get required entities for target intent
        to_intent_info = self.taxonomy.get_intent_info(to_node.intent.intent_code)
        if not to_intent_info:
            return False
        
        required_entities = set(to_intent_info.get('required_entities', []))
        
        # Check if source provides any required entities
        shared = from_entities & required_entities & self.shareable_entities
        
        return len(shared) > 0
    
    def _determine_composition_mode(self, graph: nx.DiGraph, 
                                  nodes: List[IntentNode]) -> CompositionMode:
        """
        Determine the composition mode based on graph structure.
        
        Args:
            graph: Intent relationship graph
            nodes: List of intent nodes
            
        Returns:
            Composition mode
        """
        if len(nodes) == 1:
            return CompositionMode.SINGLE
        
        # Check if graph is acyclic (can be sequential)
        if nx.is_directed_acyclic_graph(graph):
            # Check if there's a clear path through all nodes
            try:
                # Try to find a topological order
                topo_order = list(nx.topological_sort(graph))
                
                # Check if order covers all nodes and has dependencies
                if len(topo_order) == len(nodes) and graph.number_of_edges() > 0:
                    return CompositionMode.SEQUENCE
            except nx.NetworkXError:
                pass
        
        # Default to parallel execution
        return CompositionMode.PARALLEL
    
    def _determine_execution_order(self, graph: nx.DiGraph, nodes: List[IntentNode],
                                 mode: CompositionMode) -> List[int]:
        """
        Determine the execution order for intents.
        
        Args:
            graph: Intent relationship graph
            nodes: List of intent nodes
            mode: Composition mode
            
        Returns:
            List of node indices in execution order
        """
        if mode == CompositionMode.SINGLE:
            return [0]
        
        if mode == CompositionMode.SEQUENCE:
            try:
                # Use topological sort for sequential execution
                return list(nx.topological_sort(graph))
            except nx.NetworkXError:
                # Fall back to original order if cycle detected
                logger.warning("Cycle detected in intent graph, using original order")
                return list(range(len(nodes)))
        
        # For parallel mode, order by priority
        # Prioritize: retrieval > status > count > others
        priority_map = {
            'DOC': 1,
            'SPEC': 2,
            'STAT': 3,
            'COUNT': 4,
            'RESP': 5,
            'LINK': 6,
            'other': 10
        }
        
        def get_priority(node_idx):
            intent_code = nodes[node_idx].intent.intent_code
            prefix = intent_code.split(':')[0]
            return priority_map.get(prefix, priority_map['other'])
        
        # Sort by priority, maintaining relative order for same priority
        indices = list(range(len(nodes)))
        indices.sort(key=lambda i: (get_priority(i), i))
        
        return indices
    
    def _identify_join_keys(self, nodes: List[IntentNode], 
                          execution_order: List[int]) -> List[JoinKey]:
        """
        Identify entity keys that should be passed between intents.
        
        Args:
            nodes: List of intent nodes
            execution_order: Execution order indices
            
        Returns:
            List of join keys
        """
        join_keys = []
        
        # For sequential execution, identify entity flow
        if len(execution_order) > 1:
            for i in range(len(execution_order) - 1):
                from_idx = execution_order[i]
                to_idx = execution_order[i + 1]
                
                from_node = nodes[from_idx]
                to_node = nodes[to_idx]
                
                # Get entities from source
                from_entities = set(from_node.intent.entities.keys())
                
                # Get required entities for target
                to_intent_info = self.taxonomy.get_intent_info(to_node.intent.intent_code)
                if to_intent_info:
                    required = set(to_intent_info.get('required_entities', []))
                    
                    # Find shareable entities that are required
                    shared = from_entities & required & self.shareable_entities
                    
                    for entity_key in shared:
                        join_key = JoinKey(
                            from_step=from_idx,
                            to_step=to_idx,
                            key=entity_key
                        )
                        join_keys.append(join_key)
        
        return join_keys
    
    def _determine_response_policy(self, nodes: List[IntentNode], 
                                 mode: CompositionMode) -> str:
        """
        Determine how responses should be formatted.
        
        Args:
            nodes: List of intent nodes
            mode: Composition mode
            
        Returns:
            Response policy name
        """
        if mode == CompositionMode.SINGLE:
            return "single_message"
        
        # Check intent types
        intent_types = [node.intent.intent_code.split(':')[0] for node in nodes]
        
        # If all are retrieval/document intents, combine
        if all(t in ['DOC', 'DRAW', 'SPEC'] for t in intent_types):
            return "combined_results"
        
        # If mix of query and status, separate
        if any(t in ['STAT', 'COUNT'] for t in intent_types):
            return "separate_messages"
        
        # For sequential with dependencies, show progression
        if mode == CompositionMode.SEQUENCE:
            return "step_by_step"
        
        # Default to separate messages
        return "separate_messages"
    
    def _validate_composition(self, nodes: List[IntentNode], graph: nx.DiGraph,
                            execution_order: List[int]) -> List[str]:
        """
        Validate the composition plan and identify potential issues.
        
        Args:
            nodes: List of intent nodes
            graph: Intent relationship graph
            execution_order: Execution order
            
        Returns:
            List of warning messages
        """
        warnings = []
        
        # Check for incompatible intent combinations
        intent_codes = [node.intent.intent_code for node in nodes]
        
        # Warn about multiple status checks
        status_intents = [ic for ic in intent_codes if ic.startswith('STAT:')]
        if len(status_intents) > 2:
            warnings.append(f"Multiple status checks ({len(status_intents)}) may be redundant")
        
        # Check for missing dependencies
        for i, node in enumerate(nodes):
            intent_info = self.taxonomy.get_intent_info(node.intent.intent_code)
            if intent_info:
                required = intent_info.get('required_entities', [])
                available = set(node.intent.entities.keys())
                
                # Check if required entities will be provided by previous intents
                for j in execution_order:
                    if j < i:
                        available.update(nodes[j].intent.entities.keys())
                
                missing = set(required) - available
                if missing:
                    warnings.append(
                        f"Intent {node.intent.intent_code} missing required entities: {missing}"
                    )
        
        # Check for cycles in sequential mode
        if not nx.is_directed_acyclic_graph(graph):
            warnings.append("Circular dependencies detected between intents")
        
        # Check chain validity
        for i in range(len(execution_order) - 1):
            from_intent = nodes[execution_order[i]].intent.intent_code
            to_intent = nodes[execution_order[i + 1]].intent.intent_code
            
            if not self.taxonomy.is_allowed_chain(from_intent, to_intent):
                # Only warn if there's an explicit connection
                if graph.has_edge(execution_order[i], execution_order[i + 1]):
                    warnings.append(
                        f"Unusual intent sequence: {from_intent} → {to_intent}"
                    )
        
        return warnings
    
    def build_composition(self, clauses: List[ClauseSegment], 
                        intents: List[Intent]) -> Composition:
        """
        Build the final composition object.
        
        Args:
            clauses: List of clause segments
            intents: List of classified intents
            
        Returns:
            Composition object for the output schema
        """
        # Analyze composition
        analysis = self.analyze_composition(clauses, intents)
        
        # Log warnings
        for warning in analysis.warnings:
            logger.warning(f"Composition warning: {warning}")
        
        # Create composition object
        composition = Composition(
            mode=analysis.mode,
            ordering=analysis.execution_order,
            join_keys=analysis.join_keys,
            response_policy=analysis.response_policy
        )
        
        return composition
    
    def explain_composition(self, composition: Composition, intents: List[Intent]) -> str:
        """
        Generate a human-readable explanation of the composition.
        
        Args:
            composition: Composition object
            intents: List of intents
            
        Returns:
            Explanation string
        """
        explanations = []
        
        # Explain mode
        if composition.mode == CompositionMode.SINGLE:
            explanations.append("Single intent query")
        elif composition.mode == CompositionMode.SEQUENCE:
            explanations.append("Sequential execution required")
        else:
            explanations.append("Parallel execution possible")
        
        # Explain order
        if len(composition.ordering) > 1:
            ordered_intents = [intents[i].intent_code for i in composition.ordering]
            explanations.append(f"Execution order: {' → '.join(ordered_intents)}")
        
        # Explain data flow
        if composition.join_keys:
            for jk in composition.join_keys:
                from_intent = intents[jk.from_step].intent_code
                to_intent = intents[jk.to_step].intent_code
                explanations.append(
                    f"Pass {jk.key} from {from_intent} to {to_intent}"
                )
        
        # Explain response policy
        policy_explanations = {
            'single_message': "Single response message",
            'combined_results': "Combine all results",
            'separate_messages': "Separate response per intent",
            'step_by_step': "Show step-by-step progression"
        }
        
        explanations.append(
            policy_explanations.get(
                composition.response_policy, 
                composition.response_policy
            )
        )
        
        return " | ".join(explanations)
