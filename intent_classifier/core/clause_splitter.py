"""
Clause splitter module for the intent classification system.
Splits multi-intent queries into individual clauses using dependency parsing and heuristics.
"""

import re
import logging
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass

try:
    import spacy
    from spacy.tokens import Doc, Token
except ImportError:
    spacy = None
    Doc = None
    Token = None
    
from intent_classifier.models.schemas import ClauseSegment


logger = logging.getLogger(__name__)


@dataclass
class SplitPoint:
    """Represents a potential split point in the text."""
    position: int
    token_idx: int
    split_type: str  # 'conjunction', 'punctuation', 'question', 'imperative'
    confidence: float
    token_text: str


class ClauseSplitter:
    """
    Splits queries into clauses for multi-intent detection.
    Uses spaCy dependency parsing combined with rule-based heuristics.
    """
    
    def __init__(self, spacy_model: str = "en_core_web_sm", use_simple_fallback: bool = True):
        """
        Initialize the clause splitter.
        
        Args:
            spacy_model: Name of the spaCy model to use
            use_simple_fallback: Whether to use simple splitting if spaCy is unavailable
        """
        self.use_simple_fallback = use_simple_fallback
        self.nlp = None
        
        # Try to load spaCy
        if spacy is not None:
            try:
                self.nlp = spacy.load(spacy_model)
                # Disable unnecessary components for speed
                disabled = ["ner", "lemmatizer", "textcat"]
                for component in disabled:
                    if component in self.nlp.pipe_names:
                        self.nlp.disable_pipes(component)
                logger.info(f"Loaded spaCy model: {spacy_model}")
            except Exception as e:
                logger.warning(f"Failed to load spaCy model {spacy_model}: {e}")
                logger.warning("Falling back to simple clause splitting")
                self.nlp = None
        else:
            logger.warning("spaCy not installed. Using simple clause splitting")
        
        # Conjunctions that often indicate clause boundaries
        self.coordinating_conjunctions = {
            'and', 'but', 'or', 'nor', 'for', 'yet', 'so',
            'plus', 'also', 'then', 'additionally', 'furthermore',
            'moreover', 'besides', 'as well as', 'along with'
        }
        
        # Subordinating conjunctions (usually don't split on these)
        self.subordinating_conjunctions = {
            'if', 'when', 'where', 'because', 'since', 'while',
            'although', 'though', 'unless', 'until', 'before',
            'after', 'whether', 'that', 'which', 'who'
        }
        
        # Question words that might start new clauses
        self.question_words = {
            'what', 'where', 'when', 'who', 'why', 'how',
            'which', 'whose', 'whom'
        }
        
        # Imperative verbs common in construction queries
        self.imperative_verbs = {
            'show', 'find', 'get', 'list', 'display', 'retrieve',
            'check', 'verify', 'confirm', 'calculate', 'count',
            'identify', 'locate', 'search', 'provide', 'give',
            'tell', 'explain', 'describe', 'review', 'inspect'
        }
        
        # Patterns that indicate clause boundaries
        self._compile_patterns()
        
    def _compile_patterns(self):
        """Compile regex patterns for efficiency."""
        # Pattern for multiple questions in one query
        self.multi_question_pattern = re.compile(
            r'[.?!]\s*(?=[A-Z])|(?<=[.?!])\s+(?=(?:what|where|when|who|why|how|which|is|are|can|will|does|do))',
            re.IGNORECASE
        )
        
        # Pattern for semicolons as clause separators
        self.semicolon_pattern = re.compile(r';\s*')
        
        # Pattern for "and" preceded by a comma (strong indicator)
        self.comma_and_pattern = re.compile(r',\s+(?:and|or|but)\s+', re.IGNORECASE)
        
        # Pattern for numbered lists
        self.numbered_list_pattern = re.compile(r'(?:^|\s)(?:\d+[.)]|\([a-z]\))\s*', re.MULTILINE)
        
        # Pattern for bullet points
        self.bullet_pattern = re.compile(r'(?:^|\s)[-•*]\s+', re.MULTILINE)
    
    def split(self, text: str) -> List[ClauseSegment]:
        """
        Split text into clause segments.
        
        Args:
            text: Input text to split
            
        Returns:
            List of ClauseSegment objects
        """
        if not text or not text.strip():
            return []
        
        # Try spaCy-based splitting first
        if self.nlp:
            try:
                return self._spacy_split(text)
            except Exception as e:
                logger.warning(f"spaCy splitting failed: {e}")
                if self.use_simple_fallback:
                    return self._simple_split(text)
                else:
                    raise
        
        # Fallback to simple splitting
        return self._simple_split(text)
    
    def _spacy_split(self, text: str) -> List[ClauseSegment]:
        """
        Split using spaCy dependency parsing.
        
        Args:
            text: Input text
            
        Returns:
            List of ClauseSegment objects
        """
        doc = self.nlp(text)
        
        # Find potential split points
        split_points = self._find_split_points(doc)
        
        # Sort by position
        split_points.sort(key=lambda x: x.position)
        
        # Filter split points based on confidence and rules
        filtered_splits = self._filter_split_points(split_points, doc)
        
        # Create clause segments
        segments = self._create_segments_from_splits(text, filtered_splits, doc)
        
        # Post-process segments
        segments = self._post_process_segments(segments)
        
        return segments
    
    def _find_split_points(self, doc: Doc) -> List[SplitPoint]:
        """
        Find potential split points in the document.
        
        Args:
            doc: spaCy Doc object
            
        Returns:
            List of SplitPoint objects
        """
        split_points = []
        
        for i, token in enumerate(doc):
            # Check for coordinating conjunctions
            if token.text.lower() in self.coordinating_conjunctions:
                # Check if it's actually coordinating (not within a noun phrase)
                if self._is_coordinating_conjunction(token):
                    split_points.append(SplitPoint(
                        position=token.idx,
                        token_idx=i,
                        split_type='conjunction',
                        confidence=0.8,
                        token_text=token.text
                    ))
            
            # Check for punctuation splits
            elif token.is_punct:
                if token.text in [';', ':', '?', '!', '.']:
                    # Check if followed by a new clause
                    if i + 1 < len(doc) and self._starts_new_clause(doc[i + 1:]):
                        split_points.append(SplitPoint(
                            position=token.idx + len(token.text),
                            token_idx=i,
                            split_type='punctuation',
                            confidence=0.9 if token.text == ';' else 0.7,
                            token_text=token.text
                        ))
            
            # Check for question words starting new clauses
            elif token.text.lower() in self.question_words:
                if self._is_new_question(token, i):
                    split_points.append(SplitPoint(
                        position=token.idx,
                        token_idx=i,
                        split_type='question',
                        confidence=0.7,
                        token_text=token.text
                    ))
            
            # Check for imperative verbs
            elif token.text.lower() in self.imperative_verbs:
                if self._is_new_imperative(token, i):
                    split_points.append(SplitPoint(
                        position=token.idx,
                        token_idx=i,
                        split_type='imperative',
                        confidence=0.6,
                        token_text=token.text
                    ))
        
        return split_points
    
    def _is_coordinating_conjunction(self, token: Token) -> bool:
        """
        Check if a token is truly functioning as a coordinating conjunction.
        
        Args:
            token: spaCy Token
            
        Returns:
            True if it's coordinating between clauses
        """
        # Check dependency relation
        if token.dep_ == "cc":
            # Check if it's connecting verb phrases or clauses
            if token.head.pos_ in ["VERB", "AUX"]:
                return True
            
            # Check if preceded by comma (strong indicator)
            if token.i > 0 and token.doc[token.i - 1].text == ',':
                return True
        
        # Special case for "and" between independent clauses
        if token.text.lower() == "and":
            # Look for verb on both sides
            left_has_verb = any(t.pos_ in ["VERB", "AUX"] for t in token.doc[:token.i])
            right_has_verb = any(t.pos_ in ["VERB", "AUX"] for t in token.doc[token.i + 1:])
            
            if left_has_verb and right_has_verb:
                return True
        
        return False
    
    def _starts_new_clause(self, span) -> bool:
        """
        Check if a span starts a new clause.
        
        Args:
            span: spaCy Span
            
        Returns:
            True if it appears to start a new clause
        """
        if not span:
            return False
        
        first_token = span[0]
        
        # Check for question word
        if first_token.text.lower() in self.question_words:
            return True
        
        # Check for imperative verb
        if first_token.text.lower() in self.imperative_verbs:
            return True
        
        # Check for capitalization (new sentence)
        if first_token.text[0].isupper() and first_token.i > 0:
            return True
        
        # Check for subject-verb pattern
        if len(span) > 1:
            if first_token.pos_ in ["NOUN", "PROPN", "PRON"] and span[1].pos_ in ["VERB", "AUX"]:
                return True
        
        return False
    
    def _is_new_question(self, token: Token, position: int) -> bool:
        """
        Check if a question word starts a new question clause.
        
        Args:
            token: spaCy Token
            position: Token position in document
            
        Returns:
            True if it starts a new question
        """
        # Not at the beginning (that's the first clause)
        if position == 0:
            return False
        
        # Check if preceded by punctuation or conjunction
        if position > 0:
            prev_token = token.doc[position - 1]
            if prev_token.is_punct or prev_token.text.lower() in self.coordinating_conjunctions:
                return True
        
        # Check if it's a root or has verb children
        if token.dep_ == "ROOT" or any(child.pos_ == "VERB" for child in token.children):
            return True
        
        return False
    
    def _is_new_imperative(self, token: Token, position: int) -> bool:
        """
        Check if an imperative verb starts a new command clause.
        
        Args:
            token: spaCy Token
            position: Token position in document
            
        Returns:
            True if it starts a new imperative clause
        """
        # Not at the beginning
        if position == 0:
            return False
        
        # Should be at or near the start of its clause
        if token.dep_ in ["ROOT", "advcl", "conj"]:
            return True
        
        # Check if preceded by conjunction or punctuation
        if position > 0:
            prev_token = token.doc[position - 1]
            if prev_token.text.lower() in self.coordinating_conjunctions:
                return True
            if prev_token.is_punct and prev_token.text in [',', ';', '.']:
                return True
        
        return False
    
    def _filter_split_points(self, split_points: List[SplitPoint], doc: Doc) -> List[SplitPoint]:
        """
        Filter split points based on confidence and rules.
        
        Args:
            split_points: List of potential split points
            doc: spaCy Doc object
            
        Returns:
            Filtered list of split points
        """
        filtered = []
        
        for i, split in enumerate(split_points):
            # Skip if confidence too low
            if split.confidence < 0.5:
                continue
            
            # Skip if too close to previous split
            if filtered and split.position - filtered[-1].position < 10:
                # Keep the one with higher confidence
                if split.confidence > filtered[-1].confidence:
                    filtered[-1] = split
                continue
            
            # Skip conjunctions within noun phrases or lists
            if split.split_type == 'conjunction':
                token = doc[split.token_idx]
                # Check if it's within a list (e.g., "A, B, and C")
                if self._is_within_list(token):
                    continue
            
            # Additional validation for question splits
            if split.split_type == 'question':
                # Ensure there's substantial content before
                if split.position < 15:  # Too early in text
                    continue
            
            filtered.append(split)
        
        return filtered
    
    def _is_within_list(self, token: Token) -> bool:
        """
        Check if a conjunction is within a list rather than between clauses.
        
        Args:
            token: spaCy Token
            
        Returns:
            True if within a list
        """
        # Look for pattern: NOUN, NOUN, and NOUN
        if token.text.lower() in ['and', 'or']:
            # Check if surrounded by similar POS tags
            left_pos = [t.pos_ for t in token.doc[max(0, token.i - 3):token.i]]
            right_pos = [t.pos_ for t in token.doc[token.i + 1:min(len(token.doc), token.i + 4)]]
            
            # Count nouns and commas
            left_nouns = sum(1 for p in left_pos if p in ["NOUN", "PROPN"])
            right_nouns = sum(1 for p in right_pos if p in ["NOUN", "PROPN"])
            
            if left_nouns > 0 and right_nouns > 0:
                # Check for list pattern
                left_text = token.doc[max(0, token.i - 5):token.i].text
                if ',' in left_text:
                    return True
        
        return False
    
    def _create_segments_from_splits(self, text: str, split_points: List[SplitPoint], doc: Doc) -> List[ClauseSegment]:
        """
        Create clause segments from split points.
        
        Args:
            text: Original text
            split_points: List of split points
            doc: spaCy Doc object
            
        Returns:
            List of ClauseSegment objects
        """
        segments = []
        start = 0
        
        for split in split_points:
            # Extract segment text
            segment_text = text[start:split.position].strip()
            
            if segment_text:
                # Find dependencies on other segments
                dependencies = self._find_dependencies(start, split.position, split_points, doc)
                
                segments.append(ClauseSegment(
                    text=segment_text,
                    start_offset=start,
                    end_offset=split.position,
                    dependencies=dependencies
                ))
            
            # Update start position
            start = split.position
            # Skip the conjunction/punctuation if needed
            while start < len(text) and text[start] in ' ,;:':
                start += 1
        
        # Add final segment
        if start < len(text):
            final_text = text[start:].strip()
            if final_text:
                dependencies = self._find_dependencies(start, len(text), split_points, doc)
                segments.append(ClauseSegment(
                    text=final_text,
                    start_offset=start,
                    end_offset=len(text),
                    dependencies=dependencies
                ))
        
        return segments
    
    def _find_dependencies(self, start: int, end: int, split_points: List[SplitPoint], doc: Doc) -> List[int]:
        """
        Find dependencies between segments.
        
        Args:
            start: Start position of current segment
            end: End position of current segment
            split_points: All split points
            doc: spaCy Doc object
            
        Returns:
            List of segment indices this segment depends on
        """
        dependencies = []
        
        # For now, simple heuristic: segments with pronouns depend on previous segments
        segment_text = doc.text[start:end]
        
        # Check for pronouns or references
        pronouns = ['it', 'its', 'they', 'their', 'them', 'this', 'that', 'these', 'those']
        for pronoun in pronouns:
            if pronoun in segment_text.lower().split():
                # Depends on previous segment
                current_idx = len([sp for sp in split_points if sp.position <= start])
                if current_idx > 0:
                    dependencies.append(current_idx - 1)
                break
        
        return dependencies
    
    def _simple_split(self, text: str) -> List[ClauseSegment]:
        """
        Simple fallback splitting without spaCy.
        
        Args:
            text: Input text
            
        Returns:
            List of ClauseSegment objects
        """
        segments = []
        
        # First try to split on sentence boundaries
        sentences = self.multi_question_pattern.split(text)
        
        for sentence in sentences:
            if not sentence.strip():
                continue
            
            # Then split on strong conjunctions with commas
            parts = self.comma_and_pattern.split(sentence)
            
            for part in parts:
                if not part.strip():
                    continue
                
                # Finally split on semicolons
                subparts = self.semicolon_pattern.split(part)
                
                for subpart in subparts:
                    if subpart.strip():
                        # Simple approach: just track offsets
                        start_offset = text.find(subpart)
                        end_offset = start_offset + len(subpart)
                        
                        segments.append(ClauseSegment(
                            text=subpart.strip(),
                            start_offset=start_offset,
                            end_offset=end_offset,
                            dependencies=[]
                        ))
        
        # If no splits were made, return the whole text as one segment
        if not segments and text.strip():
            segments.append(ClauseSegment(
                text=text.strip(),
                start_offset=0,
                end_offset=len(text),
                dependencies=[]
            ))
        
        return segments
    
    def _post_process_segments(self, segments: List[ClauseSegment]) -> List[ClauseSegment]:
        """
        Post-process segments to clean up and merge if needed.
        
        Args:
            segments: List of ClauseSegment objects
            
        Returns:
            Post-processed segments
        """
        if not segments:
            return segments
        
        processed = []
        
        for segment in segments:
            # Skip very short segments (likely just conjunctions)
            if len(segment.text.split()) < 2:
                continue
            
            # Clean up text
            segment.text = segment.text.strip()
            
            # Remove leading conjunctions
            for conj in self.coordinating_conjunctions:
                if segment.text.lower().startswith(conj + ' '):
                    segment.text = segment.text[len(conj):].strip()
                    break
            
            if segment.text:
                processed.append(segment)
        
        # Update dependencies after filtering
        # (This is simplified - in production you'd track the mapping)
        
        return processed
    
    def get_splitting_confidence(self, text: str) -> float:
        """
        Get confidence score for whether the text contains multiple intents.
        
        Args:
            text: Input text
            
        Returns:
            Confidence score (0-1) that text contains multiple intents
        """
        # Quick heuristics
        indicators = 0
        
        # Check for multiple questions
        if len(self.multi_question_pattern.findall(text)) > 0:
            indicators += 2
        
        # Check for conjunctions with commas
        if self.comma_and_pattern.search(text):
            indicators += 1
        
        # Check for semicolons
        if ';' in text:
            indicators += 2
        
        # Check for multiple question words
        question_count = sum(1 for word in self.question_words if word in text.lower().split())
        if question_count > 1:
            indicators += 1
        
        # Check for multiple imperative verbs
        imperative_count = sum(1 for verb in self.imperative_verbs if verb in text.lower().split())
        if imperative_count > 1:
            indicators += 1
        
        # Convert to confidence score
        confidence = min(indicators * 0.2, 1.0)
        
        return confidence
