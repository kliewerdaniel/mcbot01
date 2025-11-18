"""
Entity Evaluator for GraphRAG system.

Provides confidence scoring, deduplication, type classification, and relationship
inference for extracted entities from documents.
"""

import json
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime
import re
import difflib
from collections import defaultdict, Counter
import numpy as np
from functools import lru_cache

from scripts.conversation_retriever import ConversationRetriever

@dataclass
class EntityCandidate:
    """Represents a potential entity extracted from text"""
    name: str
    entity_type: str
    confidence: float = 0.0
    sources: List[str] = field(default_factory=list)
    contexts: List[str] = field(default_factory=list)
    mentions: List[Dict[str, Any]] = field(default_factory=list)
    description: str = ""
    related_entities: Set[str] = field(default_factory=set)
    last_updated: datetime = field(default_factory=datetime.now)

    def add_mention(self, context: str, position: int = 0, source: str = "", confidence: float = 1.0):
        """Add a mention of this entity"""
        self.mentions.append({
            "context": context,
            "position": position,
            "source": source,
            "confidence": confidence,
            "timestamp": datetime.now().isoformat()
        })
        self.sources.append(source)
        self.contexts.append(context)
        self.last_updated = datetime.now()

    def calculate_confidence(self) -> float:
        """Calculate overall confidence based on mentions and consistency"""
        if not self.mentions:
            return 0.0

        # Base confidence from mention count (more mentions = higher confidence)
        mention_count = len(self.mentions)
        frequency_score = min(1.0, mention_count / 5.0)  # Cap at 5 mentions for max score

        # Consistency score (how often same type is predicted)
        type_mentions = [m.get('confidence', 0.8) for m in self.mentions]
        avg_type_confidence = np.mean(type_mentions)

        # Context quality score
        context_scores = []
        for context in self.contexts:
            score = self._score_context_quality(context)
            context_scores.append(score)
        context_quality = np.mean(context_scores) if context_scores else 0.5

        # Weighted combination
        self.confidence = (
            0.4 * frequency_score +
            0.3 * avg_type_confidence +
            0.3 * context_quality
        )

        return self.confidence

    def _score_context_quality(self, context: str) -> float:
        """Score the quality of context where entity was mentioned"""
        context = context.lower()

        # Indicators of high-quality context
        quality_indicators = [
            ' is ', ' are ', ' was ', ' were ',      # Definitional language
            ' developed ', ' created ', ' invented ', # Action words
            ' company', ' corporation', ' inc',       # Organization indicators
            ' researcher', ' scientist', ' engineer', # Person indicators
            ' technology', ' framework', ' system',   # Tech indicators
        ]

        matches = sum(1 for indicator in quality_indicators if indicator in context)
        base_score = min(1.0, matches / 3.0)  # Max score with 3+ indicators

        # Length bonus (substantial context)
        length_score = min(0.3, len(context.split()) / 100)
        base_score = min(1.0, base_score + length_score)

        # Capitalization check (proper nouns often capitalized)
        if self.name[0].isupper() and self.entity_type in ['Person', 'Organization']:
            base_score += 0.1

        return min(1.0, base_score)

    def merge_with(self, other: 'EntityCandidate') -> 'EntityCandidate':
        """Merge another entity candidate into this one"""
        # Combine mentions
        self.mentions.extend(other.mentions)
        self.sources.extend(other.sources)
        self.contexts.extend(other.contexts)
        self.related_entities.update(other.related_entities)

        # Update description if empty or other has better one
        if not self.description or (other.description and len(other.description) > len(self.description)):
            self.description = other.description

        # Recalculate confidence
        self.calculate_confidence()
        self.last_updated = datetime.now()

        return self

@dataclass
class EntityRelationship:
    """Represents a relationship between entities"""
    entity1: str
    entity2: str
    relationship_type: str
    confidence: float = 0.0
    evidence: List[str] = field(default_factory=list)
    sources: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)

class EntityEvaluator:
    """Main entity evaluation and deduplication system"""

    def __init__(self,
                 min_confidence_threshold: float = 0.3,
                 fuzzy_match_threshold: float = 0.85,
                 max_entities_per_type: int = 1000):

        self.min_confidence_threshold = min_confidence_threshold
        self.fuzzy_match_threshold = fuzzy_match_threshold
        self.max_entities_per_type = max_entities_per_type

        # Entity storage
        self.entities: Dict[str, EntityCandidate] = {}
        self.entities_by_type: Dict[str, Dict[str, EntityCandidate]] = defaultdict(dict)
        self.entity_relationships: List[EntityRelationship] = []

        # Type classification patterns
        self._load_entity_patterns()

        # Retriever for context queries
        self.retriever = ConversationRetriever()

    def _load_entity_patterns(self):
        """Load patterns for entity type classification"""
        self.entity_patterns = {
            'Person': [
                r'\b(dr\.?|prof\.?|mr\.?|mrs\.?|ms\.?|sir)\s+[A-Z][a-z]+\b',  # Titles
                r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\s+(?:jr\.|sr\.|iii|ii)?\b',      # Full names
                r'\bauthor(?:ed|s)?\s+(?:by\s+)?([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',  # Author mentions
            ],
            'Organization': [
                r'\b[A-Z][a-z]*(?:\s+[A-Z][a-z]*)*\s+(?:Inc|Corp|LLC|Ltd|Co\.?|Company|Corporation|University|Institute|Labs?|Group|Systems?|Technologies?|Solutions?)\b',
                r'\b(?:Google|Microsoft|Apple|Amazon|Meta|Facebook|Twitter|Netflix|Uber|Airbnb|Stripe|Databricks|Anthropic|OpenAI|xAI)\b',
                r'\b[A-Z]{2,}(?:\s+[A-Z]{2,})*\b',  # Acronyms like NASA, IBM, etc.
            ],
            'Technology': [
                r'\b[A-Z][a-z]*(?:Engine|Framework|Library|Language|Protocol|System|Platform|Tool|API|SDK)\b',
                r'\b(?:Python|Java|JavaScript|TypeScript|Go|Rust|C\+\+|Swift|Kotlin|R|Scala|PHP|Ruby|Perl)\b',
                r'\b(?:TensorFlow|PyTorch|React|Angular|Vue|Django|Flask|FastAPI|Spring|Docker|Kubernetes|AWS|Azure|GCP)\b',
            ],
            'Concept': [
                r'\b[A-Z][a-z]*(?:Algorithm|Model|Network|Learning|Theory|Method|Approach|Technique|Paradigm|Architecture)\b',
                r'\b(?:Machine Learning|Deep Learning|Neural Network|Computer Vision|NLP|Reinforcement Learning|Transfer Learning)\b',
            ],
            'Location': [
                r'\b[A-Z][a-z]+,\s+[A-Z]{2}\b',      # City, State
                r'\b[A-Z][a-z]+\s+Avenue|Street|Road|Boulevard|Drive|Lane\b',
                r'\bSan Francisco|New York|London|Tokyo|Berlin|Paris|Singapore|Seattle|Austin|Palo Alto\b',
            ]
        }

    def extract_entities_from_text(self, text: str, source: str = "") -> List[EntityCandidate]:
        """Extract entities from text with confidence scoring"""
        candidates = []

        # Rule-based extraction with regex patterns
        for entity_type, patterns in self.entity_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    entity_name = match.group(0).strip()
                    if len(entity_name) >= 2:  # Minimum length check
                        candidate = EntityCandidate(
                            name=entity_name,
                            entity_type=entity_type,
                            confidence=0.6  # Base confidence for regex matches
                        )
                        candidate.add_mention(text, match.start(), source, 0.6)
                        candidates.append(candidate)

        # LLM-based extraction for better context understanding
        llm_entities = self._extract_entities_llm(text, source)
        candidates.extend(llm_entities)

        # Deduplicate and score
        deduplicated = self._deduplicate_entities(candidates)

        # Filter by confidence
        filtered = [
            entity for entity in deduplicated
            if entity.confidence >= self.min_confidence_threshold
        ]

        return filtered

    def _extract_entities_llm(self, text: str, source: str) -> List[EntityCandidate]:
        """Use LLM for more sophisticated entity extraction"""
        try:
            import ollama

            # Truncate text if too long
            truncated_text = text[:2000] + "..." if len(text) > 2000 else text

            prompt = f"""Extract named entities from the following text. Focus on:

Person: Individual people, researchers, authors
Organization: Companies, institutions, universities
Technology: Software tools, frameworks, programming languages, algorithms
Concept: Key technical concepts, theories, methods
Location: Geographic locations, facilities

Text:
{truncated_text}

Return ONLY a JSON array of entities:
[{{"name": "Entity Name", "type": "Person|Organization|Technology|Concept|Location", "confidence": 0.8, "description": "brief description"}}]"""

            response = ollama.generate(
                model="granite4:micro-h",
                prompt=prompt,
                format="json",
                options={"temperature": 0.1}
            )

            result = json.loads(response['response'])

            candidates = []
            for entity_data in result:
                candidate = EntityCandidate(
                    name=entity_data['name'],
                    entity_type=entity_data['type'],
                    confidence=entity_data.get('confidence', 0.7),
                    description=entity_data.get('description', '')
                )
                candidate.add_mention(text, 0, source, entity_data.get('confidence', 0.7))
                candidates.append(candidate)

            return candidates

        except Exception as e:
            print(f"LLM entity extraction failed: {e}")
            return []

    def _deduplicate_entities(self, candidates: List[EntityCandidate]) -> List[EntityCandidate]:
        """Deduplicate and merge similar entities"""
        # Group by normalized name
        groups = defaultdict(list)

        for candidate in candidates:
            normalized = self._normalize_entity_name(candidate.name)
            groups[normalized].append(candidate)

        deduplicated = []

        for normalized_name, group in groups.items():
            if len(group) == 1:
                # Single entity, just calculate confidence
                group[0].calculate_confidence()
                deduplicated.append(group[0])
            else:
                # Multiple candidates, merge them
                merged = group[0]
                for other in group[1:]:
                    # Check if similar enough to merge
                    if self._entities_similar(merged, other):
                        merged = merged.merge_with(other)
                    else:
                        # Treat as separate entity
                        other.calculate_confidence()
                        deduplicated.append(other)

                merged.calculate_confidence()
                deduplicated.append(merged)

        return deduplicated

    def _normalize_entity_name(self, name: str) -> str:
        """Normalize entity name for comparison"""
        # Remove common title prefixes
        name = re.sub(r'^(dr\.?|prof\.?|mr\.?|mrs\.?|ms\.?|sir)\s+', '', name.lower())

        # Remove punctuation and extra spaces
        name = re.sub(r'[^\w\s]', '', name)
        name = re.sub(r'\s+', ' ', name).strip()

        return name

    def _entities_similar(self, entity1: EntityCandidate, entity2: EntityCandidate,
                         threshold: float = None) -> bool:
        """Check if two entities are similar enough to merge"""
        if threshold is None:
            threshold = self.fuzzy_match_threshold

        # Must be same type
        if entity1.entity_type != entity2.entity_type:
            return False

        # Exact name match
        if entity1.name.lower() == entity2.name.lower():
            return True

        # Fuzzy string similarity
        similarity = difflib.SequenceMatcher(None, entity1.name.lower(), entity2.name.lower()).ratio()

        return similarity >= threshold

    def evaluate_entity_quality(self, entity: EntityCandidate) -> Dict[str, Any]:
        """Comprehensive quality evaluation of an entity"""
        quality_metrics = {
            'confidence_score': entity.confidence,
            'mention_count': len(entity.mentions),
            'source_diversity': len(set(entity.sources)),
            'context_quality': np.mean([entity._score_context_quality(ctx) for ctx in entity.contexts]),
            'type_consistency': self._calculate_type_consistency(entity),
            'relationship_density': len(entity.related_entities),
            'temporal_recency': self._calculate_temporal_score(entity),
            'overall_quality': 0.0
        }

        # Calculate overall quality score
        weights = {
            'confidence_score': 0.25,
            'mention_count': 0.15,
            'source_diversity': 0.15,
            'context_quality': 0.20,
            'type_consistency': 0.10,
            'relationship_density': 0.10,
            'temporal_recency': 0.05
        }

        quality_metrics['overall_quality'] = sum(
            quality_metrics[metric] * weight
            for metric, weight in weights.items()
        )

        return quality_metrics

    def _calculate_type_consistency(self, entity: EntityCandidate) -> float:
        """Calculate how consistent the entity type is"""
        if not entity.mentions:
            return 0.5

        # In a more sophisticated system, this would compare predicted types
        # For now, assume good consistency if entity has been processed
        return 0.8

    def _calculate_temporal_score(self, entity: EntityCandidate) -> float:
        """Calculate recency score"""
        if not entity.mentions:
            return 0.5

        # More recent mentions get higher scores
        hours_since_last = (datetime.now() - entity.last_updated).total_seconds() / 3600

        # Exponential decay: 1.0 for < 1 hour, 0.5 for < 24 hours, etc.
        recency_score = np.exp(-hours_since_last / 24.0)

        return recency_score

    def find_similar_entities(self, query_entity: str, entity_type: str = None, limit: int = 10) -> List[Tuple[EntityCandidate, float]]:
        """Find entities similar to the query"""
        query_normalized = self._normalize_entity_name(query_entity)

        candidates = []
        for entity_id, entity in self.entities.items():
            if entity_type and entity.entity_type != entity_type:
                continue

            entity_normalized = self._normalize_entity_name(entity.name)
            similarity = difflib.SequenceMatcher(None, query_normalized, entity_normalized).ratio()

            if similarity >= 0.6:  # Minimum similarity threshold
                candidates.append((entity, similarity))

        # Sort by similarity and return top matches
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[:limit]

    def infer_relationships(self, entities: List[EntityCandidate], context: str = "") -> List[EntityRelationship]:
        """Infer relationships between entities"""
        relationships = []

        if len(entities) < 2:
            return relationships

        # Pairwise relationship inference
        for i, entity1 in enumerate(entities):
            for entity2 in entities[i+1:]:
                relationship = self._infer_entity_relationship(entity1, entity2, context)
                if relationship:
                    relationships.append(relationship)

        return relationships

    def _infer_entity_relationship(self, entity1: EntityCandidate, entity2: EntityCandidate, context: str) -> Optional[EntityRelationship]:
        """Infer relationship between two specific entities"""
        # Find common contexts where both entities are mentioned
        common_contexts = []
        for ctx1 in entity1.contexts:
            for ctx2 in entity2.contexts:
                # Simple overlap check
                if self._contexts_related(ctx1, ctx2):
                    common_contexts.append(ctx1)
                    break

        if not common_contexts:
            return None

        # Use LLM to infer relationship type
        relationship = self._infer_relationship_llm(entity1.name, entity2.name, context, common_contexts)

        if relationship:
            rel = EntityRelationship(
                entity1=entity1.name,
                entity2=entity2.name,
                relationship_type=relationship['type'],
                confidence=relationship['confidence'],
                evidence=common_contexts,
                sources=list(set(entity1.sources + entity2.sources))
            )
            return rel

        return None

    def _contexts_related(self, context1: str, context2: str, threshold: float = 0.3) -> bool:
        """Check if two contexts are related"""
        words1 = set(context1.lower().split())
        words2 = set(context2.lower().split())

        overlap = len(words1.intersection(words2))
        union = len(words1.union(words2))

        return (overlap / union) >= threshold if union > 0 else False

    def _infer_relationship_llm(self, entity1: str, entity2: str, context: str, evidence: List[str]) -> Optional[Dict[str, Any]]:
        """Use LLM to infer relationship type"""
        try:
            import ollama

            evidence_text = "\n".join([f"- {ctx[:200]}..." for ctx in evidence[:3]])
            context_preview = context[:500] + "..." if len(context) > 500 else context

            prompt = f"""What is the relationship between "{entity1}" and "{entity2}" based on this context?

Context:
{context_preview}

Evidence (where both are mentioned):
{evidence_text}

Possible relationships:
- RELATED_TO: General connection
- WORKS_FOR: Employment/affiliation
- USES: Technology usage
- CREATED: Development/creation
- LOCATED_IN: Geographic relationship
- PART_OF: Hierarchical relationship

Return JSON: {{"type": "RELATIONSHIP_TYPE", "confidence": 0.8, "explanation": "brief reason"}}"""

            response = ollama.generate(
                model="granite4:micro-h",
                prompt=prompt,
                format="json",
                options={"temperature": 0.1}
            )

            result = json.loads(response['response'])
            return {
                'type': result.get('type', 'RELATED_TO'),
                'confidence': result.get('confidence', 0.6),
                'explanation': result.get('explanation', '')
            }

        except Exception as e:
            print(f"Relationship inference failed: {e}")
            return None

    def add_entities_to_graph(self, entities: List[EntityCandidate], relationships: List[EntityRelationship] = None):
        """Add validated entities and relationships to the knowledge graph"""
        # Store entities
        for entity in entities:
            entity_id = f"{entity.entity_type}:{entity.name}"

            # Check limits per type
            type_entities = self.entities_by_type[entity.entity_type]
            if len(type_entities) >= self.max_entities_per_type:
                # Remove lowest confidence entity of this type
                lowest_conf = min(type_entities.values(), key=lambda x: x.confidence)
                del self.entities[lowest_conf.name]
                del type_entities[lowest_conf.name]

            # Add entity
            self.entities[entity.name] = entity
            type_entities[entity.name] = entity

        # Store relationships
        if relationships:
            self.entity_relationships.extend(relationships)

        print(f"Added {len(entities)} entities and {len(relationships or [])} relationships to knowledge graph")

    def get_entity_statistics(self) -> Dict[str, Any]:
        """Get statistics about stored entities"""
        total_entities = len(self.entities)
        type_counts = Counter(entity.entity_type for entity in self.entities.values())
        avg_confidence = np.mean([e.confidence for e in self.entities.values()]) if self.entities else 0.0

        return {
            'total_entities': total_entities,
            'entities_by_type': dict(type_counts),
            'average_confidence': avg_confidence,
            'total_relationships': len(self.entity_relationships),
            'relationships_by_type': dict(Counter(r.relationship_type for r in self.entity_relationships))
        }

# Global instance
entity_evaluator = EntityEvaluator()
