"""
Graph Builder for GraphRAG system.

Provides automatic schema inference, relationship strength calculation,
node clustering for similar entities, and graph quality metrics.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime
import math
from collections import defaultdict, Counter
import numpy as np
from neo4j import GraphDatabase

from scripts.entity_evaluator import EntityCandidate, EntityRelationship
from scripts.graph_schema import ResearchSchema

@dataclass
class GraphNode:
    """Represents a node in the knowledge graph"""
    id: str
    labels: Set[str]
    properties: Dict[str, Any]
    embeddings: Optional[List[float]] = None
    importance_score: float = 0.0
    created_at: datetime = field(default_factory=datetime.now)

@dataclass
class GraphEdge:
    """Represents an edge/relationship in the knowledge graph"""
    id: str
    start_node: str
    end_node: str
    type: str
    properties: Dict[str, Any] = field(default_factory=dict)
    strength: float = 1.0
    confidence: float = 0.8
    created_at: datetime = field(default_factory=datetime.now)

@dataclass
class ClusterGroup:
    """Represents a cluster of similar nodes"""
    cluster_id: str
    nodes: List[str]
    centroid: Optional[List[float]] = None
    topics: List[str] = field(default_factory=list)
    quality_score: float = 0.0
    description: str = ""

@dataclass
class GraphMetrics:
    """Comprehensive graph quality metrics"""
    node_count: int = 0
    edge_count: int = 0
    density: float = 0.0
    avg_degree: float = 0.0
    clustering_coefficient: float = 0.0
    diameter: Optional[int] = None
    avg_path_length: Optional[float] = None
    modularity: float = 0.0
    centralization: float = 0.0
    connectivity: float = 0.0
    entity_type_distribution: Dict[str, int] = field(default_factory=dict)
    relationship_type_distribution: Dict[str, int] = field(default_factory=dict)

class GraphBuilder:
    """Automatic graph construction and optimization system"""

    def __init__(self,
                 neo4j_uri: str = "bolt://localhost:7687",
                 neo4j_user: str = "neo4j",
                 neo4j_password: str = "password",
                 min_confidence: float = 0.6,
                 embedding_dimension: int = 384):

        self.neo4j_uri = neo4j_uri
        self.neo4j_user = neo4j_user
        self.neo4j_password = neo4j_password
        self.min_confidence = min_confidence
        self.embedding_dimension = embedding_dimension

        # Initialize Neo4j driver
        self._init_neo4j()

        # Schema inference data
        self.inferred_schema = ResearchSchema()
        self.node_types = {}
        self.relationship_types = {}

        # Graph caching
        self.node_cache = {}
        self.edge_cache = {}

    def _init_neo4j(self):
        """Initialize Neo4j connection"""
        try:
            self.driver = GraphDatabase.driver(
                self.neo4j_uri,
                auth=(self.neo4j_user, self.neo4j_password)
            )
            # Test connection
            with self.driver.session() as session:
                session.run("RETURN 1")
            print("✓ Neo4j connection initialized for graph builder")
        except Exception as e:
            print(f"✗ Neo4j connection failed: {e}")
            self.driver = None

    def infer_schema_from_data(self, entities: List[EntityCandidate],
                             relationships: List[EntityRelationship]) -> Dict[str, Any]:
        """Automatically infer graph schema from entity and relationship data"""

        # Extract node types
        node_types = {}
        for entity in entities:
            entity_type = entity.entity_type
            if entity_type not in node_types:
                node_types[entity_type] = {
                    'count': 0,
                    'properties': set(),
                    'examples': []
                }

            node_types[entity_type]['count'] += 1
            node_types[entity_type]['properties'].update(['name', 'description', 'confidence'])

            if len(node_types[entity_type]['examples']) < 3:
                node_types[entity_type]['examples'].append(entity.name)

        # Extract relationship types
        relationship_types = {}
        for rel in relationships:
            rel_type = rel.relationship_type
            if rel_type not in relationship_types:
                relationship_types[rel_type] = {
                    'count': 0,
                    'entity_pairs': set(),
                    'properties': {'confidence', 'description', 'created_at'}
                }

            relationship_types[rel_type]['count'] += 1
            entity_pair = (rel.entity1, rel.entity2)
            relationship_types[rel_type]['entity_pairs'].add(entity_pair)

        # Generate schema suggestions
        schema_suggestions = {
            'node_types': {},
            'relationship_types': {}
        }

        # Node type suggestions
        for node_type, info in node_types.items():
            # Suggest embedding property if it would be useful
            embedding_prop = f"{node_type.lower()}_embedding" if info['count'] > 5 else None

            schema_suggestions['node_types'][node_type] = {
                'properties': {
                    'name': {'type': 'string', 'required': True},
                    'description': {'type': 'string', 'required': False},
                    'confidence': {'type': 'float', 'required': False}
                },
                'embedding_property': embedding_prop,
                'estimated_nodes': info['count'],
                'example_entities': info['examples']
            }

        # Relationship type suggestions
        for rel_type, info in relationship_types.items():
            pairs = list(info['entity_pairs'])
            if len(pairs) >= 2:
                entity1_type = self._infer_entity_type(pairs[0][0], entities)
                entity2_type = self._infer_entity_type(pairs[1][1], entities)
            else:
                entity1_type = "Unknown"
                entity2_type = "Unknown"

            schema_suggestions['relationship_types'][rel_type] = {
                'valid_pairs': f"{entity1_type} -> {entity2_type}",
                'properties': {
                    'confidence': {'type': 'float', 'required': False},
                    'description': {'type': 'string', 'required': False},
                    'created_at': {'type': 'datetime', 'required': False}
                },
                'estimated_relationships': info['count'],
                'directionality': 'directed'  # Assume directed for now
            }

        self.node_types = node_types
        self.relationship_types = relationship_types

        return schema_suggestions

    def _infer_entity_type(self, entity_name: str, entities: List[EntityCandidate]) -> str:
        """Infer entity type from name (fallback when not explicitly known)"""
        for entity in entities:
            if entity.name == entity_name:
                return entity.entity_type

        # Fallback heuristics
        if any(term in entity_name.lower() for term in ['inc', 'corp', 'ltd', 'llc']):
            return 'Organization'
        elif any(term in entity_name.lower() for term in ['dr.', 'prof.', 'researcher']):
            return 'Person'
        elif any(term in entity_name.lower() for term in ['algorithm', 'method', 'framework']):
            return 'Technology'

        return 'Unknown'

    def calculate_relationship_strengths(self, entities: List[EntityCandidate],
                                       relationships: List[EntityRelationship],
                                       documents: List[Dict[str, Any]]) -> List[EntityRelationship]:
        """Calculate strength scores for relationships"""

        strengthened_relationships = []

        for rel in relationships:
            base_strength = 1.0

            # Factor 1: Co-occurrence frequency
            cooccurrence_factor = self._calculate_cooccurrence_strength(rel, entities, documents)

            # Factor 2: Semantic similarity (if available)
            semantic_factor = 1.0
            if rel.evidence:
                # Analyze evidence text for semantic closeness
                semantic_factor = self._calculate_semantic_strength(rel, entities)

            # Factor 3: Confidence and source diversity
            confidence_factor = rel.confidence
            source_diversity = len(set(rel.sources)) / max(1, len(rel.sources))

            # Factor 4: Entity importance
            entity1_importance = self._get_entity_importance(rel.entity1, entities)
            entity2_importance = self._get_entity_importance(rel.entity2, entities)
            importance_factor = (entity1_importance + entity2_importance) / 2.0

            # Combine factors
            total_strength = (
                cooccurrence_factor * 0.3 +
                semantic_factor * 0.2 +
                confidence_factor * 0.25 +
                importance_factor * 0.15 +
                source_diversity * 0.1
            )

            strengthened_rel = EntityRelationship(
                entity1=rel.entity1,
                entity2=rel.entity2,
                relationship_type=rel.relationship_type,
                confidence=rel.confidence,
                evidence=rel.evidence,
                sources=rel.sources,
                created_at=rel.created_at
            )

            # Add strength as property
            strengthened_rel.strength = min(1.0, max(0.0, total_strength))

            strengthened_relationships.append(strengthened_rel)

        return strengthened_relationships

    def _calculate_cooccurrence_strength(self, rel: EntityRelationship,
                                        entities: List[EntityCandidate],
                                        documents: List[Dict[str, Any]]) -> float:
        """Calculate relationship strength based on entity co-occurrence"""

        entity1_mentions = []
        entity2_mentions = []

        # Find documents where both entities appear
        for doc in documents:
            if rel.entity1.lower() in doc.get('content', '').lower():
                entity1_mentions.append(doc)
            if rel.entity2.lower() in doc.get('content', '').lower():
                entity2_mentions.append(doc)

        # Common documents
        common_docs = len(set(doc['filename'] for doc in entity1_mentions) &
                         set(doc['filename'] for doc in entity2_mentions))

        total_docs = len(documents)
        if total_docs == 0:
            return 0.5

        # Strength based on co-occurrence frequency
        cooccurrence_rate = common_docs / total_docs
        return min(1.0, cooccurrence_rate * 3.0)  # Scale up for visibility

    def _calculate_semantic_strength(self, rel: EntityRelationship,
                                    entities: List[EntityCandidate]) -> float:
        """Calculate semantic relationship strength from evidence"""

        # Simple semantic analysis based on common words in descriptions
        entity1_desc = self._get_entity_description(rel.entity1, entities)
        entity2_desc = self._get_entity_description(rel.entity2, entities)

        if not entity1_desc or not entity2_desc:
            return 0.5

        # Calculate word overlap
        words1 = set(entity1_desc.lower().split())
        words2 = set(entity2_desc.lower().split())

        if not words1 or not words2:
            return 0.5

        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))

        similarity = intersection / union if union > 0 else 0.0
        return min(1.0, similarity * 2.0)  # Scale up

    def _get_entity_description(self, entity_name: str, entities: List[EntityCandidate]) -> str:
        """Get entity description"""
        for entity in entities:
            if entity.name == entity_name:
                return entity.description
        return ""

    def _get_entity_importance(self, entity_name: str, entities: List[EntityCandidate]) -> float:
        """Get entity importance score"""
        for entity in entities:
            if entity.name == entity_name:
                return entity.confidence
        return 0.5

    def cluster_similar_entities(self, entities: List[EntityCandidate],
                               similarity_threshold: float = 0.7,
                               min_cluster_size: int = 3) -> List[ClusterGroup]:
        """Cluster entities based on name similarity and semantic meaning"""

        clusters = []

        if len(entities) < min_cluster_size:
            return clusters

        # Create similarity matrix
        similarity_matrix = self._build_similarity_matrix(entities)

        # Apply clustering algorithm (simple agglomerative approach)
        used_entities = set()

        for i, entity1 in enumerate(entities):
            if entity1.name in used_entities:
                continue

            cluster_entities = [entity1.name]
            used_entities.add(entity1.name)

            # Find similar entities
            for j, entity2 in enumerate(entities):
                if i != j and entity2.name not in used_entities:
                    similarity = similarity_matrix[i][j]
                    if similarity >= similarity_threshold:
                        cluster_entities.append(entity2.name)
                        used_entities.add(entity2.name)

            # Only create cluster if it meets minimum size
            if len(cluster_entities) >= min_cluster_size:
                cluster = self._create_cluster(cluster_entities, entities)
                clusters.append(cluster)

        return clusters

    def _build_similarity_matrix(self, entities: List[EntityCandidate]) -> List[List[float]]:
        """Build entity similarity matrix"""
        n = len(entities)
        matrix = [[0.0] * n for _ in range(n)]

        for i in range(n):
            for j in range(n):
                if i == j:
                    matrix[i][j] = 1.0
                elif j > i:  # Only compute upper triangle
                    similarity = self._calculate_entity_similarity(
                        entities[i], entities[j]
                    )
                    matrix[i][j] = similarity
                    matrix[j][i] = similarity

        return matrix

    def _calculate_entity_similarity(self, entity1: EntityCandidate,
                                   entity2: EntityCandidate) -> float:
        """Calculate similarity between two entities"""

        # Same type bonus
        type_similarity = 1.0 if entity1.entity_type == entity2.entity_type else 0.5

        # Name similarity
        from difflib import SequenceMatcher
        name_similarity = SequenceMatcher(None,
                                         entity1.name.lower(),
                                         entity2.name.lower()).ratio()

        # Description similarity (if available)
        desc_similarity = 0.5
        if entity1.description and entity2.description:
            desc_sim = SequenceMatcher(None,
                                      entity1.description.lower(),
                                      entity2.description.lower()).ratio()
            desc_similarity = desc_sim if desc_sim > 0.3 else 0.5

        # Context similarity
        context_similarity = self._calculate_context_similarity(entity1, entity2)

        # Weighted combination
        total_similarity = (
            name_similarity * 0.4 +
            type_similarity * 0.2 +
            desc_similarity * 0.2 +
            context_similarity * 0.2
        )

        return min(1.0, total_similarity)

    def _calculate_context_similarity(self, entity1: EntityCandidate,
                                    entity2: EntityCandidate) -> float:
        """Calculate similarity based on shared contexts"""

        contexts1 = set(ctx.get('context', '').lower()[:100] for ctx in entity1.mentions)
        contexts2 = set(ctx.get('context', '').lower()[:100] for ctx in entity2.mentions)

        if not contexts1 or not contexts2:
            return 0.5

        # Simple word overlap
        words1 = set()
        for ctx in contexts1:
            words1.update(ctx.split())

        words2 = set()
        for ctx in contexts2:
            words2.update(ctx.split())

        if not words1 or not words2:
            return 0.5

        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))

        return intersection / union if union > 0 else 0.0

    def _create_cluster(self, entity_names: List[str],
                       entities: List[EntityCandidate]) -> ClusterGroup:
        """Create a cluster from a group of entity names"""

        # Find entity objects
        cluster_entities = [e for e in entities if e.name in entity_names]

        # Determine dominant type
        type_counts = Counter(e.entity_type for e in cluster_entities)
        dominant_type = type_counts.most_common(1)[0][0]

        # Create topics based on common themes
        all_topics = []
        for entity in cluster_entities:
            all_topics.extend(getattr(entity, 'related_topics', []))

        topic_counts = Counter(all_topics)
        top_topics = [topic for topic, _ in topic_counts.most_common(3)]

        # Generate description
        description = f"Cluster of {len(cluster_entities)} {dominant_type} entities "
        if top_topics:
            description += f"related to: {', '.join(top_topics)}"

        # Calculate quality score
        quality_score = (
            len(cluster_entities) / 10.0 +  # Size factor
            (len(top_topics) / 5.0) +       # Topic coherence
            (type_counts[dominant_type] / len(cluster_entities))  # Type homogeneity
        ) / 3.0

        return ClusterGroup(
            cluster_id=f"cluster_{len(entity_names)}_{dominant_type}_{hash(str(entity_names)) % 1000}",
            nodes=entity_names,
            topics=top_topics,
            quality_score=min(1.0, quality_score),
            description=description
        )

    def calculate_graph_quality_metrics(self) -> GraphMetrics:
        """Calculate comprehensive graph quality metrics"""

        if not self.driver:
            return GraphMetrics()

        try:
            with self.driver.session() as session:
                # Basic counts
                node_result = session.run("MATCH (n) RETURN count(n) as count")
                node_count = node_result.single()['count']

                edge_result = session.run("MATCH ()-[r]->() RETURN count(r) as count")
                edge_count = edge_result.single()['count']

                # Entity type distribution
                type_result = session.run("""
                    MATCH (n)
                    WHERE n:Entity OR n:Conversation
                    RETURN labels(n) as labels, count(*) as count
                """)

                entity_type_dist = {}
                for record in type_result:
                    labels = record['labels']
                    count = record['count']
                    # Use first non-generic label
                    for label in labels:
                        if label not in ['Entity', 'Conversation', 'Node']:
                            entity_type_dist[label] = count
                            break

                # Relationship type distribution
                rel_type_result = session.run("""
                    MATCH ()-[r]->()
                    RETURN type(r) as type, count(*) as count
                """)

                rel_type_dist = {record['type']: record['count'] for record in rel_type_result}

                # Graph density
                density = (2.0 * edge_count) / (node_count * (node_count - 1)) if node_count > 1 else 0.0

                # Average degree
                avg_degree = (2.0 * edge_count) / node_count if node_count > 0 else 0.0

                # Basic connectivity check
                connected_result = session.run("""
                    MATCH (n)
                    WHERE NOT (n)--()
                    RETURN count(n) as isolated_count
                """)
                isolated_count = connected_result.single()['isolated_count']
                connectivity = 1.0 - (isolated_count / node_count) if node_count > 0 else 0.0

                return GraphMetrics(
                    node_count=node_count,
                    edge_count=edge_count,
                    density=density,
                    avg_degree=avg_degree,
                    connectivity=connectivity,
                    entity_type_distribution=entity_type_dist,
                    relationship_type_distribution=rel_type_dist
                )

        except Exception as e:
            print(f"Error calculating graph metrics: {e}")
            return GraphMetrics()

    def build_graph_from_entities(self, entities: List[EntityCandidate],
                                relationships: List[EntityRelationship],
                                documents: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Comprehensive graph building pipeline"""

        if not self.driver:
            return {"error": "Neo4j driver not initialized"}

        try:
            # Step 1: Schema inference
            schema = self.infer_schema_from_data(entities, relationships)

            # Step 2: Relationship strength calculation
            strengthened_relationships = self.calculate_relationship_strengths(
                entities, relationships, documents or []
            )

            # Step 3: Entity clustering
            clusters = self.cluster_similar_entities(entities)

            # Step 4: Build graph in Neo4j
            build_result = self._construct_neo4j_graph(
                entities, strengthened_relationships, clusters
            )

            # Step 5: Calculate quality metrics
            metrics = self.calculate_graph_quality_metrics()

            return {
                "success": True,
                "schema_inferred": schema,
                "entities_processed": len(entities),
                "relationships_processed": len(strengthened_relationships),
                "clusters_created": len(clusters),
                "build_result": build_result,
                "graph_metrics": {
                    "node_count": metrics.node_count,
                    "edge_count": metrics.edge_count,
                    "density": round(metrics.density, 4),
                    "avg_degree": round(metrics.avg_degree, 2),
                    "connectivity": round(metrics.connectivity, 3),
                    "entity_types": metrics.entity_type_distribution,
                    "relationship_types": metrics.relationship_type_distribution
                }
            }

        except Exception as e:
            print(f"Graph building failed: {e}")
            return {"success": False, "error": str(e)}

    def _construct_neo4j_graph(self, entities: List[EntityCandidate],
                             relationships: List[EntityRelationship],
                             clusters: List[ClusterGroup]) -> Dict[str, Any]:
        """Construct the graph in Neo4j"""

        nodes_created = 0
        edges_created = 0

        try:
            with self.driver.session() as session:
                # Create entity nodes
                for entity in entities:
                    result = session.run("""
                        MERGE (e:Entity {name: $name})
                        SET e.type = $entity_type,
                            e.description = $description,
                            e.confidence = $confidence,
                            e.created_at = datetime(),
                            e.importance_score = $importance
                        RETURN e
                    """, name=entity.name, entity_type=entity.entity_type,
                         description=entity.description, confidence=entity.confidence,
                         importance=entity.confidence)

                    nodes_created += 1

                # Create relationships
                for rel in relationships:
                    # Ensure entities exist first
                    session.run("""
                        MERGE (e1:Entity {name: $entity1})
                        MERGE (e2:Entity {name: $entity2})
                    """, entity1=rel.entity1, entity2=rel.entity2)

                    # Create relationship
                    result = session.run(f"""
                        MATCH (e1:Entity {{name: $entity1}})
                        MATCH (e2:Entity {{name: $entity2}})
                        MERGE (e1)-[r:`{rel.relationship_type}`]->(e2)
                        SET r.confidence = $confidence,
                            r.description = $description,
                            r.strength = $strength,
                            r.created_at = datetime()
                        RETURN r
                    """, entity1=rel.entity1, entity2=rel.entity2,
                         confidence=rel.confidence, description=getattr(rel, 'description', ''),
                         strength=getattr(rel, 'strength', rel.confidence))

                    edges_created += 1

                # Create cluster nodes (optional)
                for cluster in clusters:
                    if cluster.quality_score >= 0.6:  # Only high-quality clusters
                        session.run("""
                            MERGE (c:Cluster {id: $cluster_id})
                            SET c.description = $description,
                                c.quality_score = $quality_score,
                                c.node_count = $node_count,
                                c.topics = $topics,
                                c.created_at = datetime()
                            RETURN c
                        """, cluster_id=cluster.cluster_id, description=cluster.description,
                             quality_score=cluster.quality_score, node_count=len(cluster.nodes),
                             topics=cluster.topics)

                        # Link cluster to entities
                        for entity_name in cluster.nodes:
                            session.run("""
                                MATCH (c:Cluster {id: $cluster_id})
                                MATCH (e:Entity {name: $entity_name})
                                MERGE (e)-[:BELONGS_TO]->(c)
                            """, cluster_id=cluster.cluster_id, entity_name=entity_name)

        except Exception as e:
            print(f"Neo4j graph construction failed: {e}")
            return {"error": str(e)}

        return {
            "nodes_created": nodes_created,
            "edges_created": edges_created,
            "clusters_added": len([c for c in clusters if c.quality_score >= 0.6])
        }

    def optimize_graph_performance(self) -> Dict[str, Any]:
        """Apply performance optimizations to the graph"""

        optimizations = {}

        try:
            with self.driver.session() as session:
                # Create indexes for common queries
                session.run("CREATE INDEX entity_name IF NOT EXISTS FOR (e:Entity) ON (e.name)")
                session.run("CREATE INDEX entity_type IF NOT EXISTS FOR (e:Entity) ON (e.type)")
                session.run("CREATE INDEX conversation_filename IF NOT EXISTS FOR (c:Conversation) ON (c.filename)")

                optimizations["indexes_created"] = True

                # Statistics
                stats_result = session.run("""
                    CALL db.resample.index.all()
                    YIELD name, entityType, status, populationProgress, scanProgress
                    RETURN count(*) as indexes_resampled
                """)

                optimizations["indexes_resampled"] = stats_result.single()['indexes_resampled']

                # Memory optimization
                session.run("CALL db.checkpoint()")

                optimizations["checkpoint_completed"] = True

        except Exception as e:
            optimizations["error"] = str(e)

        return optimizations

# Global instance
graph_builder = GraphBuilder()
