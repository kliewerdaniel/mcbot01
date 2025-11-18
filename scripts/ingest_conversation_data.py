"""
Enhanced Conversation Data Ingestion with Entity Evaluator.

Uses advanced entity extraction, batch processing, and graph builder integration
for intelligent document ingestion into the knowledge graph.
"""

import ollama
from neo4j import GraphDatabase
from pathlib import Path
import csv
import json
from dotenv import load_dotenv
import os
import hashlib
from typing import List, Dict, Any, Optional, Iterator
import asyncio
from concurrent.futures import ThreadPoolExecutor
import time
from dataclasses import dataclass

# Import enhanced components
from scripts.entity_evaluator import entity_evaluator, EntityCandidate, EntityRelationship
from scripts.graph_builder import graph_builder
from scripts.upload_manager import upload_manager

# Load environment variables
load_dotenv()

@dataclass
class IngestionStats:
    """Statistics for ingestion operations"""
    documents_processed: int = 0
    entities_extracted: int = 0
    relationships_created: int = 0
    topics_identified: int = 0
    processing_time_seconds: float = 0.0
    errors: List[str] = None

    def __post_init__(self):
        if self.errors is None:
            self.errors = []

    def add_error(self, error: str):
        """Add an error to the stats"""
        self.errors.append(error)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for reporting"""
        return {
            'documents_processed': self.documents_processed,
            'entities_extracted': self.entities_extracted,
            'relationships_created': self.relationships_created,
            'topics_identified': self.topics_identified,
            'processing_time_seconds': round(self.processing_time_seconds, 2),
            'error_count': len(self.errors),
            'errors': self.errors[:10]  # Limit error messages
        }

class ConversationGraphBuilder:
    """Enhanced conversation graph builder with entity evaluator integration"""

    def __init__(self,
                 neo4j_uri=None,
                 neo4j_user=None,
                 neo4j_password=None,
                 embedding_model="mxbai-embed-large:latest",
                 ollama_model="granite4:micro-h",
                 batch_size: int = 10,
                 use_enhanced_extraction: bool = True):

        # Enhanced connection parameters
        self.neo4j_uri = neo4j_uri or os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.neo4j_user = neo4j_user or os.getenv("NEO4J_USER", "neo4j")
        self.neo4j_password = neo4j_password or os.getenv("NEO4J_PASSWORD", "password")

        self.embedding_model = embedding_model
        self.ollama_model = ollama_model
        self.batch_size = batch_size
        self.use_enhanced_extraction = use_enhanced_extraction

        # Initialize connection with enhanced error handling
        self._initialize_connection()

    def _initialize_connection(self):
        """Initialize Neo4j connection with validation"""
        try:
            if not all([self.neo4j_uri, self.neo4j_user, self.neo4j_password]):
                raise ValueError("Missing Neo4j connection parameters")

            self.driver = GraphDatabase.driver(
                self.neo4j_uri,
                auth=(self.neo4j_user, self.neo4j_password)
            )

            # Test connection
            with self.driver.session() as session:
                session.run("RETURN 1")

            print("✓ Connected to Neo4j database (enhanced mode)")

        except Exception as e:
            print(f"✗ Neo4j connection failed: {e}")
            self.driver = None
            raise

    def extract_document_entities_enhanced(self, filename: str, content: str, doc_type: str = "conversation") -> Dict[str, Any]:
        """Enhanced entity extraction using entity evaluator"""
        try:
            # Use advanced entity extractor
            entities = entity_evaluator.extract_entities_from_text(content, filename)

            # Calculate quality scores
            quality_scores = {e.name: entity_evaluator.evaluate_entity_quality(e) for e in entities}

            # Extract topics hierarchically
            topics = self._extract_topics_enhanced(content, filename)

            # Generate content summary
            summary = self._generate_content_summary(content, filename)

            return {
                'topics': topics,
                'entities': [e.name for e in entities],  # Backward compatibility
                'entity_objects': entities,  # Enhanced objects
                'entity_scores': quality_scores,
                'document_type': doc_type,
                'summary': summary
            }

        except Exception as e:
            print(f"Enhanced extraction failed: {e}. Using basic extraction.")
            return self.extract_document_entities(filename, content)

    def _extract_topics_enhanced(self, content: str, filename: str) -> List[str]:
        """Extract topics using LLM"""
        try:
            prompt = f"""Extract 3-5 key topics from this document.

Document: {filename}
Content: {content[:1000]}...

Return JSON: {{"topics": ["topic1", "topic2", "topic3"]}}"""

            response = ollama.generate(
                model=self.ollama_model,
                prompt=prompt,
                format='json'
            )

            return json.loads(response['response']).get('topics', ['document'])

        except Exception:
            return ['document']

    def _generate_content_summary(self, content: str, filename: str) -> str:
        """Generate document summary"""
        try:
            prompt = f"""Summarize this document in 1-2 sentences.

Document: {filename}
Content: {content[:1500]}...

Summary:"""

            response = ollama.generate(model=self.ollama_model, prompt=prompt)
            return response['response'].strip()

        except Exception:
            return content[:200] + '...' if len(content) > 200 else content

    def create_conversation_node_enhanced(self, conversation_id: str, title: str, content: str, stats: IngestionStats = None):
        """Create enhanced conversation node with entity evaluation"""

        if not content or not content.strip():
            if stats:
                stats.add_error(f"Empty content for conversation {conversation_id}")
            return

        try:
            # Enhanced entity extraction
            entities = self.extract_document_entities_enhanced(title, content)

            # Generate embeddings and content hash
            content_hash = hashlib.md5(content.encode()).hexdigest()
            embedding = self.generate_document_embedding(content)

            # Create node in Neo4j
            with self.driver.session() as session:
                session.run("""
                    MERGE (d:Conversation {conversation_id: $id, content_hash: $hash})
                    SET d.title = $title,
                        d.document_type = $doc_type,
                        d.summary = $summary,
                        d.content_embedding = $embedding,
                        d.raw_content = $content,
                        d.entities_count = $entities_count,
                        d.topics_count = $topics_count,
                        d.processed_at = datetime(),
                        d.content_length = $length
                    """,
                    id=conversation_id,
                    hash=content_hash,
                    title=title,
                    doc_type=entities['document_type'],
                    summary=entities['summary'],
                    embedding=embedding,
                    content=content[:10000],
                    entities_count=len(entities.get('entity_objects', [])),
                    topics_count=len(entities.get('topics', [])),
                    length=len(content)
                )

                # Create relationships for entities and topics
                for entity_obj in entities.get('entity_objects', []):
                    session.run("""
                        MERGE (e:Entity {name: $name})
                        SET e.type = $type, e.confidence = $confidence,
                            e.description = $desc, e.mention_count = COALESCE(e.mention_count, 0) + $mentions
                        MERGE (d:Conversation {conversation_id: $conv_id})-[:MENTIONS]->(e)
                        """,
                        name=entity_obj.name,
                        type=entity_obj.entity_type,
                        confidence=entity_obj.confidence,
                        desc=getattr(entity_obj, 'description', ''),
                        mentions=len(getattr(entity_obj, 'mentions', [])),
                        conv_id=conversation_id
                    )

                for topic in entities.get('topics', []):
                    session.run("""
                        MERGE (t:Topic {name: $topic})
                        SET t.mentioned_in = COALESCE(t.mentioned_in, 0) + 1
                        MERGE (d:Conversation {conversation_id: $conv_id})-[:DISCUSSES]->(t)
                        """,
                        topic=topic,
                        conv_id=conversation_id
                    )

            # Update statistics
            if stats:
                stats.documents_processed += 1
                stats.entities_extracted += len(entities.get('entity_objects', []))
                stats.topics_identified += len(entities.get('topics', []))

            print(f"✓ Enhanced node: {conversation_id} "
                  f"({len(entities.get('entity_objects', []))} entities, "
                  f"{len(entities.get('topics', []))} topics)")

        except Exception as e:
            error_msg = f"Failed to create node {conversation_id}: {str(e)}"
            print(f"❌ {error_msg}")
            if stats:
                stats.add_error(error_msg)

    def ingest_conversations_batch(self, json_path: Path, stats: IngestionStats = None) -> IngestionStats:
        """Batch ingestion with enhanced processing"""

        if stats is None:
            stats = IngestionStats()

        start_time = time.time()

        if not json_path.exists():
            stats.add_error(f"File not found: {json_path}")
            return stats

        print(f"🚀 Enhanced batch ingestion from {json_path}")

        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                conversations = json.load(f)

            total = len(conversations)
            print(f"Processing {total} conversations in batches of {self.batch_size}")

            for i in range(0, total, self.batch_size):
                batch = conversations[i:i + self.batch_size]
                batch_stats = self._process_batch(batch)

                # Update overall stats
                stats.documents_processed += batch_stats.documents_processed
                stats.entities_extracted += batch_stats.entities_extracted
                stats.topics_identified += batch_stats.topics_identified
                stats.errors.extend(batch_stats.errors)

                # Progress reporting
                processed = stats.documents_processed
                print(f"📊 Batch {i//self.batch_size + 1}: {processed}/{total} "
                      f"({stats.entities_extracted} entities, {stats.topics_identified} topics)")

            # Final graph optimization
            if stats.documents_processed > 0:
                print("🔗 Building similarity relationships...")
                self.create_similarity_relationships()

                print("📊 Optimizing graph...")
                graph_builder.optimize_graph_performance()

            stats.processing_time_seconds = time.time() - start_time
            print(f"✅ Enhanced ingestion complete! {stats.to_dict()}")

        except Exception as e:
            stats.add_error(f"Batch ingestion failed: {str(e)}")
            print(f"❌ Ingestion error: {e}")

        return stats

    def _process_batch(self, conversations: List[Dict[str, Any]]) -> IngestionStats:
        """Process a batch of conversations"""
        batch_stats = IngestionStats()

        for conv in conversations:
            conv_id = conv.get('conversation_id', '')
            title = conv.get('title', 'Untitled')

            if not conv_id:
                batch_stats.add_error("Missing conversation ID")
                continue

            content = self.extract_conversation_content(conv)
            if not content.strip():
                batch_stats.add_error(f"Empty content: {conv_id}")
                continue

            self.create_conversation_node_enhanced(conv_id, title, content, batch_stats)

        return batch_stats

    def validate_ingestion_data(self, conversations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate conversation data"""
        results = {
            'total': len(conversations),
            'valid': 0,
            'invalid': 0,
            'errors': [],
            'warnings': []
        }

        seen_ids = set()

        for conv in conversations:
            conv_id = conv.get('conversation_id')

            if not conv_id:
                results['errors'].append("Missing conversation_id")
                results['invalid'] += 1
                continue

            if conv_id in seen_ids:
                results['warnings'].append(f"Duplicate ID: {conv_id}")
            else:
                seen_ids.add(conv_id)

            content = self.extract_conversation_content(conv)
            if content and len(content.strip()) >= 10:
                results['valid'] += 1
            else:
                results['warnings'].append(f"Insufficient content: {conv_id}")

        results['summary'] = f"{results['valid']}/{results['total']} valid conversations"
        return results

    # Keep original methods for backward compatibility
    def generate_document_embedding(self, text: str) -> List[float]:
        """Generate embeddings for document content"""
        if not text or not text.strip():
            return []

        limited_text = text.strip()[:1000]
        if len(limited_text) < 10:
            return []

        try:
            response = ollama.embeddings(model=self.embedding_model, prompt=limited_text)
            return response['embedding']
        except Exception as e:
            print(f"Embeddings failed: {e}")
            return []

    def extract_document_entities(self, filename: str, content: str) -> Dict[str, Any]:
        """Basic entity extraction (backward compatibility)"""
        prompt = f"""Extract entities from: {content[:1000]}...

Return JSON: {{"topics": ["topic1"], "entities": ["entity1"], "document_type": "text", "summary": "summary"}}"""

        try:
            response = ollama.generate(model=self.ollama_model, prompt=prompt, format='json')
            return json.loads(response['response'])
        except:
            return {'topics': ['document'], 'entities': [], 'document_type': 'unknown',
                   'summary': content[:200] + '...' if len(content) > 200 else content}

    def create_conversation_node(self, conversation_id: str, title: str, content: str):
        """Basic conversation node creation"""
        if not content:
            return

        entities = self.extract_document_entities(title, content)
        content_hash = hashlib.md5(content.encode()).hexdigest()
        embedding = self.generate_document_embedding(content)

        with self.driver.session() as session:
            session.run("""
                MERGE (d:Conversation {conversation_id: $id, content_hash: $hash})
                SET d.title = $title, d.summary = $summary, d.content_embedding = $embedding,
                    d.raw_content = $content
                """,
                id=conversation_id, hash=content_hash, title=title,
                summary=entities['summary'], embedding=embedding, content=content[:10000]
            )

            for topic in entities.get('topics', []):
                session.run("""
                    MERGE (t:Topic {name: $topic})
                    MERGE (d:Conversation {conversation_id: $id})-[:DISCUSSES]->(t)
                    """, topic=topic, id=conversation_id)

            print(f"✓ Created basic conversation node: {conversation_id}")

    def extract_conversation_content(self, conversation: Dict[str, Any]) -> str:
        """Extract content from conversation JSON"""
        if not conversation or not isinstance(conversation, dict):
            return ""

        content_parts = []
        mapping = conversation.get('mapping', {})

        if not isinstance(mapping, dict):
            return ""

        for msg_data in mapping.values():
            if not isinstance(msg_data, dict):
                continue

            message = msg_data.get('message', {})
            if not isinstance(message, dict):
                continue

            content = message.get('content')
            if not isinstance(content, dict):
                continue

            parts = content.get('parts', [])
            if not isinstance(parts, list):
                continue

            for part in parts:
                if isinstance(part, str) and part.strip():
                    content_parts.append(part.strip())

        return '\n'.join(content_parts)

    def ingest_conversations_json(self, json_path: Path):
        """Basic ingestion (backward compatibility)"""
        print(f"Basic ingestion from {json_path}")

        if not json_path.exists():
            print(f"File not found: {json_path}")
            return

        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                conversations = json.load(f)

            for conv in conversations:
                conv_id = conv.get('conversation_id', '')
                title = conv.get('title', 'Untitled')

                if not conv_id:
                    continue

                content = self.extract_conversation_content(conv)
                if not content.strip():
                    continue

                self.create_conversation_node(conv_id, title, content)

            print("✓ Basic ingestion complete")

        except Exception as e:
            print(f"Basic ingestion failed: {e}")

    def create_similarity_relationships(self):
        """Create similarity relationships"""
        print("Creating similarity relationships...")

        try:
            with self.driver.session() as session:
                result = session.run("""
                    MATCH (d1:Conversation), (d2:Conversation)
                    WHERE id(d1) > id(d2)
                    WITH d1, d2,
                         reduce(dot = 0.0, i IN range(0, size(d1.content_embedding)-1) |
                           dot + d1.content_embedding[i] * d2.content_embedding[i]) AS dot_product,
                         sqrt(reduce(sum_sq = 0.0, i IN range(0, size(d1.content_embedding)-1) |
                           sum_sq + d1.content_embedding[i] * d1.content_embedding[i])) AS mag1,
                         sqrt(reduce(sum_sq = 0.0, i IN range(0, size(d2.content_embedding)-1) |
                           sum_sq + d2.content_embedding[i] * d2.content_embedding[i])) AS mag2
                    WITH d1, d2, dot_product / (mag1 * mag2) AS similarity
                    WHERE similarity > 0.7
                    CREATE (d1)-[:SIMILAR_TO {similarity: similarity}]->(d2)
                    RETURN count(*) as count
                """)

                count = result.single()['count'] if result else 0
                print(f"Created {count} similarity relationships")

        except Exception as e:
            print(f"Similarity creation failed: {e}")

    def create_vector_indexes(self):
        """Create vector indexes"""
        with self.driver.session() as session:
            try:
                session.run("""
                    CREATE VECTOR INDEX conversation_embeddings IF NOT EXISTS
                    FOR (d:Conversation) ON d.content_embedding
                    OPTIONS {indexConfig: {`vector.dimensions`: 1024, `vector.similarity_function`: 'cosine'}}
                """)

                session.run("CREATE INDEX entity_name IF NOT EXISTS FOR (e:Entity) ON (e.name)")
                session.run("CREATE INDEX entity_type IF NOT EXISTS FOR (e:Entity) ON (e.type)")
                session.run("CREATE INDEX document_hash IF NOT EXISTS FOR (d:Conversation) ON (d.content_hash)")

                print("✓ Vector indexes created")

            except Exception as e:
                print(f"Index creation failed: {e}")


# Enhanced main function with batch processing
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Enhanced conversation data ingestion")
    parser.add_argument("--json", type=str, default="conversations.json",
                       help="Path to conversations JSON file")
    parser.add_argument("--enhanced", action="store_true", default=True,
                       help="Use enhanced entity extraction")
    parser.add_argument("--batch-size", type=int, default=10,
                       help="Batch size for processing")
    parser.add_argument("--validate-only", action="store_true",
                       help="Only validate data, don't ingest")

    args = parser.parse_args()

    # Create enhanced builder
    builder = ConversationGraphBuilder(
        batch_size=args.batch_size,
        use_enhanced_extraction=args.enhanced
    )

    json_path = Path(args.json)

    if args.validate_only:
        # Only validate
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                conversations = json.load(f)

            validation = builder.validate_ingestion_data(conversations)
            print("📊 Validation Results:")
            print(validation['summary'])

            for error in validation.get('errors', []):
                print(f"❌ {error}")

            for warning in validation.get('warnings', []):
                print(f"⚠️  {warning}")

        except Exception as e:
            print(f"Validation failed: {e}")

    else:
        # Run enhanced batch ingestion
        stats = builder.ingest_conversations_batch(json_path)
        print(f"\n📈 Final Statistics: {stats.to_dict()}")
