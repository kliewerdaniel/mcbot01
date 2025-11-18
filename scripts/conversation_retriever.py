import ollama
from neo4j import GraphDatabase
from typing import List, Dict, Any
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

class ConversationRetriever:
    def __init__(self,
                 neo4j_uri=None,
                 neo4j_user=None,
                 neo4j_password=None,
                 ollama_model="granite4:micro-h",
                 embedding_model="mxbai-embed-large:latest"):

        # Use environment variables if not provided
        self.neo4j_uri = neo4j_uri or os.getenv("NEO4J_URI")
        self.neo4j_user = neo4j_user or os.getenv("NEO4J_USER")
        self.neo4j_password = neo4j_password or os.getenv("NEO4J_PASSWORD")

        self.driver = GraphDatabase.driver(
            self.neo4j_uri,
            auth=(self.neo4j_user, self.neo4j_password)
        )
        self.ollama_model = ollama_model
        self.embedding_model = embedding_model

    def retrieve_context(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Hybrid retrieval for conversation documents combining:
        1. Vector similarity search on content
        2. Topic-based graph traversal
        3. Entity relationships
        """

        # Generate query embedding
        try:
            query_embedding = ollama.embeddings(
                model=self.embedding_model,
                prompt=query
            )['embedding']
        except Exception as e:
            print(f"Error generating query embedding: {e}")
            return []

        with self.driver.session() as session:
            # Vector similarity search on conversation documents
            vector_results = session.run("""
                CALL db.index.vector.queryNodes(
                    'conversation_embeddings',
                    $limit,
                    $query_embedding
                )
                YIELD node, score
                OPTIONAL MATCH (node)-[:DISCUSSES]->(topic:Topic)
                OPTIONAL MATCH (node)-[:MENTIONS]->(entity:Entity)

                RETURN
                    node.conversation_id AS conversation_id,
                    node.title AS title,
                    node.raw_content AS content,
                    node.document_type AS document_type,
                    node.summary AS summary,
                    score AS relevance_score,
                    collect(DISTINCT topic.name) AS topics,
                    collect(DISTINCT entity.name) AS entities,
                    'vector_search' AS retrieval_method
                ORDER BY score DESC
                """,
                query_embedding=query_embedding,
                limit=limit
            ).data()
            print(f"Vector search returned {len(vector_results)} results")
            if vector_results:
                print(f"First result: conversation_id={vector_results[0].get('conversation_id')}, title='{vector_results[0].get('title', 'N/A')}', score={vector_results[0].get('relevance_score', 'N/A')}")

            # Topic-based expansion
            topic_expansion_results = []
            if vector_results and vector_results[0].get('topics'):
                try:
                    top_topics = vector_results[0]['topics'][:3]  # Top 3 topics

                    topic_results = session.run("""
                        UNWIND $topics AS topic_name
                        MATCH (t:Topic {name: topic_name})
                        MATCH (d:Conversation)-[:DISCUSSES]->(t)
                        WHERE d.conversation_id <> $exclude_conversation_id

                        RETURN DISTINCT
                            d.conversation_id AS conversation_id,
                            d.title AS title,
                            d.raw_content AS content,
                            d.document_type AS document_type,
                            d.summary AS summary,
                            count(t) AS topic_matches,
                            collect(DISTINCT topic_name) AS topics,
                            'topic_expansion' AS retrieval_method
                        ORDER BY topic_matches DESC
                        LIMIT $limit
                        """,
                        topics=top_topics,
                        exclude_conversation_id=vector_results[0]['conversation_id'],
                        limit=limit // 2
                    ).data()

                    topic_expansion_results = topic_results

                except Exception as e:
                    print(f"Topic expansion failed: {e}")
                    topic_expansion_results = []

            # Entity-based expansion
            entity_expansion_results = []
            if vector_results and vector_results[0].get('entities'):
                try:
                    top_entities = vector_results[0]['entities'][:3]  # Top 3 entities

                    entity_results = session.run("""
                        UNWIND $entities AS entity_name
                        MATCH (e:Entity {name: entity_name})
                        MATCH (d:Conversation)-[:MENTIONS]->(e)
                        WHERE d.conversation_id <> $exclude_conversation_id

                        RETURN DISTINCT
                            d.conversation_id AS conversation_id,
                            d.title AS title,
                            d.raw_content AS content,
                            d.document_type AS document_type,
                            d.summary AS summary,
                            count(e) AS entity_matches,
                            collect(DISTINCT entity_name) AS entities,
                            'entity_expansion' AS retrieval_method
                        ORDER BY entity_matches DESC
                        LIMIT $limit
                        """,
                        entities=top_entities,
                        exclude_conversation_id=vector_results[0]['conversation_id'],
                        limit=limit // 3
                    ).data()

                    entity_expansion_results = entity_results

                except Exception as e:
                    print(f"Entity expansion failed: {e}")
                    entity_expansion_results = []

            # Combine and deduplicate results
            all_results = vector_results + topic_expansion_results + entity_expansion_results
            seen_ids = set()
            unique_results = []

            for result in all_results:
                # Use conversation_id or filename for deduplication
                unique_id = result.get('conversation_id') or result.get('filename')
                if unique_id and unique_id not in seen_ids:
                    seen_ids.add(unique_id)
                    # Use conversation_id as filename for consistency in frontend
                    result['filename'] = result.get('conversation_id', result.get('filename', 'Unknown'))
                    # Normalize content for display (truncate if too long)
                    if 'content' in result and result['content']:
                        result['content_preview'] = result['content'][:800] + '...' if len(result['content']) > 800 else result['content']
                    unique_results.append(result)

            # Sort by relevance score, prioritizing vector search results
            def sort_key(result):
                method_rank = {'vector_search': 1.0, 'topic_expansion': 0.8, 'entity_expansion': 0.7}
                base_score = result.get('relevance_score', result.get('topic_matches', result.get('entity_matches', 0)) * 0.1)
                method_bonus = method_rank.get(result.get('retrieval_method', 'unknown'), 0.5)
                return base_score * method_bonus

            unique_results.sort(key=sort_key, reverse=True)

            return unique_results[:limit]

    def _extract_query_concepts(self, query: str) -> List[str]:
        """Extract key concepts from query using LLM"""
        try:
            response = ollama.generate(
                model=self.ollama_model,
                prompt=f"Extract 3-5 key concepts from this document search query: {query}. Return as comma-separated list.",
                options={'temperature': 0.1}
            )
            return [c.strip() for c in response['response'].split(',') if c.strip()]
        except Exception as e:
            print(f"Error extracting query concepts: {e}")
            # Fallback: simple keyword extraction
            return query.split()[:3]

    def search_by_topic(self, topics: List[str], limit: int = 10) -> List[Dict[str, Any]]:
        """Search EPS documents by specific topics"""
        with self.driver.session() as session:
            results = session.run("""
                UNWIND $topics AS topic_name
                MATCH (t:Topic {name: topic_name})
                MATCH (d:EPSDocument)-[:DISCUSSES]->(t)

                RETURN DISTINCT
                    d.filename AS filename,
                    d.raw_content AS content,
                    d.document_type AS document_type,
                    d.summary AS summary,
                    count(t) AS topic_matches,
                    collect(DISTINCT topic_name) AS topics,
                    'topic_search' AS retrieval_method
                ORDER BY topic_matches DESC
                LIMIT $limit
                """,
                topics=topics,
                limit=limit
            ).data()

            return results

    def search_by_entity(self, entities: List[str], limit: int = 10) -> List[Dict[str, Any]]:
        """Search EPS documents by specific entities"""
        with self.driver.session() as session:
            results = session.run("""
                UNWIND $entities AS entity_name
                MATCH (e:Entity {name: entity_name})
                MATCH (d:EPSDocument)-[:MENTIONS]->(e)

                RETURN DISTINCT
                    d.filename AS filename,
                    d.raw_content AS content,
                    d.document_type AS document_type,
                    d.summary AS summary,
                    count(e) AS entity_matches,
                    collect(DISTINCT entity_name) AS entities,
                    'entity_search' AS retrieval_method
                ORDER BY entity_matches DESC
                LIMIT $limit
                """,
                entities=entities,
                limit=limit
            ).data()

            return results

    def search_by_document_type(self, doc_type: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search EPS documents by document type"""
        with self.driver.session() as session:
            results = session.run("""
                MATCH (d:EPSDocument)
                WHERE d.document_type = $doc_type

                RETURN
                    d.filename AS filename,
                    d.raw_content AS content,
                    d.document_type AS document_type,
                    d.summary AS summary,
                    'document_type_search' AS retrieval_method
                ORDER BY d.filename
                LIMIT $limit
                """,
                doc_type=doc_type,
                limit=limit
            ).data()

            return results

    def find_similar_documents(self, filename: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Find EPS documents similar to the given document using shared topics and entities"""
        with self.driver.session() as session:
            results = session.run("""
                MATCH (seed:EPSDocument {filename: $filename})
                OPTIONAL MATCH (seed)-[:DISCUSSES]->(topic:Topic)<-[:DISCUSSES]-(similar:EPSDocument)
                OPTIONAL MATCH (seed)-[:MENTIONS]->(entity:Entity)<-[:MENTIONS]-(similar:EPSDocument)
                WITH seed, similar, topic, entity
                WHERE similar.filename <> seed.filename AND similar IS NOT NULL

                RETURN DISTINCT
                    similar.filename AS filename,
                    similar.raw_content AS content,
                    similar.document_type AS document_type,
                    similar.summary AS summary,
                    count(DISTINCT topic) AS shared_topics,
                    count(DISTINCT entity) AS shared_entities,
                    collect(DISTINCT topic.name) AS topics,
                    collect(DISTINCT entity.name) AS entities,
                    'similarity_search' AS retrieval_method
                ORDER BY (shared_topics + shared_entities) DESC
                LIMIT $limit
                """,
                filename=filename,
                limit=limit
            ).data()

            return results

    def find_related_documents(self, filename: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Find EPS documents related through topic relationships and shared entities"""
        with self.driver.session() as session:
            results = session.run("""
                MATCH (seed:EPSDocument {filename: $filename})
                OPTIONAL MATCH (seed)-[:TOPIC_RELATED]-(related:EPSDocument)
                OPTIONAL MATCH (seed)-[:SHARES_ENTITIES]-(entity_related:EPSDocument)
                OPTIONAL MATCH (seed)-[:SIMILAR_TO]-(similar_related:EPSDocument)

                WITH seed, related, entity_related, similar_related
                WHERE related IS NOT NULL OR entity_related IS NOT NULL OR similar_related IS NOT NULL

                UNWIND [
                    {doc: related, rel_type: 'topic_related'},
                    {doc: entity_related, rel_type: 'entity_related'},
                    {doc: similar_related, rel_type: 'similar_to'}
                ] AS rel_data

                WITH seed, rel_data
                WHERE rel_data.doc IS NOT NULL AND rel_data.doc.filename <> seed.filename

                RETURN DISTINCT
                    rel_data.doc.filename AS filename,
                    rel_data.doc.raw_content AS content,
                    rel_data.doc.document_type AS document_type,
                    rel_data.doc.summary AS summary,
                    rel_data.rel_type AS relationship_type,
                    'related_search' AS retrieval_method
                ORDER BY rel_data.rel_type
                LIMIT $limit
                """,
                filename=filename,
                limit=limit
            ).data()

            return results

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test EPS hybrid retrieval")
    parser.add_argument("--query", type=str, help="Query to test")
    parser.add_argument("--limit", type=int, default=5, help="Number of results")
    parser.add_argument("--topic", type=str, help="Search by topic")
    parser.add_argument("--entity", type=str, help="Search by entity")
    parser.add_argument("--doc-type", type=str, help="Search by document type")

    args = parser.parse_args()

    retriever = EPSRetriever()

    if args.query:
        results = retriever.retrieve_context(args.query, args.limit)

        print(f"Query: {args.query}")
        print(f"Results: {len(results)}")
        for i, result in enumerate(results):
            print(f"{i+1}. {result['filename']} ({result.get('document_type', 'unknown')})")
            print(f"   Score: {result.get('relevance_score', 0):.3f}")
            if 'content_preview' in result:
                print(f"   Content: {result['content_preview'][:150]}...")
            elif result.get('summary'):
                print(f"   Summary: {result['summary'][:150]}...")
            print(f"   Method: {result.get('retrieval_method', 'unknown')}")
            if result.get('topics'):
                print(f"   Topics: {', '.join(result.get('topics', [])[:3])}")
            print()

    elif args.topic:
        results = retriever.search_by_topic([args.topic], args.limit)
        print(f"Topic: {args.topic}")
        print(f"Results: {len(results)}")
        for i, result in enumerate(results):
            print(f"{i+1}. {result['filename']} ({result.get('document_type', 'unknown')})")
            if result.get('summary'):
                print(f"   Summary: {result['summary'][:100]}...")

    elif args.entity:
        results = retriever.search_by_entity([args.entity], args.limit)
        print(f"Entity: {args.entity}")
        print(f"Results: {len(results)}")
        for i, result in enumerate(results):
            print(f"{i+1}. {result['filename']} ({result.get('document_type', 'unknown')})")
            if result.get('summary'):
                print(f"   Summary: {result['summary'][:100]}...")

    elif args.doc_type:
        results = retriever.search_by_document_type(args.doc_type, args.limit)
        print(f"Document Type: {args.doc_type}")
        print(f"Results: {len(results)}")
        for i, result in enumerate(results):
            print(f"{i+1}. {result['filename']}")
            if result.get('summary'):
                print(f"   Summary: {result['summary'][:100]}...")

    else:
        print("Provide a query with --query, --topic, --entity, or --doc-type")
