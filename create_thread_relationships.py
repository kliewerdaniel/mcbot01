#!/usr/bin/env python3
"""
Script to create document relationships (TOPIC_RELATED and SIMILAR_TO)
in the Neo4j database for EPS document data.
"""
from neo4j import GraphDatabase
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

def create_document_relationships():
    """Create TOPIC_RELATED and SIMILAR_TO relationships for EPS documents"""

    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user = os.getenv("NEO4J_USER", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "research2025")

    driver = GraphDatabase.driver(uri, auth=(user, password))

    try:
        with driver.session() as session:
            print("Creating TOPIC_RELATED relationships...")

            # Create relationships between documents that discuss the same topics
            result = session.run("""
                MATCH (d1:EPSDocument)-[:DISCUSSES]->(t:Topic)<-[:DISCUSSES]-(d2:EPSDocument)
                WHERE id(d1) > id(d2)  // Avoid duplicate relationships
                WITH d1, d2, count(t) AS shared_topics
                WHERE shared_topics > 0
                MERGE (d1)-[:TOPIC_RELATED {shared_topics: shared_topics}]->(d2)
                RETURN count(*) as topic_links_created
            """).single()

            topic_count = result['topic_links_created'] if result else 0
            print(f"Created {topic_count} TOPIC_RELATED relationships")

            print("Creating SIMILAR_TO relationships based on cosine similarity...")

            # Create similarity relationships using cosine similarity on embeddings
            result = session.run("""
                MATCH (d1:EPSDocument), (d2:EPSDocument)
                WHERE id(d1) > id(d2)  // Avoid duplicate pairs
                AND d1.content_embedding IS NOT NULL
                AND d2.content_embedding IS NOT NULL
                WITH d1, d2,
                     reduce(dot = 0.0, i IN range(0, size(d1.content_embedding)-1) |
                       dot + d1.content_embedding[i] * d2.content_embedding[i]
                     ) AS dot_product,
                     sqrt(reduce(sum_sq = 0.0, i IN range(0, size(d1.content_embedding)-1) |
                       sum_sq + d1.content_embedding[i] * d1.content_embedding[i]
                     )) AS mag1,
                     sqrt(reduce(sum_sq = 0.0, i IN range(0, size(d2.content_embedding)-1) |
                       sum_sq + d2.content_embedding[i] * d2.content_embedding[i]
                     )) AS mag2
                WITH d1, d2, dot_product / (mag1 * mag2) AS similarity
                WHERE similarity > 0.75  // Only create relationships for highly similar documents
                MERGE (d1)-[:SIMILAR_TO {similarity: similarity}]->(d2)
                RETURN count(*) as similarity_links_created
            """).single()

            similarity_count = result['similarity_links_created'] if result else 0
            print(f"Created {similarity_count} SIMILAR_TO relationships")

            print("Creating DOCUMENT_SEQUENCES for related documents...")

            # Group documents by shared entities and create sequences
            result = session.run("""
                MATCH (d1:EPSDocument)-[:MENTIONS]->(e:Entity)<-[:MENTIONS]-(d2:EPSDocument)
                WHERE id(d1) > id(d2)  // Avoid duplicate pairs
                WITH d1, d2, count(e) AS shared_entities
                WHERE shared_entities > 1  // Documents share multiple entities
                MERGE (d1)-[:SHARES_ENTITIES {count: shared_entities}]->(d2)
                RETURN count(*) as entity_links_created
            """).single()

            entity_count = result['entity_links_created'] if result else 0
            print(f"Created {entity_count} SHARES_ENTITIES relationships")

            print("✓ Document relationships creation completed!")

    except Exception as e:
        print(f"Error creating document relationships: {e}")
        return False

    finally:
        driver.close()

    return True

if __name__ == "__main__":
    success = create_document_relationships()
    if success:
        print("\n✅ Document relationships creation completed successfully!")
    else:
        print("\n❌ Document relationships creation failed!")
