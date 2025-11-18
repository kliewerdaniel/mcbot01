from neo4j import GraphDatabase
import os
from dotenv import load_dotenv

load_dotenv()

# Connect to Neo4j
uri = os.getenv('NEO4J_URI', '')
user = os.getenv('NEO4J_USER', '')
password = os.getenv('NEO4J_PASSWORD', '')

if uri and user and password:
    driver = GraphDatabase.driver(uri, auth=(user, password))
    with driver.session() as session:
        # Check what nodes exist
        result = session.run('MATCH (n) RETURN DISTINCT labels(n) as labels, count(n) as count ORDER BY count DESC LIMIT 10')
        print('Node types in database:')
        for record in result:
            print(f'  {record["labels"]}: {record["count"]}')

        # Check if conversation nodes exist
        try:
            result = session.run('MATCH (n:Conversation) RETURN count(n) as count')
            conv_count = result.single()["count"] if result.single() else 0
            print(f'\nConversation nodes: {conv_count}')
        except:
            print('\nConversation nodes: Could not count')

        # Get some sample conversations to see their structure
        try:
            result = session.run('MATCH (n:Conversation) RETURN n LIMIT 3')
            print('\nSample conversation nodes:')
            for record in result:
                node = record["n"]
                print(f'  Title: {node.get("title", "N/A")}')
                print(f'  Properties: {list(node.keys())}')
                # Show some sample content if available
                if 'raw_content' in node:
                    content_preview = str(node['raw_content'])[:200] + '...' if len(str(node['raw_content'])) > 200 else str(node['raw_content'])
                    print(f'  Content preview: {content_preview}')
                print()
        except Exception as e:
            print(f'  Error getting samples: {e}')

        # Check if there are any vector indexes
        try:
            result = session.run('SHOW INDEXES YIELD name, labelsOrTypes, properties, type WHERE labelsOrTypes IS NOT NULL')
            print('\nIndexes:')
            for record in result:
                print(f'  {record["name"]} ({record["type"]}): {record["labelsOrTypes"]} on {record["properties"]}')
        except Exception as e:
            print(f'\nError accessing indexes: {e}')

        # Test the vector index specifically
        try:
            import ollama
            # Generate a real embedding for "hello"
            embedding = ollama.embeddings(model="mxbai-embed-large:latest", prompt="hello")["embedding"]
            result = session.run('''
                CALL db.index.vector.queryNodes("conversation_embeddings", $limit, $embedding)
                YIELD node, score
                RETURN count(node) as vector_count
            ''', embedding=embedding, limit=5)
            count = result.single()["vector_count"]
            print(f'\nVector index test: {count} nodes found for "hello" query')
        except Exception as e:
            print(f'\nVector index test error: {e}')

        driver.close()
else:
    print('Neo4j environment variables not set');
