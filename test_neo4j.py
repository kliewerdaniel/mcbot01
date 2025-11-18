#!/usr/bin/env python3
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

print("Environment variables:")
print(f"NEO4J_URI: {os.getenv('NEO4J_URI')}")
print(f"NEO4J_USER: {os.getenv('NEO4J_USER')}")
print(f"NEO4J_PASSWORD: {os.getenv('NEO4J_PASSWORD')}")

# Test connection
try:
    from neo4j import GraphDatabase
    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user = os.getenv("NEO4J_USER", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "research2025")

    print(f"\nTesting connection to {uri} as {user}")

    driver = GraphDatabase.driver(uri, auth=(user, password))
    with driver.session() as session:
        result = session.run("MATCH (n) RETURN count(n) as count")
        count = result.single()["count"]
        print(f"Connection successful! Nodes in database: {count}")
    driver.close()
except Exception as e:
    print(f"Connection failed: {e}")
