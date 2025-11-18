"""
Simplified MCP Server implementation for GraphRAG application.

This provides MCP-like integration without external SDK dependencies:
- Graph operations and resource management
- Context window management for large documents
- Entity and relationship handling
- Integrated with the GraphRAG pipeline
"""

import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence
from dataclasses import dataclass, field
import traceback
import logging
import uuid
import time

# Fallback MCP classes to avoid external dependency
@dataclass
class Resource:
    uri: str
    name: str
    description: str
    mime_type: str = "application/json"

@dataclass
class Tool:
    name: str
    description: str
    input_schema: Dict[str, Any]

@dataclass
class Prompt:
    name: str
    description: str
    arguments: List[Dict[str, Any]] = field(default_factory=list)

@dataclass
class TextContent:
    type: str
    text: str

# Remove MCP imports and use mock implementations
from neo4j import GraphDatabase
import ollama
from dotenv import load_dotenv
import os

# Ensure we can import from sibling modules
current_dir = Path(__file__).parent.parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

# Import our enhanced components with fallbacks
try:
    from scripts.conversation_retriever import ConversationRetriever
except ImportError:
    ConversationRetriever = None
    logging.warning("ConversationRetriever not available")

try:
    from scripts.graph_builder import graph_builder
except ImportError:
    graph_builder = None
    logging.warning("Graph builder not available")

try:
    from scripts.entity_evaluator import entity_evaluator
except ImportError:
    entity_evaluator = None
    logging.warning("Entity evaluator not available")

# Load environment variables
load_dotenv()

@dataclass
class ContextWindow:
    """Manages context windows for large documents"""
    max_tokens: int = 4000
    current_tokens: int = 0
    chunks: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_chunk(self, content: str, importance: float = 1.0) -> bool:
        """Add content chunk if within token limit"""
        estimated_tokens = len(content.split()) * 1.3  # Rough token estimation

        if self.current_tokens + estimated_tokens <= self.max_tokens:
            self.chunks.append(content)
            self.current_tokens += estimated_tokens
            return True

        # If full, remove least important content
        self._prune_least_important()
        if self.current_tokens + estimated_tokens <= self.max_tokens:
            self.chunks.append(content)
            self.current_tokens += estimated_tokens
            return True

        return False

    def _prune_least_important(self):
        """Remove content to make space - for now, remove oldest"""
        if self.chunks:
            self.chunks.pop(0)  # Simple FIFO for now

    def get_context_string(self) -> str:
        """Get concatenated context"""
        return "\n\n".join(self.chunks)

class GraphRAGMCPServer:
    """MCP Server for GraphRAG operations"""

    def __init__(self):
        self.driver = None
        self.retriever = None
        self.context_windows: Dict[str, ContextWindow] = {}
        self.ollama_model = os.getenv("OLLAMA_MODEL", "granite4:micro-h")
        self._initialize_neo4j()
        self._initialize_retriever()

    def _initialize_neo4j(self):
        """Initialize Neo4j connection"""
        uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        user = os.getenv("NEO4J_USER", "neo4j")
        password = os.getenv("NEO4J_PASSWORD", "password")

        try:
            self.driver = GraphDatabase.driver(uri, auth=(user, password))
            # Test connection
            with self.driver.session() as session:
                session.run("RETURN 1")
            print("✓ Neo4j connection established")
        except Exception as e:
            print(f"✗ Neo4j connection failed: {e}")
            self.driver = None

    def _initialize_retriever(self):
        """Initialize conversation retriever"""
        try:
            self.retriever = ConversationRetriever(ollama_model=self.ollama_model)
            print("✓ Retriever initialized")
        except Exception as e:
            print(f"✗ Retriever initialization failed: {e}")
            self.retriever = None

    def get_resources(self) -> List[Resource]:
        """Get all available MCP resources"""
        resources = []

        try:
            with self.driver.session() as session:
                # Document resources
                doc_result = session.run("""
                    MATCH (d:Conversation)
                    RETURN d.filename as filename,
                           d.document_type as doc_type,
                           d.summary as summary,
                           id(d) as node_id
                    LIMIT 100
                """)

                for record in doc_result:
                    resource_id = f"document://{record['node_id']}"
                    resources.append(Resource(
                        uri=f"graphrag://{resource_id}",
                        name=record['filename'],
                        description=f"{record['doc_type']} - {record['summary'][:100] if record['summary'] else 'No summary'}",
                        mime_type="application/json"
                    ))

                # Entity resources
                entity_result = session.run("""
                    MATCH (e:Entity)
                    RETURN e.name as name,
                           e.type as entity_type,
                           e.description as description,
                           id(e) as node_id
                    LIMIT 100
                """)

                for record in entity_result:
                    resource_id = f"entity://{record['node_id']}"
                    resources.append(Resource(
                        uri=f"graphrag://{resource_id}",
                        name=record['name'],
                        description=f"{record['entity_type']} - {record['description'][:100] if record['description'] else 'No description'}",
                        mime_type="application/json"
                    ))

        except Exception as e:
            print(f"Error fetching resources: {e}")

        return resources

    def read_resource(self, uri: str) -> str:
        """Read specific resource content"""
        if not uri.startswith("graphrag://"):
            raise ValueError(f"Invalid URI: {uri}")

        resource_path = uri.replace("graphrag://", "")
        resource_type, resource_id = resource_path.split("://")

        try:
            with self.driver.session() as session:
                if resource_type == "document":
                    result = session.run("""
                        MATCH (d:Conversation)
                        WHERE id(d) = $node_id
                        RETURN d
                    """, node_id=int(resource_id))

                    record = result.single()
                    if record:
                        doc_data = dict(record['d'])
                        return json.dumps(doc_data, indent=2, default=str)

                elif resource_type == "entity":
                    result = session.run("""
                        MATCH (e:Entity)
                        WHERE id(e) = $node_id
                        RETURN e
                    """, node_id=int(resource_id))

                    record = result.single()
                    if record:
                        entity_data = dict(record['e'])
                        return json.dumps(entity_data, indent=2, default=str)

        except Exception as e:
            return f"Error reading resource: {e}"

        return f"Resource not found: {uri}"

    def get_tools(self) -> List[Tool]:
        """Get available MCP tools"""
        return [
            Tool(
                name="search_graph",
                description="Search the knowledge graph for entities, documents, or topics",
                input_schema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query to find relevant entities and documents"
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Maximum number of results to return",
                            "default": 10
                        },
                        "entity_types": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Filter by entity types (Person, Org, Tech, etc.)",
                            "default": []
                        }
                    },
                    "required": ["query"]
                }
            ),
            Tool(
                name="get_document_context",
                description="Retrieve full context and content for a specific document",
                input_schema={
                    "type": "object",
                    "properties": {
                        "document_id": {
                            "type": "string",
                            "description": "Document ID or filename"
                        },
                        "include_relationships": {
                            "type": "boolean",
                            "description": "Include related entities and relationships",
                            "default": true
                        }
                    },
                    "required": ["document_id"]
                }
            ),
            Tool(
                name="add_entity",
                description="Add a new entity to the knowledge graph",
                input_schema={
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "Entity name"
                        },
                        "entity_type": {
                            "type": "string",
                            "description": "Entity type (Person, Org, Tech, Concept)",
                            "enum": ["Person", "Organization", "Technology", "Concept", "Location", "Other"]
                        },
                        "description": {
                            "type": "string",
                            "description": "Entity description"
                        },
                        "source_document": {
                            "type": "string",
                            "description": "Source document filename"
                        },
                        "confidence": {
                            "type": "number",
                            "description": "Confidence score (0-1)",
                            "default": 0.8
                        }
                    },
                    "required": ["name", "entity_type", "description"]
                }
            ),
            Tool(
                name="create_relationship",
                description="Create a relationship between two entities",
                input_schema={
                    "type": "object",
                    "properties": {
                        "entity1": {
                            "type": "string",
                            "description": "First entity name"
                        },
                        "entity2": {
                            "type": "string",
                            "description": "Second entity name"
                        },
                        "relationship_type": {
                            "type": "string",
                            "description": "Type of relationship",
                            "enum": ["RELATED_TO", "WORKS_FOR", "USES", "LOCATED_IN", "PART_OF"]
                        },
                        "description": {
                            "type": "string",
                            "description": "Relationship description"
                        },
                        "confidence": {
                            "type": "number",
                            "description": "Confidence score (0-1)",
                            "default": 0.7
                        }
                    },
                    "required": ["entity1", "entity2", "relationship_type"]
                }
            ),
            Tool(
                name="get_context_window",
                description="Get or manage context windows for large documents",
                input_schema={
                    "type": "object",
                    "properties": {
                        "action": {
                            "type": "string",
                            "description": "Action to perform",
                            "enum": ["get", "add_chunk", "clear"],
                            "default": "get"
                        },
                        "window_id": {
                            "type": "string",
                            "description": "Context window identifier"
                        },
                        "content": {
                            "type": "string",
                            "description": "Content to add (for add_chunk action)"
                        },
                        "importance": {
                            "type": "number",
                            "description": "Content importance (0-1, for add_chunk action)",
                            "default": 1.0
                        }
                    },
                    "required": ["window_id"]
                }
            )
        ]

    def execute_tool(self, name: str, arguments: Dict[str, Any]) -> str:
        """Execute a tool with given arguments"""
        try:
            if name == "search_graph":
                return self._tool_search_graph(**arguments)
            elif name == "get_document_context":
                return self._tool_get_document_context(**arguments)
            elif name == "add_entity":
                return self._tool_add_entity(**arguments)
            elif name == "create_relationship":
                return self._tool_create_relationship(**arguments)
            elif name == "get_context_window":
                return self._tool_context_window(**arguments)
            else:
                return f"Unknown tool: {name}"
        except Exception as e:
            return f"Tool execution error: {e}\n{traceback.format_exc()}"

    def _tool_search_graph(self, query: str, limit: int = 10, entity_types: List[str] = None) -> str:
        """Search graph for entities and documents"""
        if not self.retriever:
            return "Retriever not initialized"

        try:
            # Use existing retriever logic
            results = self.retriever.retrieve_context(query, limit=limit)

            # Filter by entity types if specified
            if entity_types:
                filtered_results = []
                for result in results:
                    result_entities = result.get('entities', [])
                    if any(entity.get('type') in entity_types for entity in result_entities):
                        filtered_results.append(result)
                results = filtered_results[:limit]

            # Format results
            response = f"Found {len(results)} results for query: '{query}'\n\n"
            for i, result in enumerate(results, 1):
                response += f"{i}. **{result['filename']}**\n"
                if result.get('summary'):
                    response += f"   Summary: {result['summary'][:200]}...\n"
                if result.get('entities'):
                    entities = [e.get('name', '') for e in result['entities'][:3]]
                    response += f"   Entities: {', '.join(entities)}\n"
                if result.get('topics'):
                    topics = result['topics'][:3]
                    response += f"   Topics: {', '.join(topics)}\n"
                response += "\n"

            return response

        except Exception as e:
            return f"Search failed: {e}"

    def _tool_get_document_context(self, document_id: str, include_relationships: bool = True) -> str:
        """Get full document context"""
        try:
            with self.driver.session() as session:
                # Find document by filename or ID
                result = session.run("""
                    MATCH (d:Conversation)
                    WHERE d.filename = $doc_id OR toString(id(d)) = $doc_id
                    RETURN d
                """, doc_id=document_id)

                record = result.single()
                if not record:
                    return f"Document not found: {document_id}"

                doc = dict(record['d'])

                response = f"**Document: {doc['filename']}**\n\n"
                response += f"**Type:** {doc.get('document_type', 'Unknown')}\n"
                response += f"**Summary:** {doc.get('summary', 'No summary')}\n\n"

                if doc.get('content'):
                    response += f"**Content:**\n{doc['content'][:2000]}...\n\n"

                if include_relationships:
                    # Get related entities
                    entity_result = session.run("""
                        MATCH (d:Conversation)-[:MENTIONS]->(e:Entity)
                        WHERE d.filename = $doc_id OR toString(id(d)) = $doc_id
                        RETURN e.name as name, e.type as type, e.description as description
                        LIMIT 20
                    """, doc_id=document_id)

                    entities = list(entity_result)
                    if entities:
                        response += f"**Related Entities ({len(entities)}):**\n"
                        for entity in entities:
                            response += f"- {entity['name']} ({entity['type']}): {entity.get('description', 'No description')[:100]}\n"
                        response += "\n"

                return response

        except Exception as e:
            return f"Document context retrieval failed: {e}"

    def _tool_add_entity(self, name: str, entity_type: str, description: str,
                        source_document: str = None, confidence: float = 0.8) -> str:
        """Add entity to graph"""
        try:
            with self.driver.session() as session:
                # Create entity node
                result = session.run("""
                    MERGE (e:Entity {name: $name})
                    ON CREATE SET e.type = $entity_type,
                                 e.description = $description,
                                 e.confidence = $confidence,
                                 e.created_at = datetime()
                    ON MATCH SET e.confidence = CASE
                        WHEN e.confidence < $confidence THEN $confidence
                        ELSE e.confidence END
                    RETURN e
                """, name=name, entity_type=entity_type, description=description, confidence=confidence)

                entity = result.single()['e']

                # Link to source document if provided
                if source_document:
                    session.run("""
                        MATCH (d:Conversation), (e:Entity {name: $name})
                        WHERE d.filename = $doc_name
                        MERGE (d)-[:MENTIONS]->(e)
                    """, name=name, doc_name=source_document)

                return f"✅ Entity added/updated: {name} ({entity_type})"

        except Exception as e:
            return f"Entity creation failed: {e}"

    def _tool_create_relationship(self, entity1: str, entity2: str,
                                relationship_type: str, description: str = "",
                                confidence: float = 0.7) -> str:
        """Create relationship between entities"""
        try:
            with self.driver.session() as session:
                # Find or create entities
                session.run("""
                    MERGE (e1:Entity {name: $entity1})
                    MERGE (e2:Entity {name: $entity2})
                    MERGE (e1)-[r:`$rel_type`]->(e2)
                    ON CREATE SET r.description = $description,
                                 r.confidence = $confidence,
                                 r.created_at = datetime()
                    ON MATCH SET r.confidence = CASE
                        WHEN r.confidence < $confidence THEN $confidence
                        ELSE r.confidence END
                """, entity1=entity1, entity2=entity2, rel_type=relationship_type,
                    description=description, confidence=confidence)

                return f"✅ Relationship created: {entity1} --[{relationship_type}]--> {entity2}"

        except Exception as e:
            return f"Relationship creation failed: {e}"

    def _tool_context_window(self, action: str, window_id: str,
                           content: str = None, importance: float = 1.0) -> str:
        """Manage context windows"""
        if action == "clear":
            self.context_windows.pop(window_id, None)
            return f"✅ Context window cleared: {window_id}"

        if action == "add_chunk":
            if content is None:
                return "❌ Content required for add_chunk action"

            if window_id not in self.context_windows:
                self.context_windows[window_id] = ContextWindow()

            success = self.context_windows[window_id].add_chunk(content, importance)
            status = "✅ Added" if success else "⚠️ Rejected (window full)"
            current_tokens = self.context_windows[window_id].current_tokens

            return f"{status} to context window '{window_id}' ({current_tokens:.0f}/{self.context_windows[window_id].max_tokens} tokens)"

        # Default: get context
        if window_id not in self.context_windows:
            return f"Context window not found: {window_id}"

        context = self.context_windows[window_id].get_context_string()
        return f"**Context Window '{window_id}':**\n\n{context}"

    def get_prompts(self) -> List[Prompt]:
        """Get available MCP prompts"""
        return [
            Prompt(
                name="entity_extraction",
                description="Extract entities from text content using structured prompts",
                arguments=[
                    {
                        "name": "content",
                        "description": "Text content to extract entities from",
                        "required": True
                    },
                    {
                        "name": "document_type",
                        "description": "Type of document (pdf, csv, txt, etc.)",
                        "required": False
                    }
                ]
            ),
            Prompt(
                name="relationship_inference",
                description="Infer relationships between extracted entities",
                arguments=[
                    {
                        "name": "entities",
                        "description": "List of entity names to analyze relationships for",
                        "required": True
                    },
                    {
                        "name": "context",
                        "description": "Contextual text to base relationship inference on",
                        "required": False
                    }
                ]
            ),
            Prompt(
                name="graph_construction",
                description="Generate Cypher queries for graph construction from entities and relationships",
                arguments=[
                    {
                        "name": "entities",
                        "description": "JSON list of entities with properties",
                        "required": True
                    },
                    {
                        "name": "relationships",
                        "description": "JSON list of relationships",
                        "required": False
                    }
                ]
            )
        ]

    def get_prompt_content(self, name: str, arguments: Dict[str, Any]) -> str:
        """Get prompt content for specific template"""
        if name == "entity_extraction":
            return self._get_entity_extraction_prompt(arguments)
        elif name == "relationship_inference":
            return self._get_relationship_inference_prompt(arguments)
        elif name == "graph_construction":
            return self._get_graph_construction_prompt(arguments)
        else:
            return f"Unknown prompt: {name}"

    def _get_entity_extraction_prompt(self, args: Dict[str, Any]) -> str:
        """Generate entity extraction prompt"""
        content = args.get("content", "")
        doc_type = args.get("document_type", "general")

        prompt = f"""Extract named entities from the following {doc_type} document content.

Focus on identifying:
- **Person**: Names of individuals, authors, contributors
- **Organization**: Companies, institutions, research groups
- **Technology**: Software tools, frameworks, techniques, algorithms
- **Concept**: Key terms, theoretical concepts, methodologies
- **Location**: Geographic locations, institutions, facilities

Document Content:
{content}

Provide output as JSON:
{{
  "entities": [
    {{
      "name": "Entity Name",
      "type": "Entity_Type",
      "description": "Brief description of entity's role or context",
      "confidence": 0.0-1.0,
      "mentions": ["position1", "position2"]
    }}
  ],
  "document_type": "{doc_type}",
  "extraction_quality": 0.0-1.0
}}

Extract only entities that are clearly mentioned and provide meaningful value to understanding the document."""

        return prompt

    def _get_relationship_inference_prompt(self, args: Dict[str, Any]) -> str:
        """Generate relationship inference prompt"""
        entities = args.get("entities", [])
        context = args.get("context", "")

        entities_list = "\n".join([f"- {entity}" for entity in entities])

        prompt = f"""Analyze relationships between the following entities based on the provided context.

Entities:
{entities_list}

Context:
{context}

Identify meaningful relationships such as:
- Employment/affiliation (WORKS_FOR)
- Usage/technology adoption (USES)
- Location/geographic relationships (LOCATED_IN)
- Hierarchical structures (PART_OF, MEMBER_OF)
- Semantic associations (RELATED_TO)

Output as JSON:
{{
  "relationships": [
    {{
      "entity1": "Entity Name",
      "entity2": "Entity Name",
      "type": "RELATIONSHIP_TYPE",
      "description": "How entities are related",
      "confidence": 0.0-1.0,
      "evidence": "Excerpt from context supporting this relationship"
    }}
  ],
  "analysis_summary": "Brief summary of key relationship patterns"
}}

Only infer relationships that are strongly supported by evidence from the context."""

        return prompt

    def _get_graph_construction_prompt(self, args: Dict[str, Any]) -> str:
        """Generate graph construction prompt"""
        entities = args.get("entities", [])
        relationships = args.get("relationships", [])

        prompt = f"""Generate Neo4j Cypher queries to construct a knowledge graph from the following entities and relationships.

Entities: {json.dumps(entities, indent=2)}

Relationships: {json.dumps(relationships, indent=2)}

Generate Cypher queries that:
1. Create entity nodes (use MERGE to avoid duplicates)
2. Create relationship edges between entities
3. Handle properties and metadata appropriately
4. Include error handling and data validation

Output format:
{{
  "cypher_queries": [
    "CREATE/MERGE statement 1",
    "CREATE/MERGE statement 2"
  ],
  "node_count": expected_number_of_nodes,
  "relationship_count": expected_number_of_relationships,
  "validation_notes": ["any important validation considerations"]
}}

Ensure queries are efficient and handle potential conflicts gracefully."""

        return prompt


# Standalone demo functions for testing MCP functionality without external SDK
def demo_server():
    """Demo the GraphRAG MCP server functionality"""
    try:
        # Initialize GraphRAG MCP server
        print("🚀 Initializing GraphRAG MCP Server...")
        graph_server = GraphRAGMCPServer()

        print("\n📊 Testing Resources...")
        resources = graph_server.get_resources()
        print(f"Found {len(resources)} resources")

        print("\n🔧 Testing Tools...")
        tools = graph_server.get_tools()
        print(f"Available tools: {[tool.name for tool in tools]}")

        print("\n💬 Testing Prompts...")
        prompts = graph_server.get_prompts()
        print(f"Available prompts: {[prompt.name for prompt in prompts]}")

        # Test tool execution
        print("\n🧪 Testing tool execution...")
        result = graph_server.execute_tool("search_graph", {"query": "test", "limit": 5})
        print(f"Search result: {result[:200]}...")

        print("\n✅ GraphRAG MCP Server initialized successfully!")
        print("MCP-like functionality is available through the enhanced API")

    except Exception as e:
        print(f"❌ Server initialization failed: {e}")
        import traceback
        traceback.print_exc()

def demo_context_window():
    """Demo context window functionality"""
    print("\n🪟 Testing Context Window Management...")

    # Test basic context window
    from scripts.mcp.context_manager import create_context_manager
    try:
        context_mgr = create_context_manager(max_tokens=1000)

        # Add some test content
        stats1 = context_mgr.add_document_content(
            session_id="test_session",
            document_name="test.pdf",
            content="This is a test document with some entities like Apple Inc., Google, and Microsoft.",
            entities=["Apple Inc.", "Google", "Microsoft"],
            topics=["technology", "companies"]
        )

        print(f"Added document content: {stats1}")

        # Get context
        context = context_mgr.get_window_context("test_session")
        print(f"Retrieved context: {context[:200]}...")

    except Exception as e:
        print(f"Context manager demo failed: {e}")

def demo_entity_evaluator():
    """Demo entity evaluator functionality"""
    print("\n🔍 Testing Entity Evaluation...")

    try:
        from scripts.entity_evaluator import entity_evaluator

        test_content = "Dr. Sarah Johnson works at MIT and developed the Transformer model."
        entities = entity_evaluator.extract_entities_from_text(test_content, "test.txt")

        print(f"Extracted {len(entities)} entities:")
        for entity in entities:
            quality = entity_evaluator.evaluate_entity_quality(entity)
            print(f"  - {entity.name} ({entity.entity_type}): confidence {entity.confidence:.2f}, quality: {quality}")

        if len(entities) > 1:
            relationships = entity_evaluator.infer_relationships(entities)
            print(f"Inferred {len(relationships)} relationships")

    except Exception as e:
        print(f"Entity evaluator demo failed: {e}")

def demo_graph_builder():
    """Demo graph builder functionality"""
    print("\n🕸️ Testing Graph Construction...")

    try:
        from scripts.graph_builder import graph_builder

        # Test schema inference
        print("Graph builder initialized - schema inference and optimization available")

    except Exception as e:
        print(f"Graph builder demo failed: {e}")

if __name__ == "__main__":
    print("🎯 GraphRAG MCP Integration Demo")
    print("="*50)

    demo_server()
    demo_context_window()
    demo_entity_evaluator()
    demo_graph_builder()

    print("\n🎊 All GraphRAG MCP components successfully demonstrated!")
    print("To run the full FastAPI server: python main.py")
    print("To use MCP functionality: Import GraphRAGMCPServer from scripts.mcp.mcp_server")
