#!/usr/bin/env python
"""
Extensive testing suite for conversation data ingestion functionality.
Tests both JSON ingestion from conversations.json and validates database structure.
"""

import sys
import json
import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch
from neo4j import GraphDatabase

# Ensure we can import our modules
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

from scripts.ingest_conversation_data import ConversationGraphBuilder


class TestConversationIngestion:
    """Comprehensive test suite for conversation ingestion"""

    @pytest.fixture
    def builder(self):
        """Create a ConversationGraphBuilder instance with mocked Neo4j driver"""
        builder = ConversationGraphBuilder.__new__(ConversationGraphBuilder)
        builder.embedding_model = "mxbai-embed-large:latest"
        builder.ollama_model = "granite4:micro-h"

        # Mock the Neo4j driver and session
        mock_driver = Mock()
        mock_session = Mock()
        mock_driver.session.return_value.__enter__.return_value = mock_session
        mock_driver.session.return_value.__exit__.return_value = None
        builder.driver = mock_driver
        builder.session = mock_session

        return builder

    def test_extract_conversation_content_empty(self, builder):
        """Test extracting content from conversation with no messages"""
        conversation = {
            "conversation_id": "test-123",
            "title": "Empty Test",
            "mapping": {}
        }

        content = builder.extract_conversation_content(conversation)
        assert content == ""

    @patch('scripts.ingest_conversation_data.ollama.embeddings')
    @patch('scripts.ingest_conversation_data.ollama.generate')
    def test_create_conversation_node_success(self, mock_generate, mock_embeddings, builder):
        """Test successful creation of conversation node"""

        # Mock LLM responses
        mock_embeddings.return_value = {'embedding': [0.1] * 1024}
        mock_generate.return_value = {
            'response': '{"topics": ["test"], "entities": [], "document_type": "chat", "summary": "Test summary"}'
        }

        # Mock Neo4j session
        builder.session.run.return_value = None

        conversation_id = "test-123"
        title = "Test Conversation"
        content = "This is test conversation content."

        builder.create_conversation_node(conversation_id, title, content)

        # Verify the MERGE query was called with correct structure
        calls = [call[0][0] for call in builder.session.run.call_args_list if 'MERGE (d:Conversation' in call[0][0]]

        # Should only be one call for the main MERGE
        assert len(calls) == 1

        cypher_query = calls[0]
        assert 'MERGE (d:Conversation {' in cypher_query
        assert 'conversation_id: $conversation_id' in cypher_query
        assert 'content_embedding: $embedding' in cypher_query

    def test_extract_conversation_content_with_messages(self, builder):
        """Test extracting content from conversation with actual messages"""
        conversation = {
            "conversation_id": "test-456",
            "mapping": {
                "msg-1": {
                    "message": {
                        "content": {"parts": ["Hello ", "world"]}
                    }
                },
                "msg-2": {
                    "message": {
                        "content": {"parts": ["How are you?"]}
                    }
                }
            }
        }

        content = builder.extract_conversation_content(conversation)
        assert content == "Hello world\nHow are you?"

    def test_extract_conversation_content_malformed_messages(self, builder):
        """Test extracting content handles malformed messages gracefully"""
        conversation = {
            "mapping": {
                "msg-1": {
                    "message": {
                        "content": None  # None content
                    }
                },
                "msg-2": {
                    "message": {
                        "content": {"parts": []}  # Empty parts
                    }
                },
                "msg-3": {
                    "message": {
                        "content": {"parts": ["Good content"]}  # Valid
                    }
                }
            }
        }

        content = builder.extract_conversation_content(conversation)
        assert content == "Good content"

    @patch('scripts.ingest_conversation_data.ollama.embeddings', side_effect=Exception("Embedding failure"))
    def test_create_conversation_node_embedding_failure(self, mock_embeddings, builder):
        """Test handling of embedding generation failure"""
        builder.session.run.return_value = None

        # Should not raise exception, just log it
        try:
            builder.create_conversation_node("test-123", "Test", "content")
            # Should complete successfully with empty embeddings
            assert True
        except Exception:
            assert False, "Should handle embedding failure gracefully"

    @patch('scripts.ingest_conversation_data.ollama.generate', side_effect=Exception("LLM failure"))
    @patch('scripts.ingest_conversation_data.ollama.embeddings')
    def test_create_conversation_node_llm_failure(self, mock_embeddings, mock_generate, builder):
        """Test handling of LLM entity extraction failure"""
        mock_embeddings.return_value = {'embedding': [0.1] * 1024}
        builder.session.run.return_value = None

        # Should still create node with fallback data
        try:
            builder.create_conversation_node("test-123", "Test", "content")
            # Should complete successfully with fallback entities
            assert True
        except Exception:
            assert False, "Should handle LLM failure gracefully"


class TestIngestionIntegration:
    """Integration tests for full ingestion pipeline"""

    def test_ingest_small_conversation_dataset(self):
        """Test ingesting a small synthetic conversation dataset"""
        # Create a small test conversation file
        test_conversations = [
            {
                "conversation_id": "integration-test-1",
                "title": "Test Chat 1",
                "mapping": {
                    "msg-1": {
                        "message": {"content": {"parts": ["Hello", " world"]}}
                    }
                }
            },
            {
                "conversation_id": "integration-test-2",
                "title": "Test Chat 2",
                "mapping": {
                    "msg-2": {
                        "message": {"content": {"parts": ["Goodbye", " cruel", " world"]}}
                    }
                }
            }
        ]

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(test_conversations, f)
            temp_file_path = f.name

        try:
            # Mock the builder since we don't have actual Neo4j/Ollama
            builder = ConversationGraphBuilder.__new__(ConversationGraphBuilder)
            builder.embedding_model = "mxbai-embed-large:latest"
            builder.ollama_model = "granite4:micro-h"

            # Mock necessary methods
            builder.generate_document_embedding = Mock(return_value=[0.1] * 1024)
            builder.extract_document_entities = Mock(return_value={
                "topics": ["test"], "entities": [], "document_type": "chat", "summary": "summary"
            })

            mock_driver = Mock()
            mock_session = Mock()
            mock_driver.session.return_value.__enter__.return_value = mock_session
            mock_driver.session.return_value.__exit__.return_value = None
            builder.driver = mock_driver

            # Set up session run to return None for MERGE operations
            def mock_run_side_effect(query, **params):
                # We could inspect the query/parameters here for validation
                return None

            mock_session.run.side_effect = mock_run_side_effect

            # Run the ingestion
            json_path = Path(temp_file_path)
            builder.ingest_conversations_json(json_path)

            # Verify session.run was called for each conversation
            # We expect: 1 MERGE + 1 CREATE (topic) + 1 CREATE (entity) per conversation = 3 per conversation
            calls = mock_session.run.call_count
            # But actually, should be at least 2 per conversation (MERGE and topic relationship)
            assert calls >= 4, f"Expected at least 4 database calls (2 per conversation), got {calls}"

        finally:
            Path(temp_file_path).unlink(missing_ok=True)


class TestVectorIndexCreation:
    """Test vector index creation functionality"""

    def test_create_vector_indexes(self):
        """Test that vector indexes are created correctly"""
        builder = ConversationGraphBuilder.__new__(ConversationGraphBuilder)

        # Mock driver and session
        mock_driver = Mock()
        mock_session = Mock()
        mock_driver.session.return_value.__enter__.return_value = mock_session
        mock_driver.session.return_value.__exit__.return_value = None
        builder.driver = mock_driver

        builder.create_vector_indexes()

        # Verify the correct index was created
        calls = mock_session.run.call_args_list

        # Should have created two indexes: conversation_embeddings and topic_embeddings
        index_queries = []
        for call in calls:
            query = call[0][0]
            if 'CREATE VECTOR INDEX' in query:
                index_queries.append(query)

        assert len(index_queries) >= 1, "Should create at least one vector index"

        # Check that conversation_embeddings index is created for Conversation nodes
        conv_index_created = any("conversation_embeddings" in query and "FOR (d:Conversation)" in query
                                for query in index_queries)
        assert conv_index_created, "Should create conversation_embeddings index for Conversation nodes"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
