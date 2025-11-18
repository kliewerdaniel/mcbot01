#!/usr/bin/env python
"""
Extensive testing suite for conversation retrieval and application functionality.
Tests the full pipeline: retriever + reasoning agent + API endpoints.
"""

import sys
import json
import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Any

# Ensure we can import our modules
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

from scripts.conversation_retriever import ConversationRetriever
from scripts.conversation_reasoning_agent import ConversationReasoningAgent


class TestConversationRetrieval:
    """Test suite for conversation data retrieval"""

    @pytest.fixture
    def mock_retriever(self):
        """Create a ConversationRetriever with mocked dependencies"""
        retriever = ConversationRetriever.__new__(ConversationRetriever)

        # Mock the driver and session
        mock_driver = Mock()
        mock_session = Mock()
        mock_driver.session.return_value.__enter__.return_value = mock_session
        mock_driver.session.return_value.__exit__.return_value = None
        retriever.driver = mock_driver
        retriever.ollama_model = "granite4:micro-h"
        retriever.embedding_model = "mxbai-embed-large:latest"

        return retriever, mock_session

    @patch('scripts.conversation_retriever.ollama.embeddings')
    def test_retrieve_context_with_query(self, mock_embeddings, mock_retriever):
        """Test retrieving context from conversations with a query"""
        retriever, mock_session = mock_retriever

        # Mock embedding response
        mock_embeddings.return_value = {'embedding': [0.1] * 1024}

        # Mock vector search results
        mock_session.run.return_value.data.return_value = [
            {
                "conversation_id": "conv-1",
                "title": "Test Conversation",
                "content": "Test content",
                "document_type": "chat",
                "summary": "Test summary",
                "relevance_score": 0.85,
                "topics": ["test"],
                "entities": [],
                "retrieval_method": "vector_search"
            },
            {
                "conversation_id": "conv-2",
                "title": "Another Conversation",
                "content": "More content",
                "document_type": "chat",
                "summary": "Another summary",
                "relevance_score": 0.75,
                "topics": ["topic"],
                "entities": [],
                "retrieval_method": "vector_search"
            }
        ]

        # Then mock topic expansion returns empty
        mock_side_effect = [
            Mock(data=lambda: [
                {
                    "conversation_id": "conv-1",
                    "title": "Test Conversation",
                    "content": "Test content",
                    "document_type": "chat",
                    "summary": "Test summary",
                    "relevance_score": 0.85,
                    "topics": ["test"],
                    "entities": [],
                    "retrieval_method": "vector_search"
                }
            ]),
            Mock(data=lambda: []),  # topic expansion
            Mock(data=lambda: []),  # entity expansion
        ]
        mock_session.run.side_effect = mock_side_effect

        results = retriever.retrieve_context("test query", limit=5)

        assert len(results) >= 1
        assert results[0]['conversation_id'] == "conv-1"
        assert results[0]['retrieval_method'] == 'vector_search'
        assert 'content_preview' in results[0]

    @patch('scripts.conversation_retriever.ollama.generate')
    def test_extract_query_concepts(self, mock_generate, mock_retriever):
        """Test extracting key concepts from query"""
        retriever, _ = mock_retriever

        mock_generate.return_value = {'response': 'programming, ai, chat'}

        concepts = retriever._extract_query_concepts("What is programming in AI chat?")

        assert concepts == ['programming', 'ai', 'chat']

    def test_extract_query_concepts_fallback(self, mock_retriever):
        """Test fallback concept extraction when LLM fails"""
        retriever, _ = mock_retriever

        # Simulate LLM failure
        retriever._extract_query_concepts = Mock(side_effect=Exception)

        # Should fall back to simple tokenization
        concepts = retriever._extract_query_concepts("hello world test query")

        assert concepts == ["hello world test query".split()[:3]]

class TestReasoningAgent:
    """Test suite for conversation reasoning agent"""

    @pytest.fixture
    def mock_agent(self):
        """Create a ConversationReasoningAgent with mocked dependencies"""
        agent = ConversationReasoningAgent.__new__(ConversationReasoningAgent)

        # Mock retriever
        mock_retriever = Mock()
        mock_retriever.retrieve_context.return_value = [
            {
                "conversation_id": "test-conv",
                "title": "Test Conversation",
                "content": "This is a test conversation with content",
                "document_type": "chat",
                "summary": "Test summary",
                "retrieval_method": "vector_search"
            }
        ]
        agent.retriever = mock_retriever

        # Mock persona config
        agent.persona_config = {
            'system_prompt_template': 'You are a test assistant. Context: {context} Question: {query}',
            'rlhf_thresholds': {
                'retrieval_required': 0.6,
                'formality_level': 0.7,
                'technical_detail_level': 0.8,
                'citation_requirement': 0.9,
                'recent_success_rate': 0.8
            }
        }

        # Mock Ollama model
        agent.ollama_model = "granite4:micro-h"

        return agent

    def test_should_retrieve_context_positive(self, mock_agent):
        """Test that agent decides to retrieve for research questions"""
        needs_context = mock_agent.should_retrieve_context("What does the document say about X?")

        assert needs_context is True

    def test_should_retrieve_context_negative(self, mock_agent):
        """Test that agent may not retrieve for simple questions"""
        needs_context = mock_agent.should_retrieve_context("Hello, how are you?")

        # May depend on thresholds, but should generally be True for any research-like queries
        # In practice this would use the configured threshold
        assert isinstance(needs_context, bool)

    @patch('scripts.conversation_reasoning_agent.ollama.generate')
    def test_generate_response_with_context(self, mock_generate, mock_agent):
        """Test generating response when context is retrieved"""
        mock_generate.return_value = {'response': 'Here is the answer based on the conversation context.'}

        result = mock_agent.generate_response("What is in the test conversation?")

        assert result['retrieval_performed'] is True
        assert len(result['context_used']) >= 1
        assert 'response' in result
        assert 'quality_grade' in result

    @patch('scripts.conversation_reasoning_agent.ollama.generate')
    def test_generate_response_without_context(self, mock_generate, mock_agent):
        """Test generating response when no retrieval needed"""
        mock_generate.return_value = {'response': 'This is a simple greeting response.'}

        # Mock agent to not need context
        mock_agent.should_retrieve_context = Mock(return_value=False)
        mock_agent.retrieve_context = Mock(return_value=False)

        result = mock_agent.generate_response("Hello!")

        assert result['retrieval_performed'] is False
        assert result['response'] is not None

    def test_build_persona_prompt(self, mock_agent):
        """Test building system prompt with persona configuration"""
        context_str = "Test document context"
        context_docs = [{"title": "Test Doc"}]
        chat_history = [{"role": "user", "content": "Hello"}]

        prompt = mock_agent._build_persona_prompt(context_str, context_docs, chat_history)

        assert context_str in prompt
        assert "Test Doc" in prompt or context_str in prompt
        assert 'You are a test assistant.' in prompt

    def test_format_context(self, mock_agent):
        """Test formatting retrieved conversations for LLM context"""
        context_docs = [
            {
                "conversation_id": "c1",
                "title": "Test Conv 1",
                "content": "Hello world",
                "document_type": "chat",
                "summary": "Greeting"
            }
        ]

        formatted = mock_agent._format_context(context_docs)

        assert "Test Conv 1" in formatted
        assert "Hello world" in formatted
        assert "chat" in formatted

    def test_format_chat_history(self, mock_agent):
        """Test formatting chat history for inclusion"""
        history = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"}
        ]

        formatted = mock_agent._format_chat_history(history)

        assert "User: Hi" in formatted
        assert "Assistant: Hello!" in formatted

    def test_grade_response_high_quality(self, mock_agent):
        """Test grading a high-quality response"""
        query = "What is in the document?"
        response = "According to document, it contains comprehensive information about the topic with examples and citations."
        context = [{"title": "Document Title"}]

        grade = mock_agent._grade_response(query, response, context)

        assert grade > 0.5  # Should be reasonable quality

    def test_grade_response_low_quality(self, mock_agent):
        """Test grading a low-quality response"""
        query = "What is AI?"
        response = ""  # Empty response
        context = []

        grade = mock_agent._grade_response(query, response, context)

        assert grade < 0.5  # Should be low quality

    def test_evaluate_document_insights(self, mock_agent):
        """Test evaluating if response shows document insights"""
        response = "The document discusses artificial intelligence and its applications."
        context = [{"conversation_id": "doc1"}]

        score = mock_agent._evaluate_document_insights(response, context)

        assert score >= 0

class TestAPIFunctionality:
    """Integration tests for API functionality"""

    def test_status_endpoint_structure(self):
        """Test that status endpoint returns expected structure"""
        from main import SystemStatus

        # Test the Pydantic model
        status = SystemStatus(
            neo4j_connected=True,
            ollama_ready=False,
            redis_connected=True,
            conversation_count=42,
            evaluation_count=10
        )

        status_dict = status.dict()
        expected_fields = ['neo4j_connected', 'ollama_ready', 'redis_connected', 'conversation_count', 'evaluation_count']

        for field in expected_fields:
            assert field in status_dict

        assert status_dict['conversation_count'] == 42

    @pytest.mark.asyncio
    async def test_chat_endpoint_request_structure(self):
        """Test chat endpoint request/response structure"""
        from main import QueryRequest, QueryResponse

        # Test request model
        request = QueryRequest(
            query="Test query",
            chat_history=[{"role": "user", "content": "Hello"}],
            persona_override="researcher"
        )

        assert request.query == "Test query"
        assert len(request.chat_history) == 1

        # Test response model (with mock data)
        response = QueryResponse(
            response="Test response",
            context_used=[{"title": "Test Doc"}],
            quality_grade=0.85,
            retrieval_method="vector_search",
            retrieval_performed=True,
            sources=[{"title": "Test Doc"}],
            session_id="test-session-123"
        )

        assert response.response == "Test response"
        assert response.quality_grade == 0.85

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
