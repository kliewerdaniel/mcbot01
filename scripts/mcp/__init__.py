"""
GraphRAG MCP (Model Context Protocol) Module

This module provides MCP server functionality for enhanced context management
and dynamic file upload capabilities in the GraphRAG application.
"""

from .mcp_server import (
    GraphRAGMCPServer,
    ContextWindow
)

from .context_manager import (
    MultiDocumentContextManager,
    ContextSession,
    ContextChunk,
    create_context_manager
)

__all__ = [
    # Main MCP Server
    "GraphRAGMCPServer",
    "ContextWindow",

    # Context Management
    "MultiDocumentContextManager",
    "ContextSession",
    "ContextChunk",
    "create_context_manager"
]

__version__ = "1.0.0"

def get_mcp_server() -> GraphRAGMCPServer:
    """
    Factory function to create and configure MCP server instance.

    Returns:
        GraphRAGMCPServer: Configured MCP server instance
    """
    return GraphRAGMCPServer()

def get_context_manager(max_tokens: int = 16000) -> MultiDocumentContextManager:
    """
    Factory function to create and configure context manager instance.

    Args:
        max_tokens: Maximum tokens for context management

    Returns:
        MultiDocumentContextManager: Configured context manager instance
    """
    return create_context_manager(max_tokens)
