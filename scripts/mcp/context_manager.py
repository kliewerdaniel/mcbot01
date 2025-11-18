"""
Context Manager for MCP Server.

Handles sliding window context aggregation for large documents with:
- Relevance scoring for context chunks
- Context pruning strategies based on token limits
- Multi-document context merging
- Intelligent chunk selection and weighting
"""

import json
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import math
import hashlib
import numpy as np
from collections import defaultdict

from .mcp_server import ContextWindow


@dataclass
class ContextChunk:
    """Represents a chunk of context with metadata"""
    id: str
    content: str
    source_document: str
    chunk_index: int
    relevance_score: float = 0.0
    importance_weight: float = 1.0
    token_count: int = 0
    entities: List[str] = field(default_factory=list)
    topics: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
    access_count: int = 0
    last_accessed: Optional[datetime] = None

    def estimated_tokens(self) -> int:
        """Estimate token count for this chunk"""
        if self.token_count > 0:
            return self.token_count
        # Rough estimation: ~1.3 tokens per word, plus markup
        word_count = len(self.content.split())
        estimated = int(word_count * 1.3) + 10  # +10 for JSON overhead
        self.token_count = estimated
        return estimated

    def update_access(self):
        """Update access tracking"""
        self.access_count += 1
        self.last_accessed = datetime.now()

    def get_age_hours(self) -> float:
        """Get age in hours since creation or last access"""
        reference_time = self.last_accessed if self.last_accessed else self.timestamp
        age = datetime.now() - reference_time
        return age.total_seconds() / 3600


@dataclass
class ContextSession:
    """Manages context for a single MCP session"""
    session_id: str
    max_tokens: int = 8000
    sliding_window: bool = True
    window_overlap: float = 0.2  # 20% overlap between windows
    chunks: Dict[str, ContextChunk] = field(default_factory=dict)
    active_documents: List[str] = field(default_factory=list)
    total_tokens: int = 0
    created_at: datetime = field(default_factory=datetime.now)

    def add_chunk(self, chunk: ContextChunk, force: bool = False) -> bool:
        """Add context chunk with intelligent pruning"""
        chunk_tokens = chunk.estimated_tokens()

        # Check if we can add without exceeding limits
        if self.total_tokens + chunk_tokens <= self.max_tokens:
            self.chunks[chunk.id] = chunk
            self.total_tokens += chunk_tokens

            # Track document
            if chunk.source_document not in self.active_documents:
                self.active_documents.append(chunk.source_document)

            chunk.update_access()
            return True

        # If force is True, make space
        if force:
            self._prune_chunks(chunk_tokens)
            self.chunks[chunk.id] = chunk
            self.total_tokens += chunk_tokens
            chunk.update_access()
            return True

        return False

    def _prune_chunks(self, tokens_needed: int):
        """Prune chunks to make space for new content"""
        # Calculate how many tokens to remove
        tokens_to_remove = tokens_needed - (self.max_tokens - self.total_tokens)

        if tokens_to_remove <= 0:
            return

        # Score chunks for pruning (lower score = more likely to be pruned)
        chunk_scores = []
        for chunk_id, chunk in self.chunks.items():
            score = self._calculate_chunk_score(chunk)
            chunk_scores.append((chunk_id, score))

        # Sort by score (ascending - lowest score first)
        chunk_scores.sort(key=lambda x: x[1])

        # Remove chunks until we have enough space
        removed_tokens = 0
        chunks_to_remove = []

        for chunk_id, _ in chunk_scores:
            if removed_tokens >= tokens_to_remove:
                break
            chunk = self.chunks[chunk_id]
            tokens = chunk.estimated_tokens()
            removed_tokens += tokens
            chunks_to_remove.append(chunk_id)

        # Remove the selected chunks
        for chunk_id in chunks_to_remove:
            if chunk_id in self.chunks:
                del self.chunks[chunk_id]
                self.total_tokens -= self.chunks[chunk_id].estimated_tokens()

    def _calculate_chunk_score(self, chunk: ContextChunk) -> float:
        """Calculate score for chunk pruning (higher = less likely to prune)"""
        base_score = (
            chunk.relevance_score * 0.4 +      # Relevance
            chunk.importance_weight * 0.3 +    # Importance
            (1.0 / (1.0 + chunk.get_age_hours())) * 0.2 +  # Recency (inverse age)
            (chunk.access_count * 0.1)         # Access frequency
        )

        # Boost for chunks with unique entities
        unique_entity_boost = min(0.2, len(set(chunk.entities)) * 0.1)
        base_score += unique_entity_boost

        return base_score

    def get_context_string(self, max_chunks: int = 50) -> str:
        """Get formatted context string with sliding window"""
        if not self.chunks:
            return ""

        # Get active chunks sorted by relevance and recency
        active_chunks = list(self.chunks.values())

        # Boost recently accessed chunks
        for chunk in active_chunks:
            if chunk.get_age_hours() < 1.0:  # Accessed within last hour
                chunk.relevance_score = min(1.0, chunk.relevance_score + 0.1)

        # Sort by composite score
        active_chunks.sort(key=lambda x: (
            -(x.relevance_score * 0.5 + x.importance_weight * 0.3 + x.access_count * 0.2)
        ))

        # Limit to max_chunks
        selected_chunks = active_chunks[:max_chunks]

        # Update access times
        for chunk in selected_chunks:
            chunk.update_access()

        # Format as context string
        context_parts = []

        for i, chunk in enumerate(selected_chunks, 1):
            header = f"[Document: {chunk.source_document}, Chunk {chunk.chunk_index}]"
            if chunk.entities:
                header += f" [Entities: {', '.join(chunk.entities[:3])}]"
            if chunk.topics:
                header += f" [Topics: {', '.join(chunk.topics[:2])}]"

            context_parts.append(f"{header}\n{chunk.content}")

        return "\n\n---\n\n".join(context_parts)

    def get_document_summary(self) -> Dict[str, Any]:
        """Get summary of documents in context"""
        doc_stats = defaultdict(lambda: {
            'chunk_count': 0,
            'total_entities': set(),
            'total_topics': set(),
            'avg_relevance': 0.0,
            'total_tokens': 0
        })

        for chunk in self.chunks.values():
            doc = chunk.source_document
            stats = doc_stats[doc]
            stats['chunk_count'] += 1
            stats['total_entities'].update(chunk.entities)
            stats['total_topics'].update(chunk.topics)
            stats['total_tokens'] += chunk.estimated_tokens()

        # Calculate averages and format
        summary = {}
        for doc, stats in doc_stats.items():
            total_relevance = sum(c.relevance_score for c in self.chunks.values()
                                if c.source_document == doc)
            summary[doc] = {
                'chunk_count': stats['chunk_count'],
                'unique_entities': len(stats['total_entities']),
                'unique_topics': len(stats['total_topics']),
                'avg_relevance': total_relevance / stats['chunk_count'],
                'total_tokens': stats['total_tokens'],
                'entities': list(stats['total_entities'])[:10],  # Top 10
                'topics': list(stats['total_topics'])[:5]     # Top 5
            }

        return summary

    def clear_stale_chunks(self, max_age_hours: float = 24.0):
        """Remove chunks older than specified age"""
        chunks_to_remove = []
        tokens_removed = 0

        for chunk_id, chunk in self.chunks.items():
            if chunk.get_age_hours() > max_age_hours:
                chunks_to_remove.append(chunk_id)
                tokens_removed += chunk.estimated_tokens()

        for chunk_id in chunks_to_remove:
            del self.chunks[chunk_id]

        self.total_tokens -= tokens_removed

        return {
            'removed_count': len(chunks_to_remove),
            'tokens_freed': tokens_removed
        }


class MultiDocumentContextManager:
    """Manages context across multiple documents with intelligent merging"""

    def __init__(self,
                 max_total_tokens: int = 16000,
                 session_max_tokens: int = 8000,
                 sliding_window: bool = True):

        self.max_total_tokens = max_total_tokens
        self.session_max_tokens = session_max_tokens
        self.sliding_window = sliding_window
        self.sessions: Dict[str, ContextSession] = {}
        self.global_chunk_index = 0

        # Similarity tracking for deduplication
        self.chunk_hashes: Dict[str, List[str]] = defaultdict(list)  # hash -> [chunk_ids]

    def create_session(self, session_id: str = None) -> str:
        """Create a new context session"""
        if not session_id:
            session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        self.sessions[session_id] = ContextSession(
            session_id=session_id,
            max_tokens=self.session_max_tokens,
            sliding_window=self.sliding_window
        )

        return session_id

    def add_document_content(self,
                           session_id: str,
                           document_name: str,
                           content: str,
                           chunk_size: int = 1000,
                           overlap: int = 200,
                           entities: List[str] = None,
                           topics: List[str] = None) -> Dict[str, Any]:
        """Add document content with intelligent chunking"""

        if session_id not in self.sessions:
            self.create_session(session_id)

        session = self.sessions[session_id]
        entities = entities or []
        topics = topics or []

        # Split content into chunks
        chunks = self._chunk_text(content, chunk_size, overlap)

        added_chunks = 0
        rejected_chunks = 0
        total_relevance = 0.0

        for i, chunk_content in enumerate(chunks):
            # Create unique chunk ID
            chunk_hash = hashlib.md5(f"{document_name}_{i}_{chunk_content[:100]}".encode()).hexdigest()[:8]
            chunk_id = f"{document_name}_{chunk_hash}"

            # Check for duplicates
            if self._is_duplicate_chunk(chunk_content):
                rejected_chunks += 1
                continue

            # Create context chunk
            chunk = ContextChunk(
                id=chunk_id,
                content=chunk_content,
                source_document=document_name,
                chunk_index=i,
                relevance_score=self._calculate_relevance(chunk_content, entities, topics),
                entities=[e for e in entities if e.lower() in chunk_content.lower()],
                topics=[t for t in topics if t.lower() in chunk_content.lower()]
            )

            # Try to add chunk to session
            if session.add_chunk(chunk, force=(i == 0)):  # Force add first chunk
                added_chunks += 1
                total_relevance += chunk.relevance_score

                # Track for deduplication
                content_hash = hashlib.md5(chunk_content.encode()).hexdigest()
                self.chunk_hashes[content_hash].append(chunk_id)

                self.global_chunk_index += 1
            else:
                rejected_chunks += 1

        avg_relevance = total_relevance / max(1, added_chunks)

        return {
            'added_chunks': added_chunks,
            'rejected_chunks': rejected_chunks,
            'total_chunks': len(chunks),
            'avg_relevance': avg_relevance,
            'session_tokens': session.total_tokens,
            'active_documents': len(session.active_documents)
        }

    def _chunk_text(self, text: str, chunk_size: int, overlap: int) -> List[str]:
        """Split text into overlapping chunks"""
        if len(text) <= chunk_size:
            return [text]

        chunks = []
        start = 0

        while start < len(text):
            end = start + chunk_size

            # Adjust end to not split words
            if end < len(text):
                # Find last space within chunk_size
                last_space = text.rfind(' ', start, end)
                if last_space > start:
                    end = last_space

            chunk = text[start:end].strip()
            if chunk:  # Only add non-empty chunks
                chunks.append(chunk)

            # Move start with overlap
            start = max(start + 1, end - overlap)

        return chunks

    def _calculate_relevance(self, content: str, entities: List[str], topics: List[str]) -> float:
        """Calculate relevance score for content chunk"""
        content_lower = content.lower()

        # Entity mentions
        entity_score = 0.0
        if entities:
            entity_mentions = sum(1 for entity in entities
                                if entity.lower() in content_lower)
            entity_score = min(1.0, entity_mentions / len(entities))

        # Topic mentions
        topic_score = 0.0
        if topics:
            topic_mentions = sum(1 for topic in topics
                               if topic.lower() in content_lower)
            topic_score = min(1.0, topic_mentions / len(topics))

        # Content quality score (sentence diversity, length, structure)
        quality_score = self._calculate_content_quality(content)

        # Weighted combination
        total_score = (
            entity_score * 0.4 +    # Entity presence
            topic_score * 0.3 +     # Topic presence
            quality_score * 0.3     # Content quality
        )

        return min(1.0, total_score)

    def _calculate_content_quality(self, content: str) -> float:
        """Calculate content quality metrics"""
        if not content.strip():
            return 0.0

        # Length score (prefer substantial chunks)
        word_count = len(content.split())
        length_score = min(1.0, word_count / 50)  # Optimal around 50+ words

        # Sentence diversity
        sentences = content.split('.')
        sentence_count = len([s for s in sentences if s.strip()])
        sentence_score = min(1.0, sentence_count / 3)  # Prefer 3+ sentences

        # Structure indicators
        structure_indicators = ['•', '-', ':', '\n']
        structure_score = min(1.0, sum(content.count(ind) for ind in structure_indicators) / 5)

        return (length_score * 0.5 + sentence_score * 0.3 + structure_score * 0.2)

    def _is_duplicate_chunk(self, content: str, threshold: float = 0.95) -> bool:
        """Check if chunk content is a near duplicate"""
        content_hash = hashlib.md5(content.encode()).hexdigest()

        if content_hash in self.chunk_hashes:
            # If we have exact hash matches, it's a duplicate
            return True

        # Simple similarity check - count common words
        existing_chunks = []
        for hash_key, chunk_ids in self.chunk_hashes.items():
            if hash_key != content_hash and chunk_ids:
                existing_chunks.extend(chunk_ids[:1])  # Check one per hash

        # For performance, only check a few recent chunks
        for chunk_id in existing_chunks[-5:]:
            for session in self.sessions.values():
                if chunk_id in session.chunks:
                    existing_content = session.chunks[chunk_id].content
                    similarity = self._calculate_similarity(content, existing_content)
                    if similarity > threshold:
                        return True

        return False

    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate simple text similarity"""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))

        return intersection / union if union > 0 else 0.0

    def get_session_context(self, session_id: str) -> str:
        """Get context string for a session"""
        if session_id not in self.sessions:
            return ""

        return self.sessions[session_id].get_context_string()

    def get_session_summary(self, session_id: str) -> Dict[str, Any]:
        """Get detailed summary for a session"""
        if session_id not in self.sessions:
            return {}

        session = self.sessions[session_id]

        return {
            'session_id': session_id,
            'total_tokens': session.total_tokens,
            'max_tokens': session.max_tokens,
            'chunk_count': len(session.chunks),
            'active_documents': session.active_documents,
            'document_summary': session.get_document_summary(),
            'created_at': session.created_at.isoformat(),
            'avg_relevance': np.mean([c.relevance_score for c in session.chunks.values()]) if session.chunks else 0.0
        }

    def merge_sessions(self, target_session: str, source_sessions: List[str]) -> Dict[str, Any]:
        """Merge multiple sessions into target session"""
        if target_session not in self.sessions:
            self.create_session(target_session)

        target = self.sessions[target_session]
        merged_stats = {
            'added_chunks': 0,
            'rejected_chunks': 0,
            'source_sessions': len(source_sessions)
        }

        for source_id in source_sessions:
            if source_id not in self.sessions:
                continue

            source = self.sessions[source_id]

            for chunk in source.chunks.values():
                # Create new chunk ID to avoid conflicts
                new_chunk_id = f"{chunk.id}_merged_{target_session}"

                new_chunk = ContextChunk(
                    id=new_chunk_id,
                    content=chunk.content,
                    source_document=chunk.source_document,
                    chunk_index=chunk.chunk_index,
                    relevance_score=chunk.relevance_score,
                    importance_weight=chunk.importance_weight,
                    entities=chunk.entities.copy(),
                    topics=chunk.topics.copy(),
                    timestamp=chunk.timestamp,
                    access_count=chunk.access_count
                )

                if target.add_chunk(new_chunk):
                    merged_stats['added_chunks'] += 1
                else:
                    merged_stats['rejected_chunks'] += 1

        return merged_stats

    def cleanup_stale_sessions(self, max_age_hours: float = 24.0) -> Dict[str, Any]:
        """Clean up old sessions and free memory"""
        sessions_to_remove = []
        total_freed_tokens = 0

        current_time = datetime.now()

        for session_id, session in self.sessions.items():
            session_age = (current_time - session.created_at).total_seconds() / 3600
            if session_age > max_age_hours:
                sessions_to_remove.append(session_id)
                total_freed_tokens += session.total_tokens

        for session_id in sessions_to_remove:
            del self.sessions[session_id]

        return {
            'removed_sessions': len(sessions_to_remove),
            'freed_tokens': total_freed_tokens,
            'remaining_sessions': len(self.sessions)
        }

    def get_global_stats(self) -> Dict[str, Any]:
        """Get global context manager statistics"""
        total_sessions = len(self.sessions)
        total_chunks = sum(len(s.chunks) for s in self.sessions.values())
        total_tokens = sum(s.total_tokens for s in self.sessions.values())
        active_documents = set()
        for s in self.sessions.values():
            active_documents.update(s.active_documents)

        return {
            'total_sessions': total_sessions,
            'total_chunks': total_chunks,
            'total_tokens': total_tokens,
            'total_documents': len(active_documents),
            'avg_tokens_per_session': total_tokens / max(1, total_sessions),
            'avg_chunks_per_session': total_chunks / max(1, total_sessions),
            'duplicate_hash_groups': len(self.chunk_hashes)
        }


# Factory function for easy instantiation
def create_context_manager(max_tokens: int = 16000) -> MultiDocumentContextManager:
    """Create a configured context manager"""
    return MultiDocumentContextManager(
        max_total_tokens=max_tokens,
        session_max_tokens=min(max_tokens // 2, 8000),
        sliding_window=True
    )
