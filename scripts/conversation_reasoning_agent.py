import json
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
import ollama
from dotenv import load_dotenv
import os

# Ensure we can import from sibling modules
current_dir = Path(__file__).parent
if str(current_dir.parent) not in sys.path:
    sys.path.insert(0, str(current_dir.parent))

from scripts.conversation_retriever import ConversationRetriever
from scripts.entity_evaluator import entity_evaluator, EntityCandidate
from scripts.mcp.context_manager import create_context_manager, ContextSession

# Load environment variables
load_dotenv()

class ConversationReasoningAgent:
    def __init__(self,
                 persona_config_path: Path = Path("data/persona.json"),
                 ollama_model: str = "granite4:micro-h"):

        self.persona_config_path = persona_config_path
        self.persona_config = self._load_persona(persona_config_path)
        self.ollama_model = ollama_model
        self.retriever = ConversationRetriever(ollama_model=ollama_model)

        # Enhanced features: MCP integration and entity analysis
        self.context_manager = create_context_manager(max_tokens=16000)
        self.session_contexts: Dict[str, str] = {}  # session_id -> context_session_id

        # Multi-pass entity extraction settings
        self.multi_pass_extraction = True
        self.max_extraction_passes = 3
        self.entity_importance_threshold = 0.6

    def _load_persona(self, config_path: Path) -> Dict[str, Any]:
        """Load persona configuration with RLHF thresholds"""

        if not config_path.exists():
            # Create default persona configuration for EPS document assistant
            default_config = {
                "name": "EPS Document Research Assistant",
                "description": "A helpful assistant that analyzes EPS documents and provides insights from document collections",
                "system_prompt_template": """You are an EPS Document Research Assistant analyzing a collection of documents and text content.
You have access to a database of document content and can retrieve relevant information from EPS document collections.

When answering questions:
1. Always cite specific document filenames and content summaries when relevant
2. Be thorough and comprehensive in your analysis
3. Explain concepts based on document evidence
4. If you don't have enough information from documents, say so
5. Organize your responses with clear structure when appropriate

Available context from EPS documents:
{context}

Question: {query}""",
                "rlhf_thresholds": {
                    "retrieval_required": 0.6,
                    "minimum_context_overlap": 0.3,
                    "formality_level": 0.7,
                    "technical_detail_level": 0.7,
                    "citation_requirement": 0.8
                },
                "recent_success_rate": 0.8
            }

            config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(config_path, 'w') as f:
                json.dump(default_config, f, indent=2)

            return default_config

        # Load existing config
        with open(config_path) as f:
            return json.load(f)

    def should_retrieve_context(self, query: str) -> bool:
        """
        Decide if we need to retrieve context based on:
        1. Query complexity and conversational nature (for chat assistant)
        2. RLHF confidence threshold
        3. Recent retrieval success rate
        """

        # For a general chat assistant, we want to retrieve conversation context
        # more broadly to reference previous similar discussions

        # Always retrieve context for conversational assistant as it has memory
        # But check if it's a fact-based or general question that might benefit from context
        query_lower = query.lower()

        # Indicators that might benefit from conversation history
        conversational_indicators = [
            'how', 'what', 'why', 'when', 'where', 'who', 'which', 'can',
            'should', 'best', 'tools', 'good', 'recommended', 'advice',
            'help', 'guide', 'tips', 'ideas', 'examples', 'explain'
        ]

        needs_retrieval = any(term in query_lower.split() for term in conversational_indicators)
        needs_retrieval = needs_retrieval or len(query.split()) > 3  # Complex queries

        # For general chat assistant, lower the threshold significantly
        confidence_threshold = self.persona_config['rlhf_thresholds']['retrieval_required'] * 0.3  # Much more lenient

        # Always retrieve for chat assistant unless it's very simple
        return needs_retrieval or confidence_threshold > 0.2 or len(query.split()) > 1

    def generate_response(self,
                         query: str,
                         chat_history: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
        """
        Main orchestration logic:
        1. Decide if retrieval needed
        2. Retrieve context if necessary
        3. Generate response with persona coloring
        4. Grade output (RLHF scoring)
        5. Update persona thresholds based on grade
        """

        # Step 1: Retrieval decision
        needs_context = self.should_retrieve_context(query)

        context_docs = []
        if needs_context:
            try:
                context_docs = self.retriever.retrieve_context(query, limit=5)
            except Exception as e:
                print(f"Error retrieving context: {e}")
                context_docs = []

        # Step 2: Format context for LLM
        context_str = self._format_context(context_docs)
        print(f"[CHAT_DEBUG] Formatted context length: {len(context_str)} characters")
        if context_str:
            print(f"[CHAT_DEBUG] Context preview: {context_str[:200]}...")

        # Step 3: Generate with persona
        system_prompt = self._build_persona_prompt(context_str, context_docs, chat_history)

        print(f"[CHAT_DEBUG] Final system prompt length: {len(system_prompt)} characters")
        print(f"[CHAT_DEBUG] System prompt preview: {system_prompt[:300]}...")

        try:
            response = ollama.generate(
                model=self.ollama_model,
                prompt=query,
                system=system_prompt
            )
            print(f"[CHAT_DEBUG] LLM generated response of {len(response['response'])} characters")
            print(f"[CHAT_DEBUG] Response preview: {response['response'][:200]}...")
        except Exception as e:
            print(f"Error generating response: {e}")
            response = {"response": "I'm sorry, I encountered an error while processing your query. Please try again."}

        # Step 4: RLHF grading
        quality_grade = self._grade_response(query, response['response'], context_docs)

        # Step 5: Update RLHF thresholds based on grade
        self._update_persona_thresholds(quality_grade)

        return {
            'response': response['response'],
            'context_used': context_docs,
            'quality_grade': quality_grade,
            'retrieval_method': context_docs[0]['retrieval_method'] if context_docs else None,
            'retrieval_performed': needs_context
        }

    def _build_persona_prompt(self, context: str, context_docs: List[Dict[str, Any]], chat_history: Optional[List[Dict[str, str]]] = None) -> str:
        """
        Build system prompt from persona configuration.
        This is the 'coloring' step mentioned in the architecture.
        """
        base_template = self.persona_config['system_prompt_template']

        # Insert context if available
        if context:
            base_template = base_template.replace("{context}", context)
        else:
            base_template = base_template.replace("{context}", "No specific document context available.")

        # Insert query placeholder (will be replaced by ollama)
        if "{query}" not in base_template:
            base_template += "\n\nQuestion: {query}"

        # Include chat history if available
        if chat_history:
            formatted_history = self._format_chat_history(chat_history)
            if formatted_history:
                base_template += f"\n\nPrevious conversation:\n{formatted_history}\n\nPlease continue this conversation naturally."

        # Add persona modifiers based on RLHF values
        formality = self.persona_config['rlhf_thresholds']['formality_level']
        if formality > 0.7:
            base_template += "\n\nUse formal, analytical language when discussing documents."
        elif formality < 0.4:
            base_template += "\n\nUse conversational language when summarizing document content."

        technical_detail = self.persona_config['rlhf_thresholds']['technical_detail_level']
        if technical_detail > 0.8:
            base_template += "\n\nInclude detailed content analysis and cross-references when relevant."
        elif technical_detail < 0.5:
            base_template += "\n\nFocus on providing clear summaries of document content."

        citation_req = self.persona_config['rlhf_thresholds']['citation_requirement']
        if citation_req > 0.8:
            base_template += "\n\nALWAYS cite specific document filenames and provide context for claims."
        elif citation_req < 0.5:
            base_template += "\n\nYou can provide general summaries without requiring specific citations."

        return base_template

    def _format_context(self, context_docs: List[Dict[str, Any]]) -> str:
        """Format retrieved EPS documents for context"""
        if not context_docs:
            return ""

        formatted = []
        for i, doc in enumerate(context_docs, 1):
            doc_info = f"""
EPS Document {i}:
Filename: {doc['filename']}
Type: {doc.get('document_type', 'Unknown')}
Summary: {doc.get('summary', 'No summary available')}
Content: {doc.get('content_preview', doc.get('content', ''))[:400]}
Topics: {', '.join(doc.get('topics', [])[:3])}
Entities: {', '.join(doc.get('entities', [])[:3])}
Retrieval Method: {doc.get('retrieval_method', 'unknown')}
"""
            formatted.append(doc_info.strip())

        return "\n\n".join(formatted)

    def _format_chat_history(self, chat_history: Optional[List[Dict[str, str]]]) -> Optional[str]:
        """Format chat history for inclusion in system prompt"""
        if not chat_history:
            return None

        # Only keep last few exchanges to avoid context overflow
        recent_history = chat_history[-6:]  # Last 3 user-assistant pairs

        formatted = []
        for msg in recent_history:
            role_prefix = "User: " if msg.get('role') == 'user' else "Assistant: "
            formatted.append(f"{role_prefix}{msg.get('content', '')}")

        return "\n".join(formatted)

    def _grade_response(self, query: str, response: str, context: List[Dict[str, Any]]) -> float:
        """
        RLHF grading: 0 (needs improvement) to 1 (excellent).
        Heuristic-based grading for document analysis.
        """

        if not response or len(response.strip()) < 10:
            return 0.1  # Too short or empty

        # Check for document insights vs available context
        insights_score = self._evaluate_document_insights(response, context)
        completeness_score = min(1.0, len(response.split()) / 150)  # Length appropriateness
        structure_score = self._evaluate_structure(response)

        # Weighted score
        overall_score = (
            0.5 * insights_score +
            0.3 * completeness_score +
            0.2 * structure_score
        )

        return min(1.0, max(0.0, overall_score))

    def _evaluate_document_insights(self, response: str, context: List[Dict[str, Any]]) -> float:
        """Check if response provides insights about documents"""

        if not context:
            return 0.3  # Some baseline if no context needed

        response_lower = response.lower()
        insights_supported = 0
        total_insights = 0

        # Check for mention of documents from context
        document_filenames = [doc['filename'].lower() for doc in context]

        mentioned_docs = sum(1 for filename in document_filenames if filename.lower() in response_lower)

        # Bonus for document analysis language
        has_analysis_terms = any(pattern in response_lower for pattern in [
            'according to the document', 'the document states', 'as shown in', 'based on the content',
            'the summary shows', 'document analysis', 'content review'
        ])

        base_score = min(0.7, mentioned_docs * 0.3)  # Up to 0.7 for document mentions
        analysis_bonus = 0.3 if has_analysis_terms else 0.0

        return min(1.0, base_score + analysis_bonus)

    def _evaluate_structure(self, response: str) -> float:
        """Evaluate response structure and readability"""

        score = 0.5  # Base score

        # Check for paragraphs (good structure)
        paragraphs = response.split('\n\n')
        if len(paragraphs) > 1:
            score += 0.2

        # Check for lists or numbered items
        has_lists = any(line.strip().startswith(('- ', '• ', '1. ', '2. ')) for line in response.split('\n'))
        if has_lists:
            score += 0.1

        # Reasonable length for document analysis
        word_count = len(response.split())
        if 30 <= word_count <= 500:
            score += 0.2

        return min(1.0, score)

    def _update_persona_thresholds(self, quality_grade: float):
        """
        Update RLHF thresholds based on response quality.
        This is the adaptive learning mechanism.
        """

        # If grade < 0.5, we need more context and formality
        if quality_grade < 0.5:
            self.persona_config['rlhf_thresholds']['retrieval_required'] += 0.05
            self.persona_config['rlhf_thresholds']['citation_requirement'] += 0.05
            self.persona_config['rlhf_thresholds']['technical_detail_level'] -= 0.02
            print("⚠️  Low quality response - increasing retrieval aggressiveness")

        # If grade > 0.8, we can be more flexible
        elif quality_grade > 0.8:
            self.persona_config['rlhf_thresholds']['retrieval_required'] -= 0.02
            self.persona_config['rlhf_thresholds']['formality_level'] -= 0.01
            print("✓ High quality response - relaxing thresholds slightly")

        # Update success rate (exponential moving average)
        alpha = 0.1
        self.persona_config['recent_success_rate'] = (
            alpha * (1.0 if quality_grade > 0.6 else 0.0) +
            (1 - alpha) * self.persona_config['recent_success_rate']
        )

        # Clamp values
        thresholds = self.persona_config['rlhf_thresholds']
        for key in thresholds:
            thresholds[key] = max(0.0, min(1.0, thresholds[key]))

        # Save updated config
        with open(self.persona_config_path, 'w') as f:
            json.dump(self.persona_config, f, indent=2)

    def _perform_multi_pass_entity_extraction(self, context_docs: List[Dict[str, Any]], query: str) -> Dict[str, Any]:
        """Perform multi-pass entity extraction with MCP integration"""

        all_entities = []
        topic_hierarchy = {}
        entity_cooccurrences = {}

        # Pass 1: Extract entities from document content
        for doc in context_docs:
            content = doc.get('content', doc.get('content_preview', ''))
            if not content:
                continue

            entities = entity_evaluator.extract_entities_from_text(content, doc['filename'])
            all_entities.extend(entities)

            # Add to MCP context manager
            context_session_id = self.session_contexts.get('default', 'default')
            if context_session_id not in self.context_manager.sessions:
                context_session_id = self.context_manager.create_session(context_session_id)
                self.session_contexts['default'] = context_session_id

            # Add document content to context manager
            self.context_manager.add_document_content(
                session_id=context_session_id,
                document_name=doc['filename'],
                content=content,
                entities=[e.name for e in entities],
                topics=doc.get('topics', [])
            )

        # Pass 2: Hierarchical topic extraction
        topic_hierarchy = self._extract_topic_hierarchy(context_docs, query)
        entity_cooccurrences = self._analyze_entity_cooccurrences(all_entities, context_docs)

        # Pass 3: Entity importance scoring
        entity_importance_scores = self._calculate_entity_importance_scores(
            all_entities, entity_cooccurrences, topic_hierarchy, query
        )

        return {
            'entities': all_entities,
            'topic_hierarchy': topic_hierarchy,
            'entity_cooccurrences': entity_cooccurrences,
            'entity_importance': entity_importance_scores,
            'mcp_session_id': self.session_contexts.get('default')
        }

    def _extract_topic_hierarchy(self, context_docs: List[Dict[str, Any]], query: str) -> Dict[str, Any]:
        """Extract hierarchical topics (parent/child relationships)"""

        all_topics = set()
        for doc in context_docs:
            all_topics.update(doc.get('topics', []))

        # Use LLM to establish topic hierarchy
        if not all_topics:
            return {}

        try:
            import ollama

            topic_list = list(all_topics)
            prompt = f"""Organize these topics into a hierarchical structure based on their relationships.
Return JSON with parent/child relationships.

Topics: {', '.join(topic_list)}

Query context: {query}

Return format:
{{
  "hierarchy": {{
    "parent_topic": ["child_topic1", "child_topic2"],
    "another_parent": ["child_topic"]
  }},
  "relationships": [
    {{"parent": "topic1", "child": "topic2", "type": "subtopic|specialization|related"}}
  ]
}}"""

            response = ollama.generate(
                model=self.ollama_model,
                prompt=prompt,
                format="json"
            )

            result = json.loads(response['response'])
            hierarchy = result.get('hierarchy', {})

            # Validate and ensure all topics are present
            for topic in topic_list:
                if topic not in hierarchy and not any(topic in children for children in hierarchy.values()):
                    hierarchy[topic] = []  # Make it a root topic

            return hierarchy

        except Exception as e:
            print(f"Topic hierarchy extraction failed: {e}")
            # Return flat hierarchy
            return {topic: [] for topic in all_topics}

    def _analyze_entity_cooccurrences(self, entities: List[EntityCandidate], context_docs: List[Dict[str, Any]]) -> Dict[str, int]:
        """Analyze which entities co-occur in the same documents/contexts"""

        cooccurrences = {}

        # Group entities by document
        doc_entities = {}
        for doc in context_docs:
            doc_entities[doc['filename']] = []
            for entity in entities:
                # Check if entity appears in this document
                entity_mentions = [m for m in entity.mentions if m.get('source') == doc['filename']]
                if entity_mentions:
                    doc_entities[doc['filename']].append(entity.name)

        # Count co-occurrences
        for doc_name, doc_entity_list in doc_entities.items():
            for i, entity1 in enumerate(doc_entity_list):
                for entity2 in doc_entity_list[i+1:]:
                    key = f"{min(entity1, entity2)}:{max(entity1, entity2)}"
                    cooccurrences[key] = cooccurrences.get(key, 0) + 1

        return cooccurrences

    def _calculate_entity_importance_scores(self, entities: List[EntityCandidate],
                                          cooccurrences: Dict[str, int],
                                          topic_hierarchy: Dict[str, Any],
                                          query: str) -> Dict[str, float]:
        """Calculate importance scores for entities"""

        importance_scores = {}

        for entity in entities:
            # Base importance factors
            confidence_weight = entity.confidence
            mention_count = len(entity.mentions)
            cooccurrence_count = sum(1 for key, count in cooccurrences.items()
                                    if entity.name in key)

            # Query relevance
            query_relevance = self._calculate_query_relevance(entity.name, query)

            # Combine factors
            importance_score = (
                confidence_weight * 0.3 +
                min(1.0, mention_count / 5.0) * 0.3 +  # Cap at 5 mentions
                min(1.0, cooccurrence_count / 3.0) * 0.2 +  # Cap at 3 cooccurrences
                query_relevance * 0.2
            )

            importance_scores[entity.name] = min(1.0, importance_score)

        return importance_scores

    def _calculate_query_relevance(self, entity_name: str, query: str) -> float:
        """Calculate how relevant an entity is to the query"""
        entity_lower = entity_name.lower()
        query_lower = query.lower()

        # Exact mention gets highest score
        if entity_lower in query_lower:
            return 1.0

        # Fuzzy matching
        from difflib import SequenceMatcher
        similarity = SequenceMatcher(None, entity_lower, query_lower).ratio()
        return similarity if similarity > 0.6 else 0.0

    def generate_enhanced_response(self,
                                  query: str,
                                  chat_history: Optional[List[Dict[str, str]]] = None,
                                  session_id: str = "default") -> Dict[str, Any]:
        """
        Enhanced response generation with multi-pass entity extraction and MCP integration
        """

        # Get or create session context
        if session_id not in self.session_contexts:
            context_session_id = self.context_manager.create_session(f"enhanced_{session_id}")
            self.session_contexts[session_id] = context_session_id

        context_session_id = self.session_contexts[session_id]

        # Step 1: Retrieve context
        needs_context = self.should_retrieve_context(query)
        context_docs = []
        if needs_context:
            try:
                context_docs = self.retriever.retrieve_context(query, limit=5)
            except Exception as e:
                print(f"Error retrieving context: {e}")
                context_docs = []

        # Step 2: Multi-pass entity extraction (if enabled)
        entity_analysis = {}
        if self.multi_pass_extraction and context_docs:
            entity_analysis = self._perform_multi_pass_entity_extraction(context_docs, query)

        # Step 3: Build enhanced system prompt with entity context
        system_prompt = self._build_enhanced_persona_prompt(
            context_docs, query, entity_analysis, chat_history, context_session_id
        )

        # Step 4: Generate response
        try:
            response = ollama.generate(
                model=self.ollama_model,
                prompt=query,
                system=system_prompt
            )
            response_text = response['response']
        except Exception as e:
            print(f"Error generating response: {e}")
            response_text = "I'm sorry, I encountered an error while processing your query. Please try again."

        # Step 5: Enhanced grading with entity analysis
        quality_grade = self._grade_enhanced_response(query, response_text, context_docs, entity_analysis)

        return {
            'response': response_text,
            'context_used': context_docs,
            'quality_grade': quality_grade,
            'retrieval_method': context_docs[0]['retrieval_method'] if context_docs else None,
            'retrieval_performed': needs_context,
            'entity_analysis': entity_analysis,
            'mcp_session_id': context_session_id
        }

    def _build_enhanced_persona_prompt(self, context_docs: List[Dict[str, Any]], query: str,
                                      entity_analysis: Dict[str, Any], chat_history: Optional[List[Dict[str, str]]] = None,
                                      context_session_id: str = None) -> str:
        """Build enhanced system prompt with entity and topic context"""

        base_template = self.persona_config['system_prompt_template']

        # Get enhanced context from MCP
        enhanced_context = ""
        if context_session_id and context_session_id in self.context_manager.sessions:
            enhanced_context = self.context_manager.get_session_context(context_session_id)
        else:
            enhanced_context = self._format_context(context_docs)

        # Add entity insights
        entity_context = ""
        if entity_analysis:
            entities = entity_analysis.get('entities', [])
            importance = entity_analysis.get('entity_importance', {})

            # Get top entities by importance
            top_entities = sorted(
                [(name, score) for name, score in importance.items() if score >= self.entity_importance_threshold],
                key=lambda x: x[1], reverse=True
            )[:5]  # Top 5

            if top_entities:
                entity_context = "\n\n**Key Entities Mentioned:**\n" + "\n".join([
                    f"- **{name}** (importance: {score:.2f})" for name, score in top_entities
                ])

            # Add topic hierarchy
            topic_hierarchy = entity_analysis.get('topic_hierarchy', {})
            if topic_hierarchy:
                entity_context += "\n\n**Topic Structure:**\n" + "\n".join([
                    f"- **{parent}**: {', '.join(children[:3])}{'...' if len(children) > 3 else ''}"
                    for parent, children in list(topic_hierarchy.items())[:3]
                ])

        # Insert enhanced context
        if enhanced_context or entity_context:
            full_context = enhanced_context + entity_context
            base_template = base_template.replace("{context}", full_context)
        else:
            base_template = base_template.replace("{context}", "No specific document context available.")

        # Add query placeholder
        if "{query}" not in base_template:
            base_template += "\n\nQuestion: {query}"

        # Include chat history
        if chat_history:
            formatted_history = self._format_chat_history(chat_history)
            if formatted_history:
                base_template += f"\n\nPrevious conversation:\n{formatted_history}\n\nPlease continue this conversation naturally."

        # Add entity-aware guidance
        if entity_analysis and entity_analysis.get('entities'):
            base_template += "\n\nWhen answering, consider mentioning relevant entities and their relationships when they provide context to the query."

        return base_template

    def _grade_enhanced_response(self, query: str, response: str, context_docs: List[Dict[str, Any]],
                                entity_analysis: Dict[str, Any]) -> float:
        """Enhanced grading that considers entity usage and topic coverage"""

        base_score = self._grade_response(query, response, context_docs)

        # Entity utilization bonus
        entity_bonus = 0.0
        if entity_analysis:
            entities = entity_analysis.get('entities', [])
            importance = entity_analysis.get('entity_importance', {})

            # Check if response mentions important entities
            response_lower = response.lower()
            mentioned_important = 0
            total_important = 0

            for entity_name, imp_score in importance.items():
                if imp_score >= self.entity_importance_threshold:
                    total_important += 1
                    if entity_name.lower() in response_lower:
                        mentioned_important += 1

            if total_important > 0:
                entity_bonus = (mentioned_important / total_important) * 0.2

        # Topic coverage bonus
        topic_bonus = 0.0
        if entity_analysis:
            topics = entity_analysis.get('topic_hierarchy', {})
            response_lower = response.lower()

            mentioned_topics = sum(
                1 for topic in topics.keys()
                if topic.lower() in response_lower
            )
            total_topics = len(topics)

            if total_topics > 0:
                topic_bonus = (mentioned_topics / total_topics) * 0.1

        enhanced_score = min(1.0, base_score + entity_bonus + topic_bonus)

        return enhanced_score

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test EPS reasoning agent")
    parser.add_argument("--query", type=str, help="Query to test")
    parser.add_argument("--history", type=str, help="JSON chat history")

    args = parser.parse_args()

    agent = EPSReasoningAgent()

    if args.query:
        chat_history = []
        if args.history:
            try:
                chat_history = json.loads(args.history)
            except:
                print("Invalid history JSON")

        result = agent.generate_response(args.query, chat_history)

        print(f"Query: {args.query}")
        print(f"Retrieved context: {len(result['context_used'])} documents")
        print(f"Quality grade: {result['quality_grade']:.2f}")
        print(f"Retrieval method: {result.get('retrieval_method', 'none')}")
        print()
        print("Response:")
        print(result['response'])
        print()
        if result['context_used']:
            print("EPS Document Sources:")
            for i, doc in enumerate(result['context_used'], 1):
                print(f"{i}. {doc['filename']} ({doc.get('document_type', 'unknown')})")
                if doc.get('summary'):
                    print(f"   Summary: {doc['summary'][:100]}...")
                print(f"   Relevance: {doc.get('relevance_score', 0):.3f}, Method: {doc.get('retrieval_method', 'unknown')}")
                print()
    else:
        print("Provide a query with --query")
