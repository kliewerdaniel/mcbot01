"""
MCP Entity Extraction Prompt Templates.

Provides structured prompt templates for entity extraction from various document types,
including few-shot examples and output schemas for consistent MCP tool responses.
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from datetime import datetime

@dataclass
class PromptTemplate:
    """Represents a prompt template with metadata"""
    name: str
    description: str
    template: str
    input_schema: Dict[str, Any]
    output_schema: Dict[str, Any]
    examples: List[Dict[str, Any]]
    created_at: datetime = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()

class MCPPromptManager:
    """Manages MCP prompt templates for entity extraction and related tasks"""

    def __init__(self):
        self.templates: Dict[str, PromptTemplate] = {}
        self._initialize_templates()

    def _initialize_templates(self):
        """Initialize all available prompt templates"""
        self.templates.update({
            'entity_extraction_general': self._create_general_entity_extraction_prompt(),
            'entity_extraction_research': self._create_research_entity_extraction_prompt(),
            'entity_extraction_business': self._create_business_entity_extraction_prompt(),
            'relationship_inference': self._create_relationship_inference_prompt(),
            'topic_hierarchy': self._create_topic_hierarchy_prompt(),
            'entity_deduplication': self._create_entity_deduplication_prompt(),
            'confidence_scoring': self._create_confidence_scoring_prompt(),
            'graph_construction': self._create_graph_construction_prompt(),
        })

    def get_template(self, name: str) -> Optional[PromptTemplate]:
        """Get a specific prompt template"""
        return self.templates.get(name)

    def list_templates(self) -> List[str]:
        """List all available template names"""
        return list(self.templates.keys())

    def format_prompt(self, template_name: str, **kwargs) -> str:
        """Format a template with provided variables"""
        template = self.templates.get(template_name)
        if not template:
            raise ValueError(f"Template '{template_name}' not found")

        try:
            return template.template.format(**kwargs)
        except KeyError as e:
            raise ValueError(f"Missing required variable for template {template_name}: {e}")

    def validate_input(self, template_name: str, input_data: Dict[str, Any]) -> bool:
        """Validate input data against template schema"""
        template = self.templates.get(template_name)
        if not template:
            return False

        # Basic validation - check required fields
        required_fields = []
        for field_name, field_spec in template.input_schema.get('properties', {}).items():
            if field_spec.get('required', False):
                required_fields.append(field_name)

        return all(field in input_data for field in required_fields)

    def _create_general_entity_extraction_prompt(self) -> PromptTemplate:
        """Create general entity extraction prompt"""
        template = """Extract named entities from the following text content.

Focus on identifying these entity types:
- **Person**: Individual people, researchers, authors, contributors
- **Organization**: Companies, institutions, universities, research groups
- **Technology**: Software tools, frameworks, programming languages, algorithms, systems
- **Concept**: Key theoretical concepts, methodologies, techniques, paradigms
- **Location**: Geographic locations, institutions, facilities, cities

Document Content:
{content}

Extraction Guidelines:
1. Extract only entities that appear in the text
2. Assign appropriate entity types based on context
3. Provide brief descriptions when possible
4. Include confidence scores (0.0-1.0) for each extraction
5. List specific mentions with position context

{examples}

Return JSON response:
{{
  "entities": [
    {{
      "name": "Entity Name",
      "type": "Entity_Type",
      "description": "Brief description of entity's role or context",
      "confidence": 0.0-1.0,
      "mentions": ["position1", "position2"],
      "context_snippet": "surrounding text context"
    }}
  ],
  "document_type": "{document_type}",
  "extraction_quality": 0.0-1.0,
  "processing_notes": ["any relevant observations"]
}}"""

        examples = [
            {
                "input": "Dr. Sarah Johnson from MIT developed a new algorithm for neural networks.",
                "output": {
                    "entities": [
                        {
                            "name": "Dr. Sarah Johnson",
                            "type": "Person",
                            "description": "Researcher who developed a neural network algorithm",
                            "confidence": 0.95,
                            "mentions": ["Dr. Sarah Johnson"],
                            "context_snippet": "Dr. Sarah Johnson from MIT developed"
                        },
                        {
                            "name": "MIT",
                            "type": "Organization",
                            "description": "Massachusetts Institute of Technology",
                            "confidence": 0.98,
                            "mentions": ["MIT"],
                            "context_snippet": "from MIT developed"
                        }
                    ]
                }
            }
        ]

        return PromptTemplate(
            name='entity_extraction_general',
            description='General purpose entity extraction for various document types',
            template=template,
            input_schema={
                'type': 'object',
                'properties': {
                    'content': {'type': 'string', 'description': 'Text content to extract entities from'},
                    'document_type': {'type': 'string', 'description': 'Type of document (pdf, research_paper, etc.)'},
                    'examples': {'type': 'string', 'description': 'Few-shot examples (optional)'}
                },
                'required': ['content']
            },
            output_schema={
                'type': 'object',
                'properties': {
                    'entities': {
                        'type': 'array',
                        'items': {
                            'type': 'object',
                            'properties': {
                                'name': {'type': 'string'},
                                'type': {'type': 'string', 'enum': ['Person', 'Organization', 'Technology', 'Concept', 'Location']},
                                'description': {'type': 'string'},
                                'confidence': {'type': 'number', 'minimum': 0.0, 'maximum': 1.0},
                                'mentions': {'type': 'array', 'items': {'type': 'string'}},
                                'context_snippet': {'type': 'string'}
                            }
                        }
                    },
                    'document_type': {'type': 'string'},
                    'extraction_quality': {'type': 'number'},
                    'processing_notes': {'type': 'array', 'items': {'type': 'string'}}
                }
            },
            examples=examples
        )

    def _create_research_entity_extraction_prompt(self) -> PromptTemplate:
        """Create research paper specific entity extraction prompt"""
        template = """Extract entities from this research/academic document.

Academic Entity Types:
- **Person**: Authors, researchers, cited scholars, reviewers
- **Organization**: Universities, research institutions, companies, conferences
- **Technology**: Algorithms, models, frameworks, datasets, tools, software
- **Concept**: Theories, methods, techniques, hypotheses, findings
- **Location**: Research institutions with geographic significance

Research Document Context:
- Citations often indicate Person or Organization entities
- Method sections describe Technology and Concept entities
- Affiliation information helps classify researchers and institutions

Document Content:
{content}

Specialized Extraction Rules:
1. Extract author names from citations and bylines
2. Identify affiliations and institutional connections
3. Recognize technical terminology as Technology/Concept entities
4. Extract dataset and model names as Technology entities
5. Identify conference and journal names as Organization entities

{examples}

Return JSON response:
{{
  "entities": [
    {{
      "name": "Entity Name",
      "type": "Entity_Type",
      "description": "Academic/research context description",
      "confidence": 0.8,
      "citations": ["paper references"],
      "affiliation": "institutional context",
      "research_area": "field of study"
    }}
  ],
  "academic_metadata": {{
    "field_of_study": "detected research area",
    "key_contributions": ["extracted contributions"],
    "methodologies_used": ["identified methods"]
  }}
}}"""

        examples = [
            {
                "input": "In their seminal paper, Vaswani et al. (2017) introduced the Transformer architecture for neural machine translation.",
                "output": {
                    "entities": [
                        {
                            "name": "Vaswani et al.",
                            "type": "Person",
                            "description": "Authors of the Transformer paper",
                            "confidence": 0.9,
                            "citations": ["Vaswani et al. (2017)"],
                            "research_area": "Natural Language Processing"
                        },
                        {
                            "name": "Transformer",
                            "type": "Technology",
                            "description": "Neural network architecture for sequence modeling",
                            "confidence": 0.95,
                            "citations": ["Transformer architecture"],
                            "research_area": "Deep Learning"
                        }
                    ],
                    "academic_metadata": {
                        "field_of_study": "Natural Language Processing",
                        "key_contributions": ["Transformer architecture"],
                        "methodologies_used": ["Neural machine translation"]
                    }
                }
            }
        ]

        return PromptTemplate(
            name='entity_extraction_research',
            description='Specialized entity extraction for research papers and academic documents',
            template=template,
            input_schema={
                'type': 'object',
                'properties': {
                    'content': {'type': 'string', 'description': 'Research document content'},
                    'examples': {'type': 'string', 'description': 'Few-shot examples (optional)'}
                },
                'required': ['content']
            },
            output_schema={
                'type': 'object',
                'properties': {
                    'entities': {'type': 'array'},
                    'academic_metadata': {
                        'type': 'object',
                        'properties': {
                            'field_of_study': {'type': 'string'},
                            'key_contributions': {'type': 'array'},
                            'methodologies_used': {'type': 'array'}
                        }
                    }
                }
            },
            examples=examples
        )

    def _create_business_entity_extraction_prompt(self) -> PromptTemplate:
        """Create business document specific entity extraction prompt"""
        template = """Extract business-relevant entities from the document.

Business Entity Types:
- **Person**: Executives, employees, stakeholders, customers, partners
- **Organization**: Companies, departments, subsidiaries, competitors, clients
- **Technology**: Products, services, platforms, tools, systems, applications
- **Concept**: Business processes, methodologies, strategies, KPIs
- **Location**: Office locations, markets, regions, facilities

Business Document Context:
- Focus on operational and strategic business entities
- Extract product and service names as Technology entities
- Identify organizational hierarchies and relationships
- Capture customer and partner information where relevant

Document Content:
{content}

Business Extraction Guidelines:
1. Extract company names and organizational entities
2. Identify products, services, and technology solutions
3. Capture executive and stakeholder names with roles
4. Extract business process and methodology names
5. Include market and geographic business contexts

{examples}

Return JSON response:
{{
  "entities": [
    {{
      "name": "Entity Name",
      "type": "Entity_Type",
      "description": "Business context description",
      "confidence": 0.8,
      "business_role": "executive/client/product etc.",
      "industry": "business sector"
    }}
  ],
  "business_metadata": {{
    "industry_sector": "detected business area",
    "key_products": ["extracted products/services"],
    "organizational_scope": "company/department/team level"
  }}
}}"""

        examples = [
            {
                "input": "Microsoft's CEO Satya Nadella announced the launch of Azure OpenAI Service, expanding AI capabilities for enterprise customers.",
                "output": {
                    "entities": [
                        {
                            "name": "Microsoft",
                            "type": "Organization",
                            "description": "Technology company offering AI services",
                            "confidence": 0.98,
                            "business_role": "parent_company",
                            "industry": "Technology"
                        },
                        {
                            "name": "Satya Nadella",
                            "type": "Person",
                            "description": "CEO of Microsoft",
                            "confidence": 0.95,
                            "business_role": "executive",
                            "industry": "Technology"
                        },
                        {
                            "name": "Azure OpenAI Service",
                            "type": "Technology",
                            "description": "AI platform service offering",
                            "confidence": 0.92,
                            "business_role": "product",
                            "industry": "Artificial Intelligence"
                        }
                    ],
                    "business_metadata": {
                        "industry_sector": "Enterprise Software",
                        "key_products": ["Azure OpenAI Service"],
                        "organizational_scope": "Enterprise Level"
                    }
                }
            }
        ]

        return PromptTemplate(
            name='entity_extraction_business',
            description='Business-focused entity extraction for corporate documents',
            template=template,
            input_schema={
                'type': 'object',
                'properties': {
                    'content': {'type': 'string', 'description': 'Business document content'},
                    'examples': {'type': 'string', 'description': 'Few-shot examples (optional)'}
                },
                'required': ['content']
            },
            output_schema={
                'type': 'object',
                'properties': {
                    'entities': {'type': 'array'},
                    'business_metadata': {
                        'type': 'object',
                        'properties': {
                            'industry_sector': {'type': 'string'},
                            'key_products': {'type': 'array'},
                            'organizational_scope': {'type': 'string'}
                        }
                    }
                }
            },
            examples=examples
        )

    def _create_relationship_inference_prompt(self) -> PromptTemplate:
        """Create relationship inference prompt"""
        template = """Analyze relationships between the provided entities based on document context.

Entities to analyze:
{entity_list}

Document Context:
{context}

Identify Relationships:
- RELATED_TO: General thematic connections
- WORKS_FOR: Employment/affiliation relationships
- USES/IMPLEMENTS: Technology usage or implementation
- CREATED/DEVELOPED: Creation or development relationships
- LOCATED_IN: Geographic relationships
- PART_OF/CONTAINS: Hierarchical relationships
- COMPETES_WITH: Competitive relationships

Relationship Analysis Guidelines:
1. Base relationships on evidence from the context
2. Assign confidence scores based on strength of evidence
3. Include directionality (entity1 -> relationship -> entity2)
4. Provide brief explanation for each relationship
5. Consider both explicit and implicit relationships

{examples}

Return JSON response:
{{
  "relationships": [
    {{
      "entity1": "First Entity Name",
      "entity2": "Second Entity Name",
      "type": "RELATIONSHIP_TYPE",
      "confidence": 0.0-1.0,
      "direction": "undirected/directed",
      "evidence": "text snippet supporting relationship",
      "contextual_strength": 0.0-1.0
    }}
  ],
  "relationship_network": {{
    "density": 0.0-1.0,
    "avg_confidence": 0.0-1.0,
    "key_clusters": ["identified entity clusters"]
  }}
}}"""

        examples = [
            {
                "input": "Dr. Smith from Stanford University developed the TensorFlow library while working at Google Brain.",
                "output": {
                    "relationships": [
                        {
                            "entity1": "Dr. Smith",
                            "entity2": "Stanford University",
                            "type": "WORKS_FOR",
                            "confidence": 0.9,
                            "direction": "undirected",
                            "evidence": "Dr. Smith from Stanford University",
                            "contextual_strength": 0.85
                        },
                        {
                            "entity1": "TensorFlow",
                            "entity2": "Dr. Smith",
                            "type": "CREATED",
                            "confidence": 0.95,
                            "direction": "directed",
                            "evidence": "developed the TensorFlow library",
                            "contextual_strength": 0.92
                        }
                    ]
                }
            }
        ]

        return PromptTemplate(
            name='relationship_inference',
            description='Infer relationships between entities from document context',
            template=template,
            input_schema={
                'type': 'object',
                'properties': {
                    'entity_list': {'type': 'string', 'description': 'Comma-separated list of entity names'},
                    'context': {'type': 'string', 'description': 'Document context for analysis'},
                    'examples': {'type': 'string', 'description': 'Few-shot examples (optional)'}
                },
                'required': ['entity_list', 'context']
            },
            output_schema={
                'type': 'object',
                'properties': {
                    'relationships': {'type': 'array'},
                    'relationship_network': {
                        'type': 'object',
                        'properties': {
                            'density': {'type': 'number'},
                            'avg_confidence': {'type': 'number'},
                            'key_clusters': {'type': 'array'}
                        }
                    }
                }
            },
            examples=examples
        )

    def _create_topic_hierarchy_prompt(self) -> PromptTemplate:
        """Create topic hierarchy extraction prompt"""
        template = """Organize the following topics into a hierarchical structure.

Topics to organize:
{topic_list}

Document Context:
{context}

Hierarchy Guidelines:
1. Group related topics under parent categories
2. Create logical parent/child relationships
3. Ensure topics don't appear multiple times in hierarchy
4. Use clear, descriptive parent category names
5. Consider semantic relationships between topics

{examples}

Return JSON response:
{{
  "hierarchy": {{
    "parent_topic_1": ["child_topic_1", "child_topic_2"],
    "parent_topic_2": ["child_topic_3"]
  }},
  "orphan_topics": ["topics not fitting hierarchy"],
  "hierarchy_quality": {{
    "coverage": 0.0-1.0,
    "depth": "shallow/medium/deep",
    "balance_score": 0.0-1.0
  }}
}}"""

        return PromptTemplate(
            name='topic_hierarchy',
            description='Create hierarchical organization of document topics',
            template=template,
            input_schema={
                'type': 'object',
                'properties': {
                    'topic_list': {'type': 'string', 'description': 'Comma-separated list of topics'},
                    'context': {'type': 'string', 'description': 'Document context for topic understanding'}
                },
                'required': ['topic_list', 'context']
            },
            output_schema={
                'type': 'object',
                'properties': {
                    'hierarchy': {'type': 'object'},
                    'orphan_topics': {'type': 'array'},
                    'hierarchy_quality': {'type': 'object'}
                }
            },
            examples=[]
        )

    def _create_entity_deduplication_prompt(self) -> PromptTemplate:
        """Create entity deduplication prompt"""
        template = """Analyze the following entities and determine which ones refer to the same real-world entity.

Entities to deduplicate:
{entity_list}

Context Information:
{context}

Deduplication Rules:
1. Same name variations (Dr. Smith vs Smith vs D. Smith)
2. Acronyms and full names (MIT vs Massachusetts Institute of Technology)
3. Organization name variations (Inc vs Incorporated vs Corp)
4. Entity type consistency check
5. Contextual similarity analysis

{examples}

Return JSON response:
{{
  "duplicate_groups": [
    {{
      "canonical_name": "Standardized Entity Name",
      "entity_type": "Entity_Type",
      "duplicates": ["variation1", "variation2"],
      "confidence": 0.0-1.0,
      "merge_strategy": "merge/keep_separate",
      "evidence": "reason for grouping"
    }}
  ],
  "unique_entities": ["entities with no duplicates"],
  "deduplication_stats": {{
    "total_groups": 0,
    "duplicate_reduction": 0.0-1.0
  }}
}}"""

        return PromptTemplate(
            name='entity_deduplication',
            description='Identify and merge duplicate entity representations',
            template=template,
            input_schema={
                'type': 'object',
                'properties': {
                    'entity_list': {'type': 'string', 'description': 'List of entities to deduplicate'},
                    'context': {'type': 'string', 'description': 'Context for entity disambiguation'}
                },
                'required': ['entity_list']
            },
            output_schema={
                'type': 'object',
                'properties': {
                    'duplicate_groups': {'type': 'array'},
                    'unique_entities': {'type': 'array'},
                    'deduplication_stats': {'type': 'object'}
                }
            },
            examples=[]
        )

    def _create_confidence_scoring_prompt(self) -> PromptTemplate:
        """Create entity confidence scoring prompt"""
        template = """Score the confidence level for each entity extraction.

Entities to score:
{entity_list}

Context:
{context}

Confidence Factors:
1. **Context Clarity**: How clearly the entity is presented
2. **Entity Specificity**: How specific/unique the entity name is
3. **Type Consistency**: How well the entity fits its assigned type
4. **Contextual Importance**: How central the entity is to the content
5. **Source Reliability**: Quality of the source context

{examples}

Return JSON response:
{{
  "confidence_scores": [
    {{
      "entity_name": "Entity Name",
      "original_confidence": 0.0-1.0,
      "adjusted_confidence": 0.0-1.0,
      "confidence_factors": {{
        "context_clarity": 0.0-1.0,
        "specificity": 0.0-1.0,
        "type_consistency": 0.0-1.0,
        "importance": 0.0-1.0,
        "source_quality": 0.0-1.0
      }},
      "should_keep": true/false,
      "confidence_reason": "brief explanation"
    }}
  ],
  "overall_quality": {{
    "avg_confidence": 0.0-1.0,
    "high_confidence_count": 0,
    "filtered_entities": 0
  }}
}}"""

        return PromptTemplate(
            name='confidence_scoring',
            description='Detailed confidence scoring for entity extractions',
            template=template,
            input_schema={
                'type': 'object',
                'properties': {
                    'entity_list': {'type': 'string', 'description': 'Entities to score'},
                    'context': {'type': 'string', 'description': 'Source context'}
                },
                'required': ['entity_list', 'context']
            },
            output_schema={
                'type': 'object',
                'properties': {
                    'confidence_scores': {'type': 'array'},
                    'overall_quality': {'type': 'object'}
                }
            },
            examples=[]
        )

    def _create_graph_construction_prompt(self) -> PromptTemplate:
        """Create graph construction guidance prompt"""
        template = """Generate Cypher queries to construct a knowledge graph from entities and relationships.

Entities:
{entities_json}

Relationships:
{relationships_json}

Document Metadata:
{document_info}

Graph Construction Requirements:
1. Use MERGE operations to avoid duplicate nodes/relationships
2. Include all relevant properties for nodes and edges
3. Create appropriate indexes for performance
4. Handle relationship directionality correctly
5. Include metadata timestamps and source information

Optimization Guidelines:
- Create nodes before relationships
- Use batch operations where possible
- Include error handling for constraints
- Add graph schema validation

{examples}

Return JSON response:
{{
  "cypher_queries": [
    {{
      "description": "What this query does",
      "query": "MERGE (n:Label {property: $value}) RETURN n",
      "parameters": {{"key": "value"}},
      "execution_order": 1
    }}
  ],
  "graph_schema": {{
    "node_labels": ["Label1", "Label2"],
    "relationship_types": ["REL_TYPE"],
    "constraints": ["unique constraints"],
    "indexes": ["performance indexes"]
  }},
  "validation_queries": [
    {{
      "description": "Schema validation query",
      "query": "MATCH (n) RETURN labels(n), count(*) as count"
    }}
  ]
}}"""

        return PromptTemplate(
            name='graph_construction',
            description='Generate Cypher queries for knowledge graph construction',
            template=template,
            input_schema={
                'type': 'object',
                'properties': {
                    'entities_json': {'type': 'string', 'description': 'JSON string of entities'},
                    'relationships_json': {'type': 'string', 'description': 'JSON string of relationships'},
                    'document_info': {'type': 'string', 'description': 'Document metadata'}
                },
                'required': ['entities_json', 'relationships_json']
            },
            output_schema={
                'type': 'object',
                'properties': {
                    'cypher_queries': {'type': 'array'},
                    'graph_schema': {'type': 'object'},
                    'validation_queries': {'type': 'array'}
                }
            },
            examples=[]
        )

# Global instance
mcp_prompt_manager = MCPPromptManager()

def get_entity_extraction_prompts() -> List[str]:
    """Get list of available entity extraction prompt names"""
    return [name for name in mcp_prompt_manager.list_templates()
            if name.startswith('entity_extraction')]

def format_extraction_prompt(prompt_name: str, **kwargs) -> str:
    """Format an extraction prompt with variables"""
    return mcp_prompt_manager.format_prompt(prompt_name, **kwargs)
