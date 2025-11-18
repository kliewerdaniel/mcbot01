"""
Upload Manager for processing uploaded files and managing ingestion tasks.

Handles file validation, parsing, progress tracking, and background processing
for the dynamic file upload system in GraphRAG.
"""

import asyncio
import json
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable, Awaitable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import uuid
import hashlib
import traceback

from neo4j import GraphDatabase
import ollama
from dotenv import load_dotenv
import os

# Ensure we can import from sibling modules
current_dir = Path(__file__).parent
if str(current_dir.parent) not in sys.path:
    sys.path.insert(0, str(current_dir.parent))

from scripts.conversation_retriever import ConversationRetriever
from scripts.graph_schema import ResearchSchema

# Load environment variables
load_dotenv()

class TaskStatus(Enum):
    """Enumeration of task statuses"""
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class ProcessingStats:
    """Statistics for processing operations"""
    files_processed: int = 0
    total_files: int = 0
    entities_extracted: int = 0
    relationships_created: int = 0
    tokens_processed: int = 0
    processing_time_seconds: float = 0.0
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

@dataclass
class UploadTask:
    """Represents a single upload processing task"""
    task_id: str
    session_id: str
    status: TaskStatus = TaskStatus.QUEUED
    files: List[Dict[str, Any]] = field(default_factory=list)
    progress: float = 0.0
    current_file: Optional[str] = None
    stats: ProcessingStats = field(default_factory=ProcessingStats)
    config: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    cancelled: bool = False

    def update_progress(self, progress: float, current_file: str = None):
        """Update progress and status"""
        self.progress = min(1.0, max(0.0, progress))
        if current_file:
            self.current_file = current_file
        self.updated_at = datetime.now()

        # Auto-update status based on progress
        if self.progress >= 1.0 and self.status == TaskStatus.PROCESSING:
            self.status = TaskStatus.COMPLETED
        elif self.progress > 0 and self.status == TaskStatus.QUEUED:
            self.status = TaskStatus.PROCESSING

    def add_error(self, error: str):
        """Add error to task"""
        self.stats.errors.append(error)
        self.updated_at = datetime.now()

    def add_warning(self, warning: str):
        """Add warning to task"""
        self.stats.warnings.append(warning)
        self.updated_at = datetime.now()

    def cancel(self):
        """Cancel the task"""
        self.cancelled = True
        self.status = TaskStatus.CANCELLED
        self.updated_at = datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses"""
        return {
            "task_id": self.task_id,
            "session_id": self.session_id,
            "status": self.status.value,
            "progress": self.progress,
            "current_file": self.current_file,
            "files_processed": self.stats.files_processed,
            "total_files": self.stats.total_files,
            "entities_extracted": self.stats.entities_extracted,
            "errors": self.stats.errors,
            "eta_seconds": self.calculate_eta(),
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }

    def calculate_eta(self) -> Optional[int]:
        """Calculate estimated time to completion in seconds"""
        if self.progress <= 0 or self.status not in [TaskStatus.PROCESSING, TaskStatus.QUEUED]:
            return None

        elapsed = (self.updated_at - self.created_at).total_seconds()
        if elapsed <= 0:
            return None

        total_estimated = elapsed / self.progress
        remaining = total_estimated - elapsed

        return max(0, int(remaining))

class UploadSession:
    """Manages a file upload session and its processing tasks"""

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.upload_dir = Path(f"data/uploads/temp/{session_id}")
        self.processed_dir = Path(f"data/uploads/processed/{session_id}")
        self.tasks: Dict[str, UploadTask] = {}
        self.created_at = datetime.now()
        self.last_accessed = datetime.now()

    def add_task(self, task: UploadTask):
        """Add a processing task to this session"""
        self.tasks[task.task_id] = task
        self.last_accessed = datetime.now()

    def get_task(self, task_id: str) -> Optional[UploadTask]:
        """Get a specific task"""
        task = self.tasks.get(task_id)
        if task:
            self.last_accessed = datetime.now()
        return task

    def get_all_tasks(self) -> List[UploadTask]:
        """Get all tasks in this session"""
        self.last_accessed = datetime.now()
        return list(self.tasks.values())

    def cleanup_temp_files(self):
        """Move processed files from temp to processed directory"""
        try:
            self.processed_dir.mkdir(parents=True, exist_ok=True)

            for file_info in self.tasks.values():
                for file_meta in file_info.files:
                    temp_path = Path(file_meta['path'])
                    if temp_path.exists():
                        # Move to processed directory
                        processed_path = self.processed_dir / temp_path.name
                        temp_path.rename(processed_path)
                        file_meta['path'] = str(processed_path)

            # Remove temp directory if empty
            if self.upload_dir.exists() and not any(self.upload_dir.iterdir()):
                self.upload_dir.rmdir()

        except Exception as e:
            print(f"Warning: Failed to cleanup temp files for session {self.session_id}: {e}")

class UploadManager:
    """Main manager for file uploads and processing"""

    def __init__(self):
        # Neo4j connection will be initialized per session as needed
        self.driver = None
        self.sessions: Dict[str, UploadSession] = {}
        self.active_tasks: Dict[str, asyncio.Task] = {}
        self.progress_callbacks: Dict[str, List[Callable]] = {}
        self.ollama_model = os.getenv("OLLAMA_MODEL", "granite4:micro-h")

        # Directory cleanup interval (hours)
        self.cleanup_interval_hours = 24

        # Background cleanup task will be started when the app starts

    def _init_neo4j(self):
        """Initialize Neo4j connection if needed"""
        if self.driver is None:
            uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
            user = os.getenv("NEO4J_USER", "neo4j")
            password = os.getenv("NEO4J_PASSWORD", "password")

            try:
                self.driver = GraphDatabase.driver(uri, auth=(user, password))
                # Test connection
                with self.driver.session() as session:
                    session.run("RETURN 1")
                print("✓ Neo4j connection initialized for upload manager")
            except Exception as e:
                print(f"✗ Neo4j connection failed for upload manager: {e}")
                self.driver = None

    def create_upload_session(self, files: List[Dict[str, Any]]) -> str:
        """Create a new upload session"""
        session_id = str(uuid.uuid4())

        # Validate session directory exists
        session_dir = Path(f"data/uploads/temp/{session_id}")
        session_dir.mkdir(parents=True, exist_ok=True)

        session = UploadSession(session_id)
        self.sessions[session_id] = session

        print(f"✓ Created upload session: {session_id} with {len(files)} files")
        return session_id

    def start_processing_task(self,
                            session_id: str,
                            config: Dict[str, Any] = None) -> str:
        """Start a processing task for an upload session"""
        if session_id not in self.sessions:
            raise ValueError(f"Session {session_id} not found")

        session = self.sessions[session_id]

        # Get files from session directory
        session_dir = Path(f"data/uploads/temp/{session_id}")
        if not session_dir.exists():
            raise ValueError(f"Session directory {session_dir} not found")

        files = []
        for file_path in session_dir.glob("*"):
            if file_path.is_file():
                files.append({
                    "filename": file_path.name,
                    "path": str(file_path),
                    "size": file_path.stat().st_size,
                    "modified": datetime.fromtimestamp(file_path.stat().st_mtime).isoformat()
                })

        if not files:
            raise ValueError(f"No files found in session {session_id}")

        # Create processing task
        task_id = str(uuid.uuid4())
        config = config or {}
        default_config = {
            "entity_types": ["Person", "Organization", "Technology", "Concept"],
            "confidence_threshold": 0.6,
            "max_chunk_size": 1000,
            "overlap_size": 200,
            "ollama_model": self.ollama_model
        }
        config = {**default_config, **config}

        task = UploadTask(
            task_id=task_id,
            session_id=session_id,
            files=files,
            config=config,
            status=TaskStatus.QUEUED
        )
        task.stats.total_files = len(files)

        session.add_task(task)

        # Start background processing
        processing_task = asyncio.create_task(
            self._process_files_async(task)
        )
        self.active_tasks[task_id] = processing_task

        print(f"✓ Started processing task: {task_id} for session {session_id}")
        return task_id

    async def _process_files_async(self, task: UploadTask):
        """Async processing of uploaded files"""
        try:
            task.status = TaskStatus.PROCESSING
            task.updated_at = datetime.now()

            print(f"🚀 Starting processing task {task.task_id}")

            # Initialize components
            self._init_neo4j()
            if not self.driver:
                task.add_error("Database connection failed")
                task.status = TaskStatus.FAILED
                return

            total_files = len(task.files)
            files_processed = 0

            for i, file_info in enumerate(task.files):
                if task.cancelled:
                    task.status = TaskStatus.CANCELLED
                    break

                try:
                    task.current_file = file_info['filename']
                    task.update_progress(i / total_files, file_info['filename'])

                    print(f"📄 Processing file {i+1}/{total_files}: {file_info['filename']}")

                    # Process individual file
                    result = await self._process_single_file(file_info, task.config)

                    task.stats.entities_extracted += result.get('entities_extracted', 0)
                    task.stats.relationships_created += result.get('relationships_created', 0)
                    task.stats.tokens_processed += result.get('tokens_processed', 0)

                    files_processed += 1
                    task.stats.files_processed = files_processed

                except Exception as e:
                    error_msg = f"Failed to process {file_info['filename']}: {str(e)}"
                    task.add_error(error_msg)
                    print(f"❌ {error_msg}")
                    traceback.print_exc()

            # Update final progress and status
            if task.cancelled:
                task.status = TaskStatus.CANCELLED
            elif task.stats.errors:
                task.status = TaskStatus.FAILED
            else:
                task.status = TaskStatus.COMPLETED

            task.update_progress(1.0)
            task.stats.processing_time_seconds = (datetime.now() - task.created_at).total_seconds()

            # Cleanup temp files
            if task.status == TaskStatus.COMPLETED:
                session = self.sessions.get(task.session_id)
                if session:
                    session.cleanup_temp_files()

            print(f"✅ Completed processing task {task.task_id}: {task.status.value}")

        except Exception as e:
            task.add_error(f"Task processing failed: {str(e)}")
            task.status = TaskStatus.FAILED
            print(f"❌ Task {task.task_id} failed: {e}")
            traceback.print_exc()

        finally:
            # Cleanup active task reference
            self.active_tasks.pop(task.task_id, None)

    async def _process_single_file(self, file_info: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single uploaded file"""
        file_path = Path(file_info['path'])
        filename = file_info['filename']

        # Determine file type and process accordingly
        file_ext = file_path.suffix.lower()

        try:
            if file_ext == '.csv':
                return await self._process_csv_file(file_path, filename, config)
            elif file_ext == '.pdf':
                return await self._process_pdf_file(file_path, filename, config)
            elif file_ext == '.txt':
                return await self._process_text_file(file_path, filename, config)
            elif file_ext == '.json':
                return await self._process_json_file(file_path, filename, config)
            elif file_ext == '.md':
                return await self._process_markdown_file(file_path, filename, config)
            else:
                raise ValueError(f"Unsupported file type: {file_ext}")

        except Exception as e:
            raise Exception(f"Failed to process {filename}: {str(e)}")

    async def _process_csv_file(self, file_path: Path, filename: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Process CSV file"""
        # Read file content
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Extract entities using LLM
        entities = await self._extract_entities_llm(content, filename, config)
        relationships = await self._infer_relationships_llm(entities, content, config)

        # Store in Neo4j
        await self._store_in_neo4j(filename, content, entities, relationships, config)

        return {
            "entities_extracted": len(entities),
            "relationships_created": len(relationships),
            "tokens_processed": len(content.split()) * 1.3,
            "file_type": "csv"
        }

    async def _process_pdf_file(self, file_path: Path, filename: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Process PDF file"""
        try:
            import PyPDF2
            with open(file_path, 'rb') as f:
                pdf_reader = PyPDF2.PdfReader(f)
                content = ""
                for page in pdf_reader.pages:
                    content += page.extract_text() + "\n"
        except ImportError:
            content = f"PDF parsing not available for {filename}"

        # Extract entities using LLM
        entities = await self._extract_entities_llm(content, filename, config)
        relationships = await self._infer_relationships_llm(entities, content, config)

        # Store in Neo4j
        await self._store_in_neo4j(filename, content, entities, relationships, config)

        return {
            "entities_extracted": len(entities),
            "relationships_created": len(relationships),
            "tokens_processed": len(content.split()) * 1.3,
            "file_type": "pdf"
        }

    async def _process_text_file(self, file_path: Path, filename: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Process text file"""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        entities = await self._extract_entities_llm(content, filename, config)
        relationships = await self._infer_relationships_llm(entities, content, config)

        await self._store_in_neo4j(filename, content, entities, relationships, config)

        return {
            "entities_extracted": len(entities),
            "relationships_created": len(relationships),
            "tokens_processed": len(content.split()) * 1.3,
            "file_type": "txt"
        }

    async def _process_json_file(self, file_path: Path, filename: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Process JSON file"""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Convert JSON to text representation for processing
        if isinstance(data, dict):
            content = json.dumps(data, indent=2)
        elif isinstance(data, list):
            content = "\n".join(json.dumps(item, indent=2) for item in data)
        else:
            content = str(data)

        entities = await self._extract_entities_llm(content, filename, config)
        relationships = await self._infer_relationships_llm(entities, content, config)

        await self._store_in_neo4j(filename, content, entities, relationships, config)

        return {
            "entities_extracted": len(entities),
            "relationships_created": len(relationships),
            "tokens_processed": len(content.split()) * 1.3,
            "file_type": "json"
        }

    async def _process_markdown_file(self, file_path: Path, filename: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Process Markdown file"""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        entities = await self._extract_entities_llm(content, filename, config)
        relationships = await self._infer_relationships_llm(entities, content, config)

        await self._store_in_neo4j(filename, content, entities, relationships, config)

        return {
            "entities_extracted": len(entities),
            "relationships_created": len(relationships),
            "tokens_processed": len(content.split()) * 1.3,
            "file_type": "markdown"
        }

    async def _extract_entities_llm(self, content: str, filename: str, config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract entities using LLM"""
        entity_types = config.get('entity_types', [])
        threshold = config.get('confidence_threshold', 0.6)
        model = config.get('ollama_model', self.ollama_model)

        # Prepare prompt
        prompt = f"""Extract entities from the following document content.

Document: {filename}
Entity types to extract: {', '.join(entity_types)}

Content:
{content[:3000]}...  # Truncated for prompt

Return JSON format:
{{
  "entities": [
    {{
      "name": "Entity Name",
      "type": "EntityType",
      "description": "Brief description",
      "confidence": 0.8
    }}
  ]
}}"""

        try:
            response = ollama.generate(
                model=model,
                prompt=prompt,
                format="json"
            )

            result = json.loads(response['response'])
            entities = result.get('entities', [])

            # Filter by confidence and type
            filtered_entities = [
                entity for entity in entities
                if entity.get('confidence', 0) >= threshold and
                entity.get('type') in entity_types
            ]

            return filtered_entities

        except Exception as e:
            print(f"Entity extraction failed for {filename}: {e}")
            return []

    async def _infer_relationships_llm(self, entities: List[Dict[str, Any]], content: str, config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Infer relationships between entities using LLM"""
        if len(entities) < 2:
            return []

        model = config.get('ollama_model', self.ollama_model)
        threshold = config.get('confidence_threshold', 0.6)

        entity_names = [e['name'] for e in entities]

        prompt = f"""Analyze relationships between these entities based on the content.

Entities: {', '.join(entity_names)}

Content preview:
{content[:2000]}...

Return JSON format:
{{
  "relationships": [
    {{
      "entity1": "Entity A",
      "entity2": "Entity B",
      "type": "RELATED_TO",
      "description": "How they are related",
      "confidence": 0.8
    }}
  ]
}}"""

        try:
            response = ollama.generate(
                model=model,
                prompt=prompt,
                format="json"
            )

            result = json.loads(response['response'])
            relationships = result.get('relationships', [])

            # Filter by confidence
            filtered_relationships = [
                rel for rel in relationships
                if rel.get('confidence', 0) >= threshold
            ]

            return filtered_relationships

        except Exception as e:
            print(f"Relationship inference failed: {e}")
            return []

    async def _store_in_neo4j(self, filename: str, content: str, entities: List[Dict[str, Any]],
                            relationships: List[Dict[str, Any]], config: Dict[str, Any]):
        """Store document, entities and relationships in Neo4j"""
        if not self.driver:
            return

        try:
            with self.driver.session() as session:
                # Create document node
                doc_result = session.run("""
                    MERGE (d:Conversation {filename: $filename})
                    SET d.content = $content,
                        d.document_type = $doc_type,
                        d.uploaded_at = datetime(),
                        d.entities_count = $entities_count
                    RETURN d
                """, filename=filename, content=content, doc_type=filename.split('.')[-1],
                     entities_count=len(entities))

                doc_node = doc_result.single()['d']

                # Create entity nodes and link to document
                entity_nodes = []
                for entity in entities:
                    entity_result = session.run("""
                        MERGE (e:Entity {name: $name})
                        SET e.type = $entity_type,
                            e.description = $description,
                            e.confidence = $confidence
                        RETURN e
                    """, name=entity['name'], entity_type=entity['type'],
                         description=entity.get('description', ''),
                         confidence=entity.get('confidence', 0.8))
                    entity_nodes.append(entity_result.single()['e'])

                    # Link document to entity
                    session.run("""
                        MATCH (d:Conversation {filename: $filename})
                        MATCH (e:Entity {name: $entity_name})
                        MERGE (d)-[:MENTIONS]->(e)
                    """, filename=filename, entity_name=entity['name'])

                # Create relationships between entities
                for rel in relationships:
                    entity1_name = rel['entity1']
                    entity2_name = rel['entity2']
                    rel_type = rel['type']
                    description = rel.get('description', '')
                    confidence = rel.get('confidence', 0.8)

                    session.run(f"""
                        MATCH (e1:Entity {{name: $entity1}})
                        MATCH (e2:Entity {{name: $entity2}})
                        MERGE (e1)-[r:`{rel_type}`]->(e2)
                        SET r.description = $description,
                            r.confidence = $confidence,
                            r.created_at = datetime()
                    """, entity1=entity1_name, entity2=entity2_name,
                       description=description, confidence=confidence)

        except Exception as e:
            print(f"Neo4j storage failed: {e}")
            raise

    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get task status"""
        for session in self.sessions.values():
            task = session.get_task(task_id)
            if task:
                return task.to_dict()
        return None

    def cancel_task(self, task_id: str) -> bool:
        """Cancel a running task"""
        task = None
        for session in self.sessions.values():
            task = session.get_task(task_id)
            if task:
                break

        if not task:
            return False

        task.cancel()

        # Cancel the asyncio task if running
        active_task = self.active_tasks.get(task_id)
        if active_task and not active_task.done():
            active_task.cancel()

        return True

    def get_session_tasks(self, session_id: str) -> List[Dict[str, Any]]:
        """Get all tasks for a session"""
        session = self.sessions.get(session_id)
        if not session:
            return []

        return [task.to_dict() for task in session.get_all_tasks()]

    async def _periodic_cleanup(self):
        """Periodic cleanup of old sessions and files"""
        while True:
            try:
                await asyncio.sleep(self.cleanup_interval_hours * 3600)  # Convert hours to seconds

                # Cleanup old sessions (older than 24 hours)
                current_time = datetime.now()
                sessions_to_remove = []

                for session_id, session in self.sessions.items():
                    age_hours = (current_time - session.last_accessed).total_seconds() / 3600
                    if age_hours > self.cleanup_interval_hours:
                        sessions_to_remove.append(session_id)

                for session_id in sessions_to_remove:
                    print(f"🧹 Cleaning up expired session: {session_id}")
                    del self.sessions[session_id]

                # Cleanup old temp directories
                temp_dir = Path("data/uploads/temp")
                if temp_dir.exists():
                    for session_dir in temp_dir.iterdir():
                        if session_dir.is_dir():
                            session_id_from_dir = session_dir.name
                            if session_id_from_dir not in self.sessions:
                                # Check if directory is old
                                age_hours = (current_time - datetime.fromtimestamp(session_dir.stat().st_mtime)).total_seconds() / 3600
                                if age_hours > self.cleanup_interval_hours:
                                    try:
                                        import shutil
                                        shutil.rmtree(session_dir)
                                        print(f"🧹 Removed old temp directory: {session_dir}")
                                    except Exception as e:
                                        print(f"Warning: Failed to remove {session_dir}: {e}")

            except Exception as e:
                print(f"Cleanup task error: {e}")

# Global instance
upload_manager = UploadManager()
