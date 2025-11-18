"""
FastAPI backend for the Research Assistant application
Following the architecture outlined in the README
"""
import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import os
import aiofiles
import shutil
from pathlib import Path
from datetime import datetime
import json
import uuid
import mimetypes

# Import our custom modules
from scripts.conversation_reasoning_agent import ConversationReasoningAgent
from scripts.conversation_retriever import ConversationRetriever
from scripts.ingest_conversation_data import ConversationGraphBuilder
from scripts.upload_manager import upload_manager
# from evaluation.run_evaluation import Evaluator

# Initialize components
app = FastAPI(title="Research Assistant API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],  # Allow Next.js dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances (will be initialized on startup)
reasoning_agent = None
retriever = None
evaluator = None

# Simple in-memory session storage for chat history
# In production, this should be stored in Redis or a database
chat_sessions: Dict[str, List[Dict[str, str]]] = {}

class QueryRequest(BaseModel):
    query: str
    chat_history: Optional[List[Dict[str, str]]] = []
    persona_override: Optional[str] = None
    session_id: Optional[str] = None

class QueryResponse(BaseModel):
    response: str
    context_used: List[Dict[str, Any]]
    quality_grade: float
    retrieval_method: Optional[str]
    retrieval_performed: bool
    sources: List[Dict[str, str]]
    session_id: Optional[str] = None

class IngestionRequest(BaseModel):
    directory: str = "data/research_papers"
    recreate_indexes: bool = False

class EvaluationRequest(BaseModel):
    dataset_path: Optional[str] = None
    output_path: Optional[str] = "evaluation/results/api_evaluation.json"

class SystemStatus(BaseModel):
    neo4j_connected: bool
    ollama_ready: bool
    redis_connected: bool
    conversation_count: int
    evaluation_count: int

# File Upload Models
class FileUploadResponse(BaseModel):
    session_id: str
    files: List[Dict[str, Any]]
    total_size: int
    message: str

class ProcessingRequest(BaseModel):
    session_id: str
    entity_types: Optional[List[str]] = ["Person", "Organization", "Technology", "Concept"]
    confidence_threshold: Optional[float] = 0.6
    max_chunk_size: Optional[int] = 1000
    overlap_size: Optional[int] = 200

class ProcessingResponse(BaseModel):
    task_id: str
    session_id: str
    estimated_files: int
    estimated_tokens: int
    message: str

class ProgressResponse(BaseModel):
    task_id: str
    status: str  # "queued", "processing", "completed", "failed"
    progress: float  # 0-1
    current_file: Optional[str]
    files_processed: int
    total_files: int
    entities_extracted: int
    errors: List[str]
    eta_seconds: Optional[int]
    created_at: str
    updated_at: str

@app.on_event("startup")
async def startup_event():
    """Initialize components on startup"""
    global reasoning_agent, retriever, evaluator

    try:
        reasoning_agent = ConversationReasoningAgent()
        retriever = ConversationRetriever()
        # evaluator = Evaluator(
        #     test_dataset_path=Path("evaluation/datasets/research_assistant_v1.json"),
        #     trace_db_path=Path("evaluation/trace.db")
        # )

        # Create upload directories
        upload_temp_dir = Path("data/uploads/temp")
        upload_temp_dir.mkdir(parents=True, exist_ok=True)

        print("✓ All components initialized")
    except Exception as e:
        print(f"✗ Component initialization failed: {e}")

ALLOWED_EXTENSIONS = {'.csv', '.pdf', '.txt', '.json', '.md'}
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB per file
MAX_TOTAL_SIZE = 200 * 1024 * 1024  # 200MB total per upload

def validate_file_type(filename: str) -> bool:
    """Validate if file extension is allowed"""
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS

def validate_file_size(file_size: int) -> bool:
    """Validate if file size is within limits"""
    return file_size <= MAX_FILE_SIZE

@app.post("/api/upload/files")
async def upload_files(files: List[UploadFile] = File(...)) -> FileUploadResponse:
    """Upload multiple files for processing"""
    try:
        if not files:
            raise HTTPException(status_code=400, detail="No files provided")

        if len(files) > 20:  # Max 20 files
            raise HTTPException(status_code=400, detail="Too many files. Maximum 20 files allowed.")

        # Create session in upload_manager first (it generates the session_id)
        session_id = upload_manager.create_upload_session([])
        session_dir = Path(f"data/uploads/temp/{session_id}")
        # Directory should already be created by upload_manager, but ensure it exists
        session_dir.mkdir(parents=True, exist_ok=True)

        uploaded_files = []
        total_size = 0

        for file in files:
            # Validate filename and size
            if not validate_file_type(file.filename):
                raise HTTPException(
                    status_code=400,
                    detail=f"Unsupported file type: {file.filename}. Supported types: {', '.join(ALLOWED_EXTENSIONS)}"
                )

            # Get file size
            file_content = await file.read()
            file_size = len(file_content)

            if not validate_file_size(file_size):
                raise HTTPException(
                    status_code=400,
                    detail=f"File too large: {file.filename}. Maximum size: {MAX_FILE_SIZE // (1024*1024)}MB"
                )

            total_size += file_size
            if total_size > MAX_TOTAL_SIZE:
                raise HTTPException(
                    status_code=400,
                    detail=f"Total upload size too large. Maximum total: {MAX_TOTAL_SIZE // (1024*1024)}MB"
                )

            # Save file
            file_path = session_dir / file.filename
            async with aiofiles.open(file_path, 'wb') as f:
                await f.write(file_content)

            # Get MIME type
            mime_type, _ = mimetypes.guess_type(file.filename)
            if not mime_type:
                mime_type = "application/octet-stream"

            uploaded_files.append({
                "filename": file.filename,
                "size": file_size,
                "mime_type": mime_type,
                "path": str(file_path),
                "uploaded_at": datetime.now().isoformat()
            })

        return FileUploadResponse(
            session_id=session_id,
            files=uploaded_files,
            total_size=total_size,
            message=f"Successfully uploaded {len(uploaded_files)} files"
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@app.post("/api/upload/process")
async def process_uploaded_files(request: ProcessingRequest) -> ProcessingResponse:
    """Start processing uploaded files"""
    print(f"🚀 /api/upload/process called - Request: {request}")
    print(f"🔄 Starting processing for session: {request.session_id}")
    print(f"🧪 upload_manager exists: {upload_manager is not None}")
    print(f"🧪 upload_manager type: {type(upload_manager)}")

    # Quick test - return early without processing
    if os.getenv("TEST_MODE") == "1":
        print("🧪 TEST MODE: Returning early without processing")
        return ProcessingResponse(
            task_id="test-task-123",
            session_id=request.session_id,
            estimated_files=1,
            estimated_tokens=1000,
            message="Test mode - no processing done"
        )

    try:
        # Verify session exists
        session_dir = Path(f"data/uploads/temp/{request.session_id}")
        if not session_dir.exists():
            print(f"❌ Session directory not found: {session_dir}")
            raise HTTPException(status_code=404, detail=f"Upload session {request.session_id} not found")

        # Count files in session
        files = list(session_dir.glob("*"))
        if not files:
            print(f"❌ No files found in session directory: {session_dir}")
            raise HTTPException(status_code=400, detail="No files found in session")

        print(f"✅ Found {len(files)} files in session")

        # Estimate processing scale
        total_size = sum(f.stat().st_size for f in files)
        estimated_tokens = total_size // 4  # Rough token estimation

        # Start processing task
        config = {
            "entity_types": request.entity_types or ["Person", "Organization", "Technology", "Concept"],
            "confidence_threshold": request.confidence_threshold or 0.6,
            "max_chunk_size": request.max_chunk_size or 1000,
            "overlap_size": request.overlap_size or 200
        }

        print(f"🚀 Starting upload_manager processing task...")
        task_id = upload_manager.start_processing_task(request.session_id, config)
        print(f"✅ Processing task started with ID: {task_id}")

        return ProcessingResponse(
            task_id=task_id,
            session_id=request.session_id,
            estimated_files=len(files),
            estimated_tokens=int(estimated_tokens),
            message=f"Started processing {len(files)} files"
        )

    except HTTPException:
        print(f"⚠️ HTTPException: {str(e)}")
        raise
    except Exception as e:
        import traceback
        print(f"❌ Processing failed with exception: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

@app.get("/api/upload/progress/{task_id}")
async def get_upload_progress(task_id: str) -> ProgressResponse:
    """Get progress of upload processing task"""
    try:
        task_status = upload_manager.get_task_status(task_id)
        if not task_status:
            raise HTTPException(status_code=404, detail=f"Task {task_id} not found")

        return ProgressResponse(**task_status)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Progress check failed: {str(e)}")

@app.delete("/api/upload/progress/{task_id}")
async def cancel_upload_task(task_id: str):
    """Cancel an upload processing task"""
    try:
        cancelled = upload_manager.cancel_task(task_id)
        if not cancelled:
            raise HTTPException(status_code=404, detail=f"Task {task_id} not found or not cancellable")

        return {"message": f"Task {task_id} cancelled", "task_id": task_id}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cancellation failed: {str(e)}")

@app.get("/api/health")
async def health_check():
    """Basic health check"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.post("/api/chat")
async def chat(request: QueryRequest) -> QueryResponse:
    """Main chat endpoint with GraphRAG"""
    if not reasoning_agent:
        raise HTTPException(status_code=503, detail="Reasoning agent not initialized")

    try:
        # Session management: Get or create session
        session_id = request.session_id
        if not session_id or session_id not in chat_sessions:
            session_id = str(uuid.uuid4())
            chat_sessions[session_id] = []

        # Use provided chat_history for backward compatibility, or build from session
        if request.chat_history and len(request.chat_history) > len(chat_sessions[session_id]):
            # If user provided more complete history, use it and update session
            chat_history_to_use = request.chat_history
            chat_sessions[session_id] = request.chat_history.copy()
        else:
            # Use session history
            chat_history_to_use = chat_sessions[session_id].copy()

        # Add current user message if not already present
        user_message = {"role": "user", "content": request.query}
        if not chat_history_to_use or chat_history_to_use[-1] != user_message:
            chat_history_to_use.append(user_message)

        # Generate response
        result = reasoning_agent.generate_response(
            request.query,
            chat_history_to_use
        )

        # Add both user and assistant messages to session history
        chat_sessions[session_id].append(user_message)
        chat_sessions[session_id].append({"role": "assistant", "content": result['response']})

        # Format sources for frontend
        sources = []
        for doc in result['context_used']:
            sources.append({
                'title': doc.get('filename', 'Unknown Document'),
                'authors': doc.get('document_type', 'Unknown'),
                'year': doc.get('filename', 'Unknown')[:10] if doc.get('filename') else 'Unknown',
                'relevance_score': f"{doc.get('relevance_score', 0.0):.3f}",
                'retrieval_method': str(doc.get('retrieval_method', 'unknown'))
            })

        return QueryResponse(
            response=result['response'],
            context_used=result['context_used'],
            quality_grade=result['quality_grade'],
            retrieval_method=result.get('retrieval_method'),
            retrieval_performed=result.get('retrieval_performed', False),
            sources=sources,
            session_id=session_id
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat processing failed: {str(e)}")

@app.get("/api/search")
async def search_papers(query: str, limit: int = 10):
    """Direct paper search endpoint"""
    if not retriever:
        raise HTTPException(status_code=503, detail="Retriever not initialized")

    try:
        results = retriever.retrieve_context(query, limit=limit)
        return {"results": results, "query": query}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@app.post("/api/ingest")
async def ingest_papers(request: IngestionRequest, background_tasks: BackgroundTasks):
    """Ingest research papers"""
    try:
        # Run ingestion in background
        background_tasks.add_task(run_ingestion, request.directory, request.recreate_indexes)
        return {"message": f"Started ingestion from {request.directory}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")

@app.post("/api/evaluate")
async def run_evaluation_endpoint(request: EvaluationRequest, background_tasks: BackgroundTasks):
    """Run evaluation"""
    if not evaluator:
        raise HTTPException(status_code=503, detail="Evaluator not initialized")

    try:
        # Run evaluation in background
        background_tasks.add_task(run_evaluation_task, request.dataset_path, request.output_path)
        return {"message": "Started evaluation"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {str(e)}")

@app.get("/api/status")
async def get_system_status() -> SystemStatus:
    """Get comprehensive system status"""
    try:
        # Check Neo4j connection
        neo4j_connected = False
        conversation_count = 0
        try:
            if retriever and retriever.driver:
                with retriever.driver.session() as session:
                    result = session.run("MATCH (d:Conversation) RETURN count(d) as count")
                    conversation_count = result.single()["count"]
                    neo4j_connected = True
        except:
            pass

        # Check Redis
        redis_connected = False
        try:
            import redis
            r = redis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379"))
            r.ping()
            redis_connected = True
        except:
            pass

        # Check Ollama (simplified)
        ollama_ready = True  # Assume ready if service started

        # Get evaluation count
        evaluation_count = 0
        try:
            trace_file = Path("evaluation/trace.db")
            if trace_file.exists():
                with open(trace_file, 'r') as f:
                    data = json.load(f)
                    evaluation_count = len(data)
        except:
            pass

        return SystemStatus(
            neo4j_connected=neo4j_connected,
            ollama_ready=ollama_ready,
            redis_connected=redis_connected,
            conversation_count=conversation_count,
            evaluation_count=evaluation_count
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Status check failed: {str(e)}")

@app.get("/api/evaluation-results")
async def get_evaluation_results():
    """Get latest evaluation results"""
    try:
        results_path = Path("evaluation/results/evaluation_output.json")
        if results_path.exists():
            with open(results_path, 'r') as f:
                return json.load(f)
        else:
            return {"error": "No evaluation results found"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load results: {str(e)}")

# Background tasks
def run_ingestion(directory: str, recreate_indexes: bool):
    """Run conversation data ingestion in background"""
    print(f"Starting background conversation ingestion from {directory}")

    try:
        builder = ConversationGraphBuilder()

        if recreate_indexes:
            # Drop existing indexes first
            try:
                with builder.driver.session() as session:
                    session.run("DROP INDEX conversation_embeddings IF EXISTS")
            except:
                pass
            builder.create_vector_indexes()

        # Check file type and ingest accordingly
        if directory.endswith('.json'):
            builder.ingest_conversations_json(Path(directory))

            if recreate_indexes:
                builder.create_vector_indexes()

            # Create similarity relationships
            builder.create_similarity_relationships()
        elif directory.endswith('.csv'):
            # For backward compatibility, though we changed to JSON
            builder.ingest_eps_csv(Path(directory))

            if recreate_indexes:
                builder.create_vector_indexes()

            # Create similarity relationships
            builder.create_similarity_relationships()
        else:
            print(f"✗ File {directory} is not supported. Ingestion expects a JSON (.json) or CSV (.csv) file.")

        print("✓ Background conversation ingestion completed")

    except Exception as e:
        print(f"✗ Background conversation ingestion failed: {e}")

def run_evaluation_task(dataset_path: str = None, output_path: str = "evaluation/results/api_evaluation.json"):
    """Run evaluation in background"""
    print("Starting background evaluation")

    try:
        if not evaluator:
            print("✗ Evaluator not initialized")
            return

        # Use default sample queries if no dataset provided
        if dataset_path and Path(dataset_path).exists():
            with open(dataset_path, 'r') as f:
                dataset = json.load(f)
                queries = dataset.get('queries', [])
        else:
            queries = [
                {
                    'query': 'What are the main approaches to attention mechanisms in deep learning?',
                    'persona': 'researcher',
                    'ground_truth_chunk_ids': ['attention paper'],
                    'reference_answer': 'Attention mechanisms in deep learning...',
                    'complexity_score': 0.7
                },
                {
                    'query': 'How do transformer models handle long-range dependencies?',
                    'persona': 'student',
                    'ground_truth_chunk_ids': ['transformer paper'],
                    'reference_answer': 'Transformer models use...',
                    'complexity_score': 0.6
                }
            ]

        results = evaluator.run_evaluation(
            queries=queries,
            output_path=Path(output_path)
        )

        print("✓ Background evaluation completed")

    except Exception as e:
        print(f"✗ Background evaluation failed: {e}")

# Mount static files if they exist (for production)
frontend_path = Path("frontend/build")
if frontend_path.exists():
    app.mount("/", StaticFiles(directory=frontend_path, html=True), name="frontend")

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
