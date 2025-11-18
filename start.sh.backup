#!/bin/bash
# Start EPS Document GraphRAG Application
# This script sets up and starts the EPS Document GraphRAG application

echo "🚀 Starting EPS Document GraphRAG setup..."

# Check prerequisites
echo "📋 Checking prerequisites..."
command -v python3 >/dev/null 2>&1 || { echo "❌ Python3 is required but not installed. Please install Python 3.8+ first."; exit 1; }
command -v node >/dev/null 2>&1 || { echo "❌ Node.js is required but not installed. Please install Node.js 16+ first."; exit 1; }
command -v pip >/dev/null 2>&1 || { echo "❌ pip is required but not installed. Please install pip first."; exit 1; }
command -v npm >/dev/null 2>&1 || { echo "❌ npm is required but not installed. Please install npm first."; exit 1; }

# Function to check if port is available
check_port() {
    local port=$1
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null ; then
        echo "❌ Port $port is already in use. Please stop the service using that port or choose a different port."
        return 1
    fi
    return 0
}

# Check if required ports are available
check_port 3001 || exit 1
# Frontend typically runs on 3000 by default
check_port 3000 || check_port 3001 || exit 1
check_port 8000 || exit 1

# Backend setup
echo "🐍 Setting up Python backend..."

# Install Python dependencies
echo "📦 Installing Python dependencies..."
pip install neo4j-graphrag[ollama] ollama python-dotenv fastapi uvicorn pydantic httpx numpy matplotlib plotly pandas redis python-multipart scikit-learn sentence-transformers nltk rouge_score || {
    echo "❌ Failed to install Python dependencies. Please check your Python/pip installation."
    exit 1
}

# Set up required services
echo "🔧 Setting up required services..."

# Check if Docker is available
if command -v docker >/dev/null 2>&1; then
    echo "🐳 Docker found - setting up services..."

    # Check if Neo4j container exists
    if ! docker ps -a --format 'table {{.Names}}' | grep -q "^neo4j$"; then
        echo "🏗️ Creating Neo4j database..."
        docker run -d --name neo4j \
            -p 7474:7474 -p 7687:7687 \
            -e NEO4J_AUTH=neo4j/research2025 \
            -e NEO4J_PLUGINS='["graph-data-science"]' \
            neo4j:latest
        echo "⏳ Waiting for Neo4j to initialize..."
        sleep 10
    else
        echo "✅ Neo4j container exists"
        # Start it if it's stopped
        if ! docker ps --format 'table {{.Names}}' | grep -q "^neo4j$"; then
            echo "▶️ Starting Neo4j container..."
            docker start neo4j
            sleep 5
        fi
    fi
else
    echo "⚠️ Docker not found. Please ensure Neo4j is running on localhost:7687 with:"
    echo "   NEO4J_AUTH=neo4j/research2025"
    echo ""
fi

# Setup Ollama
echo "🧠 Setting up Ollama AI models..."

# Check if Ollama is installed
if ! command -v ollama >/dev/null 2>&1; then
    echo "❌ Ollama not found. Please install Ollama first:"
    echo "   brew install ollama (macOS)"
    echo "   curl -fsSL https://ollama.ai/install.sh | sh (Linux)"
    echo "   Or download from https://ollama.ai/download"
    exit 1
fi

# Start Ollama service in background if not running
if ! nc -z localhost 11434 2>/dev/null; then
    echo "🖥️ Starting Ollama service..."
    ollama serve &
    OLLAMA_PID=$!
    sleep 3
fi

# Check and pull required models
echo "📥 Checking and downloading AI models..."

# Function to check if model exists
model_exists() {
    local model=$1
    ollama list 2>/dev/null | grep -q "$model"
}

if ! model_exists "granite4:micro-h"; then
    echo "⬇️ Downloading granite4:micro-h model (for reasoning)..."
    ollama pull granite4:micro-h
else
    echo "✅ granite4:micro-h model available"
fi

if ! model_exists "mxbai-embed-large:latest"; then
    echo "⬇️ Downloading mxbai-embed-large model (for embeddings)..."
    ollama pull mxbai-embed-large:latest
else
    echo "✅ mxbai-embed-large model available"
fi

# Final service verification
echo "🔍 Verifying all services are ready..."

# Neo4j connection check
if nc -z localhost 7687 2>/dev/null; then
    echo "✅ Neo4j is running on port 7687"
else
    echo "❌ Neo4j not accessible on port 7687"
    exit 1
fi

# Ollama connection check
if nc -z localhost 11434 2>/dev/null; then
    echo "✅ Ollama is running on port 11434"
else
    echo "❌ Ollama not accessible on port 11434"
    exit 1
fi

echo "🎯 All services are ready!"

# Data ingestion check
echo "📊 Checking data ingestion..."
if [ ! -f "EPS_FILES_20K_NOV2026.csv" ]; then
    echo "❌ EPS_FILES_20K_NOV2026.csv file not found"
    echo "   Please ensure the EPS_FILES_20K_NOV2026.csv file exists in the root directory"
    exit 1
fi

# Check if database is already populated
echo "🔍 Checking database status..."
python3 -c "
import sys
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

try:
    from scripts.eps_retriever import EPSRetriever
    retriever = EPSRetriever()
    result = retriever.driver.session().run('MATCH (d:EPSDocument) RETURN count(d) as count').single()
    count = result['count'] if result else 0
    print(f'Found {count} EPS document nodes in database')
    if count == 0:
        print('Database appears empty - will run ingestion')
        sys.exit(1)
    else:
        print('Database already populated - skipping ingestion')
        sys.exit(0)
except Exception as e:
    print(f'Error checking database: {e}')
    print('Will attempt to run ingestion')
    sys.exit(1)
" 2>/dev/null

INGESTION_NEEDED=$?

if [ $INGESTION_NEEDED -eq 1 ]; then
    echo "📥 Running EPS document data ingestion..."
    if ! python3 scripts/ingest_eps_data.py --csv EPS_FILES_20K_NOV2026.csv --create-indexes --create-similarities; then
        echo "❌ Data ingestion failed. Please check your Neo4j connection and try again."
        exit 1
    fi
    echo "✅ Data ingestion completed successfully"
else
    echo "✅ Database already contains data - skipping ingestion"
fi

# Update/create indexes and relationships
echo "🔗 Ensuring indexes and relationships are created..."
python3 create_indexes.py
python3 create_thread_relationships.py

# Frontend setup
echo "⚛️ Setting up Next.js frontend..."

# Check if frontend directory exists
if [ ! -d "frontend" ]; then
    echo "❌ Frontend directory not found. Please ensure the frontend folder exists."
    exit 1
fi

cd frontend

# Install Node.js dependencies
echo "📦 Installing Node.js dependencies..."
if ! npm install --force; then
    echo "❌ Failed to install Node.js dependencies. Please check your Node.js/npm installation."
    exit 1
fi

# Check if .env exists
if [ ! -f ".env" ]; then
    echo "⚠️ .env file not found in frontend directory"
    echo "   Please create frontend/.env with appropriate configuration"
    echo "   You can use the .env.example as a template"
fi

cd ..

# Start services in background
echo "🌐 Starting services..."

# Start FastAPI backend
echo "🐍 Starting FastAPI backend on port 8000..."
python3 main.py &
BACKEND_PID=$!

# Wait for backend to start
echo "⏳ Waiting for backend to start..."
BACKEND_READY=false
for i in {1..30}; do
    echo "   Checking backend (attempt $i/30)..."
    if curl -s --max-time 5 http://localhost:8000/api/health > /dev/null 2>&1; then
        echo "✅ Backend is ready!"
        BACKEND_READY=true
        break
    fi
    sleep 2
done

if [ "$BACKEND_READY" = false ]; then
    echo "❌ Backend failed to start within expected time"
    kill $BACKEND_PID 2>/dev/null || true
    exit 1
fi

# Start Next.js frontend
echo "⚛️ Starting Next.js frontend on port 3001..."
cd frontend
PORT=3001 npm run dev &
FRONTEND_PID=$!
cd ..

# Wait for frontend to start
echo "⏳ Waiting for frontend to start..."
FRONTEND_READY=false
for i in {1..20}; do
    echo "   Checking frontend (attempt $i/20)..."
    if curl -s --max-time 5 http://localhost:3001 > /dev/null 2>&1; then
        echo "✅ Frontend is ready!"
        FRONTEND_READY=true
        break
    fi
    sleep 2
done

if [ "$FRONTEND_READY" = false ]; then
    echo "❌ Frontend failed to start within expected time"
    kill $BACKEND_PID 2>/dev/null || true
    kill $FRONTEND_PID 2>/dev/null || true
    exit 1
fi

echo ""
echo "🎉 EPS Document GraphRAG application is running successfully!"
echo ""
echo "🌐 Frontend (Next.js): http://localhost:3000 (or 3001)"
echo "🔧 Backend API: http://localhost:8000"
echo "📚 API Documentation: http://localhost:8000/docs"
echo ""
echo "📊 Database Status:"
python3 -c "
try:
    from scripts.eps_retriever import EPSRetriever
    retriever = EPSRetriever()
    result = retriever.driver.session().run('MATCH (d:EPSDocument) RETURN count(d) as count').single()
    count = result['count'] if result else 0
    print(f'   • EPS documents: {count}')
    result = retriever.driver.session().run('MATCH (t:Topic) RETURN count(t) as count').single()
    count = result['count'] if result else 0
    print(f'   • Topics identified: {count}')
    result = retriever.driver.session().run('MATCH (e:Entity) RETURN count(e) as count').single()
    count = result['count'] if result else 0
    print(f'   • Entities extracted: {count}')
    result = retriever.driver.session().run('MATCH ()-[r:SIMILAR_TO]->() RETURN count(r) as count').single()
    count = result['count'] if result else 0
    print(f'   • Similarity relationships: {count}')
except Exception as e:
    print(f'   • Error checking database: {e}')
" 2>/dev/null
echo ""
echo "🛑 Press Ctrl+C to stop all services"
echo ""
echo "💡 Try asking questions like:"
echo "   • 'What documents discuss surveillance programs?'"
echo "   • 'Find content about government data collection'"
echo "   • 'Show me documents mentioning NSA or GCHQ'"

# Function to cleanup processes on exit
cleanup() {
    echo ""
    echo "🛑 Stopping services..."
    kill $BACKEND_PID 2>/dev/null || true
    kill $FRONTEND_PID 2>/dev/null || true
    echo "✅ Services stopped. Goodbye!"
    exit 0
}

# Set trap to cleanup on script termination
trap cleanup INT TERM

# Wait for either process to exit
wait $BACKEND_PID $FRONTEND_PID
