#!/bin/bash
# Start Research Assistant GraphRAG Application
# This script starts the application services (assumes setup.sh has been run)

echo "🚀 Starting Research Assistant GraphRAG Application..."

# 1. Verify setup has been completed
echo "📋 Verifying setup..."
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found. Please run ./setup.sh first."
    exit 1
fi

if [ ! -d "frontend/node_modules" ]; then
    echo "❌ Frontend dependencies not installed. Please run ./setup.sh first."
    exit 1
fi

if ! command -v ollama &> /dev/null; then
    echo "❌ Ollama not found. Please run ./setup.sh first."
    exit 1
fi

# 2. Function to check if port is available
check_port() {
    local port=$1
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null ; then
        echo "❌ Port $port is already in use. Please stop the service using that port or choose a different port."
        return 1
    fi
    return 0
}

# 3. Check if required ports are available
echo "🔍 Checking available ports..."
check_port 8000 || exit 1
check_port 3000 || check_port 3001 || exit 1

# 4. Start Docker services
echo "🐳 Starting Docker services..."
if command -v docker-compose &> /dev/null || command -v docker &> /dev/null && docker compose version &> /dev/null; then
    if command -v docker-compose &> /dev/null; then
        docker-compose up -d
    else
        docker compose up -d
    fi
    echo "⏳ Waiting for Neo4j to start..."
    sleep 10
else
    echo "⚠️  Docker Compose not found. Please ensure Neo4j is running manually."
    exit 1
fi

# 5. Verify Neo4j connection
echo "🔗 Verifying Neo4j connection..."
source venv/bin/activate
python3 -c "
from neo4j import GraphDatabase
import os
try:
    driver = GraphDatabase.driver(
        os.getenv('NEO4J_URI', 'bolt://localhost:7687'),
        auth=(os.getenv('NEO4J_USERNAME', 'neo4j'), os.getenv('NEO4J_PASSWORD', 'research2025'))
    )
    with driver.session() as session:
        result = session.run('RETURN 1 as num')
        print('✅ Neo4j connection successful')
    driver.close()
except Exception as e:
    print(f'❌ Neo4j connection failed: {e}')
    exit 1
"

# 6. Start Ollama service if not running
echo "🧠 Starting Ollama service..."
if ! nc -z localhost 11434 2>/dev/null; then
    echo "🖥️ Starting Ollama service..."
    ollama serve &
    OLLAMA_PID=$!
    sleep 3
fi

# 7. Verify Ollama is accessible
echo "🤖 Verifying Ollama models..."
source venv/bin/activate
python3 -c "
import ollama
try:
    response = ollama.generate(model='granite4:micro-h', prompt='Hello', options={'num_predict': 5})
    print('✅ Ollama service ready')
except Exception as e:
    print(f'❌ Ollama not accessible: {e}')
    exit 1
"

# 8. Start FastAPI backend
echo "🐍 Starting FastAPI backend on port 8000..."
source venv/bin/activate
python3 main.py &
BACKEND_PID=$!

# 9. Wait for backend to start
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
    kill $OLLAMA_PID 2>/dev/null || true
    exit 1
fi

# 10. Start Next.js frontend
echo "⚛️ Starting Next.js frontend..."
cd frontend
if [ -n "$OLLAMA_PID" ]; then
    # If we started Ollama, keep it running in background
    PORT=3000 npm run dev &
    FRONTEND_PID=$!
else
    PORT=3000 npm run dev &
    FRONTEND_PID=$!
fi
cd ..

# 11. Wait for frontend to start
echo "⏳ Waiting for frontend to start..."
FRONTEND_READY=false
for i in {1..20}; do
    echo "   Checking frontend (attempt $i/20)..."
    if curl -s --max-time 5 http://localhost:3000 > /dev/null 2>&1; then
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
    kill $OLLAMA_PID 2>/dev/null || true
    exit 1
fi

# 12. Display status and access information
echo ""
echo "🎉 Research Assistant GraphRAG Application is running!"
echo ""
echo "🌐 Frontend: http://localhost:3000"
echo "🔧 Backend API: http://localhost:8000"
echo "📚 API Documentation: http://localhost:8000/docs"
echo "📊 Neo4j Browser: http://localhost:7474"
echo ""
echo "📊 Database Status:"
source venv/bin/activate
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

# 13. Function to cleanup processes on exit
cleanup() {
    echo ""
    echo "🛑 Stopping services..."
    kill $BACKEND_PID 2>/dev/null || true
    kill $FRONTEND_PID 2>/dev/null || true
    kill $OLLAMA_PID 2>/dev/null || true
    echo "✅ Services stopped. Goodbye!"
    exit 0
}

# 14. Set trap to cleanup on script termination
trap cleanup INT TERM

# 15. Wait for processes
echo "🛑 Press Ctrl+C to stop all services"
echo ""
echo "💡 Ready to query your research assistant!"
echo "   Try asking questions in the web interface or via API calls."
echo ""

wait $BACKEND_PID $FRONTEND_PID
