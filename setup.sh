#!/bin/bash

echo "🔬 Setting up Research Assistant GraphRAG System with vero-eval (first-time setup)"

# 1. Check prerequisites
echo "📋 Checking prerequisites..."
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 not found. Please install Python 3.9+"
    exit 1
fi

if ! command -v node &> /dev/null; then
    echo "❌ Node.js not found. Please install Node.js 16+"
    exit 1
fi

if ! command -v ollama &> /dev/null; then
    echo "⚠️  Ollama not found. Installing Ollama..."
    # On macOS, install via Homebrew
    if command -v brew &> /dev/null; then
        brew install ollama
    else
        echo "Please install Ollama from https://ollama.ai"
        exit 1
    fi
fi

# 2. Create virtual environment and install Python dependencies
echo "📦 Setting up Python environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi
source venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt

# 3. Setup frontend dependencies
echo "⚛️ Setting up frontend dependencies..."
cd frontend && npm install --force && cd ..

# 4. Create required directories
echo "📁 Creating required directories..."
mkdir -p data/sample_papers
mkdir -p evaluation/results

# 5. Setup Docker services for initial data ingestion
echo "🐳 Starting Docker services for setup..."
if command -v docker-compose &> /dev/null || command -v docker &> /dev/null && docker compose version &> /dev/null; then
    if command -v docker-compose &> /dev/null; then
        docker-compose up -d
    else
        docker compose up -d
    fi
    echo "⏳ Waiting for Neo4j to start (this may take a minute)..."
    sleep 30
else
    echo "⚠️  Docker Compose not found. Please start Neo4j manually for setup:"
    echo "   docker-compose up -d"
    echo "   Or install Docker Desktop"
    exit 1
fi

# 6. Verify Neo4j connection
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

# 7. Pull required Ollama models
echo "🤖 Setting up Ollama models..."
ollama pull granite4:micro-h
ollama pull mxbai-embed-large:latest

# 8. Test Ollama models
echo "🧪 Testing Ollama models..."
source venv/bin/activate
python3 -c "
import ollama
try:
    response = ollama.generate(model='granite4:micro-h', prompt='Hello', options={'num_predict': 10})
    print('✅ Ollama granite4:micro-h model ready')
except Exception as e:
    print(f'⚠️  Ollama test failed: {e}')
    exit 1
"

# 9. Create Neo4j schema and indexes
echo "📊 Creating Neo4j graph schema and indexes..."
source venv/bin/activate
python3 scripts/ingest_research_data.py --setup-indexes

# 10. Check for EPS data file and perform initial ingestion if present
if [ -f "EPS_FILES_20K_NOV2026.csv" ]; then
    echo "📥 Found EPS data file - performing initial data ingestion..."
    source venv/bin/activate
    python3 scripts/ingest_eps_data.py --csv EPS_FILES_20K_NOV2026.csv --create-indexes --create-similarities
    echo "✅ Initial EPS data ingestion completed"
else
    echo "⚠️  EPS_FILES_20K_NOV2026.csv not found - skipping initial EPS ingestion"
    echo "   Place your EPS data file in the root directory and run ingestion manually if needed"
fi

# 11. Create additional indexes and relationships
echo "🔗 Creating additional indexes and relationships..."
source venv/bin/activate
python3 create_indexes.py 2>/dev/null || echo "⚠️  Additional index creation skipped"
python3 create_thread_relationships.py 2>/dev/null || echo "⚠️  Thread relationships creation skipped"

# 12. Run initial evaluation
echo "🧪 Running initial evaluation..."
source venv/bin/activate
python3 evaluation/run_evaluation.py 2>/dev/null || echo "⚠️  Initial evaluation failed - run manually later"

# 13. Stop Docker services (they will be started by start.sh)
echo "🐳 Stopping Docker services (will be restarted by start.sh)..."
if command -v docker-compose &> /dev/null; then
    docker-compose down
else
    docker compose down
fi

echo ""
echo "🎉 Setup complete!"
echo ""
echo "📋 Next steps:"
echo "   1. Start the application: ./start.sh"
echo "   2. Add research papers: cp your_papers/*.pdf data/research_papers/"
echo "   3. Run additional ingestion: python3 scripts/ingest_research_data.py"
echo "   4. Run evaluation: python3 evaluation/run_evaluation.py"
echo ""
echo "🔗 After starting, access points will be:"
echo "   - Frontend: http://localhost:3000 (or 3001)"
echo "   - Backend API: http://localhost:8000"
echo "   - Neo4j Browser: http://localhost:7474"
echo ""
echo "🚀 Your research assistant is ready to start!"
