# 🧠 Research Assistant GraphRAG System

A comprehensive, AI-powered research assistant that combines Graph Database technology with Retrieval-Augmented Generation (GraphRAG) to provide intelligent document processing, advanced entity extraction, and knowledge graph construction. Features MCP (Model Context Protocol) integration for enhanced document upload and processing capabilities.

## ✨ Features

### Core Capabilities
- **📤 Multi-format Document Upload**: Support for CSV, PDF, TXT, JSON, and Markdown files
- **🎯 Advanced Entity Extraction**: Rule-based + LLM-powered entity detection with confidence scoring
- **🕸️ Automated Knowledge Graph**: Neo4j-powered graph construction with relationship inference
- **🧠 MCP Integration**: Model Context Protocol support for enhanced AI interactions
- **💬 Intelligent Chat System**: Context-aware responses with entity references and citations
- **📊 Real-time Processing**: Live progress tracking for document processing tasks
- **🔍 Graph-based Search**: Efficient retrieval of interconnected information
- **⚡ High Performance**: Optimized for batch processing and large document collections

### Supported Entity Types
- **Person**: Researchers, authors, executives
- **Organization**: Companies, universities, research institutions
- **Technology**: Tools, frameworks, algorithms, software systems
- **Concept**: Theories, methodologies, scientific paradigms
- **Location**: Geographic entities, research facilities

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   File Upload   │───▶│  Entity Extract │───▶│ Graph Construct │
│   & Validation  │    │  & Processing   │    │  & Storage      │
│                 │    │                 │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
        │                        │                        │
        ▼                        ▼                        ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  MCP Server     │───▶│  Reasoning      │───▶│   Query         │
│  Integration    │    │  Engine         │    │   Response      │
│                 │    │                 │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Technology Stack
- **Backend**: Python 3.9+, FastAPI, Uvicorn
- **Frontend**: Next.js 16, React 19, TypeScript, Tailwind CSS
- **Database**: Neo4j (Graph Database), Redis (Caching)
- **AI/ML**: Ollama (LLM), Granite4 model, Custom entity extraction
- **Infrastructure**: Docker, Docker Compose
- **Protocols**: MCP (Model Context Protocol)

## 🚀 Quick Start

### Prerequisites

Before running the system, ensure you have the following installed:

- **Python 3.9+** with pip
- **Node.js 16+** with npm
- **Docker & Docker Compose** (for Neo4j and Redis)
- **Ollama** (for local LLM support)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/research-assistant-graphrag.git
   cd research-assistant-graphrag
   ```

2. **Run the automated setup script**
   ```bash
   ./setup.sh
   ```
   This script will:
   - Create a Python virtual environment
   - Install all Python and Node.js dependencies
   - Start Docker services (Neo4j, Redis)
   - Download required Ollama models
   - Create necessary directories and indexes

3. **Start the application**
   ```bash
   ./start.sh
   ```
   This will start:
   - Neo4j database (Graph database)
   - Redis (Caching)
   - Ollama service (LLM)
   - FastAPI backend server (port 8000)
   - Next.js frontend (port 3000)

4. **Access the application**
   - **Frontend**: http://localhost:3000
   - **API Documentation**: http://localhost:8000/docs
   - **Neo4j Browser**: http://localhost:7474 (auth: neo4j/research2025)
   - **Health Check**: http://localhost:8000/api/health

## 📤 Document Upload & Processing

### File Upload API

**Upload Files** (POST `/api/upload/files`):
```bash
curl -X POST "http://localhost:8000/api/upload/files" \
     -F "files=@document.pdf" \
     -F "files=@research_paper.txt"
```

**Process Uploaded Files** (POST `/api/upload/process`):
```bash
curl -X POST "http://localhost:8000/api/upload/process" \
     -H "Content-Type: application/json" \
     -d '{
       "session_id": "your-session-id",
       "entity_types": ["Person", "Organization", "Technology", "Concept"],
       "confidence_threshold": 0.6,
       "max_chunk_size": 1000,
       "overlap_size": 200
     }'
```

**Check Processing Progress** (GET `/api/upload/progress/{task_id}`):
```bash
curl http://localhost:8000/api/upload/progress/your-task-id
```

### Supported File Formats
- **PDF**: Research papers, technical documents
- **TXT**: Plain text files
- **JSON**: Structured data
- **CSV**: Tabular data
- **Markdown**: Documentation files

**Limits**:
- Maximum 20 files per upload
- Individual file size limit: 50MB
- Total upload size limit: 200MB

## 💬 Chat & Query System

### Basic Chat
```bash
curl -X POST "http://localhost:8000/api/chat" \
     -H "Content-Type: application/json" \
     -d '{
       "message": "What are the key findings about transformers in the uploaded papers?",
       "context": "research_papers"
     }'
```

### Advanced Query Features
- **Entity-aware responses**: Returns relevant entities with confidence scores
- **Citation tracking**: References source documents and sections
- **Topic hierarchy**: Organizes responses by related concepts
- **Co-occurrence analysis**: Shows interconnected ideas

## 🔧 Configuration

### Environment Variables

Create a `.env` file in the root directory:

```bash
# Neo4j Configuration
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=research2025

# Ollama Configuration
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=granite4:micro-h

# Redis Configuration
REDIS_URL=redis://localhost:6379

# Application Settings
UPLOAD_BATCH_SIZE=10
MAX_FILE_SIZE=50000000
CONTEXT_WINDOW_SIZE=8000
CONFIDENCE_THRESHOLD=0.6

# MCP Settings
MCP_SERVER_PORT=3001
MCP_MAX_CONTEXT_TOKENS=16000
```

### Neo4j Setup

The system uses Neo4j for graph storage. Default credentials:
- **Username**: neo4j
- **Password**: research2025
- **Bolt Port**: 7687
- **Browser Port**: 7474

### Ollama Models

Required Ollama models:
- `granite4:micro-h` (Primary reasoning model)
- `mxbai-embed-large:latest` (Embedding model for similarity)

## 🧪 Development & Testing

### Running Tests
```bash
# Backend tests
python -m pytest

# Frontend tests
cd frontend && npm test
```

### Development Mode
```bash
# Start backend in development mode
python main.py --reload

# Start frontend in development mode
cd frontend && npm run dev
```

### Database Management
```bash
# Create indexes
python create_indexes.py

# Create thread relationships
python create_thread_relationships.py

# Debug database
python debug_db.py
```

## 📊 Monitoring & Health Checks

### API Endpoints

- **Health Check**: `GET /api/health`
```json
{
  "status": "healthy",
  "timestamp": "2025-11-18T12:00:00.000Z",
  "services": {
    "neo4j": "connected",
    "ollama": "ready",
    "redis": "connected"
  }
}
```

- **System Status**: `GET /api/status`
```json
{
  "total_documents": 1250,
  "total_entities": 3420,
  "total_topics": 450,
  "uptime_seconds": 3600
}
```

### Logs

- Neo4j logs: `neo4j-logs/`
- Application logs: Check console output from running services
- Docker logs: `docker-compose logs`

## 🚢 Deployment

### Development Deployment

For local development with all services:
```bash
# Use the simple setup and start scripts
./setup.sh  # One-time setup
./start.sh  # Start all services
```

### Docker Development Deployment
```bash
# Start services using Docker Compose
docker-compose up --build

# Start in background
docker-compose up -d
```

### Production Docker Deployment

For production deployment with proper orchestration:

1. **Production Setup**
   ```bash
   # Copy production environment variables
   cp .env.example .env
   # Edit .env with your production values

   # Start production services
   docker-compose -f docker-compose.prod.yml up --build -d

   # To include Ollama and Nginx
   docker-compose -f docker-compose.prod.yml --profile with-ollama --profile with-nginx up -d
   ```

2. **Production Configuration**
   ```bash
   # Required environment variables for production
   NEO4J_URI=bolt://your-neo4j-host:7687
   NEO4J_USERNAME=production_user
   NEO4J_PASSWORD=secure_password_123
   REDIS_URL=redis://your-redis-cluster:6379
   DEBUG=false
   SECRET_KEY=your-production-secret-key-min-32-chars
   LOG_LEVEL=INFO
   HOST=0.0.0.0
   FRONTEND_URL=https://yourdomain.com
   ALLOWED_ORIGINS=https://yourdomain.com
   ```

### Production Services Configuration

#### Neo4j Enterprise Setup
```yaml
# docker-compose.prod.yml (relevant section)
neo4j:
  image: neo4j:5.20-enterprise
  environment:
    - NEO4J_AUTH=${NEO4J_USERNAME}/${NEO4J_PASSWORD}
    - NEO4J_PLUGINS=["apoc", "graph-data-science"]
    - NEO4J_dbms_memory_heap_initial__size=512m
    - NEO4J_dbms_memory_heap_max__size=4G
    - NEO4J_dbms_memory_pagecache_size=2G
```

#### Scaling Considerations
- **Neo4j**: Use Neo4j Enterprise with clustering for high availability
- **Redis**: Use Redis Cluster for horizontal scaling and persistence
- **Application**: Deploy with load balancer and multiple instances
- **Ollama**: Use dedicated GPU instances for better LLM performance

#### Reverse Proxy with Nginx
The production setup includes optional Nginx reverse proxy with:
- **Rate limiting**: Prevent API abuse
- **Load balancing**: Distribute requests across instances
- **SSL termination**: Handle HTTPS certificates
- **Security headers**: Add security-related HTTP headers
- **Static file serving**: Optimize delivery of assets

#### Monitoring and Health Checks
```bash
# Check service health
curl http://localhost/api/health

# View service logs
docker-compose -f docker-compose.prod.yml logs -f

# Monitor resource usage
docker stats
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup
```bash
# Install development dependencies
pip install -r requirements-dev.txt
cd frontend && npm install

# Run linting
python -m flake8
cd frontend && npm run lint

# Run tests
python -m pytest
cd frontend && npm test
```

## 📚 API Reference

### Upload Endpoints
- `POST /api/upload/files` - Upload multiple files
- `POST /api/upload/process` - Start document processing
- `GET /api/upload/progress/{task_id}` - Get processing status
- `DELETE /api/upload/progress/{task_id}` - Cancel processing

### Chat Endpoints
- `POST /api/chat` - Send chat messages
- `GET /api/search` - Search documents and entities
- `POST /api/graph-rag/query` - Advanced graph-based queries

### System Endpoints
- `GET /api/health` - Health check
- `GET /api/status` - System statistics
- `GET /api/evaluation-results` - Performance metrics

### MCP Endpoints
- `GET /mcp/documents/{id}` - Access document via MCP
- `GET /mcp/entities/{id}` - Access entity information
- `POST /mcp/tools/search` - MCP tool for graph search

## 🐛 Troubleshooting

### Common Issues

1. **Neo4j Connection Failed**
   - Ensure Docker services are running: `docker-compose ps`
   - Check Neo4j logs: `docker-compose logs neo4j`
   - Verify credentials in `.env` file

2. **Ollama Not Accessible**
   - Start Ollama: `ollama serve`
   - Pull required models: `ollama pull granite4:micro-h`
   - Check Ollama status: `ollama list`

3. **Port Conflicts**
   - Frontend defaults to port 3000, backend to 8000
   - Check what's using ports: `lsof -i :3000`
   - Modify ports in configuration if needed

4. **Memory Issues**
   - Increase Docker memory limits
   - Optimize chunk sizes in configuration
   - Consider GPU acceleration for Ollama

### Database Reset
```bash
# Stop all services
docker-compose down

# Remove database volumes
docker volume rm research-assistant-graphrag_neo4j-data research-assistant-graphrag_redis-data

# Rebuild and restart
docker-compose up --build
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built with [FastAPI](https://fastapi.tiangolo.com/)
- Graph database powered by [Neo4j](https://neo4j.com/)
- LLM integration via [Ollama](https://ollama.ai/)
- MCP implementation based on [Model Context Protocol](https://modelcontextprotocol.io/)
- UI components from [Radix UI](https://www.radix-ui.com/)

## 📞 Support

For support, please:
1. Check the [troubleshooting section](#-troubleshooting)
2. Review [existing issues](https://github.com/yourusername/research-assistant-graphrag/issues)
3. Open a new issue with detailed information about your problem

## 🗺️ Roadmap

- [ ] Multi-language document support
- [ ] Advanced graph visualization
- [ ] Integration with external APIs (ArXiv, PubMed)
- [ ] Machine learning model training on extracted entities
- [ ] Real-time collaborative features
- [ ] Mobile application companion
