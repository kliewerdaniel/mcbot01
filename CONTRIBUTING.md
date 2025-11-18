# Contributing to Research Assistant GraphRAG System

Thank you for your interest in contributing to the Research Assistant GraphRAG System! We welcome contributions from the community to help improve and extend this AI-powered research assistant.

## 📋 Table of Contents
- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Environment Setup](#development-environment-setup)
- [Project Structure](#project-structure)
- [Development Guidelines](#development-guidelines)
- [Making Contributions](#making-contributions)
- [Testing](#testing)
- [Documentation](#documentation)
- [Code Review Process](#code-review-process)

## 🤝 Code of Conduct

This project follows a code of conduct to ensure a welcoming environment for all contributors. By participating, you agree to:
- Be respectful and inclusive
- Focus on constructive feedback
- Accept responsibility for mistakes
- Show empathy towards other contributors
- Help create a positive community

## 🚀 Getting Started

### Prerequisites
Before you begin, ensure you have the following installed:
- **Python 3.9 or higher**
- **Node.js 16 or higher**
- **Docker & Docker Compose**
- **Ollama** (for local LLM support)
- **Git**

### Quick Setup
1. **Fork and clone the repository**
   ```bash
   git clone https://github.com/yourusername/research-assistant-graphrag.git
   cd research-assistant-graphrag
   ```

2. **Set up the development environment**
   ```bash
   ./setup.sh
   ```

3. **Start the development servers**
   ```bash
   ./start.sh
   ```

4. **Access the application**
   - Frontend: http://localhost:3000
   - API: http://localhost:8000
   - Documentation: http://localhost:8000/docs

## 🛠️ Development Environment Setup

### Backend (Python/FastAPI)
```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# For development dependencies
pip install -r requirements-dev.txt
```

### Frontend (Next.js)
```bash
cd frontend
npm install
```

### Database Services
```bash
# Start Neo4j and Redis
docker-compose up -d neo4j redis

# Wait for services to be ready
sleep 30
```

### LLM Setup
```bash
# Start Ollama service
ollama serve

# Pull required models
ollama pull granite4:micro-h
ollama pull mxbai-embed-large:latest
```

## 📁 Project Structure

```
research-assistant-graphrag/
├── scripts/                 # Python backend scripts and modules
│   ├── mcp/                # MCP server implementation
│   ├── *.py                # Core processing scripts
│   └── __init__.py
├── frontend/               # Next.js frontend application
│   ├── src/
│   │   ├── app/           # Next.js app directory
│   │   ├── components/    # React components
│   │   └── lib/           # Utility functions
│   └── package.json
├── data/                   # Data storage and sample files
├── evaluation/             # Evaluation and testing framework
├── docs/                   # Documentation (if separate from README)
├── docker-compose.yml      # Docker services configuration
├── requirements.txt        # Python dependencies
├── setup.sh               # Initial setup script
├── start.sh               # Application startup script
└── README.md              # Main documentation
```

## 💻 Development Guidelines

### Code Style

#### Python (Backend)
- Follow [PEP 8](https://peps.python.org/pep-0008/) style guidelines
- Use type hints for function parameters and return values
- Maximum line length: 88 characters
- Use `black` formatter and `isort` for imports
- Run linting: `flake8`

#### TypeScript/JavaScript (Frontend)
- Use TypeScript for all new code
- Follow the project's ESLint configuration
- Use functional components with hooks
- Maximum line length: 100 characters

### Commit Messages
Use clear, descriptive commit messages following this format:
```
type: Brief description of changes

- Detailed explanation if needed
- Additional context or breaking changes
```

Types:
- `feat`: New features
- `fix`: Bug fixes
- `docs`: Documentation changes
- `style`: Code style changes
- `refactor`: Code refactoring
- `test`: Testing related changes
- `chore`: Maintenance tasks

### Branch Naming
- `feature/description-of-feature`
- `bugfix/issue-description`
- `docs/update-documentation`
- `refactor/component-name`

## 🤲 Making Contributions

### Reporting Issues
1. **Check existing issues** to avoid duplicates
2. **Use issue templates** when available
3. **Provide detailed information**:
   - Steps to reproduce
   - Expected vs actual behavior
   - Environment details (OS, Python/Node versions)
   - Error messages and stack traces
   - Screenshots if relevant

### Feature Requests
1. **Describe the problem** you want to solve
2. **Explain your proposed solution**
3. **Consider alternative approaches**
4. **Discuss potential impacts**

### Pull Request Process
1. **Create a feature branch** from `main`
2. **Make your changes** following the guidelines
3. **Write or update tests** as needed
4. **Update documentation** if required
5. **Run the full test suite**
6. **Submit a pull request** with a clear description

### PR Checklist
- [ ] Code follows project style guidelines
- [ ] Tests pass locally
- [ ] Documentation updated
- [ ] No new linting errors
- [ ] Commit messages are clear and descriptive
- [ ] PR description explains the changes and their purpose

## 🧪 Testing

### Backend Testing
```bash
# Run all tests
python -m pytest

# Run with coverage
python -m pytest --cov=scripts --cov-report=html

# Run specific test file
python -m pytest tests/test_specific_functionality.py
```

### Frontend Testing
```bash
cd frontend

# Run tests
npm test

# Run tests with coverage
npm test -- --coverage
```

### Integration Testing
```bash
# Test the complete pipeline
python -m pytest tests/integration/

# Test API endpoints
python test_api_endpoints.py
```

### Writing Tests
- Write unit tests for individual functions
- Create integration tests for API endpoints
- Ensure test coverage above 80%
- Use descriptive test names and docstrings

## 📚 Documentation

### Code Documentation
- Add docstrings to all public functions/classes
- Document complex algorithms and business logic
- Update type hints regularly
- Keep README.md and API docs current

### Updating Documentation
```bash
# Update API documentation (if using automatic generation)
python scripts/generate_api_docs.py

# Build frontend documentation
cd frontend && npm run build-docs
```

## 🔍 Code Review Process

### Review Checklist
- [ ] Code style and formatting
- [ ] Test coverage and quality
- [ ] Security implications
- [ ] Performance considerations
- [ ] Documentation completeness
- [ ] Breaking changes identified

### Review Guidelines
- **Be constructive** and provide actionable feedback
- **Explain rationale** behind suggestions
- **Consider edge cases** and error handling
- **Check for security vulnerabilities**
- **Verify tests are adequate**

## 🚀 Deployment and Release

### Versioning
This project follows [Semantic Versioning](https://semver.org/):
- `MAJOR.MINOR.PATCH`
- Breaking changes increment MAJOR
- New features increment MINOR
- Bug fixes increment PATCH

### Release Process
1. **Update version** in relevant files
2. **Update changelog** with new features and fixes
3. **Run full test suite**
4. **Create git tag**
5. **Create GitHub release**
6. **Deploy to production**

## 📞 Getting Help

### Community Resources
- **GitHub Issues**: For bugs and feature requests
- **GitHub Discussions**: For questions and general discussion
- **Documentation**: Check the README and in-code docs first

### Communication
- **Be respectful** and patient when asking questions
- **Search existing resources** before posting
- **Provide context** when asking for help
- **Follow up** on your own issues

## 🎯 Areas Needing Contribution

### High Priority
- [ ] Additional entity types support
- [ ] Improved error handling and logging
- [ ] Performance optimizations
- [ ] Additional file format support

### Medium Priority
- [ ] Graph visualization components
- [ ] Advanced search and filtering
- [ ] Batch processing improvements
- [ ] User authentication and authorization

### Future Enhancements
- [ ] Mobile application
- [ ] Multi-language support
- [ ] Real-time collaboration features
- [ ] Integration with external APIs (ArXiv, PubMed)

## 🙏 Recognition

Contributors will be recognized in:
- Repository contributors list
- Release notes and changelogs
- Project documentation
- Community acknowledgments

Thank you for contributing to the Research Assistant GraphRAG System! 🎉
