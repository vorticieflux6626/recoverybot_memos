# memOS Project Summary

> **Updated**: 2025-12-30 | **Parent**: [Root CLAUDE.md](../CLAUDE.md) | **Version**: 0.39.0

## Project Overview
memOS is a sophisticated **intelligent research and memory management system** for the Recovery Bot platform. Originally designed for recovery community services, it has evolved into a general-purpose AI research assistant with advanced agentic search capabilities. The system combines:

- **Multi-Provider Web Search**: SearXNG metasearch, DuckDuckGo, Brave API
- **LLM-Powered Synthesis**: DeepSeek R1 classification, Qwen3 synthesis
- **Memory Management**: HIPAA-compliant storage with semantic search
- **Quest/Gamification**: Progress tracking and achievement systems
- **TTS Integration**: 6 TTS engines including EmotiVoice, Edge-TTS, Sherpa-ONNX

**Current Module Version**: v0.39.0 (Phase 26: Feature Synergy Integration)

## Architecture Overview

### Core Components
- **FastAPI Server** - Async web server with comprehensive middleware and security
- **PostgreSQL Database** - HIPAA-compliant data storage with pgvector for semantic search
- **Ollama Integration** - Local AI for embeddings and therapeutic context analysis
- **Docker Deployment** - Multi-service containerized architecture
- **Android Client Support** - Mobile-optimized API endpoints and response models

### Directory Structure
```
memOS/
├── server/                        # Main FastAPI application
│   ├── main.py                   # Application entry point (FastAPI + middleware)
│   ├── api/                      # REST API endpoints (6 modules)
│   │   ├── search.py            # Agentic search (80+ endpoints, 6000+ lines)
│   │   ├── memory.py            # Memory CRUD with HIPAA compliance
│   │   ├── quest.py             # Quest system and gamification
│   │   ├── user.py              # User settings and preferences
│   │   ├── auth.py              # JWT authentication
│   │   ├── tts.py               # Text-to-speech (EmotiVoice, Edge-TTS, OpenVoice)
│   │   └── health.py            # Health monitoring
│   ├── agentic/                  # Agentic search module (75+ files)
│   │   ├── orchestrator_universal.py  # UniversalOrchestrator (SSOT)
│   │   ├── analyzer.py          # Query analysis
│   │   ├── searcher.py          # Web search (SearXNG, Brave, DDG)
│   │   ├── synthesizer.py       # LLM-powered synthesis
│   │   ├── domain_corpus.py     # Domain-specific knowledge
│   │   ├── bge_m3_hybrid.py     # Hybrid retrieval
│   │   ├── hyde.py              # Query expansion
│   │   ├── ragas.py             # Quality evaluation
│   │   └── ...                  # 65+ additional modules
│   ├── core/                     # Business logic services
│   │   ├── memory_service.py    # Encrypted memory storage
│   │   ├── quest_service_fixed.py # Gamification system
│   │   ├── embedding_service.py # Ollama embeddings
│   │   ├── document_graph_service.py # PDF API integration
│   │   └── exceptions.py        # Unified exception handling
│   ├── config/                   # Configuration management
│   ├── models/                   # SQLAlchemy & Pydantic models
│   ├── data/                     # Databases (SQLite, caches)
│   └── tests/                    # Comprehensive test suite (30+ test files)
├── CLAUDE.md                     # AI assistant guidance (2700+ lines)
└── docker-compose.yml           # Multi-service development
```

## Technology Stack

### Backend Technologies
- **Python 3.11+** with FastAPI framework
- **Async/await** throughout for high performance
- **SQLAlchemy 2.0** with async sessions and pgvector extension
- **Pydantic v2** for data validation and serialization
- **PostgreSQL 15** with pgvector for vector similarity search

### AI/ML Integration
- **Ollama** for local LLM processing with `mxbai-embed-large` model
- **Mem0 framework** for advanced memory management
- **1024-dimensional vector embeddings** for semantic search
- **Therapeutic context scoring** and crisis detection

### Security & Compliance
- **HIPAA-compliant design** throughout all components
- **AES-256 encryption** for sensitive data at rest
- **JWT authentication** with refresh tokens
- **Comprehensive audit logging** with 7-year retention
- **Row-level security** and privacy controls

## Key Features & Capabilities

### 1. Memory Management System
- **8 Memory Types**: Conversational, Clinical, Crisis, Recovery, Resource, Procedural, Episodic, Semantic
- **4 Privacy Levels**: Minimal, Balanced, Comprehensive, Restricted
- **Semantic Search**: AI-powered similarity search with therapeutic context
- **HIPAA Compliance**: Full encryption, audit trails, and consent management
- **Smart Retention**: Configurable retention policies with automatic expiration

### 2. Quest/Gamification System
- **8 Quest Categories**: Daily, Weekly, Milestone, Life Skills, Community, Emergency, Wellness, Spiritual
- **16 Pre-built Quests** aligned with recovery journey stages
- **Achievement System**: Points, levels, and streak tracking
- **Progress Tracking**: Task-based completion with evidence collection
- **7-Level Progression**: NEWCOMER → SEEKER → PATHFINDER → WARRIOR → CHAMPION → MENTOR → LEGEND

### 3. REST API Layer
- **25+ Endpoints** across 5 API modules
- **CRUD Operations** for all major entities
- **Android-compatible responses** with optimized data structures
- **Comprehensive error handling** with structured responses
- **Health monitoring** with service status reporting

### 4. Database Layer
- **9 Core Tables** for memory and quest management
- **PostgreSQL with pgvector** for vector similarity search
- **Custom SQL functions** for therapeutic memory calculations
- **ACID transactions** with proper rollback handling
- **Comprehensive indexing** for performance optimization

## Current Status (December 2025)

### ✅ Completed Components (26 Phases)

**Agentic Search System (v0.39.0)**:
- **UniversalOrchestrator**: Single source of truth with 5 presets (minimal→full)
- **Multi-Provider Search**: SearXNG, DuckDuckGo, Brave with cascading fallback
- **Advanced Retrieval**: BGE-M3 hybrid, HyDE expansion, mixed-precision embeddings
- **Quality Control**: CRAG evaluation, Self-RAG reflection, RAGAS metrics
- **Domain Knowledge**: FANUC robotics, Raspberry Pi troubleshooting corpuses
- **SSE Streaming**: Real-time graph visualization `[A]→[P]→[S]→[E]→[W]→[V]→[Σ]→[R]→[✓]`

**Memory & Quest System**:
- **Memory management** with HIPAA compliance and semantic search
- **Quest system** with 16 sample quests across 8 categories
- **80+ REST API endpoints** across search, memory, quest, TTS modules

**TTS Integration**:
- **6 TTS engines**: EmotiVoice, Edge-TTS, OpenVoice, eSpeak-NG, Piper, Sherpa-ONNX
- **Emotion control**: Seductive, empathetic, encouraging voice presets
- **322 Neural Voices**: Microsoft Edge voices with pitch/rate control

### Agentic Search Feature Matrix

| Preset | Features | Use Case |
|--------|----------|----------|
| `minimal` | 8 | Fast, simple queries |
| `balanced` | 18 | Default for most queries (Android default) |
| `enhanced` | 28 | Complex research |
| `research` | 39 | Academic/thorough (dynamic planning + graph cache) |
| `full` | 42+ | Maximum capability (multi-agent coordination) |

### 🔧 Known Issues
1. **Async/Greenlet Conflicts** - Some quest service operations still have event loop conflicts
2. **Session Management** - SQLAlchemy async sessions partially refactored

### 🚀 Active Development
1. **HSEA Controller** - Industrial troubleshooting with semantic search
2. **IMM Corpus Builder** - Injection molding machine knowledge base
3. **PDF API Integration** - Technical documentation extraction

See `CLAUDE.md` and `server/agentic/AGENTIC_OVERVIEW.md` for detailed documentation.

## Development Workflow

### Server Management
```bash
# Start server with health monitoring
./memos_server.sh start

# Check status and health endpoints  
./memos_server.sh status

# View real-time logs
./memos_server.sh logs

# Restart for updates
./memos_server.sh restart
```

### Database Operations
```bash
# Initialize schema and extensions
python init_database.py --create

# Populate sample quest data
python create_sample_quests.py --create

# Run test suites
python test_quest_simple.py
python test_memory_chain.py
```

### Docker Development
```bash
# Start all services (postgres, redis, chromadb, etc.)
docker-compose up -d

# View service logs
docker-compose logs -f memos-server

# Health check all services
curl http://localhost:8001/health
```

## API Documentation

### Core Endpoints
- **Memory Management**: `/api/v1/memory/*` - Store, search, retrieve memories
- **Quest System**: `/api/v1/quests/*` - Quest assignment and progress tracking
- **User Management**: `/api/v1/user/*` - Settings and consent management
- **Health Monitoring**: `/api/v1/health/*` - System status and diagnostics
- **Authentication**: `/auth/*` - JWT login and token management

### Response Format
All API responses follow consistent structure with proper HTTP status codes, error handling, and HIPAA-compliant audit logging.

## Security & Compliance

### HIPAA Requirements
- **Encryption at Rest**: AES-256 for all sensitive data
- **Audit Logging**: Complete trails for all data access
- **Consent Management**: Granular consent with renewal tracking
- **Data Retention**: 7-year default with configurable policies
- **Access Controls**: User-based authorization for all resources

### Security Features  
- **JWT Authentication** with refresh token support
- **CORS Configuration** with trusted host validation
- **Rate Limiting** and request validation
- **PHI Detection** with automatic anonymization
- **Secure Deletion** with audit trail preservation

## Performance & Monitoring

### Optimization Features
- **Vector Caching**: LRU cache for embeddings and similarity calculations
- **Connection Pooling**: Optimized database connections (10 base, 20 overflow)
- **Async Operations**: Full async/await implementation
- **Custom Indexes**: Performance-optimized database indexes

### Health Monitoring
- **Comprehensive Health Checks**: Database, Ollama, encryption, and memory services
- **Structured Logging**: JSON logging with therapeutic context
- **Service Status**: Real-time monitoring of all system components
- **Performance Metrics**: Request timing and resource usage tracking

## Integration Ready

The memOS system is architecturally prepared for:
- **Android Client Integration** with optimized API responses
- **Offline Synchronization** capabilities for mobile use
- **Push Notifications** for quest reminders and achievements  
- **Care Team Collaboration** with secure data sharing
- **Crisis Intervention** integration with emergency protocols

## Technical Architecture Strengths

1. **Service-Oriented Design** - Clear separation of concerns with dependency injection
2. **HIPAA Compliance** - Built-in privacy, encryption, and audit requirements
3. **Production-Ready** - Comprehensive configuration, testing, and deployment
4. **Scalable Architecture** - Container-based with proper resource management
5. **Developer Experience** - Excellent tooling, testing, and documentation
6. **Therapeutic Focus** - Recovery-specific features and therapeutic context awareness

This memOS implementation represents a sophisticated, production-ready memory management and gamification system specifically designed for therapeutic applications requiring strict HIPAA compliance and advanced AI capabilities for supporting recovery journeys.