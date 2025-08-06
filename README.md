# RAG-Powered Multimodal Voicebot

[ Tech Stack Banner ]
| Python | FastAPI | Redis | PostgreSQL | Celery | WebSocket | LangChain | HuggingFace | FAISS | PyTorch |

[![Python](https://img.shields.io/badge/Python-3.12+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com)
[![Redis](https://img.shields.io/badge/Redis-Cache%20%26%20Queue-red.svg)](https://redis.io)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-Database-blue.svg)](https://postgresql.org)
[![Celery](https://img.shields.io/badge/Celery-Task%20Queue-37B24D.svg)](https://docs.celeryq.dev)
[![WebSocket](https://img.shields.io/badge/WebSocket-Real--time-orange.svg)](https://developer.mozilla.org/en-US/docs/Web/API/WebSockets_API)
[![LangChain](https://img.shields.io/badge/LangChain-RAG-blue.svg)](https://langchain.com)
[![Transformers](https://img.shields.io/badge/🤗%20Transformers-NLP-yellow.svg)](https://huggingface.co/transformers)
[![FAISS](https://img.shields.io/badge/FAISS-Vector%20Search-lightblue.svg)](https://faiss.ai)
[![PyTorch](https://img.shields.io/badge/PyTorch-ML%20Framework-red.svg)](https://pytorch.org)

An AI-powered query system that enables real-time voice interactions, document-augmented question answering, and multi-tenant support through WebSocket and REST APIs. Built with FastAPI, the system combines speech-to-text, retrieval-augmented generation (RAG), and text-to-speech capabilities for voice-to-voice conversations.

## Features

- Real-time voice conversations via WebSocket with Voice Activity Detection (VAD)
- Document intelligence with RAG-powered answers from uploaded documents (PDF, DOCX, TXT, web pages)
- Multi-tenant authentication with JWT and organization management
- Scalable architecture using Celery workers with Redis for background processing
- Hybrid retrieval combining FAISS vector search and BM25
- Multiple TTS options including EdgeTTS and PlayAI
- Real-time monitoring of WebSocket connections

## Architecture

| Component | Technology Stack |
|-----------|-----------------|
| **API Layer** | FastAPI, WebSocket, Swagger/OpenAPI |
| **Queue System** | Celery + Redis |
| **Databases** | PostgreSQL, Redis Cache, FAISS Vector DB |
| **AI Models** | Faster-Whisper, Transformers, BGE Reranker |
| **Audio Processing** | Silero VAD, EdgeTTS, PlayAI |
| **Document Processing** | PyPDF, python-docx, LangChain loaders |
| **Embeddings** | multilingual-e5-base |

## Prerequisites

**System Requirements:**
- Python 3.12+
- Redis Server
- PostgreSQL
- CUDA GPU (24GB VRAM recommended for local deployment)
- ffmpeg and PortAudio for audio processing

**Ubuntu/Debian:**
```bash
sudo apt update && sudo apt install -y \
    ffmpeg \
    portaudio19-dev \
    libsndfile1 \
    redis-server \
    postgresql \
    postgresql-contrib
```

## Setup

### 1. Clone and Install

```bash
git clone https://github.com/your-org/voicebot
cd voicebot
```

### 2. Create Virtual Environment

**Linux/macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Environment Configuration

Copy and configure the environment file:
```bash
cp sample.env .env
```

Edit `.env` with your configuration:
```env
# Database Configuration
DB_PASSWORD=your_secure_password
DB_NAME=voicebot_db
DB_PORT=5432
DB_HOST=localhost
DB_URL=postgresql://postgres:${DB_PASSWORD}@${DB_HOST}:${DB_PORT}/${DB_NAME}

# JWT Configuration
SECRET_KEY=your_secret_key_here  # Generate with: python -c "import secrets; print(secrets.token_urlsafe(64))"
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=10080

# API Keys
GROQ_API_KEY=your_groq_api_key_here
```

### 4. Database Setup

Create PostgreSQL database:
```bash
sudo -u postgres createdb voicebot_db
sudo -u postgres psql -c "ALTER USER postgres PASSWORD 'your_secure_password';"
```

### 5. Start the Application

**Development Mode:**
```bash
./start.sh --dev
```

**Production Mode:**
```bash
./start.sh --port 8000 --workers 4
```

The application will be available at:
- Main API: http://localhost:8000
- API Documentation: http://localhost:8000/docs
- Alternative Docs: http://localhost:8000/redoc

## API Endpoints

### Authentication
- `POST /auth/register-org` - Create organization and admin user
- `POST /auth/register-user` - Register user to existing organization
- `POST /auth/login` - User login
- `GET /auth/user` - Get current user info

### Document Management
- `POST /documents/upload` - Upload documents for RAG processing
- `GET /documents/` - List uploaded documents
- `GET /documents/{identifier}` - Get document details

### Query Processing
- `POST /query/submit` - Submit text query for RAG processing
- `POST /query/submit-with-tts` - Submit query and get TTS response
- `POST /query/audio-to-audio` - Process audio input to audio output
- `GET /query/status/{task_id}` - Check processing status

### Text-to-Speech
- `POST /tts/convert` - Convert text to speech
- `GET /tts/voices` - List available TTS voices
- `GET /tts/download/{filename}` - Download generated audio

### WebSocket
- `WS /ws/{token}` - Real-time audio streaming and processing

## Testing

### Using the Test Client

Run the included test client for end-to-end testing:

```bash
python test_client.py
```

The test client provides:
- Real-time audio recording and streaming
- WebSocket connection management
- Audio playback of TTS responses
- Authentication handling
- Error handling and reconnection

### Manual API Testing

1. **Register a new organization:**
```bash
curl -X POST "http://localhost:8000/auth/register-org" \
  -H "Content-Type: application/json" \
  -d '{"email":"admin@example.com","password":"securepassword","org_name":"Test Org"}'
```

2. **Upload a document:**
```bash
curl -X POST "http://localhost:8000/documents/upload" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -F "file=@document.pdf"
```

3. **Query the system:**
```bash
curl -X POST "http://localhost:8000/query/submit" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"text":"What is this document about?"}'
```

## Configuration

### Model Configuration

The system supports different models based on available hardware in [`config.json`](config.json):

```json
{
  "model_size": {
    "cpu": "medium",
    "cuda": "large-v2"
  },
  "compute_type": {
    "cpu": "int8",
    "cuda": "int8_float16"
  }
}
```

### Audio Processing Settings

```json
{
  "sample_rate": 16000,
  "block_duration": 0.5,
  "min_chunk_duration": 2,
  "max_chunk_duration": 20,
  "silence_threshold": -35,
  "max_silence_duration": 0.7
}
```

### RAG Configuration

```json
{
  "bm25Weight": 0.3,
  "faissWeight": 0.6,
  "reranker_model_id": "BAAI/bge-reranker-base",
  "max_new_tokens": 100,
  "temperature": 0.7
}
```

## Troubleshooting

### Common Issues

**Application won't start:**
```bash
# Check if ports are in use
sudo netstat -tulpn | grep :8000
sudo netstat -tulpn | grep :6379

# Clear Redis cache
redis-cli -n 0 FLUSHDB
redis-cli -n 1 FLUSHDB
```

**Force stop all services:**
```bash
./stop.sh
```

**GPU Memory Issues:**
- Reduce model sizes in `config.json`
- Use CPU models instead of CUDA models
- Reduce batch sizes and worker counts

**Audio Issues:**
- Ensure PortAudio is installed: `sudo apt install portaudio19-dev`
- Check microphone permissions
- Verify sample rate compatibility

### Logging

Enable debug logging:
```bash
export LOG_LEVEL=DEBUG
```

Or modify the application startup:
```bash
uvicorn backend.main:app --log-level debug
```

## Production Deployment

### Performance Considerations

- Use strong secret keys and passwords
- Configure SSL/TLS certificates
- Set up proper monitoring and logging
- Use a production WSGI server like Gunicorn
- Configure firewall rules
- Set up backup strategies for databases
- Monitor resource usage and scale accordingly

### Optimization

- Use GPU acceleration when available
- Configure Redis persistence based on needs
- Optimize database queries and indexes
- Implement caching strategies
- Monitor and tune Celery worker counts

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/new-feature`
3. Commit changes: `git commit -am 'Add new feature'`
4. Push to branch: `git push origin feature/new-feature`
5. Submit a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.



<table>
    <tr>
        <td align="center">
            <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/python/python-original.svg" width="40"/><br/>Python
        </td>
        <td align="center">
            <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/fastapi/fastapi-original.svg" width="40"/><br/>FastAPI
        </td>
        <td align="center">
            <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/redis/redis-original.svg" width="40"/><br/>Redis
        </td>
        <td align="center">
            <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/1/19/Celery_logo.png/250px-Celery_logo.png" width="50"/><br/>Celery
        </td>
        <td align="center">
            <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/pytorch/pytorch-original.svg" width="40"/><br/>PyTorch
        </td>
        <td align="center">
            <img src="https://huggingface.co/front/assets/huggingface_logo-noborder.svg" width="40"/><br/>Hugging Face
        </td>
        <!-- <td align="center">
            <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/openai/openai-original.svg" width="40"/><br/>OpenAI
        </td> -->
        <!-- <td align="center">
            <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/docker/docker-original.svg" width="40"/><br/>Docker
        </td> -->
        <td align="center">
            <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/c/cd/WebSocket_colored_logo.svg/250px-WebSocket_colored_logo.svg.png
            " width="40"/><br/>Websockets
        </td>
</table>
