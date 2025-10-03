# FDE API - Faith Driven Investor

Simple API for serving Faith Driven Investor content to ElevenLabs conversational agent.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Configure environment variables:
```bash
cp .env.example .env
# Edit .env with your credentials
```

3. Run the server:
```bash
python main.py
```

## Endpoints

- `GET /health` - Health check
- `POST /search/vector` - Vector search FDI content
- `POST /chat` - Chat with context retrieval
