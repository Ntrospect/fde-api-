"""
Simplified FDE API for ElevenLabs Integration
"""

import os
import logging
from datetime import datetime
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import asyncpg
import uvicorn
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Database connection pool
db_pool: Optional[asyncpg.Pool] = None


# Pydantic models
class ChatRequest(BaseModel):
    message: str
    user_id: str = "elevenlabs_user"
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    message: str
    session_id: Optional[str] = None
    timestamp: datetime


class SearchRequest(BaseModel):
    query: str
    limit: int = 5


class SearchResult(BaseModel):
    content: str
    document_title: str
    url: Optional[str] = None
    similarity: Optional[float] = None


class SearchResponse(BaseModel):
    results: List[SearchResult]
    total_results: int


# Create FastAPI app
app = FastAPI(
    title="FDE API for ElevenLabs",
    description="Faith Driven Investor content API",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup():
    """Initialize database connection pool."""
    global db_pool

    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        logger.error("DATABASE_URL not set")
        return

    try:
        db_pool = await asyncpg.create_pool(
            database_url,
            min_size=2,
            max_size=10,
            command_timeout=60
        )
        logger.info("Database pool created successfully")
    except Exception as e:
        logger.error(f"Failed to create database pool: {e}")


@app.on_event("shutdown")
async def shutdown():
    """Close database connection pool."""
    global db_pool

    if db_pool:
        await db_pool.close()
        logger.info("Database pool closed")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        if db_pool:
            async with db_pool.acquire() as conn:
                await conn.fetchval("SELECT 1")
            return {
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "database": True
            }
        else:
            return {
                "status": "degraded",
                "timestamp": datetime.now().isoformat(),
                "database": False
            }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "timestamp": datetime.now().isoformat(),
            "database": False,
            "error": str(e)
        }


@app.post("/search/vector", response_model=SearchResponse)
async def vector_search(request: SearchRequest):
    """
    Vector search endpoint for FDE content.
    This is what ElevenLabs will call to get relevant context.
    """
    if not db_pool:
        raise HTTPException(status_code=503, detail="Database not available")

    try:
        # Get OpenAI API key for embeddings
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise HTTPException(status_code=500, detail="OpenAI API key not configured")

        # Generate embedding for query using OpenAI
        from openai import OpenAI
        client = OpenAI(api_key=openai_api_key)

        embedding_response = client.embeddings.create(
            model="text-embedding-3-small",
            input=request.query
        )
        query_embedding = embedding_response.data[0].embedding

        # Convert embedding list to PostgreSQL vector format string
        embedding_str = '[' + ','.join(map(str, query_embedding)) + ']'

        # Search database using vector similarity
        async with db_pool.acquire() as conn:
            results = await conn.fetch("""
                SELECT
                    c.content,
                    d.title as document_title,
                    d.metadata->>'url' as document_url,
                    1 - (c.embedding <=> $1::vector) as similarity
                FROM chunks c
                JOIN documents d ON c.document_id = d.id
                WHERE c.embedding IS NOT NULL
                ORDER BY c.embedding <=> $1::vector
                LIMIT $2
            """, embedding_str, request.limit)

        # Format results
        search_results = [
            SearchResult(
                content=row["content"],
                document_title=row["document_title"] or "Unknown",
                url=row.get("document_url"),
                similarity=float(row["similarity"]) if row["similarity"] else None
            )
            for row in results
        ]

        return SearchResponse(
            results=search_results,
            total_results=len(search_results)
        )

    except Exception as e:
        logger.error(f"Vector search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Chat endpoint - performs search and returns formatted response.
    ElevenLabs can call this for conversational responses.
    """
    try:
        # Perform vector search
        search_request = SearchRequest(query=request.message, limit=3)
        search_response = await vector_search(search_request)

        # Format response
        if not search_response.results:
            response_text = "I couldn't find specific information about that in the Faith Driven Investor content."
        else:
            # Build response from top results
            response_parts = ["Based on Faith Driven Investor content:\n\n"]

            for i, result in enumerate(search_response.results[:3], 1):
                # Truncate content to ~200 chars
                content_preview = result.content[:200]
                if len(result.content) > 200:
                    content_preview += "..."

                # Include video ID if available (extract from youtube.com/watch?v=VIDEO_ID)
                video_info = ""
                if result.url:
                    # Extract video ID from URL
                    if "watch?v=" in result.url:
                        video_id = result.url.split("watch?v=")[-1].split("&")[0]
                        video_info = f"\n[Video ID: {video_id}]"

                response_parts.append(
                    f"{i}. From '{result.document_title}':{video_info}\n{content_preview}\n"
                )

            response_text = "\n".join(response_parts)

        return ChatResponse(
            message=response_text,
            session_id=request.session_id,
            timestamp=datetime.now()
        )

    except Exception as e:
        logger.error(f"Chat failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def main():
    """Run the API server."""
    port = int(os.getenv("PORT", 8058))
    host = os.getenv("HOST", "0.0.0.0")

    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=False,
        log_level="info"
    )


if __name__ == "__main__":
    main()
