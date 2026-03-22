"""
memory_client.py — Multi-Tier Mem0 Memory Layer for CI/CD Analyzer
==================================================================

Setup the Mem0 async single client configured with local Qdrant + Gemini embeddings,
as well as neo4j for Entity memory (Tier 4).
"""

import os
from mem0 import AsyncMemory
from google import genai

mem0_config = {
    "llm": {
        "provider": "gemini",
        "config": {
            "model":       "gemini-2.0-flash",
            "api_key":     os.environ.get("GEMINI_API_KEY", ""),
            "temperature": 0.1,
        }
    },
    "embedder": {
        "provider": "gemini",
        "config": {
            "model":   "models/text-embedding-004",
            "api_key": os.environ.get("GEMINI_API_KEY", ""),
        }
    },
    "vector_store": {
        "provider": "qdrant",
        "config": {
            "url":        os.environ.get("QDRANT_ENDPOINT", os.environ.get("QDRANT_URL", f"http://{os.environ.get('QDRANT_HOST', 'localhost')}:{os.environ.get('QDRANT_PORT', 6333)}")),
            "api_key":    os.environ.get("QDRANT_API_KEY", ""),
            "collection_name": "cicd_pipeline_memory",
            "embedding_model_dims": 768,   # text-embedding-004 output dim
        }
    },
    "graph_store": {            # Entity memory (Tier 4)
        "provider": "neo4j",
        "config": {
            "url":      os.environ.get("NEO4J_URL", "bolt://localhost:7687"),
            "username": os.environ.get("NEO4J_USER", "neo4j"),
            "password": os.environ.get("NEO4J_PASSWORD", "neo4j"),
        }
    },
    "history_db_path": "./data/mem0_history.db",  # SQLite for run history
    "version": "v1.1"
}

_memory_instance = None

async def get_memory() -> AsyncMemory:
    global _memory_instance
    if _memory_instance is None:
        _memory_instance = await AsyncMemory.from_config(mem0_config)
    return _memory_instance
