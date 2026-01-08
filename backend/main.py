#!/usr/bin/env python3
"""
FastAPI backend for QueryGPT evaluation frontend.
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import sys

# Add parent directory to path to import QueryGPT
sys.path.insert(0, str(Path(__file__).parent.parent))

from fireworks_querygpt import FireworksQueryGPT, load_env_file

# Import OpenAIQueryGPT - it's in evaluate_agents.py
try:
    from evaluate_agents import OpenAIQueryGPT
except ImportError:
    # Fallback: define a simple adapter if import fails
    class OpenAIQueryGPT(FireworksQueryGPT):
        """OpenAI adapter for QueryGPT."""
        def __init__(self, *args, **kwargs):
            from openai import OpenAI
            import os
            self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            self.model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
            self.intent_model = os.getenv("OPENAI_INTENT_MODEL", "gpt-4o-mini")
            self.metrics = []
            self.workspaces = self._load_workspaces()
            self.intent_mapping = self._load_intent_mapping()

# Load environment variables from parent directory
# Backend runs from backend/ directory, so .env is in parent
parent_dir = Path(__file__).parent.parent
env_path = parent_dir / ".env"
load_env_file(str(env_path))

app = FastAPI(title="QueryGPT Evaluation API")

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],  # Vite default ports
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response models
class ChatRequest(BaseModel):
    question: str
    provider: str = "fireworks"  # "fireworks" or "openai"


class ChatResponse(BaseModel):
    sql: str
    explanation: str
    metrics: Dict[str, Any]
    result: Optional[List[Dict[str, Any]]] = None
    error: Optional[str] = None


class EvaluationResults(BaseModel):
    provider: str
    summary: Dict[str, Any]
    agent_metrics: Dict[str, Any]
    test_cases: List[Dict[str, Any]]


# Initialize QueryGPT instances
fireworks_querygpt = None
openai_querygpt = None


@app.on_event("startup")
async def startup_event():
    """Initialize QueryGPT instances on startup."""
    global fireworks_querygpt, openai_querygpt
    
    # Change to parent directory so workspace files can be found
    import os
    original_cwd = os.getcwd()
    parent_dir = Path(__file__).parent.parent
    os.chdir(parent_dir)
    
    try:
        fireworks_querygpt = FireworksQueryGPT()
        print("✓ Fireworks QueryGPT initialized")
    except Exception as e:
        print(f"⚠️  Failed to initialize Fireworks QueryGPT: {e}")
        import traceback
        traceback.print_exc()
    
    try:
        openai_querygpt = OpenAIQueryGPT()
        print("✓ OpenAI QueryGPT initialized")
    except Exception as e:
        print(f"⚠️  Failed to initialize OpenAI QueryGPT: {e}")
        import traceback
        traceback.print_exc()
    
    # Restore original directory
    os.chdir(original_cwd)


@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "QueryGPT Evaluation API", "status": "running"}


@app.get("/api/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "fireworks_available": fireworks_querygpt is not None,
        "openai_available": openai_querygpt is not None,
    }


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Process a natural language query and return SQL."""
    try:
        # Select provider
        querygpt = fireworks_querygpt if request.provider == "fireworks" else openai_querygpt
        
        if querygpt is None:
            raise HTTPException(
                status_code=503,
                detail=f"{request.provider} provider not available. Check backend logs for initialization errors."
            )
        
        # Change to parent directory for workspace file access
        import os
        original_cwd = os.getcwd()
        parent_dir = Path(__file__).parent.parent
        os.chdir(parent_dir)
        
        try:
            # Process query (use generate_sql method)
            result = querygpt.generate_sql(request.question)
        finally:
            # Restore original directory
            os.chdir(original_cwd)
        
        # Get metrics from result
        metrics = {}
        if "metrics" in result and result["metrics"]:
            result_metrics = result["metrics"]
            metrics = {
                "intent_agent": {
                    "latency_ms": result_metrics.get("intent").latency_ms if result_metrics.get("intent") else 0,
                    "tokens": result_metrics.get("intent").tokens_used if result_metrics.get("intent") else 0,
                    "used_tool_calls": result_metrics.get("intent").used_tool_calls if result_metrics.get("intent") else None,
                } if result_metrics.get("intent") else {},
                "table_agent": {
                    "latency_ms": result_metrics.get("table").latency_ms if result_metrics.get("table") else 0,
                    "tokens": result_metrics.get("table").tokens_used if result_metrics.get("table") else 0,
                    "used_tool_calls": result_metrics.get("table").used_tool_calls if result_metrics.get("table") else None,
                } if result_metrics.get("table") else {},
                "column_prune_agent": {
                    "latency_ms": result_metrics.get("prune").latency_ms if result_metrics.get("prune") else 0,
                    "tokens": result_metrics.get("prune").tokens_used if result_metrics.get("prune") else 0,
                    "used_tool_calls": result_metrics.get("prune").used_tool_calls if result_metrics.get("prune") else None,
                } if result_metrics.get("prune") else {},
                "sql_generation_agent": {
                    "latency_ms": result_metrics.get("sql").latency_ms if result_metrics.get("sql") else 0,
                    "tokens": result_metrics.get("sql").tokens_used if result_metrics.get("sql") else 0,
                    "used_tool_calls": result_metrics.get("sql").used_tool_calls if result_metrics.get("sql") else None,
                } if result_metrics.get("sql") else {},
                "total_latency_ms": result.get("total_latency_ms", 0),
            }
        
        # Execute SQL and return results
        sql_result = None
        try:
            from utils import load_db, query_db
            # Load database (database is in parent directory)
            db_path = parent_dir / "Chinook.db"
            if db_path.exists():
                conn = load_db(str(db_path))
                try:
                    # Execute query
                    df_result = query_db(conn, result["sql"], return_as_df=True)
                    # Convert to list of dicts for JSON serialization
                    if df_result is not None and not df_result.empty:
                        sql_result = df_result.to_dict("records")
                finally:
                    conn.close()
            else:
                print(f"Warning: Database not found at {db_path}")
        except Exception as e:
            print(f"Warning: Could not execute SQL: {e}")
            import traceback
            traceback.print_exc()
            # Include error in response so user knows
            sql_result = None
        
        return ChatResponse(
            sql=result["sql"],
            explanation=result.get("explanation", ""),
            metrics=metrics,
            result=sql_result,
        )
    except Exception as e:
        return ChatResponse(
            sql="",
            explanation="",
            metrics={},
            error=str(e),
        )


@app.get("/api/evaluation-results")
async def get_evaluation_results():
    """Get all available evaluation results."""
    results_dir = Path(__file__).parent.parent
    results_files = {
        "fireworks": "evaluation_results_fireworks.json",
        "openai": "evaluation_results_openai.json",
        "fireworks_finetuned": "evaluation_results_post_tuning.json",
    }
    
    results = {}
    for provider, filename in results_files.items():
        filepath = results_dir / filename
        if filepath.exists():
            try:
                with open(filepath) as f:
                    data = json.load(f)
                    results[provider] = {
                        "summary": data.get("summary", {}),
                        "agent_metrics": data.get("agent_metrics", {}),
                        "test_cases": data.get("test_cases", []),
                    }
            except Exception as e:
                print(f"Error loading {filename}: {e}")
    
    return results


@app.get("/api/evaluation-results/{provider}")
async def get_provider_results(provider: str):
    """Get evaluation results for a specific provider."""
    results_dir = Path(__file__).parent.parent
    results_files = {
        "fireworks": "evaluation_results_fireworks.json",
        "openai": "evaluation_results_openai.json",
        "fireworks_finetuned": "evaluation_results_post_tuning.json",
    }
    
    if provider not in results_files:
        raise HTTPException(status_code=404, detail=f"Provider {provider} not found")
    
    filepath = results_dir / results_files[provider]
    if not filepath.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Results file for {provider} not found"
        )
    
    try:
        with open(filepath) as f:
            return json.load(f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading results: {str(e)}")


@app.get("/api/golden-dataset")
async def get_golden_dataset():
    """Get the golden dataset."""
    dataset_path = Path(__file__).parent.parent / "evaluation_data.json"
    
    if not dataset_path.exists():
        raise HTTPException(status_code=404, detail="Golden dataset not found")
    
    try:
        with open(dataset_path) as f:
            return json.load(f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading dataset: {str(e)}")


@app.get("/api/golden-dataset/stats")
async def get_golden_dataset_stats():
    """Get statistics about the golden dataset."""
    dataset_path = Path(__file__).parent.parent / "evaluation_data.json"
    
    if not dataset_path.exists():
        raise HTTPException(status_code=404, detail="Golden dataset not found")
    
    try:
        with open(dataset_path) as f:
            data = json.load(f)
        
        # Calculate statistics
        categories = {}
        for item in data:
            category = item.get("category", "unknown")
            categories[category] = categories.get(category, 0) + 1
        
        return {
            "total_cases": len(data),
            "categories": categories,
            "categories_count": len(categories),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading dataset: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

