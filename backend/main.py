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
backend_dir = Path(__file__).parent
parent_dir = backend_dir.parent
sys.path.insert(0, str(parent_dir))

# Try to import QueryGPT - if it fails, app can still start
try:
    from fireworks_querygpt import FireworksQueryGPT, load_env_file
    QUERYGPT_AVAILABLE = True
except Exception as e:
    print(f"⚠️  Failed to import FireworksQueryGPT: {e}")
    import traceback
    traceback.print_exc()
    QUERYGPT_AVAILABLE = False
    # Create a dummy class so the app can still start
    class FireworksQueryGPT:
        def __init__(self, *args, **kwargs):
            pass
        def generate_sql(self, *args, **kwargs):
            return {"error": "QueryGPT not initialized", "sql": ""}
    
    def load_env_file(*args, **kwargs):
        pass

# Import OpenAIQueryGPT - it's in evaluate_agents.py
try:
    from evaluate_agents import OpenAIQueryGPT
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    # Fallback: define a simple adapter if import fails
    class OpenAIQueryGPT(FireworksQueryGPT):
        """OpenAI adapter for QueryGPT."""
        def __init__(self, *args, **kwargs):
            try:
                from openai import OpenAI
                import os
                self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
                self.model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
                self.intent_model = os.getenv("OPENAI_INTENT_MODEL", "gpt-4o-mini")
                self.metrics = []
                self.workspaces = {}
                self.intent_mapping = {}
            except:
                pass

# Load environment variables from parent directory
# Backend runs from backend/ directory, so .env is in parent
parent_dir = Path(__file__).parent.parent
env_path = parent_dir / ".env"
load_env_file(str(env_path))

app = FastAPI(title="QueryGPT Evaluation API")

# Enable CORS for frontend
# Allow all origins in production (you can restrict this to specific domains)
cors_origins_str = os.getenv("CORS_ORIGINS", "http://localhost:5173,http://localhost:3000")
cors_origins = [origin.strip() for origin in cors_origins_str.split(",") if origin.strip()]
# In production, allow all origins (you can restrict this)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for Render deployment
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
    
    print("\n" + "="*60)
    print("QueryGPT API Starting Up...")
    print("="*60)
    
    # Print registered routes
    print("\nRegistered Routes:")
    for route in app.routes:
        if hasattr(route, 'path') and hasattr(route, 'methods'):
            print(f"  {list(route.methods)} {route.path}")
    
    # Change to parent directory so workspace files can be found
    import os
    original_cwd = os.getcwd()
    parent_dir = Path(__file__).parent.parent
    print(f"\nCurrent directory: {original_cwd}")
    print(f"Parent directory: {parent_dir}")
    print(f"Backend file: {Path(__file__)}")
    
    os.chdir(parent_dir)
    print(f"Changed to: {os.getcwd()}")
    
    # Create database if it doesn't exist
    db_path = parent_dir / "Chinook.db"
    if not db_path.exists():
        print(f"\n⚠️  Database not found at {db_path}")
        print("Creating database...")
        try:
            # Create empty database file first
            import sqlite3
            conn = sqlite3.connect(str(db_path))
            conn.close()
            print("✓ Empty database file created")
            
            # Now populate it with tables and data
            from create_dummy_db import create_dummy_database
            create_dummy_database(str(db_path))
            print("✓ Database created and populated successfully")
        except Exception as e:
            print(f"⚠️  Failed to create database: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"✓ Database found at {db_path}")
        # Verify database has tables
        try:
            import sqlite3
            conn = sqlite3.connect(str(db_path))
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = cursor.fetchall()
            conn.close()
            print(f"✓ Database has {len(tables)} tables")
        except Exception as e:
            print(f"⚠️  Could not verify database: {e}")
    
    try:
        fireworks_querygpt = FireworksQueryGPT()
        print("✓ Fireworks QueryGPT initialized")
    except Exception as e:
        print(f"⚠️  Failed to initialize Fireworks QueryGPT: {e}")
        import traceback
        traceback.print_exc()
        # Don't fail startup - allow API to work without QueryGPT if needed
    
    try:
        openai_querygpt = OpenAIQueryGPT()
        print("✓ OpenAI QueryGPT initialized")
    except Exception as e:
        print(f"⚠️  Failed to initialize OpenAI QueryGPT: {e}")
        import traceback
        traceback.print_exc()
        # Don't fail startup - allow API to work without QueryGPT if needed
    
    # Restore original directory
    os.chdir(original_cwd)
    print("\n✓ Startup complete")
    print("="*60 + "\n")


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "QueryGPT Evaluation API", 
        "status": "running",
        "routes": [
            "/api/health",
            "/api/chat",
            "/api/evaluation-results",
            "/api/golden-dataset",
        ]
    }


@app.get("/api/health")
async def health():
    """Health check endpoint."""
    results_dir = Path(__file__).parent.parent
    results_files = {
        "fireworks": "evaluation_results_fireworks.json",
        "openai": "evaluation_results_openai.json",
        "fireworks_finetuned": "evaluation_results_post_tuning.json",
    }
    
    available_files = {}
    for provider, filename in results_files.items():
        filepath = results_dir / filename
        available_files[provider] = {
            "exists": filepath.exists(),
            "path": str(filepath),
        }
    
    dataset_path = results_dir / "evaluation_data.json"
    
    return {
        "status": "healthy",
        "fireworks_available": fireworks_querygpt is not None,
        "openai_available": openai_querygpt is not None,
        "results_dir": str(results_dir),
        "evaluation_files": available_files,
        "dataset_file": {
            "exists": dataset_path.exists(),
            "path": str(dataset_path),
        },
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
        sql_error = None
        try:
            from utils import load_db, query_db
            # Load database (database is in parent directory)
            db_path = parent_dir / "Chinook.db"
            print(f"\n[SQL Execution] Database path: {db_path}")
            print(f"[SQL Execution] Database exists: {db_path.exists()}")
            print(f"[SQL Execution] SQL query: {result.get('sql', 'N/A')}")
            
            if db_path.exists():
                conn = load_db(str(db_path))
                try:
                    # Execute query
                    df_result = query_db(conn, result["sql"], return_as_df=True)
                    # Convert to list of dicts for JSON serialization
                    if df_result is not None and not df_result.empty:
                        sql_result = df_result.to_dict("records")
                        print(f"[SQL Execution] ✓ Query executed successfully, returned {len(sql_result)} rows")
                    else:
                        print(f"[SQL Execution] ⚠️  Query executed but returned no results")
                        sql_result = []
                finally:
                    conn.close()
            else:
                error_msg = f"Database not found at {db_path}"
                print(f"[SQL Execution] ❌ {error_msg}")
                sql_error = error_msg
        except Exception as e:
            error_msg = f"Could not execute SQL: {str(e)}"
            print(f"[SQL Execution] ❌ {error_msg}")
            import traceback
            traceback.print_exc()
            sql_error = error_msg
            sql_result = None
        
        return ChatResponse(
            sql=result["sql"],
            explanation=result.get("explanation", ""),
            metrics=metrics,
            result=sql_result,
            error=sql_error,  # Include SQL execution error if any
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
                    
                    # Extract test cases from sql_generation_agent results
                    test_cases = []
                    if "detailed_results" in data and "sql_generation_agent" in data["detailed_results"]:
                        sql_results = data["detailed_results"]["sql_generation_agent"]
                        # Also get intent and table results for agent_results
                        intent_results = data.get("detailed_results", {}).get("intent_agent", [])
                        table_results = data.get("detailed_results", {}).get("table_agent", [])
                        
                        for idx, sql_result in enumerate(sql_results):
                            test_case = {
                                "question": sql_result.get("question", ""),
                                "generated_sql": sql_result.get("generated_sql", ""),
                                "expected_sql": sql_result.get("expected_sql", ""),
                                "sql_similarity": sql_result.get("sql_similarity", 0),
                                "sql_exact_match": sql_result.get("sql_exact_match", False),
                                "result_match": sql_result.get("result_match", False),
                                "correct": sql_result.get("correct", False),
                                "error": sql_result.get("error"),
                            }
                            
                            # Add agent results if available
                            if idx < len(intent_results) and idx < len(table_results):
                                test_case["agent_results"] = {
                                    "intent": intent_results[idx].get("predicted_workspace"),
                                    "tables": table_results[idx].get("predicted_tables", []),
                                }
                            
                            test_cases.append(test_case)
                    
                    results[provider] = {
                        "summary": data.get("summary", {}),
                        "agent_metrics": data.get("summary", {}).get("latency_metrics", {}),
                        "test_cases": test_cases,
                    }
            except Exception as e:
                print(f"Error loading {filename}: {e}")
                import traceback
                traceback.print_exc()
    
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


# Debug endpoint to list all routes
@app.get("/api/routes")
async def list_routes():
    """List all registered routes for debugging."""
    routes = []
    for route in app.routes:
        if hasattr(route, 'path') and hasattr(route, 'methods'):
            routes.append({
                "path": route.path,
                "methods": list(route.methods),
            })
    return {"routes": routes, "total": len(routes)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

