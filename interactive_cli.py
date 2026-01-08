#!/usr/bin/env python3
"""
Interactive CLI for testing Natural Language to SQL queries.
Run queries interactively and see results.
"""

import os
import sys
from pathlib import Path
from fireworks_querygpt import FireworksQueryGPT
from utils import load_db, query_db


def load_env_file(env_path: str = ".env") -> None:
    """Load environment variables from .env file."""
    env_file = Path(env_path)
    if env_file.exists():
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    # Remove quotes if present
                    value = value.strip().strip('"').strip("'")
                    os.environ[key.strip()] = value


# Load .env file if it exists
load_env_file()


def print_header():
    """Print welcome header."""
    print("\n" + "=" * 80)
    print("QueryGPT - Natural Language to SQL")
    print("=" * 80)
    print("Ask questions in natural language and get SQL queries!")
    print("Type 'help' for commands, 'exit' to quit")
    print("=" * 80 + "\n")


def print_help():
    """Print help information."""
    print("\n" + "-" * 80)
    print("COMMANDS:")
    print("-" * 80)
    print("  help              - Show this help message")
    print("  exit / quit       - Exit the interactive session")
    print("  examples          - Show example questions")
    print("  metrics           - Show agent performance metrics")
    print("  clear             - Clear the screen")
    print("\nEXAMPLES:")
    print("-" * 80)
    print("  How many trips were completed yesterday?")
    print("  What are the top 5 drivers by rating?")
    print("  Which city has the most trips?")
    print("  Show all trips in Seattle")
    print("  Count trips by driver")
    print("-" * 80 + "\n")


def show_examples():
    """Show example questions."""
    examples = [
        "How many trips were completed yesterday?",
        "What are the top 5 drivers by rating?",
        "Which city has the most trips?",
        "Show all trips in Seattle",
        "What are the top 5 customers by total spending?",
        "Which vehicle type generates the most revenue?",
        "How many trips used promotions?",
        "Show drivers with their vehicle information",
    ]
    
    print("\n" + "-" * 80)
    print("EXAMPLE QUESTIONS:")
    print("-" * 80)
    for i, example in enumerate(examples, 1):
        print(f"  {i}. {example}")
    print("-" * 80 + "\n")


def execute_sql(db_conn, sql: str):
    """Execute SQL and return results."""
    try:
        results = query_db(db_conn, sql, return_as_df=False)
        return True, results
    except Exception as e:
        return False, str(e)


def interactive_session():
    """Run interactive session."""
    # Check API key
    if not os.getenv("FIREWORKS_API_KEY"):
        print("Error: FIREWORKS_API_KEY not found in .env file")
        print("Create a .env file with: FIREWORKS_API_KEY=your-key-here")
        return
    
    # Initialize QueryGPT
    print("Initializing QueryGPT...")
    try:
        querygpt = FireworksQueryGPT()
        print("✓ QueryGPT initialized successfully!")
    except Exception as e:
        print(f"✗ Error initializing QueryGPT: {e}")
        return
    
    # Load database
    print("Loading database...")
    try:
        db_conn = load_db()
        print("✓ Database loaded successfully!")
    except Exception as e:
        print(f"✗ Error loading database: {e}")
        return
    
    print_header()
    
    # Interactive loop
    while True:
        try:
            # Get user input
            question = input("QueryGPT> ").strip()
            
            if not question:
                continue
            
            # Handle commands
            if question.lower() in ['exit', 'quit']:
                print("\nGoodbye! 👋\n")
                break
            
            if question.lower() == 'help':
                print_help()
                continue
            
            if question.lower() == 'examples':
                show_examples()
                continue
            
            if question.lower() == 'metrics':
                metrics = querygpt.get_metrics_summary()
                print("\n" + "-" * 80)
                print("AGENT METRICS:")
                print("-" * 80)
                import json
                print(json.dumps(metrics, indent=2))
                print("-" * 80 + "\n")
                continue
            
            if question.lower() == 'clear':
                os.system('clear' if os.name != 'nt' else 'cls')
                print_header()
                continue
            
            # Process question
            print("\n" + "-" * 80)
            print(f"Question: {question}")
            print("-" * 80)
            print("Processing...")
            
            try:
                # Generate SQL
                result = querygpt.generate_sql(question)
                
                if "error" in result:
                    print(f"\n✗ Error: {result['error']}\n")
                    continue
                
                # Show results
                print(f"\n✓ Workspace: {result['workspace']}")
                print(f"✓ Tables: {', '.join(result['tables'])}")
                # Handle confidence as string or float
                confidence = result.get('intent_confidence', 0)
                if isinstance(confidence, str):
                    confidence = float(confidence)
                print(f"✓ Intent Confidence: {confidence:.2f}")
                print(f"✓ Latency: {result['total_latency_ms']:.0f}ms")
                
                print(f"\n📝 Generated SQL:")
                print("-" * 80)
                print(result['sql'])
                print("-" * 80)
                
                print(f"\n💡 Explanation:")
                print(result['explanation'])
                
                # Ask if user wants to execute
                execute = input("\nExecute this SQL query? (y/n): ").strip().lower()
                
                if execute == 'y':
                    print("\nExecuting query...")
                    success, query_result = execute_sql(db_conn, result['sql'])
                    
                    if success:
                        print(f"\n✓ Query executed successfully!")
                        print(f"✓ Returned {len(query_result)} rows")
                        
                        if len(query_result) > 0:
                            print(f"\n📊 Results (first 10 rows):")
                            print("-" * 80)
                            
                            # Show column names
                            if query_result:
                                columns = list(query_result[0].keys())
                                print(" | ".join(columns))
                                print("-" * 80)
                                
                                # Show rows
                                for row in query_result[:10]:
                                    values = [str(row[col])[:30] for col in columns]
                                    print(" | ".join(values))
                                
                                if len(query_result) > 10:
                                    print(f"\n... and {len(query_result) - 10} more rows")
                        else:
                            print("(No rows returned)")
                    else:
                        print(f"\n✗ SQL Execution Error: {query_result}")
                
                print("\n" + "=" * 80 + "\n")
                
            except Exception as e:
                print(f"\n✗ Error: {e}\n")
                import traceback
                traceback.print_exc()
        
        except KeyboardInterrupt:
            print("\n\nGoodbye! 👋\n")
            break
        except EOFError:
            print("\n\nGoodbye! 👋\n")
            break
    
    # Cleanup
    db_conn.close()


if __name__ == "__main__":
    interactive_session()

