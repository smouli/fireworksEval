#!/usr/bin/env python3
"""
Create QueryGPT-style workspace system with SQL samples, schemas, and metadata.
This mimics Uber's QueryGPT architecture with workspaces, intent mapping, and RAG samples.
"""

import json
import sqlite3
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime


def load_db(db_path: str = "Chinook.db") -> sqlite3.Connection:
    """Load the SQLite database and return a connection."""
    db_file = Path(db_path)
    if not db_file.exists():
        raise FileNotFoundError(f"Database file not found: {db_path}")

    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.row_factory = sqlite3.Row
    return conn


def get_table_schema(conn: sqlite3.Connection, table_name: str) -> Dict[str, Any]:
    """Get schema information for a table."""
    cursor = conn.cursor()
    cursor.execute(f"PRAGMA table_info({table_name})")
    columns = []
    for row in cursor.fetchall():
        columns.append({
            "name": row[1],
            "type": row[2],
            "not_null": bool(row[3]),
            "default_value": row[4],
            "primary_key": bool(row[5])
        })
    
    # Get foreign keys
    cursor.execute(f"PRAGMA foreign_key_list({table_name})")
    foreign_keys = []
    for row in cursor.fetchall():
        foreign_keys.append({
            "column": row[3],
            "references_table": row[2],
            "references_column": row[4]
        })
    
    return {
        "table_name": table_name,
        "columns": columns,
        "foreign_keys": foreign_keys
    }


def create_workspaces(conn: sqlite3.Connection) -> Dict[str, Any]:
    """Create workspace definitions with SQL samples and table mappings."""
    
    # Get all table schemas
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")
    all_tables = [row[0] for row in cursor.fetchall()]
    
    # Define workspaces (similar to Uber's system workspaces)
    workspaces = {
        "mobility": {
            "name": "Mobility",
            "description": "Queries that include trips, driver, vehicle, and city details",
            "tables": ["trips", "drivers", "vehicles", "cities", "customers"],
            "keywords": ["trip", "driver", "vehicle", "ride", "pickup", "dropoff", "city", "miles", "duration", "fare", "rating"],
            "sql_samples": [
                {
                    "question": "How many trips were completed yesterday?",
                    "sql": "SELECT COUNT(*) as trip_count FROM trips WHERE trip_status = 'completed' AND date(pickup_time) = date('now', '-1 day')",
                    "description": "Count completed trips from yesterday"
                },
                {
                    "question": "What are the top 5 cities by number of trips?",
                    "sql": "SELECT c.city_name, COUNT(t.trip_id) as trip_count FROM cities c JOIN trips t ON c.city_id = t.pickup_city_id GROUP BY c.city_id, c.city_name ORDER BY trip_count DESC LIMIT 5",
                    "description": "Aggregate trips by city with join"
                },
                {
                    "question": "Show all active drivers with their ratings",
                    "sql": "SELECT first_name || ' ' || last_name as driver_name, rating, total_trips FROM drivers WHERE status = 'active' ORDER BY rating DESC",
                    "description": "Filter active drivers and sort by rating"
                },
                {
                    "question": "What is the total revenue from trips in Seattle?",
                    "sql": "SELECT SUM(t.total_amount) as total_revenue FROM trips t JOIN cities c ON t.pickup_city_id = c.city_id WHERE c.city_name = 'Seattle' AND t.trip_status = 'completed'",
                    "description": "Sum revenue with city filter and join"
                },
                {
                    "question": "Which driver has completed the most trips?",
                    "sql": "SELECT d.first_name || ' ' || d.last_name as driver_name, COUNT(t.trip_id) as trip_count FROM drivers d JOIN trips t ON d.driver_id = t.driver_id WHERE t.trip_status = 'completed' GROUP BY d.driver_id, d.first_name, d.last_name ORDER BY trip_count DESC LIMIT 1",
                    "description": "Find driver with most completed trips using aggregation"
                },
                {
                    "question": "What are the completed trips for Tesla vehicles?",
                    "sql": "SELECT t.trip_id, t.total_amount, t.pickup_time FROM trips t JOIN vehicles v ON t.vehicle_id = v.vehicle_id WHERE v.make = 'Tesla' AND t.trip_status = 'completed'",
                    "description": "Filter trips by vehicle make with join"
                },
                {
                    "question": "How many trips does each driver have?",
                    "sql": "SELECT d.first_name || ' ' || d.last_name as driver_name, COUNT(t.trip_id) as trip_count FROM drivers d LEFT JOIN trips t ON d.driver_id = t.driver_id GROUP BY d.driver_id, d.first_name, d.last_name ORDER BY trip_count DESC",
                    "description": "Count trips per driver with left join"
                },
                {
                    "question": "What is the average trip amount by city?",
                    "sql": "SELECT c.city_name, AVG(t.total_amount) as avg_amount FROM cities c JOIN trips t ON c.city_id = t.pickup_city_id WHERE t.trip_status = 'completed' GROUP BY c.city_id, c.city_name ORDER BY avg_amount DESC",
                    "description": "Calculate average with group by"
                }
            ]
        },
        "customer_analytics": {
            "name": "Customer Analytics",
            "description": "Queries about customers, their behavior, and lifetime value",
            "tables": ["customers", "trips", "cities"],
            "keywords": ["customer", "spending", "lifetime", "value", "signup", "revenue", "total"],
            "sql_samples": [
                {
                    "question": "What are the top 5 customers by total spending?",
                    "sql": "SELECT c.first_name || ' ' || c.last_name as customer_name, SUM(t.total_amount) as total_spent FROM customers c JOIN trips t ON c.customer_id = t.customer_id WHERE t.trip_status = 'completed' GROUP BY c.customer_id, c.first_name, c.last_name ORDER BY total_spent DESC LIMIT 5",
                    "description": "Top customers by spending"
                },
                {
                    "question": "How many customers signed up in each month?",
                    "sql": "SELECT strftime('%Y-%m', signup_date) as signup_month, COUNT(*) as customer_count FROM customers WHERE signup_date IS NOT NULL GROUP BY signup_month ORDER BY signup_month DESC",
                    "description": "Customer signups by month"
                },
                {
                    "question": "What is the average lifetime value of customers?",
                    "sql": "SELECT AVG(lifetime_value) as avg_lifetime_value FROM customers",
                    "description": "Average customer lifetime value"
                },
                {
                    "question": "Which city has the most customers?",
                    "sql": "SELECT c.city_name, COUNT(cust.customer_id) as customer_count FROM cities c JOIN customers cust ON c.city_id = cust.city_id GROUP BY c.city_id, c.city_name ORDER BY customer_count DESC LIMIT 1",
                    "description": "City with most customers"
                }
            ]
        },
        "vehicle_operations": {
            "name": "Vehicle Operations",
            "description": "Queries about vehicles, their usage, and maintenance",
            "tables": ["vehicles", "trips", "drivers"],
            "keywords": ["vehicle", "car", "make", "model", "type", "license", "plate"],
            "sql_samples": [
                {
                    "question": "Which vehicle make and model has the most trips?",
                    "sql": "SELECT v.make, v.model, COUNT(t.trip_id) as trip_count FROM vehicles v JOIN trips t ON v.vehicle_id = t.vehicle_id GROUP BY v.make, v.model ORDER BY trip_count DESC LIMIT 1",
                    "description": "Most used vehicle model"
                },
                {
                    "question": "What is the total revenue generated by each vehicle type?",
                    "sql": "SELECT v.vehicle_type, SUM(t.total_amount) as total_revenue FROM vehicles v JOIN trips t ON v.vehicle_id = t.vehicle_id WHERE t.trip_status = 'completed' GROUP BY v.vehicle_type ORDER BY total_revenue DESC",
                    "description": "Revenue by vehicle type"
                },
                {
                    "question": "List all vehicles with their driver names",
                    "sql": "SELECT v.make || ' ' || v.model as vehicle, d.first_name || ' ' || d.last_name as driver_name, v.license_plate FROM vehicles v JOIN drivers d ON v.driver_id = d.driver_id",
                    "description": "Vehicles with driver information"
                }
            ]
        },
        "promotions": {
            "name": "Promotions",
            "description": "Queries about promotions, discounts, and marketing campaigns",
            "tables": ["promotions", "trip_promotions", "trips"],
            "keywords": ["promotion", "discount", "code", "campaign", "marketing"],
            "sql_samples": [
                {
                    "question": "How many trips used promotions?",
                    "sql": "SELECT COUNT(DISTINCT trip_id) as trips_with_promotions FROM trip_promotions",
                    "description": "Count trips with promotions"
                },
                {
                    "question": "What is the total discount applied from promotions?",
                    "sql": "SELECT SUM(discount_applied) as total_discount FROM trip_promotions",
                    "description": "Total discount amount"
                },
                {
                    "question": "Which promotion code has been used the most?",
                    "sql": "SELECT p.promotion_code, COUNT(tp.trip_id) as usage_count FROM promotions p JOIN trip_promotions tp ON p.promotion_id = tp.promotion_id GROUP BY p.promotion_id, p.promotion_code ORDER BY usage_count DESC LIMIT 1",
                    "description": "Most used promotion code"
                }
            ]
        }
    }
    
    # Add table schemas to each workspace
    for workspace_id, workspace in workspaces.items():
        workspace["table_schemas"] = {}
        for table_name in workspace["tables"]:
            if table_name in all_tables:
                workspace["table_schemas"][table_name] = get_table_schema(conn, table_name)
    
    return workspaces


def create_intent_mapping() -> Dict[str, Any]:
    """Create intent mapping examples for the intent agent."""
    return {
        "intent_examples": [
            {
                "question": "How many trips were completed by Teslas in Seattle yesterday?",
                "intent": "mobility",
                "reasoning": "Question is about trips, vehicles (Tesla), and location (Seattle) - all mobility domain"
            },
            {
                "question": "What are the top customers by spending?",
                "intent": "customer_analytics",
                "reasoning": "Question is about customer spending and analytics"
            },
            {
                "question": "Which vehicle type generates the most revenue?",
                "intent": "vehicle_operations",
                "reasoning": "Question is about vehicles and their revenue performance"
            },
            {
                "question": "How many trips used the WELCOME10 promotion?",
                "intent": "promotions",
                "reasoning": "Question is specifically about promotions"
            },
            {
                "question": "Show me drivers in New York with their ratings",
                "intent": "mobility",
                "reasoning": "Question about drivers and locations - mobility domain"
            }
        ],
        "intent_categories": {
            "mobility": {
                "description": "Trips, drivers, vehicles, cities, and ride-related queries",
                "primary_tables": ["trips", "drivers", "vehicles", "cities"],
                "keywords": ["trip", "driver", "vehicle", "ride", "pickup", "dropoff", "city", "miles", "duration", "fare", "rating", "completed", "cancelled"]
            },
            "customer_analytics": {
                "description": "Customer behavior, spending, and analytics",
                "primary_tables": ["customers", "trips"],
                "keywords": ["customer", "spending", "lifetime", "value", "signup", "revenue"]
            },
            "vehicle_operations": {
                "description": "Vehicle usage, performance, and operations",
                "primary_tables": ["vehicles", "trips", "drivers"],
                "keywords": ["vehicle", "car", "make", "model", "type", "license"]
            },
            "promotions": {
                "description": "Promotions, discounts, and marketing campaigns",
                "primary_tables": ["promotions", "trip_promotions"],
                "keywords": ["promotion", "discount", "code", "campaign"]
            }
        }
    }


def create_table_agent_examples() -> Dict[str, Any]:
    """Create examples for table agent to identify relevant tables."""
    return {
        "table_selection_examples": [
            {
                "question": "How many trips were completed by Teslas in Seattle yesterday?",
                "relevant_tables": ["trips", "vehicles", "cities"],
                "reasoning": "Need trips for completion status, vehicles for Tesla filter, cities for Seattle filter"
            },
            {
                "question": "What are the top customers by total spending?",
                "relevant_tables": ["customers", "trips"],
                "reasoning": "Need customers table and trips to calculate spending"
            },
            {
                "question": "Show drivers with their vehicle information",
                "relevant_tables": ["drivers", "vehicles"],
                "reasoning": "Need both drivers and vehicles tables"
            },
            {
                "question": "Which promotion code has been used the most?",
                "relevant_tables": ["promotions", "trip_promotions"],
                "reasoning": "Need promotions and trip_promotions to count usage"
            }
        ]
    }


def create_column_prune_examples() -> Dict[str, Any]:
    """Create examples for column prune agent."""
    return {
        "column_pruning_examples": [
            {
                "question": "How many trips were completed by Teslas in Seattle yesterday?",
                "table": "trips",
                "relevant_columns": ["trip_id", "trip_status", "pickup_time", "pickup_city_id", "vehicle_id"],
                "irrelevant_columns": ["customer_id", "dropoff_time", "payment_method", "rating", "tip_amount"],
                "reasoning": "Only need status, time, city, and vehicle for this query"
            },
            {
                "question": "What are the top customers by total spending?",
                "table": "trips",
                "relevant_columns": ["customer_id", "total_amount", "trip_status"],
                "irrelevant_columns": ["driver_id", "pickup_time", "dropoff_time", "distance_miles", "duration_minutes", "payment_method"],
                "reasoning": "Only need customer, amount, and status for spending calculation"
            },
            {
                "question": "Show drivers with their vehicle information",
                "table": "drivers",
                "relevant_columns": ["driver_id", "first_name", "last_name"],
                "irrelevant_columns": ["email", "phone", "total_trips", "joined_date"],
                "reasoning": "Only need driver ID and name for join"
            }
        ],
        "pruning_strategy": {
            "description": "Identify columns needed based on: SELECT clauses, WHERE filters, JOIN keys, GROUP BY, ORDER BY",
            "keep_always": ["primary_keys", "foreign_keys", "columns_in_where_clause", "columns_in_select"],
            "can_prune": ["columns_not_referenced", "unused_foreign_keys", "metadata_columns"]
        }
    }


def main():
    """Generate all QueryGPT-style workspace files."""
    print("Creating QueryGPT-style workspace system...")
    print("=" * 80)
    
    # Load database
    conn = load_db()
    print("✓ Database loaded")
    
    # Create workspaces
    print("\nCreating workspaces...")
    workspaces = create_workspaces(conn)
    print(f"✓ Created {len(workspaces)} workspaces")
    
    # Create intent mapping
    print("\nCreating intent mapping...")
    intent_mapping = create_intent_mapping()
    print("✓ Intent mapping created")
    
    # Create table agent examples
    print("\nCreating table agent examples...")
    table_agent = create_table_agent_examples()
    print("✓ Table agent examples created")
    
    # Create column prune examples
    print("\nCreating column prune examples...")
    column_prune = create_column_prune_examples()
    print("✓ Column prune examples created")
    
    # Save all to files
    output_dir = Path("querygpt_workspaces")
    output_dir.mkdir(exist_ok=True)
    
    # Save workspaces
    workspaces_file = output_dir / "workspaces.json"
    with open(workspaces_file, "w") as f:
        json.dump(workspaces, f, indent=2)
    print(f"\n✓ Saved workspaces to {workspaces_file}")
    
    # Save intent mapping
    intent_file = output_dir / "intent_mapping.json"
    with open(intent_file, "w") as f:
        json.dump(intent_mapping, f, indent=2)
    print(f"✓ Saved intent mapping to {intent_file}")
    
    # Save table agent
    table_agent_file = output_dir / "table_agent.json"
    with open(table_agent_file, "w") as f:
        json.dump(table_agent, f, indent=2)
    print(f"✓ Saved table agent examples to {table_agent_file}")
    
    # Save column prune
    column_prune_file = output_dir / "column_prune.json"
    with open(column_prune_file, "w") as f:
        json.dump(column_prune, f, indent=2)
    print(f"✓ Saved column prune examples to {column_prune_file}")
    
    # Create summary document
    summary = {
        "created_at": datetime.now().isoformat(),
        "workspaces": {
            workspace_id: {
                "name": ws["name"],
                "tables": ws["tables"],
                "sql_samples_count": len(ws["sql_samples"])
            }
            for workspace_id, ws in workspaces.items()
        },
        "total_sql_samples": sum(len(ws["sql_samples"]) for ws in workspaces.values()),
        "total_tables": len(set(table for ws in workspaces.values() for table in ws["tables"]))
    }
    
    summary_file = output_dir / "summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"✓ Saved summary to {summary_file}")
    
    conn.close()
    
    print("\n" + "=" * 80)
    print("QueryGPT workspace system created successfully!")
    print(f"\nWorkspace breakdown:")
    for workspace_id, ws in workspaces.items():
        print(f"  - {ws['name']}: {len(ws['sql_samples'])} SQL samples, {len(ws['tables'])} tables")
    print(f"\nTotal: {summary['total_sql_samples']} SQL samples across {len(workspaces)} workspaces")


if __name__ == "__main__":
    main()

