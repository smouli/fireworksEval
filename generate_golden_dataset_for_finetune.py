#!/usr/bin/env python3
"""
Generate golden dataset for fine-tuning (SFT).
Creates a separate dataset from evaluation_data.json that can be merged later.
"""

import json
import sqlite3
import re
import random
from pathlib import Path
from typing import Dict, List, Any, Tuple, Set
from datetime import datetime, timedelta
from utils import load_db, query_db


def extract_tables_from_sql(sql: str) -> List[str]:
    """Extract table names from SQL query."""
    tables = set()
    tables.update(re.findall(r'\bFROM\s+(\w+)', sql, re.IGNORECASE))
    tables.update(re.findall(r'\bJOIN\s+(\w+)', sql, re.IGNORECASE))
    return list(tables)


def validate_sql(conn: sqlite3.Connection, sql: str) -> Tuple[bool, List[Dict]]:
    """Validate SQL query and return results."""
    try:
        results = query_db(conn, sql, return_as_df=False)
        return True, results
    except Exception as e:
        return False, []


def generate_filter_variations(base_query: Dict, conn: sqlite3.Connection) -> List[Dict]:
    """Generate variations by changing filter values."""
    variations = []
    sql = base_query["sql"]
    question = base_query["question"]
    
    # Driver status variations
    if "drivers" in sql.lower() and "status" in sql.lower():
        statuses = query_db(conn, "SELECT DISTINCT status FROM drivers WHERE status IS NOT NULL LIMIT 5", return_as_df=False)
        for status_row in statuses:
            status_val = status_row["status"]
            if status_val != "active":  # Avoid duplicate of original
                new_sql = sql.replace("'active'", f"'{status_val}'")
                new_question = question.replace("active", status_val.lower())
                if new_sql != sql:
                    valid, results = validate_sql(conn, new_sql)
                    if valid and len(results) > 0:
                        variations.append({
                            "question": new_question,
                            "sql": new_sql,
                            "category": base_query["category"],
                            "expected_result": results[:100],
                            "source": "filter_variation"
                        })
    
    # Trip status variations
    if "trips" in sql.lower() and "trip_status" in sql.lower():
        statuses = query_db(conn, "SELECT DISTINCT trip_status FROM trips WHERE trip_status IS NOT NULL", return_as_df=False)
        for status_row in statuses:
            status_val = status_row["trip_status"]
            if status_val != "completed":  # Avoid duplicate
                new_sql = sql.replace("'completed'", f"'{status_val}'")
                new_question = question.replace("completed", status_val.lower())
                if new_sql != sql:
                    valid, results = validate_sql(conn, new_sql)
                    if valid:
                        variations.append({
                            "question": new_question,
                            "sql": new_sql,
                            "category": base_query["category"],
                            "expected_result": results[:100],
                            "source": "filter_variation"
                        })
    
    # City/state variations
    if "cities" in sql.lower() and "state" in sql.lower():
        states = query_db(conn, "SELECT DISTINCT state FROM cities WHERE state IS NOT NULL LIMIT 5", return_as_df=False)
        for state_row in states:
            state_val = state_row["state"]
            if state_val != "California":  # Avoid duplicate
                new_sql = sql.replace("'California'", f"'{state_val}'")
                new_question = question.replace("California", state_val)
                if new_sql != sql:
                    valid, results = validate_sql(conn, new_sql)
                    if valid:
                        variations.append({
                            "question": new_question,
                            "sql": new_sql,
                            "category": base_query["category"],
                            "expected_result": results[:100],
                            "source": "filter_variation"
                        })
    
    # City name variations
    if "city_name" in sql.lower() and "=" in sql:
        cities = query_db(conn, "SELECT DISTINCT city_name FROM cities LIMIT 10", return_as_df=False)
        for city_row in cities:
            city_val = city_row["city_name"]
            # Replace common city names
            for old_city in ["Seattle", "New York", "Los Angeles"]:
                if old_city in question:
                    new_sql = sql.replace(f"'{old_city}'", f"'{city_val}'")
                    new_question = question.replace(old_city, city_val)
                    if new_sql != sql:
                        valid, results = validate_sql(conn, new_sql)
                        if valid:
                            variations.append({
                                "question": new_question,
                                "sql": new_sql,
                                "category": base_query["category"],
                                "expected_result": results[:100],
                                "source": "filter_variation"
                            })
                    break
    
    return variations


def generate_limit_variations(base_query: Dict, conn: sqlite3.Connection) -> List[Dict]:
    """Generate variations by changing LIMIT values."""
    variations = []
    sql = base_query["sql"]
    question = base_query["question"]
    
    if "LIMIT" in sql.upper():
        # Extract current limit
        limit_match = re.search(r'LIMIT\s+(\d+)', sql, re.IGNORECASE)
        if limit_match:
            current_limit = int(limit_match.group(1))
            limits = [3, 5, 10, 20, 50]
            limits = [l for l in limits if l != current_limit]
            
            for limit in limits[:3]:  # Limit to 3 variations
                new_sql = re.sub(r'LIMIT\s+\d+', f'LIMIT {limit}', sql, flags=re.IGNORECASE)
                # Update question
                new_question = re.sub(r'top\s+\d+|first\s+\d+', f'top {limit}', question, flags=re.IGNORECASE)
                if new_sql != sql:
                    valid, results = validate_sql(conn, new_sql)
                    if valid:
                        variations.append({
                            "question": new_question,
                            "sql": new_sql,
                            "category": base_query["category"],
                            "expected_result": results[:100],
                            "source": "limit_variation"
                        })
    
    return variations


def generate_date_variations(base_query: Dict, conn: sqlite3.Connection) -> List[Dict]:
    """Generate variations with different date ranges."""
    variations = []
    sql = base_query["sql"]
    question = base_query["question"]
    
    date_replacements = [
        ("yesterday", "-1 day", "yesterday", "last 7 days", "-7 days", "last 7 days"),
        ("yesterday", "-1 day", "yesterday", "last 30 days", "-30 days", "last 30 days"),
        ("last 7 days", "-7 days", "last 7 days", "last 30 days", "-30 days", "last 30 days"),
        ("last 7 days", "-7 days", "last 7 days", "yesterday", "-1 day", "yesterday"),
    ]
    
    for old_text, old_sql, old_q, new_text, new_sql, new_q in date_replacements:
        if old_text in question.lower() or old_sql in sql.lower():
            new_sql_query = sql.replace(old_sql, new_sql)
            new_question = question.replace(old_text, new_text)
            if new_sql_query != sql:
                valid, results = validate_sql(conn, new_sql_query)
                if valid:
                    variations.append({
                        "question": new_question,
                        "sql": new_sql_query,
                        "category": base_query["category"],
                        "expected_result": results[:100],
                        "source": "date_variation"
                    })
    
    return variations


def generate_aggregation_variations(base_query: Dict, conn: sqlite3.Connection) -> List[Dict]:
    """Generate variations by changing aggregation functions."""
    variations = []
    sql = base_query["sql"]
    question = base_query["question"]
    
    # COUNT variations - try different aggregations on same column
    if "COUNT" in sql.upper() and "total_amount" in sql.lower():
        # Try SUM, AVG, MAX, MIN on total_amount
        agg_functions = [
            ("SUM", "total", "sum"),
            ("AVG", "average", "average"),
            ("MAX", "maximum", "maximum"),
            ("MIN", "minimum", "minimum"),
        ]
        
        for agg_func, metric_word, question_word in agg_functions:
            new_sql = sql.replace("COUNT", agg_func).replace("COUNT(*)", f"{agg_func}(total_amount)")
            new_question = question.replace("how many", f"what is the {question_word}").replace("count", metric_word)
            if new_sql != sql:
                valid, results = validate_sql(conn, new_sql)
                if valid:
                    variations.append({
                        "question": new_question,
                        "sql": new_sql,
                        "category": base_query["category"],
                        "expected_result": results[:100],
                        "source": "aggregation_variation"
                    })
    
    return variations


def generate_failure_pattern_queries(conn: sqlite3.Connection) -> List[Dict]:
    """Generate queries targeting common failure patterns from evaluation."""
    queries = []
    
    # Pattern 1: Column selection (biggest failure - 20% F1)
    queries.extend([
        {
            "question": "Show driver names and email addresses",
            "sql": "SELECT first_name, last_name, email FROM drivers",
            "category": "simple_filtering",
            "source": "failure_pattern",
            "focus": "column_selection"
        },
        {
            "question": "List trip IDs with customer names and driver names",
            "sql": "SELECT t.trip_id, c.first_name || ' ' || c.last_name as customer_name, d.first_name || ' ' || d.last_name as driver_name FROM trips t JOIN customers c ON t.customer_id = c.customer_id JOIN drivers d ON t.driver_id = d.driver_id LIMIT 10",
            "category": "complex_joins",
            "source": "failure_pattern",
            "focus": "column_selection"
        },
        {
            "question": "Show trip details including ID, amount, and pickup time",
            "sql": "SELECT trip_id, total_amount, pickup_time FROM trips LIMIT 10",
            "category": "simple_filtering",
            "source": "failure_pattern",
            "focus": "column_selection"
        },
    ])
    
    # Pattern 2: Aggregate functions (column pruning failures)
    queries.extend([
        {
            "question": "How many trips are there in total?",
            "sql": "SELECT COUNT(*) as total_trips FROM trips",
            "category": "simple_aggregation",
            "source": "failure_pattern",
            "focus": "aggregate_functions"
        },
        {
            "question": "What is the total sum of all trip amounts?",
            "sql": "SELECT SUM(total_amount) as total_revenue FROM trips",
            "category": "simple_aggregation",
            "source": "failure_pattern",
            "focus": "aggregate_functions"
        },
        {
            "question": "What is the minimum trip amount?",
            "sql": "SELECT MIN(total_amount) as min_amount FROM trips",
            "category": "simple_aggregation",
            "source": "failure_pattern",
            "focus": "aggregate_functions"
        },
        {
            "question": "What is the maximum trip amount?",
            "sql": "SELECT MAX(total_amount) as max_amount FROM trips",
            "category": "simple_aggregation",
            "source": "failure_pattern",
            "focus": "aggregate_functions"
        },
        {
            "question": "How many drivers are there?",
            "sql": "SELECT COUNT(*) as driver_count FROM drivers",
            "category": "simple_aggregation",
            "source": "failure_pattern",
            "focus": "aggregate_functions"
        },
        {
            "question": "How many customers are there?",
            "sql": "SELECT COUNT(*) as customer_count FROM customers",
            "category": "simple_aggregation",
            "source": "failure_pattern",
            "focus": "aggregate_functions"
        },
    ])
    
    # Pattern 3: Date filtering (SQL generation failures)
    queries.extend([
        {
            "question": "How many trips were completed today?",
            "sql": "SELECT COUNT(*) FROM trips WHERE trip_status = 'completed' AND date(pickup_time) = date('now')",
            "category": "date_filtering",
            "source": "failure_pattern",
            "focus": "date_functions"
        },
        {
            "question": "What trips happened in the last hour?",
            "sql": "SELECT * FROM trips WHERE pickup_time >= datetime('now', '-1 hour') LIMIT 10",
            "category": "date_filtering",
            "source": "failure_pattern",
            "focus": "date_functions"
        },
        {
            "question": "Show trips from last week",
            "sql": "SELECT * FROM trips WHERE pickup_time >= datetime('now', '-7 days') LIMIT 10",
            "category": "date_filtering",
            "source": "failure_pattern",
            "focus": "date_functions"
        },
    ])
    
    # Pattern 4: String concatenation
    queries.extend([
        {
            "question": "Show full customer names",
            "sql": "SELECT first_name || ' ' || last_name as full_name FROM customers LIMIT 10",
            "category": "simple_filtering",
            "source": "failure_pattern",
            "focus": "string_operations"
        },
        {
            "question": "List driver full names with their cities",
            "sql": "SELECT d.first_name || ' ' || d.last_name as driver_name, c.city_name FROM drivers d JOIN cities c ON d.city_id = c.city_id LIMIT 10",
            "category": "filtering_with_join",
            "source": "failure_pattern",
            "focus": "string_operations"
        },
    ])
    
    # Pattern 5: HAVING clauses
    queries.extend([
        {
            "question": "Which cities have more than 50 trips?",
            "sql": "SELECT c.city_name, COUNT(t.trip_id) as trip_count FROM cities c JOIN trips t ON c.city_id = t.pickup_city_id GROUP BY c.city_id, c.city_name HAVING COUNT(t.trip_id) > 50",
            "category": "group_by_having",
            "source": "failure_pattern",
            "focus": "having_clause"
        },
        {
            "question": "Show drivers who completed more than 10 trips",
            "sql": "SELECT d.first_name || ' ' || d.last_name as driver_name, COUNT(t.trip_id) as trip_count FROM drivers d JOIN trips t ON d.driver_id = t.driver_id WHERE t.trip_status = 'completed' GROUP BY d.driver_id, d.first_name, d.last_name HAVING COUNT(t.trip_id) > 10",
            "category": "group_by_having",
            "source": "failure_pattern",
            "focus": "having_clause"
        },
        {
            "question": "What customers have spent more than $1000?",
            "sql": "SELECT c.first_name || ' ' || c.last_name as customer_name, SUM(t.total_amount) as total_spent FROM customers c JOIN trips t ON c.customer_id = t.customer_id WHERE t.trip_status = 'completed' GROUP BY c.customer_id, c.first_name, c.last_name HAVING SUM(t.total_amount) > 1000",
            "category": "group_by_having",
            "source": "failure_pattern",
            "focus": "having_clause"
        },
    ])
    
    # Pattern 6: Vehicle make/model queries (intent classification failures)
    queries.extend([
        {
            "question": "What is the total revenue from Tesla trips?",
            "sql": "SELECT SUM(t.total_amount) as total_revenue FROM trips t JOIN vehicles v ON t.vehicle_id = v.vehicle_id WHERE v.make = 'Tesla' AND t.trip_status = 'completed'",
            "category": "aggregation_with_joins",
            "source": "failure_pattern",
            "focus": "vehicle_queries"
        },
        {
            "question": "How many trips used Toyota vehicles?",
            "sql": "SELECT COUNT(*) as trip_count FROM trips t JOIN vehicles v ON t.vehicle_id = v.vehicle_id WHERE v.make = 'Toyota'",
            "category": "aggregation_with_joins",
            "source": "failure_pattern",
            "focus": "vehicle_queries"
        },
    ])
    
    # Pattern 7: Payment method queries (intent classification failures)
    queries.extend([
        {
            "question": "What is the total revenue by payment method?",
            "sql": "SELECT payment_method, SUM(total_amount) as total_revenue FROM trips WHERE payment_method IS NOT NULL GROUP BY payment_method ORDER BY total_revenue DESC",
            "category": "simple_aggregation",
            "source": "failure_pattern",
            "focus": "payment_method"
        },
        {
            "question": "How many trips used credit card payment?",
            "sql": "SELECT COUNT(*) as trip_count FROM trips WHERE payment_method = 'credit_card'",
            "category": "simple_filtering",
            "source": "failure_pattern",
            "focus": "payment_method"
        },
        {
            "question": "Show payment methods and their usage count",
            "sql": "SELECT payment_method, COUNT(*) as usage_count FROM trips WHERE payment_method IS NOT NULL GROUP BY payment_method ORDER BY usage_count DESC",
            "category": "simple_aggregation",
            "source": "failure_pattern",
            "focus": "payment_method"
        },
    ])
    
    # Pattern 8: More column selection examples
    queries.extend([
        {
            "question": "Show trip ID, customer name, and total amount",
            "sql": "SELECT t.trip_id, c.first_name || ' ' || c.last_name as customer_name, t.total_amount FROM trips t JOIN customers c ON t.customer_id = c.customer_id LIMIT 10",
            "category": "filtering_with_join",
            "source": "failure_pattern",
            "focus": "column_selection"
        },
        {
            "question": "List vehicle make, model, and trip count",
            "sql": "SELECT v.make, v.model, COUNT(t.trip_id) as trip_count FROM vehicles v JOIN trips t ON v.vehicle_id = t.vehicle_id GROUP BY v.make, v.model ORDER BY trip_count DESC LIMIT 10",
            "category": "aggregation_with_joins",
            "source": "failure_pattern",
            "focus": "column_selection"
        },
        {
            "question": "Show driver name, city, and total trips",
            "sql": "SELECT d.first_name || ' ' || d.last_name as driver_name, c.city_name, d.total_trips FROM drivers d JOIN cities c ON d.city_id = c.city_id LIMIT 10",
            "category": "filtering_with_join",
            "source": "failure_pattern",
            "focus": "column_selection"
        },
    ])
    
    # Pattern 9: More aggregate function examples
    queries.extend([
        {
            "question": "What is the total number of vehicles?",
            "sql": "SELECT COUNT(*) as vehicle_count FROM vehicles",
            "category": "simple_aggregation",
            "source": "failure_pattern",
            "focus": "aggregate_functions"
        },
        {
            "question": "What is the total number of cities?",
            "sql": "SELECT COUNT(*) as city_count FROM cities",
            "category": "simple_aggregation",
            "source": "failure_pattern",
            "focus": "aggregate_functions"
        },
        {
            "question": "What is the average driver rating?",
            "sql": "SELECT AVG(rating) as avg_rating FROM drivers WHERE rating IS NOT NULL",
            "category": "simple_aggregation",
            "source": "failure_pattern",
            "focus": "aggregate_functions"
        },
        {
            "question": "What is the total distance traveled?",
            "sql": "SELECT SUM(distance_miles) as total_distance FROM trips WHERE distance_miles IS NOT NULL",
            "category": "simple_aggregation",
            "source": "failure_pattern",
            "focus": "aggregate_functions"
        },
    ])
    
    # Pattern 10: ORDER BY variations
    queries.extend([
        {
            "question": "Show trips ordered by amount ascending",
            "sql": "SELECT trip_id, total_amount FROM trips ORDER BY total_amount ASC LIMIT 10",
            "category": "simple_sorting",
            "source": "failure_pattern",
            "focus": "ordering"
        },
        {
            "question": "Show drivers ordered by total trips descending",
            "sql": "SELECT first_name, last_name, total_trips FROM drivers ORDER BY total_trips DESC LIMIT 10",
            "category": "simple_sorting",
            "source": "failure_pattern",
            "focus": "ordering"
        },
        {
            "question": "Show cities ordered by name",
            "sql": "SELECT city_name, state FROM cities ORDER BY city_name ASC LIMIT 10",
            "category": "simple_sorting",
            "source": "failure_pattern",
            "focus": "ordering"
        },
    ])
    
    # Pattern 11: NULL handling
    queries.extend([
        {
            "question": "Show trips with ratings",
            "sql": "SELECT trip_id, rating, total_amount FROM trips WHERE rating IS NOT NULL LIMIT 10",
            "category": "simple_filtering",
            "source": "failure_pattern",
            "focus": "null_handling"
        },
        {
            "question": "Show drivers without ratings",
            "sql": "SELECT first_name, last_name FROM drivers WHERE rating IS NULL LIMIT 10",
            "category": "simple_filtering",
            "source": "failure_pattern",
            "focus": "null_handling"
        },
    ])
    
    # Validate all queries
    validated = []
    for query in queries:
        valid, results = validate_sql(conn, query["sql"])
        if valid:
            query["expected_result"] = results[:100]
            validated.append(query)
    
    return validated


def generate_schema_based_queries(conn: sqlite3.Connection, num_queries: int = 150) -> List[Dict]:
    """Generate new queries based on database schema."""
    queries = []
    
    # Get actual data for realistic queries - increase limits for more variety
    cities = query_db(conn, "SELECT city_name FROM cities LIMIT 50", return_as_df=False)
    city_names = [c["city_name"] for c in cities]
    
    vehicle_makes = query_db(conn, "SELECT DISTINCT make FROM vehicles WHERE make IS NOT NULL LIMIT 50", return_as_df=False)
    makes = [v["make"] for v in vehicle_makes]
    
    vehicle_types = query_db(conn, "SELECT DISTINCT vehicle_type FROM vehicles WHERE vehicle_type IS NOT NULL LIMIT 20", return_as_df=False)
    types = [v["vehicle_type"] for v in vehicle_types]
    
    payment_methods = query_db(conn, "SELECT DISTINCT payment_method FROM trips WHERE payment_method IS NOT NULL LIMIT 20", return_as_df=False)
    payments = [p["payment_method"] for p in payment_methods]
    
    # Simple SELECT queries - more variations - use more cities
    simple_queries = []
    for city in city_names[:min(30, len(city_names))]:
        simple_queries.extend([
            {
                "question": f"Show all drivers in {city}",
                "sql": f"SELECT d.first_name, d.last_name, d.email FROM drivers d JOIN cities c ON d.city_id = c.city_id WHERE c.city_name = '{city}'",
                "category": "filtering_with_join"
            },
            {
                "question": f"List active drivers in {city}",
                "sql": f"SELECT d.first_name, d.last_name, d.email FROM drivers d JOIN cities c ON d.city_id = c.city_id WHERE c.city_name = '{city}' AND d.status = 'active'",
                "category": "filtering_with_join"
            },
            {
                "question": f"Show all customers in {city}",
                "sql": f"SELECT c.first_name, c.last_name, c.email FROM customers c JOIN cities ci ON c.city_id = ci.city_id WHERE ci.city_name = '{city}' LIMIT 10",
                "category": "filtering_with_join"
            },
        ])
    
    # Vehicle-based queries - more variations - use more makes
    vehicle_queries = []
    for make in makes[:min(30, len(makes))]:
        vehicle_queries.extend([
            {
                "question": f"List all {make} vehicles",
                "sql": f"SELECT vehicle_id, make, model, year FROM vehicles WHERE make = '{make}' LIMIT 10",
                "category": "simple_filtering"
            },
            {
                "question": f"How many {make} vehicles are there?",
                "sql": f"SELECT COUNT(*) as vehicle_count FROM vehicles WHERE make = '{make}'",
                "category": "simple_aggregation"
            },
            {
                "question": f"Show trips using {make} vehicles",
                "sql": f"SELECT t.trip_id, t.total_amount FROM trips t JOIN vehicles v ON t.vehicle_id = v.vehicle_id WHERE v.make = '{make}' LIMIT 10",
                "category": "filtering_with_join"
            },
        ])
    
    # Vehicle type queries - use more types
    for vtype in types[:min(15, len(types))]:
        vehicle_queries.extend([
            {
                "question": f"How many {vtype} vehicles are there?",
                "sql": f"SELECT COUNT(*) as vehicle_count FROM vehicles WHERE vehicle_type = '{vtype}'",
                "category": "simple_aggregation"
            },
            {
                "question": f"What is the total revenue from {vtype} trips?",
                "sql": f"SELECT SUM(t.total_amount) as total_revenue FROM trips t JOIN vehicles v ON t.vehicle_id = v.vehicle_id WHERE v.vehicle_type = '{vtype}' AND t.trip_status = 'completed'",
                "category": "aggregation_with_joins"
            },
        ])
    
    # Aggregation queries - more variations - use more cities
    agg_queries = []
    for city in city_names[:min(30, len(city_names))]:
        agg_queries.extend([
            {
                "question": f"What is the total revenue from {city}?",
                "sql": f"SELECT SUM(t.total_amount) as total_revenue FROM trips t JOIN cities c ON t.pickup_city_id = c.city_id WHERE c.city_name = '{city}' AND t.trip_status = 'completed'",
                "category": "aggregation_with_joins"
            },
            {
                "question": f"How many trips were completed in {city}?",
                "sql": f"SELECT COUNT(*) as trip_count FROM trips t JOIN cities c ON t.pickup_city_id = c.city_id WHERE c.city_name = '{city}' AND t.trip_status = 'completed'",
                "category": "aggregation_with_joins"
            },
            {
                "question": f"What is the average trip amount in {city}?",
                "sql": f"SELECT AVG(t.total_amount) as avg_amount FROM trips t JOIN cities c ON t.pickup_city_id = c.city_id WHERE c.city_name = '{city}' AND t.trip_status = 'completed'",
                "category": "aggregation_with_joins"
            },
        ])
    
    # Payment method queries - use more payment methods
    payment_queries = []
    for payment in payments[:min(15, len(payments))]:
        payment_queries.extend([
            {
                "question": f"How many trips used {payment} payment?",
                "sql": f"SELECT COUNT(*) as trip_count FROM trips WHERE payment_method = '{payment}'",
                "category": "simple_filtering"
            },
            {
                "question": f"What is the total revenue from {payment} payments?",
                "sql": f"SELECT SUM(total_amount) as total_revenue FROM trips WHERE payment_method = '{payment}' AND trip_status = 'completed'",
                "category": "simple_aggregation"
            },
        ])
    
    # Driver rating queries
    driver_queries = [
        {
            "question": "Show drivers with rating above 4.0",
            "sql": "SELECT first_name, last_name, rating FROM drivers WHERE rating > 4.0 ORDER BY rating DESC LIMIT 10",
            "category": "simple_filtering"
        },
        {
            "question": "Show drivers with rating below 3.0",
            "sql": "SELECT first_name, last_name, rating FROM drivers WHERE rating < 3.0 ORDER BY rating ASC LIMIT 10",
            "category": "simple_filtering"
        },
        {
            "question": "What is the average driver rating?",
            "sql": "SELECT AVG(rating) as avg_rating FROM drivers WHERE rating IS NOT NULL",
            "category": "simple_aggregation"
        },
        {
            "question": "How many drivers have completed more than 100 trips?",
            "sql": "SELECT COUNT(*) as driver_count FROM drivers WHERE total_trips > 100",
            "category": "simple_filtering"
        },
    ]
    
    # Customer queries
    customer_queries = [
        {
            "question": "Show customers with highest lifetime value",
            "sql": "SELECT first_name, last_name, lifetime_value FROM customers ORDER BY lifetime_value DESC LIMIT 10",
            "category": "simple_sorting"
        },
        {
            "question": "What is the average customer lifetime value?",
            "sql": "SELECT AVG(lifetime_value) as avg_lifetime_value FROM customers",
            "category": "simple_aggregation"
        },
        {
            "question": "How many customers have more than 10 rides?",
            "sql": "SELECT COUNT(*) as customer_count FROM customers WHERE total_rides > 10",
            "category": "simple_filtering"
        },
    ]
    
    # Trip amount queries
    trip_amount_queries = [
        {
            "question": "Show trips with amount greater than $50",
            "sql": "SELECT trip_id, total_amount, pickup_time FROM trips WHERE total_amount > 50 ORDER BY total_amount DESC LIMIT 10",
            "category": "simple_filtering"
        },
        {
            "question": "Show trips with amount less than $20",
            "sql": "SELECT trip_id, total_amount, pickup_time FROM trips WHERE total_amount < 20 ORDER BY total_amount ASC LIMIT 10",
            "category": "simple_filtering"
        },
        {
            "question": "What is the total revenue from trips over $100?",
            "sql": "SELECT SUM(total_amount) as total_revenue FROM trips WHERE total_amount > 100 AND trip_status = 'completed'",
            "category": "simple_aggregation"
        },
    ]
    
    # Distance queries
    distance_queries = [
        {
            "question": "Show trips longer than 10 miles",
            "sql": "SELECT trip_id, distance_miles, total_amount FROM trips WHERE distance_miles > 10 ORDER BY distance_miles DESC LIMIT 10",
            "category": "simple_filtering"
        },
        {
            "question": "What is the average trip distance?",
            "sql": "SELECT AVG(distance_miles) as avg_distance FROM trips WHERE distance_miles IS NOT NULL",
            "category": "simple_aggregation"
        },
        {
            "question": "Show the longest trips",
            "sql": "SELECT trip_id, distance_miles, duration_minutes FROM trips WHERE distance_miles IS NOT NULL ORDER BY distance_miles DESC LIMIT 10",
            "category": "simple_sorting"
        },
    ]
    
    # Duration queries
    duration_queries = [
        {
            "question": "Show trips longer than 30 minutes",
            "sql": "SELECT trip_id, duration_minutes, total_amount FROM trips WHERE duration_minutes > 30 ORDER BY duration_minutes DESC LIMIT 10",
            "category": "simple_filtering"
        },
        {
            "question": "What is the average trip duration?",
            "sql": "SELECT AVG(duration_minutes) as avg_duration FROM trips WHERE duration_minutes IS NOT NULL",
            "category": "simple_aggregation"
        },
    ]
    
    all_queries = (simple_queries + vehicle_queries + agg_queries + payment_queries + 
                   driver_queries + customer_queries + trip_amount_queries + 
                   distance_queries + duration_queries)
    
    # Validate
    validated = []
    for query in all_queries[:num_queries]:
        valid, results = validate_sql(conn, query["sql"])
        if valid:
            query["expected_result"] = results[:100]
            query["source"] = "schema_based"
            validated.append(query)
    
    return validated


def generate_expanded_golden_dataset(
    existing_data: List[Dict],
    target_size: int = 500,
    conn: sqlite3.Connection = None
) -> List[Dict]:
    """Generate expanded golden dataset for fine-tuning."""
    if conn is None:
        conn = load_db()
    
    golden_dataset = []
    seen_sql = set()
    
    # 1. Add existing data (optional - you might want to keep them separate)
    # Uncomment if you want to include existing examples:
    # for item in existing_data:
    #     sql_norm = item["sql"].strip().upper()
    #     if sql_norm not in seen_sql:
    #         seen_sql.add(sql_norm)
    #         item["source"] = "existing"
    #         golden_dataset.append(item)
    
    print(f"Starting with {len(golden_dataset)} examples")
    
    # 2. Generate variations from existing queries
    print("\nGenerating variations from existing queries...")
    variation_count = 0
    for example in existing_data:
        # Filter variations
        filter_vars = generate_filter_variations(example, conn)
        for var in filter_vars:
            sql_norm = var["sql"].strip().upper()
            if sql_norm not in seen_sql:
                seen_sql.add(sql_norm)
                golden_dataset.append(var)
                variation_count += 1
        
        # Limit variations
        limit_vars = generate_limit_variations(example, conn)
        for var in limit_vars:
            sql_norm = var["sql"].strip().upper()
            if sql_norm not in seen_sql:
                seen_sql.add(sql_norm)
                golden_dataset.append(var)
                variation_count += 1
        
        # Date variations
        if "date" in example["category"]:
            date_vars = generate_date_variations(example, conn)
            for var in date_vars:
                sql_norm = var["sql"].strip().upper()
                if sql_norm not in seen_sql:
                    seen_sql.add(sql_norm)
                    golden_dataset.append(var)
                    variation_count += 1
        
        # Aggregation variations
        if "aggregation" in example["category"]:
            agg_vars = generate_aggregation_variations(example, conn)
            for var in agg_vars:
                sql_norm = var["sql"].strip().upper()
                if sql_norm not in seen_sql:
                    seen_sql.add(sql_norm)
                    golden_dataset.append(var)
                    variation_count += 1
    
    print(f"✓ Generated {variation_count} variations")
    
    # 3. Generate failure pattern queries
    print("\nGenerating failure pattern queries...")
    failure_queries = generate_failure_pattern_queries(conn)
    failure_count = 0
    for query in failure_queries:
        sql_norm = query["sql"].strip().upper()
        if sql_norm not in seen_sql:
            seen_sql.add(sql_norm)
            golden_dataset.append(query)
            failure_count += 1
    print(f"✓ Generated {failure_count} failure pattern queries")
    
    # 4. Generate schema-based queries - generate in batches until target reached
    print("\nGenerating schema-based queries...")
    schema_count = 0
    batch_size = 200
    batch_num = 1
    while len(golden_dataset) < target_size and batch_num <= 5:
        remaining = max(0, target_size - len(golden_dataset))
        if batch_num > 1:
            print(f"  Batch {batch_num}: Generating more schema queries...")
        schema_queries = generate_schema_based_queries(conn, num_queries=min(remaining + 100, batch_size))
        batch_added = 0
        for query in schema_queries:
            if len(golden_dataset) >= target_size:
                break
            sql_norm = query["sql"].strip().upper()
            if sql_norm not in seen_sql:
                seen_sql.add(sql_norm)
                golden_dataset.append(query)
                schema_count += 1
                batch_added += 1
        if batch_num == 1:
            print(f"✓ Generated {schema_count} schema-based queries")
        else:
            print(f"  ✓ Added {batch_added} more schema queries")
        batch_num += 1
    
    # 5. Generate more paraphrases if still below target
    if len(golden_dataset) < target_size:
        print(f"\nGenerating additional paraphrases (current: {len(golden_dataset)}, target: {target_size})...")
        paraphrase_count = 0
        
        # Generate paraphrases for existing queries
        for item in golden_dataset[:min(100, len(golden_dataset))]:
            question = item["question"]
            sql = item["sql"]
            
            # Simple word replacements (only if SQL stays the same)
            paraphrases = []
            if "What are" in question:
                paraphrases.append(question.replace("What are", "List all"))
                paraphrases.append(question.replace("What are", "Show all"))
            if "Show" in question and "Show all" not in question:
                paraphrases.append(question.replace("Show", "Display"))
                paraphrases.append(question.replace("Show", "List"))
            if "How many" in question:
                paraphrases.append(question.replace("How many", "What is the count of"))
                paraphrases.append(question.replace("How many", "Count"))
            if "What is the" in question:
                paraphrases.append(question.replace("What is the", "Calculate the"))
                paraphrases.append(question.replace("What is the", "Get the"))
            
            for para_question in paraphrases:
                if para_question != question:
                    sql_norm = sql.strip().upper()
                    if sql_norm not in seen_sql:
                        seen_sql.add(sql_norm)
                        new_item = item.copy()
                        new_item["question"] = para_question
                        new_item["source"] = "paraphrase"
                        golden_dataset.append(new_item)
                        paraphrase_count += 1
                        if len(golden_dataset) >= target_size:
                            break
            if len(golden_dataset) >= target_size:
                break
        
        print(f"✓ Generated {paraphrase_count} paraphrases")
    
    # 6. Generate additional simple queries if still below target - loop until target reached
    iteration = 0
    max_iterations = 10
    while len(golden_dataset) < target_size and iteration < max_iterations:
        iteration += 1
        if iteration == 1:
            print(f"\nGenerating additional simple queries (current: {len(golden_dataset)}, target: {target_size})...")
        else:
            print(f"\nIteration {iteration}: Generating more queries (current: {len(golden_dataset)}, target: {target_size})...")
        simple_count = 0
        
        # Get more data from database - expand limits for more variety
        cities = query_db(conn, "SELECT city_name FROM cities LIMIT 50", return_as_df=False)
        city_names = [c["city_name"] for c in cities]
        
        makes = query_db(conn, "SELECT DISTINCT make FROM vehicles WHERE make IS NOT NULL LIMIT 50", return_as_df=False)
        vehicle_makes = [v["make"] for v in makes]
        
        # More simple queries - generate more variations per iteration
        additional_queries = []
        
        # City-based simple queries - use more cities
        city_start = (iteration - 1) * 10
        city_end = min(city_start + 20, len(city_names))
        for city in city_names[city_start:city_end]:
            additional_queries.extend([
                {
                    "question": f"Show trips from {city}",
                    "sql": f"SELECT trip_id, total_amount FROM trips t JOIN cities c ON t.pickup_city_id = c.city_id WHERE c.city_name = '{city}' LIMIT 10",
                    "category": "filtering_with_join"
                },
                {
                    "question": f"Count trips in {city}",
                    "sql": f"SELECT COUNT(*) as trip_count FROM trips t JOIN cities c ON t.pickup_city_id = c.city_id WHERE c.city_name = '{city}'",
                    "category": "aggregation_with_joins"
                },
            ])
        
        # Vehicle-based simple queries - use more makes
        make_start = (iteration - 1) * 10
        make_end = min(make_start + 20, len(vehicle_makes))
        for make in vehicle_makes[make_start:make_end]:
            additional_queries.extend([
                {
                    "question": f"Show {make} vehicle details",
                    "sql": f"SELECT make, model, year, vehicle_type FROM vehicles WHERE make = '{make}' LIMIT 10",
                    "category": "simple_filtering"
                },
            ])
        
        # More driver queries
        additional_queries.extend([
            {
                "question": "Show all driver emails",
                "sql": "SELECT email FROM drivers LIMIT 10",
                "category": "simple_filtering"
            },
            {
                "question": "List all customer emails",
                "sql": "SELECT email FROM customers LIMIT 10",
                "category": "simple_filtering"
            },
            {
                "question": "Show all vehicle types",
                "sql": "SELECT DISTINCT vehicle_type FROM vehicles WHERE vehicle_type IS NOT NULL",
                "category": "simple_filtering"
            },
            {
                "question": "Show all trip statuses",
                "sql": "SELECT DISTINCT trip_status FROM trips",
                "category": "simple_filtering"
            },
            {
                "question": "Show all driver statuses",
                "sql": "SELECT DISTINCT status FROM drivers",
                "category": "simple_filtering"
            },
            {
                "question": "Show all payment methods",
                "sql": "SELECT DISTINCT payment_method FROM trips WHERE payment_method IS NOT NULL",
                "category": "simple_filtering"
            },
            {
                "question": "Show all states",
                "sql": "SELECT DISTINCT state FROM cities",
                "category": "simple_filtering"
            },
            {
                "question": "Show all vehicle makes",
                "sql": "SELECT DISTINCT make FROM vehicles WHERE make IS NOT NULL",
                "category": "simple_filtering"
            },
            {
                "question": "Show all vehicle models",
                "sql": "SELECT DISTINCT model FROM vehicles WHERE model IS NOT NULL LIMIT 20",
                "category": "simple_filtering"
            },
            {
                "question": "Show all cities",
                "sql": "SELECT city_name, state FROM cities ORDER BY city_name",
                "category": "simple_sorting"
            },
            {
                "question": "Show all drivers",
                "sql": "SELECT first_name, last_name, email FROM drivers LIMIT 10",
                "category": "simple_filtering"
            },
            {
                "question": "Show all customers",
                "sql": "SELECT first_name, last_name, email FROM customers LIMIT 10",
                "category": "simple_filtering"
            },
            {
                "question": "Show all vehicles",
                "sql": "SELECT vehicle_id, make, model, year FROM vehicles LIMIT 10",
                "category": "simple_filtering"
            },
            {
                "question": "Show all trips",
                "sql": "SELECT trip_id, total_amount, pickup_time FROM trips LIMIT 10",
                "category": "simple_filtering"
            },
            {
                "question": "What is the total number of drivers?",
                "sql": "SELECT COUNT(*) as driver_count FROM drivers",
                "category": "simple_aggregation"
            },
            {
                "question": "What is the total number of customers?",
                "sql": "SELECT COUNT(*) as customer_count FROM customers",
                "category": "simple_aggregation"
            },
            {
                "question": "What is the total number of trips?",
                "sql": "SELECT COUNT(*) as trip_count FROM trips",
                "category": "simple_aggregation"
            },
            {
                "question": "What is the total number of completed trips?",
                "sql": "SELECT COUNT(*) as completed_trips FROM trips WHERE trip_status = 'completed'",
                "category": "simple_aggregation"
            },
            {
                "question": "What is the total revenue?",
                "sql": "SELECT SUM(total_amount) as total_revenue FROM trips WHERE trip_status = 'completed'",
                "category": "simple_aggregation"
            },
            {
                "question": "Show trips ordered by amount",
                "sql": "SELECT trip_id, total_amount FROM trips ORDER BY total_amount DESC LIMIT 10",
                "category": "simple_sorting"
            },
            {
                "question": "Show trips ordered by pickup time",
                "sql": "SELECT trip_id, pickup_time FROM trips ORDER BY pickup_time DESC LIMIT 10",
                "category": "simple_sorting"
            },
            {
                "question": "Show drivers ordered by rating",
                "sql": "SELECT first_name, last_name, rating FROM drivers WHERE rating IS NOT NULL ORDER BY rating DESC LIMIT 10",
                "category": "simple_sorting"
            },
        ])
        
        # Validate and add
        for query in additional_queries:
            if len(golden_dataset) >= target_size:
                break
            valid, results = validate_sql(conn, query["sql"])
            if valid:
                sql_norm = query["sql"].strip().upper()
                if sql_norm not in seen_sql:
                    seen_sql.add(sql_norm)
                    query["expected_result"] = results[:100]
                    query["source"] = "additional_simple"
                    golden_dataset.append(query)
                    simple_count += 1
        
        print(f"✓ Generated {simple_count} additional simple queries")
        
        # If we're still far from target, generate more schema-based queries
        if len(golden_dataset) < target_size and iteration >= 2:
            remaining = target_size - len(golden_dataset)
            if remaining > 100:
                print(f"\nGenerating more schema-based queries (need {remaining} more)...")
                more_schema = generate_schema_based_queries(conn, num_queries=min(remaining, 200))
                schema_added = 0
                for query in more_schema:
                    if len(golden_dataset) >= target_size:
                        break
                    sql_norm = query["sql"].strip().upper()
                    if sql_norm not in seen_sql:
                        seen_sql.add(sql_norm)
                        golden_dataset.append(query)
                        schema_added += 1
                print(f"✓ Added {schema_added} more schema-based queries")
    
    # Final expansion: Create question variations to reach target
    if len(golden_dataset) < target_size:
        print(f"\nFinal expansion: Creating question variations (current: {len(golden_dataset)}, target: {target_size})...")
        original_size = len(golden_dataset)
        expansion_count = 0
        
        # Create variations of existing questions (same SQL, different wording)
        for item in list(golden_dataset):  # Iterate over copy
            if len(golden_dataset) >= target_size:
                break
            question = item["question"]
            sql = item["sql"]
            
            # Create question variations
            variations = []
            if "Show" in question and "Show all" not in question:
                variations.append(question.replace("Show", "List"))
                variations.append(question.replace("Show", "Display"))
            if "List" in question and "List all" not in question:
                variations.append(question.replace("List", "Show"))
            if "What are" in question:
                variations.append(question.replace("What are", "List all"))
                variations.append(question.replace("What are", "Show all"))
            if "How many" in question:
                variations.append(question.replace("How many", "What is the count of"))
            if "What is the" in question and "count" not in question.lower():
                variations.append(question.replace("What is the", "Show the"))
            
            for var_question in variations[:3]:  # Allow up to 3 variations per query
                if len(golden_dataset) >= target_size:
                    break
                # Check if this variation already exists
                var_exists = any(q.get("question") == var_question for q in golden_dataset)
                if not var_exists:
                    new_item = item.copy()
                    new_item["question"] = var_question
                    new_item["source"] = new_item.get("source", "unknown") + "_variation"
                    golden_dataset.append(new_item)
                    expansion_count += 1
        
        print(f"✓ Added {expansion_count} question variations")
        
        # If still below target, create more variations from the expanded set
        if len(golden_dataset) < target_size:
            print(f"Second expansion round (current: {len(golden_dataset)}, target: {target_size})...")
            second_round_count = 0
            for item in list(golden_dataset[original_size:]):  # Only from newly added items
                if len(golden_dataset) >= target_size:
                    break
                question = item["question"]
                
                # More aggressive variations
                more_variations = []
                if "all" in question.lower():
                    more_variations.append(question.replace("all ", "").replace("All ", ""))
                if "ordered by" in question.lower():
                    more_variations.append(question.replace("ordered by", "sorted by"))
                if "sorted by" in question.lower():
                    more_variations.append(question.replace("sorted by", "ordered by"))
                if "total" in question.lower():
                    more_variations.append(question.replace("total", "sum of"))
                
                for var_question in more_variations[:2]:
                    if len(golden_dataset) >= target_size:
                        break
                    var_exists = any(q.get("question") == var_question for q in golden_dataset)
                    if not var_exists and var_question != question:
                        new_item = item.copy()
                        new_item["question"] = var_question
                        new_item["source"] = new_item.get("source", "unknown") + "_variation2"
                        golden_dataset.append(new_item)
                        second_round_count += 1
            
            print(f"✓ Added {second_round_count} more variations in second round")
            
            # Third round: Even more aggressive - use all items
            if len(golden_dataset) < target_size:
                print(f"Third expansion round (current: {len(golden_dataset)}, target: {target_size})...")
                third_round_count = 0
                for item in list(golden_dataset):  # Use all items
                    if len(golden_dataset) >= target_size:
                        break
                    question = item["question"]
                    
                    # Create even more variations
                    third_variations = []
                    # Add "please" variations
                    if not question.startswith("Please"):
                        third_variations.append("Please " + question.lower())
                    # Add question mark variations
                    if not question.endswith("?"):
                        third_variations.append(question + "?")
                    # Replace common words
                    if "get" in question.lower():
                        third_variations.append(question.replace("get", "retrieve").replace("Get", "Retrieve"))
                    if "find" in question.lower():
                        third_variations.append(question.replace("find", "get").replace("Find", "Get"))
                    
                    for var_question in third_variations[:1]:  # Limit to 1 per query
                        if len(golden_dataset) >= target_size:
                            break
                        var_exists = any(q.get("question") == var_question for q in golden_dataset)
                        if not var_exists and var_question != question:
                            new_item = item.copy()
                            new_item["question"] = var_question
                            new_item["source"] = new_item.get("source", "unknown") + "_variation3"
                            golden_dataset.append(new_item)
                            third_round_count += 1
                
                print(f"✓ Added {third_round_count} more variations in third round")
    
    print(f"\n✓ Total unique examples: {len(golden_dataset)}")
    
    return golden_dataset


def export_sft_format(data: List[Dict], output_file: str = "golden_dataset_sft.jsonl"):
    """Export golden dataset in SFT format (JSONL)."""
    with open(output_file, 'w') as f:
        for item in data:
            sft_item = {
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a SQL expert. Generate accurate SQLite queries based on user questions."
                    },
                    {
                        "role": "user",
                        "content": item["question"]
                    },
                    {
                        "role": "assistant",
                        "content": json.dumps({
                            "sql_query": item["sql"],
                            "explanation": f"SQL query for: {item['question']}",
                            "category": item.get("category", "unknown"),
                            "tables_used": extract_tables_from_sql(item["sql"])
                        }, ensure_ascii=False)
                    }
                ]
            }
            f.write(json.dumps(sft_item, ensure_ascii=False) + '\n')
    
    print(f"✓ Exported {len(data)} examples to {output_file}")


def main():
    print("=" * 80)
    print("Golden Dataset Generator for Fine-Tuning")
    print("=" * 80)
    
    # Load existing data for reference (but keep datasets separate)
    print("\n1. Loading existing evaluation data for reference...")
    try:
        existing_data = json.load(open("evaluation_data.json"))
        print(f"   Found {len(existing_data)} existing examples (will keep separate)")
    except FileNotFoundError:
        print("   No existing evaluation_data.json found (will generate from scratch)")
        existing_data = []
    
    # Load database
    print("\n2. Loading database...")
    conn = load_db()
    print("   ✓ Database loaded")
    
    # Generate golden dataset
    print("\n3. Generating golden dataset for fine-tuning...")
    target_size = 1000  # Target at least 1000 examples for better fine-tuning
    golden_dataset = generate_expanded_golden_dataset(
        existing_data, 
        target_size=target_size, 
        conn=conn
    )
    
    # Save golden dataset (separate from evaluation_data.json)
    print("\n4. Saving golden dataset...")
    output_file = "golden_dataset.json"
    with open(output_file, 'w') as f:
        json.dump(golden_dataset, f, indent=2, ensure_ascii=False)
    print(f"   ✓ Saved {len(golden_dataset)} examples to {output_file}")
    
    # Export SFT format
    print("\n5. Exporting SFT format...")
    export_sft_format(golden_dataset, "golden_dataset_sft.jsonl")
    
    # Statistics
    print("\n" + "=" * 80)
    print("Golden Dataset Statistics")
    print("=" * 80)
    print(f"Total examples: {len(golden_dataset)}")
    
    # Category distribution
    categories = {}
    sources = {}
    for item in golden_dataset:
        cat = item.get("category", "unknown")
        categories[cat] = categories.get(cat, 0) + 1
        src = item.get("source", "unknown")
        sources[src] = sources.get(src, 0) + 1
    
    print("\nCategory distribution:")
    for cat, count in sorted(categories.items()):
        print(f"  {cat}: {count}")
    
    print("\nSource distribution:")
    for src, count in sorted(sources.items()):
        print(f"  {src}: {count}")
    
    print(f"\n✓ Golden dataset ready for fine-tuning!")
    print(f"  - JSON format: {output_file}")
    print(f"  - SFT format: golden_dataset_sft.jsonl")
    print(f"  - Can be merged with evaluation_data.json later if needed")
    
    conn.close()


if __name__ == "__main__":
    main()

