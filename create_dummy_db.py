#!/usr/bin/env python3
"""
Script to replace Chinook music database with dummy tables and data
inspired by Uber's QueryGPT examples.
"""

import sqlite3
import random
from datetime import datetime, timedelta
from pathlib import Path


def create_dummy_database(db_path: str = "Chinook.db"):
    """Create dummy tables and insert dummy data."""
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Drop all existing tables (excluding system tables)
    print("Dropping existing tables...")
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")
    tables = cursor.fetchall()
    for table in tables:
        cursor.execute(f"DROP TABLE IF EXISTS {table[0]}")
    
    # Create cities table
    print("Creating cities table...")
    cursor.execute("""
        CREATE TABLE cities (
            city_id INTEGER PRIMARY KEY AUTOINCREMENT,
            city_name TEXT NOT NULL,
            state TEXT,
            country TEXT NOT NULL,
            timezone TEXT,
            population INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Create drivers table
    print("Creating drivers table...")
    cursor.execute("""
        CREATE TABLE drivers (
            driver_id INTEGER PRIMARY KEY AUTOINCREMENT,
            first_name TEXT NOT NULL,
            last_name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            phone TEXT,
            city_id INTEGER,
            status TEXT NOT NULL,
            rating REAL,
            total_trips INTEGER DEFAULT 0,
            joined_date DATE,
            vehicle_type TEXT,
            FOREIGN KEY (city_id) REFERENCES cities(city_id)
        )
    """)
    
    # Create vehicles table
    print("Creating vehicles table...")
    cursor.execute("""
        CREATE TABLE vehicles (
            vehicle_id INTEGER PRIMARY KEY AUTOINCREMENT,
            driver_id INTEGER NOT NULL,
            make TEXT NOT NULL,
            model TEXT NOT NULL,
            year INTEGER,
            color TEXT,
            license_plate TEXT UNIQUE,
            vehicle_type TEXT,
            status TEXT,
            FOREIGN KEY (driver_id) REFERENCES drivers(driver_id)
        )
    """)
    
    # Create customers table
    print("Creating customers table...")
    cursor.execute("""
        CREATE TABLE customers (
            customer_id INTEGER PRIMARY KEY AUTOINCREMENT,
            first_name TEXT NOT NULL,
            last_name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            phone TEXT,
            city_id INTEGER,
            signup_date DATE,
            total_rides INTEGER DEFAULT 0,
            lifetime_value REAL DEFAULT 0.0,
            FOREIGN KEY (city_id) REFERENCES cities(city_id)
        )
    """)
    
    # Create trips table (similar to uber.trips_data)
    print("Creating trips table...")
    cursor.execute("""
        CREATE TABLE trips (
            trip_id INTEGER PRIMARY KEY AUTOINCREMENT,
            driver_id INTEGER NOT NULL,
            customer_id INTEGER NOT NULL,
            vehicle_id INTEGER,
            pickup_city_id INTEGER NOT NULL,
            dropoff_city_id INTEGER,
            trip_status TEXT NOT NULL,
            trip_type TEXT,
            distance_miles REAL,
            duration_minutes INTEGER,
            fare_amount REAL NOT NULL,
            tip_amount REAL DEFAULT 0.0,
            total_amount REAL NOT NULL,
            pickup_time TIMESTAMP NOT NULL,
            dropoff_time TIMESTAMP,
            payment_method TEXT,
            rating INTEGER,
            FOREIGN KEY (driver_id) REFERENCES drivers(driver_id),
            FOREIGN KEY (customer_id) REFERENCES customers(customer_id),
            FOREIGN KEY (vehicle_id) REFERENCES vehicles(vehicle_id),
            FOREIGN KEY (pickup_city_id) REFERENCES cities(city_id),
            FOREIGN KEY (dropoff_city_id) REFERENCES cities(city_id)
        )
    """)
    
    # Create promotions table
    print("Creating promotions table...")
    cursor.execute("""
        CREATE TABLE promotions (
            promotion_id INTEGER PRIMARY KEY AUTOINCREMENT,
            promotion_code TEXT UNIQUE NOT NULL,
            description TEXT,
            discount_type TEXT,
            discount_value REAL,
            start_date DATE NOT NULL,
            end_date DATE,
            max_uses INTEGER,
            current_uses INTEGER DEFAULT 0,
            is_active INTEGER DEFAULT 1
        )
    """)
    
    # Create trip_promotions table (many-to-many)
    print("Creating trip_promotions table...")
    cursor.execute("""
        CREATE TABLE trip_promotions (
            trip_id INTEGER NOT NULL,
            promotion_id INTEGER NOT NULL,
            discount_applied REAL,
            PRIMARY KEY (trip_id, promotion_id),
            FOREIGN KEY (trip_id) REFERENCES trips(trip_id),
            FOREIGN KEY (promotion_id) REFERENCES promotions(promotion_id)
        )
    """)
    
    # Insert dummy cities
    print("Inserting dummy cities...")
    cities_data = [
        ("Seattle", "Washington", "USA", "PST", 750000),
        ("Los Angeles", "California", "USA", "PST", 4000000),
        ("New York", "New York", "USA", "EST", 8000000),
        ("Chicago", "Illinois", "USA", "CST", 2700000),
        ("San Francisco", "California", "USA", "PST", 900000),
        ("Boston", "Massachusetts", "USA", "EST", 700000),
        ("Miami", "Florida", "USA", "EST", 450000),
        ("Austin", "Texas", "USA", "CST", 1000000),
        ("Denver", "Colorado", "USA", "MST", 700000),
        ("Portland", "Oregon", "USA", "PST", 650000),
    ]
    cursor.executemany("""
        INSERT INTO cities (city_name, state, country, timezone, population)
        VALUES (?, ?, ?, ?, ?)
    """, cities_data)
    
    # Insert dummy drivers
    print("Inserting dummy drivers...")
    first_names = ["John", "Sarah", "Michael", "Emily", "David", "Jessica", "Robert", "Amanda", "James", "Lisa"]
    last_names = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis", "Rodriguez", "Martinez"]
    statuses = ["active", "inactive", "suspended"]
    vehicle_types = ["sedan", "suv", "luxury", "economy"]
    
    drivers_data = []
    for i in range(50):
        city_id = random.randint(1, 10)
        status = random.choice(statuses)
        rating = round(random.uniform(3.5, 5.0), 2)
        total_trips = random.randint(0, 500)
        joined_date = (datetime.now() - timedelta(days=random.randint(30, 1000))).date()
        vehicle_type = random.choice(vehicle_types)
        
        drivers_data.append((
            random.choice(first_names),
            random.choice(last_names),
            f"driver{i}@example.com",
            f"+1-555-{random.randint(100,999)}-{random.randint(1000,9999)}",
            city_id,
            status,
            rating,
            total_trips,
            joined_date.isoformat(),
            vehicle_type
        ))
    
    cursor.executemany("""
        INSERT INTO drivers (first_name, last_name, email, phone, city_id, status, rating, total_trips, joined_date, vehicle_type)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, drivers_data)
    
    # Insert dummy vehicles
    print("Inserting dummy vehicles...")
    makes = ["Toyota", "Honda", "Ford", "Chevrolet", "Nissan", "BMW", "Mercedes", "Tesla", "Hyundai", "Kia"]
    models = ["Camry", "Accord", "Fusion", "Malibu", "Altima", "3 Series", "C-Class", "Model 3", "Elantra", "Optima"]
    colors = ["Black", "White", "Silver", "Gray", "Blue", "Red"]
    vehicle_statuses = ["active", "maintenance", "retired"]
    
    vehicles_data = []
    for i in range(50):
        driver_id = i + 1
        make = random.choice(makes)
        model = random.choice(models)
        year = random.randint(2018, 2024)
        color = random.choice(colors)
        license_plate = f"{random.choice(['ABC', 'XYZ', 'DEF', 'GHI'])}{random.randint(1000, 9999)}"
        vehicle_type = random.choice(vehicle_types)
        status = random.choice(vehicle_statuses)
        
        vehicles_data.append((
            driver_id,
            make,
            model,
            year,
            color,
            license_plate,
            vehicle_type,
            status
        ))
    
    cursor.executemany("""
        INSERT INTO vehicles (driver_id, make, model, year, color, license_plate, vehicle_type, status)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, vehicles_data)
    
    # Insert dummy customers
    print("Inserting dummy customers...")
    customers_data = []
    for i in range(200):
        city_id = random.randint(1, 10)
        signup_date = (datetime.now() - timedelta(days=random.randint(1, 730))).date()
        total_rides = random.randint(0, 100)
        lifetime_value = round(random.uniform(0, 2000), 2)
        
        customers_data.append((
            random.choice(first_names),
            random.choice(last_names),
            f"customer{i}@example.com",
            f"+1-555-{random.randint(100,999)}-{random.randint(1000,9999)}",
            city_id,
            signup_date.isoformat(),
            total_rides,
            lifetime_value
        ))
    
    cursor.executemany("""
        INSERT INTO customers (first_name, last_name, email, phone, city_id, signup_date, total_rides, lifetime_value)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, customers_data)
    
    # Insert dummy trips
    print("Inserting dummy trips...")
    trip_statuses = ["completed", "cancelled", "in_progress", "scheduled"]
    trip_types = ["standard", "premium", "pool", "xl"]
    payment_methods = ["credit_card", "debit_card", "paypal", "apple_pay", "google_pay"]
    
    trips_data = []
    base_date = datetime.now() - timedelta(days=30)
    
    for i in range(1000):
        driver_id = random.randint(1, 50)
        customer_id = random.randint(1, 200)
        vehicle_id = random.randint(1, 50)
        pickup_city_id = random.randint(1, 10)
        dropoff_city_id = random.randint(1, 10)
        status = random.choice(trip_statuses)
        trip_type = random.choice(trip_types)
        distance_miles = round(random.uniform(0.5, 25.0), 2)
        duration_minutes = random.randint(5, 60)
        fare_amount = round(random.uniform(5.0, 50.0), 2)
        tip_amount = round(random.uniform(0.0, fare_amount * 0.2), 2) if random.random() > 0.3 else 0.0
        total_amount = fare_amount + tip_amount
        
        pickup_time = base_date + timedelta(
            days=random.randint(0, 30),
            hours=random.randint(0, 23),
            minutes=random.randint(0, 59)
        )
        dropoff_time = pickup_time + timedelta(minutes=duration_minutes) if status == "completed" else None
        payment_method = random.choice(payment_methods)
        rating = random.randint(1, 5) if status == "completed" and random.random() > 0.2 else None
        
        trips_data.append((
            driver_id,
            customer_id,
            vehicle_id,
            pickup_city_id,
            dropoff_city_id,
            status,
            trip_type,
            distance_miles,
            duration_minutes,
            fare_amount,
            tip_amount,
            total_amount,
            pickup_time.isoformat(),
            dropoff_time.isoformat() if dropoff_time else None,
            payment_method,
            rating
        ))
    
    cursor.executemany("""
        INSERT INTO trips (driver_id, customer_id, vehicle_id, pickup_city_id, dropoff_city_id, 
                          trip_status, trip_type, distance_miles, duration_minutes, fare_amount, 
                          tip_amount, total_amount, pickup_time, dropoff_time, payment_method, rating)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, trips_data)
    
    # Insert dummy promotions
    print("Inserting dummy promotions...")
    promotions_data = [
        ("WELCOME10", "Welcome discount", "percentage", 10.0, "2024-01-01", "2024-12-31", 1000, 450, 1),
        ("SUMMER20", "Summer promotion", "percentage", 20.0, "2024-06-01", "2024-08-31", 500, 320, 1),
        ("FIRST5", "First ride discount", "fixed", 5.0, "2024-01-01", None, None, 180, 1),
        ("WEEKEND15", "Weekend special", "percentage", 15.0, "2024-01-01", "2024-12-31", 2000, 890, 1),
        ("LOYALTY25", "Loyalty reward", "percentage", 25.0, "2024-03-01", None, 300, 95, 1),
    ]
    cursor.executemany("""
        INSERT INTO promotions (promotion_code, description, discount_type, discount_value, 
                               start_date, end_date, max_uses, current_uses, is_active)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, promotions_data)
    
    # Insert dummy trip_promotions
    print("Inserting dummy trip promotions...")
    trip_promotions_data = []
    seen_combinations = set()
    for i in range(200):
        while True:
            trip_id = random.randint(1, 1000)
            promotion_id = random.randint(1, 5)
            combination = (trip_id, promotion_id)
            if combination not in seen_combinations:
                seen_combinations.add(combination)
                discount_applied = round(random.uniform(2.0, 10.0), 2)
                trip_promotions_data.append((trip_id, promotion_id, discount_applied))
                break
    
    cursor.executemany("""
        INSERT INTO trip_promotions (trip_id, promotion_id, discount_applied)
        VALUES (?, ?, ?)
    """, trip_promotions_data)
    
    # Commit and close
    conn.commit()
    conn.close()
    
    print("\n✓ Database created successfully!")
    print("\nTables created:")
    print("  - cities (10 cities)")
    print("  - drivers (50 drivers)")
    print("  - vehicles (50 vehicles)")
    print("  - customers (200 customers)")
    print("  - trips (1000 trips)")
    print("  - promotions (5 promotions)")
    print("  - trip_promotions (200 associations)")


if __name__ == "__main__":
    db_path = Path("Chinook.db")
    if not db_path.exists():
        print(f"Error: Database file not found: {db_path}")
        print("Please run setup.sh first to create the database.")
        exit(1)
    
    print("Creating dummy database...")
    print("=" * 60)
    create_dummy_database(str(db_path))
    print("=" * 60)
    print("\nDone!")

