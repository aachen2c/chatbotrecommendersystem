import random
import pandas as pd 
# loading the data
import sqlite3
conn = sqlite3.connect('events.db')
events = pd.read_sql('SELECT * FROM events', conn)
conn.close()
# List of unique event IDs
unique_event_ids = pd.unique(events['id'])
# Number of unique users and total attendance records
num_users = 500  # Number of unique users
num_attendance_records = 10000  # Total number of attendance records to generate
# Generate the unique user IDs
user_ids = [f"user_{i}" for i in range(1, num_users + 1)]
# Randomly pair user IDs with event IDs to create attendance records
attendance_records = [
    {"userID": random.choice(user_ids), "eventID": random.choice(unique_event_ids)}
    for _ in range(num_attendance_records)
]
# Convert to a DataFrame
attendance_df = pd.DataFrame(attendance_records)
# Save users data and interactions data to CSV for later use
attendance_df.to_csv('attendance.csv', index=False)
