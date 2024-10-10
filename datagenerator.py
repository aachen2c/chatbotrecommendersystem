import random
import pandas as pd 
# loading the data
import sqlite3
'''In this document, we construct synthetic interaction data for 500 users. These "interactions" are simply implicit indicator variables connecting users and events.
They are intentionally vague, and are randomly generated because I didn't want to confound operation of the recommender with my own knowledge of the 
underlying interaction model'''

conn = sqlite3.connect('events2.db')
events = pd.read_sql('SELECT * FROM events', conn)
conn.close()
'''After quickly loading the data into pandas using pd.read_sql(QUERY, connection), we grab the list of ids (all ids should be unique, but just in case...)'''
# List of unique event IDs
unique_event_ids = pd.unique(events['id'])
# Number of unique users and total attendance records
num_users = 500  # Number of unique users
num_attendance_records = 10000  # Total number of attendance records to generate
# Generate the unique user IDs
user_ids = [f"user_{i}" for i in range(1, num_users + 1)]
# Randomly pair user IDs with event IDs to create attendance records
'''We create a list of 10000 user/event pairs, then convert this list of rows (represented by dict) into a DataFrame'''
attendance_records = [
    {"userID": random.choice(user_ids), "eventID": random.choice(unique_event_ids)}
    for _ in range(num_attendance_records)
]
# Convert to a DataFrame
attendance_df = pd.DataFrame(attendance_records)
# Save users data and interactions data to CSV for later use
attendance_df.to_csv('attendance.csv', index=False)
