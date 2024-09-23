import requests
import json
import time

# Ticketmaster API details
API_KEY = '6EZzcyC5Mrtly9yVFMBfyGkXsXVPVLdp'  # Replace with your Ticketmaster API Key
BASE_URL = 'https://app.ticketmaster.com/discovery/v2/events.json'

# Function to fetch events from the API
def fetch_events(page_num):
    params = {
        'apikey': API_KEY,
        'size': 200,  # Number of events per request
        'page': page_num,  # Page number for pagination
    }
    response = requests.get(BASE_URL, params=params)
    data = response.json()
    return data

# Fetch and store the JSON data for ~1000 events
events_data = []
page_num = 0
total_events = 0

while total_events < 1000:
    try:
        data = fetch_events(page_num)
        if '_embedded' in data and 'events' in data['_embedded']:
            events = data['_embedded']['events']
            events_data.extend(events)  # Add events to the list
            total_events += len(events)
            page_num += 1
            print(f"Fetched {len(events)} events from page {page_num}. Total events fetched: {total_events}")
            
            # Respect API rate limits (you can adjust the sleep time based on limits)
            time.sleep(10)
        else:
            print("No more events found.")
            break
    except Exception as e:
        print(f"An error occurred: {e}")
        break

# Save the fetched data to a JSON file
with open('events_data.json', 'w') as json_file:
    json.dump(events_data, json_file, indent=4)

print(f"Saved {total_events} events to 'events_data.json'.")
