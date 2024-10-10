import requests
import json
import time
''' This file contains boilerplate code for pulling event data from the Ticketmaster API. It utilizes the "requests" library to get JSON data from the API,
filtered by the event segment (Music, Sports, Misc) and source (using only events from ticketmaster and none of its subsidiaries).'''

# Ticketmaster API details
API_KEY = '6EZzcyC5Mrtly9yVFMBfyGkXsXVPVLdp' 
BASE_URL = 'https://app.ticketmaster.com/discovery/v2/events.json'

# Function to fetch events from the API
'''Since the API returns JSON data, we use the python "JSON" library to store the response'''
def fetch_events(page_num, segment):
    params = {
        'apikey': API_KEY,
        'size': 100,  # Number of events per request
        'page': page_num,  # Page number for pagination
        'segmentName': segment,
        'source': 'ticketmaster'
    }
    response = requests.get(BASE_URL, params=params)
    data = response.json()
    return data

# Fetch and store the JSON data for ~1000 events
events_data = []
events_seen = set()
page_num = 0
total_events = 0
segments = ['Sports', 'Music', 'Miscellaneous', 'Sports', 'Music', 'Miscellaneous', 'Music', 'Music', 'Miscellaneous', 'Miscellaneous']
i = 0

while total_events < 1000:
    '''Iterating through Segments to ensure that we get a mix of all event types, we access the JSON data returned by an API call and add it to a list.
    We turn multiple pages until we reach our API limit of event returns (1000, which unfortunately isn't much once we clear duplicates).'''
    segment = segments[i]
    i += 1
    try:
        data = fetch_events(page_num, segment)
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
'''Using the JSON library, we dump our list of JSON event objects into a single JSON file for intermediate storage'''
with open('events_data.json', 'w') as json_file:
    json.dump(events_data, json_file, indent=4)

print(f"Saved {total_events} events to 'events_data.json'.")
