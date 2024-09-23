import json
import sqlite3

# Load the JSON data
with open('events_data.json', 'r') as json_file:
    events_data = json.load(json_file)

# Connect to SQLite database (or create it)
conn = sqlite3.connect('events.db')
cursor = conn.cursor()

# Create the events table with combined_features column
cursor.execute('''DROP TABLE IF EXISTS events''')
cursor.execute('''CREATE TABLE IF NOT EXISTS events (
                    id TEXT PRIMARY KEY, 
                    name TEXT UNIQUE,  -- Ensure event titles are unique to prevent duplicates
                    url TEXT, 
                    locale TEXT,
                    timezone TEXT,
                    start_date TIMESTAMP, 
                    status TEXT, 
                    public_sales_start TIMESTAMP,
                    public_sales_end TIMESTAMP,
                    presales_name TEXT,
                    presales_start TIMESTAMP,
                    presales_end TIMESTAMP,
                    multiday BOOL,
                    venue_name TEXT, 
                    city TEXT, 
                    state TEXT, 
                    country TEXT, 
                    address TEXT,
                    longitude REAL,
                    latitude REAL,
                    genre TEXT, 
                    subGenre TEXT,
                    segment TEXT, 
                    type TEXT,
                    subType TEXT,
                    family BOOL,
                    currency TEXT,
                    price_min REAL, 
                    price_max REAL,
                    agerestricted BOOL,
                    attractions_string TEXT,
                    combined_features TEXT''')

# Insert events into the database
for event in events_data:
    # Extract relevant fields from the event
    event_id = event['id']
    name = event['name']
    
    # Check for duplicate event title (name)
    cursor.execute('SELECT name FROM events WHERE name = ?', (name,))
    if cursor.fetchone():
        print(f"Skipping duplicate event: {name}")
        continue  # Skip if the title is already in the database
    
    url = event['url']
    locale = event['locale']
    timezone = event['dates'].get('timezone', None)
    start_date = event['dates']['start'].get('dateTime', None)
    status = event['dates']['status'].get('code', None)

    public_sales_start = event.get('sales', {}).get('public', {}).get('startDateTime')
    public_sales_end = event.get('sales', {}).get('public', {}).get('endDateTime')
    presales = event.get('sales', {}).get('presales', [])
    presales_name = presales[0].get('name') if presales else None
    presales_start = presales[0].get('startDateTime') if presales else None
    presales_end = presales[0].get('endDateTime') if presales else None
    multiday = 'multiday' if event.get('spanMultipleDays', 0) else ''
    
    venue = event['_embedded']['venues'][0] if '_embedded' in event and 'venues' in event['_embedded'] else None
    venue_name = venue['name'] if venue else None
    city = venue['city']['name'] if venue and 'city' in venue else None
    state = venue['state']['name'] if venue and 'state' in venue else None
    country = venue['country']['name'] if venue and 'country' in venue else None
    address = venue['address']['line1'] if venue and 'address' in venue else None
    longitude = venue['location']['longitude'] if venue and 'location' in venue else None
    latitude = venue['location']['latitude'] if venue and 'location' in venue else None
    classifications = event['classifications'][0] if 'classifications' in event else None
    genre = classifications['genre']['name'] if classifications and 'genre' in classifications else None
    subGenre = classifications['subGenre']['name'] if classifications and 'subGenre' in classifications else None
    segment = classifications['segment']['name'] if classifications and 'segment' in classifications else None
    type = classifications['type']['name'] if classifications and 'type' in classifications else None
    subType = classifications['subType']['name'] if classifications and 'subType' in classifications else None
    family = 'family' if classifications.get('family', 0) else ''
    
    currency = event['priceRanges'][0]['currency'] if 'priceRanges' in event else None
    price_min = event['priceRanges'][0]['min'] if 'priceRanges' in event else None
    price_max = event['priceRanges'][0]['max'] if 'priceRanges' in event else None

    agerestricted = 'ageRestricted' if 'ageRestrictions' in event and event['ageRestrictions']['legalAgeEnforced'] else ''
    attraction_names = []
    for attraction in event['_embedded']['attractions']:
        attraction_names.append(attraction['name'])
    attractions_string = ', '.join(attraction_names)

    # Generate the combined_features field
    combined_features = f"{genre or ' '} {subGenre or ' '} {segment or ' '} {attractions_string or ' '} {type or ' '} {subType or ' '} {city or ' '} {state or ' '} {country or ' '}".strip()

    # Insert into the database
    cursor.execute('''INSERT OR IGNORE INTO events (
                        id, name, url, locale, timezone, start_date, status, public_sales_start, public_sales_end, presales_name, presales_start, 
                        presales_end, multiday, venue_name, city, state, country, address, latitude, longitude, genre, subGenre, segment, type, subType, 
                        currency, price_min, price_max, agerestricted, attractions_string, combined_features) 
                      VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''', 
                    (event_id, name, url, locale, timezone, start_date, status, public_sales_start, public_sales_end, presales_name, presales_start, 
                     presales_end, multiday, venue_name, city, state, country, address, latitude, longitude, genre, subGenre, segment, type, subType, 
                     currency, price_min, price_max, agerestricted, attractions_string, combined_features))

# Commit and close the connection
conn.commit()
conn.close()

print(f"Inserted {len(events_data)} events into the database.")
