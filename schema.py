import json
import sqlite3
'''In this file, we define the schema that maps our JSON input into a SQLlite database that we can use to efficiently store data for the webapp.'''

# Load the JSON data
'''[with open(file, "r(ead)/w(rite") as filename]: the standard line for reading in CSV, TXT, or JSON files in Python. Further use of "json.load()" is required
to load our opened JSON file into local memory (so that we can manipulate and access it for mapping)'''
with open('events_data.json', 'r') as json_file:
    events_data = json.load(json_file)

# Connect to SQLite database (or create it)
'''When using SQLlite, the first thing to do is to set a connection to your database. Then, set a user within that connection -- a "cursor". 
The cursor will run all queries on the connected database.'''
conn = sqlite3.connect('events2.db')
cursor = conn.cursor()

# Create the events table with combined_features column
'''Boilerplate schema to dump everything into a SQL table. CREATE TABLE IF NOT EXISTS tablename -- this is the essential line. The DROP TABLE line is in there 
in case I want to restart the data-processing step, which I often do.'''
cursor.execute('''DROP TABLE IF EXISTS events''')
cursor.execute('''CREATE TABLE IF NOT EXISTS events (
                    id TEXT PRIMARY KEY, 
                    name TEXT UNIQUE,  
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
                    combined_features TEXT)''')

# Insert events into the database
'''Very long and messy loop that gets the job done. Production code would need to be optimized to account for all possible cases of missing data.'''
for event in events_data:
    # Extract relevant fields from the event
    event_id = event['id']
    name = event['name']
    
    # Check for duplicate event title (name)
    '''One issue with the Ticketmaster API in this context is that it considers all showings of one event to each be different unique events.
    We only need the first occurence of each event. Since events are returned in order of recency, just checking that we aren't putting any duplicate names
    will do the trick. We could use a hashtable for faster lookup, but this will ensure that we use less memory.'''
    cursor.execute('SELECT name FROM events WHERE name = ?', (name,))
    if cursor.fetchone():
        print(f"Skipping duplicate event: {name}")
        continue  # Skip if the title is already in the database
    
    '''While mapping event properties into variables for insertion, we frequently use get(), as this will prevent runtime errors if we encounter a missing key.'''
    url = event.get('url')
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
    venue_name = venue.get('name') if venue else None
    city = venue['city']['name'] if venue and 'city' in venue else None
    state = venue['state']['name'] if venue and 'state' in venue else None
    country = venue['country']['name'] if venue and 'country' in venue else None
    address = venue['address'].get('line1') if venue and 'address' in venue else None
    longitude = venue['location']['longitude'] if venue and 'location' in venue else None
    latitude = venue['location']['latitude'] if venue and 'location' in venue else None
    classifications = event['classifications'][0] if 'classifications' in event else {}
    genre = classifications['genre']['name'] if classifications and 'genre' in classifications else None
    subGenre = classifications['subGenre']['name'] if classifications and 'subGenre' in classifications else None
    segment = classifications['segment']['name'] if classifications and 'segment' in classifications else None
    type = classifications['type']['name'] if classifications and 'type' in classifications else None
    subType = classifications['subType']['name'] if classifications and 'subType' in classifications else None
    family = 'family' if classifications.get('family', 0) else ''
    
    currency = event['priceRanges'][0].get('currency') if 'priceRanges' in event else None
    price_min = event['priceRanges'][0].get('min') if 'priceRanges' in event else None
    price_max = event['priceRanges'][0].get('max') if 'priceRanges' in event else None

    agerestricted = 'ageRestricted' if 'ageRestrictions' in event and event['ageRestrictions']['legalAgeEnforced'] else ''
    attraction_names = []
    for attraction in event['_embedded'].get('attractions',''):
        attraction_names.append(attraction['name'])
    attractions_string = ', '.join(attraction_names)

    # Generate the combined_features field
    '''This is the "bag of words" that we will eventually use within our Word2Vec content filtering system.'''
    combined_features = f"{genre or ''} {subGenre or ''} {segment or ''} {type or ''} {attractions_string or ''} {city or ''} {state or ''} {venue_name or ''} {agerestricted or ''} {family or ''} {multiday or ''}".strip()

    # Insert into the database
    '''By using INSERT OR IGNORE INTO, we handle duplicate keys (ids). After defining the columns being inserted into, 
    VALUES (?) allows us to feed in variable inputs previously defined in the loop.'''
    cursor.execute('''INSERT OR IGNORE INTO events (
                        id, name, url, locale, timezone, start_date, status, public_sales_start, public_sales_end, presales_name, presales_start, 
                        presales_end, multiday, venue_name, city, state, country, address, latitude, longitude, genre, subGenre, segment, type, subType, 
                        currency, price_min, price_max, agerestricted, attractions_string, combined_features) 
                      VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''', 
                    (event_id, name, url, locale, timezone, start_date, status, public_sales_start, public_sales_end, presales_name, presales_start, 
                     presales_end, multiday, venue_name, city, state, country, address, latitude, longitude, genre, subGenre, segment, type, subType, 
                     currency, price_min, price_max, agerestricted, attractions_string, combined_features))

# Commit and close the connection
'''Once we're finished, we must use commit() to save our changes. We then close the connection.'''
conn.commit()
conn.close()

print(f"Inserted {len(events_data)} events into the database.")
