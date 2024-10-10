import geocoder
'''This file contains some functions for handling the "near me" detection capability of the recommender system. It uses the geocoder library to get lat/long
coordinates from an IP address and then filters on events within 50km of those coordinates.'''

def get_current_location():
    g = geocoder.ip('me')
    return g.latlng  # Returns a list [latitude, longitude]

from math import radians, cos, sin, sqrt, atan2

def haversine(lat1, lon1, lat2, lon2):
    """
    Calculate the great-circle distance between two points
    on the Earth (specified in decimal degrees).
    """
    # Convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    
    # Haversine formula
    dlat = lat2 - lat1 
    dlon = lon2 - lon1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a)) 
    r = 6371  # Radius of Earth in kilometers. Use 3956 for miles
    return r * c  # Distance in kilometers

def filter_events_by_proximity(events, user_lat, user_lon, max_distance_km=50):
    """
    Filter events based on proximity to the user's location.
    :param events: DataFrame containing events with 'latitude' and 'longitude' fields.
    :param user_lat: User's latitude.
    :param user_lon: User's longitude.
    :param max_distance_km: Maximum distance from the user (in kilometers).
    :return: Filtered DataFrame of events within the specified radius.
    """
    def is_within_radius(row):
        event_lat = row['latitude']
        event_lon = row['longitude']
        distance = haversine(user_lat, user_lon, event_lat, event_lon)
        return distance <= max_distance_km
    
    return events[events.apply(is_within_radius, axis=1)]

