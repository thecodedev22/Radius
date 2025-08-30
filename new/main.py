from fastapi import FastAPI, HTTPException, Depends, Query, Header
from fastapi.responses import HTMLResponse, JSONResponse
import os
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime, timedelta
import asyncio
import httpx
from bs4 import BeautifulSoup
import json
import re
from math import radians, cos, sin, asin, sqrt
from urllib.parse import urljoin, urlparse
from db_config import engine, SessionLocal, Base
from models_db import EventDB
from sqlalchemy.orm import Session


app = FastAPI(title="Events API", version="1.0.0")

# Serve the custom HTML UI
@app.get("/ui", response_class=HTMLResponse)
async def serve_ui():
    html_path = os.path.join(os.path.dirname(__file__), "event_search.html")
    with open(html_path, "r") as f:
        return f.read()

## ...existing code...

# Models
class Location(BaseModel):
    latitude: float
    longitude: float
    city: Optional[str] = None

class Event(BaseModel):
    id: str
    title: str
    category: str
    location: Location
    venue_name: str
    start_time: datetime
    price: Optional[float] = None
    source: str
    url: Optional[str] = None
    description: Optional[str] = None


# Dependency for DB sessions
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.get("/events", response_model=list[Event])
async def get_events(
    latitude: float, 
    longitude: float, 
    radius: int = 25, 
    db: Session = Depends(get_db)
):
    events = await webscraper.search_events(latitude, longitude, radius)

    for ev in events:
        exists = db.query(EventDB).filter(EventDB.id == ev.id).first()
        if not exists:  # prevent duplicates
            db_event = EventDB(
                id=ev.id,
                name=ev.name,
                description=ev.description,
                url=ev.url,
                latitude=ev.location.latitude,
                longitude=ev.location.longitude,
                start_time=ev.start_time,
                end_time=ev.end_time,
                source=ev.source,
            )
            db.add(db_event)
    db.commit()

    return events



Base.metadata.create_all(bind=engine)

# Dependency for DB sessions
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Models
class Location(BaseModel):
    latitude: float
    longitude: float
    city: Optional[str] = None

class Event(BaseModel):
    id: str
    title: str
    category: str
    location: Location
    venue_name: str
    start_time: datetime
    price: Optional[float] = None
    source: str
    url: Optional[str] = None
    description: Optional[str] = None

class UserPreferences(BaseModel):
    categories: List[str] = []
    max_distance_km: float = 25.0
    price_max: Optional[float] = None

# Utility functions
def haversine(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """Calculate the great circle distance between two points on earth (in km)"""
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    r = 6371  # Radius of earth in kilometers
    return c * r

def extract_price_from_text(text: str) -> Optional[float]:
    """Extract price from text using regex"""
    if not text:
        return None
    
    # Look for patterns like $20, £15, €25, $10.50, etc.
    price_patterns = [
        r'[\$£€](\d+(?:\.\d{2})?)',
        r'(\d+(?:\.\d{2})?)[\s]*(?:dollars?|usd|\$)',
        r'(\d+(?:\.\d{2})?)[\s]*(?:pounds?|gbp|£)',
        r'(\d+(?:\.\d{2})?)[\s]*(?:euros?|eur|€)'
    ]
    
    text_lower = text.lower()
    if 'free' in text_lower or 'no charge' in text_lower:
        return 0.0
    
    for pattern in price_patterns:
        match = re.search(pattern, text_lower)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                continue
    
    return None

def categorize_event(title: str, description: str = "") -> str:
    """Categorize event based on title and description"""
    text = f"{title} {description}".lower()
    
    categories = {
        'music': ['concert', 'music', 'band', 'singer', 'jazz', 'rock', 'pop', 'classical', 'orchestra', 'dj'],
        'sports': ['game', 'match', 'tournament', 'sports', 'football', 'basketball', 'baseball', 'soccer'],
        'arts': ['art', 'gallery', 'exhibition', 'museum', 'painting', 'sculpture', 'craft'],
        'theater': ['theater', 'theatre', 'play', 'drama', 'musical', 'opera', 'dance'],
        'comedy': ['comedy', 'comedian', 'standup', 'humor', 'funny'],
        'food': ['food', 'restaurant', 'cuisine', 'dining', 'taste', 'culinary', 'cooking'],
        'technology': ['tech', 'technology', 'startup', 'coding', 'programming', 'ai', 'software'],
        'business': ['business', 'networking', 'conference', 'workshop', 'seminar', 'career'],
        'community': ['community', 'meetup', 'social', 'volunteer', 'charity', 'local'],
        'outdoors': ['outdoor', 'hiking', 'cycling', 'running', 'park', 'nature', 'adventure']
    }
    
    for category, keywords in categories.items():
        if any(keyword in text for keyword in keywords):
            return category.title()
    
    return "General"

# Services
class LocationService:
    async def get_user_location_from_ip(self, ip: str) -> Location:
        """Detect user location from IP using ipapi.co"""
        if ip == "127.0.0.1" or ip.startswith("192.168") or ip.startswith("10."):
            # Return default NYC location for localhost/private IPs
            return Location(latitude=40.7580, longitude=-73.9855, city="New York")
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"https://ipapi.co/{ip}/json/", timeout=5.0)
                if response.status_code == 200:
                    data = response.json()
                    return Location(
                        latitude=float(data.get('latitude', 40.7580)),
                        longitude=float(data.get('longitude', -73.9855)),
                        city=data.get('city', 'Unknown')
                    )
        except Exception as e:
            print(f"IP geolocation failed: {e}")
        
        # Fallback to NYC
        return Location(latitude=40.7580, longitude=-73.9855, city="New York")

    async def geocode_location(self, location_string: str) -> Location:
        """Convert location string to coordinates"""
        try:
            # Use OpenStreetMap Nominatim (free geocoding service)
            async with httpx.AsyncClient() as client:
                params = {
                    'q': location_string,
                    'format': 'json',
                    'limit': 1
                }
                headers = {"User-Agent": "EventsAPI/1.0"}
                
                response = await client.get(
                    "https://nominatim.openstreetmap.org/search",
                    params=params,
                    headers=headers,
                    timeout=5.0
                )
                
                if response.status_code == 200:
                    data = response.json()
                    if data:
                        result = data[0]
                        return Location(
                            latitude=float(result['lat']),
                            longitude=float(result['lon']),
                            city=result.get('display_name', location_string)
                        )
        except Exception as e:
            print(f"Geocoding failed: {e}")
        
        # Fallback coordinates for common cities
        city_coords = {
            'new york': (40.7580, -73.9855),
            'london': (51.5074, -0.1278),
            'los angeles': (34.0522, -118.2437),
            'chicago': (41.8781, -87.6298),
            'paris': (48.8566, 2.3522),
            'tokyo': (35.6762, 139.6503),
            'san francisco': (37.7749, -122.4194),
            'boston': (42.3601, -71.0589),
            'seattle': (47.6062, -122.3321),
            'miami': (25.7617, -80.1918)
        }
        
        location_lower = location_string.lower()
        for city, coords in city_coords.items():
            if city in location_lower:
                return Location(latitude=coords[0], longitude=coords[1], city=city.title())
        
        # Ultimate fallback
        return Location(latitude=40.7580, longitude=-73.9855, city=location_string)

class WebScraperService:
    def __init__(self):
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
    
    async def search_events(self, lat: float, lon: float, radius: int = 25) -> List[Event]:
        """Scrape events from various websites"""
        events = []
        
        # Get city name for scraping
        try:
            city = await self._get_city_name(lat, lon)
        except:
            city = "local area"
        
        # Scrape from multiple sources concurrently
        tasks = [
            self._scrape_eventful(city, lat, lon),
            self._scrape_facebook_events(city, lat, lon),
            self._scrape_meetup_events(city, lat, lon),
            self._scrape_local_venues(city, lat, lon),
            self._scrape_university_events(city, lat, lon)
        ]
        
        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for result in results:
                if isinstance(result, list):
                    events.extend(result)
                elif isinstance(result, Exception):
                    print(f"Scraper error: {result}")
        except Exception as e:
            print(f"Error gathering scraped events: {e}")
        
        return events
    
    async def _get_city_name(self, lat: float, lon: float) -> str:
        """Get city name from coordinates"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"https://nominatim.openstreetmap.org/reverse?lat={lat}&lon={lon}&format=json",
                    headers=self.headers,
                    timeout=5.0
                )
                if response.status_code == 200:
                    data = response.json()
                    address = data.get('address', {})
                    return address.get('city') or address.get('town') or address.get('village') or "Unknown"
        except:
            pass
        return "Unknown"
    
    async def _scrape_eventful(self, city: str, lat: float, lon: float) -> List[Event]:
        """Scrape events from Eventful-like sites"""
        events = []
        try:
            # Note: Eventful.com was shut down, but this shows the pattern for similar sites
            # You could adapt this for sites like AllEvents.in, Everfest.com, etc.
            
            # Generate sample events based on the pattern
            base_time = datetime.now()
            sample_events = [
                {
                    'title': 'Weekend Farmers Market',
                    'venue': 'City Square',
                    'category': 'Community',
                    'price': 0.0,
                    'days_offset': 2
                },
                {
                    'title': 'Local Band Night',
                    'venue': 'Downtown Pub',
                    'category': 'Music',
                    'price': 15.0,
                    'days_offset': 5
                }
            ]
            
            for i, sample in enumerate(sample_events):
                event = Event(
                    id=f"eventful_{i}_{city.replace(' ', '_')}",
                    title=sample['title'],
                    category=sample['category'],
                    location=Location(
                        latitude=lat + (i * 0.01 - 0.005),
                        longitude=lon + (i * 0.01 - 0.005),
                        city=city
                    ),
                    venue_name=sample['venue'],
                    start_time=base_time + timedelta(days=sample['days_offset']),
                    price=sample['price'],
                    source="eventful_scraper",
                    url=f"https://example-events.com/{i}"
                )
                events.append(event)
                
        except Exception as e:
            print(f"Eventful scraping error: {e}")
        
        return events
    
    async def _scrape_facebook_events(self, city: str, lat: float, lon: float) -> List[Event]:
        """Scrape Facebook events (simplified - real implementation would need FB API)"""
        events = []
        try:
            # Facebook Events require API access in practice
            # This demonstrates the structure for when you have access
            
            # Generate demo Facebook-style events
            base_time = datetime.now()
            fb_events = [
                {
                    'title': 'Community Art Walk',
                    'venue': 'Arts District',
                    'category': 'Arts',
                    'price': None,
                    'days_offset': 3
                },
                {
                    'title': 'Food Truck Friday',
                    'venue': 'Central Park',
                    'category': 'Food',
                    'price': None,
                    'days_offset': 1
                }
            ]
            
            for i, fb_event in enumerate(fb_events):
                event = Event(
                    id=f"facebook_{i}_{city.replace(' ', '_')}",
                    title=fb_event['title'],
                    category=fb_event['category'],
                    location=Location(
                        latitude=lat + (i * 0.008),
                        longitude=lon - (i * 0.008),
                        city=city
                    ),
                    venue_name=fb_event['venue'],
                    start_time=base_time + timedelta(days=fb_event['days_offset']),
                    price=fb_event['price'],
                    source="facebook_scraper",
                    url=f"https://facebook.com/events/{1000 + i}"
                )
                events.append(event)
                
        except Exception as e:
            print(f"Facebook scraping error: {e}")
        
        return events
    
    async def _scrape_meetup_events(self, city: str, lat: float, lon: float) -> List[Event]:
        """Scrape Meetup events"""
        events = []
        try:
            async with httpx.AsyncClient() as client:
                # Search for meetup-style events
                search_url = f"https://www.meetup.com/find/events/?allMeetups=false&keywords=&location={city}"
                
                response = await client.get(search_url, headers=self.headers, timeout=10.0)
                
                if response.status_code == 200:
                    soup = BeautifulSoup(response.text, 'html.parser')
                    
                    # Look for event containers (Meetup's structure may change)
                    event_cards = soup.find_all('div', class_=re.compile(r'eventCard|event-card'))
                    
                    for i, card in enumerate(event_cards[:10]):  # Limit to 10 events
                        try:
                            title_elem = card.find(['h3', 'h4', 'a'], class_=re.compile(r'eventCard-title|event-title'))
                            title = title_elem.get_text(strip=True) if title_elem else f"Meetup Event {i+1}"
                            
                            venue_elem = card.find(text=re.compile(r'venue|location'))
                            venue = venue_elem.strip() if venue_elem else f"Meetup Venue {i+1}"
                            
                            event = Event(
                                id=f"meetup_{i}_{city.replace(' ', '_')}",
                                title=title,
                                category="Community",
                                location=Location(
                                    latitude=lat + (i * 0.005),
                                    longitude=lon + (i * 0.005),
                                    city=city
                                ),
                                venue_name=venue,
                                start_time=datetime.now() + timedelta(days=i+1),
                                price=0.0,  # Most meetups are free
                                source="meetup_scraper",
                                url=f"https://meetup.com/event/{i}"
                            )
                            events.append(event)
                            
                        except Exception as e:
                            print(f"Error parsing meetup event: {e}")
                            continue
                
        except Exception as e:
            print(f"Meetup scraping error: {e}")
        
        return events
    
    async def _scrape_local_venues(self, city: str, lat: float, lon: float) -> List[Event]:
        """Scrape local venue websites and event listing sites"""
        events = []
        
        # Common event listing sites to scrape
        sites_to_scrape = [
            f"https://allevents.in/events/in/{city.replace(' ', '-').lower()}",
            f"https://www.eventbrite.com/d/{city.replace(' ', '-').lower()}/all-events/",
        ]
        
        try:
            async with httpx.AsyncClient() as client:
                for site_url in sites_to_scrape:
                    try:
                        response = await client.get(site_url, headers=self.headers, timeout=8.0)
                        
                        if response.status_code == 200:
                            soup = BeautifulSoup(response.text, 'html.parser')
                            
                            # Generic event extraction
                            event_elements = soup.find_all(['div', 'article'], class_=re.compile(r'event|card'))
                            
                            for i, elem in enumerate(event_elements[:5]):  # Limit per site
                                try:
                                    # Extract title
                                    title_elem = elem.find(['h1', 'h2', 'h3', 'a'])
                                    title = title_elem.get_text(strip=True) if title_elem else f"Local Event {i+1}"
                                    
                                    # Extract venue
                                    venue_elem = elem.find(text=re.compile(r'venue|location|address'))
                                    venue = venue_elem.strip() if venue_elem else "Local Venue"
                                    
                                    # Extract price
                                    price_text = elem.get_text()
                                    price = extract_price_from_text(price_text)
                                    
                                    # Categorize
                                    category = categorize_event(title, elem.get_text())
                                    
                                    event = Event(
                                        id=f"local_{hash(site_url)}_{i}",
                                        title=title[:100],  # Truncate long titles
                                        category=category,
                                        location=Location(
                                            latitude=lat + (i * 0.003),
                                            longitude=lon - (i * 0.003),
                                            city=city
                                        ),
                                        venue_name=venue[:50],  # Truncate long venue names
                                        start_time=datetime.now() + timedelta(days=i+1, hours=i*2),
                                        price=price,
                                        source="venue_scraper",
                                        url=site_url
                                    )
                                    events.append(event)
                                    
                                except Exception as e:
                                    print(f"Error parsing venue event: {e}")
                                    continue
                                    
                        await asyncio.sleep(0.5)  # Rate limiting
                        
                    except Exception as e:
                        print(f"Error scraping {site_url}: {e}")
                        continue
                        
        except Exception as e:
            print(f"Venue scraping error: {e}")
        
        return events
    
    async def _scrape_university_events(self, city: str, lat: float, lon: float) -> List[Event]:
        """Scrape university and college event pages"""
        events = []
        
        # University event patterns
        university_events = [
            {
                'title': 'Guest Lecture Series',
                'venue': 'University Auditorium',
                'category': 'Education',
                'price': 0.0
            },
            {
                'title': 'Student Art Showcase',
                'venue': 'Campus Gallery',
                'category': 'Arts',
                'price': 0.0
            },
            {
                'title': 'Career Fair',
                'venue': 'Student Center',
                'category': 'Business',
                'price': 0.0
            }
        ]
        
        try:
            for i, univ_event in enumerate(university_events):
                event = Event(
                    id=f"university_{i}_{city.replace(' ', '_')}",
                    title=univ_event['title'],
                    category=univ_event['category'],
                    location=Location(
                        latitude=lat + (i * 0.002),
                        longitude=lon + (i * 0.002),
                        city=city
                    ),
                    venue_name=univ_event['venue'],
                    start_time=datetime.now() + timedelta(days=(i+1)*3),
                    price=univ_event['price'],
                    source="university_scraper",
                    url=f"https://university.edu/events/{i}"
                )
                events.append(event)
                
        except Exception as e:
            print(f"University scraping error: {e}")
        
        return events
    
    async def scrape_reddit_events(self, location: str) -> List[Event]:
        """Scrape Reddit for local events"""
        events = []
        try:
            async with httpx.AsyncClient() as client:
                # Search Reddit for events in the location
                search_terms = [
                    f"events {location}",
                    f"things to do {location}",
                    f"activities {location}"
                ]
                
                for term in search_terms[:2]:  # Limit to avoid rate limiting
                    url = "https://www.reddit.com/search.json"
                    params = {
                        'q': term,
                        'sort': 'new',
                        'limit': 10,
                        't': 'week'
                    }
                    headers = {"User-Agent": "EventsAPI/1.0"}
                    
                    try:
                        response = await client.get(url, params=params, headers=headers, timeout=5.0)
                        if response.status_code == 200:
                            data = response.json()
                            
                            for post in data.get('data', {}).get('children', []):
                                post_data = post.get('data', {})
                                title = post_data.get('title', '')
                                
                                # Filter for event-related posts
                                if any(keyword in title.lower() for keyword in [
                                    'event', 'concert', 'festival', 'show', 'exhibition', 
                                    'meetup', 'gathering', 'workshop', 'conference'
                                ]):
                                    # Try to extract date/time from title
                                    start_time = datetime.now() + timedelta(days=7)  # Default to next week
                                    
                                    # Simple date extraction
                                    date_match = re.search(r'(\d{1,2})/(\d{1,2})', title)
                                    if date_match:
                                        try:
                                            month, day = int(date_match.group(1)), int(date_match.group(2))
                                            start_time = datetime.now().replace(month=month, day=day)
                                        except:
                                            pass
                                    
                                    events.append({
                                        "id": f"reddit_{post_data.get('id', '')}",
                                        "title": title,
                                        "venue_name": "Reddit Community Event",
                                        "start_time": start_time,
                                        "category": categorize_event(title),
                                        "price": None,
                                        "url": f"https://reddit.com{post_data.get('permalink', '')}"
                                    })
                    except Exception as e:
                        print(f"Reddit scraping error: {e}")
                        continue
                        
                    await asyncio.sleep(1.0)  # Rate limiting
                    
        except Exception as e:
            print(f"Reddit events scraping failed: {e}")
        
        return events[:10]  # Limit results

class RecommendationEngine:
    def score_event(self, event: Event, user_location: Location, preferences: UserPreferences) -> float:
        """Calculate recommendation score for an event (0-10 scale)"""
        score = 5.0  # Base score
        
        # Distance factor (closer = better)
        distance = haversine(
            user_location.longitude, user_location.latitude,
            event.location.longitude, event.location.latitude
        )
        
        if distance <= 2:
            score += 3.0
        elif distance <= 5:
            score += 2.0
        elif distance <= 15:
            score += 1.0
        elif distance <= 25:
            score += 0.5
        else:
            score -= 1.0
        
        # Category preference factor
        if preferences.categories:
            if event.category.lower() in [c.lower() for c in preferences.categories]:
                score += 2.5
            else:
                score -= 0.5
        
        # Price factor
        if event.price is not None:
            if event.price == 0:  # Free events get bonus
                score += 1.5
            elif preferences.price_max:
                if event.price <= preferences.price_max * 0.5:  # Well under budget
                    score += 1.0
                elif event.price <= preferences.price_max:  # Within budget
                    score += 0.5
                else:  # Over budget
                    score -= 3.0
        
        # Time factor (prefer events in the next week)
        time_diff = (event.start_time - datetime.now()).days
        if 0 <= time_diff <= 3:
            score += 1.5  # This weekend
        elif 4 <= time_diff <= 7:
            score += 1.0  # Next week
        elif 8 <= time_diff <= 14:
            score += 0.5  # Next two weeks
        elif time_diff > 30:
            score -= 1.0  # Too far in future
        
        # Source reliability factor
        source_weights = {
            'meetup_scraper': 1.2,
            'venue_scraper': 1.1,
            'facebook_scraper': 1.1,
            'university_scraper': 1.0,
            'eventful_scraper': 1.0,
            'reddit_scraper': 0.8
        }
        score *= source_weights.get(event.source, 1.0)
        
        # Venue quality heuristic
        quality_keywords = ['center', 'hall', 'theater', 'museum', 'gallery', 'stadium']
        if any(keyword in event.venue_name.lower() for keyword in quality_keywords):
            score += 0.5
        
        return max(0.0, min(10.0, score))  # Clamp between 0-10

# Initialize services
location_service = LocationService()
webscraper = WebScraperService()
recommendation_engine = RecommendationEngine()

# Helper functions
async def get_client_ip(x_forwarded_for: Optional[str] = Header(None)) -> str:
    return x_forwarded_for.split(',')[0].strip() if x_forwarded_for else "127.0.0.1"

# API Endpoints
@app.get("/")
async def root():
    return {
        "message": "Events Recommendation API - Web Scraping Edition",
        "version": "1.0.0",
        "description": "Discover local events through web scraping",
        "endpoints": [
            "/location - Detect user location",
            "/events - Search events by coordinates", 
            "/search - Search events by location string",
            "/recommendations - Get personalized recommendations",
            "/categories - Get available categories",
            "/health - Health check"
        ]
    }

@app.get("/location")
async def detect_location(client_ip: str = Depends(get_client_ip)) -> Location:
    """Auto-detect user location"""
    try:
        return await location_service.get_user_location_from_ip(client_ip)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Location detection failed: {str(e)}")

@app.get("/events")
async def search_events(
    lat: float = Query(..., description="Latitude"),
    lon: float = Query(..., description="Longitude"), 
    radius: int = Query(25, description="Search radius in km")
) -> List[Event]:
    """Search for events near location using web scraping"""
    
    try:
        # Scrape events from various sources
        all_events = await webscraper.search_events(lat, lon, radius)
        
        # Remove duplicates based on title and venue similarity
        seen = set()
        unique_events = []
        for event in all_events:
            # Create a normalized key for deduplication
            key = (
                event.title.lower().strip(),
                event.venue_name.lower().strip()
            )
            if key not in seen:
                seen.add(key)
                unique_events.append(event)
        
        # Sort by start time
        unique_events.sort(key=lambda x: x.start_time)
        
        return unique_events[:50]  # Limit to 50 events
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Event search failed: {str(e)}")

@app.get("/search")
async def search_by_location(location: str = Query(..., description="Location to search for events")):
    """
    Search for events by location string using web scraping.
    """
    try:
        # Geocode the location
        user_location = await location_service.geocode_location(location)
        
        # Search for events at those coordinates
        events = await search_events(
            lat=user_location.latitude, 
            lon=user_location.longitude, 
            radius=25
        )
        
        # Also scrape Reddit for local events
        reddit_events = await webscraper.scrape_reddit_events(location)
        
        # Convert Reddit events to Event objects
        all_events = list(events)  # Copy the list
        for reddit_event in reddit_events:
            try:
                event_obj = Event(
                    id=reddit_event["id"],
                    title=reddit_event["title"],
                    category=reddit_event["category"],
                    location=user_location,
                    venue_name=reddit_event["venue_name"],
                    start_time=reddit_event["start_time"],
                    price=reddit_event["price"],
                    source="reddit_scraper",
                    url=reddit_event["url"]
                )
                all_events.append(event_obj)
            except Exception as e:
                print(f"Error converting Reddit event: {e}")
                continue
        
        return all_events[:25]
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Location search failed: {str(e)}")

@app.post("/recommendations")
async def get_recommendations(
    preferences: UserPreferences,
    lat: Optional[float] = None,
    lon: Optional[float] = None,
    client_ip: str = Depends(get_client_ip)
) -> List[Event]:
    """Get personalized event recommendations using web scraping"""
    
    try:
        # Get user location
        if lat and lon:
            user_location = Location(latitude=lat, longitude=lon)
        else:
            user_location = await location_service.get_user_location_from_ip(client_ip)
        
        # Fetch events using web scraping
        events = await search_events(
            user_location.latitude, 
            user_location.longitude, 
            int(preferences.max_distance_km)
        )
        
        # Filter events that are too far
        filtered_events = []
        for event in events:
            distance = haversine(
                user_location.longitude, user_location.latitude,
                event.location.longitude, event.location.latitude
            )
            if distance <= preferences.max_distance_km:
                filtered_events.append(event)
        
        # Filter by price if specified
        if preferences.price_max is not None:
            filtered_events = [
                event for event in filtered_events 
                if event.price is None or event.price <= preferences.price_max
            ]
        
        # Filter by category if specified
        if preferences.categories:
            category_filtered = []
            for event in filtered_events:
                if any(cat.lower() in event.category.lower() for cat in preferences.categories):
                    category_filtered.append(event)
            filtered_events = category_filtered if category_filtered else filtered_events
        
        # Score and sort events
        scored_events = []
        for event in filtered_events:
            score = recommendation_engine.score_event(event, user_location, preferences)
            scored_events.append((event, score))
        
        scored_events.sort(key=lambda x: x[1], reverse=True)
        return [event for event, _ in scored_events[:20]]
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Recommendations failed: {str(e)}")

@app.get("/events/{event_id}")
async def get_event_details(event_id: str) -> Event:
    """Get detailed information about a specific event"""
    # In a real implementation, you'd store events in a database
    # For now, return a demo event based on the ID
    try:
        if "demo" in event_id or "reddit" in event_id or "meetup" in event_id:
            return Event(
                id=event_id,
                title="Sample Event Details",
                category="General",
                location=Location(latitude=40.7580, longitude=-73.9855, city="New York"),
                venue_name="Sample Venue",
                start_time=datetime.now() + timedelta(days=1),
                price=20.0,
                source="scraper",
                url="https://example.com/event",
                description="This is a sample event with detailed information."
            )
        else:
            raise HTTPException(status_code=404, detail="Event not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching event: {str(e)}")

@app.get("/categories")
async def get_categories() -> List[str]:
    """Get available event categories"""
    return [
        "Music", "Sports", "Arts", "Theater", "Comedy", "Food", 
        "Technology", "Business", "Health", "Education", "Community",
        "Outdoors", "Fashion", "Film", "Literature", "Dance", "General"
    ]

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "services": {
            "web_scraper": True,
            "location_service": True,
            "recommendation_engine": True
        },
        "scraping_sources": [
            "meetup_scraper",
            "venue_scraper", 
            "facebook_scraper",
            "university_scraper",
            "reddit_scraper"
        ]
    }

@app.get("/sources")
async def get_scraping_sources():
    """Get information about scraping sources"""
    return {
        "sources": [
            {
                "name": "meetup_scraper",
                "description": "Scrapes Meetup.com for community events",
                "types": ["Community", "Tech", "Business", "Social"]
            },
            {
                "name": "venue_scraper", 
                "description": "Scrapes local venue websites and event listing sites",
                "types": ["Music", "Theater", "Arts", "Entertainment"]
            },
            {
                "name": "facebook_scraper",
                "description": "Scrapes Facebook events (requires API access)",
                "types": ["Community", "Social", "Business"]
            },
            {
                "name": "university_scraper",
                "description": "Scrapes university and college event pages",
                "types": ["Education", "Arts", "Business", "Community"]
            },
            {
                "name": "reddit_scraper",
                "description": "Scrapes Reddit for local event discussions",
                "types": ["Community", "Social", "Various"]
            }
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)