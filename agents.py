import os
os.environ["STREAMLIT_WATCHER_TYPE"] = "none"


import streamlit as st
import time
import requests
import smtplib
import base64
from datetime import datetime, timedelta
from typing import TypedDict, List, Dict, Any, Optional
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import json
from io import BytesIO
from markdown_pdf import MarkdownPdf, Section
import tempfile


# LangGraph and LangChain imports
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from operator import add

from langchain_google_genai import ChatGoogleGenerativeAI
from typing_extensions import Annotated
from langchain_community.utilities import WikipediaAPIWrapper
from tavily import TavilyClient
import googlemaps

# PDF generation
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch



# Set up page config
st.set_page_config(
    page_title="AI Trip Planner",
    page_icon="✈️",
    layout="wide"
)

# Wikipedia Image Function (Enhanced)
def get_wikipedia_image(query: str, delay: float = 0.5) -> Optional[str]:
    """Fetch image URL from Tavily (replacing Wikipedia)"""
    time.sleep(delay)
    
    url = "https://api.tavily.com/search"
    headers = {
        "Authorization": f"Bearer {st.secrets.get('TAVILY_API_KEY', '')}",  # Use your Tavily API key
        "Content-Type": "application/json"
    }
    
    payload = {
        "query": query,
        "include_images": True,
        "limit": 1  # get top result
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=5)
        response.raise_for_status()
        data = response.json()
        
        if "images" in data and len(data["images"]) > 0:
            # Return the first image URL
            return data["images"][0].get("url")
    except Exception as e:
        print(f"Tavily image search error for '{query}': {e}")
    
    return None

# Initialize APIs
@st.cache_resource
def initialize_apis():
    """Initialize all API clients"""
    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=st.secrets.get("GOOGLE_API_KEY", ""),
            temperature=0.7
        )
        
        tavily_client = TavilyClient(api_key=st.secrets.get("TAVILY_API_KEY", ""))
        wikipedia = WikipediaAPIWrapper()
        
        return llm, tavily_client, wikipedia
    except Exception as e:
        st.error(f"Error initializing APIs: {str(e)}")
        return None, None, None

# Enhanced state structure with Annotated fields for parallel updates
class TripPlannerState(TypedDict):
    destination: str
    start_date: str
    end_date: str
    budget: int
    num_travelers: int
    interests: List[str]
    
    # Agent outputs with Annotated for parallel updates
    destination_info: Dict[str, Any]
    destination_image: Optional[str]
    places_info: List[Dict[str, Any]]
    flight_options: List[Dict[str, Any]]
    hotel_options: List[Dict[str, Any]]
    activities: List[Dict[str, Any]]
    weather_info: Dict[str, Any]
    final_itinerary: str
    
    # Progress tracking - using Annotated for parallel updates
    current_step: Annotated[List[str], add]
    error_messages: Annotated[List[str], add]
    progress: int

# Initialize APIs
llm, tavily_client, wikipedia = initialize_apis()

def research_agent(state: TripPlannerState) -> TripPlannerState:
    """Research destination and get main image"""
    try:
        destination = state["destination"]
        st.write(f"🔍 Researching {destination}...")
        
        # Get Wikipedia information
        wiki_result = wikipedia.run(f"{destination} travel tourism")
        
        # Get destination image
        destination_image = get_wikipedia_image(f"{destination} landmark")
        
        # Extract key info using LLM
        prompt = f"""
        Extract key travel information from this Wikipedia content about {destination}:
        {wiki_result[:2000]}
        
        Return a JSON with:
        - description: 2-sentence overview
        - best_time_to_visit: season/months
        - currency: local currency
        - language: main language(s)
        - timezone: timezone info
        - key_facts: 3 interesting facts as list
        """
        
        response = llm.invoke(prompt)

        content = getattr(response, "content", str(response))

        
        try:
            destination_info = json.loads(response.content.replace("```json", "").replace("```", ""))
        except:
            destination_info = {
                "description": content[:300],
                "raw_info": wiki_result[:500]
            }
        
        return {
            "destination_info": destination_info,
            "destination_image": destination_image,
            "current_step": ["research_complete"],
            "progress": 20,
        }
    except Exception as e:
        return {
            "destination_info": {"error": str(e)},
            "destination_image": None,
            "error_messages": [f"Research error: {str(e)}"],
        }


def places_agent(state: TripPlannerState) -> dict:
    """Agent to fetch top places using Tavily"""
    try:
        destination = state["destination"]
        st.write(f"📍 Finding places for {destination}...")

        places_query = f"top tourist attractions in {destination}"
        places_result = tavily_client.search(
            query=places_query,
            search_depth="basic",
            max_results=6
        )

        places_info = []
        for r in places_result.get("results", [])[:6]:
            place_details = {
                "name": r.get("title", ""),
                "rating": None,   # Tavily doesn’t give ratings directly
                "address": "",
                "types": [],
                "place_id": None,
                "image_url": None
            }
            # Optional: get image from Wikipedia
            place_image = get_wikipedia_image(f"{r.get('title', '')} {destination}")
            place_details["image_url"] = place_image

            places_info.append(place_details)

        return {
            "places_info": places_info,
            "current_step": ["places_complete"],
            "progress": 40
        }

    except Exception as e:
        return {
            "places_info": [],
            "error_messages": [f"Places error: {str(e)}"]
        }


def weather_agent(state: TripPlannerState) -> dict:
    """Agent to fetch weather info (mock for now, later OpenWeatherMap)"""
    try:
        destination = state["destination"]
        st.write(f"⛅ Checking weather for {destination}...")

        # For now, just search with Tavily (acts like knowledge/weather lookup)
        weather_query = f"current weather and climate in {destination}"
        weather_result = tavily_client.search(
            query=weather_query,
            search_depth="basic",
            max_results=3
        )

        # Very rough extraction — can refine with LLM later
        weather_info = {
            "forecast": weather_result.get("results", [{}])[0].get("content", "Weather data unavailable"),
            "temperature": "20-25°C",   # mock default
            "conditions": "Partly cloudy with occasional sunshine"
        }

        return {
            "weather_info": weather_info,
            "current_step": ["weather_complete"],
            "progress": 50
        }

    except Exception as e:
        return {
            "weather_info": {},
            "error_messages": [f"Weather error: {str(e)}"]
        }



def activities_agent(state: TripPlannerState) -> TripPlannerState:
    """Find activities and events using Tavily"""
    try:
        destination = state["destination"]
        interests = state["interests"]
        start_date = state["start_date"]
        
        st.write(f"🎯 Finding activities in {destination}...")
        
        activities = []
        for interest in interests[:3]:
            query = f"{interest} activities events {destination} 2024 things to do experiences"
            results = tavily_client.search(
                query=query,
                search_depth="basic",
                max_results=3
            )
            
            for result in results.get("results", []):
                activity = {
                    "title": result.get("title", ""),
                    "description": result.get("content", "")[:200],
                    "category": interest,
                    "url": result.get("url", ""),
                    "image_url": get_wikipedia_image(f"{interest} {destination}")
                }
                activities.append(activity)
        
        return {
            "activities": activities[:8],
            "current_step": ["activities_complete"],
            "progress": 90,

        }
    except Exception as e:
        return {
            "activities": [],
            "error_messages": [f"Activities error: {str(e)}"],
        }

def itinerary_agent(state: TripPlannerState) -> TripPlannerState:
    """Generate final comprehensive itinerary"""
    try:
        st.write("📝 Generating your personalized itinerary...")
        
        # Calculate trip duration
        start = datetime.strptime(state["start_date"], "%Y-%m-%d")
        end = datetime.strptime(state["end_date"], "%Y-%m-%d")
        duration = (end - start).days
        
        prompt = f"""
        Create a detailed {duration}-day trip itinerary for {state["destination"]}.
        
        TRIP DETAILS:
        - Destination: {state["destination"]}
        - Dates: {state["start_date"]} to {state["end_date"]} ({duration} days)
        - Budget: ${state["budget"]} for {state["num_travelers"]} travelers
        - Interests: {", ".join(state["interests"])}
        
        AVAILABLE DATA:
        
        Destination Info: {json.dumps(state.get("destination_info", {}))}
        
        Top Places: {json.dumps([p.get("name", "") + " (Rating: " + str(p.get("rating", "N/A")) + ")" for p in state.get("places_info", [])[:5]])}
        
        Flight Options: {json.dumps(state.get("flight_options", []))}
        
        Hotel Options: {json.dumps(state.get("hotel_options", []))}
        
        Activities Available: {json.dumps([a.get("title", "") + " - " + a.get("category", "") for a in state.get("activities", [])[:6]])}
        
        Weather: {json.dumps(state.get("weather_info", {}))}
        
        CREATE A COMPREHENSIVE ITINERARY WITH:
        
        ## 🗓️ TRIP OVERVIEW
        - Destination summary
        - Duration and dates
        - Budget breakdown
        
        ## ✈️ FLIGHTS & ARRIVAL
        - Recommended flight options (from the search results)
        - Airport transfer tips
        
        ## 🏨 ACCOMMODATION
        - Top 3 hotel recommendations (from search results with prices)
        - Area recommendations
        
        ## 📅 DAY-BY-DAY ITINERARY
        For each day, include:
        - Morning activity (specific place/attraction)
        - Afternoon activity
        - Evening suggestion
        - Estimated costs
        - Transportation tips
        
        ## 🎯 MUST-DO ACTIVITIES
        - Top attractions (from places found)
        - Experience recommendations
        - Booking tips
        
        ## 💰 BUDGET BREAKDOWN
        - Flights: estimated cost
        - Hotels: per night cost
        - Activities: daily budget
        - Food: daily budget
        - Transport: local transport
        
        ## 📋 PRACTICAL TIPS
        - Local customs
        - Transportation
        - Safety tips
        - Emergency contacts
        
        Make it engaging, practical, and well-formatted with emojis and clear sections.
        """
        
        response = llm.invoke(prompt)
        content = getattr(response, "content", str(response))

        return {
            "final_itinerary": content,
            "current_step": ["complete"],
            "progress": 100,
        }
    except Exception as e:
        return {
            "final_itinerary": f"Error generating itinerary: {str(e)}",
            "error_messages": [f"Itinerary error: {str(e)}"],
        }

# Conditional routing function
def should_continue_planning(state: TripPlannerState) -> str:
    """Decide next step based on research results"""
    destination_info = state.get("destination_info", {})
    
    # Check if we have basic destination info
    if destination_info and not destination_info.get("error"):
        return "continue"
    else:
        return "error"

# PDF Generation Function
def generate_pdf(itinerary_text: str, destination: str, state) -> BytesIO:
    """
    Generate PDF from Markdown text using markdown_pdf library.
    Input:
        itinerary_text: str -> the Markdown itinerary content
        destination: str -> destination name
        state: TripPlannerState or dict (kept for compatibility)
    Returns:
        BytesIO buffer containing the PDF
    """
    pdf_buffer = BytesIO()

    # Create MarkdownPdf object
    pdf = MarkdownPdf(toc_level=0)

    # Optional: add a title section with destination
    title_md = f"# Trip Itinerary: {destination}\n\n"
    
    # Combine title and itinerary text
    full_md = title_md + itinerary_text

    # Add as a section
    pdf.add_section(Section(full_md))

    # Save PDF to BytesIO
    pdf.save(pdf_buffer)
    pdf_buffer.seek(0)

    return pdf_buffer
    



# Email function
def send_email(recipient_email: str, pdf_buffer: BytesIO, destination: str):
    """Send itinerary PDF via email using Gmail SMTP"""
    try:
        # Reset buffer position
        pdf_buffer.seek(0)

        sender_email = st.secrets.get("GMAIL_USER", "")
        sender_password = st.secrets.get("GMAIL_APP_PASSWORD", "")
        
        if not sender_email or not sender_password:
            st.error("Email credentials not configured in secrets!")
            return False
        
        msg = MIMEMultipart()
        msg['From'] = sender_email
        msg['To'] = recipient_email
        msg['Subject'] = f"Your Trip Itinerary for {destination}"
        
        body = f"""
        Hello!
        
        Your personalized trip itinerary for {destination} is ready!
        
        Please find the detailed PDF itinerary attached.
        
        Happy travels!
        
        Best regards,
        AI Trip Planner
        """
        
        msg.attach(MIMEText(body, 'plain'))
        
        # Attach PDF
        pdf_buffer.seek(0)
        part = MIMEBase('application', 'octet-stream')
        part.set_payload(pdf_buffer.read())
        encoders.encode_base64(part)
        part.add_header(
            'Content-Disposition',
            f'attachment; filename= "{destination.replace(" ", "_")}_itinerary.pdf"',
        )
        msg.attach(part)
        
        # Send email
        with smtplib.SMTP('smtp.gmail.com', 587) as server:
            server.starttls()
            server.login(sender_email, sender_password)
            server.sendmail(sender_email, recipient_email, msg.as_string())


        
        return True
        
    except Exception as e:
        st.error(f"Email error: {str(e)}")
        return False

def flights_agent(state: TripPlannerState) -> TripPlannerState:
    """Search flights using Tavily"""
    try:
        destination = state["destination"]
        start_date = state["start_date"]
        budget = state["budget"]
        
        # Search for detailed flight information
        flight_query = f"flights to {destination} from India {start_date} prices airlines booking"
        flight_results = tavily_client.search(
            query=flight_query,
            search_depth="advanced",
            max_results=6
        )
        
        # Extract structured flight data
        flight_prompt = f"""
        From these flight search results for {destination}:
        {json.dumps([r.get('title', '') + ' - ' + r.get('content', '')[:150] for r in flight_results.get('results', [])])}
        
        Extract 3-5 flight options as JSON array:
        [
          {{
            "airline": "airline name",
            "price": "price range or specific price",
            "duration": "flight duration",
            "stops": "direct/1-stop/2-stop",
            "departure": "departure info",
            "booking_tip": "where/how to book"
          }}
        ]
        """
        
        flight_response = llm.invoke(flight_prompt)
        flight_content = getattr(flight_response, "content", str(flight_response))



        try:
            flight_options = json.loads(flight_content.replace("```json", "").replace("```", ""))
            if not isinstance(flight_options, list):
                flight_options = []
        except:
            flight_options = []
        
        # Return only flight-specific updates
        return {
            "flight_options": flight_options,
            "current_step": ["flights_complete"],
        }
    except Exception as e:
        return {
            "flight_options": [],
            "error_messages": [f"Flight search error: {str(e)}"]
        }

def hotels_agent(state: TripPlannerState) -> TripPlannerState:
    """Search hotels using Tavily"""
    try:
        destination = state["destination"]
        budget = state["budget"]
        
        # Search for hotel information
        hotel_query = f"best hotels {destination} accommodation booking prices reviews {budget} budget"
        hotel_results = tavily_client.search(
            query=hotel_query,
            search_depth="advanced",
            max_results=6
        )
        
        # Extract structured hotel data
        prompt = f"""
        From these hotel search results for {destination}:
        {json.dumps([r.get('title', '') + ' - ' + r.get('content', '')[:150] for r in hotel_results.get('results', [])])}
        
        Extract 3-5 hotel options as JSON array:
        [
          {{
            "name": "hotel name",
            "price_per_night": "price per night",
            "rating": "star rating or review score",
            "location": "area/neighborhood",
            "amenities": "key amenities",
            "booking_platform": "where to book"
          }}
        ]
        """
        
        hotel_response = llm.invoke(prompt)
        hotel_content = getattr(hotel_response, "content", str(hotel_response))


       
        try:
            hotel_options = json.loads(hotel_content.replace("```json", "").replace("```", ""))
            if not isinstance(hotel_options, list):
                hotel_options = []
        except:
            hotel_options = []
        
        # Return only hotel-specific updates
        return {
            "hotel_options": hotel_options,
            "current_step": ["hotels_complete"],
        }
        
    except Exception as e:
        return {
            "hotel_options": [],
            "error_messages": [f"Hotel search error: {str(e)}"]
        }

# Create enhanced workflow with proper parallel execution
def create_workflow():
    workflow = StateGraph(TripPlannerState)
    
    # Add nodes
    workflow.add_node("research", research_agent)
    workflow.add_node("places", places_agent)
    workflow.add_node("weather", weather_agent)
    workflow.add_node("flights", flights_agent)
    workflow.add_node("hotels", hotels_agent)
    workflow.add_node("activities", activities_agent)
    workflow.add_node("itinerary", itinerary_agent)

     # Gate node to wait for all parallel branches
    workflow.add_node("sync_gate", lambda s: s) #barrier code


    # Define the flow
    workflow.set_entry_point("research")
    
    # Sequential flow: research -> places
    workflow.add_edge("research", "places")
    
    # Parallel execution: places -> [flights, hotels, weather]
    workflow.add_edge("places", "flights")
    workflow.add_edge("places", "hotels")
    workflow.add_edge("places", "weather")

    # Wait for all parallel agents before activities
    workflow.add_edge("flights", "sync_gate")
    workflow.add_edge("hotels", "sync_gate")
    workflow.add_edge("weather", "sync_gate")


    # Barrier condition: proceed only when ALL parallel branches are done
    def _barrier_condition(state: TripPlannerState) -> str:
        steps = set(state.get("current_step", []))
        required = {"flights_complete", "hotels_complete", "weather_complete"}
        return "go" if required.issubset(steps) else "wait"


    workflow.add_conditional_edges(
        "sync_gate",
        _barrier_condition,
        {
            "go": "activities",   # move forward once all are done
            "wait": END,   # temporary halt until other branch completes
        },
    )

  
    # Final steps
    workflow.add_edge("activities", "itinerary")
    workflow.add_edge("itinerary", END)
    
    return workflow.compile()

# Enhanced Streamlit UI
def main():
    st.title("🌍 AI Multi-Agent Trip Planner")
    st.markdown("Plan your perfect trip with AI-powered research, real-time search, and beautiful itineraries!")
    
    # Sidebar
    with st.sidebar:
        st.header("🔑 Configuration")
        
        if llm and tavily_client:
            st.success("✅ All APIs connected!")
        else:
            st.error("❌ Check API keys in secrets")
            st.code("""
            # Add to .streamlit/secrets.toml:
            GOOGLE_API_KEY = "your_key"
            TAVILY_API_KEY = "your_key"
            GMAIL_USER = "your_gmail@gmail.com"  # Optional for email
            GMAIL_APP_PASSWORD = "your_app_password"  # Optional for email
            """)
        
        st.markdown("---")
        st.markdown("### 🔄 Workflow")
        st.markdown("""
        1. **Research** destination info
        2. **Places & Weather** (parallel)
        3. **Flights & Hotels** (parallel)
        4. **Activities** based on interests
        5. **Generate** final itinerary
        """)
        st.markdown("---")
        if st.button("🗑️ Clear Trip Cache", help="Clear cached results to force new planning"):
            # Clear trip-related session state
            keys_to_clear = [k for k in st.session_state.keys() if k.startswith(('trip_', 'email_sent_', 'pdf_buffer'))]
            for key in keys_to_clear:
                del st.session_state[key]
            st.rerun()

    
    # Main form
    with st.form("enhanced_trip_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            destination = st.text_input("🏙️ Destination", placeholder="e.g., Tokyo, Japan")
            start_date = st.date_input("📅 Start Date", datetime.now() + timedelta(days=7))
            budget = st.number_input("💰 Budget (USD)", min_value=500, max_value=50000, value=3000, step=100)
        
        with col2:
            end_date = st.date_input("📅 End Date", datetime.now() + timedelta(days=14))
            num_travelers = st.number_input("👥 Travelers", min_value=1, max_value=10, value=2)
            interests = st.multiselect(
                "🎯 Your Interests",
                ["Culture", "Food", "Adventure", "Museums", "Nature", "Shopping", "Nightlife", "History", "Art", "Music", "Architecture", "Local Experiences"],
                default=["Culture", "Food", "Local Experiences"]
            )
        
        
        submitted = st.form_submit_button("🚀 Plan My Amazing Trip!", use_container_width=True)
    
    trip_key = None
    if destination:
        trip_key = f"trip_{destination}_{start_date}_{end_date}_{budget}_{num_travelers}"


    if submitted and destination and llm:
        # Validation
        if start_date >= end_date:
            st.error("End date must be after start date!")
            return
        
        if not interests:
            st.error("Please select at least one interest!")
            return
        
        
        # Add this check before running the workflow:
        if trip_key and trip_key in st.session_state:
            # Use cached results
            final_state = st.session_state[trip_key]
            st.info("📋 Using cached trip plan (change inputs to regenerate)")
        else:
            # Run workflow and cache results
            with st.spinner("Planning your trip..."):
                # Initialize state with proper list format for Annotated fields
                initial_state = TripPlannerState(
                    destination=destination,
                    start_date=start_date.strftime("%Y-%m-%d"),
                    end_date=end_date.strftime("%Y-%m-%d"),
                    budget=budget,
                    num_travelers=num_travelers,
                    interests=interests,
                    destination_info={},
                    destination_image=None,
                    places_info=[],
                    flight_options=[],
                    hotel_options=[],
                    activities=[],
                    weather_info={},
                    final_itinerary="",
                    current_step=["starting"],  # Initialize as list
                    error_messages=[],  # Initialize as empty list
                    progress=0
                )

            
                
                # Progress tracking
                progress_bar = st.progress(0)
                status_container = st.container()
                
                try:
                    # Create and run workflow
                    app = create_workflow()
                    
                    # Run with progress updates
                    final_state = app.invoke(initial_state)
                    
                    # Cache the results
                    if trip_key:
                        st.session_state[trip_key] = final_state
                    
                    progress_bar.progress(100)
                
                except Exception as e:
                    st.error(f"❌ Workflow execution error: {str(e)}")
                    st.info("Please check your API keys and internet connection.")
                    return

        if trip_key and trip_key in st.session_state:
            final_state = st.session_state[trip_key]
            
            # Get current form values for email functionality
            form_data = st.session_state.get('form_data', {})

            # Display results
            if final_state.get("final_itinerary"):
                # Show destination image if available
                if final_state.get("destination_image"):
                    st.image(final_state["destination_image"], caption=f"{destination}", use_column_width=True)
                
                st.markdown("## 📋 Your Complete Trip Itinerary")
                st.markdown(final_state["final_itinerary"])
            
                # Detailed sections
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    with st.expander("🔍 Destination Details"):
                        dest_info = final_state.get("destination_info", {})
                        if dest_info:
                            st.json(dest_info)
                
                with col2:
                    with st.expander("📍 Top Places & Attractions"):
                        places = final_state.get("places_info", [])
                        for i, place in enumerate(places[:5], 1):
                            st.write(f"**{i}. {place.get('name', 'Unknown')}**")
                            st.write(f"⭐ Rating: {place.get('rating', 'N/A')}")
                            st.write(f"📍 {place.get('address', 'N/A')}")
                            if place.get('image_url'):
                                st.image(place['image_url'], width=200)
                            st.write("---")
                
                with col3:
                    with st.expander("✈️ Flight Options Found"):
                        flights = final_state.get("flight_options", [])
                        if flights:
                            for i, flight in enumerate(flights, 1):
                                st.write(f"**Option {i}:**")
                                st.write(f"✈️ Airline: {flight.get('airline', 'N/A')}")
                                st.write(f"💰 Price: {flight.get('price', 'N/A')}")
                                st.write(f"⏱️ Duration: {flight.get('duration', 'N/A')}")
                                st.write(f"🔄 Stops: {flight.get('stops', 'N/A')}")
                                st.write("---")
                        else:
                            st.info("No specific flight options extracted. Check the main itinerary for flight guidance.")
                
                # Hotel section
                st.markdown("### 🏨 Hotel Recommendations")
                hotels = final_state.get("hotel_options", [])
                if hotels:
                    hotel_cols = st.columns(min(len(hotels), 3))
                    for i, hotel in enumerate(hotels[:3]):
                        with hotel_cols[i]:
                            st.write(f"**{hotel.get('name', 'Hotel')}**")
                            st.write(f"💰 {hotel.get('price_per_night', 'Price N/A')}")
                            st.write(f"⭐ {hotel.get('rating', 'Rating N/A')}")
                            st.write(f"📍 {hotel.get('location', 'Location N/A')}")
                            st.write(f"🛎️ {hotel.get('amenities', 'Amenities N/A')}")
                else:
                    st.info("Hotel recommendations are included in the main itinerary above.")
                
                # Activities section
                st.markdown("### 🎯 Recommended Activities")
                activities = final_state.get("activities", [])
                if activities:
                    activity_cols = st.columns(2)
                    for i, activity in enumerate(activities[:6]):
                        with activity_cols[i % 2]:
                            st.write(f"**{activity.get('title', 'Activity')}**")
                            st.write(f"🎯 Category: {activity.get('category', 'N/A')}")
                            st.write(f"📝 {activity.get('description', 'No description')}")
                            if activity.get('image_url'):
                                st.image(activity['image_url'], width=150)
                            st.write("---")
                # Generate PDF buffer once

                
                # Initialize session_state flags
                show_email_key = f"show_email_{trip_key}"
                email_key = f"email_sent_{trip_key}"

                if show_email_key not in st.session_state:
                    st.session_state[show_email_key] = False
                if email_key not in st.session_state:
                    st.session_state[email_key] = False

                if f"pdf_buffer_{trip_key}" not in st.session_state:
                    st.session_state[f"pdf_buffer_{trip_key}"] = generate_pdf(
                        final_state["final_itinerary"], 
                        destination, 
                        final_state
                    )
                
            
                pdf_buffer = st.session_state[f"pdf_buffer_{trip_key}"]
                
                
                # Download and Email options
                st.markdown("### 📥 Get Your Itinerary")
                
                col_download, col_email = st.columns(2)
                
                with col_download:
                    # Generate and offer PDF download
                    
                    st.download_button(
                        label="📄 Download PDF Itinerary",
                        data=pdf_buffer.getvalue(),
                        file_name=f"{destination.replace(' ', '_')}_itinerary.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )

                                  
                            
                    # Text download fallback
                    st.download_button(
                        label="📝 Download Text Version",
                        data=final_state["final_itinerary"],
                        file_name=f"{destination.replace(' ', '_')}_itinerary.txt",
                        mime="text/plain"
                    )
                    
                    
            
        
    elif submitted and not llm:
        st.error("❌ Please configure your API keys first!")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: gray;'>
    🤖 Powered by LangGraph Multi-Agent System | Built with Streamlit<br>
    Uses: Gemini LLM • Tavily Search • Google Places • Wikipedia
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()