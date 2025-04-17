import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
import re
import requests
import json
import logging
from typing import Dict, List, Any, Optional

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

class FlightAPI:
    
    def __init__(self, api_key: str, base_url: str):
        """
        Initialize the FlightAPI client
        
        Args:
            api_key: Your FlightAPI.io API key
            base_url: The base URL for the API 
        """
        self.api_key = api_key
        self.base_url = base_url
        self.session = requests.Session()
    
    def _handle_response(self, response):
        """Handle API response, raising exceptions for errors"""
        try:
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            logging.error(f"API request failed: {e}")
            try:
                error_details = response.json()
                logging.error(f"Error details: {error_details}")
            except:
                logging.error(f"Status code: {response.status_code}, Response: {response.text}")
            raise
        except json.JSONDecodeError:
            logging.error(f"Failed to parse API response: {response.text}")
            raise

    def search_oneway_flights(
        self, 
        origin: str, 
        destination: str, 
        departure_date: str,
        adults: int = 1,
        children: int = 0,
        infants: int = 0,
        cabin_class: str = "Economy",
        currency: str = "USD",
        region: str = "US"
    ) -> Dict[str, Any]:
        """
        Search for one-way flights using the FlightAPI.io onewaytrip endpoint
        
        Args:
            origin: Departure airport IATA code (e.g., "HEL")
            destination: Arrival airport IATA code (e.g., "OUL")
            departure_date: Date of departure in YYYY-MM-DD format
            adults: Number of adult passengers (default: 1)
            children: Number of children passengers (default: 0)
            infants: Number of infant passengers (default: 0)
            cabin_class: Class of seat (options: "Economy", "Business", "First", "Premium_Economy")
            currency: Currency code (e.g., "USD", "EUR", "INR")
            region: ISO country code to check local prices (e.g., "US")
            
        Returns:
            Dict containing flight search results
        """
        # Validate cabin class
        valid_cabin_classes = ["Economy", "Business", "First", "Premium_Economy"]
        if cabin_class not in valid_cabin_classes:
            raise ValueError(f"Invalid cabin class. Must be one of: {valid_cabin_classes}")
        
        # Construct URL according to API schema
        endpoint = f"{self.base_url}/onewaytrip/{self.api_key}/{origin}/{destination}/{departure_date}/{adults}/{children}/{infants}/{cabin_class}/{currency}"
        
        # Make GET request
        logging.info(f"Searching for one-way flights: {origin} to {destination} on {departure_date}")
        response = self.session.get(endpoint)
        
        return self._handle_response(response)
    
    def parse_flight_results(self, response_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Parse flight search results into a simplified format
        
        Args:
            response_data: The raw API response data
            
        Returns:
            List of simplified flight options
        """
        simplified_results = []
        
        # Check if required data exists
        if not all(key in response_data for key in ["itineraries", "legs", "segments"]):
            logging.error("Invalid response format - missing required data")
            return simplified_results
        
        for itinerary in response_data.get("itineraries", []):
            # Get the pricing options
            pricing_options = itinerary.get("pricing_options", [])
            if not pricing_options:
                continue
                
            # Use the first pricing option
            price_info = pricing_options[0].get("price", {})
            price = price_info.get("amount")
            
            # Get leg information
            leg_ids = itinerary.get("leg_ids", [])
            if not leg_ids:
                continue
                
            leg_id = leg_ids[0]
            leg_info = next((leg for leg in response_data.get("legs", []) if leg.get("id") == leg_id), {})
            
            departure_time = leg_info.get("departure")
            arrival_time = leg_info.get("arrival")
            duration = leg_info.get("duration")
            stops = leg_info.get("stop_count", 0)
            
            # Get carrier information if available
            carrier_ids = leg_info.get("marketing_carrier_ids", [])
            carriers = []
            if "carriers" in response_data:
                carriers = [
                    next((carrier.get("name", str(carrier_id)) for carrier in response_data.get("carriers", []) 
                         if carrier.get("id") == carrier_id), str(carrier_id))
                    for carrier_id in carrier_ids
                ]
            
            simplified_results.append({
                "itinerary_id": itinerary.get("id"),
                "price": price,
                "departure_time": departure_time,
                "arrival_time": arrival_time,
                "duration_minutes": duration,
                "stops": stops,
                "carriers": carriers
            })
        
        return simplified_results


class HotelAPI:
    
    def __init__(self, jwt_token: str, base_url: str):
        """
        Initialize the HotelAPI client
        
        Args:
            jwt_token: JWT authorization token for the API
            base_url: The base URL for the API
        """
        self.jwt_token = jwt_token
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"JWT {jwt_token}",
            "Content-Type": "application/json",
            "Accept": "application/json"
        })
    
    def _handle_response(self, response):
        """Handle API response, raising exceptions for errors"""
        try:
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            logging.error(f"API request failed: {e}")
            try:
                error_details = response.json()
                logging.error(f"Error details: {error_details}")
            except:
                logging.error(f"Status code: {response.status_code}, Response: {response.text}")
            raise
        except json.JSONDecodeError:
            logging.error(f"Failed to parse API response: {response.text}")
            raise

    def search_hotels(self, city: str) -> List[Dict[str, Any]]:
        """
        Search for hotels in a specific city
        
        Args:
            city: Name of the city to search hotels in
            
        Returns:
            List of hotel data with pricing information
        """
        endpoint = f"{self.base_url}/free/{city}"
        
        # Make GET request
        logging.info(f"Searching for hotels in {city}")
        response = self.session.get(endpoint)
        
        return self._handle_response(response)
    
    def parse_hotel_results(self, response_data: List[Any]) -> List[Dict[str, Any]]:
        """
        Parse hotel search results into a simplified format
        
        Args:
            response_data: The raw API response data
            
        Returns:
            List of simplified hotel options with pricing
        """
        simplified_results = []
        
        try:
            for hotel_entry in response_data:
                if len(hotel_entry) < 2:
                    continue
                
                hotel_info = hotel_entry[0]
                price_options = hotel_entry[1]
                
                hotel_name = hotel_info.get("hotelName")
                hotel_id = hotel_info.get("hotelId")
                
                if not hotel_name:
                    continue
                
                # Extract price options
                vendor_prices = []
                
                for price_option in price_options:
                    for i in range(1, 5):  # The API provides up to 4 vendors
                        price_key = f"price{i}"
                        vendor_key = f"vendor{i}"
                        tax_key = f"tax{i}"
                        
                        if price_key in price_option and vendor_key in price_option:
                            price = price_option.get(price_key)
                            vendor = price_option.get(vendor_key)
                            tax = price_option.get(tax_key)
                            
                            # Skip if price or vendor is missing/null
                            if price is None or vendor is None:
                                continue
                                
                            vendor_prices.append({
                                "vendor": vendor,
                                "price": float(price) if price else None,
                                "tax": float(tax) if tax else 0
                            })
                
                # Sort by price (lowest first)
                vendor_prices.sort(key=lambda x: x["price"] if x["price"] is not None else float('inf'))
                
                # Calculate total prices (price + tax)
                for vendor_price in vendor_prices:
                    if vendor_price["price"] is not None:
                        vendor_price["total_price"] = vendor_price["price"] + vendor_price["tax"]
                
                simplified_results.append({
                    "hotel_name": hotel_name,
                    "hotel_id": hotel_id,
                    "price_options": vendor_prices,
                    "lowest_price": vendor_prices[0]["price"] if vendor_prices else None,
                    "lowest_total_price": vendor_prices[0]["total_price"] if vendor_prices else None,
                    "lowest_price_vendor": vendor_prices[0]["vendor"] if vendor_prices else None
                })
            
            # Sort by lowest total price
            simplified_results.sort(key=lambda x: x["lowest_total_price"] if x["lowest_total_price"] is not None else float('inf'))
            
        except Exception as e:
            logging.error(f"Error parsing hotel results: {e}")
        
        return simplified_results
    
    def get_best_hotel_deals(self, city: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Get the best hotel deals for a city
        
        Args:
            city: Name of the city to search hotels in
            limit: Maximum number of results to return (default: 5)
            
        Returns:
            List of best hotel deals, sorted by lowest price
        """
        # Get raw hotel data
        raw_results = self.search_hotels(city)
        
        # Parse results into a more usable format
        parsed_results = self.parse_hotel_results(raw_results)
        
        # Return top N results
        return parsed_results[:limit]


class TravelAssistant:
    def __init__(self, 
                 model_path="./models/travel_llm_model", 
                 use_optimized=False,
                 flight_api_key=None,
                 hotel_jwt_token=None):
        self.use_optimized = use_optimized
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"Loading {'quantized' if use_optimized else 'fine-tuned'} model...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        if use_optimized:
            self.model = AutoModelForCausalLM.from_pretrained(model_path)
            self.model.load_state_dict(torch.load(f"{model_path}/quantized_model.pt"))
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                trust_remote_code=True,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            )
        
        self.model.to(self.device)
        self.model.eval()
        
        # Initialize API clients if credentials are provided
        self.flight_api = FlightAPI(api_key=flight_api_key) if flight_api_key else None
        self.hotel_api = HotelAPI(jwt_token=hotel_jwt_token) if hotel_jwt_token else None
    
    def _detect_flight_search_intent(self, query):
        """Detect if the query is asking for flight information"""
        flight_patterns = [
            r"(?:flights?|airlines?).*?(?:from|to|between)\s+([a-zA-Z\s]+)",
            r"(?:book|search|find|looking for).*?(?:flights?|airlines?)",
            r"(?:travel|flying|fly).*?(?:from|to)\s+([a-zA-Z\s]+)",
            r"(?:airport|iata).*?(?:code|for)\s+([a-zA-Z\s]+)"
        ]
        
        for pattern in flight_patterns:
            if re.search(pattern, query.lower()):
                return True
        return False
    
    def _extract_flight_parameters(self, query):
        """Extract flight search parameters from the query"""
        params = {}
        
        # Extract origin
        origin_patterns = [
            r"from\s+([a-zA-Z\s]+?)(?:\s+to|\s+on|\s+for|\s+in|\s+at|\s+with|\s+\?|$)",
            r"(?:departing|leaving|departure)\s+(?:from\s+)?([a-zA-Z\s]+?)(?:\s+to|\s+on|\s+for|\s+in|\s+at|\s+with|\s+\?|$)"
        ]
        
        for pattern in origin_patterns:
            match = re.search(pattern, query.lower())
            if match:
                params["origin"] = match.group(1).strip()
                break
        
        # Extract destination
        dest_patterns = [
            r"to\s+([a-zA-Z\s]+?)(?:\s+from|\s+on|\s+for|\s+in|\s+at|\s+with|\s+\?|$)",
            r"(?:arriving|arrival)\s+(?:at|in)\s+([a-zA-Z\s]+?)(?:\s+from|\s+on|\s+for|\s+in|\s+at|\s+with|\s+\?|$)"
        ]
        
        for pattern in dest_patterns:
            match = re.search(pattern, query.lower())
            if match:
                params["destination"] = match.group(1).strip()
                break
        
        # Extract date
        date_patterns = [
            r"(?:on|for|date)\s+(\d{1,2}(?:st|nd|rd|th)?\s+(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{4})",
            r"(?:on|for|date)\s+(\d{4}-\d{2}-\d{2})",
            r"(?:on|for|date)\s+(\d{1,2}/\d{1,2}/\d{4})",
            r"(?:on|for|date)\s+(\d{1,2}/\d{1,2}/\d{2})"
        ]
        
        for pattern in date_patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                params["date"] = match.group(1).strip()
                break
        
        # Extract cabin class
        cabin_pattern = r"(?:in|class|cabin)\s+(economy|business|first|premium)"
        match = re.search(cabin_pattern, query.lower())
        if match:
            cabin = match.group(1).strip()
            if cabin == "premium":
                cabin = "Premium_Economy"
            else:
                cabin = cabin.capitalize()
            params["cabin_class"] = cabin
        
        # Extract adults count
        adults_pattern = r"(\d+)\s+(?:adult|adults|people|passengers)"
        match = re.search(adults_pattern, query.lower())
        if match:
            params["adults"] = int(match.group(1))
        
        return params
    
    def _detect_hotel_search_intent(self, query):
        """Detect if the query is asking for hotel information"""
        hotel_patterns = [
            r"(?:hotels?|accommodations?|places to stay|lodging).*?(?:in|at|near)\s+([a-zA-Z\s]+)",
            r"(?:book|search|find|looking for).*?(?:hotels?|accommodations?|rooms?)",
            r"(?:stay|staying).*?(?:in|at)\s+([a-zA-Z\s]+)"
        ]
        
        for pattern in hotel_patterns:
            if re.search(pattern, query.lower()):
                return True
        return False
    
    def _extract_hotel_parameters(self, query):
        """Extract hotel search parameters from the query"""
        params = {}
        
        # Extract city
        city_patterns = [
            r"(?:in|at|near|to)\s+([a-zA-Z\s]+?)(?:\s+on|\s+for|\s+from|\s+with|\s+\?|$)",
            r"(?:hotels?|accommodations?|places to stay|lodging)\s+(?:in|at|near)\s+([a-zA-Z\s]+?)(?:\s+on|\s+for|\s+from|\s+with|\s+\?|$)"
        ]
        
        for pattern in city_patterns:
            match = re.search(pattern, query.lower())
            if match:
                params["city"] = match.group(1).strip()
                break
        
        return params
    
    def search_flights(self, params):
        """Search for flights using the FlightAPI"""
        if not self.flight_api:
            return "Flight API is not configured. Please provide an API key."
        
        required_params = ["origin", "destination", "date"]
        missing_params = [param for param in required_params if param not in params]
        
        if missing_params:
            return f"Missing required flight search parameters: {', '.join(missing_params)}"
        
        try:
            # Convert date to YYYY-MM-DD format if needed
            # (This is a simplified example and would need more robust date parsing)
            date = params["date"]
            if not re.match(r"\d{4}-\d{2}-\d{2}", date):
                # Very basic date conversion - in a real app, use dateutil or similar
                if re.match(r"\d{1,2}/\d{1,2}/\d{4}", date):
                    parts = date.split("/")
                    date = f"{parts[2]}-{parts[0].zfill(2)}-{parts[1].zfill(2)}"
            
            # Map city names to IATA codes (in a real app, use a proper airport database)
            # This is a simplified mapping for demonstration
            city_to_iata = {
                "new york": "JFK",
                "los angeles": "LAX",
                "chicago": "ORD",
                "san francisco": "SFO",
                "miami": "MIA",
                "london": "LHR",
                "paris": "CDG",
                "tokyo": "HND",
                "sydney": "SYD",
                "dubai": "DXB"
            }
            
            origin = params.get("origin").lower()
            destination = params.get("destination").lower()
            
            origin_code = city_to_iata.get(origin, origin.upper() if len(origin) == 3 else None)
            dest_code = city_to_iata.get(destination, destination.upper() if len(destination) == 3 else None)
            
            if not origin_code or not dest_code:
                return f"Could not determine airport codes for {params.get('origin')} and/or {params.get('destination')}"
            
            # Search for flights
            results = self.flight_api.search_oneway_flights(
                origin=origin_code,
                destination=dest_code,
                departure_date=date,
                adults=params.get("adults", 1),
                cabin_class=params.get("cabin_class", "Economy")
            )
            
            # Parse results
            parsed_results = self.flight_api.parse_flight_results(results)
            
            if not parsed_results:
                return f"No flights found from {params.get('origin')} to {params.get('destination')} on {params.get('date')}."
            
            # Format response
            response = f"Found {len(parsed_results)} flights from {origin_code} to {dest_code} on {date}:\n\n"
            
            for i, flight in enumerate(parsed_results[:5], 1):
                carriers = ", ".join(flight.get("carriers", ["Unknown Airline"]))
                departure = flight.get("departure_time", "Unknown")
                arrival = flight.get("arrival_time", "Unknown")
                price = flight.get("price", "Unknown")
                stops = flight.get("stops", 0)
                
                response += f"{i}. {carriers} - ${price}\n"
                response += f"   Departure: {departure}, Arrival: {arrival}\n"
                response += f"   Stops: {stops}, Duration: {flight.get('duration_minutes', 0)} minutes\n\n"
            
            return response
            
        except Exception as e:
            logging.error(f"Error searching flights: {e}")
            return f"Error searching for flights: {str(e)}"
    
    def search_hotels(self, params):
        """Search for hotels using the HotelAPI"""
        if not self.hotel_api:
            return "Hotel API is not configured. Please provide a JWT token."
        
        if "city" not in params:
            return "Missing required parameter: city"
        
        try:
            city = params.get("city")
            
            # Search for hotels
            best_deals = self.hotel_api.get_best_hotel_deals(city, limit=5)
            
            if not best_deals:
                return f"No hotels found in {city}."
            
            # Format response
            response = f"Found {len(best_deals)} hotels in {city}:\n\n"
            
            for i, hotel in enumerate(best_deals, 1):
                name = hotel.get("hotel_name", "Unknown Hotel")
                lowest_price = hotel.get("lowest_total_price", "Unknown")
                vendor = hotel.get("lowest_price_vendor", "Unknown")
                
                response += f"{i}. {name}\n"
                response += f"   Best price: ${lowest_price:.2f} ({vendor})\n"
                
                # Add other vendor options
                other_options = hotel.get("price_options", [])[1:3]  # Get next 2 options
                if other_options:
                    response += "   Other options:\n"
                    for option in other_options:
                        response += f"     ${option.get('total_price', 0):.2f} ({option.get('vendor', 'Unknown')})\n"
                
                response += "\n"
            
            return response
            
        except Exception as e:
            logging.error(f"Error searching hotels: {e}")
            return f"Error searching for hotels: {str(e)}"
    
    def generate_response(self, query, max_length=250, temperature=0.7):
        """Generate a response to the user's travel query"""
        start_time = time.time()
        
        # Detect if the query needs real-time data
        flight_intent = self._detect_flight_search_intent(query)
        hotel_intent = self._detect_hotel_search_intent(query)
        
        api_response = None
        
        # Process flight search intent
        if flight_intent and self.flight_api:
            logging.info("Detected flight search intent")
            flight_params = self._extract_flight_parameters(query)
            logging.info(f"Extracted flight parameters: {flight_params}")
            
            if flight_params:
                api_response = self.search_flights(flight_params)
        
        # Process hotel search intent
        elif hotel_intent and self.hotel_api:
            logging.info("Detected hotel search intent")
            hotel_params = self._extract_hotel_parameters(query)
            logging.info(f"Extracted hotel parameters: {hotel_params}")
            
            if hotel_params:
                api_response = self.search_hotels(hotel_params)
        
        # If we got data from an API, include it in the prompt
        if api_response:
            formatted_query = f"""Instruction: 
The user asked: {query}

I found the following real-time travel data:
{api_response}

Please provide a helpful response to the user's query using this data.
Response:"""
        else:
            # Regular query without API data
            formatted_query = f"Instruction: {query}\nResponse:"
        
        # Generate response from the model
        inputs = self.tokenizer(formatted_query, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=len(inputs["input_ids"][0]) + max_length,
                temperature=temperature,
                top_p=0.9,
                do_sample=True
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        response_time = time.time() - start_time
        
        # Extract only the response part
        response = response.split("Response:")[-1].strip()
        
        return response, response_time


if __name__ == "__main__":
    # Configuration
    use_optimized = False
    model_path = "./models/travel_llm_model" if not use_optimized else "./models/travel_llm_mobile"
    
    flight_api_key = FlightAPI.api_key
    hotel_jwt_token = HotelAPI.api_key
    # Initialize the travel assistant with API integrations
    assistant = TravelAssistant(
        model_path=model_path, 
        use_optimized=use_optimized,
        flight_api_key=flight_api_key,
        hotel_jwt_token=hotel_jwt_token
    )
    
    print("\nTravel Assistant initialized. API integration enabled.")
    if not flight_api_key or flight_api_key == "your_flight_api_key":
        print("Warning: Flight API is not properly configured. Using placeholder key.")
    if not hotel_jwt_token or hotel_jwt_token == "your_hotel_jwt_token":
        print("Warning: Hotel API is not properly configured. Using placeholder token.")
    
    print("\nYou can ask about flights, hotels, or general travel questions.")
    
    while True:
        user_query = input("\nEnter your travel question (or type 'exit' to quit): ")
        if user_query.lower() == 'exit':
            break
        
        response, response_time = assistant.generate_response(user_query)
        print(f"\nResponse: {response}")
        print(f"Generated in {response_time:.2f} seconds")