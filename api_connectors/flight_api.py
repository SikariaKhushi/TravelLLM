import requests
import logging
import json
from typing import Dict, List, Any, Optional

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