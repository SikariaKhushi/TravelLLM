import requests
import logging
import json
from typing import Dict, List, Any, Optional

class HotelAPI:
    
    def __init__(self, jwt_token: str, base_url: str ):
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
            # The API returns a list where each item is a hotel entry
            # Each hotel entry is a list containing:
            # - First element: Dict with hotelName and hotelId
            # - Second element: List of price options from different vendors
            
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
                    # Each vendor has its own set of fields (price1, vendor1, tax1, etc.)
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