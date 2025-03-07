import json
import random
from datetime import datetime, timedelta

class TravelDatasetGenerator:
    def __init__(self):
        # Common cities and tourist destinations
        self.cities = {
            "Paris": ["Eiffel Tower", "Louvre Museum", "Notre-Dame", "Champs-Élysées", "Montmartre"],
            "London": ["Big Ben", "Tower Bridge", "British Museum", "Buckingham Palace", "London Eye"],
            "New York": ["Times Square", "Central Park", "Statue of Liberty", "Empire State Building", "Broadway"],
            "Tokyo": ["Shibuya Crossing", "Senso-ji Temple", "Tokyo Skytree", "Shinjuku Gyoen", "Meiji Shrine"],
            "Rome": ["Colosseum", "Vatican Museums", "Trevi Fountain", "Roman Forum", "Pantheon"]
        }
        
        # Hotel amenities and types
        self.hotel_amenities = ["free Wi-Fi", "pool", "spa", "gym", "restaurant", "room service", "parking"]
        self.hotel_types = ["luxury", "boutique", "budget-friendly", "business", "resort"]
        
        # Airlines and flight preferences
        self.airlines = ["Delta", "United", "American Airlines", "British Airways", "Air France", "Emirates"]
        self.flight_preferences = ["direct flight", "shortest duration", "lowest price", "specific airline", "flexible dates"]
        
    def generate_dates(self, start_offset=7, range_days=30):
        """Generate random future dates for travel scenarios"""
        start_date = datetime.now() + timedelta(days=start_offset)
        random_days = random.randint(0, range_days)
        travel_date = start_date + timedelta(days=random_days)
        return travel_date.strftime("%Y-%m-%d")
        
    def generate_flight_queries(self, num_samples=1000):
        """Generate flight search and booking related queries"""
        queries = []
        
        templates = [
            {
                "instruction": "Find flights from {origin} to {destination} for {date}",
                "response": "I'll search for flights from {origin} to {destination} for {date}. Here are the best options I found:\n\n1. {airline1}: {time1} - ${price1}\n- Duration: {duration1}\n- {type1}\n\n2. {airline2}: {time2} - ${price2}\n- Duration: {duration2}\n- {type2}\n\nWould you like to know more about any of these options?"
            },
            {
                "instruction": "What's the cheapest flight from {origin} to {destination} next week?",
                "response": "Let me search for the most economical flights from {origin} to {destination} for next week. The best value options are:\n\n1. ${price1} on {airline1}\n- {date1} at {time1}\n- {duration1} duration\n\n2. ${price2} on {airline2}\n- {date2} at {time2}\n- {duration2} duration\n\nWould you like to proceed with booking any of these flights?"
            },
            {
                "instruction": "I need a {preference} flight from {origin} to {destination}",
                "response": "I understand you're looking for a {preference} flight from {origin} to {destination}. Here's what I found:\n\nBest matching option:\n- {airline1} Airlines\n- Departure: {time1}\n- Price: ${price1}\n- {additional_info}\n\nAlternative option:\n- {airline2} Airlines\n- Departure: {time2}\n- Price: ${price2}\n\nShall I provide more details about either of these options?"
            }
        ]
        
        for _ in range(num_samples):
            template = random.choice(templates)
            origin, destination = random.sample(list(self.cities.keys()), 2)
            airline1, airline2 = random.sample(self.airlines, 2)
            preference = random.choice(self.flight_preferences)
            date = self.generate_dates()
            
            # Create a dictionary with all possible parameters
            format_params = {
                "origin": origin,
                "destination": destination,
                "date": date,
                "preference": preference,
                "airline1": airline1,
                "airline2": airline2,
                "time1": f"{random.randint(1, 12):02d}:{random.randint(0, 59):02d}",
                "time2": f"{random.randint(1, 12):02d}:{random.randint(0, 59):02d}",
                "price1": random.randint(200, 1500),
                "price2": random.randint(200, 1500),
                "duration1": f"{random.randint(1, 12)}h {random.randint(0, 59)}m",
                "duration2": f"{random.randint(1, 12)}h {random.randint(0, 59)}m",
                "type1": random.choice(["Direct", "1 stop", "2 stops"]),
                "type2": random.choice(["Direct", "1 stop", "2 stops"]),
                "date1": self.generate_dates(),
                "date2": self.generate_dates(),
                "additional_info": f"Includes {random.choice(self.flight_preferences)}"
            }
            
            instruction = template["instruction"].format(**format_params)
            response = template["response"].format(**format_params)
            
            queries.append({"instruction": instruction, "response": response})
            
        return queries

    def generate_hotel_queries(self, num_samples=1000):
        """Generate hotel search and booking related queries"""
        queries = []
        
        templates = [
            {
                "instruction": "Find a {hotel_type} hotel in {city} near {attraction}",
                "response": "I've found these {hotel_type} hotels near {attraction} in {city}:\n\n1. {hotel1} Hotel\n- {distance1} from {attraction}\n- ${price1} per night\n- Amenities: {amenities1}\n\n2. {hotel2} Hotel\n- {distance2} from {attraction}\n- ${price2} per night\n- Amenities: {amenities2}\n\nWould you like more details about either of these options?"
            },
            {
                "instruction": "Book a hotel in {city} with {amenity}",
                "response": "I've found several hotels in {city} with {amenity}:\n\n1. {hotel1} Hotel - ${price1} per night\n- Rating: {rating1}/5\n- Additional amenities: {amenities1}\n\n2. {hotel2} Hotel - ${price2} per night\n- Rating: {rating2}/5\n- Additional amenities: {amenities2}\n\nWould you like to proceed with booking either of these options?"
            }
        ]
        
        for _ in range(num_samples):
            template = random.choice(templates)
            city = random.choice(list(self.cities.keys()))
            attraction = random.choice(self.cities[city])
            hotel_type = random.choice(self.hotel_types)
            amenity = random.choice(self.hotel_amenities)
            
            format_params = {
                "hotel_type": hotel_type,
                "city": city,
                "attraction": attraction,
                "amenity": amenity,
                "hotel1": f"The {random.choice(['Grand', 'Royal', 'Plaza', 'Central', 'Park'])}",
                "hotel2": f"The {random.choice(['Imperial', 'Metropolitan', 'Continental', 'Riverside', 'Summit'])}",
                "distance1": f"{random.randint(1, 20)} minutes",
                "distance2": f"{random.randint(1, 20)} minutes",
                "price1": random.randint(100, 500),
                "price2": random.randint(100, 500),
                "amenities1": ", ".join(random.sample(self.hotel_amenities, 3)),
                "amenities2": ", ".join(random.sample(self.hotel_amenities, 3)),
                "rating1": round(random.uniform(3.5, 5.0), 1),
                "rating2": round(random.uniform(3.5, 5.0), 1)
            }
            
            instruction = template["instruction"].format(**format_params)
            response = template["response"].format(**format_params)
            
            queries.append({"instruction": instruction, "response": response})
            
        return queries

    def generate_itinerary_queries(self, num_samples=500):
        """Generate itinerary planning related queries"""
        queries = []
        
        templates = [
            {
                "instruction": "Create a {days}-day itinerary for {city}",
                "response": "Here's a {days}-day itinerary for {city}:\n\n{itinerary}\n\nWould you like any modifications to this itinerary?"
            },
            {
                "instruction": "What are the must-visit places in {city} for a {days}-day trip?",
                "response": "For a {days}-day trip to {city}, here are the must-visit places:\n\n{itinerary}\n\nWould you like specific details about any of these attractions?"
            }
        ]
        
        for _ in range(num_samples):
            template = random.choice(templates)
            city = random.choice(list(self.cities.keys()))
            days = random.randint(2, 5)
            
            # Generate daily itinerary
            itinerary_days = []
            attractions = self.cities[city].copy()
            random.shuffle(attractions)
            
            for day in range(days):
                daily_attractions = attractions[:2]
                attractions = attractions[2:] + attractions[:2]  # Rotate attractions
                itinerary_days.append(f"Day {day + 1}:\n- Morning: Visit {daily_attractions[0]}\n- Afternoon: Explore {daily_attractions[1]}\n- Evening: {random.choice(['Local dining', 'Cultural show', 'Night tour', 'Shopping'])}")
            
            format_params = {
                "days": days,
                "city": city,
                "itinerary": "\n\n".join(itinerary_days)
            }
            
            instruction = template["instruction"].format(**format_params)
            response = template["response"].format(**format_params)
            
            queries.append({"instruction": instruction, "response": response})
            
        return queries

    def generate_full_dataset(self, output_file="travel_training_data.json"):
        """Generate complete training dataset"""
        dataset = []
        dataset.extend(self.generate_flight_queries(1000))
        dataset.extend(self.generate_hotel_queries(1000))
        dataset.extend(self.generate_itinerary_queries(500))
        
        # Shuffle the dataset
        random.shuffle(dataset)
        
        # Save to file
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, indent=2, ensure_ascii=False)
            
        print(f"Generated {len(dataset)} training examples")
        return dataset

# Example usage
if __name__ == "__main__":
    generator = TravelDatasetGenerator()
    dataset = generator.generate_full_dataset()
    
    # Print sample entries
    print("\nSample entries from the dataset:")
    for i in range(3):
        print(f"\nExample {i+1}:")
        print(f"Instruction: {dataset[i]['instruction']}")
        print(f"Response: {dataset[i]['response']}")