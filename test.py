import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time

class TravelAssistant:
    def __init__(self, model_path="./models/travel_llm_model", use_optimized=False):
        self.use_optimized = use_optimized
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"Loading {'quantized' if use_optimized else 'fine-tuned'} model...")

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        if use_optimized:
            self.model = torch.load(f"{model_path}/quantized_model.pt")  # Load full model
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                trust_remote_code=True,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            )

        self.model.to(self.device)
        self.model.eval()

    def generate_response(self, query, max_length=150, temperature=0.7):
        formatted_query = f"Instruction: {query}\nResponse:"
        inputs = self.tokenizer(formatted_query, return_tensors="pt").to(self.device)

        start_time = time.time()
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

        return response.split("Response:")[-1].strip(), response_time

if __name__ == "__main__":
    use_optimized = True
    model_path = "./models/travel_llm_model" if not use_optimized else "./models/travel_llm_mobile"
    
    assistant = TravelAssistant(model_path, use_optimized)
    
    while True:
        user_query = input("\nEnter your travel question (or type 'exit' to quit): ")
        if user_query.lower() == 'exit':
            break
        response, response_time = assistant.generate_response(user_query)
        print(f"\nResponse: {response}")
        print(f"Generated in {response_time:.2f} seconds")
