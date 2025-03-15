import torch
import numpy as np
from PIL import Image
import ollama

class ImageGrayscaleNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),  # ComfyUI handles images as tensors
            },
            "optional": {
                "intensity": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.1
                })
            },
        }

    RETURN_TYPES = ("IMAGE",)  # Output is also an image
    FUNCTION = "process"
    CATEGORY = "Image Processing"
    TITLE = "Convert to Grayscale"

    def process(self, image: torch.Tensor, intensity=1.0):
        """
        Convert an image tensor to grayscale.
        """

        # Convert PyTorch tensor to NumPy array (C, H, W) -> (H, W, C)
        image_np = (image[0].cpu().numpy() * 255).astype(np.uint8)
        pil_image = Image.fromarray(image_np)

        print(f"Processing image with intensity: {intensity} # This is just an example")
        
        # Convert to grayscale
        grayscale_image = pil_image.convert("L")

        # Convert back to tensor format expected by ComfyUI
        grayscale_tensor = torch.tensor(np.array(grayscale_image) / 255.0).unsqueeze(0)

        return (grayscale_tensor,)


class QueryOllama:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"default": "Hello, world!"}),
            },
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "process"
    CATEGORY = "LLM"
    TITLE = "Query Ollama"

    
    def query_ollama(self, model, prompt, system_message=None, verbose=False, host="localhost", port=11434):
        ollama.api_host = f"http://{host}:{port}"
        messages = [{"role": "user", "content": prompt}]
        if system_message:
            messages.insert(0,{"role": "system", "content": system_message})

        result = ollama.chat(
            model=model,  # Replace with the model you're using
            messages=messages
        )
        response = result['message']['content']
        if verbose: self.pretty_print_prompt(prompt, system_message, response)
        return response

    def pretty_print_prompt(self, prompt, system_message, response):
        print("-------SYSTEM MESSAGE--------")
        print(system_message)
        print("----------PROMPT---------")
        print(prompt)
        print("----------RESPONSE---------")
        print(response)
        print("\n\n")
        print(f"Estimated tokens: {self.estimate_token_length(system_message) + self.estimate_token_length(prompt)}")

    def estimate_token_length(self, text: str) -> int:
        """Estimate the number of tokens in a string."""
        avg_chars_per_token = 4
        return max(1, len(text) // avg_chars_per_token)

    def process(self, prompt: str):
        """
        Query Ollama with a given prompt.
        """
        response = self.query_ollama("gemma3", prompt, verbose=True)

        return (response,)


# Register node in ComfyUI
NODE_CLASS_MAPPINGS = {
    "ImageGrayscaleNode": ImageGrayscaleNode,
    "QueryOllama": QueryOllama,
}
