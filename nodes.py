from ollama_query import ollama_query

# def get_ollama_models():
#     models = ollama.list()
#     return [model["model"] for model in models.get("models", [])]

class OllamaQuery:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # "model": (get_ollama_models(),),
                "model": ("STRING", {"default": "gemma3:1b"}),
                "system_message": ("STRING", {"default": "You are a helpful assistant."}),
                "prompt": ("STRING", {"default": "Hello, world!"}),
                "host"  : ("STRING", {"default": "localhost"}),
                "port" : ("INT", {"default": 11434}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    FUNCTION = "process"
    CATEGORY = "LLM"
    TITLE = "Ollama Query"

    def process(self, model: str, prompt: str):
        """
        Query Ollama with a given prompt.
        """
        response, debug_text = ollama_query(model=model, prompt=prompt, system_message=system_message, host=host, port=port)

        return (response, debug_text)

# Register node in ComfyUI
NODE_CLASS_MAPPINGS = {
    "OllamaQuery": OllamaQuery,
}
