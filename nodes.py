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
                "prompt": ("STRING", {"default": "Hello, world!"}),
            },
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "process"
    CATEGORY = "LLM"
    TITLE = "Ollama Query"

    def process(self, model: str, prompt: str):
        """
        Query Ollama with a given prompt.
        """
        response = ollama_query("gemma3:1b", prompt, host="10.0.0.2")

        return (response,)



# Register node in ComfyUI
NODE_CLASS_MAPPINGS = {
    "OllamaQuery": OllamaQuery,
}
