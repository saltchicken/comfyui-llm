from ollama_query import ollama_query
import rag_milvus

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
                "system_message": ("STRING", {"default": ""}),
                "prompt": ("STRING", {"default": "Hello, world!"}),
                "host"  : ("STRING", {"default": "localhost"}),
                "port" : ("STRING", {"default": 11434}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    FUNCTION = "process"
    CATEGORY = "LLM"
    TITLE = "Ollama Query"

    def process(self, model: str, system_message: any, prompt: str, host: str, port: str):
        """
        Query Ollama with a given prompt.
        """
        if system_message == "":
            system_message = None

        response, debug_text = ollama_query(model=model, prompt=prompt, system_message=system_message, host=host, port=port)

        return (response, debug_text)

class MilvusQuery:
    def __init__(self):
        super().__init__()
        self.rag = rag_milvus.Rag()

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "query": ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "process"
    CATEGORY = "LLM"
    TITLE = "Milvus Query"

    def process(self, query: str):
        """
        Query Milvus with a given prompt.
        """

        knowledge = self.rag.search_knowledge(query, 0.5)
        return (repr(knowledge),)

# Register node in ComfyUI
NODE_CLASS_MAPPINGS = {
    "OllamaQuery": OllamaQuery,
    "MilvusQuery": MilvusQuery,
}
