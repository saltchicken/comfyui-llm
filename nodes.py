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
                "temperature" : ("STRING",),
                "seed" : ("STRING",),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    FUNCTION = "process"
    CATEGORY = "LLM"
    TITLE = "Ollama Query"

    def process(self, model: str, system_message: any, prompt: str, host: str, port: str, temperature: any, seed: any):
        """
        Query Ollama with a given prompt.
        """
        if system_message == "":
            system_message = None

        if temperature == "":
            temperature = None

        if seed == "":
            print("SETTING SEED TO NONE")
            seed = None

        response, debug_text = ollama_query(model=model, prompt=prompt, system_message=system_message, host=host, port=port, temperature=temperature)

        return (response, debug_text)

class MilvusQuery:
    def __init__(self):
        super().__init__()
        self.rag = rag_milvus.Rag(host = "10.0.0.7")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "query": ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("knowledge_string",)
    FUNCTION = "process"
    CATEGORY = "LLM"
    TITLE = "Milvus Query"

    def process(self, query: str):
        """
        Query Milvus with a given prompt.
        """

        knowledge = self.rag.search_knowledge(query, 0.5)
        # knowledge_string = "\n".join(knowledge)
        # knowledge_string = "\n".join(map(str, knowledge))
        knowledge_string = "\n".join(f"<knowledge>{item}</knowledge>" for item in knowledge)
        knowledge_string = "<knowledge_base>\n" + knowledge_string + "\n</knowledge_base>"
        return (knowledge_string,)

class ConcatenateStrings:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "string1": ("STRING", {"default": ""}),
                "string2": ("STRING", {"default": ""}),
            },
        }
    RETURN_TYPES = ("STRING",)
    FUNCTION = "process"
    CATEGORY = "LLM"
    TITLE = "Concatenate Strings"

    def process(self, string1, string2):
        return (string1 + "\n" + string2,)

class ShowText:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"forceInput": True}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    INPUT_IS_LIST = True
    RETURN_TYPES = ("STRING",)
    FUNCTION = "notify"
    OUTPUT_NODE = True
    TITLE = "Show Text"

    CATEGORY = "LLM"

    def notify(self, text, unique_id=None, extra_pnginfo=None):
        if unique_id is not None and extra_pnginfo is not None:
            if not isinstance(extra_pnginfo, list):
                print("Error: extra_pnginfo is not a list")
            elif (
                not isinstance(extra_pnginfo[0], dict)
                or "workflow" not in extra_pnginfo[0]
            ):
                print("Error: extra_pnginfo[0] is not a dict or missing 'workflow' key")
            else:
                workflow = extra_pnginfo[0]["workflow"]
                node = next(
                    (x for x in workflow["nodes"] if str(x["id"]) == str(unique_id[0])),
                    None,
                )
                if node:
                    node["widgets_values"] = [text]

        return {"ui": {"text": text}, "result": (text,)}

# Register node in ComfyUI
NODE_CLASS_MAPPINGS = {
    "OllamaQuery": OllamaQuery,
    "MilvusQuery": MilvusQuery,
    "ConcatenateStrings": ConcatenateStrings,
    "ShowText": ShowText,
}
