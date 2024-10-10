import os

OPENAI_API_KEY="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"
# HUGGINGFACEHUB_API_TOKEN="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"

def set_environment():
 variable_dict = globals().items()
 for key, value in variable_dict:
    if "API" in key or "ID" in key:
        os.environ[key] = value