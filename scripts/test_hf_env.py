import sys
try:
    import huggingface_hub
    print("huggingface_hub installed version:", huggingface_hub.__version__)
except ImportError:
    print("huggingface_hub NOT installed")

