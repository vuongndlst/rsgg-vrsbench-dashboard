try:
    import bitsandbytes as bnb
    print("bitsandbytes installed:", bnb.__version__)
except ImportError:
    print("bitsandbytes NOT installed")

