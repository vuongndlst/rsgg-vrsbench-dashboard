import sys
try:
    import datasets
    print("datasets version:", datasets.__version__)
except ImportError:
    print("datasets is not installed")

