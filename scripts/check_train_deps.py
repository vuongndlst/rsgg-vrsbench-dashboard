import sys, subprocess

def check_pkg(pkg):
    try:
        __import__(pkg)
        print(f"Package {pkg}: INSTALLED")
    except ImportError:
        print(f"Package {pkg}: NOT INSTALLED")

for p in ['torch', 'transformers', 'peft', 'accelerate', 'bitsandbytes', 'datasets', 'timm', 'pil']:
    check_pkg(p)

