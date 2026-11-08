import sys, os, subprocess

print("Python:", sys.version)
try:
    import datasets
    print("datasets version:", datasets.__version__)
except ImportError:
    print("datasets library NOT installed.")

try:
    import evaluate
    print("evaluate installed")
except ImportError:
    print("evaluate NOT installed")

try:
    import rouge_score
    print("rouge_score installed")
except ImportError:
    print("rouge_score NOT installed")

try:
    import bert_score
    print("bert_score installed")
except ImportError:
    print("bert_score NOT installed")

# Check git-lfs or curl
res = subprocess.run(['which', 'git-lfs'], capture_output=True, text=True)
print("git-lfs:", res.stdout.strip() if res.returncode == 0 else "not found")

