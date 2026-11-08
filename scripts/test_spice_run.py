import sys
from pycocoevalcap.spice.spice import Spice

print("Testing Spice initialization...")
spice = Spice()
print("Spice instance created.")

# Test on a dummy pair
gts = {0: ['There is a plane on the airport runway.']}
res = {0: ['A plane is parked on the airport runway.']}
try:
    score, scores = spice.compute_score(gts, res)
    print("SPICE score:", score)
except Exception as e:
    print("SPICE compute error:", e)
    # Check if get_stanford_models is needed
    import pycocoevalcap
    from pathlib import Path
    base = Path(pycocoevalcap.__file__).parent / 'spice'
    print("Base spice contents:", [f.name for f in base.iterdir()])

