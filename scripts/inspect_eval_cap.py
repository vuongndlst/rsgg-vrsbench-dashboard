import json

with open("data/vrsbench/VRSBench_EVAL_Cap.json", "r") as f:
    d = json.load(f)

print("Total keys in VRSBench_EVAL_Cap:", len(d))
sample_k = list(d.keys())[0]
print("Sample key:", sample_k)
print("Sample value type:", type(d[sample_k]))
print("Sample value:", d[sample_k])
if len(d) > 1:
    print("Second sample:", list(d.keys())[1], d[list(d.keys())[1]])

