import os, json

os.chdir(os.path.expanduser('~/rsg-former'))

with open("reports/VRSBENCH_QWEN_G2T_1000_EVAL_REPORT.json") as f:
    d = json.load(f)
    print("Keys in 1000 report:", d.keys())
    if 'metrics' in d:
        print("1000 METRICS:", json.dumps(d['metrics'], indent=2))
    elif 'results' in d:
        print("1000 RESULTS:", json.dumps(d['results'], indent=2))
    elif 'eval_metrics' in d:
        print("1000 EVAL METRICS:", json.dumps(d['eval_metrics'], indent=2))
    else:
        for k in d:
            if not isinstance(d[k], list):
                print(f"  {k}: {d[k]}")
