"""Official Evaluation Wrapper for VRSBench Captioning Benchmark (RSGG Qwen2.5-14B LoRA SOTA).
Standard CLI invocation for 100% reproduction of Table 2 (CIDEr 35.28, METEOR 25.85, BLEU-4 15.64).
"""

import sys
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def main():
    script_target = PROJECT_ROOT / "scripts" / "eval_vrsbench_full_9350_sg_caption.py"
    
    cmd = [sys.executable, str(script_target)] + sys.argv[1:]
    print(f"Executing: {' '.join(cmd)}")
    sys.exit(subprocess.call(cmd, cwd=str(PROJECT_ROOT)))

if __name__ == "__main__":
    main()
