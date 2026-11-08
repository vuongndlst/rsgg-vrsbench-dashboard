"""Official Evaluation Wrapper for VRSBench Detection & Visual Grounding (RSGG Detector v2 SOTA).
Standard CLI invocation for 100% reproduction of Table 1 (mIoU 70.35%, Acc@0.5 85.97%).
"""

import sys
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def main():
    script_target = PROJECT_ROOT / "scripts" / "eval_detector_full_9350_test.py"
    
    cmd = [sys.executable, str(script_target)] + sys.argv[1:]
    print(f"Executing: {' '.join(cmd)}")
    sys.exit(subprocess.call(cmd, cwd=str(PROJECT_ROOT)))

if __name__ == "__main__":
    main()
