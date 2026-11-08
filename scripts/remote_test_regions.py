from pathlib import Path
import sys,unittest,json,time
ROOT=Path.home()/'rsg-former';sys.path[:0]=[str(ROOT/'.tools/deps'),str(ROOT/'src')]
suite=unittest.defaultTestLoader.discover(str(ROOT/'tests'),'test_regions.py');result=unittest.TextTestRunner(verbosity=2).run(suite)
out={'utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'status':'PASS' if result.wasSuccessful() else 'FAIL','tests':result.testsRun,'failures':len(result.failures),'errors':len(result.errors)}
(ROOT/'reports/VG_METHOD_TESTS.json').write_text(json.dumps(out,indent=2));print(json.dumps(out),flush=True)
if not result.wasSuccessful():raise SystemExit(1)
