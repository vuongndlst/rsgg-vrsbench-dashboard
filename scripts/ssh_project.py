"""SSH project operations with sanitized output and no credential logging."""
from pathlib import Path
import argparse, json, re, sys, shlex, time, os
sys.path.insert(0,str(Path(__file__).absolute().parents[1]/'research'/'tools'))
from p0_audit import connect
ROOT=Path(__file__).absolute().parents[1]
def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['run','sync','fetch']);parser.add_argument('path');parser.add_argument('--report');parser.add_argument('--environment',choices=['system','project'],default='system');parser.add_argument('--timeout',type=int,default=1800);args=parser.parse_args()
    c=None
    try:
        c=connect();s=c.open_sftp();home=s.normalize('.');base=home+'/rsg-former'
        def safe(v):
            return re.sub(r'/(home|users)/[^/\s]+',r'/\1/${ACCOUNT}',v.replace(home,'${REMOTE_HOME}'))
        def mkdir(path):
            try:s.stat(path)
            except OSError:
                parent=path.rsplit('/',1)[0]
                if parent and parent!=home:mkdir(parent)
                s.mkdir(path)
        if args.action=='sync':
            local=ROOT/args.path
            files=[local] if local.is_file() else [p for p in local.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix not in ['.pyc']]
            for p in files:
                rel=p.relative_to(ROOT).as_posix();target=base+'/'+rel;mkdir(target.rsplit('/',1)[0]);s.put(str(p),target)
            print(json.dumps({'synced_files':len(files),'project_root':'${REMOTE_HOME}/rsg-former'}))
        elif args.action=='fetch':
            rel=args.path.replace('\\','/');assert '..' not in Path(rel).parts
            target=ROOT/rel;target.parent.mkdir(parents=True,exist_ok=True);s.get(base+'/'+rel,str(target))
            # Only reports/code are downloaded; sanitize all text output returned from SSH.
            if target.suffix in ['.json','.jsonl','.md','.txt','.log']:
                target.write_text(safe(target.read_text(encoding='utf-8',errors='replace')),encoding='utf-8')
            print(json.dumps({'fetched':rel,'bytes':target.stat().st_size}))
        else:
            payload=(ROOT/args.path).read_text(encoding='utf-8')
            exe=base+'/.venv/bin/python' if args.environment=='project' else '/usr/bin/python3'
            command='PYTHONDONTWRITEBYTECODE=1 CUDA_CACHE_DISABLE=1 '
            if args.environment=='project':command+='PYTHONPATH='+shlex.quote(base+'/.tools/deps:'+base+'/src')+' '
            command+=shlex.quote(exe)+' -B -'
            stdin,stdout,stderr=c.exec_command(command,timeout=args.timeout)
            stdin.write(payload);stdin.channel.shutdown_write()
            ch=stdout.channel;chunks=[];err=[]
            while not ch.exit_status_ready() or ch.recv_ready() or ch.recv_stderr_ready():
                if ch.recv_ready():
                    piece=ch.recv(65536).decode(errors='replace');chunks.append(piece);print(safe(piece),end='',flush=True)
                if ch.recv_stderr_ready():err.append(ch.recv_stderr(65536).decode(errors='replace'))
                time.sleep(.1)
            status=ch.recv_exit_status()
            if err:print(safe(''.join(err))[-6000:],file=sys.stderr)
            if args.report:
                target=ROOT/args.report;target.parent.mkdir(parents=True,exist_ok=True);target.write_text(safe(''.join(chunks)),encoding='utf-8')
            if status:sys.exit(status)
    except Exception as e:
        print(json.dumps({'error_type':type(e).__name__}),file=sys.stderr);sys.exit(1)
    finally:
        if c:c.close()
if __name__=='__main__':main()
