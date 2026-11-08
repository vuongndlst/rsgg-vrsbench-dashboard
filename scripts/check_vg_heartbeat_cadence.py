"""Local preflight; avoid duplicate/early scheduled SSH checks."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,json

def decision(last_snapshot_utc,requested_utc,minimum_hours=12,direct=False):
    requested=datetime.fromisoformat(requested_utc.replace('Z','+00:00'))
    previous=datetime.fromisoformat(last_snapshot_utc.replace('Z','+00:00')) if last_snapshot_utc else None
    assert requested.utcoffset() is not None and (previous is None or previous.utcoffset() is not None)
    elapsed=(requested-previous).total_seconds() if previous else None
    allowed=direct or previous is None or elapsed>=minimum_hours*3600
    return {'requested_utc':requested.isoformat(),'last_successful_remote_snapshot_utc':last_snapshot_utc,
            'elapsed_seconds':elapsed,'minimum_interval_seconds':minimum_hours*3600,
            'allow_remote_check':allowed,'direct_user_request':direct,
            'decision':'ALLOW_DIRECT_REQUEST' if direct else 'ALLOW_MISSING_SNAPSHOT' if previous is None else
                       'ALLOW_DUE_CHECK' if allowed else 'SKIP_DUPLICATE_EARLY_HEARTBEAT'}

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--heartbeat-utc',required=True)
    parser.add_argument('--minimum-hours',type=int,default=12);parser.add_argument('--direct-request',action='store_true')
    args=parser.parse_args();assert args.minimum_hours>0
    root=Path(__file__).absolute().parents[1];snapshots=[]
    for name in ['VG_V2_LAST_STATUS.json','VG_COMPONENT_CONTROL_TEST_LAST_STATUS.json','VG_CAPTION_PREPARATION_LAST_CHECK.json']:
        try:snapshots.append(json.loads((root/'reports'/name).read_text(encoding='utf8'))['utc'])
        except (FileNotFoundError,json.JSONDecodeError,KeyError):pass
    last=max(snapshots,key=lambda v:datetime.fromisoformat(v.replace('Z','+00:00'))) if snapshots else None
    out=decision(last,args.heartbeat_utc,args.minimum_hours,args.direct_request)
    out.update(automation_id='theo-d-i-checkpoint-vg',additional_monitor_calls=0,additional_ssh_calls=0)
    target=root/'reports/VG_HEARTBEAT_CADENCE_LAST_CHECK.json';temporary=target.with_suffix('.tmp')
    temporary.write_text(json.dumps(out,indent=2),encoding='utf8');temporary.replace(target)
    print(json.dumps(out))

if __name__=='__main__':main()
