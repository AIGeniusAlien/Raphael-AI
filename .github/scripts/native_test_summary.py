"""Expose exact per-profile xcresult counts without clinical or credential content."""
import argparse
import json
from pathlib import Path
import subprocess

FIELDS=('passedTests','failedTests','skippedTests','totalTestCount')


def summarize(bundles,reader):
    profiles=[]
    for bundle in sorted(bundles,key=lambda p:p.parent.name):
        pieces=bundle.parent.name.split('.')
        profile=pieces[1] if len(pieces)>=3 else 'legacy_unlabelled'
        item={'profile':profile,'status':'unavailable'}
        try:
            parsed=reader(bundle)
            counts={key:parsed.get(key) for key in FIELDS}
            if any(type(value) is not int or value<0 for value in counts.values()):raise ValueError()
            if counts['passedTests']+counts['failedTests']+counts['skippedTests']!=counts['totalTestCount']:raise ValueError()
            item.update(status='reported',**counts)
        except (OSError,ValueError,subprocess.SubprocessError):pass
        profiles.append(item)
    valid=bool(profiles) and all(p['status']=='reported' for p in profiles)
    result={'source':'xcresulttool','status':'reported' if valid else 'unavailable','profiles':profiles,'native_job_result_is_separate':True}
    if valid:result.update({key:sum(p[key] for p in profiles) for key in FIELDS})
    return result


def accepted(summary,requirements):
    if summary['status']!='reported':return False
    if len(summary['profiles'])!=len(requirements):return False
    for profile,minimum in requirements.items():
        rows=[p for p in summary['profiles'] if p['profile']==profile]
        if len(rows)!=1 or rows[0]['passedTests']<minimum or rows[0]['failedTests'] or rows[0]['skippedTests']:return False
    return True


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--require',action='append',default=[],help='Exact profile:minimum-passing-count;missing/duplicate/failing/skipped profiles fail.');args=parser.parse_args()
    required={}
    for value in args.require:
        name,number=value.split(':',1)
        if name in required or name not in ('development','consumer_candidate') or int(number)<1:parser.error('Unique known profile and positive minimum required.')
        required[name]=int(number)
    def read(bundle):
        output=subprocess.run(['xcrun','xcresulttool','get','test-results','summary','--path',str(bundle)],text=True,capture_output=True,timeout=30,check=True)
        return json.loads(output.stdout)
    summary=summarize(Path('mobile/artifacts').glob('NativeRun.*/NativeTests.xcresult'),read)
    print('::notice title=Raphael native test summary::'+json.dumps(summary))
    return int(bool(required) and not accepted(summary,required))
if __name__=='__main__':raise SystemExit(main())
