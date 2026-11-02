"""Expose xcresult counts through API-readable notices,without clinical/log content."""
import json
from pathlib import Path
import subprocess

bundles=sorted(Path('mobile/artifacts').glob('NativeRun.*/NativeTests.xcresult'),key=lambda p:p.stat().st_mtime,reverse=True)
summary={'source':'xcresulttool','status':'unavailable','native_job_result_is_separate':True}
if bundles:
    try:
        result=subprocess.run(['xcrun','xcresulttool','get','test-results','summary','--path',str(bundles[0])],text=True,capture_output=True,timeout=30,check=True)
        parsed=json.loads(result.stdout)
        counts={key:parsed.get(key) for key in ('passedTests','failedTests','skippedTests','totalTestCount')}
        if all(isinstance(value,int) and value>=0 for value in counts.values()):summary.update(status='reported',**counts)
    except (OSError,ValueError,subprocess.SubprocessError):pass
print('::notice title=Raphael native test summary::'+json.dumps(summary))
