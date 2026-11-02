"""Export actual synthetic simulator attachments;never browser substitutes."""
import json
from pathlib import Path
import subprocess


results=sorted(Path('mobile/artifacts').glob('NativeRun.*/NativeTests.xcresult'),key=lambda item:item.stat().st_mtime,reverse=True)
if not results:
    print('::notice title=Raphael native screenshots::No xcresult exists;no simulator screenshots exported.')
else:
    output=results[0].parent/'NativeAttachments'
    subprocess.run(['xcrun','xcresulttool','export','attachments','--path',str(results[0]),'--output-path',str(output)],check=True,timeout=90)
    print('::notice title=Raphael native screenshots::'+json.dumps({'exported_png_files':len(list(output.rglob('*.png'))),'source':'actual_xcresult_attachments','authenticated_workflow_result_is_separate':True}))
