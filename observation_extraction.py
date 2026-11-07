"""Bounded,deterministic synthetic tabular observations;no NLP,OCR or inference."""
import csv
from datetime import datetime,timezone
import hashlib
import io
import math

HEADERS=('patient_id','kind','label','recorded_at','value','unit','reference_low','reference_high')


def preview(content,patient_id):
    if not content or len(content)>65536:raise ValueError('Use a nonempty tabular text document up to64KiB.')
    try:
        text=content.decode('utf-8-sig')
        if '\x00' in text:raise ValueError('NUL bytes are not supported.')
        reader=csv.DictReader(io.StringIO(text),strict=True)
        if tuple(reader.fieldnames or ())!=HEADERS:raise ValueError('Use the exact documented observation header.')
        rows=[];errors=[];physical=text.splitlines()
        for index,row in enumerate(reader):
            if index>=50:raise ValueError('At most50observation rows are supported.')
            number=reader.line_num
            if row.get('patient_id')!=str(patient_id):raise ValueError('Document patient identifier does not match the selected synthetic chart.')
            try:
                if set(row)!=set(HEADERS) or any(not isinstance(v,str) or '\n' in v or '\r' in v for v in row.values()):raise ValueError()
                kind=row['kind'];label=row['label'].strip();unit=row['unit'].strip()
                if kind not in ('lab','vital') or not 1<=len(label)<=200 or not 1<=len(unit)<=100:raise ValueError()
                stamp=datetime.fromisoformat(row['recorded_at'])
                if stamp.tzinfo is None or stamp>datetime.now(timezone.utc):raise ValueError()
                value=float(row['value']);low=float(row['reference_low']) if row['reference_low'] else None;high=float(row['reference_high']) if row['reference_high'] else None
                if any(v is not None and not math.isfinite(v) for v in (value,low,high)) or low is not None and high is not None and low>high:raise ValueError()
                rows.append({'line_number':number,'line_sha256':hashlib.sha256(physical[number-1].encode()).hexdigest(),'fields':{'kind':kind,'label':label,'recorded_at':stamp.astimezone(timezone.utc).isoformat(),'value':value,'unit':unit,'reference_low':low,'reference_high':high}})
            except (ValueError,TypeError,OverflowError):errors.append({'line_number':number,'code':'invalid_observation','detail':'Check kind,label,timezone-aware past date,finite value,unit and reference bounds.'})
        if not rows and not errors:raise ValueError('No observation rows found.')
        return {'parser_version':1,'rows':rows,'errors':errors,'limitations':'Exact synthetic CSV-in-text format only;no free-text/PDF/image extraction,unit conversion,patient identity matching or medical validation. No confidence score is assigned. Clinician review is required before creating any chart record.'}
    except (UnicodeError,csv.Error):raise ValueError('Malformed UTF-8 tabular text.') from None
