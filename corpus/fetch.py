import re
import httpx
from pypdf import PdfReader
import io

def html_to_text(html: str) -> str:
    html = re.sub(r"(?is)<script.*?>.*?</script>", " ", html)
    html = re.sub(r"(?is)<style.*?>.*?</style>", " ", html)
    html = re.sub(r"(?is)<.*?>", " ", html)
    html = re.sub(r"\s+", " ", html).strip()
    return html

async def fetch_url(url: str, user_agent: str = "RaphaelAI/1.0") -> tuple[str, str]:
    """
    Returns (content_type, extracted_text).
    Supports HTML and PDF.
    """
    headers = {"User-Agent": user_agent}
    async with httpx.AsyncClient(timeout=60, follow_redirects=True) as client:
        r = await client.get(url, headers=headers)
        r.raise_for_status()
        ctype = (r.headers.get("Content-Type", "") or "").split(";")[0].strip().lower()

        # PDF
        if "application/pdf" in ctype or url.lower().endswith(".pdf"):
            reader = PdfReader(io.BytesIO(r.content))
            txt = " ".join([(p.extract_text() or "") for p in reader.pages])
            txt = re.sub(r"\s+", " ", txt).strip()
            return "application/pdf", txt

        # HTML/text
        txt = html_to_text(r.text)
        return ctype or "text/html", txt
