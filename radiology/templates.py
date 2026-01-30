def radiology_impression_template(findings: list[str], conclusion: str, followup: list[str]) -> str:
    f = "\n".join(f"- {x}" for x in findings) if findings else "- (none)"
    fu = "\n".join(f"- {x}" for x in followup) if followup else "- (none)"
    return f"""IMPRESSION (structured):
FINDINGS:
{f}

CONCLUSION:
{conclusion}

FOLLOW-UP / RECOMMENDATIONS:
{fu}
"""
