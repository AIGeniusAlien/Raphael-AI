def enforce_citations(output_dict: dict):
    for item in output_dict.get("ranked_differential", []):
        if not item.get("citations"):
            raise ValueError("Safety violation: differential item missing citations.")
    if output_dict.get("working_diagnosis"):
        any_cited = any(i.get("citations") for i in output_dict.get("ranked_differential", []))
        if not any_cited:
            raise ValueError("Safety violation: working diagnosis without cited evidence.")
