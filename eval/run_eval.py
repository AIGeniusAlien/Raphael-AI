import json, requests, os

def main():
    api = os.getenv("API_URL", "http://localhost:8000")
    api_key = os.getenv("API_KEY", "")
    headers = {"X-API-Key": api_key}

    cases = json.load(open("eval_cases.json","r",encoding="utf-8"))
    results = {"cases": []}

    for c in cases:
        r = requests.post(f"{api}/cases", json={"patient_id": c.get("patient_id"), "context": c.get("context",{})}, headers=headers)
        r.raise_for_status()
        case_id = r.json()["case_id"]

        for a in c.get("artifacts", []):
            requests.post(f"{api}/cases/{case_id}/artifact", json=a, headers=headers).raise_for_status()

        q = requests.post(f"{api}/query", json={"case_id": case_id, "question": c["question"], "mode": c.get("mode","general")}, headers=headers)
        q.raise_for_status()
        out = q.json()

        citations_ok = all(bool(d.get("citations")) for d in out.get("ranked_differential", []))
        results["cases"].append({"name": c.get("name",""), "case_id": case_id, "citations_ok": citations_ok, "confidence": out.get("confidence")})

    json.dump(results, open("results.json","w",encoding="utf-8"), indent=2)
    print("Wrote results.json")

if __name__ == "__main__":
    main()
