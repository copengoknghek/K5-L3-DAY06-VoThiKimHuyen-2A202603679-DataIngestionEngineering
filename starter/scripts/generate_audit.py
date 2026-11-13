"""Build manual-audit scaffolding from retrieval traces (metadata + filter checks)."""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "results"
BENCHMARK = REPO / "benchmark" / "questions.json"


def meta_ok(hit: dict, question: dict) -> bool:
    if hit.get("document_id") != question.get("gold_source_id"):
        return False
    if hit.get("page") != question.get("gold_page"):
        return False
    if question.get("filter") and hit.get("chapter") != question.get("filter", {}).get("chapter"):
        return False
    return True


def main() -> None:
    questions = {q["id"]: q for q in json.loads(BENCHMARK.read_text(encoding="utf-8"))["questions"]}
    test_ids = [qid for qid, q in questions.items() if q["split"] == "test"]

    audit: dict = {"metadata_checks": [], "filter_compliance": [], "notes": []}
    for strategy in ("A", "B", "C"):
        traces = json.loads((RESULTS / f"traces_{strategy}.json").read_text(encoding="utf-8"))
        by_id = {t["question_id"]: t for t in traces}

        for qid in test_ids:
            q = questions[qid]
            hits = by_id[qid]["hits"]
            for rank, hit in enumerate(hits[:3], start=1):
                audit["metadata_checks"].append(
                    {
                        "strategy": strategy,
                        "question_id": qid,
                        "rank": rank,
                        "ok": meta_ok(hit, q),
                        "chunk_id": hit.get("chunk_id"),
                        "page": hit.get("page"),
                        "chapter": hit.get("chapter"),
                    }
                )
            if q.get("filter"):
                compliant = all(
                    h.get("chapter") == q["filter"]["chapter"] for h in hits[:3]
                )
                audit["filter_compliance"].append(
                    {
                        "strategy": strategy,
                        "question_id": qid,
                        "filter": q["filter"],
                        "top3_compliant": compliant,
                    }
                )

    meta = audit["metadata_checks"]
    meta_ok_count = sum(1 for m in meta if m["ok"])
    audit["summary"] = {
        "metadata_ok": f"{meta_ok_count}/{len(meta)}",
        "filter_rows": len(audit["filter_compliance"]),
        "filter_ok": sum(1 for f in audit["filter_compliance"] if f["top3_compliant"]),
    }
    out = RESULTS / "audit_metadata.json"
    out.write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(audit["summary"], indent=2))


if __name__ == "__main__":
    main()
