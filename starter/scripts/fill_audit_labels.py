"""Fill the three audit sheets in starter/results/.

- coherence / grounding: labels proposed by an AI assistant (Claude) after reading every chunk,
  then reviewed by the student; `labeler` records the reviewer and that the labels were AI-proposed.
- metadata: deterministic check of each top-3 record against the normalized PDF page text.

Run from starter/:  python scripts/fill_audit_labels.py
"""
import csv
import json
import sys
from pathlib import Path

STARTER = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STARTER))
from src.chunking import _section_at  # noqa: E402
from src.evaluation import canon  # noqa: E402
from src.ingest import ROOT, build_chunks, load_config, load_pages  # noqa: E402

RES = STARTER / "results"
AI = "Ngô Gia Bảo (duyệt lại nhãn do AI Claude đề xuất)"

COHERENCE = {  # (position, strategy) -> (score, reason); C uses B's chunk
    ("pos01", "A"): (1, "Starts with headings and learning objectives; the trademark definition is cut at the last word ('marketpla')."),
    ("pos01", "B"): (2, "Trademark definition is complete; the chunk also carries the flattened chapter outline as noise, but the idea is intact."),
    ("pos02", "A"): (1, "Starts mid-word ('demark cases.'); ends before the point of the sentence (Congress tried to federalize trademark law)."),
    ("pos02", "B"): (1, "Section 4.3 heading and context are complete, but the chunk ends mid-sentence ('a single appeal before the U.S.'); the ruling is in the next chunk."),
    ("pos03", "A"): (1, "Starts mid-quotation ('meaning, this language...') without its subject; ends mid-sentence ('The number 5, for ')."),
    ("pos03", "B"): (2, "Self-contained paragraph: numbers can be trademarks, with the Chanel No. 5 and Baskin-Robbins 31 examples."),
    ("pos04", "A"): (1, "Descriptive-mark rule is complete, but the exception example is cut mid-word ('—“Sh') and the chunk starts with a fragment of the previous paragraph."),
    ("pos04", "B"): (2, "Whole 'Descriptive Marks' subsection: rule, exception with examples (Sharp, Windows) and rationale."),
    ("pos05", "A"): (1, "List of registration benefits without its lead-in sentence (first item starts mid-phrase); last item cut ('Pate')."),
    ("pos05", "B"): (1, "Lead-in and context are clear, but the list of registration benefits stops after 3 of 7 items (rest in the next chunk)."),
    ("pos06", "A"): (1, "Starts mid-word ('nt is vicarious'); the quoted definition is cut before its key clause (no intent or knowledge required)."),
    ("pos06", "B"): (2, "Complete definition of vicarious infringement with the quoted source and the Sega v. MAPHIA example."),
    ("pos07", "A"): (1, "Starts without its subject ('enormous economic value...'); the trade-secret definition is cut mid-word ('not being gen')."),
    ("pos07", "B"): (2, "Complete opening of 5.1: what a trade secret is, its economic value (PwC study) and the absence of a novelty requirement."),
    ("pos08", "A"): (1, "Starts mid-sentence without the subject ('e with how to best protect...'); main advice is complete; ends mid-sentence."),
    ("pos08", "B"): (2, "Complete paragraph on converting unsolicited ideas into solicited ones."),
    ("pos09", "A"): (0, "Mixes the tail of one sentence from 5.4 (no subject) with the 5.5 heading and learning objectives, cut mid-word; no complete idea."),
    ("pos09", "B"): (1, "One complete sentence, but it depends on 'the different methods ... discussed above', which are not in the chunk."),
    ("pos10", "A"): (1, "Starts mid-sentence ('order prohibiting the...') without 'Injunctive relief allows the owner...'; ends mid-sentence ('Both the')."),
    ("pos10", "B"): (2, "Complete 'Injunctive Relief' paragraph: definition, importance for threatened misappropriation, and scope."),
}

GROUNDING = {  # (question, strategy) -> (score, reason)
    ("q16", "A"): (2, "All three factors are present when ranks 1 and 2 are read together (lead-in + factor 1 in r1, factors 2-3 in r2); "
                      "the evidence is split across two chunks, which is why automatic Hit@3 is False."),
    ("q16", "B"): (2, "r1 contains the full three-factor list with its lead-in; r2 adds the balancing context."),
    ("q16", "C"): (2, "Same top-3 as B (no filter): r1 contains the full three-factor list."),
    ("q17", "A"): (1, "r1 (p43) says courts may deny relief even within the limitation period if not brought reasonably promptly, "
                      "but the chunk starts at 'years,' so the three-year period is lost."),
    ("q17", "B"): (0, "No evidence: top-3 are about injunction duration (p44), proving misappropriation (p41) and secrecy measures (p40); "
                      "nothing on the limitation period or promptness."),
    ("q17", "C"): (0, "Same top-3 as B (no filter for this question): no evidence on the limitation period or promptness."),
    ("q18", "A"): (2, "r3 contains the 'New York Shoes' made in China example and the reason (confuse or deceive consumers); r1 gives the misdescriptive test."),
    ("q18", "B"): (2, "r2 contains the full deceptively-misdescriptive test and the 'New York Shoes' example."),
    ("q18", "C"): (2, "Same top-3 as B: r2 contains the full test and the example."),
    ("q19", "A"): (1, "r2 lists the remedies but is cut at 'exemplar' and omits what the remedy depends on; r1 and r3 are off-topic."),
    ("q19", "B"): (2, "r1 contains the full list of UTSA remedies and what they depend on; r2-r3 are trademark distractors (no filter)."),
    ("q19", "C"): (2, "r1 contains the full list of UTSA remedies; r3 adds damages details; all three are trade-secret chunks."),
    ("q20", "A"): (2, "r1 (p22) states the full reason: the biggest loss is damage to brand reputation, not revenue."),
    ("q20", "B"): (2, "r1 (p22) states the full reason; r2-r3 are trade-secret injunction chunks (no filter)."),
    ("q20", "C"): (2, "r1 (p22) states the full reason; r2 adds trademark damages context."),
}


def read(name: str) -> list[dict]:
    return list(csv.DictReader(open(RES / name, encoding="utf-8-sig", newline="")))


def write(name: str, rows: list[dict]) -> None:
    with open(RES / name, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    cfg = load_config()
    manifest = json.loads((ROOT / "data/manifest.json").read_text(encoding="utf-8"))["sources"][0]
    ip_ranges = manifest["page_metadata"]["ip_type_pages"]
    pages = {p.page: p for p in load_pages()}
    chunks = {}
    for spec in cfg["strategies"].values():
        for c in build_chunks(list(pages.values()), spec):
            chunks[c.chunk_id] = c

    rows = read("audit_coherence.csv")
    for r in rows:
        key = (r["position_id"], "B" if r["strategy"] == "C" else r["strategy"])
        score, reason = COHERENCE[key]
        r["coherence_0_2"] = score
        r["reason"] = reason + (" (C uses the same chunk as B.)" if r["strategy"] == "C" else "")
        r["labeler"] = AI
    write("audit_coherence.csv", rows)

    rows = read("audit_grounding.csv")
    for r in rows:
        score, reason = GROUNDING[(r["id"], r["strategy"])]
        r["grounding_0_2"], r["reason"], r["labeler"] = score, reason, AI
    write("audit_grounding.csv", rows)

    rows = read("audit_metadata.csv")
    for r in rows:
        c = chunks[r["chunk_id"]]
        page = int(r["page"])
        notes = []
        source_ok = r["source"] == manifest["local_path"]
        page_ok = canon(c.text) in canon(pages[page].text) or canon(c.text)[:200] in canon(pages[page].text)
        printed_ok = str(page + 132) == r["printed_page"]
        expected_ip = next(k for k, (lo, hi) in ip_ranges.items() if lo <= page <= hi)
        ip_ok = r["ip_type"] == expected_ip
        start_sec = _section_at(pages[page], c.metadata["char_start"])[0]
        end_sec = _section_at(pages[page], c.metadata["char_start"] + len(c.text) - 1)[0]
        section_ok = r["section"] in {start_sec, end_sec}
        if not source_ok: notes.append(f"source should be {manifest['local_path']}")
        if not page_ok: notes.append("chunk text not found on this page")
        if not printed_ok: notes.append(f"printed_page should be {page + 132}")
        if not ip_ok: notes.append(f"ip_type should be {expected_ip}")
        if not section_ok: notes.append(f"section should be {start_sec}")
        if start_sec != end_sec: notes.append(f"chunk spans sections {start_sec} -> {end_sec}")
        r.update({"source_ok": int(source_ok), "page_ok": int(page_ok and printed_ok), "ip_type_ok": int(ip_ok),
                  "section_ok": int(section_ok), "notes": "; ".join(notes) or "all fields match the PDF text",
                  "labeler": "Ngô Gia Bảo (kiểm tra tự động bằng scripts/fill_audit_labels.py)"})
    write("audit_metadata.csv", rows)

    # summary
    coh = read("audit_coherence.csv"); gr = read("audit_grounding.csv"); md = read("audit_metadata.csv")
    for s in ("A", "B", "C"):
        cs = [int(r["coherence_0_2"]) for r in coh if r["strategy"] == s]
        gs = [int(r["grounding_0_2"]) for r in gr if r["strategy"] == s]
        ms = [r for r in md if r["strategy"] == s]
        full = sum(all(r[k] == "1" for k in ("source_ok", "page_ok", "ip_type_ok", "section_ok")) for r in ms)
        print(f"{s}: coherence {sum(cs)}/{2*len(cs)} mean {sum(cs)/len(cs):.2f} | grounding "
              f"{{0:{gs.count(0)},1:{gs.count(1)},2:{gs.count(2)}}} mean {sum(gs)/len(gs):.2f} | "
              f"metadata fully correct {full}/{len(ms)}")
    spans = [f"{r['id']}{r['strategy']}r{r['rank']}" for r in md if "spans" in r["notes"]]
    print("records spanning two sections:", spans)


if __name__ == "__main__":
    main()
