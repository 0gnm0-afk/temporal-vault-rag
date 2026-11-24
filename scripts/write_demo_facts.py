"""Write demo/facts.jsonl, the curated fact observations of the synthetic demo vault.

The demo vault is small enough to curate by hand; this script keeps the observations
readable (evidence written as "s:<date>" = that day's session, "t:<topic>#<heading>")
and resolves them to chunk ids. Run after the demo chunks exist:
  python -m tvrag.chunk --config configs/demo.yaml   (or python -m tvrag.run ...)
  python scripts/write_demo_facts.py
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHUNKS = ROOT / "demo/build/chunks.jsonl"
OUT = ROOT / "demo/facts.jsonl"

AM, MEAL, GYM, INB = "아침 공복", "저녁 식후", "운동 직후", "인바디 오전 공복"
HOME, GYMSCALE, INBODY = "집 체중계", "헬스장 체중계", "인바디"
COND_RULE = "체중 기록 > 측정 원칙"

# (id, attr, value, unit, date, series, device, nature, evidence, extra)
OBS = [
    ("w-0106", "체중", 81.2, "kg", "2025-01-06", AM, HOME, "측정", ["s:2025-01-06", "t:체중 기록#2025-01-06"], {}),
    ("w-0120", "체중", 82.0, "kg", "2025-01-20", MEAL, HOME, "측정", ["s:2025-01-20", "t:체중 기록#2025-01-20"],
     {"note": "식후라 추세 비교에서 제외"}),
    ("w-0203", "체중", 80.1, "kg", "2025-02-03", AM, HOME, "측정", ["s:2025-02-03", "t:체중 기록#2025-02-03"], {}),
    ("w-0217", "체중", 79.4, "kg", "2025-02-17", AM, HOME, "측정", ["s:2025-02-17", "t:체중 기록#2025-02-17"], {}),
    ("w-0303", "체중", 78.6, "kg", "2025-03-03", AM, HOME, "측정", ["s:2025-03-03", "t:체중 기록#2025-03-03"], {}),
    ("w-0317", "체중", 77.5, "kg", "2025-03-17", GYM, GYMSCALE, "측정", ["s:2025-03-17", "t:체중 기록#2025-03-17"],
     {"note": "운동 직후, 참고만"}),
    ("w-0331", "체중", 78.0, "kg", "2025-03-31", AM, HOME, "측정", ["s:2025-03-31", "t:체중 기록#2025-03-31"], {}),
    ("w-0414", "체중", 77.2, "kg", "2025-04-14", AM, HOME, "측정", ["s:2025-04-14", "t:체중 기록#2025-04-14"], {}),
    ("w-0428", "체중", 76.9, "kg", "2025-04-28", AM, HOME, "측정", ["s:2025-04-28", "t:체중 기록#2025-04-28"], {}),
    ("w-0512", "체중", 76.5, "kg", "2025-05-12", AM, HOME, "측정", ["s:2025-05-12", "t:체중 기록#2025-05-12"], {}),
    ("w-0526", "체중", 76.8, "kg", "2025-05-26", AM, HOME, "측정", ["s:2025-05-26", "t:체중 기록#2025-05-26"], {}),
    ("w-0609", "체중", 76.1, "kg", "2025-06-09", AM, HOME, "측정", ["s:2025-06-09", "t:체중 기록#2025-06-09"], {}),
    ("w-0623", "체중", 75.9, "kg", "2025-06-23", AM, HOME, "측정", ["s:2025-06-23", "t:체중 기록#2025-06-23"], {}),
    # InBody weight: a different device, its own series (the vault says not to mix it into the trend)
    ("iw-0110", "체중", 81.0, "kg", "2025-01-10", INB, INBODY, "측정", ["s:2025-01-10", "t:인바디 기록"], {}),
    ("iw-0412", "체중", 77.4, "kg", "2025-04-12", INB, INBODY, "측정", ["s:2025-04-12", "t:인바디 기록"], {}),
    ("iw-0616", "체중", 76.0, "kg", "2025-06-16", INB, INBODY, "측정", ["s:2025-06-16", "t:인바디 기록"], {}),
    ("m-0110", "골격근량", 31.4, "kg", "2025-01-10", "", INBODY, "측정", ["s:2025-01-10", "t:인바디 기록"], {}),
    ("m-0412", "골격근량", 31.9, "kg", "2025-04-12", "", INBODY, "측정", ["s:2025-04-12", "t:인바디 기록"], {}),
    ("m-0616", "골격근량", 32.3, "kg", "2025-06-16", "", INBODY, "측정", ["s:2025-06-16", "t:인바디 기록"], {}),
    ("p-0110", "체지방률", 27.8, "%", "2025-01-10", "", INBODY, "측정", ["s:2025-01-10", "t:인바디 기록"], {}),
    ("p-0412", "체지방률", 25.1, "%", "2025-04-12", "", INBODY, "측정", ["s:2025-04-12", "t:인바디 기록"], {}),
    ("p-0616", "체지방률", 23.6, "%", "2025-06-16", "", INBODY, "측정", ["s:2025-06-16", "t:인바디 기록"], {}),
    ("k-0106", "식단 한도", 2300, "kcal", "2025-01-06", "", None, "사용자 결정",
     ["s:2025-01-06", "t:식단 기준#변경 이력", "s:2025-06-23"], {"detail": "하루"}),
    ("k-0203", "식단 한도", 2100, "kcal", "2025-02-03", "", None, "사용자 결정",
     ["s:2025-02-03", "t:식단 기준#2025-02-03 첫 조정", "s:2025-06-23"], {"detail": "하루"}),
    ("k-0317", "식단 한도", 1900, "kcal", "2025-03-17", "", None, "사용자 결정",
     ["s:2025-03-17", "t:식단 기준#2025-03-17 두 번째 조정", "s:2025-06-23"], {"detail": "하루"}),
    ("k-0512", "식단 한도", 2000, "kcal", "2025-05-12", "", None, "사용자 결정",
     ["s:2025-05-12", "t:식단 기준#2025-05-12 유지기 전환", "s:2025-06-23", "t:식단 기준"], {"detail": "하루"}),
    ("g-0106", "단백질 목표", 110, "g", "2025-01-06", "", None, "사용자 결정",
     ["s:2025-01-06", "t:식단 기준#변경 이력"], {"detail": "하루"}),
    ("g-0303", "단백질 목표", 140, "g", "2025-03-03", "", None, "사용자 결정",
     ["s:2025-03-03", "t:식단 기준#2025-03-03 단백질 상향"], {"detail": "하루"}),
    ("g-0602", "단백질 목표", 130, "g", "2025-06-02", "", None, "사용자 결정",
     ["s:2025-06-02", "t:식단 기준#2025-06-02 단백질 조정", "s:2025-06-23", "t:식단 기준"], {"detail": "하루"}),
    ("r-0106", "운동 루틴", "주 3회 전신(고블릿 스쿼트, 덤벨 벤치프레스, 랫풀다운, 플랭크) + 매일 7천 보 걷기", None,
     "2025-01-06", "", None, "사용자 결정", ["s:2025-01-06", "t:운동 루틴#이전 루틴과 비교"], {}),
    ("r-0303", "운동 루틴", "주 4회 상하체 분할(상체 벤치프레스·바벨 로우·오버헤드 프레스 / 하체 백스쿼트·루마니안 데드리프트·런지) + 걷기", None,
     "2025-03-03", "", None, "사용자 결정", ["s:2025-03-03", "t:운동 루틴#바뀐 이유"], {}),
    ("r-0324", "운동 루틴", "분할 4회, 런지 대신 레그프레스(2주)", None,
     "2025-03-24", "", None, "사용자 결정", ["s:2025-03-24", "t:운동 루틴#바뀐 이유"], {}),
    ("r-0407", "운동 루틴", "분할 4회 + 레그 익스텐션 3세트(가벼운 무게), 런지 미복귀", None,
     "2025-04-07", "", None, "사용자 결정", ["s:2025-04-07"], {}),
    ("r-0505", "운동 루틴", "걷기 대신 인터벌 달리기 시험(1분 빠르게·2분 천천히 8세트)", None,
     "2025-05-05", "", None, "제안", ["s:2025-05-05"], {"note": "이번 주 두 번 시험 후 결정"}),
    ("r-0512", "운동 루틴", "주 4회 상하체 분할 + 인터벌 달리기 주 2회, 하체 보조는 불가리안 스플릿 스쿼트", None,
     "2025-05-12", "", None, "사용자 결정", ["s:2025-05-12", "t:운동 루틴", "t:운동 루틴#바뀐 이유", "s:2025-06-23"], {}),
]


def main() -> None:
    rows = [json.loads(line) for line in CHUNKS.read_text(encoding="utf-8").splitlines() if line.strip()]

    def resolve(ref: str) -> str:
        kind, _, spec = ref.partition(":")
        if kind == "s":
            ids = [r["chunk_id"] for r in rows if r["doc_type"] == "session" and Path(r["doc_path"]).stem.startswith(spec)]
        else:
            topic, _, head = spec.partition("#")
            ids = [r["chunk_id"] for r in rows if r["doc_type"] == "topic" and Path(r["doc_path"]).stem == topic
                   and ((r["heading_path"] or [""])[-1] == head if head else len(r["heading_path"]) == 1)]
        if len(ids) != 1:
            raise SystemExit(f"cannot resolve {ref!r}: {ids}")
        return ids[0]

    with OUT.open("w", encoding="utf-8", newline="\n") as f:
        for oid, attr, value, unit, date, series, device, nature, ev, extra in OBS:
            o = {"id": oid, "attr": attr, "value": value, "unit": unit, "date": date, "series": series,
                 "device": device, "nature": nature, "evidence": [resolve(e) for e in ev], "review": "확인"}
            if attr == "체중":
                o["cond_source"] = COND_RULE
            o.update(extra)
            f.write(json.dumps({k: v for k, v in o.items() if v is not None}, ensure_ascii=False) + "\n")
    print(f"{len(OBS)} observations -> {OUT}")


if __name__ == "__main__":
    main()
