"""
compare_results.py
==================
2つのevaluate.py出力JSONを読み込み、Semantic vs Fixed Sizeのスコアを比較表示する。

使い方:
    python compare_results.py --semantic result_semantic.json --fixed result_fixed.json
"""

import argparse
import json


def load_result(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def difficulty_label(qid: str) -> str:
    """qa_test_set.txt の定義に基づく難易度マッピング"""
    mapping = {
        "Q01": "Easy", "Q02": "Easy", "Q03": "Easy",
        "Q04": "Medium", "Q05": "Medium", "Q06": "Medium", "Q07": "Medium",
        "Q08": "Hard", "Q09": "Hard", "Q10": "Hard",
    }
    return mapping.get(qid, "Unknown")


def group_by_difficulty(results: list) -> dict:
    groups = {"Easy": [], "Medium": [], "Hard": []}
    for r in results:
        label = difficulty_label(r["id"])
        if label in groups:
            groups[label].append(r)
    return groups


def avg(scores):
    return sum(scores) / len(scores) if scores else 0.0


def print_comparison(semantic: dict, fixed: dict):
    sem_results = semantic["results"]
    fix_results = fixed["results"]

    # id順に並べる
    sem_map = {r["id"]: r for r in sem_results}
    fix_map = {r["id"]: r for r in fix_results}
    all_ids = sorted(set(sem_map) | set(fix_map))

    # ── ヘッダー ──────────────────────────────────────────────────
    print("=" * 72)
    print("  Chunking手法 比較レポート")
    print(f"  Semantic DB : {semantic.get('db_path', 'N/A')}")
    print(f"  Fixed    DB : {fixed.get('db_path', 'N/A')}")
    print("=" * 72)

    # ── 全体サマリー ───────────────────────────────────────────────
    print(f"\n{'':20s} {'Semantic':>12s} {'Fixed':>12s} {'Diff(S-F)':>12s}")
    print("-" * 60)
    for label, key in [("Retrieval Score", "avg_ret_score"), ("Generation Score", "avg_gen_score")]:
        s = semantic[key]
        f = fixed[key]
        diff = s - f
        sign = "+" if diff > 0 else ""
        print(f"  {label:18s} {s:>12.3f} {f:>12.3f} {sign}{diff:>11.3f}")

    # ── 難易度別サマリー ────────────────────────────────────────────
    print("\n--- 難易度別 Retrieval Score ---")
    print(f"{'':12s} {'Semantic':>12s} {'Fixed':>12s} {'Diff':>10s}")
    print("-" * 48)
    sem_groups = group_by_difficulty(sem_results)
    fix_groups = group_by_difficulty(fix_results)
    for diff_label in ["Easy", "Medium", "Hard"]:
        s = avg([r["ret_score"] for r in sem_groups[diff_label]])
        f = avg([r["ret_score"] for r in fix_groups[diff_label]])
        d = s - f
        sign = "+" if d > 0 else ""
        print(f"  {diff_label:10s} {s:>12.2f} {f:>12.2f} {sign}{d:>9.2f}")

    # ── 問題別詳細 ─────────────────────────────────────────────────
    print("\n--- 問題別スコア詳細 ---")
    header = f"{'QID':>5s} {'難易度':>8s} | {'Ret(S)':>7s} {'Ret(F)':>7s} {'ΔRet':>6s} | {'Gen(S)':>7s} {'Gen(F)':>7s} {'ΔGen':>6s}"
    print(header)
    print("-" * len(header))
    for qid in all_ids:
        s = sem_map.get(qid)
        f = fix_map.get(qid)
        if s is None or f is None:
            continue
        d_label = difficulty_label(qid)
        d_ret = s["ret_score"] - f["ret_score"]
        d_gen = s["gen_score"] - f["gen_score"]
        ret_sign = "+" if d_ret > 0 else ""
        gen_sign = "+" if d_gen > 0 else ""
        print(
            f"  {qid:>4s} {d_label:>8s} | "
            f"{s['ret_score']:>7d} {f['ret_score']:>7d} {ret_sign}{d_ret:>5d} | "
            f"{s['gen_score']:>7d} {f['gen_score']:>7d} {gen_sign}{d_gen:>5d}"
        )

    print("\n凡例: Semantic > Fixed の場合 Diff は正値")
    print("=" * 72)


def main():
    parser = argparse.ArgumentParser(description="Semantic vs Fixed Size Chunking 評価結果比較")
    parser.add_argument("--semantic", required=True, help="Semantic Chunkingの評価結果JSON")
    parser.add_argument("--fixed", required=True, help="Fixed Size Chunkingの評価結果JSON")
    args = parser.parse_args()

    semantic = load_result(args.semantic)
    fixed = load_result(args.fixed)
    print_comparison(semantic, fixed)


if __name__ == "__main__":
    main()
