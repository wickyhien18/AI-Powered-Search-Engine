"""
Lấy mẫu ngẫu nhiên bài báo theo từng category để tự đặt query gán nhãn thủ công
cho GOLDEN_SET. Chạy: python sample_for_labeling.py [--per-category N] [--seed S]
"""

import argparse
import pandas as pd

CATEGORIES = ["business", "entertainment", "politics", "sport", "tech"]
SNIPPET_LEN = 220  # số ký tự đầu in ra để đọc nhanh


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="tfidf_dataset.csv")
    parser.add_argument("--per-category", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = pd.read_csv(args.csv)  # cột đầu (unnamed) tự động thành index -> article_id
    df.index.name = "article_id"

    for cat in CATEGORIES:
        subset = df[df["category"] == cat]
        sample = subset.sample(n=min(args.per_category, len(subset)), random_state=args.seed)

        print(f"\n{'=' * 70}")
        print(f"CATEGORY: {cat}  (tổng {len(subset)} bài, đang lấy mẫu {len(sample)})")
        print("=" * 70)

        for article_id, row in sample.iterrows():
            snippet = row["text"][:SNIPPET_LEN]
            print(f"\n  [article_id={article_id}]")
            print(f"  {snippet}...")

    print(f"\n{'=' * 70}")
    print("Gợi ý: với mỗi bài in ở trên, tự đặt 1 câu query (proper_noun / paraphrase / general)")
    print("mà bài đó (và có thể vài bài liên quan khác) sẽ là câu trả lời đúng, rồi thêm vào GOLDEN_SET:")
    print('  {"query": "...", "expected_article_ids": {<article_id>, ...}, "category": "..."}')


if __name__ == "__main__":
    main()
