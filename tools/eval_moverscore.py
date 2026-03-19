import os
import argparse
import pandas as pd
from moverscore_v2 import get_idf_dict, word_mover_score  # 官方推荐的加速版接口
from statistics import meanban

def main():
    parser = argparse.ArgumentParser(description="Compute MoverScore for a CSV with predict/label columns.")
    parser.add_argument("--csv", required=True, help="Path to input CSV.")
    parser.add_argument("--pred_col", default="predict", help="Column name for predictions.")
    parser.add_argument("--label_col", default="label", help="Column name for references (GT).")
    parser.add_argument("--out", default="with_moverscore.csv", help="Output CSV path.")
    parser.add_argument("--model", default=None,
                        help="HF model name for embeddings. "
                             "e.g., distilbert-base-uncased (en), xlm-roberta-base (multi/zh). "
                             "If not set, moverscore_v2 default is used.")
    parser.add_argument("--n_gram", type=int, default=1, help="1 for unigram (default), 2 for bigram.")
    parser.add_argument("--remove_subwords", type=str, default="auto",
                        help="auto|true|false. 'auto'→英文True、中文/多语False。")
    args = parser.parse_args()

    # （可选）指定底层模型：必须在导入/首次调用前设置环境变量
    if args.model:
        os.environ["MOVERSCORE_MODEL"] = args.model  # 官方 README 中提供的方式
        # 例如：英文/默认 -> distilbert-base-uncased；中文/多语 -> xlm-roberta-base 或 bert-base-multilingual-cased

    # 读表
    df = pd.read_csv(args.csv)
    if args.pred_col not in df.columns or args.label_col not in df.columns:
        raise ValueError(f"CSV缺少列：{args.pred_col} 或 {args.label_col}")

    preds = df[args.pred_col].fillna("").astype(str).tolist()
    refs  = df[args.label_col].fillna("").astype(str).tolist()
    if len(preds) != len(refs):
        raise ValueError("predict 与 label 行数不一致。")

    # 自动判断是否中文/多语：粗略规则（包含中文字符或非 ASCII）
    def looks_multilingual(texts, sample=100):
        import re
        pat = re.compile(r"[^\x00-\x7F]")  # 非 ASCII
        take = texts[:min(sample, len(texts))]
        return any(pat.search(t or "") for t in take)

    multilingual = looks_multilingual(preds + refs)

    # remove_subwords 设定
    if args.remove_subwords.lower() == "true":
        remove_subwords = True
    elif args.remove_subwords.lower() == "false":
        remove_subwords = False
    else:  # auto
        remove_subwords = False if multilingual else True

    # 计算 IDF 字典（官方推荐做法，可用 defaultdict(lambda:1.) 退化为等权）
    idf_hyp = get_idf_dict(preds)
    idf_ref = get_idf_dict(refs)

    # 逐行计算 MoverScore；word_mover_score 支持批量列表
    scores = word_mover_score(
        refs, preds, idf_ref, idf_hyp,
        stop_words=[], n_gram=args.n_gram, remove_subwords=remove_subwords
    )

    df["moverscore"] = scores
    print(f"Mean MoverScore: {mean(scores):.4f}")
    df.to_csv(args.out, index=False)
    print(f"Saved → {args.out}")

if __name__ == "__main__":
    main()
