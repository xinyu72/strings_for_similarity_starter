from scipy.stats import pearsonr
import argparse
from util import parse_sts


def main(sts_data):
    """Calculate pearson correlation between semantic similarity scores and string similarity metrics.
    Data is formatted as in the STS benchmark"""

    # read the dataset
    # TODO: implement in util.py
    texts, labels = parse_sts(sts_data)

    print(f"Found {len(texts)} STS pairs")

    score_types = ["NIST", "BLEU", "Word Error Rate", "Longest common substring", "Levenshtein distance"]
    scores = {score_type: [] for score_type in score_types}

    # TODO: Calculate the metrics here to fill the lists in scores


    # This can stay as-is to print similar output to the sample
    print(f"Semantic textual similarity for {sts_data}\n")
    for metric_name, dists in scores.items():
        score, sig = pearsonr(dists, labels)
        print(f"{metric_name} correlation: {score:.03}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sts_data", type=str, default="stsbenchmark/sts-dev.csv",
                        help="tab separated sts data in benchmark format")
    args = parser.parse_args()

    main(args.sts_data)

