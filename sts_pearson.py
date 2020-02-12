from scipy.stats import pearsonr
import argparse
from util import parse_sts
from lab import nist_cal, bleu_cal, lcs_cal, ld_cal, wer_cal

def maxminNormalization(x,Max,Min):
    return (x - Min) / (Max - Min)


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
    
    scores['NIST'] = nist_cal(sts_data)
    scores['NIST'] = [maxminNormalization(i,5,0) for i in scores['NIST']]
    
    #use sentence_bleu function in nltk package to calculate BLEU score
    scores['BLEU'] = bleu_cal(sts_data)
    scores['BLEU'] = [maxminNormalization(i,5,0) for i in scores['BLEU']]
    
    
    scores['Word Error Rate'] = wer_cal(sts_data)
    scores['Word Error Rate'] = [maxminNormalization(i,5,0) for i in scores['Word Error Rate']]
    
    scores['Longest common substring'] = lcs_cal(sts_data)
    scores['Longest common substring'] = [maxminNormalization(i,5,0) for i in scores['Longest common substring']]
    
    scores['Levenshtein distance'] = ld_cal(sts_data)
    scores['Levenshtein distance'] = [maxminNormalization(i,5,0) for i in scores['Levenshtein distance']]


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

