from nltk import word_tokenize
from nltk.translate.nist_score import sentence_nist
from util import parse_sts
import argparse

def main(sts_data):
    """Calculate NIST metric for pairs of strings
    Data is formatted as in the STS benchmark"""

    # read the dataset
    texts, labels = parse_sts(sts_data)


    print(f"Found {len(texts)} STS pairs")

    for i,pair in enumerate(texts[120:140]):
        t1, t2 = pair
        print(f"Sentences: {t1}\t{t2}")

        # TODO: Calculate for each pair of sentences
        # catch any exceptions and assign 0.0

        nist_score = 0.0
        print(f"Label: {labels[i]}, NIST: {nist_score:0.02f}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sts_data", type=str, default="stsbenchmark/sts-dev.csv",
                        help="sts data")
    args = parser.parse_args()

    main(args.sts_data)
