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

    start_index = 120
    for i,pair in enumerate(texts[start_index:start_index+20]):

        this_label = labels[i + start_index]

        t1, t2 = pair
        print(f"Sentences: {t1}\t{t2}")

        # input tokenized text
        t1_toks = word_tokenize(t1.lower())
        t2_toks = word_tokenize(t2.lower())

        # try / except for each side because of ZeroDivision Error
        # 0.0 is lowest score - give that if ZeroDivision Error
        try:
            nist_1 = sentence_nist([t1_toks,], t2_toks)
        except ZeroDivisionError:
            #print(f"\n\n\nno NIST, {i}")
            nist_1 = 0.0


        try:
            nist_2 = sentence_nist([t2_toks, ], t1_toks)
        except ZeroDivisionError:
            #print(f"\n\n\nno NIST, {i}")
            nist_2 = 0.0

        # sum to produce one metric
        nist_total = nist_1 + nist_2
        #print(nist_1, nist_2)
        print(f"Label: {this_label}, NIST: {nist_total:0.02f}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sts_data", type=str, default="stsbenchmark/sts-dev.csv",
                        help="sts data")
    args = parser.parse_args()

    main(args.sts_data)
