from nltk import word_tokenize
from nltk.translate.nist_score import sentence_nist
from util import parse_sts
from nltk.translate.bleu_score import sentence_bleu
import Levenshtein
from jiwer import wer
import pylcs

def nist_cal(sts_data):
    """Calculate NIST metric for pairs of strings
    Data is formatted as in the STS benchmark"""

    # read the dataset
    texts, labels = parse_sts(sts_data)

    nist_list = []
    for i, pair in enumerate(texts):
        t1, t2 = pair
        
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
        
        nist_list.append(nist_total)
        
    return nist_list



def bleu_cal(sts_data):
    texts, labels = parse_sts(sts_data)

    bleu_list = []
    for i, pair in enumerate(texts):
        t1, t2 = pair
        
        # input tokenized text
        t1_toks = word_tokenize(t1.lower())
        t2_toks = word_tokenize(t2.lower())
        
        try:
            bleu_1 = sentence_bleu([t1_toks,], t2_toks)
        except ZeroDivisionError:
            bleu_1 = 0.0


        try:
            bleu_2 = sentence_bleu([t2_toks, ], t1_toks)
        except ZeroDivisionError:
            bleu_2 = 0.0

        # sum to produce one metric
        bleu_total = bleu_1 + bleu_2
        
        bleu_list.append(bleu_total)
        
    return bleu_list


    
def lcs_cal(sts_data):
    texts, labels = parse_sts(sts_data)
    
    lcs_list = []
    for i, pair in enumerate(texts):
        t1, t2 = pair
        
        # input tokenized text
        t1 = t1.lower()
        t2 = t2.lower()
        
        lcs_score = pylcs.lcs2(t1, t2)

        
        lcs_list.append(lcs_score)
        
    return lcs_list



def ld_cal(sts_data):
    texts, labels = parse_sts(sts_data)
    
    ld_list = []
    for i, pair in enumerate(texts):
        t1, t2 = pair
        
        # input tokenized text
        t1 = t1.lower()
        t2 = t2.lower()
        
        ld_score = Levenshtein.distance(t1,t2)
 
        ld_list.append(ld_score)
        
    return ld_list


def wer_cal(sts_data):
    texts, labels = parse_sts(sts_data)
    
    wer_list = []
    for i, pair in enumerate(texts):
        t1, t2 = pair
        
        # input tokenized text
        t1 = t1.lower()
        t2 = t2.lower()
        
        wer_score1 = wer(word_tokenize(t1),word_tokenize(t2))
        wer_score2 = wer(word_tokenize(t2),word_tokenize(t1))
        
        wer_total = wer_score1 + wer_score2
        
        
        wer_list.append(wer_total)
        
    return wer_list


        
        
    
    
    
    
        