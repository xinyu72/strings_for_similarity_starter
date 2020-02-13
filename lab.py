from nltk import word_tokenize
from nltk.translate.nist_score import sentence_nist
from util import parse_sts
from nltk.translate.bleu_score import sentence_bleu
from nltk.metrics import *

import numpy


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



def lcs(s1, s2): 
	m=[[0 for i in range(len(s2)+1)]  for j in range(len(s1)+1)]  #生成0矩阵，为方便后续计算，比字符串长度多了一列
	mmax=0   #最长匹配的长度
	p=0  #最长匹配对应在s1中的最后一位
	for i in range(len(s1)):
		for j in range(len(s2)):
			if s1[i]==s2[j]:
				m[i+1][j+1]=m[i][j]+1
				if m[i+1][j+1]>mmax:
					mmax=m[i+1][j+1]
					p=i+1
	return mmax   #返回最长子串及其长度




def lcs_cal(sts_data):
    texts, labels = parse_sts(sts_data)
    
    lcs_list = []
    for i, pair in enumerate(texts):
        t1, t2 = pair
        
        # input tokenized text
        t1 = t1.lower()
        t2 = t2.lower()
        
        lcs_score = lcs(t1,t2)

        
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
        
        ld_score = edit_distance(t1,t2)
 
        ld_list.append(ld_score)
        
    return ld_list

def editDistance(r, h):
    '''
    This function is to calculate the edit distance of reference sentence and the hypothesis sentence.
    Main algorithm used is dynamic programming.
    Attributes: 
        r -> the list of words produced by splitting reference sentence.
        h -> the list of words produced by splitting hypothesis sentence.
    '''
    d = numpy.zeros((len(r)+1)*(len(h)+1), dtype=numpy.uint8).reshape((len(r)+1, len(h)+1))
    for i in range(len(r)+1):
        d[i][0] = i
    for j in range(len(h)+1):
        d[0][j] = j
    for i in range(1, len(r)+1):
        for j in range(1, len(h)+1):
            if r[i-1] == h[j-1]:
                d[i][j] = d[i-1][j-1]
            else:
                substitute = d[i-1][j-1] + 1
                insert = d[i][j-1] + 1
                delete = d[i-1][j] + 1
                d[i][j] = min(substitute, insert, delete)
    return d

def wer(r, h):
    """
    This is a function that calculate the word error rate in ASR.
    You can use it like this: wer("what is it".split(), "what is".split()) 
    """
    # build the matrix
    d = editDistance(r, h)
    # print the result in aligned way
    result = float(d[len(r)][len(h)]) / len(r) 

    return result
      
def wer_cal(sts_data):
    texts, labels = parse_sts(sts_data)
    
    wer_list = []
    for i, pair in enumerate(texts):
        t1, t2 = pair
        
        # input tokenized text
        t1 = t1.lower()
        t2 = t2.lower()
        
        wer1 = wer(word_tokenize(t1),word_tokenize(t2))
        wer2 = wer(word_tokenize(t2),word_tokenize(t1))
        wer_total = wer1 + wer2
        
        
          
        wer_list.append(wer_total) 
        
    return wer_list


        
        
    
    
    
    
        