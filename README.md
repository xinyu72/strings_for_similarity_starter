Semantic textual similarity using string similarity
---------------------------------------------------

This project examines string similarity metrics for semantic textual similarity.
Though semantics go beyond the surface representations seen in strings, some of these
metrics constitute a good benchmark system for detecting STS.


Data is from the [STS benchmark](http://ixa2.si.ehu.es/stswiki/index.php/STSbenchmark).


## lab.py

`lab.py` calculates NIST machine translation metric for sentence pairs in an STS dataset.

Example usage:

`python lab.py --sts_data stsbenchmark/sts-dev.csv`

## dev_output.txt

Desired output for `sts_pearson.py` on `sts-dev.csv`. Homework will be evaluated against the test data.

## sts_pearson.py

Calculate pearson's correlation of semantic similarity with the metrics specified in the starter code.
Calculate the metrics between lowercased inputs and ensure that the metric is the same for either order of the sentences (i.e. sim(A,B) == sim(B,A))
Use SmoothingFunction method0 for BLEU, as described in the nltk documentation.

TODO: Update these instructions under `sts_pearson.py` with a description of your code.
* ~ 1 sentence about each of the metrics used.
* Describe what your script does. Be affirmative and efficient 
(see [guidelines for documenting the Python language]( 
https://devguide.python.org/documenting/#affirmative-tone) )
* Include a usage example showing command line flags
* Describe your output
* More README philosophy [here](https://bulldogjob.com/news/449-how-to-write-a-good-readme-for-your-github-project) 

### Introduction of metrics 
* NIST a metric which is used to evaluate the effctiveness of machine translation, thus, this metric can also be used to evaluate the similarity between two sentences.
* BLEU is another metric which is also used to evaluate the effctiveness of machine translation
* Word error rate is a automatic speech recognition metric. It computes the minimum-edit distance between the ground-truth sentence and the hypothesis sentence of a speech-to-text API
* Longet common substring is literally used to find out the longest common substring between two sentences.
* Levenshtein distance is the number of operations to transfer sentence A to sentence B, if two sentences have short Levenshtein distance, that means they are similar to each other.

### Introduction of my script
`sts_pearson.py` uses `sts-dev.csv` as example to calculate different similarity metrics for each pair, and then calculate the pearson correlation rate between true label and each similarity matrics.

Example usage:
`python sts_pearson.py --sts_data stsbenchmark/sts-dev.csv`

### output
NIST correlation: 0.593
BLEU correlation: 0.433
Word Error Rate correlation: -0.459
Longest common substring correlation: 0.468
Levenshtein distance correlation: -0.175
