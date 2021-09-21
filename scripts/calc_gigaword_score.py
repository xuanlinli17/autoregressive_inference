from vizseq.scorers.rouge import Rouge1Scorer, Rouge2Scorer, RougeLScorer
# vizseq RIBES Scorer implementation seems buggy and gives very low scores
# vizseq BLEU needs to take in ' '.join(tokenized_input) instead of detokenized input, 
# or it gives very low scores compared to sacrebleu
from mosestokenizer import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    '--files', type=str, nargs='+', help="hypothesis file, reference file")

args = parser.parse_args()
assert len(args.files) == 2

scorers = [Rouge1Scorer, Rouge2Scorer, RougeLScorer]
scorers = [s(corpus_level=True, sent_level=False, n_workers=2, verbose=False, extra_args=None)
                   for s in scorers]
l1 = []
l2 = []
f1 = open(args.files[0], "r")
f2 = open(args.files[1], "r")
    
for line in f1:
    line = line.strip()
    l1.append(line)

for line in f2:
    line = line.strip()
    l2.append(line)
    
print("VIZSEQ scores:")
for s in scorers:
    print(s, s.score(l1, [l2])[0]*100)