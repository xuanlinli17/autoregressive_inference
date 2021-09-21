from vizseq.scorers.bleu import BLEUScorer
from vizseq.scorers.meteor import METEORScorer
from vizseq.scorers.ter import TERScorer
# vizseq RIBES Scorer implementation seems buggy and gives very low scores
# vizseq BLEU needs to take in ' '.join(tokenized_input) instead of detokenized input, 
# or it gives very low scores compared to sacrebleu
from mosestokenizer import *
import sacrebleu
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    '--files', type=str, nargs='+', help="hypothesis file, reference file")

args = parser.parse_args()
assert len(args.files) == 2

scorers = [BLEUScorer, METEORScorer, TERScorer]
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

# BLEU for shorter sentences for debugging
# tmp1 = []
# tmp2 = []
# for i in range(len(l2)):
#     if len(l2[i].split(' ')) > 30:
#         continue
#     tmp1.append(l1[i])
#     tmp2.append(l2[i])
# l1 = tmp1
# l2 = tmp2

detok = MosesDetokenizer('en')
det_l1 = [detok(x.split(' ')) for x in l1]
det_l2 = [detok(x.split(' ')) for x in l2]

bleu = sacrebleu.corpus_bleu(det_l1, [det_l2])
print("Sacre BLEU", bleu.score)
    
print("VIZSEQ scores:")
for s in scorers:
    if not isinstance(s, METEORScorer):
        print(s, s.score(l1, [l2]))
    else:
        print(s, s.score(det_l1, [det_l2]))