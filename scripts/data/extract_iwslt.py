from collections import defaultdict
import tensorflow as tf
import os
import argparse
import json
import tensorflow_datasets as tfds
from mosestokenizer import *


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",
                        type=str,
                        default=None)
    args = parser.parse_args()
    
    src_tokenizer = MosesTokenizer('de')
    src_nm = MosesPunctuationNormalizer('de')
    tgt_tokenizer = MosesTokenizer('en')
    tgt_nm = MosesPunctuationNormalizer('en')
    
    def proc_line(line, t, n):
        result = t(n(line.lower()))
        l = len(result)
        result = ' '.join(result)
        return l, result
        
    dsets = ['train', 'dev', 'test']
    for ds in dsets:
        orig_file = open(os.path.join(args.data_dir, "ted-talk-iwslt2014.{}.sents.tsv".format(ds)), "r")
        f1 = open(os.path.join(args.data_dir, "src_raw_{}.txt".format(ds)), "w")
        f2 = open(os.path.join(args.data_dir, "tgt_raw_{}.txt".format(ds)), "w")
        for line in orig_file:
            tmp = line.split("\t")
            assert len(tmp) == 4
            _, l0, l1, l2 = tmp
            l0 = l0.strip()
            l1 = l1.strip()
            l2 = l2.strip()
            l0, result0 = proc_line(l0, tgt_tokenizer, tgt_nm)
            l1, result1 = proc_line(l1, src_tokenizer, src_nm)
            l2, result2 = proc_line(l2, tgt_tokenizer, tgt_nm)
            if ds != 'test' and (l1 > 100 or l2 > 100):
                continue
            result1 = ' '.join(result0.split(' ') + ['|'] + result1.split(' '))
            print(result1, file=f1)
            print(result2, file=f2)
        f1.close()
        f2.close()