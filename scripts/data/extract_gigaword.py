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

    # Download should take 1-2 hours, as servers limit bandwidth
    datasets, metadata = tfds.load("gigaword", 
                               data_dir = args.data_dir, with_info=True,
                               as_supervised=True)
    
    src_tokenizer = MosesTokenizer('en')
    src_nm = MosesPunctuationNormalizer('en')
    tgt_tokenizer = MosesTokenizer('en')
    tgt_nm = MosesPunctuationNormalizer('en')
    
    def proc_line(line, t, n):
        result = t(n(line.decode('utf-8').lower()))
        l = len(result)
        result = ' '.join(result)
        return l, result
        
    dsets = ['train', 'validation', 'test']
    for ds in dsets:
        npds = tfds.as_numpy(datasets)[ds]
        f1 = open(os.path.join(args.data_dir, "gigaword", "src_raw_{}.txt".format(ds)), "w")
        f2 = open(os.path.join(args.data_dir, "gigaword", "tgt_raw_{}.txt".format(ds)), "w")
        for line in npds:
            l1, result1 = proc_line(line[0], src_tokenizer, src_nm)
            l2, result2 = proc_line(line[1], tgt_tokenizer, tgt_nm)
            if ds != 'test' and (l1 > 100 or l2 > 100):
                continue
            print(result1, file=f1)
            print(result2, file=f2)
        f1.close()
        f2.close()