from collections import defaultdict
import tensorflow as tf
import os
import argparse
import json
import tensorflow_datasets as tfds
from mosestokenizer import *


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--language_pair', nargs='+', type=str)
    parser.add_argument("--data_dir",
                        type=str,
                        default=None)
    parser.add_argument('--truecase', action='store_true')
    args = parser.parse_args()
    
    assert len(args.language_pair) == 3
    yr, src, tgt = args.language_pair

    # Download should take 1-2 hours, as servers limit bandwidth
    datasets, metadata = tfds.load("wmt{}_translate/{}-{}".format(yr, src, tgt), 
                               data_dir = args.data_dir, with_info=True,
                               as_supervised=True)
    
    src_tokenizer = MosesTokenizer(args.language_pair[1])
    src_nm = MosesPunctuationNormalizer(args.language_pair[1])
    tgt_tokenizer = MosesTokenizer(args.language_pair[2])
    tgt_nm = MosesPunctuationNormalizer(args.language_pair[2])
    
    def proc_line(line, t, n):
        if args.truecase:
            result = t(n(line.decode('utf-8')))
        else:
            result = t(n(line.decode('utf-8').lower()))
        l = len(result)
        result = ' '.join(result)
        return l, result
        
    dsets = ['train', 'validation', 'test']
    for ds in dsets:
        npds = tfds.as_numpy(datasets)[ds]
        if args.truecase:
            f1 = open(os.path.join(args.data_dir, "wmt{}_translate/{}-{}".format(yr, src, tgt), "src_truecase_{}.txt".format(ds)), "w")
            f2 = open(os.path.join(args.data_dir, "wmt{}_translate/{}-{}".format(yr, src, tgt), "tgt_truecase_{}.txt".format(ds)), "w")
        else:
            f1 = open(os.path.join(args.data_dir, "wmt{}_translate/{}-{}".format(yr, src, tgt), "src_raw_{}.txt".format(ds)), "w")
            f2 = open(os.path.join(args.data_dir, "wmt{}_translate/{}-{}".format(yr, src, tgt), "tgt_raw_{}.txt".format(ds)), "w")
            
        for line in npds:
            l1, result1 = proc_line(line[0], src_tokenizer, src_nm)
            l2, result2 = proc_line(line[1], tgt_tokenizer, tgt_nm)
            if ds != 'test' and (l1 > 100 or l2 > 100):
                continue
            print(result1, file=f1)
            print(result2, file=f2)
        f1.close()
        f2.close()