from collections import defaultdict
import tensorflow as tf
import os
import argparse
import json
import tensorflow_datasets as tfds
import nltk
import re

QUOTED_STRING_RE = re.compile(r"(?P<quote>['\"])(?P<string>.*?)(?<!\\)(?P=quote)")

#https://github.com/pcyin/NL2code/blob/f9732f1f5caafa73a0f767cc4f5ce9f5961c46d6/dataset.py
def canonicalize_query(query):
    """
    canonicalize the query, replace strings to a special place holder
    """
    str_count = 0
    str_map = dict()

    matches = QUOTED_STRING_RE.findall(query)
    # de-duplicate
    cur_replaced_strs = set()
    for match in matches:
        # If one or more groups are present in the pattern,
        # it returns a list of groups
        quote = match[0]
        str_literal = quote + match[1] + quote

        if str_literal in cur_replaced_strs:
            continue

        # FIXME: substitute the ' % s ' with
        if str_literal in ['\'%s\'', '\"%s\"']:
            continue

        str_repr = '_STR:%d_' % str_count
        str_map[str_literal] = str_repr

        query = query.replace(str_literal, str_repr)

        str_count += 1
        cur_replaced_strs.add(str_literal)

    # tokenize
    query_tokens = nltk.word_tokenize(query)

    new_query_tokens = []
    # break up function calls like foo.bar.func
    for token in query_tokens:
        new_query_tokens.append(token)
        i = token.find('.')
        if 0 < i < len(token) - 1:
            new_tokens = ['['] + token.replace('.', ' . ').split(' ') + [']']
            new_query_tokens.extend(new_tokens)

    query = ' '.join(new_query_tokens)

    return query, str_map

#https://github.com/pcyin/NL2code/blob/f9732f1f5caafa73a0f767cc4f5ce9f5961c46d6/dataset.py
def canonicalize_example(query, code):
    #from lang.py.parse import parse_raw, parse_tree_to_python_ast, canonicalize_code as make_it_compilable
    import astor, ast

    canonical_query, str_map = canonicalize_query(query)
    canonical_code = code

    for str_literal, str_repr in str_map.items():
        canonical_code = canonical_code.replace(str_literal, '\'' + str_repr + '\'')

    #canonical_code = make_it_compilable(canonical_code)
    #query_tokens = canonical_query.split(' ')

    return canonical_query, canonical_code, str_map

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",
                        type=str,
                        default=None)
    args = parser.parse_args()
    
    tokenizer = nltk.tokenize.WhitespaceTokenizer()
    
    src_file = open(os.path.join(args.data_dir, 'all.anno'), "r")
    tgt_file = open(os.path.join(args.data_dir, 'all.code'), "r")
    src_lines = []
    tgt_lines = []
    for line in src_file:
        src_lines.append(re.sub(' +',' ',line.strip()))
    for line in tgt_file:
        tgt_lines.append(re.sub(' +',' ',line.strip()))
    
    src_train_out = open(os.path.join(args.data_dir, 'src_raw_train.txt'), "w")
    src_dev_out = open(os.path.join(args.data_dir, 'src_raw_dev.txt'), "w")
    src_test_out = open(os.path.join(args.data_dir, 'src_raw_test.txt'), "w")
    tgt_train_out = open(os.path.join(args.data_dir, 'tgt_raw_train.txt'), "w")
    tgt_dev_out = open(os.path.join(args.data_dir, 'tgt_raw_dev.txt'), "w")
    tgt_test_out = open(os.path.join(args.data_dir, 'tgt_raw_test.txt'), "w")  

    def parse_file(src_lines, tgt_lines, src_f, tgt_f, test=False):
        src_lines = [x.strip() for x in src_lines]
        tgt_lines = [x.strip() for x in tgt_lines]
        assert len(src_lines) == len(tgt_lines)
        for idx in range(len(src_lines)):
            s, t, _ = canonicalize_example(src_lines[idx], tgt_lines[idx])
            if test or (len(s.split(' ')) <= 70 and len(t.split(' ')) <= 50):
                print(s, file=src_f)
                print(t, file=tgt_f)
    
    parse_file(src_lines[:-2805], tgt_lines[:-2805], src_train_out, tgt_train_out)
    parse_file(src_lines[-2805:-1805], tgt_lines[-2805:-1805], src_dev_out, tgt_dev_out)
    parse_file(src_lines[-1805:], tgt_lines[-1805:], src_test_out, tgt_test_out, test=True)
    
    src_train_out.close()
    src_test_out.close()
    tgt_train_out.close()
    tgt_test_out.close()