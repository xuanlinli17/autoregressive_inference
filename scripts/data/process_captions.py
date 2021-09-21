from voi.process.captions import process_captions
import argparse


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--out_feature_folder', type=str, default='captions_features')
    parser.add_argument(
        '--in_folder', type=str, default='captions')
    parser.add_argument(
        '--tagger_file', type=str, default='tagger.pkl')
    parser.add_argument(
        '--vocab_file', type=str, default='vocab.txt')
    parser.add_argument(
        '--max_length', type=int, default=20)
    parser.add_argument(
        '--min_word_frequency', type=int, default=5)
    parser.add_argument(
        '--dataset_type', type=str, default='train')
    args = parser.parse_args()

    process_captions(args.out_feature_folder,
                     args.in_folder,
                     args.tagger_file,
                     args.vocab_file,
                     args.max_length,
                     args.min_word_frequency)
