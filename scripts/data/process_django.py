from voi.process.django import process_django
import argparse


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--out_feature_folder', type=str, default='/home/django/')
    parser.add_argument(
        "--data_folder", type=str, default=None)
    parser.add_argument(
        '--vocab_file', type=str, default='/home/django/vocab.txt')
    parser.add_argument(
        '--max_length', type=int, default=70)
    parser.add_argument(
        '--min_word_frequency', type=int, default=1)
    parser.add_argument(
        '--dataset_type', type=str, default='train', choices=['train', 'dev', 'test'])
    args = parser.parse_args()

    process_django(args.out_feature_folder,
                     args.data_folder,
                     args.vocab_file,
                     args.max_length,
                     args.min_word_frequency,
                     args.dataset_type)
