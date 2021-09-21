from voi.data.tagger import create_tagger
import argparse


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--out_tagger_file', type=str, default='tagger.pkl')
    args = parser.parse_args()

    create_tagger(args.out_tagger_file)
