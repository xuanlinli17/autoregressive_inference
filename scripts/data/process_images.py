from voi.process.images import process_images
import argparse


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--out_feature_folder', type=str, default='images_features')
    parser.add_argument(
        '--in_folder', type=str, default='images')
    parser.add_argument(
        '--batch_size', type=int, default=32)
    args = parser.parse_args()

    process_images(args.out_feature_folder,
                   args.in_folder,
                   args.batch_size)
