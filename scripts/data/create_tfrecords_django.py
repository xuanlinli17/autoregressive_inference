from voi.data.tfrecords_django import create_tfrecord_django
import argparse


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--out_tfrecord_folder', type=str, default='/home/django/')
    parser.add_argument(
        '--feature_folder', type=str, default='/home/django/')
    parser.add_argument(
        '--dataset_type', type=str, default='test', choices=['train', 'dev', 'test'])    
    parser.add_argument(
        '--samples_per_shard', type=int, default=4096)
    args = parser.parse_args()

    create_tfrecord_django(args.out_tfrecord_folder,
                    args.feature_folder,
                    args.dataset_type,
                    args.samples_per_shard)