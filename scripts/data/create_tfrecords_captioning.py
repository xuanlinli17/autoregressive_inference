from voi.data.tfrecords_captioning import create_tfrecord
import argparse


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--out_tfrecord_folder', type=str, default='tfrecords')
    parser.add_argument(
        '--caption_folder', type=str, default='captions_features')
    parser.add_argument(
        '--image_folder', type=str, default='images_features')
    parser.add_argument(
        '--samples_per_shard', type=int, default=4096)
    args = parser.parse_args()

    create_tfrecord(args.out_tfrecord_folder,
                    args.caption_folder,
                    args.image_folder,
                    args.samples_per_shard)
