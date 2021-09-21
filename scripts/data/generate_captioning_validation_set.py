import tensorflow as tf
import argparse
import os
import shutil
import random

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--in_image_folder",
                        type=str,
                        default="flickr30k")
    parser.add_argument("--in_caption_folder",
                        type=str,
                        default="captions_flickr30k")
    parser.add_argument("--out_image_folder",
                        type=str,
                        default="flickr30k_val")
    parser.add_argument("--out_caption_folder",
                        type=str,
                        default="captions_flickr30k_val")
    parser.add_argument("--k",
                        type=int,
                        default=1000)
    args = parser.parse_args()

    caption_files = tf.io.gfile.glob(
        os.path.join(args.in_caption_folder, "*.txt"))

    tf.io.gfile.makedirs(args.out_image_folder)
    tf.io.gfile.makedirs(args.out_caption_folder)

    for caption_file_name in random.sample(caption_files, args.k):

        shutil.move(caption_file_name, os.path.join(
            args.out_caption_folder, os.path.basename(caption_file_name)))

        image_file_name = os.path.join(
            args.in_image_folder, os.path.basename(
                caption_file_name).replace(".txt", ".jpg"))

        shutil.move(image_file_name, os.path.join(
            args.out_image_folder, os.path.basename(image_file_name)))
