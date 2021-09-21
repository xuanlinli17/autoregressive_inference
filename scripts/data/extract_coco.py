from collections import defaultdict
import tensorflow as tf
import os
import argparse
import json


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--out_caption_folder",
                        type=str,
                        default="captions_train2017")
    parser.add_argument("--annotations_file",
                        type=str,
                        default="annotations/captions_train2017.json")
    args = parser.parse_args()

    tf.io.gfile.makedirs(args.out_caption_folder)
    with tf.io.gfile.GFile(args.annotations_file, "r") as f:
        data = json.load(f)

    ids_to_captions = defaultdict(list)
    ids_to_name = defaultdict(str)

    for x in data["images"]:
        ids_to_name[x["id"]] = x["file_name"]

    for x in data["annotations"]:
        ids_to_captions[x["image_id"]].append(x["caption"])

    for image_id, captions in ids_to_captions.items():

        text_file_name = os.path.join(
            args.out_caption_folder,
            ids_to_name[image_id][:-3] + "txt")

        with tf.io.gfile.GFile(text_file_name, "w") as f:
            f.write("\n".join(captions))
