from voi.core.visualize_generation import visualize_generation
from voi.nn.transformer import Transformer
from voi.nn.permutation_transformer import PermutationTransformer
from voi.process.captions import Vocabulary
import tensorflow as tf
import argparse
from common_argparse import add_common_arguments
from build_model_utils import build_model

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_common_arguments(parser)
    args = parser.parse_args()
    print(args)

    model, order, vocab, tags_vocab, strategy = build_model(args, return_pt=True, no_dropout=True)

    visualize_generation(
        args.validate_folder,
        args.caption_ref_folder,
        args.batch_size,
        args.beam_size,
        model,
        args.model_ckpt,
        order if args.order == 'soft' else args.order,
        vocab,
        tags_vocab,
        strategy,
        args.dataset)
