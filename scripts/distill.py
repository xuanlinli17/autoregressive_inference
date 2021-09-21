from voi.core.distill import distill_dataset
from voi.nn.transformer import Transformer
from voi.process.captions import Vocabulary
from voi.scripts.common_argparse import add_common_arguments
import tensorflow as tf
import argparse
from common_argparse import add_common_arguments
from build_model_utils import build_model

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_common_arguments(parser) 
    args = parser.parse_args()
    
    # distillation unavailable for captioning tasks since there are
    # already multiple references
    assert args.dataset != 'captioning'
    
    model, vocab, strategy = build_model(args, return_pt=False, no_dropout=True)

    distill_dataset(args.train_folder,
                    args.batch_size,
                    args.beam_size,
                    model,
                    args.model_ckpt,
                    vocab,
                    args.dataset,
                    strategy,
                    args.distillation_save_path)
