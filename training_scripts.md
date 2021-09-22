Please see `scripts/common_argparse.py` for specific descriptions of arguments.

Here we refer to the "encoder permutation generator" as the "encoder Permutation Transformer" or "PT" for short. The decoder Transformer, on the other hand, is the autoregressive insertion Transformer language model.

Remarks:

(1) As we did not have access to a large computational resource (i.e. we did not have access to industry-level infrastructures to utilize a large amount of GPU memory on multiple machines), we did not exhaustively search for hyperparameters and training schemes. We believe that there exist better training schemes (e.g. using larger batch size, # permutations to sample per data, ratio of decoder learning rate and PT learning rate, etc).

(2) The number of GPUs below are based on the assumption that each GPU has 12 Gigabytes of memory.

(3) Since we save the model snapshots during training, the saved path in `--model_ckpt` will have `_ckpt` or `_iter{iternum}` automatically added before the file extension. For example, if `--model_ckpt ckpt_refinement/nsds_coco_voi.h5`, then the saved model (which contains 4 files) is: `ckpt_refinement/nsds_coco_voi_ckpt.h5` for autoregressive decoder, `ckpt_refinement/nsds_coco_voi_ckpt.pt.h5` for PT, `ckpt_refinement/nsds_coco_voi_ckpt_optim.obj` for optimizer statistics of autoregressive decoder, and `ckpt_refinement/nsds_coco_voi_ckpt_pt_optim.obj` for optimizer statistics of PT. 

During training, when we specify `--model_ckpt ckpt_refinement/nsds_coco_voi.h5`, the above 4 files will be automatically loaded (i.e. `_ckpt` is added before the file extension). However, for evaluation and visualization (see `evaluation_visualization_scripts.md`), `_ckpt` will not be automatically added, so we need to explicitly specify which file to use (e.g. `nsds_coco_voi_ckpt.h5`, `nsds_coco_voi_iter10000.h5`).

#### COCO
```bash
# First, train with token embedding and Transformer encoder shared between the encoder Permutation Transformer and the decoder Transformer language model
nohup bash -c "CUDA_VISIBLE_DEVICES=0,1,2,3 python -u scripts/train.py \
--dataset captioning \
--train_folder ./train2017_tfrecords \
--vocab_file ./train2017_vocab.txt \
--model_ckpt ckpt_refinement/nsds_coco_voi.h5 \
--batch_size 32 \
--num_epochs 4 \
--embedding_size 512 --heads 8 --num_layers 6 \
--first_layer region --final_layer indigo \
--decoder_pretrain -1 \
--decoder_init_lr 0.0001 --pt_init_lr 0.00001 --lr_schedule constant \
--label_smoothing 0.1 \
--order soft \
--kl_coeff 0.5 --action_refinement 4 \
--share_embedding \
--share_encoder \
--pt_pg_type sinkhorn --pt_positional_attention \
--hungarian_op_path {path_to hungarian.so}" > nohup_coco_voi1.txt

# Then train with Transformer encoder and token embedding separated;
# Also add cosine embedding alignment loss between the encoder and decoder models
# We also observe that the performance is slightly harmed if the model is trained for too long, so 
# we save every 10k iterations and evaluate using the best model.
Step1: nohup bash -c "CUDA_VISIBLE_DEVICES=0,1,2,3 python -u scripts/train.py \
--dataset captioning \
--train_folder ~/train2017_tfrecords \
--vocab_file train2017_vocab.txt \
--model_ckpt ckpt_refinement/nsds_coco_voi.h5 \
--batch_size 32 \
--num_epochs 10 \
--embedding_size 512 --heads 8 --num_layers 6 \
--first_layer region --final_layer indigo \
--decoder_pretrain -1 \
--decoder_init_lr 0.00005 --pt_init_lr 0.000005 --lr_schedule constant \
--label_smoothing 0.1 \
--order soft \
--kl_coeff 0.5  --kl_log_linear 0.1 \
--action_refinement 4 \
--pt_pg_type sinkhorn \
--pt_positional_attention \
--embedding_align_coeff 100.0 \
--hungarian_op_path {path_to hungarian.so} \
--save_interval 10000" > nohup_coco_voi2.txt

Step2: Use the same script as above, but set 
`--num_epochs 5 --decoder_init_lr 0.00002 --pt_init_lr 0.000002 --lr_schedule constant --kl_coeff 0.1 --kl_log_linear 0.03`
```

#### Django
```bash
# First, train with token embedding and Transformer encoder shared between the encoder Permutation Transformer and the decoder Transformer language model
nohup bash -c "CUDA_VISIBLE_DEVICES=0,1,2,3 python -u scripts/train.py \
--dataset django \
--train_folder django_data/train \
--vocab_file django_data/djangovocab.txt \
--model_ckpt ckpt_refinement/nsds_django_voi.h5 \
--batch_size 32 \
--num_epochs 50 \
--embedding_size 512 --heads 8 --num_layers 6 \
--first_layer discrete --final_layer indigo \
--decoder_pretrain -1 \
--decoder_init_lr 0.0001 --pt_init_lr 0.00001 --lr_schedule constant \
--label_smoothing 0.1 \
--order soft \
--kl_coeff 0.5 \
--action_refinement 4 \
--share_embedding \
--share_encoder \
--use_ppo \
--pt_pg_type sinkhorn --pt_positional_attention \
--hungarian_op_path {path_to hungarian.so}" > nohup_nsds_django_voi1.txt

# Then train with Transformer encoder and token embedding separated, and kl regularization annealed
nohup bash -c "CUDA_VISIBLE_DEVICES=0,1,2,3 python -u scripts/train.py \
--dataset django \
--train_folder django_data/train \
--vocab_file django_data/djangovocab.txt \
--model_ckpt ckpt_refinement/nsds_django_voi.h5 \
--batch_size 32 \
--num_epochs 250 \
--embedding_size 512 --heads 8 --num_layers 6 \
--first_layer discrete --final_layer indigo \
--decoder_pretrain -1 \
--decoder_init_lr 0.00003 --pt_init_lr 0.000003 --lr_schedule constant \
--label_smoothing 0.1 \
--order soft \
--kl_coeff 0.5 --kl_log_linear 0.002 \
--action_refinement 4 \
--embedding_align_coeff 100.0 \
--pt_special_encoder_block EncoderWithRelativePositionLayer \
--use_ppo \
--pt_pg_type sinkhorn --pt_positional_attention \
--hungarian_op_path {path_to hungarian.so}" > nohup_nsds_django_voi2.txt

# Finally fix the encoder Permutation Transformer and finetune the decoder autoregressive Transformer with larger batch size.
nohup bash -c "CUDA_VISIBLE_DEVICES=0,1 python -u scripts/train.py \
--dataset django \
--train_folder django_data/train \
--vocab_file django_data/djangovocab.txt \
--model_ckpt ckpt_refinement/nsds_django_voi.h5 \
--batch_size 64 \
--num_epochs 50 \
--embedding_size 512 --heads 8 --num_layers 6 \
--first_layer discrete --final_layer indigo \
--decoder_pretrain -1 \
--decoder_init_lr 0.00003 --pt_init_lr 0.0 --lr_schedule linear \
--label_smoothing 0.1 \
--order soft \
--finetune_decoder_transformer \
--pt_special_encoder_block EncoderWithRelativePositionLayer \
--pt_pg_type sinkhorn --pt_positional_attention \
--hungarian_op_path {path_to hungarian.so}" > nohup_nsds_django_voi3.txt
```

#### Gigaword

```bash
# First, train with token embedding and Transformer encoder shared between the encoder Permutation Transformer and the decoder Transformer language model
nohup bash -c "CUDA_VISIBLE_DEVICES=0,1,2,3,4 python -u scripts/train.py \
--dataset gigaword \
--train_folder {path_to_gigaword}/train \
--vocab_file {path_to_gigaword}/gigaword_vocab.txt \
--model_ckpt ckpt_refinement/nsds_gigaword_voi.h5 \
--batch_size 50 \
--num_epochs 3 \
--embedding_size 512 --heads 8 --num_layers 6 \
--first_layer discrete --final_layer indigo \
--decoder_pretrain -1 \
--decoder_init_lr 0.00005 --pt_init_lr 0.000005 --lr_schedule constant \
--label_smoothing 0.1 \
--order soft \
--kl_coeff 0.5 \
--action_refinement 4 \
--share_embedding \
--share_encoder \
--pt_pg_type sinkhorn --pt_positional_attention \
--hungarian_op_path {path_to hungarian.so}" > nohup_nsds_gigaword_voi1.txt

# Then train with Transformer encoder and token embedding separated, and kl regularization annealed
nohup bash -c "CUDA_VISIBLE_DEVICES=0,1,2,3,4 python -u scripts/train.py \
--dataset gigaword \
--train_folder {path_to_gigaword}/train \
--vocab_file {path_to_gigaword}/gigaword_vocab.txt \
--model_ckpt ckpt_refinement/nsds_gigaword_voi.h5 \
--batch_size 50 \
--num_epochs 8 \
--embedding_size 512 --heads 8 --num_layers 6 \
--first_layer discrete --final_layer indigo \
--decoder_pretrain -1 \
--decoder_init_lr 0.00005 --pt_init_lr 0.000005 --lr_schedule constant \
--label_smoothing 0.1 \
--order soft \
--kl_coeff 0.5 --kl_log_linear 0.03 \
--action_refinement 4 \
--embedding_align_coeff 100.0 \
--pt_special_encoder_block EncoderWithRelativePositionLayer \
--pt_pg_type sinkhorn --pt_positional_attention \
--hungarian_op_path {path_to hungarian.so} \
--save_interval 50000" > nohup_nsds_gigaword_voi2.txt

# Finally fix the encoder Permutation Transformer and finetune the decoder autoregressive Transformer with larger batch size.
nohup bash -c "CUDA_VISIBLE_DEVICES=0,1,2,3 python -u scripts/train.py \
--dataset gigaword \
--train_folder {path_to_gigaword}/train \
--vocab_file {path_to_gigaword}/gigaword_vocab.txt \
--model_ckpt ckpt_refinement/nsds_gigaword_voi.h5 \
--batch_size 128 \
--num_epochs 5 \
--embedding_size 512 --heads 8 --num_layers 6 \
--first_layer discrete --final_layer indigo \
--decoder_pretrain -1 \
--decoder_init_lr 0.00007 --pt_init_lr 0.0 --lr_schedule linear \
--label_smoothing 0.1 \
--order soft \
--finetune_decoder_transformer \
--pt_special_encoder_block EncoderWithRelativePositionLayer \
--pt_pg_type sinkhorn --pt_positional_attention \
--hungarian_op_path {path_to hungarian.so}" > nohup_nsds_gigaword_voi3.txt
```

#### WMT

```bash
# First, train with token embedding and Transformer encoder shared between the encoder Permutation Transformer and the decoder Transformer language model
nohup bash -c "CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8 python -u scripts/train.py \
--dataset wmt \
--train_folder {path_to_wmt}/distillation \
--vocab_file ro_en_vocab.txt \
--model_ckpt ckpt_refinement/nsds_wmt_voi.h5 \
--batch_size 54 \
--num_epochs 20 \
--embedding_size 512 --heads 8 --num_layers 6 \
--first_layer discrete --final_layer indigo \
--decoder_pretrain -1 \
--decoder_init_lr 0.00005 --pt_init_lr 0.000005 --lr_schedule constant \
--label_smoothing 0.1 \
--order soft \
--kl_coeff 0.3 \
--action_refinement 3 \
--share_embedding \
--share_encoder \
--pt_pg_type sinkhorn --pt_positional_attention \
--hungarian_op_path {path_to hungarian.so}" > nohup_nsds_wmt_voi1.txt

# Then train with Transformer encoder and token embedding separated, and kl regularization annealed
nohup bash -c "CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8 python -u scripts/train.py \
--dataset wmt \
--train_folder {path_to_wmt}/distillation \
--vocab_file ro_en_vocab.txt \
--model_ckpt ckpt_refinement/nsds_wmt_voi.h5 \
--batch_size 54 \
--num_epochs 40 \
--embedding_size 512 --heads 8 --num_layers 6 \
--first_layer discrete --final_layer indigo \
--decoder_pretrain -1 \
--decoder_init_lr 0.00005 --pt_init_lr 0.000005 --lr_schedule constant \
--label_smoothing 0.1 \
--order soft \
--kl_coeff 0.3 --kl_log_linear 0.01 \
--embedding_align_coeff 10.0 \
--action_refinement 3 \
--pt_special_encoder_block EncoderWithRelativePositionLayer \
--pt_pg_type sinkhorn --pt_positional_attention \
--hungarian_op_path {path_to hungarian.so} \
--save_interval 50000" > nohup_nsds_wmt_voi2.txt

nohup bash -c "CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8 python -u scripts/train.py \
--dataset wmt \
--train_folder {path_to_wmt}/distillation \
--vocab_file ro_en_vocab.txt \
--model_ckpt ckpt_refinement/nsds_wmt_voi.h5 \
--batch_size 54 \
--num_epochs 25 \
--embedding_size 512 --heads 8 --num_layers 6 \
--first_layer discrete --final_layer indigo \
--decoder_pretrain -1 \
--decoder_init_lr 0.00002 --pt_init_lr 0.000002 --lr_schedule constant \
--label_smoothing 0.1 \
--order soft \
--kl_coeff 0.01 --kl_log_linear 0.0007 \
--embedding_align_coeff 10.0 \
--action_refinement 3 \
--pt_special_encoder_block EncoderWithRelativePositionLayer \
--pt_pg_type sinkhorn --pt_positional_attention \
--hungarian_op_path {path_to hungarian.so} \
--save_interval 50000" > nohup_nsds_wmt_voi3.txt

# Finally fix the encoder Permutation Transformer and finetune the decoder autoregressive Transformer with larger batch size.
nohup bash -c "CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -u scripts/train.py \
--dataset wmt \
--train_folder {path_to_wmt}/distillation \
--vocab_file ro_en_vocab.txt \
--model_ckpt ckpt_refinement/nsds_wmt_voi.h5 \
--batch_size 128 \
--num_epochs 20 \
--embedding_size 512 --heads 8 --num_layers 6 \
--first_layer discrete --final_layer indigo \
--decoder_pretrain -1 \
--decoder_init_lr 0.00002 --pt_init_lr 0.0 --lr_schedule linear \
--label_smoothing 0.1 \
--order soft \
--finetune_decoder_transformer \
--action_refinement 1 \
--pt_special_encoder_block EncoderWithRelativePositionLayer \
--pt_pg_type sinkhorn --pt_positional_attention \
--hungarian_op_path {path_to hungarian.so} \
--save_interval 50000" > nohup_nsds_wmt_voi4.txt
```