# Discovering Non-monotonic Autoregressive Orderings with Variational Inference

## Description

This package contains the source code implementation of the paper "Discovering Non-monotonic Autoregressive Orderings with Variational Inference" [arxiv]().

Inferring good generation orders in natural sequences is challenging. In our main contribution, we propose Variational Order Inference (VOI), which can be efficiently trained to discover autoregressive sequence generation orders in a data driven way without a domain-specific prior.

In VOI, the encoder permutation generator generates non-monotonic autoregressive orders as the latent variable, and the decoder autoregressive (language) model maximizes the joint probability of generating the target sequence under these non-monotonic orders. In conditional text generation tasks, the encoder is implemented as Transformer with non-causal attention, and the decoder is implemented as Transformer-InDIGO (Gu et al., 2019) which generates target sequences through insertion.

![](readme_imgs/high_level.PNG)
![](readme_imgs/computation_graph.PNG)
![](readme_imgs/arch.PNG)

## Installation

To install this package, first download the package from github, then install it using pip. For CUDA 10.1 (as configured in `setup.py`), the package versions are Tensorflow 2.3 and PyTorch 1.5, with their corresponding `tensorflow_probability` and `torchvision` versions. For CUDA 11.0, you may need to change the package versions in `setup.py` to be `tensorflow==2.4`, `torch==1.6`, `tensorflow_probability==0.12.1`, and `torchvision==0.7.0`.

```bash
git clone git@github.com:{name/voi}
pip install -e voi
```

Install helper packages for word tokenization and part of speech tagging. Enter the following statements into the python interpreter where you have installed our package.

```python
import nltk
nltk.download('punkt')
nltk.download('brown')
nltk.download('universal_tagset')
```

Install `nlg-eval` that contains several helpful metrics for evaluating image captioning. Tasks other than captioning are evaluated through the `vizseq` package we already installed through `setup.py`.

```bash
pip install git+https://github.com/Maluuba/nlg-eval.git@master
nlg-eval --setup
```

Clone `wmt16-scripts` for machine translation preprocessing.

```
git clone https://github.com/rsennrich/wmt16-scripts
```

## Configure tensorflow-hungarian

During training, one process of order inference is to obtain permutation matrices from doubly stochastic matrices. This is accomplished through the Hungarian algorithm. Since `tf.py_function` only allows one gpu to run the function at any time, multi-gpu training is very slow if we use `scipy.optimize.linear_sum_assignment` (which requires wrapping it with `tf.py_function` to call). Therefore, we use a pre-written Hungarian-op script and compile it through g++ into dynamic library. During runtime, we can import the dynamic library using tensorflow api. This leads to much faster distributed training.

```bash
git clone https://github.com/brandontrabucco/tensorflow-hungarian
cd tensorflow-hungarian
make hungarian_op
```
If you encounter `fatal error: third_party/gpus/cuda/include/cuda_fp16.h: No such file or directory`, this could be resolved via [link](https://github.com/tensorflow/tensorflow/issues/31912#issuecomment-547475301). The generated op could be found in `tensorflow-hungarian/tensorflow_hungarian/python/ops/_hungarian_ops.so`

Alternatively, we could also generate the op from the repo `munkres-tensorflow`. 
```bash
git clone https://github.com/mbaradad/munkres-tensorflow
TF_CFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))') )
TF_LFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))') )
g++ -std=c++11 -shared munkres-tensorflow/hungarian.cc -o hungarian.so -fPIC ${TF_CFLAGS[@]} ${TF_LFLAGS[@]} -O2
```
However, this function requires all entries in a matrix to be different (otherwise some strange behaviors will occur), so we also need to uncomment the line `sample_permu = sample_permu * 1000 + tf.random.normal(tf.shape(sample_permu)) * 1e-7` in `voi/nn/layers/permutation_sinkhorn.py`

## Setup

### Captioning

In this section, we will walk you through how to create a training dataset, using COCO 2017 as an example. In the first step, download COCO 2017. Place the annotations at `~/annotations` and the images at `~/train2017` and `~/val2017` for the training and validation set respectively.

Create a part of speech tagger first. This information is used to visualize the generation orders of captions learnt by our model, and is not used during training.

```bash
cd {folder_with_voi_repo}
python scripts/data/create_tagger.py --out_tagger_file tagger.pkl
```

Extract COCO 2017 into a format compatible with our package. There are several arguments that you can specify to control how the dataset is processed. You may leave all arguments as default except `out_caption_folder` and `annotations_file`.

```bash
python scripts/data/extract_coco.py --out_caption_folder ~/captions_train2017 --annotations_file ~/annotations/captions_train2017.json
python scripts/data/extract_coco.py --out_caption_folder ~/captions_val2017 --annotations_file ~/annotations/captions_val2017.json
```

Process the COCO 2017 captions and extract integer features on which to train a non sequential model. There are again several arguments that you can specify to control how the captions are processed. You may leave all arguments as default except `out_feature_folder` and `in_folder`, which depend on where you extracted the COCO dataset in the previous step. Note that if `vocab_file` doesn't exist before, it will be automatically generated. Since we have provided the `train2017_vocab.txt` we used to train our model, this vocab file will be directly loaded to create integer representations of tokens.

```bash
python scripts/data/process_captions.py --out_feature_folder ~/captions_train2017_features --in_folder ~/captions_train2017 \
--tagger_file tagger.pkl --vocab_file train2017_vocab.txt --min_word_frequency 5 --max_length 100
python scripts/data/process_captions.py --out_feature_folder ~/captions_val2017_features --in_folder ~/captions_val2017 \
--tagger_file tagger.pkl --vocab_file train2017_vocab.txt --max_length 100
```

Process images from the COCO 2017 dataset and extract features using a pretrained Faster RCNN FPN backbone from pytorch checkpoint. Note this script will distribute inference across all visible GPUs on your system. There are several arguments you can specify, which you may leave as default except `out_feature_folder` and `in_folder`, which depend on where you extracted the COCO dataset.

```bash
python scripts/data/process_images.py --out_feature_folder ~/train2017_features --in_folder ~/train2017 --batch_size 4
python scripts/data/process_images.py --out_feature_folder ~/val2017_features --in_folder ~/val2017 --batch_size 4
```

Finally, convert the processed features into a TFRecord format for efficient training. Record where you have extracted the COCO dataset in the previous steps and specify `out_tfrecord_folder`, `caption_folder` and `image_folder` at the minimum.

```bash
python scripts/data/create_tfrecords_captioning.py --out_tfrecord_folder ~/train2017_tfrecords \
--caption_folder ~/captions_train2017_features --image_folder ~/train2017_features --samples_per_shard 4096
python scripts/data/create_tfrecords_captioning.py --out_tfrecord_folder ~/val2017_tfrecords \
--caption_folder ~/captions_val2017_features --image_folder ~/val2017_features --samples_per_shard 4096
```

### Django

For convenience, we ran the script from [NL2code](https://github.com/pcyin/NL2code) to extract the cleaned dataset from [drive](https://drive.google.com/drive/folders/0B14lJ2VVvtmJWEQ5RlFjQUY2Vzg) and place them in `django_data`. Alternatively, you may download raw data from [ase15-django](https://github.com/odashi/ase15-django-dataset) and run `python scripts/data/extract_django.py --data_dir {path to all.anno and all.code)`

```bash
cd {folder_with_voi_repo}
CUDA_VISIBLE_DEVICES=0 python scripts/data/process_django.py --data_folder ./django_data \
--vocab_file ./django_data/djangovocab.txt --dataset_type train/dev/test \
--out_feature_folder ./django_data
CUDA_VISIBLE_DEVICES=0 python scripts/data/create_tfrecords_django.py --out_tfrecord_folder ./django_data \
--dataset_type train/dev/test --feature_folder ./django_data
```

### Gigaword

First, extract the dataset and learn byte-pair encoding.

```bash
cd {folder_with_voi_repo}
CUDA_VISIBLE_DEVICES=0 python scripts/data/extract_gigaword.py --data_dir {dataroot}
cd {dataroot}/gigaword
subword-nmt learn-joint-bpe-and-vocab --input src_raw_train.txt tgt_raw_train.txt -s 32000 -o joint_bpe.code --write-vocabulary src_vocab.txt tgt_vocab.txt
subword-nmt apply-bpe -c joint_bpe.code --vocabulary src_vocab.txt --vocabulary-threshold 50 < src_raw_train.txt > src_train.BPE.txt
subword-nmt apply-bpe -c joint_bpe.code --vocabulary src_vocab.txt --vocabulary-threshold 50 < src_raw_validation.txt > src_validation.BPE.txt
subword-nmt apply-bpe -c joint_bpe.code --vocabulary src_vocab.txt --vocabulary-threshold 50 < src_raw_test.txt > src_test.BPE.txt
subword-nmt apply-bpe -c joint_bpe.code --vocabulary tgt_vocab.txt --vocabulary-threshold 50 < tgt_raw_train.txt > tgt_train.BPE.txt
subword-nmt apply-bpe -c joint_bpe.code --vocabulary tgt_vocab.txt --vocabulary-threshold 50 < tgt_raw_validation.txt > tgt_validation.BPE.txt
subword-nmt apply-bpe -c joint_bpe.code --vocabulary tgt_vocab.txt --vocabulary-threshold 50 < tgt_raw_test.txt > tgt_test.BPE.txt
```

Then, generate the vocab file. Alternately you may use the `gigaword_vocab.txt` provided in our repo, which we used to train our model. To do this, set the `vocab_file` argument to be `{voi_repo}/gigaword_vocab.txt`. 

```bash
cd {folder_with_voi_repo}
CUDA_VISIBLE_DEVICES=0 python scripts/data/process_gigaword.py --out_feature_folder {dataroot}/gigaword \
--data_folder {dataroot}/gigaword --vocab_file {dataroot}/gigaword/gigaword_vocab.txt \
--dataset_type train/validation/test
```

Finally, generate the train/validation/test tfrecords files.
```bash
CUDA_VISIBLE_DEVICES=0 python scripts/data/create_tfrecords_gigaword.py --out_tfrecord_folder {dataroot}/gigaword \
--feature_folder {dataroot}/gigaword --samples_per_shard 4096 --dataset_type train/validation/test
```


### WMT

Here, we use WMT16 Ro-En as an example.

First extract the dataset and learn byte-pair encoding.
```bash
cd {folder_with_voi_repo}
CUDA_VISIBLE_DEVICES=0 python scripts/data/extract_wmt.py --language_pair 16 ro en --data_dir {dataroot}
cd {dataroot}/wmt16_translate/ro-en
subword-nmt learn-joint-bpe-and-vocab --input src_raw_train.txt tgt_raw_train.txt -s 32000 -o joint_bpe.code --write-vocabulary src_vocab.txt tgt_vocab.txt
subword-nmt apply-bpe -c joint_bpe.code --vocabulary src_vocab.txt --vocabulary-threshold 50 < src_raw_train.txt > src_train.BPE.txt
subword-nmt apply-bpe -c joint_bpe.code --vocabulary src_vocab.txt --vocabulary-threshold 50 < src_raw_validation.txt > src_validation.BPE.txt
subword-nmt apply-bpe -c joint_bpe.code --vocabulary src_vocab.txt --vocabulary-threshold 50 < src_raw_test.txt > src_test.BPE.txt
subword-nmt apply-bpe -c joint_bpe.code --vocabulary tgt_vocab.txt --vocabulary-threshold 50 < tgt_raw_train.txt > tgt_train.BPE.txt
subword-nmt apply-bpe -c joint_bpe.code --vocabulary tgt_vocab.txt --vocabulary-threshold 50 < tgt_raw_validation.txt > tgt_validation.BPE.txt
subword-nmt apply-bpe -c joint_bpe.code --vocabulary tgt_vocab.txt --vocabulary-threshold 50 < tgt_raw_test.txt > tgt_test.BPE.txt
```

Extract corpus with truecase to train the truecaser, which is used for evaluation.

```bash
cd {repo_with_mosesdecoder}
git clone https://github.com/moses-smt/mosesdecoder
cd {folder_with_voi_repo}
CUDA_VISIBLE_DEVICES=0 python scripts/data/extract_wmt.py --language_pair 16 ro en --data_dir {dataroot} --truecase
{path_to_mosesdecoder}/scripts/recaser/train-truecaser.perl -corpus {dataroot}/wmt16_translate/ro-en/src_truecase_train.txt -model {dataroot}/wmt16_translate/ro-en/truecase-model.ro
{path_to_mosesdecoder}/scripts/recaser/train-truecaser.perl -corpus {dataroot}/wmt16_translate/ro-en/tgt_truecase_train.txt -model {dataroot}/wmt16_translate/ro-en/truecase-model.en
```

Remove the diacritics of Romanian:
```bash
git clone https://github.com/rsennrich/wmt16-scripts
python wmt16-scripts/preprocess/remove-diacritics.py < src_train.BPE.txt > src_train.BPE.txt
python wmt16-scripts/preprocess/remove-diacritics.py < src_validation.BPE.txt > src_validation.BPE.txt
python wmt16-scripts/preprocess/remove-diacritics.py < src_test.BPE.txt > src_test.BPE.txt
```

Generate the vocab file (joint vocab for the source and target languages). Since we forgot to remove the diacritics during our initial experiments and we appended all missing vocabs in the diacritics-removed corpus afterwards, the vocab file we used to train our model is slightly different from the one generated through the scripts below, so we have uploaded the vocab file we used to train our model as `ro_en_vocab.txt`. To use this vocab file, set the `vocab_file` argument to be `{voi_repo}/ro_en_vocab.txt`
```
cd {folder_with_voi_repo}
CUDA_VISIBLE_DEVICES=0 python scripts/data/process_wmt.py --out_feature_folder {dataroot}/wmt16_translate/ro-en \
--data_folder {dataroot}/wmt16_translate/ro-en --vocab_file {dataroot}/wmt16_translate/ro_en_vocab.txt \
--dataset_type train/validation/test
```

Finally, generate the train/validation/test tfrecords files.
```
CUDA_VISIBLE_DEVICES=0 python scripts/data/create_tfrecords_wmt.py --out_tfrecord_folder {dataroot}/wmt16_translate/ro-en \
--feature_folder {dataroot}/wmt16_translate/ro-en --samples_per_shard 4096 --dataset_type train/validation/test
```

**Note**
In practice, training with the sequence-level distillation dataset ([Link](https://arxiv.org/pdf/1606.07947.pdf)) generated using the L2R model with beam size 5 leads to about 2 BLEU improvement on WMT16 Ro-En, intuitively because the target sequences in this new dataset are more consistent. We release the this distilled dataset at {placeholder}. To train on this dataset, replace `src_train.BPE.txt` and `tgt_train.BPE.txt` accordingly before running `create_tfrecords_wmt.py`. Training on this distilled dataset obtains very similar ordering observations (i.e. the model generates all descriptive tokens before generating the auxillary tokens) compared to training on the original dataset.

## Training

You may train several kinds of models using our framework. For example, you can replicate our results and train a non-sequential VOI model using the following command in the terminal.

Run `python scripts/train.py --help` for specific details about the commands.

Remarks:

(1) Due to limited computational budget, the hyperparameters and training schemes of our VOI model were not tuned carefully, but we still got strong results. We believe that there exist better training schemes (e.g. using larger batch size, # permutations to sample per data, ratio of decoder learning rate and PT learning rate, etc). 

(2) For some datasets (COCO, WMT16 RO-EN), training the decoder Transformer for too long after PT has converged to a single permutation per data **while keeping the constant learning rate** could lead to overfitting and a degradation of performance. Thus we save the model every 10k iterations for COCO and 50k iterations for WMT. Finetuning with fixed PT and linear learning rate decay, along with the evaluation afterwards, are done using the best model.

(3) We find that, after PT has converged to a single permutation per data, finetuning the decoder Transformer with larger batch size, fixed PT, and linear learning rate decay can improve the performance of Gigaword and WMT (by about 2.0 ROUGE / 1.5 BLEU). The performance slightly improves for Django, but is harmed for COCO (in fact, we observe that if the baseline fixed ordering models, e.g. L2R, are trained for too long on COCO, then the performance also drops). However, for COCO and Django, finetuning is not necessary to outperform fixed orderings like L2R.

(4) `embedding_align_coeff` adds a cosine alignment loss between the PT's vocab and decoder Transformer's vocab to the loss of PT. We found this helpful in Gigaword and WMT to encourage PT to learn better orderings. For COCO and Django we didn't add this loss when we trained our models, but this could also improve our results.

(5) The number of GPUs below are based on the assumption that each GPU has 12 Gigabytes of memory.

(6) Since we save the model snapshots during training, the saved `model_ckpt` will have `ckpt` or `iter{iternum}` automatically appended to the names. This also means that when we run the evaluation scripts, these strings will not be automatically appended to the input file names.

#### COCO
```bash
# Train with embedding shared between encoder Permutation Transformer and decoder Transformer first
nohup bash -c "CUDA_VISIBLE_DEVICES=0,1,2,3 python -u scripts/train.py \
--dataset captioning \
--train_folder ~/train2017_tfrecords \
--vocab_file train2017_vocab.txt \
--model_ckpt ckpt_refinement/nsds_coco_voi.h5 \
--batch_size 32 \
--num_epochs 4 \
--embedding_size 512 --heads 8 --num_layers 6 \
--first_layer region --final_layer indigo \
--decoder_pretrain -1 \
--decoder_init_lr 0.0001 --pt_init_lr 0.00001 --lr_schedule constant \
--label_smoothing 0.1 \
--order soft --policy_gradient without_bvn \
--kl_coeff 0.5 --action_refinement 4 --share_embedding True \
--pt_pg_type sinkhorn --pt_relative_embedding False --pt_positional_attention True \
--reward_std False \
--hungarian_op_path {path_to hungarian.so}" > nohup_coco_voi1.txt

# Then train with embedding separated; 
# if the sampled permutations start to become identical (i.e. the encoder converges),
# then we can further half the learning rate.
# Note that we could also add embedding alignment loss like in gigaword. We didn't 
# try this when we ran the experiments. This could further improve the results.
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
--order soft --policy_gradient without_bvn \
--kl_coeff 0.5  --kl_log_linear 0.1 \
--action_refinement 4 \
--share_embedding False \
--pt_pg_type sinkhorn --pt_relative_embedding False --pt_positional_attention True \
--reward_std False \
--hungarian_op_path {path_to hungarian.so} \
--save_interval 10000" > nohup_coco_voi2.txt

Step2: Use the same script as above, but set --num_epochs 5 --decoder_init_lr 0.00002 --pt_init_lr 0.000002 --lr_schedule constant --kl_coeff 0.1 --kl_log_linear 0.05
```

#### Django
```bash
# Train with embedding shared between encoder Permutation Transformer and decoder Transformer first
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
--order soft --policy_gradient without_bvn \
--kl_coeff 0.5 \
--action_refinement 4 \
--share_embedding True \
--use_ppo \
--pt_pg_type sinkhorn --pt_relative_embedding False --pt_positional_attention True \
--reward_std False \
--hungarian_op_path {path_to hungarian.so}" > nohup_nsds_django_voi1.txt

# Then train with embedding separated and kl regularization annealed
# Note that we could also add embedding alignment loss like in gigaword. We didn't 
# try this when we ran the experiments. This could further improve the results.
nohup bash -c "CUDA_VISIBLE_DEVICES=0,1,2,3 python -u scripts/train.py \
--dataset django \
--train_folder django_data/train \
--vocab_file django_data/djangovocab.txt \
--model_ckpt ckpt_refinement/nsds_django_voi.h5 \
--batch_size 32 \
--num_epochs 300 \
--embedding_size 512 --heads 8 --num_layers 6 \
--first_layer discrete --final_layer indigo \
--decoder_pretrain -1 \
--decoder_init_lr 0.00003 --pt_init_lr 0.000003 --lr_schedule constant \
--label_smoothing 0.1 \
--order soft --policy_gradient without_bvn \
--kl_coeff 0.5 --kl_log_linear 0.002 \
--action_refinement 4 \
--share_embedding False \
--use_ppo \
--pt_pg_type sinkhorn --pt_relative_embedding False --pt_positional_attention True \
--reward_std False \
--hungarian_op_path {path_to hungarian.so}" > nohup_nsds_django_voi2.txt

# Finally fix the encoder Transformer and finetune the decoder Transformer with larger batch size.
# This can be achieved through the "alternate_training" argument, which designates the number of
# training steps for the decoder and for the encoder, respectively, before switching to training
# the other model.
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
--order soft --policy_gradient without_bvn \
--alternate_training 1000000 1
--action_refinement 1 \
--share_embedding False \
--pt_pg_type sinkhorn --pt_relative_embedding False --pt_positional_attention True \
--reward_std False \
--hungarian_op_path {path_to hungarian.so}" > nohup_nsds_django_voi4.txt
```

#### Gigaword

```bash
# Train with embedding shared between encoder Transformer and decoder Transformer first
nohup bash -c "CUDA_VISIBLE_DEVICES=0,1,2,3 python -u scripts/train.py \
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
--order soft --policy_gradient without_bvn \
--kl_coeff 0.5 \
--action_refinement 4 \
--share_embedding True \
--pt_pg_type sinkhorn --pt_relative_embedding False --pt_positional_attention True \
--reward_std False --use_ppo \
--hungarian_op_path {path_to hungarian.so}" > nohup_nsds_gigaword_voi1.txt

# Then train with embedding separated and kl regularization annealed.
# Note that we also add a cosine alignment loss between the encoder Transformer and
# decoder Transformer's embeddings, achieved through the "embedding_align_coeff" argument.
nohup bash -c "CUDA_VISIBLE_DEVICES=0,1,2,3 python -u scripts/train.py \
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
--order soft --policy_gradient without_bvn \
--kl_coeff 0.5 --kl_log_linear 0.03 \
--action_refinement 4 \
--share_embedding False \
--embedding_align_coeff 100.0 \ 
--pt_pg_type sinkhorn --pt_relative_embedding False --pt_positional_attention True \
--reward_std False --use_ppo \
--hungarian_op_path {path_to hungarian.so}" > nohup_nsds_gigaword_voi2.txt

# Finally fix the encoder Transformer and finetune the decoder Transformer with larger batch size.
# This can be achieved through the "alternate_training" argument, which designates the number of
# training steps for the decoder and for the encoder, respectively, before switching to training
# the other model.
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
--alternate_training 500000 1 \        # this fixes PT throughout training
--decoder_init_lr 0.00007 --pt_init_lr 0.0 --lr_schedule linear \
--label_smoothing 0.1 \
--order soft --policy_gradient without_bvn \
--action_refinement 1 \
--share_embedding False \
--pt_pg_type sinkhorn --pt_relative_embedding False --pt_positional_attention True \
--reward_std False \
--hungarian_op_path {path_to hungarian.so}" > nohup_nsds_gigaword_voi3.txt
```

#### WMT

```bash
# Train with embedding shared between encoder Transformer and decoder Transformer first
nohup bash -c "CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8 python -u scripts/train.py \
--dataset wmt \
--train_folder {path_to_wmt}/train \
--vocab_file ro_en_vocab.txt \
--model_ckpt ckpt_refinement/nsds_wmt_voi.h5 \
--batch_size 54 \
--num_epochs 20 \
--embedding_size 512 --heads 8 --num_layers 6 \
--first_layer discrete --final_layer indigo \
--decoder_pretrain -1 \
--decoder_init_lr 0.00005 --pt_init_lr 0.000005 --lr_schedule constant \
--label_smoothing 0.1 \
--order soft --policy_gradient without_bvn \
--kl_coeff 0.3 \
--action_refinement 3 \
--share_embedding True \
--pt_pg_type sinkhorn --pt_relative_embedding False --pt_positional_attention True \
--reward_std False \
--hungarian_op_path {path_to hungarian.so}
--save_interval 50000" > nohup_nsds_wmt_voi1.txt

# Then train with embeddings separated and with alignment loss added to PT. 
# Train with learning rate 5e-5/5e-6 for 40 epochs and then 3e-5/3e-6 for
# 40 epochs (not carefully tuned).
nohup bash -c "CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8 python -u scripts/train.py \
--dataset wmt \
--train_folder {path_to_wmt}/train \
--vocab_file ro_en_vocab.txt \
--model_ckpt ckpt_refinement/nsds_wmt_voi.h5 \
--batch_size 54 \
--num_epochs 40 \                               # 40 afterwards
--embedding_size 512 --heads 8 --num_layers 6 \
--first_layer discrete --final_layer indigo \
--decoder_pretrain -1 \
--decoder_init_lr 0.00005 --pt_init_lr 0.000005 --lr_schedule constant \   # 3e-5/3e-6 afterwards
--label_smoothing 0.1 \
--order soft --policy_gradient without_bvn \
--kl_coeff 0.3 --kl_log_linear 0.01 \           # --kl_coeff 0.01 --kl_log_linear 0.0007 afterwards
--embedding_align_coeff 10.0 \
--action_refinement 3 \
--share_embedding False \
--pt_pg_type sinkhorn --pt_relative_embedding False --pt_positional_attention True \
--reward_std False \
--hungarian_op_path {path_to hungarian.so}
--save_interval 50000" > nohup_nsds_wmt_voi2.txt

# Finetune the model with PT fixed, larger batch size, and learning rate linear decay.
nohup bash -c "CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -u scripts/train.py \
--dataset wmt \
--train_folder {path_to_wmt}/train \
--vocab_file ro_en_vocab.txt \
--model_ckpt ckpt_refinement/nsds_wmt_voi.h5 \
--batch_size 128 \
--num_epochs 20 \                               
--embedding_size 512 --heads 8 --num_layers 6 \
--first_layer discrete --final_layer indigo \
--decoder_pretrain -1 \
--decoder_init_lr 0.00003 --pt_init_lr 0.0 --lr_schedule linear \   
--label_smoothing 0.1 \
--order soft --policy_gradient without_bvn \
--alternate_training 1000000 1 \       # this makes PT fixed throughout training
--action_refinement 1 \
--share_embedding False \
--pt_pg_type sinkhorn --pt_relative_embedding False --pt_positional_attention True \
--reward_std False \
--hungarian_op_path {path_to hungarian.so}
--save_interval -1" > nohup_nsds_wmt_voi3.txt
```

## Validation / Test

You may evaluate a trained model with the following commands. Interestingly, on COCO and Gigaword, we found that our model achieves better performance when the beam size is small and larger than 1.

#### COCO

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/validate.py \
--dataset captioning \
--validate_folder ~/val2017_tfrecords --ref_folder ~/captions_val2017 \
--batch_size 32 --beam_size 2 \
--vocab_file train2017_vocab.txt \
--model_ckpt ckpt/nsds_coco_voi_ckpt.h5 \
--embedding_size 512 --heads 8 --num_layers 6 \
--first_layer region --final_layer indigo
```


#### Django

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/validate.py \
--dataset django \
--validate_folder django_data/(dev/test) --ref_folder "" \
--batch_size 8 --beam_size 5 \
--vocab_file django_data/djangovocab.txt \
--model_ckpt ckpt/nsds_django_voi_ckpt.h5 \
--embedding_size 512 --heads 8 --num_layers 6 \
--first_layer discrete --final_layer indigo
```

#### Gigaword

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/validate.py \
--dataset gigaword \
--validate_folder {path_to_gigaword}/(dev/test) --ref_folder "" \
--batch_size 50 --beam_size 2 \
--vocab_file {path_to_gigaword}/gigaword_vocab.txt \
--model_ckpt ckpt/nsds_gigaword_voi_ckpt.h5 \
--embedding_size 512 --heads 8 --num_layers 6 \
--first_layer discrete --final_layer indigo
```

After validation is done, post-process the output.

```bash
sed -r 's/(@@ )|(@@ ?$)//g' hyp_caps_list.txt > hyp_caps_list_cleaned.txt
sed -r 's/(@@ )|(@@ ?$)//g' ref_caps_list.txt > ref_caps_list_cleaned.txt
python scripts/calc_gigaword_score.py --files hyp_caps_list_cleaned.txt ref_caps_list_cleaned.txt
```

#### WMT16 Ro-En

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/validate.py \
--dataset wmt \
--validate_folder {path_to_wmt}/(dev/test) --ref_folder "" \
--batch_size 4 --beam_size 5 \
--vocab_file ro_en_vocab.txt \
--model_ckpt ckpt/nsds_wmt_voi_ckpt.h5 \
--embedding_size 512 --heads 8 --num_layers 6 \
--first_layer discrete --final_layer indigo
```

After validation is done, post-process the output.
```bash
sed -r 's/(@@ )|(@@ ?$)//g' hyp_caps_list.txt > hyp_caps_list_cleaned.txt
sed -r 's/(@@ )|(@@ ?$)//g' ref_caps_list.txt > ref_caps_list_cleaned.txt
{path_to_mosesdecoder}/scripts/recaser/truecase.perl -model {dataroot}/wmt16_translate/ro-en/truecase-model.en < ref_caps_list_cleaned.txt > ref_caps_list_cleaned2.txt
{path_to_mosesdecoder}/scripts/recaser/truecase.perl -model {dataroot}/wmt16_translate/ro-en/truecase-model.en < hyp_caps_list_cleaned.txt > hyp_caps_list_cleaned2.txt
python scripts/calc_wmt_score.py --files hyp_caps_list_cleaned2.txt ref_caps_list_cleaned2.txt
```


## Visualization

These scripts visualize the generation orders of our model.

#### COCO

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/inspect_order.py \
--dataset captioning \
--validate_folder ~/val2017_tfrecords --ref_folder ~/captions_val2017 \
--batch_size 32 --beam_size 2 \
--vocab_file train2017_vocab.txt \
--model_ckpt ckpt/nsds_coco_voi_ckpt.h5 \
--embedding_size 512 --heads 8 --num_layers 6 \
--first_layer region --final_layer indigo \
--policy_gradient without_bvn --pt_positional_attention True \
--save_path inspect_generation_stats_coco.txt
```


#### Django

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/inspect_order.py \
--dataset django \
--validate_folder django_data/(dev/test) --ref_folder "" \
--batch_size 8 --beam_size 5 \
--vocab_file django_data/djangovocab.txt \
--model_ckpt ckpt/nsds_django_voi_ckpt.h5 \
--embedding_size 512 --heads 8 --num_layers 6 \
--first_layer discrete --final_layer indigo \
--policy_gradient without_bvn --pt_positional_attention True \
--save_path inspect_generation_stats_coco.txt
```

#### Gigaword

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/inspect_order.py \
--dataset gigaword \
--validate_folder {path_to_gigaword}/(dev/test) --ref_folder "" \
--batch_size 50 --beam_size 2 \
--vocab_file {path_to_gigaword}/gigaword_vocab.txt \
--model_ckpt ckpt/nsds_gigaword_voi_ckpt.h5 \
--embedding_size 512 --heads 8 --num_layers 6 \
--first_layer discrete --final_layer indigo \
--policy_gradient without_bvn --pt_positional_attention True \
--save_path inspect_generation_stats_coco.txt
```

#### WMT16 Ro-En

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/inspect_order.py \
--dataset wmt \
--validate_folder {path_to_wmt}/(dev/test) --ref_folder "" \
--batch_size 4 --beam_size 5 \
--vocab_file ro_en_vocab.txt \
--model_ckpt ckpt/nsds_wmt_voi_ckpt.h5 \
--embedding_size 512 --heads 8 --num_layers 6 \
--first_layer discrete --final_layer indigo \
--policy_gradient without_bvn --pt_positional_attention True \
--save_path inspect_generation_stats_coco.txt
```
