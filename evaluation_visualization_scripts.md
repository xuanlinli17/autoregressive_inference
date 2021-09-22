## Validation / Test

You may evaluate a trained model with the following commands. Interestingly, on COCO and Gigaword, we found that our model achieves better performance when the beam size is small but larger than 1.

#### COCO

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/validate.py \
--dataset captioning \
--validate_folder ~/val2017_tfrecords --caption_ref_folder ~/captions_val2017 \
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
--validate_folder django_data/(dev/test) \
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
--validate_folder {path_to_gigaword}/(dev/test) \
--batch_size 50 --beam_size 2 \
--vocab_file gigaword_vocab.txt \
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
--validate_folder {path_to_wmt}/(dev/test) \
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

These scripts visualize the generation orders of our model. Note that for example, when `--model_ckpt ckpt/nsds_coco_voi_ckpt.h5` (the model checkpoint for the decoder autoregressive language model) is specified, `ckpt/nsds_coco_voi_ckpt.pt.h5` (the model checkpoint for the encoder permutation generator) will also be automatically loaded.

#### COCO

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/inspect_order.py \
--dataset captioning \
--validate_folder ~/val2017_tfrecords --caption_ref_folder ~/captions_val2017 \
--batch_size 32 --beam_size 2 \
--vocab_file train2017_vocab.txt \
--model_ckpt ckpt/nsds_coco_voi_ckpt.h5 \
--embedding_size 512 --heads 8 --num_layers 6 \
--first_layer region --final_layer indigo \
--pt_positional_attention \
--visualization_save_path inspect_generation_stats_coco.txt
```


#### Django

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/inspect_order.py \
--dataset django \
--validate_folder django_data/(dev/test) \
--batch_size 8 --beam_size 5 \
--vocab_file django_data/djangovocab.txt \
--model_ckpt ckpt/nsds_django_voi_ckpt.h5 \
--embedding_size 512 --heads 8 --num_layers 6 \
--first_layer discrete --final_layer indigo \
--pt_positional_attention \
--pt_special_encoder_block EncoderWithRelativePositionLayer \
--visualization_save_path inspect_generation_stats_django.txt
```

#### Gigaword

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/inspect_order.py \
--dataset gigaword \
--validate_folder {path_to_gigaword}/(dev/test) \
--batch_size 50 --beam_size 2 \
--vocab_file {path_to_gigaword}/gigaword_vocab.txt \
--model_ckpt ckpt/nsds_gigaword_voi_ckpt.h5 \
--embedding_size 512 --heads 8 --num_layers 6 \
--first_layer discrete --final_layer indigo \
--pt_positional_attention \
--pt_special_encoder_block EncoderWithRelativePositionLayer \
--visualization_save_path inspect_generation_stats_gigaword.txt
```

#### WMT16 Ro-En

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/inspect_order.py \
--dataset wmt \
--validate_folder {path_to_wmt}/(dev/test) \
--batch_size 4 --beam_size 5 \
--vocab_file ro_en_vocab.txt \
--model_ckpt ckpt/nsds_wmt_voi_ckpt.h5 \
--embedding_size 512 --heads 8 --num_layers 6 \
--first_layer discrete --final_layer indigo \
--pt_positional_attention \
--pt_special_encoder_block EncoderWithRelativePositionLayer \
--visualization_save_path inspect_generation_stats_wmt.txt
```
