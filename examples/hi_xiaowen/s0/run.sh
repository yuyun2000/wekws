#!/bin/bash
# Copyright 2021  Binbin Zhang(binbzha@qq.com)

. ./path.sh

stage=1
stop_stage=4
num_keywords=1

config=conf/mdtc-fbank.yaml
norm_mean=true
norm_var=true
gpus="0"

checkpoint=
dir=exp/mdtc

noise_lmdb=/hy-tmp/noi-lmdb
reverb_lmdb=/hy-tmp/rir-lmdb

num_average=30
score_checkpoint=$dir/avg_${num_average}.pt

download_dir=./data/local # your data dir

. tools/parse_options.sh || exit 1;
window_shift=50

if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ]; then
  echo "Download and extracte all datasets"
  local/mobvoi_data_download.sh --dl_dir $download_dir
fi


if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
  echo "Preparing datasets..."
  mkdir -p dict
  echo "<FILLER> -1" > dict/dict.txt
  echo "<HI_XIAOWEN> 0" >> dict/dict.txt
  echo "<NIHAO_WENWEN> 1" >> dict/dict.txt
  awk '{print $1}' dict/dict.txt > dict/words.txt

  for folder in train dev test; do
    mkdir -p data/$folder
    for prefix in p n; do
      mkdir -p data/${prefix}_$folder
      json_path=$download_dir/mobvoi_hotword_dataset_resources/${prefix}_$folder.json
      local/prepare_data.py $download_dir/mobvoi_hotword_dataset $json_path \
        dict/dict.txt data/${prefix}_$folder
    done
    cat data/p_$folder/wav.scp data/n_$folder/wav.scp > data/$folder/wav.scp
    cat data/p_$folder/text data/n_$folder/text > data/$folder/text
    rm -rf data/p_$folder data/n_$folder
  done
fi


if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  echo "Compute CMVN and Format datasets"
  tools/compute_cmvn_stats.py --num_workers 16 --train_config $config \
    --in_scp data/train/wav.scp \
    --out_cmvn data/train/global_cmvn

  for x in train dev; do
    tools/wav_to_duration.sh --nj 8 data/$x/wav.scp data/$x/wav.dur
    tools/make_list.py data/$x/wav.scp data/$x/text \
      data/$x/wav.dur data/$x/data.list
  done
fi


if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  echo "Start training ..."
  mkdir -p $dir
  cmvn_opts=
  $norm_mean && cmvn_opts="--cmvn_file data/train/global_cmvn"
  $norm_var && cmvn_opts="$cmvn_opts --norm_var"
  num_gpus=$(echo $gpus | awk -F ',' '{print NF}')
  torchrun --standalone --nnodes=1 --nproc_per_node=$num_gpus \
    wekws/bin/train.py --gpus $gpus \
      --config $config \
      --train_data data/train/data.list \
      --cv_data data/dev/data.list \
      --model_dir $dir \
      --num_workers 8 \
      --num_keywords $num_keywords \
      --min_duration 50 \
      --seed 666 \
      --dict ./dict \
      $cmvn_opts \
      ${reverb_lmdb:+--reverb_lmdb $reverb_lmdb} \
      ${noise_lmdb:+--noise_lmdb $noise_lmdb} \
      ${checkpoint:+--checkpoint $checkpoint}
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  echo "Do model average, Compute FRR/FAR ..."
  python wekws/bin/average_model.py \
    --dst_model $score_checkpoint \
    --src_path $dir  \
    --num ${num_average} \
    --val_best
fi


if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
  jit_model=$(basename $score_checkpoint | sed -e 's:.pt$:.zip:g')
  onnx_model=$(basename $score_checkpoint | sed -e 's:.pt$:.onnx:g')
  python wekws/bin/export_jit.py \
    --config $dir/config.yaml \
    --checkpoint $score_checkpoint \
    --jit_model $dir/$jit_model
  python wekws/bin/export_onnx.py \
    --config $dir/config.yaml \
    --checkpoint $score_checkpoint \
    --onnx_model $dir/$onnx_model
fi