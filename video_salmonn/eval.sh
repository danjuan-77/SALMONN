#! /bin/bash
export CUDA_VISIBLE_DEVICES=0


python eval.py --cfg-path config/test.yaml

python eval2.py --cfg-path config/test.yaml

# nohup bash eval.sh > /share/nlp/tuwenming/projects/HAVIB/logs/eval_videosalmonn_gpu0_$(date +%Y%m%d%H%M%S).log 2>&1 &
