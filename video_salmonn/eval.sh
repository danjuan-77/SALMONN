#! /bin/bash
export CUDA_VISIBLE_DEVICES=0


python eval.py --cfg-path config/test.yaml

python eval2.py --cfg-path config/test.yaml

# nohup bash eval.sh > /share/nlp/tuwenming/projects/HAVIB/logs/eval_videosalmonn_gpu0_$(date +%Y%m%d%H%M%S).log 2>&1 &


# export CUDA_VISIBLE_DEVICES=0
# nohup python eval.py --cfg-path config/gpu0.yaml > /share/nlp/tuwenming/projects/HAVIB/logs/eval_videosalmonn_unimodal_gpu0_$(date +%Y%m%d%H%M%S).log 2>&1 &

# export CUDA_VISIBLE_DEVICES=1
# nohup python eval.py --cfg-path config/gpu1.yaml > /share/nlp/tuwenming/projects/HAVIB/logs/eval_videosalmonn_unimodal_gpu1_$(date +%Y%m%d%H%M%S).log 2>&1 &

# export CUDA_VISIBLE_DEVICES=2
# nohup python eval.py --cfg-path config/gpu2.yaml > /share/nlp/tuwenming/projects/HAVIB/logs/eval_videosalmonn_unimodal_gpu2_$(date +%Y%m%d%H%M%S).log 2>&1 &

# export CUDA_VISIBLE_DEVICES=3
# nohup python eval.py --cfg-path config/gpu3.yaml > /share/nlp/tuwenming/projects/HAVIB/logs/eval_videosalmonn_unimodal_gpu3_$(date +%Y%m%d%H%M%S).log 2>&1 &

# export CUDA_VISIBLE_DEVICES=0
# nohup python eval.py --cfg-path config/gpu4.yaml > /share/nlp/tuwenming/projects/HAVIB/logs/eval_videosalmonn_unimodal_gpu4_$(date +%Y%m%d%H%M%S).log 2>&1 &

# export CUDA_VISIBLE_DEVICES=1
# nohup python eval.py --cfg-path config/gpu5.yaml > /share/nlp/tuwenming/projects/HAVIB/logs/eval_videosalmonn_unimodal_gpu5_$(date +%Y%m%d%H%M%S).log 2>&1 &

# export CUDA_VISIBLE_DEVICES=2
# nohup python eval.py --cfg-path config/gpu6.yaml > /share/nlp/tuwenming/projects/HAVIB/logs/eval_videosalmonn_unimodal_gpu6_$(date +%Y%m%d%H%M%S).log 2>&1 &

# export CUDA_VISIBLE_DEVICES=3
# nohup python eval.py --cfg-path config/gpu7.yaml > /share/nlp/tuwenming/projects/HAVIB/logs/eval_videosalmonn_unimodal_gpu7_$(date +%Y%m%d%H%M%S).log 2>&1 &

###############

# export CUDA_VISIBLE_DEVICES=0
# nohup python eval2.py --cfg-path config/gpu0.yaml > /share/nlp/tuwenming/projects/HAVIB/logs/eval_videosalmonn_unimodal_gpu0_$(date +%Y%m%d%H%M%S).log 2>&1 &

# export CUDA_VISIBLE_DEVICES=1
# nohup python eval2.py --cfg-path config/gpu1.yaml > /share/nlp/tuwenming/projects/HAVIB/logs/eval_videosalmonn_unimodal_gpu1_$(date +%Y%m%d%H%M%S).log 2>&1 &

# export CUDA_VISIBLE_DEVICES=2
# nohup python eval2.py --cfg-path config/gpu2.yaml > /share/nlp/tuwenming/projects/HAVIB/logs/eval_videosalmonn_unimodal_gpu2_$(date +%Y%m%d%H%M%S).log 2>&1 &

# export CUDA_VISIBLE_DEVICES=3
# nohup python eval2.py --cfg-path config/gpu3.yaml > /share/nlp/tuwenming/projects/HAVIB/logs/eval_videosalmonn_unimodal_gpu3_$(date +%Y%m%d%H%M%S).log 2>&1 &

# export CUDA_VISIBLE_DEVICES=0
# nohup python eval2.py --cfg-path config/gpu4.yaml > /share/nlp/tuwenming/projects/HAVIB/logs/eval_videosalmonn_unimodal_gpu4_$(date +%Y%m%d%H%M%S).log 2>&1 &

# export CUDA_VISIBLE_DEVICES=1
# nohup python eval2.py --cfg-path config/gpu5.yaml > /share/nlp/tuwenming/projects/HAVIB/logs/eval_videosalmonn_unimodal_gpu5_$(date +%Y%m%d%H%M%S).log 2>&1 &

# export CUDA_VISIBLE_DEVICES=2
# nohup python eval2.py --cfg-path config/gpu6.yaml > /share/nlp/tuwenming/projects/HAVIB/logs/eval_videosalmonn_unimodal_gpu6_$(date +%Y%m%d%H%M%S).log 2>&1 &

# export CUDA_VISIBLE_DEVICES=3
# nohup python eval2.py --cfg-path config/gpu7.yaml > /share/nlp/tuwenming/projects/HAVIB/logs/eval_videosalmonn_unimodal_gpu7_$(date +%Y%m%d%H%M%S).log 2>&1 &
