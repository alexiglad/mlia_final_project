singularity exec --env HF_HOME=$HF_HOME --env PYTHONPATH=~/.local/lib/python3.8/site-packages:/usr/local/lib/python3.8/site-packages/ --nv /scratch/$(whoami)/containers/mlia_fp.sif torchrun --nproc_per_node 1 train.py \
./Classification_data_formatted \
--model mixnet_s \
-b 32 \
--sched step \
--epochs 200 \
--decay-epochs 2.4 \
--decay-rate .969 \
--opt rmsproptf \
--opt-eps .001 \
-j 8 \
--warmup-lr 1e-6 \
--weight-decay 1e-5 \
--drop 0.3 \
--drop-path 0.2 \
--model-ema \
--model-ema-decay 0.9999 \
--aa rand-m9-mstd0.5 \
--remode pixel \
--reprob 0.3 \
--amp \
--lr 0.003 \
--dist-bn reduce \
--wandb_name '3.5.7' \
--num_classes 2 \
--log-wandb
#TODO manually set nproc_per_node value
