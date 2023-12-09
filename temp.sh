NUM_PROC=$1

singularity exec --env HF_HOME=$HF_HOME --env PYTHONPATH=/usr/local/lib/python3.8/site-packages/ --nv /scratch/ssb3vk/MLIA/mlia_fp.sif torchrun --nproc_per_node=$NUM_PROC train.py \
/scratch/ssb3vk/MLIA/AD_Classification_data/Classification_data_formatted \
--model mixnet_s \
-b 32 \
--sched step \
--epochs 450 \
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
--lr 0.16 \
--dist-bn reduce

#dont worry about lines 2-14 they are just hparams for this specific project, replace them with the hparams you sent