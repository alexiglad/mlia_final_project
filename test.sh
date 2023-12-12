singularity exec --env HF_HOME=$HF_HOME --env PYTHONPATH=~/.local/lib/python3.8/site-packages:/usr/local/lib/python3.8/site-packages/ --nv /scratch/$(whoami)/containers/mlia_fp.sif torchrun --nproc_per_node 1 test.py \
--data_dir './Classification_data_formatted' \
--checkpoint_path 'output/train/20231211-185247-mixnet_s-224/checkpoint-16.pth.tar' \