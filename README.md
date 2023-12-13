# Setup
* First step is to download the container to your local scratch dir on Rivanna (/scratch/[computing_id]/containers/mlia_fp.sif)
  * Please grab the container from https://drive.google.com/file/d/1KJz2xQEB9jcH1dRTF3rjcl6pf9PqcCKB/view?usp=sharing
* Second step is to make sure you request a Jupyter Lab session with an A100 GPU (a single one was how we did training/testing.)
* Third step is to start training! This is as simple as going to the base directory of the repo and running *bash train.sh*
  * This will save checkpoints in output/train
* To test saved models please run bash test.sh with the model checkpoint to test specified!
  * For example, if you want to run testing with the code downloaded please copy the uploaded to canvas model file into the repo and then specify the path to that file. I.e. '--checkpoint_path './last.pth.tar' \'
* It's that easy :) - credit goes to Timm: https://github.com/huggingface/pytorch-image-models/tree/main#models

repo link: https://github.com/alexiglad/mlia_final_project