conda activate base
export CUDA_HOME=/mnt/cephfs/smil/cuda/cuda-11.0
export PATH=${CUDA_HOME}/bin${PATH:+:${PATH}}
export CC=gcc-7 CXX=g++-7
export TORCH_CUDA_ARCH_LIST="6.1 8.0"
CUDA_VISIBLE_DEVICES=1 python eval_epsad.py --num_sub 490 --t 1000 --adv_batch_size 500 --detection_flag --detection_ensattack_flag
