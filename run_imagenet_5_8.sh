export CUDA_HOME=/mnt/cephfs/smil/cuda/cuda-11.0
export PATH=/mnt/cephfs/smil/cuda/cuda-11.0/bin:/home/zhangshuhai/.vscode-server/bin/30d9c6cd9483b2cc586687151bcbcd635f373630/bin/remote-cli:/mnt/cephfs/home/zhangshuhai/anaconda3/bin:/mnt/cephfs/home/zhangshuhai/anaconda3/condabin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin
export CC=gcc-7
export CXX=g++-7
export TORCH_CUDA_ARCH_LIST=8.0

for t in 1 2 5 10 100 20 50; do
  for adv_eps in 0.01961 0.02353 0.02745 0.03137; do

    export CUDA_VISIBLE_DEVICES=0 
    python eval_sde_adv.py --datapath '/mnt/cephfs/mixed/dataset/imagenet'\
        --num_sub 500\
        --adv_batch_size 24\
        --detection_datapath './score_diffusion_t_imagenet_stand'\
        --config imagenet.yml\
        -i imagenet\
        --domain imagenet\
        --classifier_name imagenet-resnet50\
        --diffuse_t $t\
        --epsilon $adv_eps\
        --detection_flag\
        --detection_ensattack_flag\
        --single_vector_norm_flag\
        --perb_image\

  done
done