# apt-get update
# apt-get install ffmpeg libsm6 libxext6  -y

DATASET=lego
PYTHONPATH=/root/miniconda3/envs/rookie/bin

CUDA_VISIBLE_DEVICES=1 ${PYTHONPATH}/python run_nerf.py --config configs/${DATASET}.txt