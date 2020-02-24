set -ex
python train.py \
  --dataroot /local_data/cosmiq/wdata/rareplanes/cyclegan_train_set \
  --name rp_cyclegan_v1 \
  --gpu_ids 0,1 \
  --checkpoints-dir /local_data/cosmiq/wdata/rareplanes/cyclegan_checkpoints/v1/ \
  --model cycle_gan  \
  --dataset_mode template \
  --pool_size 50 \
  --no_dropout \
  --num_threads 12 \
  --crop_size 512 \
  --preprocess crop \
  --verbose
