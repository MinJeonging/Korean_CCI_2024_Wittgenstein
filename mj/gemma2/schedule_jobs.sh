#실험 예약하는 코드 예시

# 첫 번째 작업 예약
# echo "nohup bash -c 'CUDA_VISIBLE_DEVICES=0,1,2,3 python -m run.train \
#     --model_id rtzr/ko-gemma-2-9b-it \
#     --save_dir /root/gemma2/resource/results/gemma_ft_v0_1 \
#     --batch_size 4 \
#     --gradient_accumulation_steps 32 \
#     --epoch 1 \
#     --lr 2e-5 \
#     --warmup_steps 20' > log_gemma_train1.txt 2>&1 &" | at now + 1 minute

# # 두 번째 작업 예약
# echo "nohup bash -c 'CUDA_VISIBLE_DEVICES=0,1,2,3 python -m run.train \
#     --model_id rtzr/ko-gemma-2-9b-it \
#     --save_dir /root/gemma2/resource/results/gemma_ft_v0_2 \
#     --batch_size 4 \
#     --gradient_accumulation_steps 32 \
#     --epoch 1 \
#     --lr 2e-5 \
#     --warmup_steps 20' > log_gemma_train2.txt 2>&1 &" | at now + 10 minute
