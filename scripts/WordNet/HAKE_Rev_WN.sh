DATA_DIR=dataset

MODEL_NAME=HAKE
DATASET_NAME=WN18RR
DATA_PATH=$DATA_DIR/$DATASET_NAME
LITMODEL_NAME=KGERevLitModel
TRAIN_SAMPLER_CLASS=RevUniSampler
MAX_EPOCHS=1000
EMB_DIM=500
LOSS=Adv_Loss
ADV_TEMP=0.5
TRAIN_BS=512
EVAL_BS=8
NUM_NEG=1024
MARGIN=6.0
LR=5e-5
CHECK_PER_EPOCH=10
MILESTONES=300
EARLY_STOP_PATIENCE=10
PHASE_WEIGHT=0.5
MODULUS_WEIGHT=0.5
NUM_WORKERS=32
GPU=3


CUDA_VISIBLE_DEVICES=$GPU python -u main.py \
    --model_name $MODEL_NAME \
    --dataset_name $DATASET_NAME \
    --data_path $DATA_PATH \
    --litmodel_name $LITMODEL_NAME \
    --train_sampler_class $TRAIN_SAMPLER_CLASS \
    --max_epochs $MAX_EPOCHS \
    --emb_dim $EMB_DIM \
    --loss $LOSS \
    --adv_temp $ADV_TEMP \
    --train_bs $TRAIN_BS \
    --eval_bs $EVAL_BS \
    --num_neg $NUM_NEG \
    --margin $MARGIN \
    --lr $LR \
    --check_per_epoch $CHECK_PER_EPOCH \
    --milestones $MILESTONES \
    --early_stop_patience $EARLY_STOP_PATIENCE \
    --phase_weight $PHASE_WEIGHT \
    --modulus_weight $MODULUS_WEIGHT \
    --num_workers $NUM_WORKERS \
    --use_weight \
    --use_wandb \
    #--save_config \