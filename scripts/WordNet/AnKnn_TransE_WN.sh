DATA_DIR=dataset

MODEL_NAME=AnKnn
DATASET_NAME=WN18RR
DATA_PATH=$DATA_DIR/$DATASET_NAME
LITMODEL_NAME=AnKnnLitModel
TRAIN_SAMPLER_CLASS=KnnSampler
TEST_SAMPLER_CLASS=TestSampler
LOSS=AnKnn_Loss

BASE_MODEL_NAME=TransE
BASE_MODEL_PATH=epoch=109-Eval\|mrr=0.224.ckpt
ENT_KNN=1
REL_KNN=1
TRIPLE_KNN=20
TRIPLE_ENT_KNN=1000
TRIPLE_REL_KNN=5
TRANS_ALPHA=0
SET_LEVEL=4
SET_LOSS_DISTANCE=1
ENT_LAMBDA=0.01
REL_LAMBDA=0.3
TRIPLE_LAMBDA=0.3
MILESTONES="200"

MAX_EPOCHS=1000
EMB_DIM=500
ADV_TEMP=1.0
TRAIN_BS=2048
EVAL_BS=32
NUM_NEG=0
MARGIN=9.0
LR=1e-3

ANAFUNC='cos'
EARLY_STOP_PATIENCE=10
REGULARIZATION=0
CHECK_PER_EPOCH=10
NUM_WORKERS=10
GPU=0

CUDA_VISIBLE_DEVICES=$GPU python -u main.py \
    --model_name $MODEL_NAME \
    --dataset_name $DATASET_NAME \
    --data_path $DATA_PATH \
    --litmodel_name $LITMODEL_NAME \
    --train_sampler_class $TRAIN_SAMPLER_CLASS \
    --test_sampler_class $TEST_SAMPLER_CLASS \
    --loss $LOSS \
    --calculate_knn \
    --base_model_name $BASE_MODEL_NAME \
    --base_model_path $BASE_MODEL_PATH \
    --ent_knn $ENT_KNN \
    --rel_knn $REL_KNN \
    --triple_knn $TRIPLE_KNN \
    --triple_ent_knn $TRIPLE_ENT_KNN \
    --triple_rel_knn $TRIPLE_REL_KNN \
    --trans_alpha $TRANS_ALPHA \
    --set_level $SET_LEVEL \
    --set_loss_distance $SET_LOSS_DISTANCE \
    --ent_lambda $ENT_LAMBDA \
    --rel_lambda $REL_LAMBDA \
    --triple_lambda $TRIPLE_LAMBDA \
    --max_epochs $MAX_EPOCHS \
    --emb_dim $EMB_DIM \
    --adv_temp $ADV_TEMP \
    --train_bs $TRAIN_BS \
    --eval_bs $EVAL_BS \
    --num_neg $NUM_NEG \
    --margin $MARGIN \
    --milestones $MILESTONES \
    --lr $LR \
    --anafunc $ANAFUNC \
    --early_stop_patience $EARLY_STOP_PATIENCE \
    --regularization $REGULARIZATION \
    --check_per_epoch $CHECK_PER_EPOCH \
    --num_workers $NUM_WORKERS \
    --use_wandb \
    #--save_config \








