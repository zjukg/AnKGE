DATA_DIR=dataset

MODEL_NAME=AnKnn
DATASET_NAME=FB15K237
DATA_PATH=$DATA_DIR/$DATASET_NAME
LITMODEL_NAME=AnKnnLitModel
TRAIN_SAMPLER_CLASS=KnnSampler
TEST_SAMPLER_CLASS=RevTestSampler
LOSS=AnKnn_Loss

BASE_MODEL_NAME=TransE
BASE_MODEL_PATH=epoch=19-Eval\|mrr=0.321.ckpt
ENT_KNN=1
REL_KNN=1
TRIPLE_KNN=3
TRIPLE_ENT_KNN=1000
TRIPLE_REL_KNN=40
SET_LEVEL=4
SET_LOSS_DISTANCE=1
ENT_LAMBDA=0.01
REL_LAMBDA=0.2
TRIPLE_LAMBDA=0.02

MAX_EPOCHS=1000
EMB_DIM=500
ADV_TEMP=1.0
TRAIN_BS=4096
EVAL_BS=32
NUM_NEG=0
MARGIN=9.0
LR=1e-4
MILESTONES="200"

EARLY_STOP_PATIENCE=10
REGULARIZATION=1e-7
CHECK_PER_EPOCH=50
NUM_WORKERS=8
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
    --lr $LR \
    --milestones $MILESTONES \
    --early_stop_patience $EARLY_STOP_PATIENCE \
    --regularization $REGULARIZATION \
    --check_per_epoch $CHECK_PER_EPOCH \
    --num_workers $NUM_WORKERS \
    --use_wandb \
    #--save_config \







