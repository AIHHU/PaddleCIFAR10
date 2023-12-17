TRAIN_BATCH_SIZE=512 #batch_size 32 64 128 默认：64
NUM_EPOCHS=100 #训练轮数
LR=0.001 #固有学习率 0.01 0.001 0.0001 默认:0.001
MODE="normal"
SEED = 1919810 #定义随机种子，保证一切可复现
coeff=0.0001 #正则化系数 后面weight_decay删掉直接无效 默认无正则化