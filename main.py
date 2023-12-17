import random
import numpy as np
import paddle

from config import *
from dataset import *

from argparser import read_options
from model import choose_model

from trainer import trainer

from paddle.regularizer import L1Decay, L2Decay

#设定随机种子，保证结果可复现
random.seed(SEED)
np.random.seed(SEED)
paddle.seed(SEED)

train_dataloader = paddle.io.DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
test_dataloader = paddle.io.DataLoader(test_dataset, places=paddle.CPUPlace(), batch_size=TRAIN_BATCH_SIZE,shuffle=True)

THRESHOLD=len(test_dataloader)//2
iters = int(len(train_dataset)/TRAIN_BATCH_SIZE) * NUM_EPOCHS #训练次数
lr = paddle.optimizer.lr.CosineAnnealingDecay(LR, T_max=(iters // 3), last_epoch=0.5)


def main():
    options = read_options()
    model = choose_model(options)
    optimizer = paddle.optimizer.Adam(lr, parameters=model.parameters())

    Trainer=trainer(NUM_EPOCHS,THRESHOLD,model,MODE,optimizer,train_dataloader,test_dataloader)
    Trainer.run(early_stop=True,patience=10)

if __name__ == '__main__':
    main()
# optimizer = paddle.optimizer.Adam(lr, parameters=model.parameters())#,weight_decay=L2Decay(coeff)) #sgd / momentum 0.9 /adam，weight_decay是正则化项




