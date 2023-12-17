import matplotlib.pyplot as plt
import paddle.nn.functional as F
import paddle

net_loss = paddle.nn.CrossEntropyLoss()
metric = paddle.metric.Accuracy()

class trainer(object):
    def __init__(self,
                 epochs,
                 threshold,
                 model,
                 mode,
                 opt,
                 train_dataloader,
                 test_dataloader):
        """
        threshold: int，划分多少出来作为验证集，主要用于模型集成
        opt:给一个优化器对象
        mode: ["normal","assemble"],用于表示是否开启单模型集成
        """
        self.model=model
        self.opt=opt
        self.train_dataloader=train_dataloader
        self.test_dataloader=test_dataloader

        self.mode=mode
        self.threshold=threshold
        self.loss=[]
        self.acc=[]

        self.val_loss=[]
        self.val_acc=[] 

        self.max_epoch=epochs
        self.epoch=0

    def train(self):
        self.model.train()
        total_loss=0.0
        num_batchs = len(self.train_dataloader)
        LOSS_temp=[]
        ACC_temp=[]

        for batch_id,data in enumerate(self.train_dataloader()):
            x_data = data[0]
            y_data = data[1]
            predicts=self.model(x_data)
            loss=net_loss(predicts,y_data)
            acc=metric.compute(predicts,y_data)
            acc = acc.mean(axis=0)
            LOSS_temp.append(loss.numpy()[0])
            ACC_temp.append(acc.numpy()[0])
            loss.backward()
            self.opt.step()
            self.opt.clear_grad()
        total_loss=sum(LOSS_temp)/num_batchs
        total_acc=sum(ACC_temp)/num_batchs
        self.loss.append(total_loss)
        self.acc.append(total_acc)
        print("epoch: {},  loss is: {}, acc is: {}".format(self.epoch, total_loss, total_acc))
        return total_loss
        
    def eval(self,mode,early_stop_mode=False):
        """
        mode:区别于self.mode,这里主要用于区别是验证还是测试
        early_stop_mode:在启动提前停止时，只有第一次验证才必须为true
        """
        self.model.eval()
        total_loss=0.0
        num_batchs=len(self.test_dataloader)
        # print(num_batchs)

        LOSS_temp=[]
        ACC_temp=[]
        print("testing in {} mode……".format(mode))
        for batch_id, data in enumerate(self.test_dataloader()):
            # if mode=="eval" and batch_id>self.threshold: break
            # if mode=="test" and batch_id <= self.threshold: continue
            x_data = data[0]
            y_data = data[1]
            predicts = self.model(x_data)
            loss = net_loss(predicts, y_data)
            acc = metric.compute(predicts, y_data)
            acc = acc.mean(axis=0)
            # print(acc.mean(axis=0))
            # assert 0
            LOSS_temp.append(loss.numpy()[0])
            ACC_temp.append(acc.numpy()[0])
        # if mode=="eval":
            #print(max(ACC_temp))
        total_loss=sum(LOSS_temp)/len(LOSS_temp)
        total_acc=sum(ACC_temp)/len(ACC_temp)
        # else:
        #     total_loss=sum(LOSS_temp)/(num_batchs-self.threshold-1)
        #     total_acc=sum(ACC_temp)/(num_batchs-self.threshold-1)
        print("epoch: {},  loss is: {}, acc is: {}".format(self.epoch, total_loss, total_acc))
        if mode=="eval" and early_stop_mode !=True: 
            self.val_loss.append(total_loss)
            self.val_acc.append(total_acc)
            print("Now continue training……")
        return total_loss
        
    def run(self,early_stop=False, patience=None):
        """
        early_stop:是否启动提前停止
        patien:提前停止步长
        """
        if early_stop==True:
            v_loss = self.eval(mode="eval",early_stop_mode=True)
            best_loss = v_loss
            no_improve = 0
        while self.epoch<self.max_epoch:
            paddle.device.set_device("gpu") #保证一下gpu被充分利用
            self.train()
            v_loss=self.eval(mode="eval")
            if early_stop==True: #提前停止算法
                if v_loss >= best_loss:
                    no_improve += 1
                else:
                    best_loss = v_loss
                    no_improve = 0
                    self.save_checkpoint(self.epoch)
                
                if no_improve == patience:
                    break
            self.epoch +=1

        self.eval(mode="test")
            
    def visualize(self,item):
        """
        item:用于表示可视化的项目["loss","acc"]
        """
        if(self.epoch>= self.max_epoch):
            x = [i for i in range(self.max_epoch)]
        else:
            x = [i for i in range(self.epoch+1)]
        if item=="loss":
            plt.title("Loss of train and test")
            plt.plot(x, self.loss, 'b-', label=u'train_loss',linewidth=0.8)
            plt.plot(x, self.val_loss, 'c-', label=u'val_loss',linewidth=0.8)
        if item=="acc":
            plt.title("Accuracy of train and test")
            plt.plot(x, self.acc, 'b-', label=u'train_acc',linewidth=0.8)
            plt.plot(x, self.val_acc, 'c-', label=u'val_acc',linewidth=0.8)       
        plt.legend()
        #plt.xticks(l, lx)
        plt.ylabel('loss')
        plt.xlabel('epoch')
        #plt.savefig('ConformerDPCL_loss.png')

    def save_checkpoint(self, epoch):
        '''
           save model
           best: the best model
        '''
        #shutil.rmtree(self.save_path,ignore_errors=True)
        #os.makedirs(self.save_path)
        obj={'model': self.model.state_dict(), 'acc': self.val_acc[-1]} #保存一下验证精度，可以用来做模型集成
        path='model/model_'+str(epoch)+'.pdparams'
        paddle.save(obj,path)
        print("New model has been saved!\n")