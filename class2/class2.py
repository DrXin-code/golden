import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim
from scipy.linalg import orth

np.set_printoptions(threshold=np.inf)
class class2(object):

    def __init__(self, input_size, output_size, hidden1_units, hidden2_units, hidden3_units, l1_penalty, epoch_num, lr, filename1,filename2, x_tr, y_tr, x_va, y_va, x_te, y_te,beta):

        # 输入输出、三个隐藏层节点数
        self.input_size = input_size
        self.output_size = output_size
        self.hidden1_units = hidden1_units
        self.hidden2_units = hidden2_units
        self.hidden3_units = hidden3_units

        # l1正则项系数、epoch数量、学习率、文件名
        self.l1_penalty = l1_penalty
        self.epoch_num = epoch_num
        self.lr = lr
        self.filename1 = filename1
        self.filename2 = filename2

        self.parameter = []
        # 训练、验证、测试数据
        self.x_tr = x_tr
        self.y_tr = y_tr
        self.x_va = x_va
        self.y_va = y_va
        self.x_te = x_te
        self.y_te = y_te

        self.beta = beta

    class MyDataSet(data.Dataset):
        def __init__(self, x=None, y=None):
            self.X = x
            self.Y = y

        def __getitem__(self, index):
            return self.X[index], self.Y[index]

        def getall(self):
            return self.X, self.Y

        def __len__(self):
            return len(self.X)

    def adjust_learning_rate(self, optimizer, epoch):
        """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
        lr = self.lr * (0.1 ** (epoch // 2000))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


    def train(self):

        tr_best = torch.FloatTensor([100])
        va_best = torch.FloatTensor([100])
        va_best_te = torch.FloatTensor([100])
        va_best_S = torch.zeros((self.y_va.shape[0]))
        train_set = self.MyDataSet(self.x_tr, self.y_tr)
        train_loader = data.DataLoader(train_set, batch_size=200, shuffle=True)
        x_va_data = self.x_va
        x_te_data = self.x_te

        net = nn.Sequential(


            nn.Linear(self.input_size, self.hidden1_units, bias=False),  # bias=False相当于是bias=0
            # nn.Tanh(),   第一层是identity没有activation function （文章里的定理2就是第一层不加激活函数只是转换）
            nn.Linear(self.hidden1_units, self.hidden2_units),
            nn.Tanh(),
            nn.Linear(self.hidden2_units, self.output_size),


            # nn.Linear(self.input_size, self.output_size)
        )
        # 选择设备：优先使用 MPS，如果不可用则使用 CPU
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        net.to(device)
        optimizer = optim.Adam(net.parameters(), lr=self.lr)
        cri_loss = nn.MSELoss()
        # if not os.path.exists('./log/'+self.filename1+'/log1/'):
        #     os.makedirs('./log/'+self.filename1+'/log1/')
        # outlog = open('./log/'+self.filename1 +'/log1/'+self.filename2+'—h1:'+str(self.hidden1_units)+'—h2:'+str(self.hidden2_units)+'—h3:'+str(self.hidden3_units)+ '.log', 'w')
        x_va_data = x_va_data.to(device)
        x_te_data = x_te_data.to(device)
        self.y_va = self.y_va.to(device)
        self.y_te = self.y_te.to(device)
        for epoch in range(self.epoch_num):
            #self.adjust_learning_rate(optimizer, epoch)
            va_now = 0
            for i, batch in enumerate(train_loader, 0):
                x_batch, y_batch = batch
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                pre = net(x_batch)
                loss = cri_loss(pre, y_batch)
                tr_best = min(loss.data.cpu(), tr_best)

                l1_loss = 0
                flag=0
                for param in net.parameters():
                    flag+=1
                    if flag<3:
                        continue
                    if len(param.shape) == 2:
                        l1_loss += torch.sum(torch.abs(param))

                loss += self.l1_penalty * l1_loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                with torch.no_grad():
                    pre_va = net(x_va_data)
                    pre_te = net(x_te_data)

                    pre_loss_va = cri_loss(pre_va, self.y_va).cpu()
                    pre_loss_te = cri_loss(pre_te, self.y_te).cpu()

                    S = (pre_va - self.y_va)** 2

                    va_now = pre_loss_va.data
                    if pre_loss_va.data < va_best:
                        va_best = pre_loss_va.data
                        va_best_te = pre_loss_te.data
                        va_best_S = S
                    va_best = min(pre_loss_va.data, va_best)

                    #print('epoch: ', epoch, 'batch: ', i, ' tr_loss: ', loss.data, 'va_loss: ', pre_loss_va.data, 'te_loss: ', pre_loss_te.data)
                    #outlog.write('epoch: '+str(epoch)+'   batch: '+str(i)+'    tr_loss: '+str(loss.data)+'    va_loss: '+str(pre_loss_va.data)+'    te_loss: '+str(pre_loss_te.data)+'\n')
                    sys.stdout.flush()
            # if va_now - va_best > 0.1:
            #     break
            self.parameter = list(net.parameters())
        net.to('cpu')
        #outlog.write('\n\n第一层weights矩阵:'+str(self.parameter[0].detach().numpy().shape)+'\n'+str(self.parameter[0].detach().numpy())+'\n')
        #outlog.write('\n\ntr_best'+str(tr_best)+'\n')
        # print('tr_best', tr_best)
        #outlog.write('va_test'+str(va_best)+'\n')
        # print('va_best', va_best)
        #outlog.write('va_best_te'+str(va_best_te)+'\n')
        # print('va_best_te', va_best_te)
        '''
        计算q
        '''

        p1 = self.parameter[0].detach().numpy()
        p1 = orth(p1.T).T
        self.beta = orth(self.beta.T)
        self.beta = np.dot(self.beta, self.beta.T)

        tmp = np.matmul(p1,self.beta)

        tmp = np.matmul(tmp, p1.T)
        q = np.linalg.det(tmp)
        q = np.sqrt(q)
        #outlog.write('q:' + str(q) + '\n')
        # print('q:', q)
        #outlog.close()
        print(list(net.parameters()))
        for param in net.parameters():
            print(param.shape)
        return tr_best, va_best, va_best_te, list(net.parameters()),q, va_best_S.cpu()

