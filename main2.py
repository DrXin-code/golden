import argparse
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import torch
import numpy as np
from class2 import class2
from scipy.linalg import orth
from scipy.stats import ortho_group
from generate_data import get_data
import pandas as pd
import torch.nn as nn
import time
device = torch.device("cpu")

right_d = 4

def get_arguments():
    """Parse all the arguments provided from the CLI.
    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Maoye")
    parser.add_argument("--epoch-num", type=int, default=100,
                        help="epoch-num,if data_times == 10, 1000 is enough if data_times==1,10000 is required")
    parser.add_argument("--hidden1-units", type=int, default=20,
                        help="6 or 60")
    parser.add_argument("--hidden2-units", type=int, default=10,
                        help="6 or 60")
    parser.add_argument("--hidden3-units", type=int, default=0,
                        help="6 or 60")
    parser.add_argument("--lr", type=float, default=5e-3,
                        help="5e-2")
    parser.add_argument("--l1-penalty", type=float, default=0.0001,
                        help="5e-2")
    parser.add_argument("--features-num", type=int, default=10,
                        help="5e-2")
    parser.add_argument("--filename", type=str, default='test1_2',
                        help="file name")
    parser.add_argument("--noise", type=float, default=0.1,
                        help="0.1-0.5")
    parser.add_argument("--k", type=str, default=5,
                        help="number of train")
    parser.add_argument("--lx", type=float, default=0.1,
                        help="l_x")
    parser.add_argument("--lam", type=float, default=1,
                        help="lam")
    parser.add_argument("--alp", type=float, default=0.33,
                        help="alp")
    parser.add_argument("--penalty", type=int, default=1,
                        help="1 for true,0 for false")
    return parser.parse_args()

'''
划分训练，验证，测试集
'''
def split_data(x, y, data_times=1):
    return x[:800*data_times],y[:800*data_times],x[800*data_times:1000*data_times],y[800*data_times:1000*data_times],x[1000*data_times:],y[1000*data_times:]




# 进行一次训练
def get_one_mse(path, mse_dic, trained, hidden1_units, k, index, file,result):
    print("####################\n第一层节点：", hidden1_units)
    file.write("\n####################\n第一层节点："+str(hidden1_units))
    path.append(hidden1_units)
    min_mse_tr = np.array([10000000.])
    min_mse_va = np.array([10000000.])
    min_mse_te = np.array([10000000.])
    min_va_best_S = torch.zeros((y_va.shape[0]))

    w = np.ones((hidden1_units, args.features_num))
    q = 100
    for i in range(k):

        f = class2(input_size=input_size, output_size=output_size,
                   hidden1_units=hidden1_units, hidden2_units=args.hidden2_units, hidden3_units=args.hidden3_units,
                   l1_penalty=args.l1_penalty, epoch_num=args.epoch_num, lr=args.lr, filename1=args.filename,
                   filename2 =str(index)+" " + str(i),
                   x_tr=x_tr, y_tr=y_tr, x_va=x_va, y_va=y_va, x_te=x_te, y_te=y_te, beta=beta)

        # 调用变量选择函数，返回值为最终选择好变量后，网络的参数。
        tr_best, va_best, te_best, net_parameter, net_q, va_best_S = f.train()
        # min_mse = min(min_mse, va_best.data.numpy())
        if min_mse_va >= va_best.data.numpy():
            min_mse_tr = tr_best.data.numpy()
            min_mse_va = va_best.data.numpy()
            min_mse_te = te_best.data.numpy()
            min_va_best_S = va_best_S
            w = net_parameter[0]
            q = net_q
    print("训练", str(k), "次min_MSE_tr:", str(min_mse_tr))
    file.write("\n训练"+str(k)+ "次min_MSE_tr:"+ str(min_mse_tr))
    print("min_MSE_va:", str(min_mse_va))
    print("min_MSE_te:", str(min_mse_te))
    print("q:", str(q))
    print("\n#############################")
    file.write("\nmin_MSE_va:"+ str(min_mse_va))
    file.write("\nmin_MSE_te:"+ str(min_mse_te))
    file.write("\nq:"+ str(q))
    file.write("\n#############################")
    mse_dic[str(hidden1_units)] = [min_mse_tr.tolist(), min_mse_va.tolist(), min_mse_te.tolist(), (min_mse_va + args.lam * x_tr.shape[0]**(-args.alp) * hidden1_units).tolist(), q]
    # print("fgd", min_mse_te)
    trained[str(hidden1_units)] = [min_mse_tr, min_mse_va, min_mse_te, w, q, min_va_best_S]

    if hidden1_units == right_d:
            # if result.has_key(str(index)) == False or (result[str(index)][1] > min_mse_va.tolist())[0] == True:
            #     result[str(index)] = [min_mse_tr.tolist(), min_mse_va.tolist(), min_mse_te.tolist(), q]
            # if str(index) in result == False or (result[str(index)][1] > min_mse_va.tolist())[0] == True:
            #     result[str(index)] = [min_mse_tr.tolist(), min_mse_va.tolist(), min_mse_te.tolist(), q]
            if (str(index) not in result) == True:
                print("not in")
                result[str(index)] = [min_mse_tr.tolist(), min_mse_va.tolist(), min_mse_te.tolist(), q]
            else:
                result_1=list(result.values())
                result_2=list(result.keys())
                print(result_1,result_2)
                if result[str(index)][1] > min_mse_va.tolist():
                    result[str(index)] = [min_mse_tr.tolist(), min_mse_va.tolist(), min_mse_te.tolist(), q]


    return min_mse_tr, min_mse_va, min_mse_te, w, q, min_va_best_S


# 获取参数
args = get_arguments()

# 生成数据
# x, y, beta = get_data(random_seed=123, data_times=1)
# x_tr, y_tr, x_va, y_va, x_te, y_te = split_data(x, y, data_times=1)
# input_size = x.shape[1]
# output_size = y.shape[1]

def train(index,result):
    if not os.path.exists("./log/"+args.filename+"/log2/"):
        os.makedirs("./log/"+args.filename+"/log2/")
    f = open("./log/"+args.filename+"/log2/"+str(index)+".log",'w')
    for arg in vars(args):
        f.write("%s: %s\n" % (arg, getattr(args, arg)))
        print("%s: %s\n" % (arg, getattr(args, arg)))
    path = []
    mse_dic = {}
    trained = {}


    start = 1
    p = args.features_num
    tr_len = x_tr.shape[0]
    va_len = x_va.shape[0]
    # 容忍出差错的范围
    l_x = args.lx   #alg line 2 的epsilon
    lam = args.lam
    alp = args.alp
    pen =  tr_len**(-alp)

    m = start
    n = p

    k1 = m + 0.382 * (n - m)
    k2 = m + 0.618 * (n - m)

    tmp,p_mse, ___, _, __,Sp = get_one_mse(path, mse_dic, trained, p, args.k, index, f,result)
    tmp,k1_mse, ___, _, __,S1 = get_one_mse(path, mse_dic, trained, int(k1), args.k, index, f,result)
    tmp,k2_mse, ___, _, __,S2 = get_one_mse(path, mse_dic, trained, int(k2), args.k, index, f,result)
    n_mse = p_mse

    while True:
        print("Sd:", (S1 - S2).numpy().std())
        f.write("Sd:"+ str((S1 - S2).numpy().std()))
        if (args.penalty==0 and k1_mse < (1 + l_x) * k2_mse and k2_mse < (1 + l_x) * n_mse) or (
            args.penalty==1 and k1_mse - k2_mse < lam*(k2-k1)*pen and k2_mse - n_mse < lam*(n-k2)*pen):

            print("窗口左滑")
            f.write("窗口左滑")
            path.append("窗口左滑")
            # print(k1_mse, k2_mse, k1_mse + lam * pen * k1, k2_mse + lam * pen * k2)
            n = k2
            if n - m >= 4:
                n_mse = k2_mse
                k2 = k1
                k2_mse = k1_mse
                k1 = m + 0.382 * (n - m)
                tmp, k1_mse, ___, _, __, S1 = get_one_mse(path, mse_dic, trained, int(k1), args.k,index,f,result)
            else:
                break
        else:
            print("窗口右滑")
            f.write("窗口右滑")
            path.append("窗口右滑")
            m = k1
            if n - m >= 4:
                k1 = k2
                k1_mse = k2_mse
                k2 = m + 0.618 * (n - m)
                tmp, k2_mse, _, __, ___, S2 = get_one_mse(path, mse_dic, trained, int(k2), args.k,index,f,result)
            else:
                break
    '''
    取出最优结果，单独列出来
    '''
    d = n
    # print("sfd", str(int(d)), str(int(d)) in trained)
    if (str(int(d)) in trained) == True:
        print('直接获取:', int(d))
        f.write("\n直接获取:" + str(int(d)))
        d_mse_tr = trained[str(int(d))][0]
        d_mse = trained[str(int(d))][1]
        d_mse_te = trained[str(int(d))][2]
        d_w = trained[str(int(d))][3]
        d_q = trained[str(int(d))][4]
        Sd = trained[str(int(d))][5]
    else:
        d_mse_tr, d_mse, d_mse_te, d_w, d_q, Sd = get_one_mse(path, mse_dic, trained, int(d), args.k, index, f,result)
    best_d = d
    best_d_mse_tr = d_mse_tr
    best_d_mse = d_mse
    best_d_mse_te = d_mse_te
    best_d_w = d_w
    best_d_q = d_q

    l = d - 1
    if (str(int(l)) in trained) == True:
        print('直接获取:', int(l))
        f.write("\n直接获取:" + str(int(l)))
        l_mse_tr = trained[str(int(l))][0]
        l_mse = trained[str(int(l))][1]
        l_mse_te = trained[str(int(l))][2]
        l_w = trained[str(int(l))][3]
        l_q = trained[str(int(l))][4]
    else:
        l_mse_tr, l_mse, l_mse_te, l_w, l_q, _ = get_one_mse(path, mse_dic, trained, int(l), args.k, index, f,result)

    while (args.penalty==0 and int(l) > 0 and l_mse < (1 + l_x) * d_mse) or (
        args.penalty==1 and int(l) > 0 and l_mse-d_mse < lam*pen*1) :

        best_d = l
        best_d_mse_tr = l_mse_tr
        best_d_mse = l_mse
        best_d_mse_te = l_mse_te
        best_d_w = l_w
        best_d_q = l_q
        d_mse = l_mse

        print("节点减1：")
        path.append("节点-1")
        l -= 1
        if int(l) > 0:
            if (str(int(l)) in trained) == True:
                l_mse_tr = trained[str(int(l))][0]
                l_mse = trained[str(int(l))][1]
                l_mse_te = trained[str(int(l))][2]
                l_w = trained[str(int(l))][3]
                l_q = trained[str(int(l))][4]
            else:
                l_mse_tr, l_mse, l_mse_te, l_w, l_q, _ = get_one_mse(path, mse_dic, trained, int(l), args.k, index, f,result)

    print(path)
    # print(mse_dic)
    print("best_d:", str(int(best_d)))
    print("best_d_mse_tr:", str(int(best_d_mse_tr)))
    print("best_d_mse_va:", str(best_d_mse))
    print("best_d_mse_te:", str(best_d_mse_te))
    print("best_d_w:", str(best_d_w.shape), '\n', str(best_d_w))
    print("best_d_q:", str(best_d_q))
    print(mse_dic)

    f.write("\n"+str(path))
    # print(mse_dic)
    f.write("\nbest_d:"+ str(int(best_d)))
    f.write("\nbest_d_mse_tr:"+ str(int(best_d_mse_tr)))
    f.write("\nbest_d_mse_va:"+ str(best_d_mse))
    f.write("\nbest_d_mse_te:"+ str(best_d_mse_te))
    f.write("\nbest_d_w:"+ str(best_d_w.shape)+ '\n'+ str(best_d_w))
    f.write("\nbest_d_q:"+ str(best_d_q))
    f.write("\n"+str(mse_dic))
    f.close()
    # write to csv file
    submission = pd.DataFrame(mse_dic)
    if not os.path.exists("./log/"+args.filename+"/log4/"):
        os.makedirs("./log/"+args.filename+"/log4/")
    submission.to_csv("./log/"+args.filename+"/log4/"+str(index)+".csv", mode='a', index=False)
    return best_d_mse_tr, best_d_mse, best_d_mse_te

def main():
    start_time = time.time()
    seed = [26316, 64148, 98246, 13995, 80418, 90809, 22060, 60059, 36648, 45799]
    tr = []
    va = []
    te = []
    result={}
    for i in range(10):
        global x, y, beta, x_tr, y_tr, x_va, y_va, x_te, y_te, input_size, output_size
        # 生成数据
        x, y, beta = get_data(random_seed=seed[i], data_times=1)
        x_tr, y_tr, x_va, y_va, x_te, y_te = split_data(x, y, data_times=1)
        input_size = x.shape[1]
        output_size = y.shape[1]

        tr_mse,va_mse,te_mse = train(i,result)
        tr.append(tr_mse)
        va.append(va_mse)
        te.append(te_mse)

    submission1 = pd.DataFrame(result)
    if not os.path.exists("./log/"+args.filename+"/log5/"):
        os.makedirs("./log/"+args.filename+"/log5/")
    submission1.to_csv("./log/"+args.filename+"/log5/result.csv", mode='w', index=False)

    if not os.path.exists("./log/"+args.filename+"/log3/"):
        os.makedirs("./log/"+args.filename+"/log3/")
    f = open("./log/"+args.filename+"/log3/res.log",'w')
    print("tr:",np.mean(np.array(tr)),np.std(np.array(tr)))
    print("va:", np.mean(np.array(va)), np.std(np.array(va)))
    print("te:", np.mean(np.array(te)), np.std(np.array(te)))
    elapsed = time.time() - start_time
    print("runtime_sec:", elapsed)
    f.write("tr:mean:"+str(np.mean(np.array(tr)))+" std:"+str(np.std(np.array(tr)))+"\n")
    f.write("va:mean"+str(np.mean(np.array(va)))+" std:"+str(np.std(np.array(va)))+"\n")
    f.write("te:mean"+str(np.mean(np.array(te)))+" std:"+str(np.std(np.array(te)))+"\n")
    f.write("runtime_sec:"+str(elapsed)+"\n")

if __name__ == '__main__':
    main()
