import os
import re
path = './res/'
# 用于返回指定的文件夹包含的文件或文件夹的名字的列表
parents = os.listdir(path)
print(path)
dic = {}

for parent in parents:
    # 把目录和文件名合并为一个路径
    child = os.path.join(path,parent)
    if child.endswith(".log"):
        # 用正则表达式把中间的第一层节点数取出来
        res = re.findall('h1:(\d{1,3})',child)

        if res[0] not in dic:
            dic[res[0]] = []
        f = open(child)
        lines = f.readlines()

        # 取出关心的那一行
        line = lines[-2]
        # 正则表达取出数字
        res2 = re.findall('(\(\d.\d{4}\))', line)
        dic[res[0]].append(res2[0])
        f.close()

# 按照keys排序
sorted(dic,reverse=True)
for k in dic:
    print(k," ",dic[k])

'''
epoch 100:
[20, 8, 12, '窗口左滑', 5, '窗口左滑', 3, '窗口右滑', 6, '窗口左滑', 6, '节点-1', 5, '节点-1', 4]
epoch 1000:
[20, 8, 12, '窗口右滑', 15, '窗口左滑', 11, '窗口右滑', 13, '窗口左滑', 13, '节点-1', 12, '节点-1', 11, '节点-1', 10, '节点-1', 9, '节点-1', 8, '节点-1', 7, '节点-1', 6, '节点-1', 5, '节点-1', 4]
'''
