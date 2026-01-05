import os
from operator import itemgetter

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data_path = "./log/log12.csv"

data = pd.read_csv(data_path)
data = pd.DataFrame(data)
units = []
y_train = []
y_valid = []
y_test = []
q = []
# data = sorted(data, key=itemgetter("units"))
for k, v in data.items():
    units.append(int(k))
    y_train.append(v[0])
    y_valid.append(v[1])
    y_test.append(v[2])
    q.append([3])

# sorted the list
idx = np.argsort(units)
units = np.array(units)[idx]
y_train = np.array(y_train)[idx]
y_valid = np.array(y_valid)[idx]
y_test = np.array(y_test)[idx]
q = np.array(q)[idx]

plt.xlabel("Number of Nodes")
plt.ylabel("Minimum MSE")
plt.title("p=30")
plt.plot(units, y_train, color="red", marker="+", linestyle='-', lw=0.5, label="train")
plt.plot(units, y_valid, color="blue", marker="v", linestyle='--', lw=0.8, label="valid")
plt.plot(units, y_test, color="magenta", marker="^", linestyle='-', lw=0.5, label="test")
xticklable = [str(i) for i in list(units)]
plt.xticks(units, xticklable, size=12, color='black')
#plt.ylim((0.025,0.15))
plt.legend(loc="best")
file_name = "./log/log12.png"
plt.savefig(file_name)
plt.show()

# plt.xlabel("Trial")
# plt.ylabel("Units")
# plt.title("Our method")
# plt.plot(units, color="blue", marker="+", linestyle='-', lw=0.5)
# # yticklable = [str(i) for i in list(units)]
# # plt.yticks(units, yticklable, size=12, color='black')
# # plt.legend(loc="best")
# file_name = "./text.pdf"
# plt.savefig(file_name)
# plt.show()

if __name__ == '__main__':
    pass
