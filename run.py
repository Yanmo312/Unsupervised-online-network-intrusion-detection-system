import KitNET as kit
import numpy as np
import pandas as pd
import time
from scipy.stats import norm
from matplotlib import pyplot as plt
import zipfile

print("解压样本数据集中")
with zipfile.ZipFile("dataset.zip", "r") as zip_ref:
    zip_ref.extractall()
print("读取样本数据集中")
X = pd.read_csv("mirai3.csv", header=None).values

maxAE = 10
FMgrace = 5000
ADgrace = 50000

K = kit.KitNET(X.shape[1], maxAE, FMgrace, ADgrace)
RMSEs = np.zeros(X.shape[0])
print("启动KitNET模型:")
start = time.time()
for i in range(X.shape[0]):
    if i % 1000 == 0:
        print(i)
    RMSEs[i] = K.process(X[i,])
stop = time.time()
print("已完成！耗时: " + str(stop - start))

benignSample = np.log(RMSEs[FMgrace + ADgrace + 1:71000])
logProbs = norm.logsf(np.log(RMSEs), np.mean(benignSample), np.std(benignSample))

print("绘制结果图：")

plt.figure(figsize=(10, 5))
timestamps = pd.read_csv("mirai3_ts.csv", header=None).values
fig = plt.scatter(timestamps[FMgrace + ADgrace + 1:], RMSEs[FMgrace + ADgrace + 1:], s=0.1,
                  c=logProbs[FMgrace + ADgrace + 1:], cmap='RdYlGn')
plt.yscale("log")
plt.title("Anomaly Scores from KitNET's Execution Phase")
plt.ylabel("RMSE (log scaled)")
plt.xlabel("Time elapsed [min]")
plt.annotate('Mirai C&C channel opened [Telnet]', xy=(timestamps[71662], RMSEs[71662]), xytext=(timestamps[58000], 1),
             arrowprops=dict(facecolor='black', shrink=0.05), )
plt.annotate('Mirai Bot Activated\nMirai scans network for vulnerable devices', xy=(timestamps[72662], 1),
             xytext=(timestamps[55000], 5), arrowprops=dict(facecolor='black', shrink=0.05), )
figbar = plt.colorbar()
figbar.ax.set_ylabel('Log Probability\n ', rotation=270)
plt.show()
