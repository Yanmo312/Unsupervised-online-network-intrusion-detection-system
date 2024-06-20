import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
import KitNET as kit
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


# 创建真实标签数据
y_true = np.zeros(X.shape[0])
y_true[FMgrace+ADgrace+1:] = 1 # 后面的观测值为异常

# 计算模型输出的异常分数
anomaly_scores = RMSEs

# 计算准确率、精确率、召回率和 F1 分数
accuracy = accuracy_score(y_true, anomaly_scores > 0)
precision = precision_score(y_true, anomaly_scores > 0)
recall = recall_score(y_true, anomaly_scores > 0)
f1 = f1_score(y_true, anomaly_scores > 0)

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

# 绘制 ROC 曲线并计算 AUC 值
fpr, tpr, thresholds = roc_curve(y_true, anomaly_scores)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (AUC = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()

print(f"AUC: {roc_auc:.4f}")
