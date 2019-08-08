import pathlib

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

print(tf.__version__)

dataset_path = keras.utils.get_file("auto-mpg.data","http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data")

# 设置列名
column_names = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight', 'Acceleration', 'Model Year', 'Origin']
# 读入csv数据，指定列名，na值指定?，
raw_dataset = pd.read_csv(dataset_path, names=column_names, na_values= "?", comment='\t', sep=" ", skipinitialspace=True)
# 
dataset = raw_dataset.copy()
# 打印最后几条数据
print(dataset.tail())
# 统计空数据
print(dataset.isna().sum())
# 删除空数据
dataset = dataset.dropna()
# 弹出Origin列
origin = dataset.pop('Origin')
# 对origin列数据处理， 分成USA, Europe，Japan，并转换成one-hot编码
dataset['USA'] = (origin == 1)*1.0
dataset['Europe'] = (origin == 2)*1.0
dataset['Japan'] = (origin == 3)*1.0
print(dataset.tail(1))

# 取80%作为训练集
train_dataset = dataset.sample(frac=0.8, random_state=0)
# 剩余部分作为验证集
test_dataset = dataset.drop(train_dataset.index)

# sns pairplot 特征对比图， 对比4列, diag_kind单变量对比设定(自己与自己比较的设定),kde(单变量使用线形图，其他使用散点)
sns.pairplot(train_dataset[["MPG", "Cylinders", "Displacement", "Weight"]], diag_kind="kde")
plt.show()

# 展示一些统计数据
print("statistics...")
train_stats = train_dataset.describe()
train_stats.pop("MPG")
train_stats = train_stats.transpose()
print(train_stats)

# 提取标签, mpg 油耗(每加仑汽油行驶里程数可以看作是热机效率)
train_labels = train_dataset.pop("MPG")
test_labels = test_dataset.pop("MPG")

# 标准化数据，虽然normalization是归一化的引文，但是此处操作是标准化，均值为0，方差为1的处理
# 标准化可以用于加速损失函数收敛，因为降低了分布的偏移
# 归一化用于去除不同维度量纲和量纲单位，比如特征x值一百万，而特征y值为10，显然x的影响要更大，所以归一化，消除影响
# 正则化， 避免过拟合，选择经验风险和模型复杂度同时较小的模型
def norm(x):
    return (x-train_stats['mean'])/train_stats['std']
normed_train_data = norm(train_dataset)
normed_test_data = norm(test_dataset)

def build_model():
    model = keras.Sequential([
        layers.Dense(64, activation=tf.nn.relu, input_shape=[len(train_dataset.keys())]),
        layers.Dense(64, activation=tf.nn.relu),
        layers.Dense(1)
    ])

    # 使用RMSprop 随机梯度下降
    optimizer = tf.keras.optimizers.RMSprop(0.001)
    # 损失函数使用mse， 评估使用mae和mse（将会记录每次迭代这两个值）
    model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['mean_absolute_error', 'mean_squared_error'])

    return model

model = build_model()
# 打印模型基本信息
model.summary()

# 提取前十个， 用初始模型预测下
example_batch = normed_train_data[:10]
example_result = model.predict(example_batch)
print(example_result)

# 设置一个回调，打印进度
class PrintDot(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        if epoch % 100 == 0: print('')
        print('.', end='')
EPOCHS = 1000
# 设置验证集为训练集的20%
history = model.fit(
    normed_train_data, train_labels,
    epochs=EPOCHS, validation_split=0.2, verbose=0, callbacks=[PrintDot()]
)

print("\n")
# 把训练history转换成DataFrame
hist = pd.DataFrame(history.history)
# 把迭代次数传给DataFrame
hist['epoch'] = history.epoch
print(hist.tail())

# 打印训练过程中的评估值mse和mae
def plot_history(history):
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch

    # 创建一个新窗口
    plt.figure()
    # 迭代次数为x轴
    plt.xlabel("Epoch")
    # y轴展示mae
    plt.ylabel("Mean Abs Error [MPG]")
    # 绘制一个实线图 训练集上的epoch-mae
    plt.plot(hist['epoch'], hist['mean_absolute_error'], label='Train Error')
    # 绘制一个实线图 验证集上的epoch-mae
    plt.plot(hist['epoch'], hist['val_mean_absolute_error'], label='Val Error')
    # y轴0-5
    plt.ylim([0, 5])
    plt.legend()

    # 创建新窗口展示训练集和验证集各自的epoch-mse
    plt.figure()
    plt.xlabel("Epoch")
    plt.ylabel("Mean Square Error [MPG]")
    plt.plot(hist['epoch'], hist['mean_squared_error'], label='Train Error')
    plt.plot(hist['epoch'], hist['val_mean_squared_error'], label='Val Error')
    plt.ylim([0, 20])
    plt.legend()
    plt.show()

plot_history(history)

# 由于第一个模型在验证集上出现了mse和mae上升，所以认为出现过拟合
# 新的模型增加一个early_stop回调，用于在patience迭代内验证集损失函数不下降就提前终止训练，以防止过拟合
model2 = build_model()
early_stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=10)
history2 = model2.fit(normed_train_data, train_labels, epochs=EPOCHS, validation_split=0.2, verbose=0, callbacks=[early_stop, PrintDot()])

plot_history(history2)

# 使用测试集评估下新模型
loss, mae, mse = model2.evaluate(normed_test_data, test_labels, verbose=0)
# 打印mae
print("Testing set Mean Absolute Error: {:5.2f} MPG".format(mae))

# 使用新模型预测测试集
test_predictions = model2.predict(normed_test_data).flatten()
# 绘制测试集y-y^散点
plt.scatter(test_labels, test_predictions)
plt.xlabel("True Values [MPG]")
plt.ylabel("Predictions [MPG]")
# 两个坐标轴刻度相同
plt.axis("equal")
# 坐标系限制为方形
plt.axis("square")
plt.xlim([0, plt.xlim()[1]])
plt.ylim([0, plt.ylim()[1]])
# 绘制一个45度分界线，因为真实值和预测值基本接近，所以应该都聚拢在这条分界线上
_ = plt.plot([-100, 100], [-100, 100])
plt.show()

# 预测差值的分布柱状图，基本上是一个均值为0的高斯分布，符合训练的预期
error = test_predictions - test_labels
plt.hist(error, bins=25)
plt.xlabel("Prediction Error [MPG]")
_ = plt.ylabel("Count")
plt.show()