import pathlib

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras import layers

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
dataset.dropna()
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

    model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['mean_absolute_error', 'mean_squared_error'])

    return model

model = build_model()
# 打印模型基本信息
model.summary()

example_batch = normed_train_data[:10]
example_result = model.predict(example_batch)
print(example_result)

class PrintDot(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        if epoch % 100 == 0:
            print('')
        print('.', end='')
EPOCHS = 1000
history = model.fit(
    normed_train_data, train_labels,
    epochs=EPOCHS, validation_split=0.2, verbose=0, callbacks=[PrintDot()]
)

hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
print(hist.tail())
