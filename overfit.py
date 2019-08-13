import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)

NUM_WORDS = 10000
(train_data, train_labels), (test_data, test_labels) = keras.datasets.imdb.load_data(num_words=NUM_WORDS)

def multi_hot_sequence(sequences, dimension):
    results = np.zeros((len(sequences), dimension))
    for i, word_indices in enumerate(sequences):
        results[i,word_indices] = 1.0
    return results

train_data = multi_hot_sequence(train_data, dimension=NUM_WORDS)
test_data = multi_hot_sequence(test_data, dimension=NUM_WORDS)
# 展示第一条影评的独热编码，由于特征空间的词语是经过词频排序的，所以靠近x=0的地方有更多的y=1
plt.plot(train_data[0])
plt.show()

# 创建一个基础模型
baseline_model = keras.Sequential([
    keras.layers.Dense(16, activation=tf.nn.relu, input_shape=(NUM_WORDS,)),
    keras.layers.Dense(16, activation=tf.nn.relu),
    keras.layers.Dense(1, activation=tf.nn.sigmoid)
])
baseline_model.compile(optimizer='adam',loss='binary_crossentropy', metrics=['accuracy', 'binary_crossentropy'])
baseline_model.summary()

baseline_history = baseline_model.fit(train_data, train_labels,epochs=20, batch_size=512, validation_data=(test_data, test_labels), verbose=2)


# 创建一个小容量模型
smaller_model = keras.Sequential([
    keras.layers.Dense(4, activation=tf.nn.relu, input_shape=(NUM_WORDS, )),
    keras.layers.Dense(4, activation=tf.nn.relu),
    keras.layers.Dense(1,activation=tf.nn.sigmoid)
])

smaller_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', 'binary_crossentropy'])
smaller_model.summary()
smaller_history = smaller_model.fit(train_data, train_labels,epochs=20, batch_size=512, validation_data=(test_data,test_labels),verbose=2)

# 创建一个大模型
bigger_model = keras.Sequential([
    keras.layers.Dense(512, activation=tf.nn.relu, input_shape=(NUM_WORDS, )),
    keras.layers.Dense(512, activation=tf.nn.relu),
    keras.layers.Dense(1, activation=tf.nn.sigmoid)
])

bigger_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', 'binary_crossentropy'])
bigger_model.summary()

bigger_history = bigger_model.fit(train_data, train_labels, epochs=20, batch_size=512, validation_data=(test_data, test_labels), verbose=2)

# 构建图表, 展示训练集和验证集的损失函数值
def plot_history(histories, key='binary_crossentropy'):
    plt.figure(figsize=(16,10))
    
    for name, history in histories:
        val = plt.plot(history.epoch, history.history['val_'+key], '--', label=name.title()+' Val')
        plt.plot(history.epoch, history.history[key], color=val[0].get_color(), label=name.title()+' Train')

    plt.xlabel("Epochs")
    plt.ylabel(key.replace('_', ' ').title())
    plt.legend()

    plt.xlim([0, max(history.epoch)])
    plt.show()

plot_history([
    ('baseline', baseline_history),
    ('smaller', smaller_history),
    ('bigger', bigger_history)
])

# l2正则化的模型， 防止过拟合
# l1,l2正则化的作用， 模型的复杂度(模型的熵)由参数数量和参数值决定，无法修改参数数量就减小参数值，l1符合拉普拉斯分布会出现一些坐标轴上的值(会导致某些维度值为0)，l2正则化会使参数值一直减小(权值衰减)
l2_model = keras.Sequential([
    keras.layers.Dense(16, kernel_regularizer=keras.regularizers.l2(0.001), activation=tf.nn.relu, input_shape=(NUM_WORDS, )),
    keras.layers.Dense(16, kernel_regularizer=keras.regularizers.l2(0.001), activation=tf.nn.relu),
    keras.layers.Dense(1, activation=tf.nn.sigmoid)
])

l2_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', 'binary_crossentropy'])

l2_model_history = l2_model.fit(train_data, train_labels, epochs=20, batch_size=512, validation_data=(test_data, test_labels), verbose=2)
# 比较l2和普通模型交叉熵的收敛，可见l2正则化后，在验证集上损失函数值比普通模型更小，泛华性更好
plot_history([
    ('baseline', baseline_history),
    ('l2', l2_model_history)
])


# 使用丢弃层(Dropout)， 防止过拟合， 在验证集上的损失要小于base版本，与l2差不多
# 丢弃是有Hinton和他的学生们发明的算法， 在训练时随机把本层的某些输出置0，就是丢弃， 而在测试时不丢弃，只是把本层的输出按等同于丢弃率的比例进行缩减
dpt_model = keras.Sequential([
    keras.layers.Dense(16, activation=tf.nn.relu, input_shape=(NUM_WORDS, )),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(16, activation=tf.nn.relu),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(1, activation=tf.nn.sigmoid)
])
dpt_model.compile(optimizer='adam', loss='binary_crossentropy',metrics=['accuracy', 'binary_crossentropy'])
dpt_model_history = dpt_model.fit(train_data, train_labels, epochs=20, batch_size=512, validation_data=(test_data, test_labels), verbose=2)

plot_history([
    ('baseline', baseline_history),
    ('l2', l2_model_history),
    ('dropout', dpt_model_history)
])