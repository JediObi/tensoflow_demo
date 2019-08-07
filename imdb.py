import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt

imdb = keras.datasets.imdb

# num_words 保留出现频次在前10000的字词，其他罕见字词会被舍弃。
# 影评的字词被数字替代
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)
word_index = imdb.get_word_index()

# 遍历字典的k词语和v索引，然后所有索引+3，因为要另行指定四个标志词语索引0-3
word_index = {k:(v+3) for k,v in word_index.items()}
# 插入三个特殊词
word_index['<PAD>'] = 0
word_index['<START>'] = 1
word_index['<UNK>'] = 2
word_index['<UNUSED>'] = 3

# 把经过以上处理的dict key与value交换
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()] )
# 遍历一条影评数据，把数字替换为对应的词，如果索引不存在则返回问号?
def decode_review(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])

print(decode_review(train_data[0]))

# 由于影评的长短不一， 神经网络需要相同长度的输入
# 使用最大长度填充，影评有最大长度，以此为标准，把长度不足的后续补0
# 在官网的demo中提到过独热编码（但是没使用），因为独热编码会把一个特征值序列化为特征空间长度的二进制，索引位为1，其他位为0
# 但是不理解官网的解释，为什么独热编码是num_words*num_reviews的张量，而填充则是man_length*num_reviews
# 独热编码: 比如某个特征空间有10个值，那么该特征出现时，可以用一个10位寄存器表示(索引位为1，其他位为0)
train_data = keras.preprocessing.sequence.pad_sequences(train_data, value=word_index["<PAD>"], padding="post", maxlen=256)

test_data = keras.preprocessing.sequence.pad_sequences(test_data, value=word_index["<PAD>"], padding="post", maxlen=256)

# 有效词汇数量
vocab_size = 10000

model = keras.Sequential()
# embedding层可以把索引转换成向量
# 具体的实现就是把特征空间转换为嵌入矩阵，然后输入样本里的每个特征值转换成嵌入矩阵里的索引
# 比如某个one-hot编码的10000维度特征值 [1,0,0,...]转换后在嵌入矩阵中变成[25,26]，然后再转换成一个索引指向嵌入矩阵中的[25,26]这行
# 可以对输入的稀疏矩阵降维，比如语言处理常用的one-hot编码
model.add(keras.layers.Embedding(vocab_size, 16))

# 时域信号全局平均池化 可以理解为计算领域平均值来降维
model.add(keras.layers.GlobalAveragePooling1D())

# 使用relu做激活函数,relu的特点是函数和微分的数学形式都很简单，输入为正时计算速度很快
model.add(keras.layers.Dense(16, activation=tf.nn.relu))
# 最后使用sigmoid做激活函数
model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))
# 打印模型
model.summary()

model.compile(optimizer=tf.train.AdamOptimizer(), loss="binary_crossentropy", metrics=['accuracy'])

x_val = train_data[:10000]
partial_x_train = train_data[10000:]

y_val = train_labels[:10000]
partial_y_train = train_labels[10000:]

# 迭代40次，数据分成512大小的batch处理，一次迭代，将会执行batch数量的梯度下降，使用验证集验证，verbose打印训练细节
history = model.fit(partial_x_train, partial_y_train, epochs=40, batch_size=512, validation_data=(x_val, y_val), verbose=1)

# 评估模型
results = model.evaluate(test_data, test_labels)
print(results)

# 创建准确率和损失值随时间变化的图
history_dict = history.history
# 准确率
acc = history_dict['acc']
# 验证集准确率
val_acc = history_dict['val_acc']
# 损失值
loss = history_dict['loss']
# 验证集损失值
val_loss = history_dict['val_loss']

# 迭代次数， [start, stop)
epochs = range(1, len(acc)+1)

# 损失值随迭代次数变化，蓝色点
plt.plot(epochs, loss, 'bo', label='Training loss')

# 验证集损失值随迭代次数的变化，蓝色实线
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
# 默认所有图例
plt.legend()

plt.show()

# 创建准确率随迭代次数的变化图
plt.clf()
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation loss')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()