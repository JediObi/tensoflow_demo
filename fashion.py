import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt

# 打印tensorflow版本
print("tensorflow版本: ",tf.__version__)

# 导入数据集
fashion_mnist = keras.datasets.fashion_mnist

# 加载训练集和验证集
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# 衣物名称，样本标签是0-9的数字不含语义
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# plt.figure()
# plt.imshow(train_images[0])
# plt.colorbar()
# plt.grid(True)
# plt.show()

# 像素规则化
train_images = train_images / 255.0
test_images = test_images / 255.0

# 创建模型，Flatten层第一层输入层扁平化多维数组为一维， Dense第一层全连接层，128个神经元所以输出也是128个使用relu，第三层Dense10个神经元作为输出层使用softmax计算概率
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(128, activation = tf.nn.relu),
    keras.layers.Dense(10, activation = tf.nn.softmax)
])
# 编译模型，损失函数，优化器，评估方式数组（当前数组只有accuracy表示只使用成功率）
# metrics 评估函数类似于损失函数，但评估函数不参与模型更新
# 评估函数在一个训练迭代结束时评估当前模型的性能，比如accuracy会计算当前模型在本次训练集合上的正确率。不同的评估函数关注点会有不同。
model.compile(optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# 训练模型 训练数据集， 训练集的标签， 迭代次数
model.fit(train_images, train_labels, epochs=5)

# 在测试机上评估模型
test_loss, test_acc = model.evaluate(test_images, test_labels)

print('Test loss: ', test_loss)
print('Test accuracy: ', test_acc)

# 进行预测
predictions = model.predict(test_images)
# 打印第一张图片的预测结果(由于输出层采用softmax，所以会输出所有分类的概率)
print("first prediction: ",predictions[0])
# 打印第一张图预测概率最高的下标，也就是打印最高概率预测分类
print("first prediction: ", np.argmax(predictions[0]))
# 打印第一张图的真实标签
print("first real label: ", test_labels[0])

# 打印图片和预测结果（最高概率分类），并把预测结果语义化

def plot_image(i, predictions_array, true_label, img):
    predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'
    
    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],100*np.max(predictions_array),class_names[predicted_label]), color=color)

def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array[i], true_label[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    thisplot = plt.bar(range(10),predictions_array, color = '#777777')
    plt.ylim([0,1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('green')

i = 0
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i,predictions, test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i,predictions,test_labels)
plt.show()

    