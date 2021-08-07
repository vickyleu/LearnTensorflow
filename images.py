import tensorflow as tf
from tensorflow import keras
# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager


def detected_text_sets():
    a = sorted([f.name for f in matplotlib.font_manager.fontManager.ttflist])
    for i in a:
        print(i)


def load_image_mnist():
    fashion_mnist = keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    print(len(train_labels))
    print(train_images.shape)
    train_images = train_images / 255.0
    test_images = test_images / 255.0

    plt.rcParams['font.sans-serif'] = ['PingFang HK']  # 显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 这两行需要手动设置
    training(train_images, train_labels, test_images, class_names)


def training(train_images, train_labels, test_images, class_names):
    # 设置层
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(10)
    ])
    # 编译模型配置
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    # 向模型馈送数据
    model.fit(train_images, train_labels, epochs=10)

    # # 评估准确率
    # test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
    #
    # print('\nTest accuracy:', test_acc)
    # 进行预测
    probability_model = tf.keras.Sequential([model,
                                             tf.keras.layers.Softmax()])
    predictions = probability_model.predict(test_images)
    draw_result(predictions, train_labels, test_images, class_names)


def draw_result(predictions, test_labels, test_images, class_names):
    num_rows = 5
    num_cols = 3
    num_images = num_rows * num_cols
    # figsize 长宽像素
    plt.figure(figsize=(2 * 2 * num_cols, 2 * num_rows))
    for i in range(num_images):
        # subplot 指定行或列,并移动到相应索引位置
        plt.subplot(num_rows, 2 * num_cols, 2 * i + 1)
        plot_image(i, predictions[i], test_labels, test_images, class_names)
        plt.subplot(num_rows, 2 * num_cols, 2 * i + 2)
        plot_value_array(i, predictions[i], test_labels)
    plt.tight_layout()
    plt.show();


def plot_image(idx, predictions_array, true_label, img, class_names):
    predictions_array, true_label, img = predictions_array, true_label[idx], img[idx]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    # 画出左侧模型样本图片
    plt.imshow(img, cmap=plt.cm.binary)

    # np.argmax预测模型取最大置信度值,得到预测文本
    predicted_label = np.argmax(predictions_array)

    # 比较预测类型和真实类型是否是一样的
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    # np.max预测模型取最值,由轴方向决定最大或最小,默认为Row(跨行)方向
    plt.xlabel("嘿嘿嘿{} {:2.0f}%".format(class_names[predicted_label],
                                       100 * np.max(predictions_array)),
               color=color);


def plot_value_array(idx, predictions_array, true_label):
    predictions_array, true_label = predictions_array, true_label[idx]
    plt.grid(False)
    # 横坐标标线范围10位
    plt.xticks(range(10))
    # 纵坐标不画标线
    plt.yticks([])
    # 指定一个柱状图
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    # 限制绘制上下限
    plt.ylim([0, 1])
    # np.argmax预测模型取最大置信度值,得到预测文本
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue');
