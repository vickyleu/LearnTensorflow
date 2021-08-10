import pathlib

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers


def load_data_predict():
    print(tf.__version__)
    # Auto MPG 数据集
    dataset_path = keras.utils.get_file("auto-mpg.data", "http://archive.ics.uci.edu/ml/machine-learning-databases"
                                                         "/auto-mpg/auto-mpg.data")
    # 使用 pandas 导入数据集
    column_names = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight',
                    'Acceleration', 'Model Year', 'Origin']
    raw_dataset = pd.read_csv(dataset_path, names=column_names,
                              na_values="?", comment='\t',
                              sep=" ", skipinitialspace=True)

    package_data(raw_dataset)


def package_data(raw_dataset):
    dataset = raw_dataset.copy()
    dataset.tail()
    # 数据清洗
    dataset.isna().sum()
    # 为了保证这个初始示例的简单性，删除这些行
    dataset = dataset.dropna()
    # 转换为独热码 （one - hot）
    origin = dataset.pop('Origin')
    dataset['USA'] = (origin == 1) * 1.0
    dataset['Europe'] = (origin == 2) * 1.0
    dataset['Japan'] = (origin == 3) * 1.0
    dataset.tail()
    # 拆分训练数据集和测试数据集
    train_dataset = dataset.sample(frac=0.8, random_state=0)
    test_dataset = dataset.drop(train_dataset.index)
    # 数据检查
    sns.pairplot(train_dataset[["MPG", "Cylinders", "Displacement", "Weight"]], diag_kind="kde")
    # 查看总体的数据统计
    train_stats = train_dataset.describe()
    train_stats.pop("MPG")
    train_stats = train_stats.transpose()
    train_stats
    # 从标签中分离特征
    train_labels = train_dataset.pop('MPG')
    test_labels = test_dataset.pop('MPG')

    # 数据规范化
    def norm(x):
        return (x - train_stats['mean']) / train_stats['std']

    normed_train_data = norm(train_dataset)
    normed_test_data = norm(test_dataset)
    # 构建模型
    train(normed_train_data, normed_test_data, train_labels, test_labels,train_dataset)


def build_model(train_dataset):
    _model = keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=[len(train_dataset.keys())]),
        layers.Dense(64, activation='relu'),
        layers.Dense(1)
    ])

    optimizer = tf.keras.optimizers.RMSprop(0.001)

    _model.compile(loss='mse',
                   optimizer=optimizer,
                   metrics=['mae', 'mse'])
    return _model


def train(normed_train_data, normed_test_data, train_labels, test_labels, train_dataset):
    model = build_model(train_dataset)
    # 检查模型
    model.summary()
    # 从训练数据中批量获取‘10’条例子并对这些例子调用 model.predict
    example_batch = normed_train_data[:10]
    example_result = model.predict(example_batch)
    example_result

    # 通过为每个完成的时期打印一个点来显示训练进度
    class PrintDot(keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs):
            if epoch % 100 == 0: print('')
            print('.', end='')

    EPOCHS = 1000
    # 对模型进行1000个周期的训练，并在 history 对象中记录训练和验证的准确性
    history = model.fit(
        normed_train_data, train_labels,
        epochs=EPOCHS, validation_split=0.2, verbose=0,
        callbacks=[PrintDot()])
    # 使用 history 对象中存储的统计信息可视化模型的训练进度
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch
    hist.tail()

    plot_history(history)

    model = build_model(train_dataset)

    # patience 值用来检查改进 epochs 的数量
    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

    history = model.fit(normed_train_data, train_labels, epochs=EPOCHS,
                        validation_split=0.2, verbose=0, callbacks=[early_stop, PrintDot()])

    plot_history(history)

    loss, mae, mse = model.evaluate(normed_test_data, test_labels, verbose=2)

    print("Testing set Mean Abs Error: {:5.2f} MPG".format(mae))
    # 最后，使用测试集中的数据预测 MPG 值
    test_predictions = model.predict(normed_test_data).flatten()

    plt.scatter(test_labels, test_predictions)
    plt.xlabel('True Values [MPG]')
    plt.ylabel('Predictions [MPG]')
    plt.axis('equal')
    plt.axis('square')
    plt.xlim([0, plt.xlim()[1]])
    plt.ylim([0, plt.ylim()[1]])
    _ = plt.plot([-100, 100], [-100, 100])
    # 误差分布
    error = test_predictions - test_labels
    plt.hist(error, bins=25)
    plt.xlabel("Prediction Error [MPG]")
    _ = plt.ylabel("Count")
    plt.show()


def plot_history(history):
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Abs Error [MPG]')
    plt.plot(hist['epoch'], hist['mae'],
             label='Train Error')
    plt.plot(hist['epoch'], hist['val_mae'],
             label='Val Error')
    plt.ylim([0, 5])
    plt.legend()

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Square Error [$MPG^2$]')
    plt.plot(hist['epoch'], hist['mse'],
             label='Train Error')
    plt.plot(hist['epoch'], hist['val_mse'],
             label='Val Error')
    plt.ylim([0, 20])
    plt.legend()
    plt.show()
