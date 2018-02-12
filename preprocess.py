#-*- coding:utf-8 -*-
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
from sklearn import preprocessing
from sklearn.feature_extraction import DictVectorizer


def preprocess():
    train_size = 750

    df = pd.read_csv('data/train.csv')

    labelEncoder = preprocessing.LabelEncoder()

    df['Sex'] = labelEncoder.fit_transform(df['Sex'])
    df['Cabin'] = labelEncoder.fit_transform(df['Cabin'])
    df['Embarked'] = labelEncoder.fit_transform(df['Embarked'])

    x_np = np.array(df[['Age', 'Pclass', 'Sex', 'SibSp', 'Parch', 'Fare', 'Cabin', 'Embarked']].fillna(0))

    d = df[['Survived']].to_dict('record')
    vectorizer = DictVectorizer(sparse=False)
    y_np = vectorizer.fit_transform(d)
    y_np = np.eye(2)[y_np.astype(np.int64)]

    [x_train, x_test] = np.vsplit(x_np, [train_size]) # 入力データを訓練データとテストデータに分ける
    [y_train, y_test] = np.vsplit(y_np, [train_size]) # ラベルを訓練データをテストデータに分ける
    return [x_train, x_test], [y_train, y_test]


def inference(inputs):
    fc1 = tf.layers.dense(inputs=inputs, units=100, activation=tf.nn.leaky_relu,
                          bias_initializer=tf.truncated_normal_initializer, name="fc1")
    fc2 = tf.layers.dense(inputs=fc1, units=100, activation=tf.nn.leaky_relu,
                          bias_initializer=tf.truncated_normal_initializer, name="fc2")
    output = tf.layers.dense(inputs=fc2, units=2, activation=None, name="output")
    return output


def loss(truth, predict):
    losses = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=predict, labels=truth))
    return losses


def training(losses):
    return tf.train.AdamOptimizer(learning_rate=0.01).minimize(losses)


def main(argv=None):
    x_train, y_train = preprocess()
    x = tf.placeholder(tf.float32, shape=(None, 8), name='inputs')
    y = tf.placeholder(tf.float32, shape=(None, 1, 2), name='truth')
    batch_size = 100
    predict = inference(x)

    losses = loss(y, predict)

    train_step = training(losses)

    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(100):
            for i in range(100000):
                ind = np.random.choice(750, batch_size)
                x_train_batch = x_train[0][ind]
                y_train_batch = y_train[0][ind]

                sess.run(train_step, feed_dict={x: x_train_batch, y: y_train_batch})
                if i % 1000 == 0:
                    loss_val = sess.run(losses, feed_dict={x: x_train_batch, y: y_train_batch})
                    print ('Step:%d, Loss:%f' % (i, loss_val))

                if i % 10000 == 0:
                    saver.save(sess, 'ckpt/model', global_step=i)

                    correct_prediction = tf.equal(tf.argmax(predict, 1), tf.argmax(y[0], 1))
                    
                    # Calculate accuracy
                    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
                    print "Accuracy:", accuracy.eval({x: x_train[1], y: y_train[1]})


if __name__ == '__main__':
    main()
