#-*- coding:utf-8 -*-
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import json
import pandas as pd
from sklearn import preprocessing
from sklearn.feature_extraction import DictVectorizer

train_size = 800


def preprocess():

    df = pd.read_csv('data/train.csv')

    #変換用dict jsonを読み込む
    targets = ['Sex', 'Embarked']
    with open('dict/Sex.json') as f:
        json_sex = json.load(f)

    with open('dict/Embarked.json') as f:
        json_embarked = json.load(f)

    with open('dict/Survived.json') as f:
        json_survived = json.load(f)
    ##print json_embarked

    #挿入用DataFrameを生成
    list_sex = []
    for i, v in df['Sex'].fillna('').iteritems():
        list_sex.append(json_sex[v][1:])
    list_embarked = []
    for i, v in df['Embarked'].fillna('').iteritems():
        list_embarked.append(json_embarked[v][1:])
    ##print list_sex
    ##print list_embarked
    df_sex = pd.DataFrame(list_sex,columns=['male','female'])
    df_embarked = pd.DataFrame(list_embarked,columns=['S','C','Q'])
    ##print df_sex
    ##print df_embarked

    #全体dfにマージ
    df_preprocessed = pd.concat([df[['Age', 'Pclass', 'SibSp', 'Parch', 'Fare']], df_sex, df_embarked], axis=1).fillna(0)

    print df_preprocessed
    x_np = np.array(df_preprocessed)

    #preprocess label data
    list_survived = []
    for i, v in df['Survived'].fillna('').iteritems():
        list_survived.append(json_survived[str(v)][1:])
    y_np = np.array(list_survived)

    #train test dataset 分割
    [x_train, x_test] = np.vsplit(x_np, [train_size]) # 入力データを訓練データとテストデータに分ける
    [y_train, y_test] = np.vsplit(y_np, [train_size]) # ラベルを訓練データをテストデータに分ける
    return [x_train, x_test], [y_train, y_test]


def inference(inputs, isTrain):
    fc1 = tf.layers.dense(inputs=inputs, units=100, activation=tf.nn.leaky_relu,
                          bias_initializer=tf.truncated_normal_initializer,
                          activity_regularizer=tf.contrib.layers.l2_regularizer, name="fc1")
    drop1 = tf.layers.dropout(fc1, rate=0.5, training=isTrain)
    fc2 = tf.layers.dense(inputs=drop1, units=100, activation=tf.nn.leaky_relu,
                          bias_initializer=tf.truncated_normal_initializer,
                          activity_regularizer=tf.contrib.layers.l2_regularizer, name="fc2")
    drop2 = tf.layers.dropout(fc2, rate=0.5, training=isTrain)

    output = tf.layers.dense(inputs=drop2, units=2, activation=None, name="output")
    return output


def loss(truth, predict):
    losses = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=predict, labels=truth))
    return losses


def training(losses):
    #return tf.train.MomentumOptimizer(learning_rate=0.01, momentum=0.25).minimize(losses)
    return tf.train.AdamOptimizer(learning_rate=0.0005).minimize(losses)


def main(argv=None):
    x_train, y_train = preprocess()
    x = tf.placeholder(tf.float32, shape=(None, 10), name='inputs')
    y = tf.placeholder(tf.float32, shape=(None, 2), name='truth')
    train = tf.placeholder(tf.bool, name='isTrain')
    batch_size = 100
    predict = inference(x, train)

    losses = loss(y, predict)

    train_step = training(losses)

    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(100):
            for i in range(100000):
                ind = np.random.choice(train_size, batch_size)
                x_train_batch = x_train[0][ind]
                y_train_batch = y_train[0][ind]

                sess.run(train_step, feed_dict={x: x_train_batch, y: y_train_batch, train: True})
                if i % 1000 == 0:
                    loss_val = sess.run(losses, feed_dict={x: x_train_batch, y: y_train_batch, train: False})
                    print ('Step:%d, Loss:%f' % (i, loss_val))

                if i % 10000 == 0:
                    saver.save(sess, 'ckpt/model', global_step=i)

                    correct_prediction = tf.equal(tf.argmax(predict, 1), tf.argmax(y, 1))
                    # Calculate accuracy
                    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
                    print "Accuracy:", accuracy.eval({x: x_train[1], y: y_train[1], train: False})


if __name__ == '__main__':
    main()
