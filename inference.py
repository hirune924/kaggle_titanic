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

    df = pd.read_csv('data/test.csv')

    #変換用dict jsonを読み込む
    targets = ['Sex', 'Embarked']
    with open('dict/Sex.json') as f:
        json_sex = json.load(f)

    with open('dict/Embarked.json') as f:
        json_embarked = json.load(f)
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

    return x_np


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

    output_softmax = tf.nn.softmax(output)
    output_argmax = tf.argmax(output, 1)
    return output_argmax


def main(argv=None):
    x_test = preprocess()
    x = tf.placeholder(tf.float32, shape=(None, 10), name='inputs')
    train = tf.placeholder(tf.bool, name='isTrain')
    batch_size = 100
    predict = inference(x, train)

    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, "ckpt/model-9-70000")

        for i, inp in enumerate(x_test):
            ##x_batch = [inp]
            ##print x_batch
            inf = sess.run(predict, feed_dict={x: [inp], train: False})
            #print i,int(inf[0,0]*100),int(inf[0,1]*100)
            print inf[0]

if __name__ == '__main__':
    main()
