import tensorflow as tf


def preprocess(line):
    #clms = line.decode('utf-8').split(',')
    line = tf.Print([line], [line])
    return line

#sess = tf.InteractiveSession()

dataset = tf.data.TextLineDataset("data/test.csv")\
        .skip(1)\
        .map(lambda x: preprocess(x))\
        .shuffle(4)\
        .batch(1)


iterator = dataset.make_initializable_iterator()
next_elem = iterator.get_next()
#sess.run(iterator.initializer)
#for i in range(10000):
#    print(sess.run(next_elem))
num_batch = 0
with tf.train.MonitoredTrainingSession() as sess:
    for epoch in range(100):
        sess.run(iterator.initializer)
        while not sess.should_stop():
            value = sess.run(next_elem)
            num_batch += 1
            print("Num Batch: ", num_batch)
            print(value)
