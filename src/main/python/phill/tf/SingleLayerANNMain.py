import numpy as np
import tensorflow as tf
import src.main.python.phill.tf.MySubjects as util


def test_train_indices(n, batch_size, test_to_train_ratio):
    indices = np.random.choice(n, size=[n], replace=False)
    xs = []
    for i in range(int(n / batch_size)):
        start_incl = i * batch_size
        end_excl = (i+1) * batch_size
        print("(%d, %d] in %s" % (start_incl, end_excl, np.shape(indices)))
        batch = indices[start_incl:end_excl]
        end_test_incl = int(batch_size * test_to_train_ratio / (test_to_train_ratio + 1))
        test = batch[0:end_test_incl]
        train = batch[end_test_incl:-1]
        xs.append((test, train))
    return xs


def train_and_test_in_batches(x, out, y, sparse_tfidf_texts, targets, layer_1):
    #(optimizer, loss) = util.optimiser_loss(out, y, learning_rate=0.01)

    # ValueError: Only call `softmax_cross_entropy_with_logits` with named arguments (labels=..., logits=..., ...)
    #loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(out, y))


    loss = tf.reduce_mean(tf.abs(y - out))
    my_opt = tf.train.AdamOptimizer(0.005)
    optimizer = my_opt.minimize(loss)

    reg_lambda = 0.1
    diff_plus_regularization = tf.add(tf.reduce_sum(tf.square(y - out)), tf.multiply(reg_lambda, tf.reduce_sum(tf.square(out))))
    print(x.shape[1], x.shape[0])
    loss = tf.div(diff_plus_regularization, 20000)

    epoch = 1000
    batch_size = 128

    accuracy = util.accuracy_fn(out, y)

    testing_training = test_train_indices(len(targets), 1000, 2.0)

    all_test = range(sparse_tfidf_texts.shape[0])

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(epoch):
            print("epoch %d: training..." % i)
            i_batch = 0
            for (test_indices, train_indices) in testing_training:
                rand_index = np.random.choice(train_indices, size=batch_size)
                rand_x = sparse_tfidf_texts[rand_index].todense()
                rand_y = util.one_hot(rand_index, out.shape[1], targets)
                f_dict = {x: rand_x, y: rand_y}
                sess.run([loss, optimizer], feed_dict=f_dict)
                if (i+2) % 10 == 0:
                    print("accuracy", sess.run(accuracy, feed_dict=f_dict), "loss", sess.run(loss, feed_dict=f_dict))
                    i_batch = i_batch + 1
            if ((i+1) % 10 == 0):
                print("Testing on all data...")
                acc = sess.run(accuracy, feed_dict={x: sparse_tfidf_texts[all_test].todense(), y: util.one_hot(all_test, out.shape[1], targets)})
                print("batch accuracy", acc)



def train_and_test(nn_init_fn):
    n_features = 9000
    (sparse_tfidf_texts, targets) = util.do_tf_idf(n_features)
    output_size = len(util.subjects)

    (x, out, y, layer1) = nn_init_fn(n_features, output_size)
    train_and_test_in_batches(x, out, y, sparse_tfidf_texts, targets, layer1)


if __name__ == '__main__':
    train_and_test(util.neural_net) # about 70%
    # train_and_test(util.neural_net_w_hidden)


