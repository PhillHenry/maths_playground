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


if __name__ == '__main__':
    n_features = 9000
    (sparse_tfidf_texts, targets) = util.do_tf_idf(n_features)
    output_size = len(util.subjects)

    (x, out, y) = util.neural_net(n_features, output_size)

    (optimizer, loss) = util.optimiser_loss(out, y)

    epoch = 1000
    batch_size = 10

    accuracy = util.accuracy_fn(out, y)

    testing_training = test_train_indices(len(targets), 1000, 1.0)

    i_batch = 0
    with tf.Session() as sess:
        print("training...")
        sess.run(tf.global_variables_initializer())
        for (test_indices, train_indices) in testing_training:
            for i in range(epoch):
                rand_index = np.random.choice(train_indices, size=batch_size)
                rand_x = sparse_tfidf_texts[rand_index].todense()
                rand_y = util.one_hot(rand_index, output_size, targets)
                f_dict = {x: rand_x, y: rand_y}
                sess.run([loss, optimizer], feed_dict=f_dict)
                if (i+1) % 100 == 0:
                    print("accuracy", sess.run(accuracy, feed_dict=f_dict), "loss", sess.run(loss, feed_dict=f_dict))
            print("batch %d finished" % i_batch)
            i_batch = i_batch + 1
            acc = sess.run(accuracy, feed_dict={x: sparse_tfidf_texts[test_indices].todense(), y: util.one_hot(test_indices, output_size, targets)})
            print("batch accuracy", acc)

