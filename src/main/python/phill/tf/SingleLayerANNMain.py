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


def train_and_test_in_batches(x, out, y, sparse_tfidf_texts, targets):
    #(optimizer, loss) = util.optimiser_loss(out, y, learning_rate=0.01)

    # ValueError: Only call `softmax_cross_entropy_with_logits` with named arguments (labels=..., logits=..., ...)
    #loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(out, y))


    loss = tf.reduce_mean(tf.abs(y - out))
    my_opt = tf.train.AdamOptimizer(0.005)
    optimizer = my_opt.minimize(loss)

    reg_lambda = 1.0
    diff_plus_regularization = tf.add(tf.reduce_sum(tf.square(y - out)), tf.multiply(reg_lambda, tf.reduce_sum(tf.square(out))))
    print(x.shape[1], x.shape[0])
    loss = tf.div(diff_plus_regularization, 20000)

    epoch = 300

    accuracy = util.accuracy_fn(out, y)

    testing_training = test_train_indices(len(targets), 500, 1.0)

    all_test = range(sparse_tfidf_texts.shape[0])

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(epoch):
            i_batch = 0
            total_batch_train_acc = 0.0
            total_batch_test_acc = 0.0
            total_batch_train_loss = 0.0
            total_batch_test_loss = 0.0
            log_every = 10
            for (test_indices, train_indices) in testing_training:
                rand_index = train_indices
                rand_x = sparse_tfidf_texts[rand_index].todense()
                rand_y = util.one_hot(rand_index, out.shape[1], targets)
                f_dict = {x: rand_x, y: rand_y}
                sess.run([loss, optimizer], feed_dict=f_dict)
                if i % log_every == 0:
                    train_acc = sess.run(accuracy, feed_dict=f_dict)
                    train_loss = sess.run(loss, feed_dict=f_dict)
                    f_dict_test = {x: sparse_tfidf_texts[test_indices].todense(),
                                            y: util.one_hot(test_indices, out.shape[1], targets)}
                    test_acc = sess.run(accuracy, feed_dict=f_dict_test)
                    test_loss = sess.run(loss, feed_dict=f_dict_test)
                    i_batch = i_batch + 1
                    total_batch_test_acc += test_acc
                    total_batch_train_acc += train_acc
                    total_batch_train_loss += train_loss
                    total_batch_test_loss += test_loss
            if i % log_every == 0:
                print("\nEpoch %d " % i)
                print("Average test accuracy ", (total_batch_test_acc / i_batch))
                print("Average train accuracy", (total_batch_train_acc / i_batch))
                print("Average test loss     ", (total_batch_test_loss / i_batch))
                print("Average train loss    ", (total_batch_train_loss / i_batch))
                acc = sess.run(accuracy, feed_dict={x: sparse_tfidf_texts[all_test].todense(), y: util.one_hot(all_test, out.shape[1], targets)})
                print("batch accuracy        ", acc)



def train_and_test(nn_init_fn):
    n_features = 9000
    (sparse_tfidf_texts, targets) = util.do_tf_idf(n_features)

    print("input shape", sparse_tfidf_texts.shape)  # (18781, 9000)
    print(sparse_tfidf_texts)

    output_size = len(util.subjects)

    (x, out, y) = nn_init_fn(n_features, output_size)
    train_and_test_in_batches(x, out, y, sparse_tfidf_texts, targets)


if __name__ == '__main__':
    train_and_test(util.neural_net) # about 70%, 80% (after about 500 epochs, 256 batch size and regularization of 0.1), 79.8% with regularizer of 0.1
    # removing batch_size and using all the training data (about 500 vs. 128) and regularizer of 1.0 gives 86.5% after 300 epochs, 87.2% after 600 epochs
    # train_and_test(util.neural_net_w_hidden)


