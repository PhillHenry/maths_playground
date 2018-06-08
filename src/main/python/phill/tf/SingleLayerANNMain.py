import numpy as np
import tensorflow as tf
import MySubjects as util
import src.main.python.phill.text.TermCategoryVectorizer as term_category
import matplotlib.pyplot as plt

log_every = 2


def test_train_indices(n, batch_size, test_to_train_ratio):
    indices = np.random.choice(n, size=[n], replace=False)
    xs = []
    for i in range(int(n / batch_size)):
        start_incl = i * batch_size
        end_excl = (i+1) * batch_size
        # print("(%d, %d] in %s" % (start_incl, end_excl, np.shape(indices)))
        batch = indices[start_incl:end_excl]
        end_test_incl = int(batch_size * test_to_train_ratio / (test_to_train_ratio + 1))
        test = batch[0:end_test_incl]
        train = batch[end_test_incl:-1]
        xs.append((test, train))
    return xs


def train_and_test_in_batches(x, out, y, sparse_tfidf_texts, targets, epoch):
    (optimizer, loss) = util.optimiser_loss(out, y)

    accuracy = util.accuracy_fn(out, y)

    testing_training = test_train_indices(len(targets), 500, 1.0)

    all_test = range(sparse_tfidf_texts.shape[0])

    test_accs = []
    train_accs = []
    total_accs = []

    p_dropout = 0.9  # 1.0->88.8% accuracy. 0.8->68%, 0.95->81%, 0.85->71.9%, 0.9->76.5%, 0.97->83.3%, 0.99->84.7%, 1.0->85.3%

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(epoch):
            i_batch = 0
            total_batch_train_acc = 0.0
            total_batch_test_acc = 0.0
            total_batch_train_loss = 0.0
            total_batch_test_loss = 0.0
            for (test_indices, train_indices) in testing_training:
                rand_index = train_indices
                rand_x = sparse_tfidf_texts[rand_index] #.todense()
                rand_y = util.one_hot(rand_index, out.shape[1], targets)
                f_dict = {x: rand_x, y: rand_y}
                train_loss, _, train_acc = sess.run([loss, optimizer, accuracy], feed_dict=f_dict)
                if i % log_every == log_every - 1:
                    f_dict_test = {x: sparse_tfidf_texts[test_indices], #.todense(),
                                   y: util.one_hot(test_indices, out.shape[1], targets)}
                    test_acc = sess.run(accuracy, feed_dict=f_dict_test)
                    test_loss = sess.run(loss, feed_dict=f_dict_test)
                    i_batch = i_batch + 1
                    total_batch_test_acc += test_acc
                    total_batch_train_acc += train_acc
                    total_batch_train_loss += train_loss
                    total_batch_test_loss += test_loss
            if i % log_every == log_every - 1:
                print("\nEpoch %d " % i)
                test_acc = (total_batch_test_acc / i_batch)
                train_acc = (total_batch_train_acc / i_batch)
                print("Average test accuracy ", test_acc)
                print("Average train accuracy", train_acc)
                print("Average test loss     ", (total_batch_test_loss / i_batch))
                print("Average train loss    ", (total_batch_train_loss / i_batch))
                acc = sess.run(accuracy, feed_dict={x: sparse_tfidf_texts[all_test], #.todense(),
                                                    y: util.one_hot(all_test, out.shape[1], targets)})
                print("batch accuracy        ", acc)
                test_accs.append(test_acc)
                train_accs.append(train_acc)
                total_accs.append(acc)
    return test_accs, train_accs, total_accs


def train_and_test(nn_init_fn, epoch):
    n_features = len(util.subjects)
    # (sparse_tfidf_texts, targets) = util.do_tf_idf(n_features)
    (sparse_tfidf_texts, targets) = util.do_term_cat()

    output_size = len(util.subjects)

    (x, out, y) = nn_init_fn(n_features, output_size)
    return train_and_test_in_batches(x, out, y, sparse_tfidf_texts, targets, epoch) #, dropout_keep_prob)


def plot_training_vs_testing():
    epoch = log_every * 500
    (test, train, total) = train_and_test(util.neural_net, epoch)  # about 70%, 80% (after about 500 epochs, 256 batch size and regularization of 0.1), 79.8% with regularizer of 0.1
    # removing batch_size and using all the training data (about 500 vs. 128) and regularizer of 1.0 gives 86.5% after 300 epochs, 87.2% after 600 epochs
    # train_and_test(util.neural_net_w_hidden)
    xs = [i * log_every for i in range(len(total))]
    print("xs", xs)
    print("total", total)
    plt.plot(xs, total)
    plt.plot(xs, test, 'o', c='b')
    plt.plot(xs, train, 'o', c='r')
    plt.ylabel('accuracy')
    plt.xlabel('epochs')
    plt.show()


def plot_accuracy_vs_data():
    epoch = log_every
    n_features = 20
    # (sparse_tfidf_texts, targets) = util.do_tf_idf(n_features)
    (sparse_tfidf_texts, targets) = util.do_term_cat()

    output_size = len(util.subjects)

    (x, out, y, dropout_keep_prob) = util.neural_net(n_features, output_size)

    n = 16
    total = []
    test = []
    train = []
    for i in range(n - 1):
        i += 1
        print("i", i)
        pc_of_data = float(i) / float(n)
        indices = np.random.choice(sparse_tfidf_texts.shape[0], round(pc_of_data * sparse_tfidf_texts.shape[0]), replace=False)
        print(indices)
        doc_vec_sample = sparse_tfidf_texts[indices]
        target_sample = np.array(targets)[indices]
        (test_accs, train_accs, total_accs) = train_and_test_in_batches(x, out, y, doc_vec_sample, target_sample, epoch, dropout_keep_prob)
        print(test_accs, train_accs, total_accs)
        total.extend(total_accs)
        test.extend(test_accs)
        train.extend(train_accs)

    xs = [float(i) / n for i in range(len(total))]
    print("xs", xs)
    print("total", total)
    plt.plot(xs, total)
    plt.plot(xs, test, 'o', c='b')
    plt.plot(xs, train, 'o', c='r')
    plt.ylabel('accuracy')
    plt.xlabel('fraction of all data')
    plt.show()


if __name__ == '__main__':
    plot_training_vs_testing()


