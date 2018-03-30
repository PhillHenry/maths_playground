import numpy as np
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
    print("hello world")