import tensorflow as tf
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
import string
import numpy as np

subjects = ["alt.atheism",
            "comp.graphics",
            "comp.os.ms-windows.misc",
            "comp.sys.ibm.pc.hardware",
            "comp.sys.mac.hardware",
            "comp.windows.x",
            "misc.forsale",
            "rec.autos",
            "rec.motorcycles",
            "rec.sport.baseball",
            "rec.sport.hockey",
            "sci.crypt",
            "sci.electronics",
            "sci.med",
            "sci.space",
            "soc.religion.christian",
            "talk.politics.guns",
            "talk.politics.mideast",
            "talk.politics.misc",
            "talk.religion.misc"]


def zip_with_index(xs):
    x2i = {}
    for (i, x)in enumerate(xs):
        x2i[x] = i
    return x2i


def read_lines(file):
    lines = []
    for line in open(file, 'r'):
        lines.append(line)
    return lines


def tokenizer(text):
    words = nltk.word_tokenize(text)
    return words


# stolen from https://github.com/nfmcclure/tensorflow_cookbook/blob/master/07_Natural_Language_Processing/03_Implementing_tf_idf/03_implementing_tf_idf.py
def clean_text(texts):
    texts = [x.lower() for x in texts]
    texts = [''.join(c for c in x if c not in string.punctuation) for x in texts]
    texts = [''.join(c for c in x if c not in '0123456789') for x in texts]
    texts = [' '.join(x.split()) for x in texts]
    return texts


def parse_file():
    sub_2_indx = zip_with_index(subjects)
    sub_2_lines = {}
    lines = []
    targets = []
    for (x, i) in sub_2_indx.items():
        print(i, x)
        subject_lines = read_lines("/home/henryp/Code/Temp/" + x + ".txt")
        lines += subject_lines
        sub_2_lines[x] = lines
        targets += [i] * len(subject_lines)
    print(len(lines))
    return (lines, targets)


if __name__ == '__main__':
    (lines, targets) = parse_file()
    n_features = 9000
    tfidf = TfidfVectorizer(tokenizer=tokenizer, stop_words='english', max_features=n_features)
    text = clean_text(lines)
    print("Number of lines", len(lines), "number of targest", len(targets))
    # print(text)
    sparse_tfidf_texts = tfidf.fit_transform(text)

    train_indices = np.random.choice(sparse_tfidf_texts.shape[0], round(0.8*sparse_tfidf_texts.shape[0]), replace=False)
    test_indices = np.array(list(set(range(sparse_tfidf_texts.shape[0])) - set(train_indices)))
    texts_train = sparse_tfidf_texts[train_indices]
    texts_test = sparse_tfidf_texts[test_indices]

#    print("sparse_tfidf_texts shape = " + np.shape(sparse_tfidf_texts))

    hidden_dim = len(subjects)
    x = tf.placeholder(dtype=tf.float32, shape=[None, n_features], name="x")
    y = tf.placeholder(dtype=tf.float32, shape=[None, hidden_dim], name="y")
    weights = tf.Variable(tf.random_normal([n_features, hidden_dim], dtype=tf.float32), name='weights')
    biases = tf.Variable(tf.zeros([hidden_dim]), name='biases')
    output = tf.nn.tanh(tf.matmul(x, weights) + biases)

    print("output shape ", output.shape, "x shape ", x.shape, "weights shape", weights.shape, "bias shape", biases.shape, "hidden_dim", hidden_dim)

    epoch = 10000

    loss = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(y, output))))

    learning_rate = 0.0125

    train_op = tf.train.RMSPropOptimizer(learning_rate).minimize(loss)

    batch_size = 10

    predictions_correct = tf.subtract(tf.constant(1.), tf.squared_difference(y, output))  #tf.cast(tf.equal(output, y), tf.float32)
    accuracy = tf.reduce_mean(predictions_correct)

    def one_hot(indxs):
        ys = []
        for i in indxs:
            bit = targets[i]
            # y = np.zeros(shape=[1, hidden_dim])
            # y[0, ] = 1.
            y = [0.] * hidden_dim
            y[bit] = 1.
            ys.append(y)
        return np.matrix(ys)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(epoch):
            rand_index = np.random.choice(texts_train.shape[0], size=batch_size)
            rand_x = texts_train[rand_index].todense()
            rand_y = one_hot(rand_index)
            # print("rand_x shape", rand_x.shape, "rand_y shape", rand_y.shape)
            f_dict = {x: rand_x, y: rand_y}
            sess.run([loss, train_op], feed_dict=f_dict)
            if (i+1)%100==0:
                train_acc_temp = sess.run(accuracy, feed_dict=f_dict)
                train_loss_temp = sess.run(loss, feed_dict=f_dict)
                print("accuracy", train_acc_temp, "loss", train_loss_temp)

        print("trained")

        # TODO check the training with test data


