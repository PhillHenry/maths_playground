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
    """
    Prepare your files from the "20 newsgroups" data with:

    for DIR in `ls 20news-18828` ; do { echo $DIR ; head -2 20news-18828/$DIR/* | grep ^Subject | perl -pe s/^Subject:\ //g > $DIR.txt ; } done

    :return: a tuple of all the subject texts and all the correct group IDs
    """
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
    return lines, targets


def neural_net(n_in, n_out):
    x = tf.placeholder(dtype=tf.float32, shape=[None, n_in], name="x")
    y = tf.placeholder(dtype=tf.float32, shape=[None, n_out], name="y")
    weights = tf.Variable(tf.random_normal([n_in, n_out], dtype=tf.float32), name='weights')
    biases = tf.Variable(tf.zeros([n_out]), name='biases')
    out = tf.nn.tanh(tf.matmul(x, weights) + biases)
    print("out shape ", out.shape, "x shape ", x.shape, "weights shape", weights.shape, "bias shape", biases.shape, "hidden_dim", n_out)
    return x, out, y


def optimiser_loss(actual, expected):
    loss_fn = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(expected, actual))))
    learning_rate = 0.0125
    optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(loss_fn)
    return optimizer, loss_fn


def one_hot(indxs, hidden_dim, targets):
    ys = []
    for i in indxs:
        bit = targets[i]
        y = [0.] * hidden_dim
        y[bit] = 1.
        ys.append(y)
    return np.matrix(ys)


def do_tf_idf(n_features):
    (lines, targets) = parse_file()
    tfidf = TfidfVectorizer(tokenizer=tokenizer, stop_words='english', max_features=n_features)
    text = clean_text(lines)
    print("Number of lines", len(lines), "number of targest", len(targets))
    tf_idf_matrix = tfidf.fit_transform(text)
    return tf_idf_matrix, targets


def accuracy_fn():
    # see https://stackoverflow.com/questions/42607930/how-to-compute-accuracy-of-cnn-in-tensorflow
    prediction = tf.argmax(out, 1)
    equality = tf.equal(prediction, tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(equality, tf.float32))
    return accuracy


if __name__ == '__main__':
    n_features = 9000
    (sparse_tfidf_texts, targets) = do_tf_idf(n_features)

    train_indices = np.random.choice(sparse_tfidf_texts.shape[0], round(0.8*sparse_tfidf_texts.shape[0]), replace=False)
    test_indices = np.array(list(set(range(sparse_tfidf_texts.shape[0])) - set(train_indices)))

    output_size = len(subjects)

    (x, out, y) = neural_net(n_features, output_size)

    (optimizer, loss) = optimiser_loss(out, y)

    epoch = 10000
    batch_size = 10

    accuracy = accuracy_fn()

    with tf.Session() as sess:
        print("training...")
        sess.run(tf.global_variables_initializer())
        for i in range(epoch):
            rand_index = np.random.choice(train_indices, size=batch_size)
            rand_x = sparse_tfidf_texts[rand_index].todense()
            rand_y = one_hot(rand_index, output_size, targets)
            f_dict = {x: rand_x, y: rand_y}
            sess.run([loss, optimizer], feed_dict=f_dict)
            if (i+1) % 100 == 0:
                print("accuracy", sess.run(accuracy, feed_dict=f_dict), "loss", sess.run(loss, feed_dict=f_dict))

        print("trained")
        print("Calculating accuracy on test data...")
        overall_accuracy = sess.run(accuracy, feed_dict={x: sparse_tfidf_texts[test_indices].todense(), y: one_hot(test_indices, output_size, targets)})
        print("accuracy", overall_accuracy)

