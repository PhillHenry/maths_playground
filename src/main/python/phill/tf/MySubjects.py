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


if __name__ == '__main__':
    sub_2_indx = zip_with_index(subjects)
    sub_2_lines = {}
    lines = []
    targets = []
    for (x, i) in sub_2_indx.items():
        print(i, x)
        subjects = read_lines("/home/henryp/Code/Temp/" + x + ".txt")
        lines += subjects
        sub_2_lines[x] = lines
        targets += [i] * len(subjects)
    print(len(lines))
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
    target_train_ids = np.array([x for ix, x in enumerate(targets) if ix in train_indices])
    target_test_ids = np.array([x for ix, x in enumerate(targets) if ix in test_indices])

    print("target_test shape", target_test_ids.shape)
    print("texts_test shape", texts_test.shape)

#    print("sparse_tfidf_texts shape = " + np.shape(sparse_tfidf_texts))

    hidden_dim = n_features
    x = tf.placeholder(dtype=tf.float32, shape=[None, n_features])
    weights = tf.Variable(tf.random_normal([n_features, hidden_dim], dtype=tf.float32), name='weights')
    biases = tf.Variable(tf.zeros([hidden_dim]), name='biases')
    output = tf.nn.tanh(tf.matmul(x, weights) + biases)

    print("output shape ", output.shape)
    print("x shape ", x.shape)

    epoch = 10

    loss = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(x, output))))

    learning_rate = 0.0125

    train_op = tf.train.RMSPropOptimizer(learning_rate).minimize(loss)

    batch_size = 10

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(epoch):
            # for j in target_test_ids:
            #     datum = texts_test[j]
            #     print("datum", datum, "type", type(datum))
            #     l, _ = sess.run([loss, train_op], feed_dict={x: datum})
            rand_index = np.random.choice(texts_train.shape[0], size=batch_size)
            rand_x = texts_train[rand_index].todense()
            f_dict = {x: rand_x}
            sess.run([loss, train_op], feed_dict=f_dict)

    print("trained")


