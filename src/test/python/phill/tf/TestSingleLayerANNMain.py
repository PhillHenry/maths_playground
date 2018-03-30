import unittest

import src.main.python.phill.tf.SingleLayerANNMain as to_test


class MyTestCase(unittest.TestCase):

    def test_batching_indices(self):
        n = 16000
        batch_size = 1000
        test_to_train_ratio = 1.0
        batches = to_test.test_train_indices(n, batch_size, test_to_train_ratio)
        self.assertTrue(len(batches), batch_size)
        for batch in batches:
            batch_size = n / batch_size
            self.assertTrue(len(batch), batch_size)
            (test, train) = batch
            self.assertTrue(len(test), (batch_size * (test_to_train_ratio / (test_to_train_ratio + 1))))
            self.assertTrue(len(train), (batch_size / (test_to_train_ratio + 1)))


if __name__ == '__main__':
    unittest.main()