import unittest

import src.main.python.phill.lang.Star as x


class TestStar(unittest.TestCase):
    def test_fn_with_star_arg_no_star(self):
        self.assertEqual(x.with_star((1, 2)), [(1, 2)])

    def test_fn_with_star_arg_with_star(self):
        self.assertEqual(x.with_star(*(1, 2)), [1, 2])

    def test_fn_without_star_arg_no_star(self):
        self.assertEqual(x.without_star((1, 2)), [1, 2])

    def test_fn_without_star_arg_with_star(self):
        with self.assertRaises(TypeError) as context:
            x.without_star(*(1, 2))
