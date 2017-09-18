import unittest
from unittest import TestCase

import numpy as np

from .. import imaging

__author__ = 'palpatine'


class TestCleanDiode(TestCase):
    test_array = np.array([0, 5, 6, 7, 8, 10, 11, 12, 15, 19, 20, 22, 25, 30, 35, 40, 45, 50, 55,
                           60, 65, 70, 75])
    test_array *= 100
    cleaned_array = np.array(
        [0, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000, 6500, 7000,
         7500])

    def test_clean_diode(self):
        result = imaging.clean_diode(self.test_array, self.test_array)
        np.testing.assert_array_equal(result, self.cleaned_array)


if __name__ == '__main__':
    unittest.main()
