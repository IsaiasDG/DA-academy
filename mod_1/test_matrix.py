from matrix import Matrix

from unittest import TestCase


class MatrixTest(TestCase):

    def setUp(self):
        pass

    def test_build(self):
        m = Matrix([[1, 2, 3, 4], [5, 6, 7, 8]])
        self.assertEqual(m[0][1], 2)

    def test_build_fail_elements_incogruents(self):
        with self.assertRaises(ValueError) as e:
            Matrix([[1, 2, 3, 4, 5], [6, 7]])

        self.assertEqual("Row not the same size.", e.exception.args[0])

    def test_add(self):
        addend1 = Matrix([[1, 2, 3], [3, 4, 5], [5, 6, 7]])
        addend2 = Matrix([[3, 5, 6], [8, 6, 9], [1, 2, 4]])
        sum = Matrix([[4, 7, 9], [11, 10, 14], [6, 8, 11]])
        self.assertEqual(sum, addend1.add(addend2))

    def test_add_addend_gte_other(self):
        addend1 = Matrix([[1, 2, 3], [5, 6, 7]])
        addend2 = Matrix([[3, 5, 6], [8, 6, 9], [1, 2, 4]])
        with self.assertRaises(ValueError) as e:
            addend1.add(addend2)

        msg = "Operands could not be broadcast together with shape " \
            "(2, 3) (3, 3)."
        self.assertEqual(msg, e.exception.args[0])

    def test_multiplication(self):
        multiplicand = Matrix([[1, 2, 3], [4, 5, 6]])
        multiplier = Matrix([[7, 8], [9, 10], [11, 12]])
        mul = Matrix([[58,  64], [139, 154]])
        self.assertEqual(mul, multiplicand.mult(multiplier))

    def test_multiplication_fail_shape(self):
        multiplicand = Matrix([[1, 2, 3], [4, 5, 6], [4, 5, 6], [4, 5, 6]])
        multiplier = Matrix([[7, 8], [9, 10]])

        with self.assertRaises(ValueError) as e:
            multiplicand.mult(multiplier)

        msg = "Operands could not be broadcast together with shape " \
            "(4, 3) (2, 2)."
        self.assertEqual(msg, e.exception.args[0])

    def test_transpose(self):
        matrix = Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
        matrix_expected = Matrix([[1, 4, 7, 10],
                                  [2, 5, 8, 11],
                                  [3, 6, 9, 12]])
        self.assertEqual(matrix_expected, matrix.transpose())

    def test_multiplication_scalar(self):
        multiplicand = Matrix([[1, 2, 3], [4, 5, 6]])
        mul = Matrix([[2, 4, 6], [8, 10, 12]])
        self.assertEqual(mul, multiplicand.mult_scalar(2))

    def test_determinat_2x2(self):
        matrix = Matrix([[1, 2], [3, 4]])
        self.assertEqual(-2, matrix.det())

    def test_determinat_nxn(self):
        matrix = Matrix([[2, 6, 7], [4, 5, 6], [7, 8, 9]])
        self.assertEqual(9, matrix.det())

        matrix = Matrix([[2, 0, 7, 8],
                         [4, 5, 0, 0],
                         [0, 8, 9, 5],
                         [1, 8, 2, 5]])
        self.assertEqual(1957, matrix.det())

    def test_determinat_nxn_fails(self):
        matrix = Matrix([[2, 6, 7], [4, 5, 6]])
        with self.assertRaises(ValueError) as e:
            matrix.det()

        msg = "Operands could not be broadcast together with shape (2, 3)."
        self.assertEqual(msg, e.exception.args[0])

    def test_adjugate_nxn(self):
        matrix = Matrix([[0, 9, 3], [2, 0, 4], [3, 7, 0]])
        matrix_expected = Matrix([[-28, 21, 36], [12, -9, 6], [14, 27, -18]])
        self.assertEqual(matrix_expected, matrix.adj())

    # TODO: Carry errors
    def test_inv_nxn(self):
        matrix = Matrix([[0, 9, 3], [2, 0, 4], [3, 7, 0]])
        matrix_inv = matrix.inv()
        matrix_identity = Matrix([[1, 0, 0], [-0.00, 1, -0.00], [0, 0, 1]])
        self.assertEqual(str(matrix_identity), str(matrix_inv.mult(matrix)))

        matrix = Matrix([[0, 9, 3, 2],
                         [2, 0, 4, 0],
                         [3, 7, 0, 9],
                         [0, 1, 0, 2]])
        matrix_inv = matrix.inv()
        matrix_identity = Matrix([[1, 0, 0, 0],
                                  [0, 1, 0, 0],
                                  [0, 0, 1, -0.00],
                                  [0, 0, 0, 1]])
        self.assertEqual(str(matrix_identity), str(matrix_inv.mult(matrix)))

    def test_get_rows(self):
        matrix = Matrix([[0, 9, 3, 2],
                         [2, 0, 4, 0],
                         [3, 7, 0, 9],
                         [0, 1, 0, 2],
                         [2, 3, 6, 4],
                         [2, 3, 5, 6]])
        matrix_expected = Matrix([[2, 0, 4, 0],
                                  [0, 1, 0, 2],
                                  [2, 3, 6, 4]])
        self.assertEqual(matrix_expected,
                         matrix.get_rows([1, 3, 4, 10, 1, 1]))

    def test_get_columns(self):
        matrix = Matrix([[0, 1, 9, 3, 2, 0, 2],
                         [2, 4, 0, 4, 0, 3, 6],
                         [3, 6, 7, 0, 9, 2, 9],
                         [0, 4, 1, 0, 2, 1, 1],
                         [2, 0, 3, 6, 4, 3, 0],
                         [2, 0, 3, 5, 6, 4, 2]])
        matrix_expected = Matrix([[0, 9, 3, 0],
                                  [2, 0, 4, 3],
                                  [3, 7, 0, 2],
                                  [0, 1, 0, 1],
                                  [2, 3, 6, 3],
                                  [2, 3, 5, 4]])
        self.assertEqual(str(matrix_expected),
                         str(matrix.get_colums([0, 2, 3, 5])))

        matrix_expected = Matrix([[1, 2],
                                  [4, 6],
                                  [6, 9],
                                  [4, 1],
                                  [0, 0],
                                  [0, 2]])
        self.assertEqual(str(matrix_expected),
                         str(matrix.get_colums([6, 1, 1, 1, 23])))

    def test_get_sub_matrix(self):
        matrix = Matrix([[0, 1, 9, 3, 2, 0, 2],
                         [2, 4, 0, 4, 0, 3, 6],
                         [3, 6, 7, 0, 9, 2, 9],
                         [0, 4, 1, 0, 2, 1, 1],
                         [2, 0, 3, 6, 4, 3, 0],
                         [2, 0, 3, 5, 6, 4, 2]])
        matrix_expected = Matrix([[9, 3, 2],
                                  [0, 4, 0],
                                  [7, 0, 9]])
        self.assertEqual(str(matrix_expected),
                         str(matrix.get_sub_matrix(3, 3, 2)))

    def test_get_sub_matrix_fail(self):
        matrix = Matrix([[0, 1, 9, 3, 2, 0, 2],
                         [2, 4, 0, 4, 0, 3, 6],
                         [3, 6, 7, 0, 9, 2, 9],
                         [0, 4, 1, 0, 2, 1, 1],
                         [2, 0, 3, 6, 4, 3, 0],
                         [2, 0, 3, 5, 6, 4, 2]])

        with self.assertRaises(ValueError) as e:
            matrix.get_sub_matrix(3, 5, 3)

        msg = "Operands could not be broadcast together with shape (6, 7)."
        self.assertEqual(msg, e.exception.args[0])
