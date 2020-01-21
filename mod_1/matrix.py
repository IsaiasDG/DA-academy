from typing import List, Callable

MATRIX_ROW = List[float]
MATRIX = List[MATRIX_ROW]


def valid_equal_size(func: Callable):

    def inner(addend1, addend2):
        if addend1.get_size() != addend2.get_size():
            msg = "Operands could not be broadcast together with shape %s %s."
            raise ValueError(msg % (addend1.get_size(), addend2.get_size()))

        return func(addend1, addend2)

    return inner


def valid_mult_shape(func: Callable):

    def inner(multiplicand, multiplier):
        _, multiplicand_column = multiplicand.get_size()
        multiplier_row, _ = multiplier.get_size()
        if multiplicand_column != multiplier_row:
            msg = "Operands could not be broadcast together with shape %s %s."
            raise ValueError(msg %
                             (multiplicand.get_size(), multiplier.get_size()))

        return func(multiplicand, multiplier)

    return inner


def valid_square(func: Callable):

    def inner(matrix):
        rows, columns = matrix.get_size()
        if rows != columns:
            msg = "Operands could not be broadcast together with shape %s."
            raise ValueError(msg % (str(matrix.get_size())))
        return func(matrix)

    return inner


def valid_sub_matrix(func: Callable):

    def inner(matrix, row, col, from_col):
        rows, columns = matrix.get_size()
        if row > rows or (col + from_col) > columns:
            msg = "Operands could not be broadcast together with shape %s."
            raise ValueError(msg % (str(matrix.get_size())))
        return func(matrix, row, col, from_col)

    return inner


class Matrix(list):

    def __init__(self, init_list: MATRIX):
        if init_list:
            prev_size_column = len(init_list[0])
            self.__set_size(len(init_list), prev_size_column)
            for row in init_list[1:]:
                if len(row) != prev_size_column:
                    raise ValueError("Row not the same size.")
                prev_size_column = len(row)

        super(Matrix, self).__init__(init_list)

    def get_rows(self, rows: List[int]) -> 'Matrix':
        return Matrix(
            [self[i_row] for i_row in set(rows) if i_row <= self.__rows])

    def get_colums(self, cols: List[int]) -> 'Matrix':
        m = []
        searched_col = set(cols)
        for i_row in range(self.__rows):
            m.append([self[i_row][i_col]
                      for i_col in searched_col if i_col <= self.__columns])
        return Matrix(m)

    @valid_sub_matrix
    def get_sub_matrix(self, row: int, col: int, from_col: int) -> 'Matrix':
        matrix = Matrix([[0] * col for _ in range(row)])
        for i_row in range(row):
            for i_col in range(col):
                matrix[i_row][i_col] = self[i_row][from_col + i_col]
        return Matrix(matrix)

    @valid_equal_size
    def add(self, addend: 'Matrix') -> 'Matrix':
        sum = Matrix([[0] * self.__columns for _ in range(self.__rows)])
        sum.__set_size(self.__rows, self.__columns)
        for i_row in range(self.__rows):
            for i_col in range(self.__columns):
                sum[i_row][i_col] = self[i_row][i_col] + addend[i_row][i_col]
        return sum

    @valid_mult_shape
    def mult(self, multiplier: 'Matrix') -> 'Matrix':
        mult = Matrix([[0] * multiplier.__columns for _ in range(self.__rows)])
        mult.__set_size(self.__rows, multiplier.__columns)
        for i_row in range(self.__rows):
            for i_col in range(multiplier.__columns):
                for n in range(self.__columns):
                    mult[i_row][i_col] += self[i_row][n] * multiplier[n][i_col]
        return mult

    def mult_scalar(self, scalar: float) -> 'Matrix':
        mult = Matrix([[0] * self.__columns for _ in range(self.__rows)])
        mult.__set_size(self.__rows, self.__columns)
        for i_row in range(self.__rows):
            for i_col in range(self.__columns):
                mult[i_row][i_col] = self[i_row][i_col] * scalar

        return mult

    def transpose(self) -> 'Matrix':
        tran = Matrix([[0] * self.__rows for _ in range(self.__columns)])
        tran.__set_size(self.__columns, self.__rows)
        for i_row in range(self.__rows):
            for i_col in range(self.__columns):
                tran[i_col][i_row] = self[i_row][i_col]
        return tran

    @valid_square
    def det(self) -> float:
        det = 0
        if self.__rows == 2 and self.__columns == 2:
            return self[0][0] * self[1][1] - self[1][0] * self[0][1]
        for i_col in range(self.__columns):
            if self[0][i_col] != 0:
                sub_matrix = self.__get_SubMatrix(0, i_col)
                det += pow(-1, i_col) * self[0][i_col] * sub_matrix.det()
        return det

    @valid_square
    def cof(self) -> 'Matrix':
        cof = Matrix([[0] * self.__columns for _ in range(self.__rows)])
        for i_row in range(self.__rows):
            for i_col in range(self.__columns):
                sub_matrix = self.__get_SubMatrix(i_row, i_col)
                cof[i_row][i_col] = pow(-1, i_row + i_col) * sub_matrix.det()
        return cof

    @valid_square
    def adj(self) -> 'Matrix':
        return self.cof().transpose()

    @valid_square
    def inv(self) -> 'Matrix':
        return self.adj().mult_scalar(1/self.det())

    def __get_SubMatrix(self, row: float, col: float) -> 'Matrix':
        return Matrix([self.__get_SubItem(r, col)
                      for r in range(self.__rows) if r != row])

    def __get_SubItem(self, row: float, col: float) -> float:
        return self[row][:col] + self[row][col + 1:]

    def get_size(self):
        return (self.__rows, self.__columns)

    def __set_size(self, rows: int, columns: int):
        self.__rows = rows
        self.__columns = columns

    def __str__(self):
        m = []
        for i_row in range(self.__rows):
            m.append("[%s]" % ", ".join(
                ["%.2f" % round(item, 2) for item in self[i_row]]))
        return "[%s]" % ("\n".join(m))
