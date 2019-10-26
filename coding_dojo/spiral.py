"""
    the challenge is to create a matrix of spiraling numbers given n, e.g.
    e.g. n = 3:
        1    2    3    
        8    9    4    
        7    6    5  
    n = 4:
        1   2   3   4
        12  13  14  5
        11  16  15  6
        10  9   8   7
    n = 5:
        1   2   3   4   5    
        16  17  18  19  6    
        15  24  25  20  7    
        14  23  22  21  8    
        13  12  11  10  9
"""
import numpy as np


def perimeter(array, base=0):
    i = j = 0
    n = len(array)
    if n == 1:
        array[0, 0] = base + 1
        return array

    in_upside = lambda i, j:  j == 0
    in_rightside = lambda i, j: i == (n - 1)
    in_downside = lambda i, j: j == (n - 1)
    in_leftside = lambda i, j: i == 0

    at_topright_corner = lambda i, j: i == (n - 1) and j == 0
    at_downright_corner = lambda i, j: i == (n - 1) and j == (n - 1)
    at_downleft_corner = lambda i, j: i == 0 and j == (n - 1)

    for k in range(4*(n-1)):
        array[j, i] = k + base + 1
        if in_upside(i, j) and not at_topright_corner(i, j):
            i += 1
        elif in_rightside(i, j) and not at_downright_corner(i, j):
            j += 1
        elif in_downside(i, j) and not at_downleft_corner(i, j):
            i -= 1
        elif in_leftside(i, j):
            j -= 1
        if (j - 1) == 0 and i == 0:
            array[j][i] = 4*(n-1) + base
            array[1:-1, 1:-1] = perimeter(array[1:-1, 1:-1], base=4*(n-1) + base)
            return array


def to_int_string(array):
    return ''.join([str(s) for s in array.astype(int).tolist()])


def create_spiral(n):
    if type(n) is int and n >= 1:
        array = np.zeros((n,n))
        array = perimeter(array)
        return to_int_string(array)
    return ''


if __name__ == "__main__":
    create_spiral(5)
