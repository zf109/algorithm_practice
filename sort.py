# from heapq import merge

from random import randint
from time import time
from functools import reduce
import operator as op

def merge(sorted1, sorted2):
    if not sorted1:
        return sorted2
    if not sorted2:
        return sorted1

    merged = []
    i, j = 0, 0
    while i < len(sorted1) and j < len(sorted2):
        if sorted1[i] < sorted2[j]:
            merged.append(sorted1[i])
            i += 1
        else:
            merged.append(sorted2[j])
            j += 1
    rest = sorted1[i:] if j == len(sorted2) else sorted2[j:]
    return merged + rest


def merge_sort(seq):
    if len(seq) <= 1:
        return seq

    middle = len(seq) // 2
    left = seq[:middle]
    right = seq[middle:]

    left = merge_sort(left)
    right = merge_sort(right)
    return list(merge(left, right))

def bubble_sort(seq):
    changed = True
    while changed:
        changed = False
        for i in range(len(seq) - 1):
            if seq[i] > seq[i+1]:
                seq[i], seq[i+1] = seq[i+1], seq[i]
                changed = True
    return None

def insertion_sort(l):
    for i in range(1, len(l)):
        j = i-1
        key = l[i]
        while (l[j] > key) and (j >= 0):
           l[j+1] = l[j]
           j -= 1
        l[j+1] = key


def _test_time(sort_func):
    test_list = [randint(0,100) for x in range(10000)]
    start = time()
    res = sort_func(test_list)
    print(sort_func.__name__, "takes time: ", time()-start)
    assert res == sorted(test_list)


def ncr(n, r):
    r = min(r, n-r)
    if r == 0: return 1
    numer = reduce(op.mul, range(n, n-r, -1))
    denom = reduce(op.mul, range(1, r+1))
    return numer//denom

if __name__ == "__main__":
    # _test_time(bubble_sort)
    _test_time(merge_sort)
    # _test_time(insertion_sort)
