from random import gauss
from numpy import argmin
from itertools import product
from copy import deepcopy


def dist(vec1: tuple, vec2: list, p: int=2) -> float:
    """
        calculate the distance between vec1 and vec2, it uses L_p norm to do the calculation,
        by default its euclidean distance (L_2 distance), but can be set to manhattan distance (L_1 distance)
        or other L_p measure
    """
    return sum([abs(x - y)**p for x, y in zip(vec1, vec2)])**(1/p)

def find_closest(vec: tuple, veclist: list) -> tuple:
    """
        given a vector vec and a list of vectors, it returns the closest vector to vec in the second list.
    """
    min_ind = argmin([dist(vec, vec_) for vec_ in veclist])
    return veclist[min_ind]

def match(veclist1: list, veclist2:list) -> [dict]:
    """
        The first algorithm come to mind, this is a very readable but not efficient algoirthm (N^2 time complexity and N space complexity)
        .. Parameters:
            :veclist1: A list of tuples represents list of approximate locations
            :veclist2: A list of tuples represetns list of actual location
        ..Return:
            list of dictionaries with key vec for the vector in veclist1 and closest be the cloest location in veclist2
    """ 
    return [{"vec": vec, "closest": find_closest(vec, veclist2)} for vec in veclist1]

def in_place_match(veclist1: list, veclist2: list) -> [tuple]:
    """
        This is a in place matching algorithm, meaning it will change the order of veclist 2.
        This is potentially more suitable for big data situation.
        It is slightly more efficient (N^2/2 time complexity and 1 space complexity)
        .. Parameters:
            :veclist1: A list of tuples represents list of approximate locations
            :veclist2: A list of tuples represetns list of actual location
        ..Return:
            sorted veclist2, it will sort veclist2 in memory to make sure the closest location
            matches to the location in veclist1
    """ 
    i = 0
    for estlocation in veclist1[:-1]:
        min_ind = argmin([dist(estlocation, loc) for loc in veclist2[i:]])
        veclist2[i], veclist2[min_ind + i] = veclist2[min_ind + i], veclist2[i]
        i += 1
    return veclist2

def random_lists(n: int=100, mean: int=0, var: int=100, dim: int=2) -> (list, list):
    """
        Random list of tuple generator
    """
    veclist1 = [tuple(gauss(mean, var) for _ in range(dim)) for _ in range(n)]
    veclist2 = [tuple(gauss(mean, var) for _ in range(dim)) for _ in range(n)]
    return veclist1, veclist2


if __name__ == "__main__":
    """Setup run config"""
    from time import time
    list_length = 3000
    tuple_dim = 2
    time_message = "time taken by using match with list of length {} (dimention {}) is {:.2f}s"

    """Initialise lists """
    veclist1, veclist2 = random_lists(list_length, dim=tuple_dim)
    veclist2_copy = deepcopy(veclist2)

    """Check in match() time"""
    start = time()
    match(veclist1, veclist2)
    match_time = time() - start
    print(time_message.format(list_length, tuple_dim, match_time))

    """Check in in_place_match() time"""
    start = time()
    in_place_match(veclist1, veclist2_copy)
    in_place_match_time = time() - start
    print(time_message.format(list_length, tuple_dim, in_place_match_time))



