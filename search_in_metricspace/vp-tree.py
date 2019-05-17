
from random import shuffle, sample
from random import gauss
from numpy import argmin
from copy import deepcopy, copy
from statistics import median, variance
import matplotlib.pyplot as plt

def random_lists(n: int=100, mean: int=0, var: int=100, dim: int=2) -> (list, list):
    """
        Random list of tuple generator
    """
    veclist1 = [tuple(gauss(mean, var) for _ in range(dim)) for _ in range(n)]
    veclist2 = [tuple(gauss(mean, var) for _ in range(dim)) for _ in range(n)]
    return veclist1, veclist2

def dist(vec1: tuple, vec2: list, p: int=2) -> float:
    """
        calculate the distance between vec1 and vec2, it uses L_p norm to do the calculation,
        by default its euclidean distance (L_2 distance), but can be set to manhattan distance (L_1 distance)
        or other L_p measure
    """
    return sum([abs(x - y)**p for x, y in zip(vec1, vec2)])**(1/p)


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


class VPtreeNode():
    def __init__(self, p=None, mu=None, left=None, right=None):
        self.id = id(self)
        self.p = p
        self.mu = mu
        self.left = left or None
        self.right = right or None
        self.list_ = []

    def __repr__(self):
        # return "{}, {}, {}".format(self.left or "", self.p, self.right or "")
        return str(self.p)
    
    def tolist(self):
        print("self.id:{}".format(self.id))
        if self.left:
            self.list_ += self.left.tolist()
        if self.right:
            self.list_ += self.right.tolist()
        self.list_ += [self]
        return self.list_


def select_vp(list_=None, rate=.1):
    if len(list_) == 1:
        return list_[0]
    best_spread = 0
    length = len(list_)
    n_sample = int(rate*length) if length > 20 else length
    sample1 = sample(list_, n_sample)

    try:
        for p in sample1:
            dist2p = [dist(p, x) for x in sample(list_, n_sample)]
            mu = median(dist2p)
            spread = variance([d - mu for d in dist2p])
            if spread > best_spread:
                best_spread, best_p = spread, p
    except Exception as e:
        print("list_: {}".format(list_))
        raise e
    return best_p


def make_vp(list_=None):
    if not list_:
        return
    vp = VPtreeNode()
    vp.p = select_vp(list_)
    vp.mu = median([dist(vp.p, s) for s in list_])
    left_list = [s for s in list_ if dist(vp.p, s) < vp.mu]
    right_list = [s for s in list_ if dist(vp.p, s) >= vp.mu and s != vp.p]
    vp.left = make_vp(left_list)
    vp.right = make_vp(right_list)
    return vp


def search_vp(q, vp: VPtreeNode, tau=float("inf"), best=None):
    if not vp:
        return tau, best
    x = dist(q, vp.p)
    if x < tau:
        tau, best = x, vp.p
    if x <= vp.mu + tau:
        tau_, best_ = search_vp(q, vp.left, tau=tau, best=best)
        if tau_ < tau:
            tau, best = tau_, best_
    if x >= vp.mu - tau:
        tau_, best_ = search_vp(q, vp.right, tau=tau, best=best)
        if tau_ < tau:
            tau, best = tau_, best_
    return tau, best


if __name__ == "__main__":
    from time import time
    list_length = 5
    tuple_dim = 2
    time_message = "time taken by using match with list of length {} (dimention {}) is {:.2f}s"

    """Initialise lists """
    veclist1, veclist2 = random_lists(list_length, dim=tuple_dim)
    veclist2_copy = deepcopy(veclist2)

    """Check in in_place_match() time"""
    start = time()
    in_place_match(veclist1, veclist2_copy)
    in_place_match_time = time() - start
    print(time_message.format(list_length, tuple_dim, in_place_match_time))


    """Check in search_vp() time"""
    start = time()
    vptree = make_vp(veclist2)
    nearest = []
    for q in veclist1:
        tau, best = search_vp(q, vptree)
        x = best
        nearest.append(x)
    vp_time = time() - start
    print(time_message.format(list_length, tuple_dim, vp_time))

    assert nearest == veclist2_copy
