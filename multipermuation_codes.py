import numpy as np
import itertools as it
import functools


def multipermutation(sigma, r):
    """
    sigma is a permutation of n (i.e. a length-n sequence of each
    of the numbers from range(n))
    r is a positive integer that divides n
    returns a length-n sequence m of values from range(n//r) defined by
    m[i] = sigma[i]//r
    """

    return np.array([int(s)//r for s in sigma])

# the following works but is too slow due (at least in part) to
# redundancy. it calculates each multipermutation 2^(n/2) times. hence
# the need for multipermutations_faster, below.


def multipermutations(n, r):
    """returns an np array consisting of all r-multipermutations of n"""
    return np.unique(np.array([multipermutation(s, r) for s in
                              it.permutations(range(n))]), axis=0)


@functools.lru_cache(maxsize=None)
def multipermutations_faster(n, r):
    """returns an np array consisting of all r-multipermutations of n"""
    if n % r != 0:
        return "Invalid input: r must divide n"
    if n == 0:
        return np.array([[]], dtype=int)
    else:
        multiperms = np.empty(shape=(0, n), dtype=int)
        multiperms_prev = multipermutations_faster(n-r, r)
        for combo in it.combinations(range(n), r):
            for x in multiperms_prev:
                for i in combo:
                    x = np.insert(x, i, n//r-1)
                multiperms = np.append(multiperms, [x], axis=0)
        return multiperms

    
def multiperm_equivalence(sigma, tau, r):
    """returns True iff multipermutation(sigma) == multipermutation(tau)"""
    return np.array_equal(multipermutation(sigma, r), multipermutation(tau, r))


def step_downs(sigma):
    """returns the bits (digits, places, etc) where the next bit is is
    (strictly) less than the current bit

    """
    t = np.zeros(len(sigma)-1, dtype=int)
    for i in range(len(sigma)-1):
        if sigma[i] > sigma[i+1]:
            t[i] = 1
    return t


def remove_reinsert(sigma, i, j):
    """
    moves the element in position i to position j;
    other elements shift as necessary.

    """
    sigma = np.array(sigma)
    if j <= i:
        return np.concatenate((sigma[0:j], sigma[i:i+1], sigma[j:i],
                              sigma[i+1:]))
    else:
        return np.concatenate((sigma[0:i], sigma[i+1:j+1], sigma[i:i+1],
                              sigma[j+1:]))

    
def remove_reinsert_neighbors(sigma):
    """returns an np array of all strings that can be obtained by removing a
    single bit (digit etc.) from sigma and reinserting it in any
    place

    """
    neighbors = np.array([list(sigma)])

    for p in it.permutations(range(len(sigma)), 2):
        neighbors = np.append(neighbors, [remove_reinsert(sigma, p[0],
                                                          p[1])], axis=0)
        neighbors = np.append(neighbors, [remove_reinsert(sigma, p[1],
                                                          p[0])], axis=0)
    return np.unique(neighbors, axis=0)


def positional_sum(sigma):
    """
    returns the dot product of the length-n input string sigma with
    [1, 2, ..., n]

    """
    sum = 0
    for i in range(len(sigma)):
        sum += (i+1)*sigma[i]
    return sum


def check_lev_cons(n, r):
    """returns a dictionary indexed by the r-multipermutations of n; the
    values are the remove-reinsert neighbors whose step downs' positional
    sum is 0 mod n
    """
    # t_start = time.time()
    # witnesses_to_failure = []
    good_neighbors = {}
    for m in multipermutations_faster(n, r):
        m = tuple(m)
        good_neighbors[m] = np.empty(shape=(0, n), dtype=int)
        for neighb in remove_reinsert_neighbors(m):
            if positional_sum(step_downs(neighb)) % n == 0:
                good_neighbors[m] = np.append(good_neighbors[m],
                                              [neighb], axis=0)
        # else:
        #     t_end = time.time()
        #     return ''.join(map(str, m)), False, t_end - t_start
        #     # witnesses_to_failure.append(m)
    # return witnesses_to_failure
    # t_end =time.time()
    return good_neighbors

# the following taken from
# https://www.geeksforgeeks.org/python-program-for-longest-common-subsequence/


def lcs(X, Y, len_X, len_Y):
    """
    returns the length of the longest common subsequence
    (not to be confused with subword) of X and Y
    len_X and len_Y are X's and Y's lengths
    
    """
    if len_X == 0 or len_Y == 0:
        return 0
    elif X[len_X-1] == Y[len_Y-1]:
        return 1 + lcs(X, Y, len_X-1, len_Y-1)
    else:
        return max(lcs(X, Y, len_X, len_Y-1), lcs(X, Y, len_X-1, len_Y))


def ulam_distance(sigma, tau):
    """"""
    m = len(sigma)
    return m-lcs(sigma, tau, m, m)

# distance_matrix=((ulam_distance(sigma, tau) for tau in S_n) for sigma in S_n)


# for n=4, t=1, any two permutations that are distance 3 apart form a
# quasi-permutation code; e.g. (0, 1, 2, 3) & (3, 2, 1, 0)

# for n=4, t=2, not possible b/c 2t+1=4 and no words are that far apart

# now for n>=5 and t=1. let's try building a code greedily. add the
# first and last permutations (since they're the max distance, n-1,
# apart). then for each remaining string, add it if it's at least
# 2t+1=3 away from all the current codewords.

# proposed_code = [S_n[0]] # , S_n[math.factorial(n)-1]]

# for s in S_n:
#     for c in proposed_code:
#         if ulam_distance(s, c)< 3:
#             break
#     else:
#         proposed_code.append(s)


# now check if that's quasi-perfect by checking that there's a codeword
def qp_predicate(code, S, t):
    """"""
    qp_witnesses = []
    for s in S:
        for c in code:
            if ulam_distance(s, c) <= t+1:
                qp_witnesses.append((s, c))
                break
        else:
            return s, "has no witnesses"
    return True


def binary_counter_ex(n):
    """"""
    return np.concatenate(([1], [0 for x in range(n//2 - 3)], [1, 0,
                                                               0, 0],
                           [1 for
                            x in
                            range(n//2
                                  - 2)]))
