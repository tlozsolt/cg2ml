from more_itertools import pairwise, value_chain

def pairwise_value_chain(*args):
    return pairwise(value_chain(*args))

    