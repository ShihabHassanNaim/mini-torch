"""
Mathematical operators for MiniTorch.

These form the foundation of all neural network operations.
You'll implement each function to understand how deep learning
frameworks handle basic mathematics.
"""

import math
from typing import Callable, Iterable


# TODO: Implement these functions in Task 0.1
def mul(x: float, y: float) -> float:
    return x * y

def id(x: float) -> float:
    return x

def add(x: float, y: float) -> float:
    return x + y

def neg(x: float) -> float:
    return -x

def lt(x: float, y: float) -> float:
    return 1.0 if x < y else 0.0

def eq(x: float, y: float) -> float:
    return 1.0 if x == y else 0.0

def max(x: float, y: float) -> float:
    return x if x > y else y

def is_close(x: float, y: float) -> float:
    return 1.0 if abs(x - y) < 1e-2 else 0.0

def sigmoid(x: float) -> float:
    if x >= 0:
        result = 1.0 / (1.0 + math.exp(-x))
    else:
        exp_x = math.exp(x)
        result = exp_x / (1.0 + exp_x)

    return min(1.0 - 1e-15, max(1e-15 , result))

def relu(x: float) -> float:
    return max(0.0, x)

def log(x: float) -> float:
    return math.log(x)

def exp(x: float) -> float:
    return math.exp(x)

def inv(x: float) -> float:
    return 1.0 / x

def log_back(x: float, grad: float) -> float:
    return grad / x

def inv_back(x: float, grad: float) -> float:
    return -grad / (x * x)

def relu_back(x: float, grad: float) -> float:
    return grad if x > 0 else 0.0
  


# TODO: Implement these in Task 0.3
def map(fn: Callable[[float], float]) -> Callable[[Iterable[float]], Iterable[float]]:
    def mapped(ls: Iterable[float]) -> Iterable[float]:
        return [fn(x) for x in ls]
    return mapped


def zipWith(fn: Callable[[float, float], float]) -> Callable[[Iterable[float], Iterable[float]], Iterable[float]]:
    def zipped(ls1: Iterable[float], ls2: Iterable[float]) -> Iterable[float]:
        return [fn(x , y) for x , y in zip(ls1 , ls2)]
    return zipped


def reduce(fn: Callable[[float, float], float], init: float) -> Callable[[Iterable[float]], float]:
    def reduced(ls: Iterable[float]) -> float:
        result = init
        for x in ls:
            result = fn(result, x)
        return result
    return reduced


def sum(ls: Iterable[float]) -> float:
    return reduce(add, 0.0)(ls)


def prod(ls: Iterable[float]) -> float:
    return reduce(mul, 1.0)(ls)


def negList(ls: Iterable[float]) -> Iterable[float]:
    return map(neg)(ls)


def addLists(ls1: Iterable[float], ls2: Iterable[float]) -> Iterable[float]:
    return zipWith(add)(ls1, ls2)
