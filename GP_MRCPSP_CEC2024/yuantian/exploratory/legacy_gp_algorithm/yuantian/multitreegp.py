"""
This module implements MultiPrimitiveTree, a data structure to handle multiple
primitive trees in genetic programming, along with crossover and mutation operations
that operate type-wise on these trees. It also includes decorators for controlling
bloat in genetic programming trees.
This code is based on DEAP (Distributed Evolutionary Algorithms in Python) library.
"""
import copy
from typing import Hashable, Union
from functools import wraps
import random
from enum import Enum
from deap.gp import PrimitiveTree, cxOnePoint, PrimitiveSet, cxOnePointLeafBiased


class TerminalTypeEnum(Enum):
    ACTIVITY = "activity"
    MODE = "mode"
    INTEGRATED = "integrated"


class MultiPrimitiveTree(dict):
    def __init__(self, content: dict[Hashable, PrimitiveTree]):
        """Create a MultiPrimitiveTree from a dictionary.
        The key is a TerminalTypeEnum (ACTIVITY, MODE or INTEGRATED)

        """
        for type, subtree in content.items():
            self[type] = PrimitiveTree(subtree)

    def __str__(self) -> str:
        return super().__str__()

    @classmethod
    def from_string(
        cls,
        content: Union[dict[Hashable, str], str],
        pset: Union[PrimitiveSet, dict[Hashable, PrimitiveSet]],
    ):
        """Try to convert a dict which contains expression into a MultiPrimitiveTree given a
        a PrimitiveSet `pset`. The primirive set needs to contain every primitive
        present in the expression.
        {
            "activity": "add(a, b)" -> str,
            "mode": "mul(a, b)" -> str
        }
        =======>
        {
            "activity": add(a, b) -> PrimitiveTree,
            "mode": mul(a, b) -> PrimitiveTree
        }

        Args:
            content (dict): String representation of a Python expression.
            pset (dict): Primitive set from which primitives are selected.
        """
        _content: dict = None
        if isinstance(content, str):
            # try to convert the string into dict
            try:
                _content = eval(content)
            except NameError as e:
                raise ValueError(f"Argument content should be dict like.")
        elif isinstance(content, dict):
            _content = content
        else:
            raise ValueError("Content should be either dict or str")
        tree_dict: dict = {}
        for type, string in _content.items():
            # convert type from strin to enum type
            _pset = None
            if isinstance(pset, PrimitiveSet):
                _pset = pset
            elif isinstance(pset, dict):
                if not pset.get(type, None):
                    raise ValueError(f"{type} is not in {pset=}")
                _pset = pset[type]
            else:
                raise ValueError(f"pset should be PrimitiveSet or dict")

            tree_dict[type] = PrimitiveTree.from_string(string, _pset)
        return MultiPrimitiveTree(tree_dict)

    def __str__(self) -> str:
        return str({type: str(ind) for type, ind in self.items()})


def cxOnePoint_type_wise(ind1: MultiPrimitiveTree, ind2: MultiPrimitiveTree):
    """Exchange subtrees between each individual with the same tree type

    Args:
        ind1 (MultiPrimitiveTree): First tree participating in the crossover
        ind2 (MultiPrimitiveTree): Second tree participating in the crossover

    Retutrn:
        A tuple of two trees
    """
    if ind1.keys() != ind2.keys():
        raise ValueError("ind1 and ind2 doesn't have the same tree type")
    keys = ind1.keys()
    for k in keys:
        ind1[k], ind2[k] = cxOnePoint(ind1=ind1[k], ind2=ind2[k])
    return ind1, ind2

def cxOnePoint_type_wise_leaf_biased(ind1: MultiPrimitiveTree, ind2: MultiPrimitiveTree, termpb=0.1):
    """Exchange subtrees between each individual with the same tree type

    Args:
        ind1 (MultiPrimitiveTree): First tree participating in the crossover
        ind2 (MultiPrimitiveTree): Second tree participating in the crossover

    Retutrn:
        A tuple of two trees
    """
    if ind1.keys() != ind2.keys():
        raise ValueError("ind1 and ind2 doesn't have the same tree type")
    keys = ind1.keys()
    for k in keys:
        ind1[k], ind2[k] = cxOnePointLeafBiased(ind1=ind1[k], ind2=ind2[k], termpb=termpb)
    return ind1, ind2


def multi_tree_mutate(individual: MultiPrimitiveTree, expr, pset, mutate_func:callable):
    for k in individual.keys():
        mutate_func(individual[k], expr[k], pset[k])
    return (individual,)


    


######################################
# Multi-tree GP bloat control decorators        #
######################################


def staticLimit(key, max_value):
    """Implement a static limit on some measurement on a GP tree, as defined
    by Koza in [Koza1989]. It may be used to decorate both crossover and
    mutation operators. When an invalid (over the limit) child is generated,
    it is simply replaced by one of its parents, randomly selected.

    This operator can be used to avoid memory errors occurring when the tree
    gets higher than 90 levels (as Python puts a limit on the call stack
    depth), because it can ensure that no tree higher than this limit will ever
    be accepted in the population, except if it was generated at initialization
    time.

    :param key: The function to use in order the get the wanted value. For
                instance, on a GP tree, ``operator.attrgetter('height')`` may
                be used to set a depth limit, and ``len`` to set a size limit.
    :param max_value: The maximum value allowed for the given measurement.
    :returns: A decorator that can be applied to a GP operator using \
    :func:`~deap.base.Toolbox.decorate`

    .. note::
       If you want to reproduce the exact behavior intended by Koza, set
       *key* to ``operator.attrgetter('height')`` and *max_value* to 17.

    .. [Koza1989] J.R. Koza, Genetic Programming - On the Programming of
        Computers by Means of Natural Selection (MIT Press,
        Cambridge, MA, 1992)

    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            keep_inds = [copy.deepcopy(ind) for ind in args]
            new_inds = list(func(*args, **kwargs))
            for i, ind in enumerate(new_inds):
                for type, sub_ind in ind.items():
                    if key(sub_ind) > max_value[type]:
                        new_inds[i][type] = random.choice(keep_inds)[type]
            return new_inds

        return wrapper

    return decorator
