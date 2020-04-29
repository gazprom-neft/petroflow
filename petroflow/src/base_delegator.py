"""Implements base delegator - a metaclass that creates absent abstract
methods of `WellBatch` and `Well` classes."""

from abc import ABCMeta


class BaseDelegator(ABCMeta):
    """Base metaclass that searches for absent abstract methods and creates
    them."""

    def __new__(mcls, name, bases, namespace):
        abstract_methods = [base.__abstractmethods__ for base in bases if hasattr(base, "__abstractmethods__")]
        abstract_methods = frozenset().union(*abstract_methods)
        for method in abstract_methods:
            if method not in namespace:
                mcls._create_method(method, namespace)
        return super().__new__(mcls, name, bases, namespace)

    @classmethod
    def _create_method(mcls, method, namespace):
        """Create a method, absent in the `namespace`. Must be overridden in
        child classes."""
        raise NotImplementedError
