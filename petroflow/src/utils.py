"""Miscellaneous utility functions."""

import re
import warnings
import functools

import pint
import numpy as np
from numba import njit

UNIT_REGISTRY = pint.UnitRegistry()


def to_list(obj):
    """Cast an object to a list. Almost identical to `list(obj)` for 1-D
    objects, except for `str`, which won't be split into separate letters but
    transformed into a list of a single element.
    """
    return np.array(obj).ravel().tolist()


def process_columns(*dec_args, dst_from_result=False):
    """Decorate a `method` so that it is applied to `src` columns of an `attr`
    well attribute and the result is saved to `dst` columns of this attribute.

    A `dst_from_result` keyword argument can be passed to the decorator
    to redefine the default value of the added argument with the same name.

    Adds the following additional arguments to the decorated method:
    ----------------------------------------------------------------
    attr : str, optional
        `WellSegment` attribute to get the data from. Defaults to "logs".
    src : str or list of str or None, optional
        `attr` columns to pass to the method. Defaults to all columns.
    except_src : str or list of str or None, optional
        All `attr` columns, except these, will be passed to the method. Can't
        be specified together with `src`. By default, all columns are passed.
    dst : str or list of str or None, optional
        `attr` columns to save the result into. Defaults to `src`.
    drop_src : bool, optional
        Specifies whether to drop `src` columns from `attr` after the method
        call. Defaults to `False`.
    dst_from_result : bool, optional
        Whether use column names of the dataframe returned by class method as
        `dst` when saving final result. Defaults to `False`.
    """
    def wrapper_caller(method):
        @functools.wraps(method)
        def wrapper(self, *args, attr="logs", src=None, except_src=None, dst=None,
                    drop_src=False, dst_from_result=None, **kwargs):
            df = getattr(self, attr)
            if (src is not None) and (except_src is not None):
                raise ValueError("src and except_src can't be specified together")
            if src is not None:
                src = to_list(src)
            elif except_src is not None:
                # Calculate the difference between df.columns and except_src, preserving the order of columns in df
                except_src = np.unique(except_src)
                src = np.setdiff1d(df.columns, except_src, assume_unique=True)
            else:
                src = df.columns

            dst_from_result = dst_from_result_ if dst_from_result is None else dst_from_result
            result = method(self, df[src], *args, **kwargs)
            if dst is None:
                dst = result.columns if dst_from_result else src
            else:
                if dst_from_result:
                    warnings.warn('Column names of dataframe returned by {} are overwritten by your custom `dst`. '
                                  'To suppress this warning explicitly pass `dst_from_result=False` to the method call.'
                                  .format(method.__qualname__))
                dst = to_list(dst)
            df[dst] = result

            if drop_src:
                df.drop(set(src) - set(dst), axis=1, inplace=True)
            return self
        return wrapper

    dst_from_result_ = dst_from_result
    if len(dec_args) == 1 and callable(dec_args[0]):
        return wrapper_caller(method=dec_args[0])
    if len(dec_args) != 0:
        raise ValueError("Decorator `process_columns` takes only named arguments")
    return wrapper_caller


def parse_depth(depth, check_positive=False, var_name="Depth/length"):
    """Convert `depth` to centimeters and validate, that it has `int` type.
    Optionally check that it is positive.

    Parameters
    ----------
    depth : int or str
        Depth value to parse.
    check_positive : bool, optional
        Specifies, whether to check that depth is positive. Defaults to
        `False`.
    var_name : str, optional
        Variable name to check, used to create meaningful exception messages.
        Defaults to "Depth/length".

    Returns
    -------
    depth : int
        Depth value converted to centimeters.
    """
    if isinstance(depth, str):
        regexp = re.compile(r"(?P<value>[-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?)(?P<units>[a-zA-Z]+)")
        match = regexp.fullmatch(depth)
        if not match:
            raise ValueError("{} must be specified in a <value><units> format".format(var_name))
        depth = float(match.group("value")) * UNIT_REGISTRY(match.group("units")).to("cm").magnitude
        if depth.is_integer():
            depth = int(depth)
    if not isinstance(depth, (int, np.integer)):
        raise ValueError("{} must have int type".format(var_name))
    if check_positive and depth <= 0:
        raise ValueError("{} must be positive".format(var_name))
    return depth


@njit
def fast_fill_intervals(arr, starts, ends, values):
    """Accelerate interval-based array numeric value fill."""
    for start, end, value in zip(starts, ends, values):
        arr[start:end] = value
    return arr
