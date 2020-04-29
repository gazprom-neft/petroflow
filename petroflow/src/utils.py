"""Miscellaneous utility functions."""

import re
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


def process_columns(method):
    """Decorate a `method` so that it is applied to `src` columns of an `attr`
    well attribute and stores the result in `dst` columns of the same
    attribute.

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
    """
    @functools.wraps(method)
    def wrapper(self, *args, attr="logs", src=None, except_src=None, dst=None, drop_src=False, **kwargs):
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
        dst = src if dst is None else to_list(dst)
        df[dst] = method(self, df[src], *args, **kwargs)
        if drop_src:
            df.drop(set(src) - set(dst), axis=1, inplace=True)
        return self
    return wrapper


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
def fill_intervals(arr, starts, ends, values):
    """Accelerate interval-based array value fill."""
    for start, end, value in zip(starts, ends, values):
        arr[start:end] = value
    return arr
