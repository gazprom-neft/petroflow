"""Various types of DataFrame joins."""

import numpy as np
import pandas as pd


def cross_join(left, right, suffixes=("_left", "_right")):
    """Return the cartesian product of rows from both tables.

    Parameters
    ----------
    left, right : pandas.DataFrame
        Tables to join.
    suffixes : tuple of two str
        Suffix to append to overlapping column names in the left and right
        side respectively. Defaults to `("_left", "_right")`.

    Returns
    -------
    df : pandas.DataFrame
        Cross-joined `DataFrame`.

    Examples
    --------
    >>> df_left = pd.DataFrame({"DEPTH": [1, 2],
    ...                         "VALUE": [3, 4]})
    >>> df_left
       DEPTH  VALUE
    0      1      3
    1      2      4

    >>> df_right = pd.DataFrame({"DEPTH": [1, 2],
    ...                          "VALUE": [5, 6]})
    >>> df_right
       DEPTH  VALUE
    0      1      5
    1      2      6

    >>> cross_join(df_left, df_right)
       DEPTH_left  VALUE_left  DEPTH_right  VALUE_right
    0           1           3            1            5
    1           1           3            2            6
    2           2           4            1            5
    3           2           4            2            6
    """
    left["_TMP_KEY"] = 1
    right["_TMP_KEY"] = 1
    res = pd.merge(left, right, on="_TMP_KEY", suffixes=suffixes).drop("_TMP_KEY", axis=1)
    left.drop("_TMP_KEY", axis=1, inplace=True)
    right.drop("_TMP_KEY", axis=1, inplace=True)
    return res


def between_join(left, right, left_on="DEPTH", right_on=("DEPTH_FROM", "DEPTH_TO"), suffixes=("_left", "_right")):
    """Return a `DataFrame`, consisting of all combinations of rows from both
    `left` and `right`, so that `left.left_on` is between `right.right_on[0]`
    (inclusive) and `right.right_on[1]` (exclusive).

    Parameters
    ----------
    left : pandas.DataFrame
        Table with depth values.
    right : pandas.DataFrame
        Table with depth ranges.
    left_on : str
        Depth column in `left`. Defaults to `"DEPTH"`.
    right_on : tuple of two str
        Columns, specifying depth ranges in `right`. Defaults to
        `("DEPTH_FROM", "DEPTH_TO")`.
    suffixes : tuple of two str
        Suffix to append to overlapping column names in the left and right
        side respectively. Defaults to `("_left", "_right")`.

    Returns
    -------
    df : pandas.DataFrame
        Joined `DataFrame`.

    Examples
    --------
    >>> df_left = pd.DataFrame({"DEPTH": [2, 4],
    ...                         "VALUE": [1, 2]})
    >>> df_left
       DEPTH  VALUE
    0      2      1
    1      4      2

    >>> df_right = pd.DataFrame({"DEPTH_FROM": [1, 3],
    ...                          "DEPTH_TO": [3, 5],
    ...                          "VALUE": [1, 2]})
    >>> df_right
       DEPTH_FROM  DEPTH_TO  VALUE
    0           1         3      1
    1           3         5      2

    >>> between_join(df_left, df_right)
       DEPTH  VALUE_left  DEPTH_FROM  DEPTH_TO  VALUE_right
    0      2           1           1         3            1
    3      4           2           3         5            2
    """
    # TODO: avoid cross join
    right_from, right_to = right_on
    cross = cross_join(left, right, suffixes=suffixes)
    mask = (cross[left_on] >= cross[right_from]) & (cross[left_on] < cross[right_to])
    return cross[mask]


def fdtd_join(left, right, left_on=("DEPTH_FROM", "DEPTH_TO"), right_on=("DEPTH_FROM", "DEPTH_TO")):
    """Return inner join of two tables on depth ranges, specified in `left_on`
    and `right_on` (from depth - to depth join).

    Parameters
    ----------
    left, right : pandas.DataFrame
        Tables to join.
    left_on, right_on : tuple of two str
        Columns, specifying depth ranges in `left` and `right` respectively.
        Defaults to `("DEPTH_FROM", "DEPTH_TO")`.

    Returns
    -------
    df : pandas.DataFrame
        Joined `DataFrame`.

    Examples
    --------
    >>> df_left = pd.DataFrame({"DEPTH_FROM": [0, 2],
    ...                         "DEPTH_TO": [2, 5],
    ...                         "VALUE": [1, 2]})
    >>> df_left
       DEPTH_FROM  DEPTH_TO  VALUE
    0           0         2      1
    1           2         5      2

    >>> df_right = pd.DataFrame({"DEPTH_FROM": [1, 3],
    ...                          "DEPTH_TO": [3, 5],
    ...                          "VALUE": [1, 2]})
    >>> df_right
       DEPTH_FROM  DEPTH_TO  VALUE
    0           1         3      1
    1           3         5      2

    >>> fdtd_join(df_left, df_right)
       DEPTH_FROM  DEPTH_TO  VALUE_left  VALUE_right
    2           1         2           1            1
    4           2         3           2            1
    7           3         5           2            2
    """
    # TODO: avoid cross joins
    left_from, left_to = left_on
    right_from, right_to = right_on

    depths = pd.concat([left[left_from], left[left_to], right[right_from], right[right_to]])
    depths = np.sort(pd.unique(depths))
    res = pd.DataFrame({"_TMP_DEPTH_FROM": depths[:-1],
                        "_TMP_DEPTH_TO": depths[1:]})

    res = cross_join(res, left)
    mask = (res["_TMP_DEPTH_FROM"] >= res[left_from]) & (res["_TMP_DEPTH_TO"] <= res[left_to])
    res = res[mask].drop([left_from, left_to], axis=1)

    res = cross_join(res, right)
    mask = (res["_TMP_DEPTH_FROM"] >= res[right_from]) & (res["_TMP_DEPTH_TO"] <= res[right_to])
    res = res[mask].drop([right_from, right_to], axis=1)

    res = res.rename(columns={"_TMP_DEPTH_FROM": "DEPTH_FROM", "_TMP_DEPTH_TO": "DEPTH_TO"})
    return res
