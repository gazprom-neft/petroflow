"""Implements WellSegment - a class, representing a contiguous part of a well.
"""
# pylint: disable=no-member,protected-access

import os
import re
import json
import base64
import shutil
import warnings
from copy import copy, deepcopy
from glob import glob
from functools import reduce
from itertools import chain, repeat

import numpy as np
import pandas as pd
import lasio
import PIL
from scipy.interpolate import interp1d
from sklearn.linear_model import LinearRegression
import cv2
from plotly import graph_objs as go
from plotly.subplots import make_subplots
from plotly.offline import init_notebook_mode, plot, iplot

from .abstract_classes import AbstractWellSegment
from .matching import select_contigious_intervals, match_boring_sequence, find_best_shifts, create_zero_shift
from .joins import cross_join, between_join, fdtd_join
from .utils import to_list, process_columns, parse_depth
from .exceptions import SkipWellException, DataRegularityError


def add_attr_properties(cls):
    """Add missing properties for lazy loading of `WellSegment` table-based
    attributes."""
    for attr in cls.attrs_depth_index + cls.attrs_fdtd_index + cls.attrs_no_index:
        if hasattr(cls, attr):
            continue
        def prop(self, attr=attr):
            if getattr(self, "_" + attr) is None:
                getattr(self, "load_" + attr)()
            return getattr(self, "_" + attr)
        setattr(cls, attr, property(prop))
    return cls


def add_attr_loaders(cls):
    """Add missing loaders of `WellSegment` table-based attributes."""
    attr_iter = chain(
        zip(cls.attrs_depth_index, repeat(cls._load_depth_df)),
        zip(cls.attrs_fdtd_index, repeat(cls._load_fdtd_df)),
        zip(cls.attrs_no_index, repeat(cls._load_df))
    )
    for attr, loader in attr_iter:
        if hasattr(cls, "load_" + attr):
            continue
        def load_factory(attr, loader):
            def load(self, *args, **kwargs):
                data = loader(self, self._get_full_name(self.path, attr), *args, **kwargs)
                setattr(self, "_" + attr, data)
                return self
            return load
        setattr(cls, "load_" + attr, load_factory(attr, loader))
    return cls


@add_attr_properties
@add_attr_loaders
class WellSegment(AbstractWellSegment):
    """A class, representing a contiguous part of a well.

    Initially an instance of `Well` class consists of a single segment,
    representing the whole well. Subsequently, several `Well` methods, such as
    `crop`, `random_crop` or `drop_nans`, increase the number of segments,
    storing them in a tree-based structure.

    Unlike `Well`, which nearly always redirects method calls to its segments,
    `WellSegment` actually stores well data and implements its processing
    logic.

    Each table-based attribute of a `WellSegment`, described in `Attributes`
    section, can be loaded in two different ways:
    1. via corresponding load method, e.g. `load_logs` or `load_layers`. All
       specified arguments will be passed to a loader, responsible for file's
       extension.
    2. via lazy loading mechanism, when an attribute is loaded at the time of
       the first access. In this case, default loading parameters are used.
    `core_dl` and `core_uv` attributes are loaded either by accessing them for
    the first time, or by calling `load_core` method.

    Depth and length values in all the attributes are assumed to be stored in
    centimeters for all formats, except for `.las`, where depths units are
    assumed to be meters and are converted to centimeters by the `load_las`
    method. Log units are not parsed from a `.las` file header since they are
    optional and their format is not strictly fixed.

    Parameters
    ----------
    path : str
        A path to a directory with well data, containing:
        - `meta.json` - a json dict with the following keys:
            - `name` - well name
            - `field` - field name
            - `depth_from` - minimum depth entry in the well logs (in cm)
            - `depth_to` - maximum depth entry in the well logs (in cm)
          These values will be stored as instance attributes.
        - `samples_dl` and `samples_uv` (optional) - directories, containing
          daylight and ultraviolet images of core samples respectively. Images
          of the same sample must have the same name in both dirs.
        - Optional `.csv`, `.las` or `.feather` file for certain class
          attributes (see more details in the `Attributes` section).
    core_width : positive float, optional
        The width of core samples in cm. Defaults to 10 cm.
    pixels_per_cm : positive float, optional
        The number of pixels in cm used to determine the loaded width of core
        sample images. Image height is calculated so as to keep the aspect
        ratio. Defaults to 5 pixels.
    validate : bool, optional
        Specifies whether to check well data for correctness and consistency.
        Slightly reduces processing speed. Defaults to `True`.

    Attributes
    ----------
    name : str
        Well name, loaded from `meta.json`.
    field : str
        Field name, loaded from `meta.json`.
    depth_from : int or str
        Minimum depth entry in the well logs in cm, loaded from `meta.json`.
        If `str`, must be specified in a <value><units> format (e.g. "10m").
    depth_to : int or str
        Maximum depth entry in the well logs in cm, loaded from `meta.json`.
        If `str`, must be specified in a <value><units> format (e.g. "10m").
    actual_depth_to : int
        Actual maximum segment depth in cm. It is defined if the segment has
        been padded by the `crop` method and is then used in `Well.aggregate`
        to drop padded part of the segment.
    logs : pandas.DataFrame
        Well logs, indexed by depth. Depth log in a source file must have
        `DEPTH` mnemonic. Mnemonics of the same log type in `logs` and
        `core_logs` should match. Loaded from the file with the same name from
        the well directory.
    logs_step: positive int
        Step between measurements in `logs` in centimeters.
    inclination : pandas.DataFrame
        Well inclination. Loaded from the file with the same name from the
        well directory.
    layers : pandas.DataFrame
        Stratum layers names, indexed by depth range. Loaded from the file
        with the same name, having the following structure: DEPTH_FROM -
        DEPTH_TO - LAYER.
    boring_intervals : pandas.DataFrame
        Depths of boring intervals with core recovery in centimeters, indexed
        by depth range. Loaded from the file with the same name, having the
        following structure: DEPTH_FROM - DEPTH_TO - CORE_RECOVERY.
    boring_sequences : pandas.DataFrame
        Depth ranges of contiguous boring intervals, extracted one after
        another. If the file with the same name exists in the well directory,
        then `boring_sequences` is loaded from the file. Otherwise, it is
        calculated from `boring_intervals`. If core-to-log matching is
        performed, then extra `MODE` and `R2` columns with matching parameters
        and results are created.
    core_properties : pandas.DataFrame
        Properties of core plugs, indexed by depth. Depth column in source
        file must be called `DEPTH`. Loaded from the file with the same name
        from the well directory.
    core_lithology : pandas.DataFrame
        Lithological description of core samples, indexed by depth range.
        Loaded from the file with the same name, having the following
        structure: DEPTH_FROM - DEPTH_TO - FORMATION - COLOR - GRAINSIZE -
        GRAINCONTENT.
    core_logs : pandas.DataFrame
        Core logs, indexed by depth. Depth log in a source file must have
        `DEPTH` mnemonic. Mnemonics of the same log type in `logs` and
        `core_logs` should match. Loaded from the file with the same name from
        the well directory.
    samples : pandas.DataFrame
        Names of core sample images with their depth ranges. Loaded from the
        file with the same name, having the following structure: DEPTH_FROM -
        DEPTH_TO - SAMPLE.
    core_dl : numpy.ndarray
        Concatenated daylight image of all core samples in the segment. If
        core samples are absent for several depth ranges, corresponding
        `core_dl` values are equal to `numpy.nan`. Loaded from images in
        `samples_dl` directory, requires `samples` file to exist in the well
        directory.
    core_uv : numpy.ndarray
        Concatenated ultraviolet image of all core samples in the segment. If
        core samples are absent for several depth ranges, corresponding
        `core_uv` values are equal to `numpy.nan`. Loaded from images in
        `samples_uv` directory, requires `samples` file to exist in the well
        directory.
    """

    attrs_depth_index = ("logs", "core_properties", "core_logs")
    attrs_fdtd_index = ("layers", "boring_sequences", "boring_intervals", "core_lithology", "samples")
    attrs_no_index = ("inclination",)
    attrs_image = ("core_uv", "core_dl")

    def __init__(self, path, *args, core_width=10, pixels_per_cm=5, validate=True, **kwargs):
        super().__init__()
        _ = args, kwargs
        self.path = path
        self.core_width = core_width
        self.pixels_per_cm = pixels_per_cm
        self.validate = validate

        with open(os.path.join(self.path, "meta.json")) as meta_file:
            meta = json.load(meta_file)
        self.name = meta["name"]
        self.field = meta["field"]
        self.depth_from = parse_depth(meta["depth_from"], var_name="depth_from")
        self.depth_to = parse_depth(meta["depth_to"], var_name="depth_to")
        self.actual_depth_to = None
        if self.depth_from >= self.depth_to:
            raise ValueError("depth_from must be less than depth_to")

        self.has_samples = self._has_file("samples")

        self.logs_step = None
        self._logs = None
        self._inclination = None
        self._layers = None
        self._boring_intervals = None
        self._boring_sequences = None
        self._core_properties = None
        self._core_lithology = None
        self._core_logs = None
        self._samples = None
        self._core_dl = None
        self._core_uv = None
        self._boring_intervals_deltas = None
        self._core_lithology_deltas = None

        # In order to unify aggregate behavior in case of loaded and calculated `boring_sequences`,
        # they should be computed explicitly during the creation of a segment.
        if self._has_file("boring_intervals") and not self._has_file("boring_sequences"):
            _ = self.boring_sequences

    @property
    def length(self):
        """float: Length of the segment in centimeters."""
        return self.depth_to - self.depth_from

    def load_logs(self, *args, **kwargs):
        """Load well logs and calculate logs step in centimeters.

        Parameters
        ----------
        args : misc
            Any additional positional arguments to pass to the file loader,
            depends on its extension.
        kwargs : misc
            Any additional keyword arguments to pass as keyword arguments to
            the file loader, depends on its extension.

        Returns
        -------
        self : type(self)
            Self with loaded well logs.

        Raises
        ------
        ValueError
            If logs don't have a fixed sampling rate.
        """
        self._logs = self._load_depth_df(self._get_full_name(self.path, "logs"), *args, **kwargs)
        steps = self.logs.index[1:] - self.logs.index[:-1]
        unique_steps = np.unique(steps)
        if self.validate and (len(unique_steps) > 1):
            raise ValueError("Well logs must have a fixed sampling rate")
        self.logs_step = unique_steps[0]
        return self

    @property
    def boring_sequences(self):
        """pandas.DataFrame: Depth ranges of contiguous boring intervals,
        extracted one after another."""
        if self._boring_sequences is None:
            if self._has_file("boring_sequences"):
                self.load_boring_sequences()
            else:
                self._calc_boring_sequences()
        return self._boring_sequences

    def _calc_boring_sequences(self):
        """Calculate boring sequences given boring intervals."""
        data = []
        for segment in select_contigious_intervals(self.boring_intervals.reset_index()):
            data.append([segment["DEPTH_FROM"].min(), segment["DEPTH_TO"].max()])
        self._boring_sequences = pd.DataFrame(data, columns=["DEPTH_FROM", "DEPTH_TO"])
        self._boring_sequences.set_index(["DEPTH_FROM", "DEPTH_TO"], inplace=True)

    @staticmethod
    def _get_extension(path):
        """Get file extension from its name."""
        return os.path.splitext(path)[1][1:]

    @staticmethod
    def _load_las(path, *args, **kwargs):
        """Load a `.las` file into a `DataFrame`."""
        df = lasio.read(path, *args, **kwargs).df().reset_index().rename(columns={"DEPT": "DEPTH"})
        # DEPTH values in a .las file are assumed to be in meters and converted to centimeters in the next line. Units
        # are not parsed from .las file header since they are optional and their format is not strictly fixed.
        df["DEPTH"] = np.round(df["DEPTH"] * 100).astype(int)
        return df

    @staticmethod
    def _load_csv(path, *args, **kwargs):
        """Load a `.csv` file into a `DataFrame`."""
        return pd.read_csv(path, *args, **kwargs)

    @staticmethod
    def _load_feather(path, *args, **kwargs):
        """Load a `.feather` file into a `DataFrame`."""
        return pd.read_feather(path, *args, **kwargs)

    def _load_df(self, path, *args, **kwargs):
        """Load a `DataFrame` from a table format (`.las`, `.csv` or
        `.feather`) depending on its extension."""
        ext = self._get_extension(path)
        if not hasattr(self, "_load_" + ext):
            raise ValueError("A loader for data in {} format is not implemented".format(ext))
        return getattr(self, "_load_" + ext)(path, *args, **kwargs)

    def _filter_depth_df(self, df):
        """Keep only depths between `self.depth_from` and `self.depth_to` in a
        `DataFrame`, indexed by depth."""
        df = df.loc[self.depth_from:self.depth_to]
        if len(df) == 0:
            return df
        if (self.depth_from == df.index[0]) and (self.depth_to == df.index[-1]):
            # See __getitem__ docstring for an explanation of this behaviour
            df.drop(df.index[-1], inplace=True)
        return df

    @staticmethod
    def _validate_depth_df(df):
        """Check depth-indexed `DataFrame` for data consistency.

        The following checks are performed:
        1. If `DEPTH` is not int.
        2. If index is not unique.
        3. If index is not monotonically increasing.

        Raises
        ------
        DataRegularityError
            If any of checks above failed.
        """
        depth = df.index

        # Check if depth is not int
        if not pd.api.types.is_integer_dtype(depth):
            raise DataRegularityError("non_int_index", depth)

        # Check if depth values are not unique
        if not depth.is_unique:
            raise DataRegularityError("non_unique_index", depth)

        # Check if depth values are not monotonically increasing
        if not depth.is_monotonic_increasing:
            raise DataRegularityError("non_increasing_index", depth)

    def _load_depth_df(self, path, *args, **kwargs):
        """Load a `DataFrame`, indexed by depth, from a table format and keep
        only depths between `self.depth_from` and `self.depth_to`."""
        df = self._load_df(path, *args, **kwargs).set_index("DEPTH")
        if self.validate:
            self._validate_depth_df(df)
        df = self._filter_depth_df(df)
        return df

    def _filter_fdtd_df(self, df):
        """Keep only depths between `self.depth_from` and `self.depth_to` in a
        `DataFrame`, indexed by depth range."""
        if len(df) == 0:
            return df
        depth_from, depth_to = zip(*df.index.values)
        mask = (np.array(depth_from) < self.depth_to) & (self.depth_from < np.array(depth_to))
        return df[mask]

    @staticmethod
    def _validate_fdtd_df(df):
        """Check fdtd-indexed `DataFrame` for data consistency.

        The following checks are performed:
        1. If `DEPTH_FROM` and `DEPTH_TO` are not int.
        2. If index is not unique.
        3. If index is not monotonically increasing.
        4. If `DEPTH_FROM` is greater that or equal to `DEPTH_TO`.
        5. If any adjacent [`DEPTH_FROM`, `DEPTH_TO`) intervals are
           overlapping.

        Raises
        ------
        DataRegularityError
            If any of checks above failed.
        """
        depth_from = df.index.get_level_values("DEPTH_FROM")
        depth_to = df.index.get_level_values("DEPTH_TO")

        # Check if depth_from and depth_to are not int
        if not pd.api.types.is_integer_dtype(depth_from) or not pd.api.types.is_integer_dtype(depth_to):
            raise DataRegularityError("non_int_index", df.index)

        # Check if depth values are not unique
        if not df.index.is_unique:
            raise DataRegularityError("non_unique_index", df.index)

        # Check if depth_from or depth_to values are not monotonically increasing
        if not depth_from.is_monotonic_increasing or not depth_to.is_monotonic_increasing:
            raise DataRegularityError("non_increasing_index", df.index)

        # Check if depth_from is greater than depth_to
        disordered = df[depth_from >= depth_to]
        if len(disordered):
            raise DataRegularityError("disordered_index", disordered)

        # Check if any adjacent [depth_from, depth_to) intervals are overlapping
        if (depth_from.values[1:] < depth_to.values[:-1]).any():
            raise DataRegularityError("overlapping_index", df.index)

    def _load_fdtd_df(self, path, *args, **kwargs):
        """Load a `DataFrame`, indexed by depth range, from a table format and
        keep only depths between `self.depth_from` and `self.depth_to`."""
        df = self._load_df(path, *args, **kwargs).set_index(["DEPTH_FROM", "DEPTH_TO"])
        if self.validate:
            self._validate_fdtd_df(df)
        df = self._filter_fdtd_df(df)
        return df

    def _has_file(self, name):
        """Check that exactly one file with a given name and any extension
        exists in a well directory."""
        files = glob(os.path.join(self.path, name + ".*"))
        if len(files) == 1:
            return True
        return False

    @classmethod
    def _get_full_name(cls, path, name):
        """Get full name of a file with a given `name` in a dir, specified in
        `path`.

        Parameters
        ----------
        path : str
            A path to a directiry to search for a file.
        name : str
            File name with or without extension.

        Returns
        -------
        full_name : str
            Full name of a file.

        Raises
        ------
        FileNotFoundError
            If a file does not exist in a dir.
        OSError
            If extension is not specified and several files with given name
            exist.
        """
        ext = cls._get_extension(name)
        if ext != "":
            full_name = os.path.join(path, name)
            if os.path.exists(full_name):
                return full_name
            raise FileNotFoundError("A file {} does not exist in {}".format(name, path))

        files = glob(os.path.join(path, name + ".*"))
        if len(files) == 0:
            raise FileNotFoundError("A file {} does not exist in {}".format(name, path))
        if len(files) > 1:
            raise OSError("Several files called {} are found in {}".format(name, path))
        return files[0]

    @property
    def core_dl(self):
        """numpy.ndarray: Concatenated daylight image of all core samples in
        the segment."""
        if self._core_dl is None:
            self.load_core()
        return self._core_dl

    @property
    def core_uv(self):
        """numpy.ndarray: Concatenated ultraviolet image of all core samples
        in the segment."""
        if self._core_uv is None:
            self.load_core()
        return self._core_uv

    @staticmethod
    def _load_image(path):
        """Open an image in `PIL` format."""
        return PIL.Image.open(path) if os.path.isfile(path) else None

    @staticmethod
    def _match_samples(dl_img, uv_img, height, width):
        """Match core samples in daylight and ultraviolet by resizing them to
        the given shape."""
        if (dl_img is not None) and (dl_img is not None):
            # TODO: contour matching instead of resizing
            dl_img = np.array(dl_img.resize((width, height), resample=PIL.Image.LANCZOS))
            uv_img = np.array(uv_img.resize((width, height), resample=PIL.Image.LANCZOS))
        elif dl_img is not None:
            dl_img = np.array(dl_img.resize((width, height), resample=PIL.Image.LANCZOS))
            uv_img = np.full((height, width, 3), np.nan, dtype=np.float32)
        else:
            dl_img = np.full((height, width, 3), np.nan, dtype=np.float32)
            uv_img = np.array(uv_img.resize((width, height), resample=PIL.Image.LANCZOS))
        return dl_img, uv_img

    def _cm_to_pixels(self, length):
        """Convert centimeters to pixels given conversion ratio in
        `self.pixels_per_cm`."""
        # int is required to correctly handle numpy float dtypes, which are not converted to int by round
        return int(round(length * self.pixels_per_cm))

    def load_core(self, core_width=None, pixels_per_cm=None):
        """Load core images in daylight and ultraviolet.

        If any of method arguments are not specified, those, passed to
        `__init__`, will be used. Otherwise, they will be overridden in
        `self`.

        Parameters
        ----------
        core_width : positive float, optional
            The width of core samples in centimeters.
        pixels_per_cm : positive float, optional
            The number of pixels in centimeters used to determine the loaded
            width of core sample images. Image height is calculated so as to
            keep the aspect ratio.

        Returns
        -------
        self : type(self)
            Self with loaded core images.
        """
        self.core_width = core_width if core_width is not None else self.core_width
        self.pixels_per_cm = pixels_per_cm if pixels_per_cm is not None else self.pixels_per_cm

        height = self._cm_to_pixels(self.length)
        width = self._cm_to_pixels(self.core_width)

        exist_dl = os.path.isdir(os.path.join(self.path, "samples_dl"))
        exist_uv = os.path.isdir(os.path.join(self.path, "samples_uv"))
        if not exist_dl and not exist_uv:
            raise FileNotFoundError("At least one of samples_dl or samples_uv must exist")
        core_dl = np.full((height, width, 3), np.nan, dtype=np.float32)
        core_uv = np.full((height, width, 3), np.nan, dtype=np.float32)

        for (sample_depth_from, sample_depth_to), sample_name in self.samples["SAMPLE"].iteritems():
            sample_height = self._cm_to_pixels(sample_depth_to - sample_depth_from)
            sample_name = str(sample_name)

            dl_path = self._get_full_name(os.path.join(self.path, "samples_dl"), sample_name)
            dl_img = self._load_image(dl_path)
            uv_path = self._get_full_name(os.path.join(self.path, "samples_uv"), sample_name)
            uv_img = self._load_image(uv_path)
            dl_img, uv_img = self._match_samples(dl_img, uv_img, sample_height, width)

            top_crop = max(0, self._cm_to_pixels(self.depth_from - sample_depth_from))
            bottom_crop = sample_height - max(0, self._cm_to_pixels(sample_depth_to - self.depth_to))
            dl_img = dl_img[top_crop:bottom_crop]
            uv_img = uv_img[top_crop:bottom_crop]

            insert_pos = max(0, self._cm_to_pixels(sample_depth_from - self.depth_from))
            core_dl[insert_pos:insert_pos+dl_img.shape[0]] = dl_img
            core_uv[insert_pos:insert_pos+uv_img.shape[0]] = uv_img

        self._core_dl = core_dl if exist_dl else None
        self._core_uv = core_uv if exist_uv else None
        return self

    def dump(self, path):
        """Dump well segment data.

        Segment attributes are saved in the following manner:
        - `name`, `field`, `depth_from` and `depth_to` attributes are saved in
          `meta.json` file.
        - `core_dl` and `core_uv` are not saved. Instead, `samples_dl` and
          `samples_uv` directories are copied if exist.
        - All other attributes are dumped in feather format.

        Parameters
        ----------
        path : str
            A path to a directory, where well dir with dump will be created.

        Returns
        -------
        self : WellSegment
            Self unchanged.
        """
        path = os.path.join(path, self.name)
        if not os.path.exists(path):
            os.makedirs(path)

        meta = {
            "name": self.name,
            "field": self.field,
            "depth_from": self.depth_from,
            "depth_to": self.depth_to,
        }
        with open(os.path.join(path, "meta.json"), "w") as meta_file:
            json.dump(meta, meta_file)

        for attr in self.attrs_depth_index + self.attrs_fdtd_index + self.attrs_no_index:
            attr_val = getattr(self, "_" + attr)
            if attr_val is None:
                try:
                    shutil.copy2(self._get_full_name(self.path, attr), path)
                except Exception:  # pylint: disable=broad-except
                    pass
            else:
                if attr not in self.attrs_no_index:
                    attr_val = attr_val.reset_index()
                attr_val.to_feather(os.path.join(path, attr + ".feather"))

        if not self.has_samples:
            return self

        # Copy DL and UV images, specified in samples
        def ignore(directory, files):
            _ = directory
            return sorted(set(files) - set(self.samples["SAMPLE"]))

        samples_dl_path = os.path.join(self.path, "samples_dl")
        if os.path.exists(samples_dl_path):
            shutil.copytree(samples_dl_path, os.path.join(path, "samples_dl"),
                            copy_function=os.link, ignore=ignore)

        samples_uv_path = os.path.join(self.path, "samples_uv")
        if os.path.exists(samples_uv_path):
            shutil.copytree(samples_uv_path, os.path.join(path, "samples_uv"),
                            copy_function=os.link, ignore=ignore)

        return self

    @staticmethod
    def _encode(img_path):
        """Encode an image into a `plotly` representation."""
        with open(img_path, "rb") as img:
            encoded_img = base64.b64encode(img.read()).decode()
        encoded_img = "data:image/png;base64," + encoded_img
        return encoded_img

    def _plot_core(self, subplot_titles, traces, images, plot_core, image_dir, title):
        """Create a trace with core images if `plot_core` is `True` and
        `image_dir` exists in a well dir. Return the index of the created
        subplot.

        Note, that the method has side effects: it updates `subplot_titles`,
        `traces` and `images` lists inplace.
        """
        plot_core = plot_core and os.path.isdir(os.path.join(self.path, image_dir))
        if not plot_core:
            return None
        col = len(traces) + 1
        subplot_titles += [title]
        traces.append(go.Scatter(x=[0, 1], y=[self.depth_from / 100, self.depth_to / 100], opacity=0))
        samples = self.samples.reset_index()[["DEPTH_FROM", "DEPTH_TO", "SAMPLE"]]
        for _, (depth_from, depth_to, sample_name) in samples.iterrows():
            depth_from /= 100
            depth_to /= 100
            sample_name = str(sample_name)

            sample_path = self._get_full_name(os.path.join(self.path, image_dir), sample_name)
            sample_image = self._encode(sample_path)
            sample_image = go.layout.Image(source=sample_image, xref="x"+str(col), yref="y", x=0, y=depth_from,
                                           sizex=1, sizey=depth_to-depth_from, sizing="stretch", layer="below")
            images.append(sample_image)
        return col

    def plot(self, plot_core=True, interactive=True, subplot_height=700, subplot_width=200):
        """Plot well logs and core images.

        All well logs and core images in daylight and ultraviolet are plotted
        on separate subplots. If called for a `Well` or `WellBatch` instance,
        an extra `aggregate` argument can be passed.

        Parameters
        ----------
        plot_core : bool
            Specifies whether to plot core images or not. Defaults to `True`.
        interactive : bool
            Specifies whether to draw a plot directly inside a Jupyter
            notebook. Defaults to `True`.
        subplot_height : positive int
            Height of each subplot with well log or core samples images in
            pixels. Defaults to 700.
        subplot_width : positive int
            Width of each subplot with well log or core samples images in
            pixels. Defaults to 200.
        aggregate : bool
            Specifies whether to plot all segments of the well on the same
            plot or create a separate plot for each segment. Creates one plot
            by default.

        Returns
        -------
        self : type(self)
            Self unchanged.
        """
        init_notebook_mode(connected=True)

        # Approximate size of the left margin of a plot, that will be added to the calculated
        # figsize in order to prevent figure shrinkage in case of small number of subplots
        margin = 120

        # Create traces for well logs
        index = self.logs.index / 100
        subplot_titles = list(self.logs.columns)
        traces = [go.Scatter(x=self.logs[mnemonic], y=index, mode="lines") for mnemonic in subplot_titles]

        # Create DL and UV traces
        images = []
        dl_col = self._plot_core(subplot_titles, traces, images, plot_core, "samples_dl", "CORE DL")
        uv_col = self._plot_core(subplot_titles, traces, images, plot_core, "samples_uv", "CORE UV")

        # Create a figure
        fig = make_subplots(rows=1, cols=len(traces), shared_yaxes=True, subplot_titles=subplot_titles)
        for i, trace in enumerate(traces, 1):
            fig.append_trace(trace, 1, i)

        # Update figure layout
        layout = fig.layout
        fig_layout = go.Layout(title="{} {}".format(self.field.capitalize(), self.name), showlegend=False,
                               width=len(traces)*subplot_width + margin, height=subplot_height,
                               yaxis=dict(range=[self.depth_to / 100, self.depth_from / 100]), images=images)
        layout.update(fig_layout)

        for key in layout:
            if key.startswith("xaxis"):
                layout[key]["fixedrange"] = True

        if dl_col is not None:
            layout["xaxis" + str(dl_col)]["showticklabels"] = False
            layout["xaxis" + str(dl_col)]["showgrid"] = False

        if uv_col is not None:
            layout["xaxis" + str(uv_col)]["showticklabels"] = False
            layout["xaxis" + str(uv_col)]["showgrid"] = False

        for ann in layout["annotations"]:
            ann["font"]["size"] = 12

        plot_fn = iplot if interactive else plot
        plot_fn(fig)
        return self

    def __getitem__(self, key):
        """Select well logs by mnemonics or slice the well along the wellbore.

        Parameters
        ----------
        key : str, list of str or slice
            - If `key` is `str` or `list` of `str`, preserve only those logs
              in `self.logs`, that are in `key`.
            - If `key` is `slice` - perform well slicing along the wellbore.
              If both `start` and `stop` are in `self.logs.index`, then only
              `start` is kept in the resulting segment to ensure, that such
              methods as `crop` always return the same number of samples
              regardless of cropping position if crop length is given in
              centimeters and no less than `logs_step`. If only one of the
              ends of the slice present in the index, it is kept in the result
              contrary to usual python slices.

        Returns
        -------
        well : WellSegment
            A segment with filtered logs or depths.
        """
        if not isinstance(key, slice):
            return self.keep_logs(key)
        res = self.copy()
        if key.step is not None:
            raise ValueError("A well does not support slicing with a specified step")

        if key.start is not None:
            res.depth_from = max(parse_depth(key.start, var_name="depth_from"), res.depth_from)

        if key.stop is not None:
            res.depth_to = min(parse_depth(key.stop, var_name="depth_to"), res.depth_to)

        if res.depth_from > res.depth_to:
            raise SkipWellException("Slicing interval is out of segment bounds")

        # Slice attributes
        attr_iter = chain(
            zip(res.attrs_depth_index, repeat(res._filter_depth_df)),
            zip(res.attrs_fdtd_index, repeat(res._filter_fdtd_df))
        )
        for attr, filt in attr_iter:
            attr_val = getattr(res, "_" + attr)
            if attr_val is not None:
                setattr(res, "_" + attr, filt(attr_val))

        # Slice images
        start_pos = self._cm_to_pixels(res.depth_from - self.depth_from)
        stop_pos = self._cm_to_pixels(res.depth_to - self.depth_from)
        if res._core_dl is not None:
            res._core_dl = res._core_dl[start_pos:stop_pos]
        if res._core_uv is not None:
            res._core_uv = res._core_uv[start_pos:stop_pos]
        return res

    def copy(self):
        """Perform a shallow copy of an object.

        Returns
        -------
        self : WellSegment
            Shallow copy.
        """
        return copy(self)

    def deepcopy(self):
        """Perform a deep copy of an object.

        Returns
        -------
        self : WellSegment
            Deep copy.
        """
        return deepcopy(self)

    def validate_core(self, validate_lithology=True):
        """Check core data for consistency.

        The following checks are performed for `boring_intervals` dataframe:
        1. All the checks from `_validate_fdtd_df`.
        2. If any values of `CORE_RECOVERY` are nan.
        3. If any values of `CORE_RECOVERY` are non-positive.
        4. If any values of `CORE_RECOVERY` are greater than the length of the
           corresponding interval.

        The following checks are performed for `core_lithology` dataframe:
        1. All the checks from `_validate_fdtd_df`.
        2. If any [`DEPTH_FROM`, `DEPTH_TO`) interval is not included in the
           corresponding interval from `boring_intervals` dataframe.
        3. If total length of all lithology intervals of a boring interval
           does not match its core recovery.

        Parameters
        ----------
        check_lithology : bool, optional
            Specifies whether to check lithology data. Can be turned off if
            core-to-log matching is performed only by shifting boring
            intervals. Defaults to `True`.

        Returns
        -------
        self : type(self)
            Self unchanged.

        Raises
        ------
        DataRegularityError
            If any of checks above failed.
        """
        if not self._has_file("boring_intervals"):
            raise SkipWellException("boring_intervals file is reqiured to perform the checks")

        # Run fdtd checks for boring intervals
        self._validate_fdtd_df(self.boring_intervals)

        # Check if any CORE_RECOVERY values are nan
        nan_recovery_mask = self.boring_intervals["CORE_RECOVERY"].isna()
        if nan_recovery_mask.sum():
            raise DataRegularityError("nan_recovery", self.boring_intervals[nan_recovery_mask])

        # Check if any CORE_RECOVERY values non-positive
        non_positive_mask = self.boring_intervals["CORE_RECOVERY"] <= 0
        if non_positive_mask.sum():
            raise DataRegularityError("non_positive_recovery", self.boring_intervals[non_positive_mask])

        # Check if any CORE_RECOVERY values are greater than the length of the corresponding interval
        index = self.boring_intervals.index
        interval_lengths = index.get_level_values("DEPTH_TO") - index.get_level_values("DEPTH_FROM")
        wrong_recovery_mask = self.boring_intervals["CORE_RECOVERY"] > interval_lengths
        if wrong_recovery_mask.sum():
            raise DataRegularityError("wrong_recovery", self.boring_intervals[wrong_recovery_mask])

        if not validate_lithology or not self._has_file("core_lithology"):
            return self

        # Run fdtd checks for lithology intervals
        self._validate_fdtd_df(self.core_lithology)

        boring_intervals = self.boring_intervals.reset_index()[["DEPTH_FROM", "DEPTH_TO", "CORE_RECOVERY"]]
        lithology_intervals = self.core_lithology.reset_index()[["DEPTH_FROM", "DEPTH_TO"]]
        lithology_intervals["SUM_LENGTH"] = lithology_intervals["DEPTH_TO"] - lithology_intervals["DEPTH_FROM"]

        joined_intervals = cross_join(boring_intervals, lithology_intervals, suffixes=("_BI", "_LI"))
        join_mask = ((joined_intervals["DEPTH_FROM_BI"] <= joined_intervals["DEPTH_FROM_LI"]) &
                     (joined_intervals["DEPTH_TO_BI"] >= joined_intervals["DEPTH_TO_LI"]))
        joined_intervals = joined_intervals[join_mask]

        # Check if any lithology interval is not included in a boring interval
        if len(joined_intervals) != len(lithology_intervals):
            left = joined_intervals[["DEPTH_FROM_LI", "DEPTH_TO_LI"]]
            left.columns = ["DEPTH_FROM", "DEPTH_TO"]
            right = lithology_intervals[["DEPTH_FROM", "DEPTH_TO"]]
            diff = pd.concat([left, right]).drop_duplicates(keep=False)
            raise DataRegularityError("lithology_ranges", diff)

        # Check if total length of all lithology intervals of a boring interval does not match its core recovery
        agg_dict = {
            "DEPTH_FROM_LI": "min",
            "DEPTH_TO_LI": "max",
            "CORE_RECOVERY": "min",  # must be unique
            "SUM_LENGTH": "sum"
        }
        groupped_intervals = joined_intervals.groupby(["DEPTH_FROM_BI", "DEPTH_TO_BI"]).agg(agg_dict)
        groupped_intervals["TOTAL_LENGTH"] = groupped_intervals["DEPTH_TO_LI"] - groupped_intervals["DEPTH_FROM_LI"]
        length_mask = ((groupped_intervals["TOTAL_LENGTH"] != groupped_intervals["CORE_RECOVERY"]) |
                       (groupped_intervals["SUM_LENGTH"] != groupped_intervals["CORE_RECOVERY"]))
        if length_mask.sum():
            raise DataRegularityError("lithology_length", groupped_intervals[length_mask])

        return self

    def validate_samples(self):
        """Check core samples data for consistency.

        The following checks are performed:
        1. If duplicates exist in samples.feather dataframe.
        2. If both samples_dl and samples_uv dirs are missing.
        3. If an image does not exist for a core sample.
        4. If multiple images exist for a core sample and file extension is
           not specified.

        Returns
        -------
        self : type(self)
            Self unchanged.

        Raises
        ------
        DataRegularityError
            If any of checks above failed.
        """
        if not self._has_file("samples"):
            raise SkipWellException("samples file is reqiured to perform the checks")

        samples = [str(sample) for sample in self.samples["SAMPLE"].values]
        if len(samples) > len(set(samples)):
            raise DataRegularityError("duplicated_files", "samples.feather")

        samples_folders = {"samples_dl", "samples_uv"}.intersection(os.listdir(self.path))
        if not samples_folders:
            raise DataRegularityError("missing_samples_dirs")

        for folder in samples_folders:
            path = os.path.join(self.path, folder)
            try:
                _ = [self._get_full_name(path, sample) for sample in samples]
            except Exception as err:
                raise DataRegularityError(str(err))

        return self

    def _apply_matching(self):
        """Update depths in all core-related attributes given calculated
        deltas."""
        # Update DataFrames with depth index
        attrs_depth_index = [attr for attr in self.attrs_depth_index if attr.startswith("core_")]
        for attr in attrs_depth_index:
            if not self._has_file(attr):
                continue
            attr_df = getattr(self, attr).reset_index()
            columns = attr_df.columns
            merged_df = between_join(attr_df, self._core_lithology_deltas)
            merged_df["DEPTH"] += merged_df["DELTA"]
            setattr(self, "_" + attr, merged_df[columns].set_index("DEPTH").sort_index())

        # TODO: carfully update samples and reload core images if needed

        if self._has_file("core_lithology"):
            core_lithology = cross_join(self.core_lithology.reset_index(),
                                        self._core_lithology_deltas[["DEPTH_FROM", "DEPTH_TO", "DELTA"]],
                                        suffixes=("", "_deltas"))
            mask = ((core_lithology["DEPTH_FROM"] >= core_lithology["DEPTH_FROM_deltas"]) &
                    (core_lithology["DEPTH_TO"] <= core_lithology["DEPTH_TO_deltas"]))
            core_lithology = core_lithology[mask]
            core_lithology["DEPTH_FROM"] += core_lithology["DELTA"]
            core_lithology["DEPTH_TO"] += core_lithology["DELTA"]
            core_lithology = core_lithology[self.core_lithology.reset_index().columns]
            self._core_lithology = core_lithology.set_index(["DEPTH_FROM", "DEPTH_TO"]).sort_index()

        boring_intervals = pd.merge(self._boring_intervals.reset_index(),
                                    self._boring_intervals_deltas[["DEPTH_FROM", "DEPTH_TO", "DELTA"]],
                                    on=["DEPTH_FROM", "DEPTH_TO"])
        boring_intervals["DEPTH_FROM"] += boring_intervals["DELTA"]
        boring_intervals["DEPTH_TO"] += boring_intervals["DELTA"]
        boring_intervals = boring_intervals.drop("DELTA", axis=1)
        self._boring_intervals = boring_intervals.set_index(["DEPTH_FROM", "DEPTH_TO"]).sort_index()

    def _save_matching_report(self):
        """Save matching report in a well directory, specified in `self.path`.

        The report consists of two `.csv` files, containing depths of boring
        and lithology intervals respectively before and after matching.
        """
        boring_sequences = self.boring_sequences.reset_index()[["DEPTH_FROM", "DEPTH_TO", "MODE"]]
        not_none_mask = boring_sequences["MODE"].map(lambda x: x is not None)
        boring_sequences = boring_sequences[not_none_mask]
        if len(boring_sequences) == 0:
            return

        core_logs_list = []
        for _, (depth_from, depth_to, mode) in boring_sequences.iterrows():
            _, core_mnemonic, core_attr, _ = self._parse_matching_mode(mode)
            core_log_segment = getattr(self, core_attr)[core_mnemonic].dropna().loc[depth_from:depth_to]
            core_log_segment.index /= 100
            core_logs_list.append(core_log_segment.to_frame(name=mode))
        core_logs = pd.concat(core_logs_list)
        core_logs.index.rename("Matched depth of a core plug", inplace=True)
        core_logs.reset_index(inplace=True)

        boring_intervals = self._boring_intervals_deltas.reset_index()
        boring_intervals["DEPTH_FROM_DELTA"] = boring_intervals["DEPTH_FROM"] + boring_intervals["DELTA"]
        boring_intervals["DEPTH_TO_DELTA"] = boring_intervals["DEPTH_TO"] + boring_intervals["DELTA"]
        boring_intervals = boring_intervals[["DEPTH_FROM", "DEPTH_TO", "DEPTH_FROM_DELTA", "DEPTH_TO_DELTA"]]
        boring_intervals.columns = [
            "Top of a boring interval",
            "Bottom of a boring interval",
            "Matched top of a boring interval",
            "Matched bottom of a boring interval"
        ]
        boring_intervals /= 100
        boring_intervals = between_join(core_logs, boring_intervals, left_on="Matched depth of a core plug",
                                        right_on=("Matched top of a boring interval",
                                                  "Matched bottom of a boring interval"))
        boring_intervals.to_csv(os.path.join(self.path, self.name + "_boring_intervals_matching.csv"),
                                index=False, float_format="%.2f")

        lithology_intervals = self._core_lithology_deltas.reset_index()
        lithology_intervals["DEPTH_FROM_DELTA"] = lithology_intervals["DEPTH_FROM"] + lithology_intervals["DELTA"]
        lithology_intervals["DEPTH_TO_DELTA"] = lithology_intervals["DEPTH_TO"] + lithology_intervals["DELTA"]
        lithology_intervals = lithology_intervals[["DEPTH_FROM", "DEPTH_TO", "DEPTH_FROM_DELTA", "DEPTH_TO_DELTA"]]
        lithology_intervals.columns = [
            "Top of a lithology interval",
            "Bottom of a lithology interval",
            "Matched top of a lithology interval",
            "Matched bottom of a lithology interval"
        ]
        lithology_intervals /= 100
        lithology_intervals.to_csv(os.path.join(self.path, self.name + "_lithology_intervals_matching.csv"),
                                   index=False, float_format="%.2f")

    @staticmethod
    def _parse_matching_mode(mode):
        """Split matching mode string into well log mnemonic, core log or
        property mnemonic, class attribute to get core data from and a sign of
        the theoretical correlation between well and core data."""
        mode = mode.replace(" ", "")
        split_mode = mode.split("~")
        if len(split_mode) != 2:
            raise ValueError("Incorrect mode format")
        log_mnemonic = split_mode[0]
        split_core_mode = split_mode[1].split(".")
        if len(split_core_mode) != 2:
            raise ValueError("Incorrect mode format")
        core_attr, core_mnemonic = split_core_mode

        sign = "+"
        if core_attr[0] in {"+", "-"}:
            sign = core_attr[0]
            core_attr = core_attr[1:]
        sign = 1 if sign == "+" else -1
        return log_mnemonic, core_mnemonic, core_attr, sign

    def _select_matching_mode(self, segment, mode_list, min_points, min_points_per_meter):
        """Select appropriate matching mode based on data, availible for given
        segment."""
        segment_depth_from = segment["DEPTH_FROM"].min()
        segment_depth_to = segment["DEPTH_TO"].max()
        core_len = segment["CORE_RECOVERY"].sum()
        min_points_per_cm = min_points_per_meter / 100
        for mode in mode_list:
            log_mnemonic, core_mnemonic, core_attr, _ = self._parse_matching_mode(mode)
            if log_mnemonic in self.logs and self._has_file(core_attr) and core_mnemonic in getattr(self, core_attr):
                well_log = self.logs[log_mnemonic].dropna()
                core_log = getattr(self, core_attr)[core_mnemonic].dropna()
                well_log_len = len(well_log.loc[segment_depth_from:segment_depth_to])
                core_log_len = len(core_log.loc[segment_depth_from:segment_depth_to])
                if min(well_log_len, core_log_len) >= max(min_points_per_cm * core_len, min_points):
                    return mode
        return None

    @classmethod
    def _unify_matching_mode(cls, mode):
        """Delete all spaces and add `+` sign if omitted to a matching mode
        string."""
        log_mnemonic, core_mnemonic, core_attr, sign = cls._parse_matching_mode(mode)
        mode = "{}~{}{}.{}".format(log_mnemonic, "+" if sign == 1 else "-", core_attr, core_mnemonic)
        return mode

    @classmethod
    def _unify_matching_modes(cls, modes):
        """Delete all spaces and add `+` sign if omitted to each matching mode
        string in a `modes` list."""
        return [cls._unify_matching_mode(mode) for mode in to_list(modes)]

    @staticmethod
    def _blur_log(log, win_size):
        """Blur a log with a Gaussian filter of size `win_size`."""
        if win_size is None:
            return log
        old_index = log.index
        new_index = np.arange(old_index.min(), old_index.max() + 1)
        log = log.reindex(index=new_index).interpolate(method="index", limit_direction="both")
        std = win_size / 6  # three-sigma rule
        log = log.rolling(window=win_size, min_periods=1, win_type="gaussian", center=True).mean(std=std)
        log = log.loc[old_index]
        return log

    def match_core_logs(self, mode="GK ~ core_logs.GK", split_lithology_intervals=True, gaussian_win_size=None,
                        min_points=3, min_points_per_meter=1, min_gap="0.5m", max_shift="10m", delta_from="-8m",
                        delta_to="8m", delta_step="0.1m", max_iter=100, max_iter_time=0.25, save_report=False):
        """Perform core-to-log matching by shifting core samples in order to
        maximize correlation between well and core logs.

        Parameters
        ----------
        mode : str or list of str, optional
            Matching mode precedence from highest to lowest. The mode is
            independently selected for each boring sequence. Each mode has the
            following structure: <well_log> ~ <sign><core_attr>.<core_log>,
            where:
            - well_log - mnemonic of a well log to use
            - sign - a sign of the theoretical correlation between well and
              core data, defaults to `+` if not given
            - core_attr - an attribute of `self` to get core data from
            - core_log - mnemonic of a core log or property to use
            Defaults to gamma ray matching.
        split_lithology_intervals : bool, optional
            Specifies whether to independently shift lithology intervals
            inside a boring interval. Defaults to `True`.
        gaussian_win_size : int, optional
            A Gaussian filter size in samples to perform log blurring before
            matching. No blurring is performed by default.
        min_points : int, optional
            A minimum number of samples in logs to perform matching. Defaults
            to 3.
        min_points_per_meter : int, optional
            A minimum number of samples per meter in logs to perform matching.
            Defaults to 1.
        min_gap : float, optional
            A minimum gap between two boring intervals in meters to set the
            bottom depth of the upper interval equal to the top depth of the
            lower one due to inaccuracies in depth measurements. Defaults to
            0.5.
        max_shift : positive float, optional
            Maximum shift of a boring sequence in meters. Defaults to 10.
        delta_from : float, optional
            Start of the grid of initial shifts in meters. Defaults to -8.
        delta_to : float, optional
            End of the grid of initial shifts in meters. Defaults to 8.
        delta_step : float, optional
            Step of the grid of initial shifts in meters. Defaults to 0.1.
        max_iter : positive int, optional
            Maximum number of `SLSQP` iterations. Defaults to 50.
        max_iter_time, optional
            Maximum time for an optimization iteration in seconds. Defaults to
            0.25.
        save_report : bool, optional
            Specifies whether to save matching report in a well directory.
            Defaults to `False`.

        Returns
        -------
        self : type(self)
            Self with core-to-log matching performed. Changes all core-related
            depths inplace.
        """
        min_gap = parse_depth(min_gap, check_positive=True, var_name="min_gap")
        max_shift = parse_depth(max_shift, check_positive=True, var_name="max_shift")
        delta_from = parse_depth(delta_from, var_name="delta_from")
        delta_to = parse_depth(delta_to, var_name="delta_to")
        delta_step = parse_depth(delta_step, check_positive=True, var_name="max_shift")
        if delta_from > delta_to:
            raise ValueError("delta_to must be greater than delta_from")
        if max(np.abs(delta_from), np.abs(delta_to)) > max_shift:
            raise ValueError("delta_from and delta_to must not exceed max_shift in absolute value")

        # TODO: extra checks for min_points
        min_points = max(min_points, 2)

        mode_list = self._unify_matching_modes(mode)

        # If a gap between any two boring intervals is less than `min_gap`, treat it as inaccuracies in depth
        # measurements and set the bottom depth of the upper interval equal to the top depth of the lower one.
        boring_intervals = self.boring_intervals.reset_index()
        bi_depth_from = boring_intervals["DEPTH_FROM"]
        bi_depth_to = boring_intervals["DEPTH_TO"]
        boring_intervals["DEPTH_TO"] = np.where((bi_depth_from.shift(-1) - bi_depth_to) < min_gap,
                                                bi_depth_from.shift(-1), bi_depth_to).astype(bi_depth_to.dtype)
        self._boring_intervals = boring_intervals.set_index(["DEPTH_FROM", "DEPTH_TO"])
        self._calc_boring_sequences()

        split_lithology_intervals = split_lithology_intervals and self._has_file("core_lithology")
        lithology_df = self.core_lithology if split_lithology_intervals else self.boring_intervals
        lithology_intervals = lithology_df.reset_index()[["DEPTH_FROM", "DEPTH_TO"]]

        # `boring_sequences` is a list of DataFrames, containing contiguous boring intervals, extracted one after
        # another. They are considered together since they must be shifted by the same delta.

        # `boring_groups` is a list of DataFrames, containing a number of boring sequences, extracted close to each
        # other. They are considered together since they can possibly overlap during optimization.
        boring_groups = select_contigious_intervals(self.boring_intervals.reset_index(), max_gap=2*max_shift)

        matched_boring_sequences = []
        matched_lithology_intervals = []
        sequences_modes = []
        sequences_r2 = []

        for group in boring_groups:
            boring_sequences = select_contigious_intervals(group)
            sequences_shifts = []

            # Independently optimize R^2 for each boring sequence
            for sequence in boring_sequences:
                sequence_depth_from = sequence["DEPTH_FROM"].min()
                sequence_depth_to = sequence["DEPTH_TO"].max()

                mode = self._select_matching_mode(sequence, mode_list, min_points, min_points_per_meter)
                sequences_modes.append(mode)
                if mode is None:
                    # Don't shift a sequence if there's no data to perform matching
                    sequences_shifts.append([create_zero_shift(sequence_depth_from, sequence_depth_to)])
                    continue

                log_mnemonic, core_mnemonic, core_attr, sign = self._parse_matching_mode(mode)
                well_log = self.logs[log_mnemonic].dropna()
                well_log = well_log.loc[sequence_depth_from - max_shift : sequence_depth_to + max_shift]
                well_log = self._blur_log(well_log, gaussian_win_size)
                core_log = sign * getattr(self, core_attr)[core_mnemonic].dropna()
                core_log = core_log.loc[sequence_depth_from:sequence_depth_to]
                core_log = self._blur_log(core_log, gaussian_win_size)

                shifts = match_boring_sequence(sequence, lithology_intervals, well_log, core_log,
                                               max_shift, delta_from, delta_to, delta_step,
                                               max_iter, timeout=max_iter*max_iter_time)
                sequences_shifts.append(shifts)

            best_shifts = find_best_shifts(sequences_shifts, self.name, self.field)

            # Store shift deltas, mode and R^2
            for sequence, shift in zip(boring_sequences, best_shifts):
                mask = ((lithology_intervals["DEPTH_FROM"] >= sequence["DEPTH_FROM"].min()) &
                        (lithology_intervals["DEPTH_TO"] <= sequence["DEPTH_TO"].max()))
                sequence_lithology_intervals = lithology_intervals[mask].copy()  # to avoid SettingWithCopyWarning
                sequence_lithology_intervals["DELTA"] = shift.interval_deltas
                sequence["DELTA"] = shift.sequence_delta

                sequences_r2.append(shift.loss**2)
                matched_boring_sequences.append(sequence)
                matched_lithology_intervals.append(sequence_lithology_intervals)

        if all(mode is None for mode in sequences_modes):
            raise SkipWellException("None of the boring sequences can be matched")

        self._boring_intervals_deltas = pd.concat(matched_boring_sequences)
        self._core_lithology_deltas = pd.concat(matched_lithology_intervals)
        self._apply_matching()

        self._calc_boring_sequences()
        self._boring_sequences["MODE"] = sequences_modes
        self._boring_sequences["R2"] = sequences_r2

        if save_report:
            self._save_matching_report()
        return self

    @staticmethod
    def _calc_matching_r2(well_log, core_log, eps=1e-8):
        """Calculate squared correlation coefficient between well and core
        logs.

        If well and core logs are defined on different sets of depths, then at
        first well log values are estimated at core log depths by linear
        interpolation and then `R^2` is calculated for the resulting arrays.
        """
        well_log = well_log.dropna()
        interpolator = interp1d(well_log.index, well_log, kind="linear", fill_value="extrapolate")
        well_log = interpolator(core_log.index)
        cov = np.mean(well_log * core_log) - well_log.mean() * core_log.mean()
        cor = np.clip(cov / ((well_log.std() + eps) * (core_log.std() + eps)), -1, 1)
        return cor**2

    def plot_matching(self, mode=None, scale=False, interactive=True, subplot_height=700, subplot_width=200):
        """Plot well log and the corresponding core log for each boring
        sequence.

        This method can be used to illustrate the results of core-to-log
        matching. If called for a `Well` or `WellBatch` instance, an extra
        `aggregate` argument can be passed.

        Parameters
        ----------
        mode : str or list of str
            Specify type of well log and core log or property to plot. If
            `str`, the same mode will be used for all boring sequences. If
            `list` of `str`, than each boring sequence will have its own mode.
            In this case length of the `list` should match the number of
            boring sequences.

            Each mode has the same structure as `mode` in `match_core_logs`.
            If `None` and core-to-log matching was performed beforehand,
            chosen matching modes are used.
            Defaults to `None`.
        scale : bool
            Specifies whether to lineary scale core log values to well log
            values. Defaults to `False`.
        interactive : bool
            Specifies whether to draw a plot directly inside a Jupyter
            notebook. Defaults to `True`.
        subplot_height : positive int
            Height of each subplot with well and core logs. Defaults to 700.
        subplot_width : positive int
            Width of each subplot with well and core logs. Defaults to 200.
        aggregate : bool
            Specifies whether to plot all segments of the well on the same
            plot or create a separate plot for each segment. Creates one plot
            by default.

        Returns
        -------
        self : type(self)
            Self unchanged.
        """
        init_notebook_mode(connected=True)

        # Approximate size of the left margin of a plot, that will be added to the calculated
        # figsize in order to prevent figure shrinkage in case of small number of subplots
        margin = 120

        # Calculate matching R^2 if core-to-log matching was not performed
        boring_sequences = self.boring_sequences.reset_index()
        if mode is None and "MODE" not in boring_sequences.columns:
            raise ValueError("Core-to-log matching has to be performed beforehand if mode is not specified")
        if mode is not None:
            mode_list = self._unify_matching_modes(mode)
            if len(mode_list) == 1:
                mode_list = mode_list * len(boring_sequences)
            if len(mode_list) != len(boring_sequences):
                raise ValueError("Mode length must match the number of matching intervals")
            boring_sequences["MODE"] = mode_list
            r2_list = []
            for _, (depth_from, depth_to, _mode) in boring_sequences[["DEPTH_FROM", "DEPTH_TO", "MODE"]].iterrows():
                log_mnemonic, core_mnemonic, core_attr, sign = self._parse_matching_mode(_mode)
                well_log = self.logs[log_mnemonic].dropna()
                core_log_segment = sign * getattr(self, core_attr)[core_mnemonic].dropna().loc[depth_from:depth_to]
                r2_list.append(self._calc_matching_r2(well_log, core_log_segment))
            boring_sequences["R2"] = r2_list
        boring_sequences = boring_sequences[["DEPTH_FROM", "DEPTH_TO", "MODE", "R2"]]
        not_none_mask = boring_sequences["MODE"].map(lambda x: x is not None)
        boring_sequences = boring_sequences[not_none_mask]

        # Create a figure and traces for well and core logs
        depth_from_list = boring_sequences["DEPTH_FROM"]
        depth_to_list = boring_sequences["DEPTH_TO"]
        mode_list = boring_sequences["MODE"]
        r2_list = boring_sequences["R2"]

        n_cols = len(boring_sequences)
        subplot_titles = ["{}<br>R^2 = {:.3f}".format(mode, r2) for mode, r2 in zip(mode_list, r2_list)]
        fig = make_subplots(rows=1, cols=n_cols, subplot_titles=subplot_titles)

        for i, (depth_from, depth_to, _mode) in enumerate(zip(depth_from_list, depth_to_list, mode_list), 1):
            log_mnemonic, core_mnemonic, core_attr, sign = self._parse_matching_mode(_mode)
            well_log_segment = self.logs[log_mnemonic].dropna().loc[depth_from - 300 : depth_to + 300]
            core_log_segment = sign * getattr(self, core_attr)[core_mnemonic].dropna().loc[depth_from:depth_to]

            if scale and min(len(well_log_segment), len(core_log_segment)) > 1:
                log_interpolator = interp1d(well_log_segment.index, well_log_segment, kind="linear",
                                            fill_value="extrapolate")
                X = np.array(core_log_segment).reshape(-1, 1)
                y = log_interpolator(core_log_segment.index)
                reg = LinearRegression().fit(X, y)
                core_log_segment = pd.Series(reg.predict(X), index=core_log_segment.index)

            well_log_trace = go.Scatter(x=well_log_segment, y=well_log_segment.index / 100, showlegend=False,
                                        line=dict(color="rgb(255, 127, 14)"))
            drawing_mode = "markers" if core_attr == "core_properties" else None
            core_log_trace = go.Scatter(x=core_log_segment, y=core_log_segment.index / 100, showlegend=False,
                                        line=dict(color="rgb(31, 119, 180)"), mode=drawing_mode)
            fig.append_trace(well_log_trace, 1, i)
            fig.append_trace(core_log_trace, 1, i)

        # Update figure layout
        layout = fig.layout
        fig_layout = go.Layout(title="{} {}".format(self.field.capitalize(), self.name),
                               legend=dict(orientation="h"), width=n_cols*subplot_width + margin,
                               height=subplot_height)
        layout.update(fig_layout)

        for key in layout:
            if key.startswith("xaxis"):
                layout[key]["fixedrange"] = True

        for ann in layout["annotations"]:
            ann["font"]["size"] = 14

        # Reverse y-axis on all the subplots
        for ix in range(n_cols):
            axis_ix = str(ix + 1) if ix > 0 else ""
            axis_name = "yaxis" + axis_ix
            layout[axis_name]["autorange"] = "reversed"

        plot_fn = iplot if interactive else plot
        plot_fn(fig)
        return self

    def add_depth_log(self, dst='DEPTH'):
        """Copy the index of `self.logs` to its column.

        Parameters
        ----------
        dst : str
            Specify logs dst column to save depth data. Defaults to 'DEPTH'.

        Returns
        -------
        self : type(self)
            Self with created depth log.
        """
        self.logs[dst] = self.logs.index
        return self

    def drop_logs(self, mnemonics):
        """Drop well logs whose mnemonics are in `mnemonics`.

        Parameters
        ----------
        mnemonics : str or list of str
            Mnemonics of well logs to be dropped.

        Returns
        -------
        self : type(self)
            Self with filtered logs.
        """
        res = self.copy()
        res._logs.drop(to_list(mnemonics), axis=1, inplace=True)
        return res

    def keep_logs(self, mnemonics):
        """Drop well logs whose mnemonics are not in `mnemonics`.

        Parameters
        ----------
        mnemonics : str or list of str
            Mnemonics of well logs to be kept.

        Returns
        -------
        self : type(self)
            Self with filtered logs.
        """
        missing_mnemonics = np.setdiff1d(mnemonics, self.logs.columns)
        if len(missing_mnemonics) > 0:
            err_msg = "The following logs were not recorded for the well: {}".format(", ".join(missing_mnemonics))
            raise SkipWellException(err_msg)
        res = self.copy()
        res._logs = res.logs[to_list(mnemonics)]
        return res

    def rename_logs(self, rename_dict):
        """Rename well logs with corresponding mnemonics in `rename_dict`.

        Parameters
        ----------
        rename_dict : dict
            Dictionary containing `(old mnemonic : new mnemonic)` pairs.

        Returns
        -------
        self : type(self)
            Self with renamed logs. Changes `logs` inplace.
        """
        self.logs.rename(columns=rename_dict, inplace=True)
        return self

    def _filter_layers(self, layers, connected, invert_mask):
        """Keep or drop layers whose names match any pattern from `layers`
        depending on an `invert_mask` flag.

        Parameters
        ----------
        layers : str or list of str
            Regular expressions, specifying layer names to keep or drop.
        connected : bool
            Specifies whether to join segments with kept layers, that go one
            after another.
        invert_mask : bool
            If `False`, drop layers, whose names don't match any pattern from
            `layers`. If `True`, drop layers, whose names match any of them.

        Returns
        -------
        well : list of WellSegment
            Segments, representing kept layers.
        """
        if not self._has_file("layers"):
            raise SkipWellException("Layers are not defined for a well {}".format(self.name))
        reg_list = [re.compile(layer, re.IGNORECASE) for layer in to_list(layers)]
        mask = np.array([any(regex.fullmatch(layer) for regex in reg_list) for layer in self.layers["LAYER"]])
        if invert_mask:
            mask = ~mask
        # TODO: unify with _create_segments_by_fdtd
        depth_df = self.layers.reset_index()[mask]
        if connected:
            depth_df = self._core_chunks(depth_df)
        segments = [self[top:bottom] for _, (top, bottom) in depth_df[['DEPTH_FROM', 'DEPTH_TO']].iterrows()]
        return segments

    def drop_layers(self, layers, connected=True):
        """Drop layers, whose names match any pattern from `layers`.

        Parameters
        ----------
        layers : str or list of str
            Regular expressions, specifying layer names to drop.
        connected : bool, optional
            Specifies whether to join segments with kept layers, that go one
            after another. Defaults to `True`.

        Returns
        -------
        res : list of WellSegment or type(self)
            If called for a `WellSegment`, returns a list of segments,
            representing kept layers. Otherwise, return `self` with dropped
            layers.
        """
        return self._filter_layers(layers, connected, invert_mask=True)

    def keep_layers(self, layers, connected=True):
        """Drop layers, whose names don't match any pattern from `layers`.

        Parameters
        ----------
        layers : str or list of str
            Regular expressions, specifying layer names to keep.
        connected : bool, optional
            Specifies whether to join segments with kept layers, that go one
            after another. Defaults to `True`.

        Returns
        -------
        res : list of WellSegment or type(self)
            If called for a `WellSegment`, returns a list of segments,
            representing kept layers. Otherwise, return `self` with kept
            layers.
        """
        return self._filter_layers(layers, connected, invert_mask=False)

    def keep_matched_sequences(self, mode=None, threshold=0.6):
        """Keep boring sequences, matched using given `mode` with `R^2`
        greater than `threshold`.

        Parameters
        ----------
        mode : str or list of str
            Chosen matching mode to keep a sequence. It has the same structure
            as `mode` in `match_core_logs`.
        threshold : float
            Minimum value of `R^2` to keep a sequence.

        Returns
        -------
        res : list of WellSegment or type(self)
            If called for a `WellSegment`, returns a list of segments,
            representing kept boring sequences. Otherwise, return `self` with
            kept matched segments.
        """
        mask = self.boring_sequences["R2"] > threshold
        if mode is not None:
            mode_list = self._unify_matching_modes(mode)
            mask &= self.boring_sequences["MODE"].isin(mode_list)
        sequences = self.boring_sequences[mask].reset_index()[["DEPTH_FROM", "DEPTH_TO"]]
        res_segments = []
        for _, (depth_from, depth_to) in sequences.iterrows():
            res_segments.append(self[depth_from:depth_to])
        return res_segments

    def create_segments(self, src, connected=True):
        """Split `self` into segments with depth ranges, specified in
        attributes in `src`.

        Parameters
        ----------
        src : str or iterable
            Names of attributes to get depths ranges for splitting. If `src`
            consists of attributes in fdtd format then each row will represent
            a new segment. Otherwise, an exception will be raised.
        connected : bool, optional
            Join segments that follow one another. Defaults to `True`.

        Returns
        -------
        res : list of WellSegment or type(self)
            If called for a `WellSegment`, returns a list of split segments.
            Otherwise, return `self` with split segments.
        """
        if not isinstance(src, list):
            src = [src]
        if all([item in self.attrs_fdtd_index for item in src]):
            res = self._create_segments_by_fdtd(src, connected)
        else:
            # TODO: create_segments from depth_index
            raise ValueError(
                'All `src` must be in fdtd format:',
                [item for item in src if item not in self.attrs_fdtd_index]
            )
        return res

    def _create_segments_by_fdtd(self, src, connected):
        """Get segments from depth ranges, specified in fdtd attributes."""
        tables = [getattr(self, item).reset_index() for item in src]
        df = tables[0] if len(tables) == 1 else reduce(fdtd_join, tables)
        if connected:
            df = self._core_chunks(df)
        segments = [self[top:bottom] for _, (top, bottom) in df[['DEPTH_FROM', 'DEPTH_TO']].iterrows()]
        return segments

    @staticmethod
    def _core_chunks(df):
        if len(df) > 0:
            chunks = [(item.DEPTH_FROM.min(), item.DEPTH_TO.max()) for item in select_contigious_intervals(df)]
            chunks = pd.DataFrame(chunks, columns=["DEPTH_FROM", "DEPTH_TO"])
            return chunks
        return pd.DataFrame(columns=["DEPTH_FROM", "DEPTH_TO"])

    def random_crop(self, length, n_crops=1):
        """Create random crops from the segment. All cropped segments have the
        same length, their positions are sampled uniformly from the segment.

        Parameters
        ----------
        length : positive int or str
            Length of each crop in centimeters. If `str`, must be specified in
            a <value><units> format (e.g. "10m").
        n_crops : positive int, optional
            The number of crops from the segment. Defaults to 1.

        Returns
        -------
        segments : list of WellSegment
            Cropped segments.
        """
        length = parse_depth(length, check_positive=True, var_name="length")
        bounds = self.depth_from, max(self.depth_from, self.depth_to - length)
        crops_starts = np.floor(np.sort(np.random.uniform(*bounds, size=n_crops))).astype(int).tolist()
        crops = [self[start:start+length] for start in crops_starts]
        return crops

    def crop(self, length, step, drop_last=False, fill_value=0):
        """Create crops from the segment. All cropped segments have the same
        length and are cropped with some fixed step.

        Parameters
        ----------
        length : positive int or str
            Length of each crop in centimeters. If `str`, must be specified in
            a <value><units> format (e.g. "10m").
        step : positive int or str
            Step of cropping in centimeters.  If `str`, must be specified in a
            <value><units> format (e.g. "10m").
        drop_last : bool, optional
            If `True`, only crops that lie within the segment will be kept.
            If `False`, the first crop which comes out of segment bounds will
            also be kept to cover the whole segment with crops. Its `logs`
            will be padded by `fill_value` at the end to have given `length`.
            Defaults to `False`.
        fill_value : float, optional
            Value to fill padded part of `logs`. Defaults to 0.

        Returns
        -------
        segments : list of WellSegment
            Cropped segments.
        """
        length = parse_depth(length, check_positive=True, var_name="length")
        step = parse_depth(step, check_positive=True, var_name="step")
        n_crops, mod = divmod(self.length - length, step)
        n_crops += 1  # The number of crops strictly within the segment
        if not drop_last and mod and self._has_file("logs"):  # Pad logs by length
            # TODO: pad core images if loaded
            n_crops += 1  # Create an extra segment
            logs = self.logs  # Preload logs, since segment depths will be updated further
            self.actual_depth_to = self.depth_to
            self.depth_to += length
            index = logs.index[-1] + np.arange(self.logs_step, length, self.logs_step)
            data = np.full((len(index), len(logs.columns)), fill_value)
            pad_df = pd.DataFrame(data, index=index, columns=logs.columns)
            self._logs = pd.concat([logs, pad_df])
        crops_starts = self.depth_from + np.arange(n_crops) * step
        crops = [self[start:start+length] for start in crops_starts]
        return crops

    def create_mask(self, src, column, mapping=None, mode="logs", default=np.nan, dst="mask"):
        """Transform a column from some `WellSegment` attribute into a mask
        correponding to a well log or to a core image.

        Parameters
        ----------
        src : str
            Attribute to get the column from.
        column : str
            Name of the column to transform.
        mapping : dict or None
            Mapping for column values. If `None`, original attribure values
            will be kept in the mask. Defaults to `None`.
        mode : 'logs' or 'core'
            A mode, specifying the length of computed mask. If 'logs', mask
            lenght will match the length of well logs. If 'core', mask length
            will match the heigth of `core_dl` in pixels. Defaults to
            `'logs'`.
        default : float, optional
            Default value for mask if `src` doesn't contain information for
            corresponding depth. Defaults to `numpy.nan`.
        dst : str, optional
            `WellSegment` attribute to save the mask to. Defaults to `mask`.

        Returns
        -------
        self : type(self)
            Self with created mask.
        """
        if src in self.attrs_fdtd_index:
            self._create_mask_fdtd(src, column, mapping, mode, default, dst)
        elif src in self.attrs_depth_index:
            self._create_mask_depth_index(src, column, mapping, mode, default, dst)
        else:
            ValueError("Unknown src: ", src)
        return self

    def _create_empty_mask(self, mode, default):
        """Create a mask, filled with `default` value."""
        if mode == "core":
            mask = np.full(len(self.core_dl), default)
        elif mode == "logs":
            mask = np.full(len(self.logs), default)
        else:
            raise ValueError("Unknown mode: ", mode)
        return mask

    def _create_mask_fdtd(self, src, column, mapping, mode, default, dst):
        """Create mask from fdtd data."""
        mask = self._create_empty_mask(mode, default)

        table = getattr(self, src)
        factor = len(mask) / self.length
        for row in table.iterrows():
            depth_from, depth_to = row[1].name
            start = np.floor((max(depth_from, self.depth_from) - self.depth_from) * factor)
            end = np.ceil((min(depth_to, self.depth_to) - self.depth_from) * factor)
            mask[int(start):int(end)] = row[1][column] if mapping is None else mapping[row[1][column]]
        setattr(self, dst, mask)

    def _create_mask_depth_index(self, src, column, mapping, mode, default, dst):
        """Create mask from depth-indexed data."""
        # TODO: fix interpolation
        mask = self._create_empty_mask(mode, default)

        table = getattr(self, src)
        factor = len(mask) / self.length
        for row in table.iterrows():
            depth = row[1].name
            pos = np.floor((depth - self.depth_from) * factor)
            mask[int(pos)] = row[1][column] if mapping is None else mapping[row[1][column]]
        setattr(self, dst, mask)

    @process_columns
    def apply(self, df, fn, *args, axis=None, **kwargs):
        """Apply a function to each row of a segment attribute.

        Parameters
        ----------
        fn : callable
            A function to be applied. See Notes section for caveats.
        axis : {0, 1, None}, optional
            An axis, along which the function is applied:
            * 0: apply the function to each column
            * 1: apply the function to each row
            * `None`: apply the function to the whole `DataFrame`
            Defaults to `None`.
        args : misc
            Any additional positional arguments to pass to `fn` after data
            from `attr`.
        kwargs : misc
            Any additional keyword arguments to pass as keyword arguments to
            `fn`.

        Notes
        -----
        Currently, callables from `numpy` without pure Python implementation
        can't be passed as `fn` if they get more than one non-keyword
        argument and `axis` is specified.

        E.g. `apply(np.divide, 1000, axis=1, src='DEPTH')` will fail.
        Use `apply(lambda x: x / 1000, axis=1, src='DEPTH')` instead.

        Returns
        -------
        self : type(self)
            Self with applied function.
        """
        _ = self
        if axis is None:
            res = fn(df, *args, **kwargs)
        else:
            res = df.apply(fn, axis=axis, raw=True, result_type="expand", args=args, **kwargs)
        if isinstance(res, pd.Series):
            res = res.to_frame()
        return res

    def _filter_depth_attrs(self, attrs=None):
        """Return intersection of `attrs` and `self.attrs_depth_index`."""
        attrs = self.attrs_depth_index if attrs is None else attrs
        return np.intersect1d(attrs, self.attrs_depth_index)

    def reindex(self, step, interpolate=False, attrs=None):
        """Conform depth-indexed `attrs` of the segment to a new index,
        starting from `self.depth_from` to `self.depth_to` with a step `step`.
        If `interpolate` is `False`, `nan` values will be placed in locations
        having no value in the original index. Otherwise the data will be
        linearly interpolated.

        Parameters
        ----------
        step : positive int or str
            Distance between any two adjacent values in the new index in
            centimeters.  If `str`, must be specified in a <value><units>
            format (e.g. "10m").
        interpolate : bool, optional
            Specifies whether to interpolate attributes after reindexation.
            Defaults to `False`.
        attrs : str or list of str
            Depth-indexed attributes of the segment to be reindexed.

        Returns
        -------
        self : type(self)
            Self with reindexed `attrs`.
        """
        step = parse_depth(step, check_positive=True, var_name="step")
        new_index = np.arange(self.depth_from, self.depth_to, step)
        for attr in self._filter_depth_attrs(attrs):
            attr_val = getattr(self, attr)
            if interpolate:
                tmp_index = attr_val.index.union(new_index)
                tmp_index.name = attr_val.index.name
                res = attr_val.reindex(index=tmp_index)
                res = res.interpolate(method="index", limit_direction="both").loc[new_index]
            else:
                res = attr_val.reindex(index=new_index)
            setattr(self, "_" + attr, res)
            if attr == "logs":
                self.logs_step = step
        return self

    def interpolate(self, *args, attrs=None, **kwargs):
        """Interpolate `nan` values in `attrs`.

        Parameters
        ----------
        attrs : str or list of str
            Depth-indexed attributes of the segment to be interpolated.
        args : misc
            Any additional positional arguments to `pandas.interpolate`.
        kwargs : misc
            Any additional named arguments to `pandas.interpolate`.

        Returns
        -------
        self : type(self)
            Self with interpolated values in `attrs`.
        """
        for attr in self._filter_depth_attrs(attrs):
            res = getattr(self, attr).interpolate(method="index", *args, **kwargs)
            setattr(self, "_" + attr, res)
        return self

    def gaussian_blur(self, win_size, std=None, attrs=None):
        """Blur columns of `attrs` with a Gaussian filter.

        Parameters
        ----------
        win_size : int
            Size of the kernel.
        std : float, optional
            The standard deviation of the normal distribution. Equals
            `win_size / 6' by default.
        attrs : str or list of str
            Depth-indexed attributes of the segment to be blurred.

        Returns
        -------
        self : type(self)
            Self with blurred logs in `attrs`.
        """
        if std is None:
            std = win_size / 6  # three-sigma rule
        for attr in self._filter_depth_attrs(attrs):
            val = getattr(self, attr)
            nan_mask = val.isna()
            val = val.rolling(window=win_size, min_periods=1, win_type="gaussian", center=True).mean(std=std)
            val[nan_mask] = np.nan
            setattr(self, "_" + attr, val)
        return self

    def drop_nans(self, logs=None):
        """Split a well into contiguous segments, that do not contain `nan`
        values in logs, specified in `logs`.

        Parameters
        ----------
        logs : None or int or list of str
            - If `None`, create segments without `nan` values in all logs.
            - If `int`, create segments so that each row of logs has at least
              `logs` not-nan values.
            - If `list`, create segments without `nan` values in logs with
              mnemonics in `logs`.
            Defaults to `None`.

        Returns
        -------
        res : list of WellSegment or type(self)
            If called for a `WellSegment`, returns a list of segments without
            `nan` values. Otherwise, return `self` with dropped `nan` values.
        """
        logs = self.logs.columns if logs is None else logs
        if isinstance(logs, int):
            not_nan_mask = (~np.isnan(self.logs)).sum(axis=1) >= logs
        else:
            not_nan_mask = np.all(~np.isnan(self.logs[logs]), axis=1)

        if not not_nan_mask.any():
            return []

        borders = not_nan_mask[not_nan_mask ^ not_nan_mask.shift(1)].index.tolist()
        # If last mask element is True (segment ends with not nan value in logs) then the xor trick above misses
        # the last slicing border and therefore None should be added so the last slice could be done as `self[a:None]`
        if not_nan_mask.iloc[-1]:
            borders.append(None)
        borders = zip(borders[0::2], borders[1::2])
        return [self[a:b] for a, b in borders]

    @process_columns
    def norm_mean_std(self, df, mean=None, std=None, eps=1e-10):
        """Standardize well logs by subtracting the mean and scaling to unit
        variance.

        Parameters
        ----------
        mean : None or ndarray, optional
            Mean to be subtracted. If `None`, it is calculated independently
            for each log in the segment.
        std : None or ndarray, optional
            Standard deviation to be divided by. If `None`, it is calculated
            independently for each log in the segment.
        eps: float, optional
            A small float to be added to the denominator to avoid division by
            zero.

        Returns
        -------
        self : type(self)
            Self with standardized logs.
        """
        _ = self
        if mean is None:
            mean = df.mean()
        if std is None:
            std = df.std()
        return (df - mean) / (std + eps)

    @process_columns
    def norm_min_max(self, df, min=None, max=None, q_min=None, q_max=None, clip=True):  # pylint: disable=redefined-builtin
        """Linearly scale well logs to a [0, 1] range.

        Parameters
        ----------
        min : None or ndarray, optional
            Minimum values of the logs. If `None`, it is calculated
            independently for each log in the segment.
        max : None or ndarray or tuple or list of ndarrays
            Maximum values of the logs. If `None`, it is calculated
            independently for each log in the segment.
        q_min : float
            A quantile of logs to use as `min` if `min` is not given.
        q_max : float
            A quantile of logs to use as `max` if `max` is not given.
        clip : bool
            Specify, whether to clip scaled logs to a [0, 1] range.

        Returns
        -------
        self : type(self)
            Self with normalized logs.
        """
        _ = self

        if min is None and q_min is None:
            min = df.min()
        elif q_min is not None:
            min = df.quantile(q_min)

        if max is None and q_max is None:
            max = df.max()
        elif q_max is not None:
            max = df.quantile(q_max)

        df = (df - min) / (max - min)
        if clip:
            df.clip(0, 1, inplace=True)
        return df

    def equalize_histogram(self, src=None, dst=None, channels='last'):
        """Normalize core images by histogram equalization.

        Parameters
        ----------
        src : None, str or iterable
            Attributes to normalize. If `None`, `src` will be
            `('core_dl', 'core_uv')`. Defaults to `None`.
        dst : None, str or iterable
            Attributes to save normalized images to. If `None`, images will be
            saved into `src`. Defaults to `None`.
        channels : 'last' or 'first'
            Channels axis in images. Defaults to 'last'.

        Returns
        -------
        self : type(self)
            Self with normalized images.
        """
        src = to_list(src)
        for item in src:
            _ = getattr(self, item)
        src = ['_' + item if item in ['core_dl', 'core_uv'] else item for item in src]
        if dst is None:
            dst = src
        else:
            dst = to_list(dst)
        for _src, _dst in zip(src, dst):
            img = getattr(self, _src)
            if img.ndim == 3:
                img = cv2.cvtColor(img.astype('uint8'), cv2.COLOR_RGB2YCrCb)
                _slice = [slice(None)] * 3
                axis = -1 if channels == 'last' else 0
                _slice[axis] = 0
                img[_slice] = cv2.equalizeHist(img[_slice])
                img = cv2.cvtColor(img, cv2.COLOR_YCrCb2RGB)
            else:
                img = cv2.equalizeHist(img)
            setattr(self, _dst, img)
        return self

    def random_shift_logs(self, max_shift, src=None):
        """Shift `logs` attr columns by a step sampled from discrete uniform
        distribution in [-`max_period`, `max_period`]. Resulted empty positions
        are filled with first/last column value depending on shift direction.

        Parameters
        ----------
        max_shift : positive int or str
            Maximum possible shift in centimeters. If `str`, must be specified
            in a <value><units> format (e.g. "10m").

        Returns
        -------
        self : type(self)
            Self with randomly shifted `logs` columns.
        """
        df = self.logs
        src = df.columns if src is None else [src] if isinstance(src, str) else src
        max_shift = parse_depth(max_shift, check_positive=True, var_name="max_shift") // self.logs_step
        if max_shift == 0:
            warnings.warn('Passed `max_shift` is smaller then dataframe index step and therefore no shift was applied.')
        for column, series in df[src].iteritems():
            periods = np.random.randint(-max_shift, max_shift + 1)
            fill_value = series.iloc[0] if periods > 0 else series.iloc[-1]
            df[column] = series.shift(periods=periods, fill_value=fill_value)
        return self

    @process_columns(dst_from_result_=True)
    def one_hot_encode(self, df, encoder):
        """One-hot encode columns of WellSegment attribute.

        Parameters
        ----------
        encoder : callable
            Fitted sklearn.preprocessing.OneHotEncoder

        Returns
        -------
        self : type(self)
            Self with one-hot encoded logs columns.
        """
        _ = self
        columns = ['{}_{}'.format(col, cat) for col, categories in zip(df, encoder.categories_) for cat in categories]
        encoded = encoder.transform(df).toarray() if encoder.sparse else encoder.transform(df)
        return pd.DataFrame(index=df.index, data=encoded, columns=columns)
