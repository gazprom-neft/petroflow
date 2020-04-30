"""Implements abstract classes for WellSegment, Well and WellBatch."""
# pylint: disable=missing-docstring

from abc import ABCMeta, abstractmethod


class AbstractWellSegment(metaclass=ABCMeta):
    """Abstract class to check that all nesessary methods are implemented in
    `WellSegment` class."""
    @abstractmethod
    def __getitem__(self, key):
        pass

    @abstractmethod
    def load_core(self):
        pass

    @abstractmethod
    def dump(self):
        pass

    @abstractmethod
    def copy(self):
        pass

    @abstractmethod
    def deepcopy(self):
        pass

    @abstractmethod
    def validate_core(self):
        pass

    @abstractmethod
    def validate_samples(self):
        pass

    @abstractmethod
    def match_core_logs(self):
        pass

    @abstractmethod
    def plot(self):
        pass

    @abstractmethod
    def plot_matching(self):
        pass

    @abstractmethod
    def add_depth_log(self):
        pass

    @abstractmethod
    def drop_logs(self):
        pass

    @abstractmethod
    def keep_logs(self):
        pass

    @abstractmethod
    def rename_logs(self):
        pass

    @abstractmethod
    def create_mask(self):
        pass

    @abstractmethod
    def apply(self):
        pass

    @abstractmethod
    def reindex(self):
        pass

    @abstractmethod
    def interpolate(self):
        pass

    @abstractmethod
    def gaussian_blur(self):
        pass

    @abstractmethod
    def norm_mean_std(self):
        pass

    @abstractmethod
    def norm_min_max(self):
        pass

    @abstractmethod
    def equalize_histogram(self):
        pass

    @abstractmethod
    def random_shift_logs(self):
        pass

    @abstractmethod
    def one_hot_encode(self):
        pass


class AbstractWell(AbstractWellSegment):
    """Abstract class to check that all nesessary methods are implemented in
    `Well` and `WellBatch` classes."""
    @abstractmethod
    def drop_layers(self):
        pass

    @abstractmethod
    def keep_layers(self):
        pass

    @abstractmethod
    def keep_matched_sequences(self):
        pass

    @abstractmethod
    def create_segments(self):
        pass

    @abstractmethod
    def drop_short_segments(self):
        pass

    @abstractmethod
    def crop(self):
        pass

    @abstractmethod
    def random_crop(self):
        pass

    @abstractmethod
    def drop_nans(self):
        pass

    @abstractmethod
    def aggregate(self):
        pass
