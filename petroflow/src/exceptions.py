"""Contains package-specific exceptions."""


class SkipWellException(Exception):
    """Raised if a well should be dropped from a batch."""


class DataRegularityError(SkipWellException):
    """Raised if any data regularity checks are not passed."""

    error_templates = {
        # general errors
        "non_int_index": "Index must have int type:\n\n{}",
        "non_unique_index": "Index is not unique:\n\n{}",
        "non_increasing_index": "Index is not monotonically increasing:\n\n{}",
        "disordered_index": "DEPTH_FROM is greater than DEPTH_TO:\n\n{}",
        "overlapping_index": "Overlaping intervals found in index:\n\n{}",

        # boring_intervals errors
        "nan_recovery": "Missing `CORE_RECOVERY` values in boring_intervals:\n\n{}",
        "non_positive_recovery": "`CORE_RECOVERY` values must be positive:\n\n{}",
        "wrong_recovery": "`CORE_RECOVERY` is greater than the length of the corresponding interval:\n\n{}",

        # lithology_intervals errors
        "lithology_ranges": ("The following lithology intervals are not included in any of the boring "
                             "intervals:\n\n{}"),
        "lithology_length": ("Total length of all lithology intervals of a boring interval does not match "
                             "its core recovery:\n\n{}"),

        # samples errors
        "duplicated_files": "Duplicated file names in {}",
        "missing_samples_dirs": "Both samples_dl and samples_uv dirs are missing",
    }

    def __init__(self, error_type, *args):
        message = self.error_templates.get(error_type, error_type)
        message = message.format(*args)
        super().__init__(message)
