# PetroFlow well format

A petrophysicist works with a huge amount of different well data stored in various formats from `.las` to `.xls`. Usually, the structure and headers of these files is changing over time, which makes automatic processing of such data difficult and inconvenient. That's why `PetroFlow` introduces its own well data format, described below.

## Format description

Data for each well must be stored in a separate directory, containing at least a single mandatory `meta.json` file with the following structure:
```python
{
    "name": "well_name",
    "field": "well_field",
    "depth_from": 100,  # wellhead depth in centimeters
    "depth_to": 300000  # bottom hole depth in centimeters
}
```

The well directory can optionally contain any of the files listed in the table below: they can have `.csv`, `.las` or `.feather` extension and will be automatically loaded into the corresponding `Well` or `WellSegment` attribute as a `pandas.DataFrame` at the time of the first access.

Depth and length values are assumed to be stored in centimeters for all formats, except for `.las`, where depths units are assumed to be meters and are automatically converted to centimeters during file loading. Log units are not parsed from a `.las` file header since they are optional and their format is not strictly fixed.

The well dir can also contain `samples_dl` and `samples_uv` subdirectories, containing daylight and ultraviolet images of core samples respectively.

Optional contents of a well dir are summarized in the following table:

| Name | Type | Index | Description |
| --- | --- | --- |--- |
| logs | file | `DEPTH` | Well logs. Depth log must have `DEPTH` mnemonic. Mnemonics of the same log type in `logs` and `core_logs` should match. |
| layers | file | `DEPTH_FROM`, `DEPTH_TO` | Stratum layers' names in a `LAYER` column. |
| boring_intervals | file | `DEPTH_FROM`, `DEPTH_TO` | Depths of boring intervals with core recovery in centimeters, stored in `CORE_RECOVERY` column. |
| boring_sequences | file | `DEPTH_FROM`, `DEPTH_TO` | Depth ranges of contiguous boring intervals, extracted one after another. If the file exists, then `boring_sequences` is loaded from it. Otherwise, it is calculated from `boring_intervals`. If core-to-log matching is performed, then extra `MODE` and `R2` columns with matching parameters and results are created. |
| core_properties | file | `DEPTH` | Physical properties of extracted core plugs. |
| core_lithology | file | `DEPTH_FROM`, `DEPTH_TO` | Lithological description of core samples. |
| core_logs | file | `DEPTH` | Logs of the core. Depth log must have `DEPTH` mnemonic. Mnemonics of the same log type in `logs` and `core_logs` should match. |
| samples | file | `DEPTH_FROM`, `DEPTH_TO` | Names of core sample images in a `SAMPLE` column. |
| samples_dl | dir | No | A directory, containing daylight images of core samples. Loaded into the `core_dl` attribute of `Well` and `WellSegment`, requires `samples` file to exist in the well dir. |
| samples_uv | dir | No | A directory, containing ultraviolet images of core samples. Loaded into the `core_uv` attribute of `Well` and `WellSegment`, requires `samples` file to exist in the well dir. |
