[![License](https://img.shields.io/github/license/analysiscenter/batchflow.svg)](https://www.apache.org/licenses/LICENSE-2.0)
[![Python](https://img.shields.io/badge/python-3.6-blue.svg)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-1.15-orange.svg)](https://tensorflow.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.5-orange.svg)](https://pytorch.org)


# PetroFlow

`PetroFlow` is a library that allows processing well data (logs, core images, etc.) and conveniently train machine learning models.

Main features:
* load and process various types of well data:
    * well logs
    * core images in daylight (DL) and ultraviolet light (UV)
    * logs of the core
    * inclination
    * stratum layers
    * boring intervals
    * properties of core plugs
    * lithological description of core samples
* perform core-to-log matching
* train various deep learning models, e.g. to:
    * predict porosity by well logs
    * detect mismatched DL and UV core images
    * recover missing logs by the available ones


## About PetroFlow

> `PetroFlow` is based on [BatchFlow](https://github.com/analysiscenter/batchflow). You might benefit from reading [its documentation](https://analysiscenter.github.io/batchflow). However, it is not required, especially at the beginning.

In addition to the `BatchFlow` primitives, `PetroFlow` defines its own `Well` and `WellBatch` classes.

`Well` is a class, representing a well. It implements various methods for well data processing, such as filtering, visualization, normalization and cropping. In order to create a `Well`, you have to convert well data into a special [format](https://github.com/gazprom-neft/petroflow/blob/master/well_format.md).

`WellBatch` is designed to processes several wells at once to conveniently build multi-staged workflows that can involve machine learning model training.

You can learn more about using `PetroFlow` from the [tutorials](https://github.com/gazprom-neft/petroflow/tree/master/tutorials).


## Basic usage

Here is an example of a pipeline that loads well data, makes preprocessing and trains a model for porosity prediction for 1000 epochs:

```python
train_pipeline = (
  bf.Pipeline()
    .add_namespace(np)
    .init_variable("loss", default=[])
    .init_model("dynamic", "UNet", config=model_config)
    .keep_logs(LOG_MNEMONICS)
    .drop_nans()
    .drop_short_segments(CROP_LENGTH_CM)
    .add_depth_log()
    .reindex(step=LOGS_REINDEXATION_STEP, interpolate=True, attrs="logs")
    .reindex(step=PROPS_REINDEXATION_STEP, attrs="core_properties")
    .norm_mean_std(logs_mean, logs_std)
    .random_crop(CROP_LENGTH_CM, N_CROPS)
    .update(B("logs"), WS("logs").ravel())
    .stack(B("logs"), save_to=B("logs"))
    .swapaxes(B("logs"), 1, 2, save_to=B("logs"))
    .array(B("logs"), dtype=np.float32, save_to=B("logs"))
    .update(B("mask"), WS("core_properties")["POROSITY"].ravel())
    .stack(B("mask"), save_to=B("mask"))
    .expand_dims(B("mask"), 1, save_to=B("mask"))
    .divide(B("mask"), 100, save_to=B("mask"))
    .array(B("mask"), dtype=np.float32, save_to=B("mask"))
    .train_model("UNet", B("logs"), B("mask"), fetches="loss", save_to=V("loss", mode="a"))
    .run(batch_size=4, n_epochs=1000, shuffle=True, drop_last=True, bar=True, lazy=True)
)
```


## Installation

> `PetroFlow` module is in the beta stage. Your suggestions and improvements are very welcome.

> `PetroFlow` supports python 3.6 or higher.


### Installation as a python package

With [pipenv](https://docs.pipenv.org/):

    pipenv install git+https://github.com/gazprom-neft/petroflow.git#egg=petroflow

With [pip](https://pip.pypa.io/en/stable/):

    pip3 install git+https://github.com/gazprom-neft/petroflow.git

After that just import `PetroFlow`:
```python
import petroflow
```


### Installation as a project repository

When cloning repo from GitHub use flag `--recursive` to make sure that `BatchFlow` submodule is also cloned.

    git clone --recursive https://github.com/gazprom-neft/petroflow.git


## Citing PetroFlow

Please cite `PetroFlow` in your publications if it helps your research.

```
Khudorozhkov R., Kuvaev A., Kozhevin A., Goryachev S. PetroFlow library for data science research of well data. 2019.
```

```
@misc{
  author       = {R. Khudorozhkov and A. Kuvaev and A. Kozhevin and S. Goryachev},
  title        = {PetroFlow library for data science research of well data},
  year         = 2019
}
```
