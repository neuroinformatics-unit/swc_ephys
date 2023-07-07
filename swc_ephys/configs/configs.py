import glob
import os
from pathlib import Path
from typing import Dict, Tuple

import yaml

from ..utils import utils


def get_configs(name: str) -> Tuple[Dict, Dict, Dict]:
    """
    Loads the config yaml file in the same folder
    (swc_ephys/configs) containing preprocessing (pp)
    and sorter options.

    Once loaded, the list containing preprocesser name
    and kwargs is cast to tuple. This keeps the type
    checker happy while not requiring a tuple
    in the .yaml which require ugly tags.

    Parameters
    ----------

    name: name of the configs to load. Should not include the
          .yaml suffix.

    Returns
    -------

    pp_steps : a dictionary containing the preprocessing
               step order (keys) and a [pp_name, kwargs]
               list containing the spikeinterface preprocessing
               step and keyword options.

    sorter_options : a dictionary with sorter name (key) and
                     a dictionary of kwargs to pass to the
                     spikeinterface sorter class.
    """
    config_dir = Path(os.path.dirname(os.path.realpath(__file__)))

    available_files = glob.glob((config_dir / "*.yaml").as_posix())
    available_files = [Path(path_).stem for path_ in available_files]

    if name not in available_files:  # then assume it is a full path
        assert Path(name).is_file(), (
            f"{name} is neither the name of an existing "
            f"config or valid path to configuration file."
        )

        assert Path(name).suffix in [
            ".yaml",
            ".yml",
        ], f"{name} is not the path to a .yaml file"

        config_filepath = Path(name)

    else:
        config_filepath = config_dir / f"{name}.yaml"

    with open(config_filepath) as file:
        config = yaml.full_load(file)

    pp_steps = config["preprocessing"]
    sorter_options = config["sorting"]
    waveform_options = config["waveforms"]

    utils.cast_pp_steps_values(pp_steps, "tuple")

    return pp_steps, sorter_options, waveform_options
