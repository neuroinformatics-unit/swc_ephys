import json
from typing import Dict, List, Optional, Tuple

import numpy as np
import spikeinterface.preprocessing as spre

from ..configs import configs
from ..data_classes.preprocessing import PreprocessingData
from ..utils import utils


def preprocess(
    preprocess_data: PreprocessingData,
    pp_steps: Optional[Dict] = None,
    verbose: bool = True,
) -> PreprocessingData:
    """
    Returns an updated PreprocessingData dictionary of SpikeInterface
    recording objects setup in the order and with the options specified
    in pp_steps.

    Spikeinterface preprocessing is lazy - no preprocessing is done
    until the data is written to file or the get_traces() method on
    the recording object is called.

    Parameters
    ----------
    preprocess_data : PreprocessingData
        A preprocessing data object that has as attributes the
        paths to rawdata. The pp_steps attribute is set on
        this class during execution of this function.

    pp_steps: pp_steps dictionary, see configs/configs.py for details.

    Returns
    -------
    preprocess_data : PreprocessingData
        Preprocessing data class with pp_steps updated and
        dictionary field filled with key-value pairs indicating
        the name and order of preprocessing keys, and value containing
        associated SpikeInterface recording objects.

    """
    if not pp_steps:
        # TODO: should this ever be done? Might be
        # very confusing if user forgets to pass pp_steps
        pp_steps, _, _ = configs.get_configs("test")

    pp_funcs = get_pp_funcs()

    checked_pp_steps, pp_step_names = check_and_sort_pp_steps(pp_steps, pp_funcs)

    preprocess_data.set_pp_steps(pp_steps)

    for step_num, pp_info in checked_pp_steps.items():
        perform_preprocessing_step(
            step_num, pp_info, preprocess_data, pp_step_names, pp_funcs, verbose
        )

    return preprocess_data


def check_and_sort_pp_steps(pp_steps: Dict, pp_funcs: Dict) -> Tuple[Dict, List[str]]:
    """
    Sort the preprocessing steps dictionary by order to be run (based on the
    keys) and check the dictionary is valid.

    Parameters
    ----------
    pp_steps : Dict
        A dictionary with keys as numbers indicating the order that
        preprocessing steps are run (starting at "1"). The values are a
        (preprocessing name, preprocessing kwargs) tuple containing the
        spikeinterface preprocessing function name, and kwargs to pass to it.

    pp_funcs : Dict
        A dictionary linking preprocessing step names to the underlying
        SpikeInterface function objects that conduct the preprocessing.

    Returns
    -------
    sorted_pp_steps :Dict
        A sorted and checked preprocessing steps dictionary.

    pp_step_names : List
        Preprocessing step names (e.g. "bandpass_filter") in order
        that they are to be run.

    Notes
    -------
    This will soon be deprecated and replaced by validation
    of the config file itself on load.
    """
    sorted_pp_steps = {k: pp_steps[k] for k in sorted(pp_steps.keys())}
    pp_step_names = [item[0] for item in sorted_pp_steps.values()]

    # Check keys are numbers starting at 1 increasing by 1
    assert all(
        key.isdigit() for key in sorted_pp_steps.keys()
    ), "pp_steps keys must be integers"

    key_nums = [int(key) for key in sorted_pp_steps.keys()]

    assert np.min(key_nums) == 1, "dict keys must start at 1"

    diffs = np.diff(key_nums)
    assert np.unique(diffs).size == 1, "all dict keys must increase in steps of 1"
    assert diffs[0] == 1, "all dict keys must increase in steps of 1"

    # Check the preprocessing function names are valid and print steps used
    canonical_step_names = list(pp_funcs.keys())

    for user_passed_name in pp_step_names:
        assert (
            user_passed_name in canonical_step_names
        ), f"{user_passed_name} not in allowed names: ({canonical_step_names}"

    utils.message_user(
        f"\nThe preprocessing options dictionary is "
        f"{json.dumps(sorted_pp_steps, indent=4, sort_keys=True)}"
    )

    return sorted_pp_steps, pp_step_names


def perform_preprocessing_step(
    step_num: str,
    pp_info: Tuple[str, Dict],
    preprocess_data: PreprocessingData,
    pp_step_names: List[str],
    pp_funcs: Dict,
    verbose: bool = True,
) -> None:
    """
    Given the preprocessing step and preprocess_data UserDict containing
    spikeinterface BaseRecordings, apply a preprocessing step to the
    last preprocessed recording and save the recording object to PreprocessingData.
    For example, if step_num = "3", get the recording of the second
    preprocessing step from preprocess_data and apply the 3rd preprocessing step
    as specified in pp_info.

    Parameters
    ----------
    step_num : str
        Preprocessing step to run (e.g. "1", corresponds to the key in pp_dict).

    pp_info : Tuple[str, Dict]
        Preprocessing name, preprocessing kwargs) tuple (they value from
        the pp_dict).

    preprocess_data : PreprocessingData
        swc_ephys PreprocessingData class (a UserDict in which key-values are
        the preprocessing chain name : spikeinterface recording objects).

    pp_step_names : List[str]
        Ordered list of preprocessing step names that are being
        applied across the entire preprocessing chain.

    pp_funcs : Dict
        The cannonical SpikeInterface preprocessing functions. The key
        are the function name and value the function object.

    verbose : bool
        If True, messages will be printed to consolve updating on the
        progress of preprocessing / sorting.
    """
    pp_name, pp_options = pp_info

    utils.message_user(
        f"Running preprocessing step: {pp_name} with options {pp_options}", verbose
    )

    last_pp_step_output, __ = utils.get_dict_value_from_step_num(
        preprocess_data, step_num=str(int(step_num) - 1)
    )

    new_name = f"{step_num}-" + "-".join(["raw"] + pp_step_names[: int(step_num)])

    assert pp_funcs[pp_name].__name__ == pp_name, "something is wrong in func dict"

    preprocess_data[new_name] = pp_funcs[pp_name](last_pp_step_output, **pp_options)


def get_pp_funcs() -> Dict:
    """
    Get the spikeinterface preprocessing function
    from its name.

    TODO
    -----
    It should be possible to generate this on the fly from
    SI __init__ rather than hard code like this
    """
    pp_funcs = {
        "phase_shift": spre.phase_shift,
        "bandpass_filter": spre.bandpass_filter,
        "common_reference": spre.common_reference,
    }

    return pp_funcs
