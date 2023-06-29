from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Dict, List, Literal, Tuple, Union

if TYPE_CHECKING:
    from spikeinterface.core import BaseRecording

    from ..data_classes.preprocessing import PreprocessingData
    from ..data_classes.sorting import SortingData

import copy
import json
import os.path
import subprocess
from pathlib import Path

import numpy as np
from spikeinterface import concatenate_recordings


def canonical_names(name: str) -> str:
    """
    Store the canonical names e.g. filenames, tags
    that are used throughout the project. This setup
    means filenames can be edited without requiring
    extensive code changes.

    Parameters
    ----------
    name : str
        short-hand name of the full name of interest.

    Returns
    -------
    filenames[name] : str
        The full name of interest e.g. filename.

    """
    filenames = {
        "preprocessed_yaml": "preprocess_data_attributes.yaml",
    }
    return filenames[name]


def get_keys_first_char(
    data: Union[PreprocessingData, SortingData], as_int: bool = False
) -> Union[List[str], List[int]]:
    """
    Get the first character of all keys in a dictionary. Expected
    that the first characters are integers (as str type).

    Parameters
    ----------
    data : Union[PreprocessingData, SortingData]
        swc_ephys PreprocessingData class holding filepath information.

    as_int : bool
        If True, the first character of the keys are cast to
        integer type.

    Returns
    -------
    list_of_numbers : Union[List[str], List[int]]
        A list of numbers of string or integer type, that are
        the first numbers of the Preprocessing / Sorting Data
        .data dictionary keys.
    """
    list_of_numbers = [int(key[0]) if as_int else key[0] for key in data.keys()]
    return list_of_numbers


def get_dict_value_from_step_num(
    data: Union[PreprocessingData, SortingData], step_num: str
) -> Tuple[BaseRecording, str]:
    """
    Get the value of the PreprocessingData dict from the preprocessing step number.

    PreprocessingData contain keys indicating the preprocessing steps,
    starting with the preprocessing step number.
    e.g. 0-raw, 1-raw-bandpass_filter, 2-raw_bandpass_filter-common_average

    Return the value of the dict (spikeinterface recording object)
    from the dict using only the step number.

    Parameters
    ----------
    data : Union[PreprocessingData, SortingData]
        swc_ephys PreprocessingData class holding filepath information.

    step_num : str
        The preprocessing step number to get the value (i.e. recording object)
        from.

    Returns
    -------
    dict_value : BaseRecording
        The SpikeInterface recording stored in the dict at the
        given preprocessing step number.

    pp_key : str
        The key of the preprocessing dict at the given
        step number.
    """
    if step_num == "last":
        pp_key_nums = get_keys_first_char(data, as_int=True)

        # Complete overkill as a check but this is critical.
        step_num = str(int(np.max(pp_key_nums)))
        assert (
            int(step_num) == len(data.keys()) - 1
        ), "the last key has been taken incorrectly"

    select_step_pp_key = [key for key in data.keys() if key[0] == step_num]

    assert len(select_step_pp_key) == 1, "pp_key must always have unique first char"

    pp_key: str = select_step_pp_key[0]
    dict_value = data[pp_key]

    return dict_value, pp_key


def message_user(message: str, verbose: bool = True) -> None:
    """
    Method to interact with user.

    Parameters
    ----------
    message : str
        Message to print.

    verbose : bool
        The mode of the application. If verbose is False,
        nothing is printed.
    """
    if verbose:
        print(message)


def concatenate_runs(recording: BaseRecording) -> BaseRecording:
    """
    Convenience function to concatenate the segments
    of a recording object.

    Parameters
    ----------
    recording : BaseRecording
        A spikeinterface recording object.

    Returns
    -------
    concatenated_recording : BaseRecording
        The SpikeInterface recording object with all
        segments concatenated into a single segments.
    """
    message_user(f"Concatenating {recording.get_num_segments()} into a single segment.")

    concatenated_recording = concatenate_recordings([recording])

    return concatenated_recording


def get_local_sorter_path(sorter: str) -> Path:
    """
    Return the path to a sorter singularity image. The sorters are
    stored by swc_ephys in the home folder.

    Parameters
    ----------
    sorter : str
        The name of the sorter to get the path to (e.g. kilosort2_5).

    Returns
    ---------
    local_path : Path
        The path to the sorter image on the local machine.
    """
    local_path = (
        Path.home() / ".swc_ephys" / "sorter_images" / get_sorter_image_name(sorter)
    )
    local_path.parent.mkdir(exist_ok=True, parents=True)
    return local_path


def get_hpc_sorter_path(sorter: str) -> Path:
    """
    Return the path to the sorter image on the SWC HCP (ceph).

    Parameters
    ----------
    sorter : str
        The name of the sorter to get the path to (e.g. kilosort2_5).

    Returns
    -------
    sorter_path : Path
        The base to the sorter image on SWC HCP (ceph).
    """
    base_path = Path("/ceph/neuroinformatics/neuroinformatics/scratch/sorter_images")
    sorter_path = base_path / sorter / get_sorter_image_name(sorter)
    return sorter_path


def get_sorter_image_name(sorter: str) -> str:
    """
    Get the sorter image name, as defined by how
    SpikeInterface names the docker images it provides.

    Parameters
    ----------
    sorter : str
        The name of the sorter to get the path to (e.g. kilosort2_5).

    Returns
    -------
    sorter_name : str
        The SpikeInterface filename of the docker image for that sorter.
    """
    if "kilosort" in sorter:
        sorter_name = f"{sorter}-compiled-base.sif"
    else:
        if sorter == "spykingcircus":
            sorter = "spyking-circus"
        sorter_name = f"{sorter}-base.sif"
    return sorter_name


def check_singularity_install() -> bool:
    """
    Check the system install of singularity.

    Returns
    -------
    is_installed : bool
        Indicates whether Singularity is installed on the machine.
    """
    try:
        subprocess.run("singularity --version", shell=True)
        is_installed = True
    except FileNotFoundError:
        is_installed = False

    return is_installed


def sort_list_of_paths_by_datetime_order(list_of_paths: List[Path]) -> List[Path]:
    """
    Given a list of paths to folders, sort the paths by the creation
    time of the folders they point to. Return the sorted
    list of paths.

    Parameters
    ----------
    list_of_paths : List[Path]
        A list of paths to sort into datetime order.


    Returns
    -------
    list_of_paths_by_creation_time : List[Path]
        A list containing `list_of_paths` ordered by the
        folder creation timestamp.
    """
    list_of_paths_by_creation_time = copy.deepcopy(list_of_paths)
    list_of_paths_by_creation_time.sort(key=os.path.getctime)

    return list_of_paths_by_creation_time


def list_of_files_are_in_datetime_order(
    list_of_paths: List[Path], creation_or_modification: str = "creation"
) -> bool:
    """
    Assert whether a list of paths are in order. By default, check they are
    in order by creation date. Can also check if they are ordered by
    modification date.

    Parameters
    ----------
    list_of_paths : List[Path]
        A list of paths to sort into datetime order.

    creation_or_modification : str
        If "creation", check the list of paths are ordered by creation datetime.
        Otherwise if "modification", check they are sorterd by modification datetime.

    Returns
    -------
    is_in_time_order : bool
        Indicates whether `list_of_paths` was in creation or modification time order.
        depending on the value of `creation_or_modification`.
    """
    assert creation_or_modification in [
        "creation",
        "modification",
    ], "creation_or_modification must be 'creation' or 'modification."

    filter: Callable
    filter = (
        os.path.getctime if creation_or_modification == "creation" else os.path.getmtime
    )

    list_of_paths_by_time = copy.deepcopy(list_of_paths)
    list_of_paths_by_time.sort(key=filter)

    is_in_time_order = list_of_paths == list_of_paths_by_time

    return is_in_time_order


def make_preprocessing_plot_title(
    run_name: str,
    full_key: str,
    shank_idx: int,
    recording_to_plot: BaseRecording,
    total_used_shanks: int,
) -> str:
    """
    For visualising data, make the plot titles (with headers in bold). If
    more than one shank is used, the title will also contain information
    on the displayed shank.

    Parameters
    ----------
    run_name : str
        The name of the preprocessing run (e.g. "1-phase_shift").

    full_key : str
        The full preprocessing key (as defined in preprocess.py).

    shank_idx : int
        The SpikeInterface group number representing the shank number.

    recording_to_plot : BaseRecording
        The SpikeInterface recording object that is being displayed.

    total_used_shanks : int
        The total number of shanks used in the recording. For a 4-shank probe,
        this could be between 1 - 4 if not all shanks are mapped.

    Returns
    -------
    plot_title : str
        The formatted plot title.
    """
    plot_title = (
        r"$\bf{Run \ name:}$" + f"{run_name}"
        "\n" + r"$\bf{Preprocessing \ step:}$" + f"{full_key}"
    )
    if total_used_shanks > 1:
        plot_title += (
            "\n"
            + r"$\bf{Shank \ group:}$"
            + f"{shank_idx}, "
            + r"$\bf{Num \ channels:}$"
            + f"{recording_to_plot.get_num_channels()}"
        )
    return plot_title


def cast_pp_steps_values(
    pp_steps: Dict, list_or_tuple: Literal["list", "tuple"]
) -> None:
    """
    The settings in the pp_steps dictionary that defines the options
    for preprocessing should be stored in Tuple as they are not to
    be edited. However, when dumping Tuple to .yaml, there are tags
    displayed on the .yaml file which are very ugly.

    These are not shown when storing list, so this function serves
    to convert Tuple and List values in the preprocessing dict when
    loading / saving the preprocessing dict to .yaml files. This
    function converts `pp_steps` in place.

    Parameters
    ----------
    pp_steps : Dict
        The dictionary indicating the preprocessing steps to perform.

    list_or_tuple : Literal["list", "tuple"]
        The direction to convert (i.e. if "tuple", will convert to Tuple).
    """
    assert list_or_tuple in ["list", "tuple"], "Must cast to `list` or `tuple`."
    func = tuple if list_or_tuple == "tuple" else list

    for key in pp_steps.keys():
        pp_steps[key] = func(pp_steps[key])


# Misc. --------------------------------------------------------------------------------


def get_probe_num_groups(data: Union[PreprocessingData, SortingData]) -> int:
    """
    Get the number of probe groups on a recording-objects `probe` attribute.
    This is typically the number of shanks on the probe, which are treated
    separately by SI.

    By default, uses the first step on the recording, as probe metadata will not
    change throughout preprocessing.

    Parameters
    ----------
    data : Union[PreprocessingData, SortingData]
        The data object on which the recordings are stored.

    Returns
    -------
    num_groups : int
        The number of groups on the probe associated with the data recordings.
    """
    num_groups = np.unique(data[data.init_data_key].get_property("group")).size
    return num_groups


# TODO: unit test this properly.
# TODO: fix typing (why str not Union[path, str])
def update_metadata_base_paths(
    project_folder: Union[Path, str], old_base_path: str, new_base_path: str
) -> None:
    """
    Method to change the base_path in spikeinterface
    metadata files. SpikeInterface saves  paths to raw data
    in .json metadata files. This can cause problems if the data is accessed
    from a different location (e.g. mounted drive). This function
    cycles through all known SI metadata filenames and replaces
    the base_path on the folder path.

    TODO: Note: this function is experimental! Not all spikeinterface
    metadata formats have been investigated. This is a workaround
    and it should be possible to do this directly in SI,
    pending issue https://github.com/SpikeInterface/spikeinterface/issues/1752

    project_folder : str
        Folder to replace metadata files in.

    param old_base_path : str
        Old part of the path that is replaced 1:1 with `new_base_path`.
        This is achieved with a simple str replace().

    param new_base_path : str
        New base path, which 1:1 replaces `old_base_path` in the metadata folder path.
    """
    project_folder = Path(project_folder)

    all_metadata_paths = []
    for filename in ["spikeinterface_recording.json", "sorting.json", "recording.json"]:
        paths = project_folder.glob("**/" + filename)
        all_metadata_paths.extend(list(paths))

    for metadata_path in all_metadata_paths:
        with open(metadata_path, "r") as file:
            dict_ = json.load(file)

        assert (
            "module" in dict_ and dict_["module"] == "spikeinterface"
        ), f"{metadata_path} is not a spikeinterface metadata file - rename this file."

        if metadata_path.name in ["spikeinterface_recording.json", "recording.json"]:
            for recording_dict in dict_["kwargs"]["recording_list"]:
                check_and_set_spikeinterface_folder_path_metdata(
                    recording_dict, old_base_path, new_base_path
                )
        else:
            check_and_set_spikeinterface_folder_path_metdata(
                dict_, old_base_path, new_base_path
            )

        with open(metadata_path, "w") as file:
            json.dump(dict_, file)


def check_and_set_spikeinterface_folder_path_metdata(
    dict_: Dict, old_base_path: str, new_base_path: str
):
    """
    Perform checks on and replace SpikeInterface metadata with
    new_base_path. See update_metadata_base_paths().
    """
    folder_path = dict_["kwargs"]["folder_path"]

    assert old_base_path in folder_path, (
        f"The base path {old_base_path} was not found in the metadata folder path "
        f"{folder_path}"
    )

    new_folder_path = dict_["kwargs"]["folder_path"].replace(
        old_base_path, new_base_path
    )

    assert Path(new_folder_path).is_dir(), (
        f"The new folder path does not exist. The new folder path was created by "
        f"replacing {old_base_path} with {new_base_path} in the metadata folder path {folder_path}"
    )

    dict_["kwargs"]["folder_path"] = new_folder_path
