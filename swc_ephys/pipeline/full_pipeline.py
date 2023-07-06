from __future__ import annotations

import copy
from pathlib import Path
from typing import TYPE_CHECKING, List, Literal, Union

if TYPE_CHECKING:
    from ..data_classes.preprocessing import PreprocessingData

from ..configs.configs import get_configs
from ..utils import slurm, utils
from .load_data import load_spikeglx_data
from .postprocess import run_postprocess
from .preprocess import preprocess
from .sort import run_sorting


def run_full_pipeline(
    base_path: Union[Path, str],
    sub_name: str,
    run_names: Union[List[str], str],
    config_name: str = "test",
    sorter: str = "kilosort2_5",
    existing_preprocessed_data: Literal[
        "overwrite", "load_if_exists", "fail_if_exists"
    ] = "fail_if_exists",
    overwrite_existing_sorter_output: bool = False,
    verbose: bool = True,
    slurm_batch: bool = False,
) -> None:
    """
    Run preprocessing, sorting and post-processing on SpikeGLX data.
    see README.md for detailed information on use.

    This function must be run in main as uses multiprocessing e.g.
    if __name__ == "__main__":
        run_full_pipeline(args...)

    Parameters
    ----------
    base_path : Union[Path, str]
        Path to the rawdata folder containing subjects folders.

    sub_name : str
        Subject to preprocess. The subject top level dir should reside in
        base_path/rawdata/ .

    run_names : Union[List[str], str],
        The SpikeGLX run name (i.e. not including the gate index). This can
        also be a list of run names. Preprocessing will still occur per-run.
        Runs are concatenated in the order passed prior to sorting.

    config_name : str
        The name of the configuration to use. Note this must be the name
        of a .yaml file (not including the extension) stored in
        swc_ephys/configs.

    sorter : str
        name of the sorter to use e.g. "kilosort2_5".

    existing_preprocessed_data : Literal[
                                 "overwrite", "load_if_exists", "fail_if_exists"]
        Determines how existing preprocessed data (e.g. from a prior pipeline run)
        is treated.
            "overwrite" : will overwrite any existing preprocessed data output.
            "load_if_exists" : will search for existing data and load if it exists.
                               Otherwise, will use the preprocessing from the
                               current run.
            "fail_if_exists" : If existing preprocessed data is found, an error
                               will be raised.

     overwrite_existing_sorter_output : bool
         If `False` an error will be raised if sorting output already
         exists. If True, existing sorting output will be overwritten.

    verbose : bool
        If True, messages will be printed to console updating on the
        progress of preprocessing / sorting.

    slurm_batch : bool
        If True, the pipeline will be run in a SLURM job. Set False
        if running on an interactive job, or locally.
    """
    if slurm_batch:
        local_args = copy.deepcopy(locals())
        slurm.run_full_pipeline_slurm(**local_args)
        return
    assert slurm_batch is False, "SLURM run has slurm_batch set True"

    pp_steps, sorter_options, waveform_options = get_configs(config_name)

    preprocess_data = load_spikeglx_data(base_path, sub_name, run_names)

    preprocess_data = preprocess(preprocess_data, pp_steps, verbose)

    save_preprocessed_data_if_required(preprocess_data, existing_preprocessed_data)

    sorting_data = run_sorting(
        preprocess_data.preprocessed_data_path,
        sorter,
        sorter_options,
        overwrite_existing_sorter_output,
        verbose,
    )
    # TODO: update this doc.
    # Save spikeinterface 'waveforms' output (TODO: currently, this is large)
    # to the sorter output dir. Quality checks are run and .csv of checks
    # output in the sorter folder as quality_metrics.csv
    run_postprocess(
        sorting_data.preprocessed_data_path,
        sorter,
        verbose,
        waveform_options=waveform_options,
    )


def save_preprocessed_data_if_required(
    preprocess_data: PreprocessingData,
    existing_preprocessed_data: Literal[
        "overwrite", "load_if_exists", "fail_if_exists"
    ],
) -> None:
    """
    Handle the loading of existing preprocessed data.
    See `run_full_pipeline()` for details.
    """
    preprocess_path = preprocess_data.preprocessed_data_path

    if existing_preprocessed_data == "overwrite":
        if preprocess_path.is_dir():
            utils.message_user(f"Removing existing file at {preprocess_path}\n")

        utils.message_user(f"Saving preprocessed data to {preprocess_path}")

        preprocess_data.save_all_preprocessed_data(overwrite=True)

    elif existing_preprocessed_data == "load_if_exists":
        if preprocess_path.is_dir():
            utils.message_user(
                f"\nSkipping preprocessing, using file at "
                f"{preprocess_path} for sorting.\n"
            )
        else:
            utils.message_user(
                f"No data found at {preprocess_path}, saving" f"preprocessed data."
            )
            preprocess_data.save_all_preprocessed_data(overwrite=False)

    elif existing_preprocessed_data == "fail_if_exists":
        if preprocess_path.is_dir():
            raise FileExistsError(
                f"Preprocessed binary already exists at "
                f"{preprocess_path}. "
                f"To overwrite, set 'existing_preprocessed_data' to 'overwrite'"
            )
        preprocess_data.save_all_preprocessed_data(overwrite=False)

    else:
        raise ValueError(
            "`existing_preproessed_data` argument not recognised."
            "Must be: 'load_if_exists', 'fail_if_exists' or 'overwrite'."
        )
