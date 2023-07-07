"""
"""
from __future__ import annotations

import time
from typing import TYPE_CHECKING, Dict, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
from spikeinterface.core import compute_sparsity

if TYPE_CHECKING:
    from spikeinterface.core import BaseSorting

from pathlib import Path

import pandas as pd
import spikeinterface as si
from spikeinterface import curation
from spikeinterface.extractors import KiloSortSortingExtractor

from ..configs.configs import get_configs
from ..data_classes.sorting import SortingData
from ..pipeline.load_data import load_data_for_sorting
from ..utils import utils
from .waveform_compare import get_waveform_similarity


def run_postprocess(
    sorting_data: Union[Path, str, SortingData],
    sorter: str = "kilosort2_5",
    verbose: bool = True,
    waveform_options: Optional[Dict] = None,
) -> None:
    """
    Run post-processing, including ave quality metrics on sorting
    output to a quality_metrics.csv file.

    Parameters
    ----------

    sorting_data : Union[Path, str, SortingData]
        The path to the 'preprocessed' folder in the subject / run
        folder used for sorting or a SortingData object. If a

        SortingData object, the path will be read from the
        `preprocessed_data_path` attribute.

    sorter : str
        The name of the sorter (e.g. "kilosort2_5").

    verbose : bool
        If True, messages will be printed to console updating on the
        progress of preprocessing / sorting.
    """
    # Load sorting
    if not isinstance(sorting_data, SortingData):
        sorting_data = load_data_for_sorting(
            Path(sorting_data),
        )
    assert isinstance(sorting_data, SortingData), "type narrow `sorting_data`."

    # Create / load waveforms
    if waveform_options is None:
        _, _, waveform_options = get_configs("test")

    sorting_data.set_sorter_output_paths(sorter)

    utils.message_user(
        f"Quality Checks: sorting path used: {sorting_data.sorter_run_output_path}",
        verbose,
    )

    if not sorting_data.waveforms_output_path.is_dir():
        utils.message_user(f"Saving waveforms to {sorting_data.waveforms_output_path}")

        sorting_without_excess_spikes = load_sorting_output(sorting_data, sorter)

        waveforms = si.extract_waveforms(
            sorting_data.data["0-preprocessed"],
            sorting_without_excess_spikes,
            folder=sorting_data.waveforms_output_path,
            use_relative_path=True,
            **waveform_options,
        )
    else:
        utils.message_user(
            f"Loading existing waveforms from: {sorting_data.waveforms_output_path}",
            verbose,
        )

        waveforms = si.load_waveforms(sorting_data.waveforms_output_path)

    save_plots_of_templates(sorting_data.waveforms_output_path, waveforms)

    # Perform postprocessing
    quality_metrics = si.qualitymetrics.compute_quality_metrics(waveforms)
    quality_metrics.to_csv(sorting_data.quality_metrics_path)

    unit_locations = si.postprocessing.compute_unit_locations(
        waveforms, outputs="by_unit"
    )
    unit_locations_pandas = pd.DataFrame.from_dict(
        unit_locations, orient="index", columns=["x", "y"]
    )
    unit_locations_pandas.to_csv(sorting_data.unit_locations_path)

    save_waveform_similarities(sorting_data.waveforms_output_path, waveforms)

    utils.message_user(f"Quality metrics saved to {sorting_data.quality_metrics_path}")
    utils.message_user(f"Unit locations saved to {sorting_data.unit_locations_path}")


def save_plots_of_templates(waveforms_output_path, waveforms):
    """ """
    t = time.perf_counter()

    fs = waveforms.sampling_frequency
    n_samples = waveforms.nsamples
    time_ = np.arange(n_samples) / fs * 1000

    sparsity = compute_sparsity(
        waveforms, peak_sign="neg", method="radius", radius_um=75
    )

    for idx, unit_id in enumerate(waveforms.sorting.get_unit_ids()):
        unit_template = waveforms.get_template(unit_id)
        unit_best_chan_idxs = sparsity.unit_id_to_channel_indices[unit_id]

        # TODO: own function
        # TODO: why are some templates all zeros?
        # if unit_id in [33, "33"]:
        #    breakpoint()
        plt.plot(time_, np.mean(unit_template[:, unit_best_chan_idxs], axis=1))

        peak_chan_idx = np.argmin(
            np.mean(unit_template[:, unit_best_chan_idxs], axis=0)
        )
        plt.plot(time_, unit_template[:, unit_best_chan_idxs[peak_chan_idx]])

        plt.legend(["max signal channel", "mean across best channels"])
        plt.xlabel("Time (ms)")
        plt.ylabel("Voltage (uV)")
        plt.title(f"Unit {unit_id} Template")

        output_folder = waveforms_output_path / "images"
        output_folder.mkdir(exist_ok=True)
        plt.savefig(waveforms_output_path / "images" / f"unit_{unit_id}.png")
        plt.clf()

    print(f"Saving plots of templates took: {time.perf_counter() - t}")


def load_sorting_output(sorting_data: SortingData, sorter: str) -> BaseSorting:
    """
    Load the output of a sorting run.

    TODO: understand remove_excess_spikes.
    """
    if not sorting_data.sorter_run_output_path.is_dir():
        raise FileNotFoundError(
            f"{sorter} output was not found at "
            f"{sorting_data.sorter_run_output_path}.\n"
            f"Quality metrics were not generated."
        )

    assert len(sorting_data) == 1, (
        "unexpected number of entries in " "`sorting_data` dict."
    )

    recording = sorting_data[sorting_data.init_data_key]

    sorting = KiloSortSortingExtractor(
        folder_path=sorting_data.sorter_run_output_path,
        keep_good_only=False,
    )

    # TODO: use upcoming SI option, see
    #  https://github.com/SpikeInterface/spikeinterface/issues/1760
    sorting = sorting.remove_empty_units()
    sorting_without_excess_spikes = curation.remove_excess_spikes(sorting, recording)

    return sorting_without_excess_spikes


def save_waveform_similarities(waveforms_output_path, waveforms):
    out_path = waveforms_output_path / "similarity_matricies"
    out_path.mkdir(exist_ok=True)

    time.perf_counter()
    for unit_id in waveforms.sorting.get_unit_ids():
        sim_matrix, spike_times = get_waveform_similarity(waveforms, unit_id, "jax")

        sim_matrix_pd = pd.DataFrame(sim_matrix, columns=spike_times, index=spike_times)
        sim_matrix_pd.to_csv(out_path / f"waveform_similarity_unit_{unit_id}.csv")

    # TODO: use message utils
    print(f"Saving waveform similarities took: {time.perf_counter()} - t")
