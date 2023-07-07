from typing import Literal

import jax.numpy as jnp
import numpy as np
from jax import jit, vmap
from jax.lax import scan
from spikeinterface.core import compute_sparsity


def get_waveform_similarity(
    waveforms, unit_id, backend: Literal["numpy", "jax"] = "numpy"
):
    data = waveforms.get_waveforms(unit_id=unit_id)

    # TODO: check with SI these assumptions are correct
    # Own function
    select_waveform_tuples = waveforms.get_sampled_indices(unit_id)
    select_waveform_idxs, seg_idxs = zip(*select_waveform_tuples)
    assert np.all(np.array(seg_idxs) == 0), "multi-segment waveforms not tested."

    all_waveform_peak_idxs = waveforms.sorting.get_unit_spike_train(unit_id)
    selected_waveform_idxs = all_waveform_peak_idxs[np.array(select_waveform_idxs)]
    selected_waveform_peak_times = selected_waveform_idxs / waveforms.sampling_frequency

    if data.shape[0] == 1:
        print("only one cluster")  # TODO: better message, use utils
        return np.nan, selected_waveform_peak_times

    # TODO: how to determine "neg", "pos", "both", how to decide the
    #  peak_sign / radius when computing sparisty. Own function.
    sparsity = compute_sparsity(
        waveforms, peak_sign="neg", method="radius", radius_um=75
    )
    unit_best_chan_idxs = sparsity.unit_id_to_channel_indices[unit_id]

    if backend == "numpy":
        sim = calculate_similarity_numpy(data, unit_best_chan_idxs)
    elif backend == "jax":
        sim = calculate_similarity_jax(data, unit_best_chan_idxs)

    return sim, selected_waveform_peak_times


@jit
def calculate_similarity_jax(data, unit_best_chan_idxs):
    # This actually duplicates the upper and lower triangular,
    # but is so fast it doesn't really matter. vmapping occurs
    # across the entire axis so not sure it can be re-configured.
    # Can think a bit more about this though.
    data_mean = jnp.mean(data[:, :, unit_best_chan_idxs], axis=2)

    def func(carry, i):
        y = data_mean[i, :]
        sim_row = vmap(
            lambda x: jnp.dot(x, y.T) / (jnp.linalg.norm(x) * jnp.linalg.norm(y)),
            in_axes=0,
        )(data_mean)
        return carry, sim_row

    # TOOD: is this stylistically okay? because carry does not change.
    # Using is to hack a loop.
    return scan(func, data_mean, np.arange(data_mean.shape[0]))[1]


def calculate_similarity_numpy(data, unit_best_chan_idxs):  # TODO: variable naming
    """ """
    num_spikes = data.shape[0]
    data_mean = np.mean(data[:, :, unit_best_chan_idxs], axis=2)

    sim = np.zeros((num_spikes, num_spikes))
    for j in range(num_spikes):
        for i in range(j + 1):
            x = data_mean[i, :]
            y = data_mean[j, :]

            sim[i, j] = np.dot(x, y.T) / (np.linalg.norm(x) * np.linalg.norm(y))

    i_lower = np.tril_indices(sim.shape[0], -1)
    sim[i_lower] = sim.T[i_lower]

    return sim
