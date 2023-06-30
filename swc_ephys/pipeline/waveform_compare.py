from typing import Literal

import jax.numpy as jnp
import numpy as np
from jax import jit, vmap
from jax.lax import scan
from spikeinterface.core import compute_sparsity

# TODO: checkout  similarity method from kilosort (see notes)
# x = data[i, :, unit_best_chan_idxs].flatten("F")  # this will not perform well
# without drift shifting. Could just take subset around peak.
# y = data[i, :, unit_best_chan_idxs].flatten("F")

# TODO: add jax, jaxlib to dependencies
# Save some example waveforms to a PDF in the waveforms file.


def get_waveform_similarity(
    waveforms, unit_id, backend: Literal["numpy", "jax"] = "numpy"
):  # TODO: where to inferface with spike window?
    data = waveforms.get_waveforms(unit_id=unit_id)

    if data.shape[0] == 1:
        print("only one cluster")  # TODO: better
        return  # or whatever

    # TODO: how to determine "neg", "pos", "both", how to decide best radius
    sparsity = compute_sparsity(
        waveforms, peak_sign="neg", method="radius", radius_um=75
    )
    unit_best_chan_idxs = sparsity.unit_id_to_channel_indices[unit_id]

    if backend == "numpy":
        sim = calculate_similarity_numpy(data, unit_best_chan_idxs)
    elif backend == "jax":
        sim = calculate_similarity_jax(data, unit_best_chan_idxs)

    return sim


@jit  # dont think this makes a difference, see Jax docs compiled under the hood anyway.
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

    # TOOD: is this okay? because carry does not change. Using is to hack a loop.
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


# fill sim with flip

# TODO
# 1) read quality metrics
# 2) profile with larger clusters
# 3) use Jax for fun
# 4) think - what exactly do we want from waveform comparison in the current use-case? (noise, low sampling rate)
# 5) Package up
# 6) Talk to steve, submit to SI?


# Then can plot these next to spike times. Can also smooth other these
# thing of other ML or probabilistic ways to capture this information.
# array([  7876, 158707, 176768, 176987, 181303, 210568])
# waveforms.sorting.get_unit_spike_train

# gonna have to handle 'channel' and 'shank'
# can get sorter when loading waveform?!?!  need it for spike times. DOn't necessarily need this.s
