import numpy as np
import scipy.spatial.distance as distance
from scipy.stats.stats import pearsonr
from spikeinterface.core import compute_sparsity

unit_id = 1
data = waveforms.get_waveforms(unit_id=unit_id)

num_spikes = data.shape[0]

if num_spikes == 1:
    print("only one cluster")
    # return # or whatever

sparsity = compute_sparsity(
    waveforms, peak_sign="neg", method="radius", radius_um=75
)  # TODO: how to determine "neg", "pos", "both", how to decide best radius
unit_best_chan_idxs = sparsity.unit_id_to_channel_indices[unit_id]

# TODO: checkout  similarity method from kilosort (see notes)
# x = data[i, :, unit_best_chan_idxs].flatten("F")  # this will not perform well
# without drift shifting. Could just take subset around peak.
# y = data[i, :, unit_best_chan_idxs].flatten("F")

method = "2"
sim = np.zeros((num_spikes, num_spikes))
for j in range(num_spikes):
    for i in range(j + 1):
        x = np.mean(
            data[i, :, unit_best_chan_idxs], axis=0
        )  # I guess this is similar to the template but per-unit.
        y = np.mean(data[j, :, unit_best_chan_idxs], axis=0)

        # cosine
        if method == "1":
            sim[i, j] = np.dot(x, y.T) / (
                np.sqrt(np.sum(x**2)) * np.sqrt(np.sum(y**2))
            )  # TODO: unsure of this rounding error

        elif method == "2":
            sim[i, j] = np.dot(x, y.T) / np.linalg.norm(x) / np.linalg.norm(y)

        elif method == "3":
            sim[i, j] = 1 - distance.cosine(x, y)

        # corr
        elif method == "4":
            sim[i, j] = pearsonr(x, y).statistic

        elif method == "5":
            demean_x = x - np.mean(x)
            demean_y = y - np.mean(y)
            sim[i, j] = np.sum(demean_x * demean_y) / np.sqrt(
                np.sum(demean_x**2) * np.sum(demean_y**2)
            )

print(sim)
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
