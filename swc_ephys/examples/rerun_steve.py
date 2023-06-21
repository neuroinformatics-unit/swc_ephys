from pathlib import Path

from swc_ephys.pipeline.full_pipeline import run_full_pipeline

base_path = Path(r"/ceph/margrie/slenzi/2023/probe/raw/sc_pag_experiments/sub-004_id-1119976/experiment")
sub_name = "1119976" # "sub-004_id-1119976"
run_names = [
    "1119976_shank3_pretest",
    "1119976_shank3_lse",
    "1119976_shank3_posttest",
]

from swc_ephys.pipeline.load_data import load_spikeglx_data
from swc_ephys.configs.configs import get_configs
from swc_ephys.pipeline.preprocess import preprocess

config_name = "test"
pp_steps, sorter_options = get_configs(config_name)

preprocess_data = load_spikeglx_data(base_path, sub_name, run_names)
breakpoint()

preprocess_data = preprocess(preprocess_data, pp_steps, verbose=True)

sorter = "kilosort2_5"
# /ceph/margrie/slenzi/2023/probe/raw/sc_pag_experiments/sub-004_id-1119976/experiment/derivatives/1119976/1119976_shank3_pretest_lse_posttest/preprocessed