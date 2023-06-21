from pathlib import Path

from swc_ephys.pipeline.postprocess import run_postprocess

base_path = Path(
    Path(r"/ceph/margrie/slenzi/2023/probe/raw/sc_pag_experiments/sub-004_id-1119976/experiment")
)

sub_name = "1119976"
run_name = "1119976_shank3_pretest_lse_posttest"

output_path = base_path / "derivatives" / sub_name / f"{run_name}" / "preprocessed"

run_postprocess(output_path, sorter="kilosort2_5")
