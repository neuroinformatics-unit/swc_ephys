from pathlib import Path

from swc_ephys.pipeline.postprocess import run_postprocess

base_path = Path(
    Path(
        r"/ceph/neuroinformatics/neuroinformatics/scratch/jziminski/ephys/test_data/steve_multi_run/1119617/time-mid"
    )
)
# TODO: why is this taking the preprocessed path? isn't sorting more intuitive? or top level? think...
sub_name = "1119617"
run_name = "1119617_LSE1_shank12_posttest1_pretest1"

output_path = (
    base_path / "derivatives" / sub_name / f"{run_name}" / "preprocessed"
)  # This is the most stupidly named variable of all time. This is INPUT preprocessing.

run_postprocess(output_path, sorter="kilosort2_5")
