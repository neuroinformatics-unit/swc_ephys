"""
TODO: these tests don't check any output, only that things run without error
"""

import os
import shutil
from pathlib import Path

import pytest

from swc_ephys.pipeline import full_pipeline

ON_HPC = False


class TestFirstEphys:
    @pytest.fixture(scope="function")
    def test_info(self):
        """ """
        script_path = Path(os.path.dirname(os.path.realpath(__file__)))
        data_path = script_path.parent
        test_path = data_path / "data" / "steve_multi_run"
        sub_name = "1119617"
        run_names = [
            "1119617_LSE1_shank12",
            "1119617_posttest1_shank12",
            "1119617_pretest1_shank12",
        ]

        output_path = test_path / "derivatives"
        if output_path.is_dir():
            print("CHECK THIS")
            shutil.rmtree(output_path)

        yield [test_path, sub_name, run_names, output_path]

        if output_path.is_dir():
            print("CHECK THIS")
            shutil.rmtree(output_path)

    def run_full_pipeline(
        self,
        base_path,
        sub_name,
        run_names,
        use_existing_preprocessed_file=True,
        overwrite_existing_sorter_output=True,
        slurm_batch=False,
    ):
        full_pipeline.run_full_pipeline(
            base_path,
            sub_name,
            run_names,
            config_name="test",
            sorter="kilosort2_5",
            use_existing_preprocessed_file=use_existing_preprocessed_file,
            overwrite_existing_sorter_output=overwrite_existing_sorter_output,
            slurm_batch=slurm_batch,
        )

    def test_single_run_local(self, test_info):
        test_info.pop(3)

        test_info[2] = test_info[2][0]

        self.run_full_pipeline(*test_info)

    def test_multi_run_local(self, test_info):
        test_info.pop(3)

        test_info[2] = test_info[2][0]

        self.run_full_pipeline(*test_info)

    def test_single_run_slurm(self, test_info):
        test_info.pop(3)

        test_info[2] = test_info[2][0]

        self.run_full_pipeline(*test_info, slurm_batch={"wait": True})

    @pytest.mark.skipif(ON_HPC is False, reason="ON_HPC is false")
    def test_multi_run_slurm(self, test_info):
        test_info.pop(3)

        self.run_full_pipeline(*test_info, slurm_batch=True)

    def test_preprocessing_exists_error(self):
        raise NotImplementedError

    def test_use_existing_preprocessing_errror(self):
        raise NotImplementedError

    def test_sorter_exists_error(self):
        raise NotImplementedError

    def test_overwrite_sorter(self):
        raise NotImplementedError

    def test_sorting_only_local(self):
        raise NotImplementedError

    def test_sorting_only_slumr(self):
        raise NotImplementedError
