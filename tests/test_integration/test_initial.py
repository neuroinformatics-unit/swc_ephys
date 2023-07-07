"""
TODO: these tests don't check any output, only that things run without error
"""
import os
import shutil
from pathlib import Path

import pytest

from swc_ephys.pipeline import full_pipeline

ON_HPC = True


class TestFirstEphys:
    @pytest.fixture(scope="class")
    def output_data_path(self):
        script_path = Path(os.path.dirname(os.path.realpath(__file__)))
        data_path = script_path.parent
        output_data_path = data_path / "data" / "steve_multi_run"
        return output_data_path

    @pytest.fixture(scope="function")
    def test_info(self, output_data_path):
        """ """
        sub_name = "1119617"
        run_names = [
            "1119617_LSE1_shank12",
            "1119617_posttest1_shank12",
            "1119617_pretest1_shank12",
        ]

        output_path = output_data_path / "derivatives"
        if output_path.is_dir():
            shutil.rmtree(output_path)

        yield [output_data_path, sub_name, run_names, output_path]

        if output_path.is_dir():
            shutil.rmtree(output_path)

    def run_full_pipeline(
        self,
        base_path,
        sub_name,
        run_names,
        existing_preprocessed_data="fail_if_exists",
        overwrite_existing_sorter_output=True,
        slurm_batch=False,
    ):
        full_pipeline.run_full_pipeline(
            base_path,
            sub_name,
            run_names,
            config_name="test",
            sorter="kilosort2_5",
            existing_preprocessed_data=existing_preprocessed_data,
            overwrite_existing_sorter_output=overwrite_existing_sorter_output,
            slurm_batch=slurm_batch,
        )

    def test_single_run_local(self, test_info):
        test_info.pop(3)
        test_info[2] = test_info[2][0]
        self.run_full_pipeline(*test_info)

    def test_single_run_local_overwrite(self, test_info):
        test_info.pop(3)
        test_info[2] = test_info[2][0]

        self.run_full_pipeline(*test_info)

        self.run_full_pipeline(*test_info, existing_preprocessed_data="overwrite")

        with pytest.raises(BaseException) as e:
            self.run_full_pipeline(
                *test_info, existing_preprocessed_data="fail_if_exists"
            )

        assert "To overwrite, set 'existing_preprocessed_data' to 'overwrite'" in str(
            e.value
        )

    def test_multi_run_local(self, test_info):
        test_info.pop(3)

        test_info[2] = test_info[2][0]

        self.run_full_pipeline(*test_info)

    @pytest.mark.skipif(ON_HPC is False, reason="ON_HPC is false")
    def test_single_run_slurm(self, test_info, output_data_path):
        test_info.pop(3)

        test_info[2] = test_info[2][0]

        self.clear_slurm_logs(output_data_path)

        self.run_full_pipeline(*test_info, slurm_batch={"wait": True})

        self.check_slurm_log(output_data_path)

    def check_slurm_log(self, output_data_path):
        slurm_run = output_data_path.glob("slurm_logs/*/*log.out")
        slurm_run = list(slurm_run)[0]

        with open(slurm_run, "r") as log:
            log_output = log.readlines()

        assert "Stopping container" in log_output
        assert "Saving waveforms to" in log_output
        assert "Quality metrics saved to" in log_output
        assert "Job completed successfully" in log_output

    @pytest.mark.skipif(ON_HPC is False, reason="ON_HPC is false")
    def test_multi_run_slurm(self, test_info, output_data_path):
        test_info.pop(3)

        self.clear_slurm_logs(output_data_path)

        self.run_full_pipeline(*test_info, slurm_batch=True)

        self.check_slurm_log(output_data_path)

    def clear_slurm_logs(self, output_data_path):
        slurm_path = output_data_path / "slurm_logs"
        [shutil.rmtree(path_) for path_ in slurm_path.glob("*-*-*_*-*-*")]

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
