from .data_class import Data

def load_spikeglx_data():  # TODO: currently only spikeglx supported

    data = Data(base_path, sub_name, run_name)

    data.run_level_path = data.rawdata_path / sub_name / (run_name + "_g0")
    data.set_preprocessing_output_path()

    data["0-raw"] = se.read_spikeglx(
        folder_path=data.run_level_path, stream_id="imec0.ap", all_annotations=True
    )

    return data