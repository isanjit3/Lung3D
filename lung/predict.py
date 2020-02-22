#predict the outcome with the trained model.
import os
from lung.config import config
import tensorflow as tf
from unet3d.prediction import run_validation_cases, run_validation_case
from unet3d.training import load_old_model
import tables


tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True
tf_config.gpu_options.per_process_gpu_memory_fraction = 0.98
tf.keras.backend.set_session(tf.Session(config=tf_config))


def run_validation_at_index(index):
    case_directory = os.path.join(os.path.abspath("prediction"), "validation_case_{}".format(index))
    data_file = tables.open_file(config["data_file"], "r")
    model = load_old_model(config["model_file"])
    result = run_validation_case(data_index=index, 
                            output_dir=case_directory, 
                            model=model, 
                            data_file=data_file,
                            training_modalities=None, 
                            output_label_map=True, 
                            labels=config["labels"]
                            )

def main():
    prediction_dir = os.path.abspath("prediction")
    run_validation_cases(validation_keys_file=config["validation_file"],
                         model_file=config["model_file"],
                         training_modalities=None,
                         labels=config["labels"],
                         hdf5_file=config["data_file"],
                         output_label_map=True,
                         output_dir=prediction_dir)


if __name__ == "__main__":
    #main()
    run_validation_at_index(893)
