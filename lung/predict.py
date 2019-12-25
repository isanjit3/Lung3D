#predict the outcome with the trained model.
import os
import config
import tensorflow as tf
from unet3d.prediction import run_validation_cases

tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True
tf_config.gpu_options.per_process_gpu_memory_fraction = 0.98
tf.keras.backend.set_session(tf.Session(config=tf_config))

def main():
    prediction_dir = os.path.abspath("prediction")
    run_validation_cases(validation_keys_file=config.config["validation_file"],
                         model_file=config.config["model_file"],
                         training_modalities=None,
                         labels=config.config["labels"],
                         hdf5_file=config.config["data_file"],
                         output_label_map=True,
                         output_dir=prediction_dir)


if __name__ == "__main__":
    main()
