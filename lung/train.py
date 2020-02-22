import os
import glob
import tensorflow as tf

from unet3d import dicom_data_util, generator, model, training
from lung.config import config

tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True
tf_config.gpu_options.per_process_gpu_memory_fraction = 0.98
tf.keras.backend.set_session(tf.Session(config=tf_config))

def main(overwrite=False):
   
    #open the pre-processed data file.    
    data_file_opened = dicom_data_util.open_data_file(config["data_file"])

    if not overwrite and os.path.exists(config["model_file"]):
        cnn_model = training.load_old_model(config["model_file"])
    else:
        # instantiate new model
        cnn_model = model.unet_model_3d(input_shape=config["input_shape"],
                              pool_size=config["pool_size"],
                              n_labels=config["n_labels"],
                              initial_learning_rate=config["initial_learning_rate"],
                              deconvolution=config["deconvolution"],
                              depth=4 # 4 is the default.
                            )

    # get training and testing generators
    train_generator, validation_generator, n_train_steps, n_validation_steps = generator.get_training_and_validation_generators(
        data_file_opened,
        batch_size=config["batch_size"],
        data_split=config["validation_split"],
        overwrite=overwrite,
        validation_keys_file=config["validation_file"],
        training_keys_file=config["training_file"],
        n_labels=config["n_labels"],
        labels=config["labels"],
        patch_shape= config["patch_shape"],
        validation_batch_size=config["validation_batch_size"],
        validation_patch_overlap=config["validation_patch_overlap"],
        training_patch_start_offset=config["training_patch_start_offset"],
        permute=config["permute"],
        augment=config["augment"],
        skip_blank=config["skip_blank"],
        augment_flip=config["flip"],
        augment_distortion_factor=config["distort"])

    # run training
    training.train_model(model=cnn_model,
                model_file=config["model_file"],
                training_generator=train_generator,
                validation_generator=validation_generator,
                steps_per_epoch=n_train_steps,
                validation_steps=n_validation_steps,
                initial_learning_rate=config["initial_learning_rate"],
                learning_rate_drop=config["learning_rate_drop"],
                learning_rate_patience=config["patience"],
                early_stopping_patience=config["early_stop"],
                n_epochs=config["n_epochs"])
    
    #Close the data file.            
    data_file_opened.close()

if __name__ == "__main__":
    main(overwrite=config["overwrite"])
