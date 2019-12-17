import os
import glob
import tensorflow as tf

from unet3d import dicom_data_util, generator, model, training
import config

def main(overwrite=False):
   
    #open the pre-processed data file.    
    data_file_opened = dicom_data_util.open_data_file(config.config["data_file"])

    if not overwrite and os.path.exists(config.config["model_file"]):
        cnn_model = training.load_old_model(config.config["model_file"])
    else:
        # instantiate new model
        cnn_model = model.unet_model_3d(input_shape=config.config["input_shape"],
                              pool_size=config.config["pool_size"],
                              n_labels=config.config["n_labels"],
                              initial_learning_rate=config.config["initial_learning_rate"],
                              deconvolution=config.config["deconvolution"])

    # get training and testing generators
    train_generator, validation_generator, n_train_steps, n_validation_steps = generator.get_training_and_validation_generators(
        data_file_opened,
        batch_size=config.config["batch_size"],
        data_split=config.config["validation_split"],
        overwrite=overwrite,
        validation_keys_file=config.config["validation_file"],
        training_keys_file=config.config["training_file"],
        n_labels=config.config["n_labels"],
        labels=config.config["labels"],
        patch_shape=config.config["patch_shape"],
        validation_batch_size=config.config["validation_batch_size"],
        validation_patch_overlap=config.config["validation_patch_overlap"],
        training_patch_start_offset=config.config["training_patch_start_offset"],
        permute=config.config["permute"],
        augment=config.config["augment"],
        skip_blank=config.config["skip_blank"],
        augment_flip=config.config["flip"],
        augment_distortion_factor=config.config["distort"])

    # run training
    training.train_model(model=cnn_model,
                model_file=config.config["model_file"],
                training_generator=train_generator,
                validation_generator=validation_generator,
                steps_per_epoch=n_train_steps,
                validation_steps=n_validation_steps,
                initial_learning_rate=config.config["initial_learning_rate"],
                learning_rate_drop=config.config["learning_rate_drop"],
                learning_rate_patience=config.config["patience"],
                early_stopping_patience=config.config["early_stop"],
                n_epochs=config.config["n_epochs"])
    
    #Close the data file.            
    data_file_opened.close()

if __name__ == "__main__":
    main(overwrite=config.config["overwrite"])
