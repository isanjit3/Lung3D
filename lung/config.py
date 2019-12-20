import os

config = dict()
config["data_path"] = "/home/sanjit/datasets/lung/"
config["processed_data_path"] = "/home/sanjit/datasets/lung_processed/"

config["min_bound"] = -1000.0   #Minimum HU value we care about
config["max_bound"] = 400.0     #Max HU value we care about.
config["pixel_mean"] = 0.25
config["image_shape"] = (144, 144, 144)
config["data_file"] = "lung_data_file.h5"

config["labels"] = (1, 2, 4)  # the label numbers on the input image
config["n_labels"] = len(config["labels"])
config["patch_shape"] = None #(64,64,64)
config["nb_channels"] = 1 #We have only one modalities.

#config["all_modalities"] = ["t1", "t1ce", "flair", "t2"]
#config["training_modalities"] = config["all_modalities"]  # change this if you want to only use some of the modalities
#config["nb_channels"] = len(config["training_modalities"])
if "patch_shape" in config and config["patch_shape"] is not None:
    config["input_shape"] = tuple([config["nb_channels"]] + list(config["patch_shape"]))
else:
    config["input_shape"] = tuple([config["nb_channels"]] + list(config["image_shape"]))

#config["input_shape"] = tuple([1] + list(config["image_shape"]) )
config["truth_channel"] = 1
config["deconvolution"] = True  # if False, will use upsampling instead of deconvolut

config["pool_size"] = (2, 2, 2)  # pool size for the max pooling operations
config["batch_size"] = 1
config["validation_batch_size"] = 1
config["n_epochs"] = 500  # cutoff the training after this many epochs
config["patience"] = 10  # learning rate will be reduced after this many epochs if the validation loss is not improving
config["early_stop"] = 50  # training will be stopped after this many epochs without the validation loss improving
config["initial_learning_rate"] = 0.00001
config["learning_rate_drop"] = 0.5  # factor by which the learning rate will be reduced
config["validation_split"] = 0.8  # portion of the data that will be used for training
config["flip"] = False  # augments the data by randomly flipping an axis during
config["permute"] = True  # data shape must be a cube. Augments the data by permuting in various directions
config["distort"] = None  # switch to None if you want no distortion
config["augment"] = config["flip"] or config["distort"]
config["validation_patch_overlap"] = 0  # if > 0, during training, validation patches will be overlapping
config["training_patch_start_offset"] = (16, 16, 16)  # randomly offset the first patch index by up to this offset
config["skip_blank"] = True  # if True, then patches without any target will be skipped

config["model_file"] = os.path.abspath("lung_tumor_segmentation_model.h5")
config["training_file"] = os.path.abspath("lung_training_ids.pkl")
config["validation_file"] = os.path.abspath("lung_validation_ids.pkl")
config["overwrite"] = False  # If True, will previous files. If False, will use previously written files.
