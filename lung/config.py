config = dict()
config["data_path"] = "/home/sanjit/datasets/lung/"
config["processed_data_path"] = "/home/sanjit/datasets/lung_processed/"

config["min_bound"] = -1000.0   #Minimum HU value we care about
config["max_bound"] = 400.0     #Max HU value we care about.
config["pixel_mean"] = 0.25
config["image_shape"] = (512, 512, 8)
config["data_file"] = "lung_data_file.h5"
