from os.path import join
from pandas import read_excel

_one_drive_location = "/Users/gyanantaran/Library/CloudStorage/OneDrive-PlakshaUniversity"

_home_dir = _one_drive_location
proj_dir = join(_home_dir, "gym-exercise-feedback")

_data_relative_path = "data/labeled_data"
data_dir = join(proj_dir, _data_relative_path)

_labels_rel_path = "labels.xlsx"
labels_abs_path = join(data_dir, _labels_rel_path)

_vid_rel_path = "videos"
vid_dir = join(data_dir, _vid_rel_path)

labels_df = read_excel(labels_abs_path)
vid_names = labels_df["VidName"].to_list()

_landmarks_rel_path = "extracted_landmarks"
landmarks_dir = join(data_dir, _landmarks_rel_path)

_train_test_rel_path = "train_test"
train_test_dir = join(data_dir, _train_test_rel_path)
