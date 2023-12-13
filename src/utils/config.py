from os.path import join
from pandas import read_excel

_home = "/Users/gyanantaran/Library/CloudStorage/OneDrive-PlakshaUniversity"
parent_dir = join(_home, "gym-exercise-feedback")

_data_relative_path = "data/labeled_data"
data_dir = join(parent_dir, _data_relative_path)

_labels_rel_path = "labels.xlsx"
labels_abs_path = join(data_dir, _labels_rel_path)

_videos_rel_path = "videos"
vids_dir = join(data_dir, _videos_rel_path)

labels_df = read_excel(labels_abs_path)
vid_names = labels_df["VidName"].to_list()

_landmarks_rel_path = "extracted_landmarks"
landmarks_dir = join(data_dir, _landmarks_rel_path)

_train_test_rel_path = "train_test"
train_test_dir = join(data_dir, _train_test_rel_path)


# others
num_frames_per_new = 30