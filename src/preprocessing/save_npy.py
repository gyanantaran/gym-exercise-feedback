#%% Imports
from numpy import save, array, isnan
from os.path import join

from src.paths.paths import train_test_dir, vid_names, labels_df
from src.preprocessing.data_augmentation import augment_frames


# %% Save X.npy
def save_X() -> None:
    subdir_name = './slidify'
    dir_loc = train_test_dir
    x_file_name = 'X.npy'
    y_file_name = 'Y.npy'
    x_save_loc = join(dir_loc, subdir_name, x_file_name)
    y_save_loc = join(dir_loc, subdir_name, y_file_name)

    num_videos = len(vid_names)
    X = []
    Y = []
    for vid_name in vid_names:
        fps_val = labels_df[labels_df['VidName'] == vid_name]['FPS'].iloc[0]

        if isnan(fps_val):
            continue

        vid_label = int(labels_df[labels_df['VidName'] == vid_name]['Label'].iloc[0])
        augmented_frames = augment_frames(vid_name=vid_name)

        num_new_videos, _, _, _ = augmented_frames.shape
        # print('New augmented frames shape:\t', num_new_videos)

        X += list(augmented_frames)
        Y += [vid_label] * num_new_videos  # adding the same label for the augmented sub-parts

    X = array(X)
    Y = array(Y)

    # save the X and Y '.npy'
    save(x_save_loc, X)
    save(y_save_loc, Y)

    return


if __name__ == "__main__":
    save_X()