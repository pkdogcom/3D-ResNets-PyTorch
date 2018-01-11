from __future__ import print_function, division
import os
import sys
import subprocess


def class_process(dir_path, class_name):
    class_path = os.path.join(dir_path, class_name)
    if not os.path.isdir(class_path):
        return []

    n_frames_list = []
    for file_name in os.listdir(class_path):
        video_dir_path = os.path.join(class_path, file_name)
        image_indices = []
        for image_file_name in os.listdir(video_dir_path):
            if 'image' not in image_file_name:
                continue
            image_indices.append(int(image_file_name[6:11]))

        if len(image_indices) == 0:
            print('no image files', video_dir_path)
            n_frames = 0
        else:
            image_indices.sort(reverse=True)
            n_frames = image_indices[0]
            print(video_dir_path, n_frames)
        n_frames_list.append('{}/{} {}'.format(class_name, file_name, n_frames))
        # with open(os.path.join(video_dir_path, 'n_frames'), 'w') as dst_file:
        #     dst_file.write(str(n_frames))
    return n_frames_list


if __name__ == "__main__":
    dir_path = sys.argv[1]
    output_file = sys.argv[2]
    n_frames_list = []
    for class_name in os.listdir(dir_path):
        n_frames_list += class_process(dir_path, class_name)
    with open(output_file, 'w') as f:
        f.write('\n'.join(n_frames_list))
    # class_name = 'test'
    # class_process(dir_path, class_name)
