from __future__ import print_function, division
import sys
import json
import pandas as pd


def convert_csv_to_dict(csv_path, sublist, subset):
    data = pd.read_csv(csv_path)
    with open(sublist, 'r') as f:
        ids_to_keep = set([line.strip() for line in f.readlines()])
    keys = []
    key_labels = []
    for i in range(data.shape[0]):
        row = data.ix[i, :]
        if row['youtube_id'] not in ids_to_keep:
            continue
        basename = '%s_%s_%s' % (row['youtube_id'],
                                 '%06d' % row['time_start'],
                                 '%06d' % row['time_end'])
        keys.append(basename)
        if subset != 'testing':
            key_labels.append(row['label'])

    database = {}
    for i in range(len(keys)):
        key = keys[i]
        database[key] = {}
        database[key]['subset'] = subset
        if subset != 'testing':
            label = key_labels[i]
            database[key]['annotations'] = {'label': label}
        else:
            database[key]['annotations'] = {}

    return database, list(set(key_labels))


def convert_kinetics_csv_to_activitynet_json(train_csv_path, mini_train_list_path, val_csv_path, mini_val_list_path, dst_json_path):

    train_database, labels = convert_csv_to_dict(train_csv_path, mini_train_list_path, 'training')
    val_database, _ = convert_csv_to_dict(val_csv_path, mini_val_list_path, 'validation')

    dst_data = {'labels': labels, 'database': {}}
    dst_data['database'].update(train_database)
    dst_data['database'].update(val_database)

    with open(dst_json_path, 'w') as dst_file:
        json.dump(dst_data, dst_file)


if __name__ == "__main__":
    train_csv_path = sys.argv[1]
    mini_train_list_path = sys.argv[2]
    val_csv_path = sys.argv[3]
    mini_val_list_path = sys.argv[4]
    dst_json_path = sys.argv[5]

    convert_kinetics_csv_to_activitynet_json(
        train_csv_path, mini_train_list_path, val_csv_path, mini_val_list_path, dst_json_path)