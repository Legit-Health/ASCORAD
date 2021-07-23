###########################################
# Author:     Alfonso Medela              #
# Contact     alfonso@legit.health        #
# Copyright:  Legit.Health                #
# Website:    https://legit.health/       #
###########################################

# Imports
import json

def merge_labels(json_1_path, json_2_path):

    '''
    Function to merge the annotation labels of different dataset versions
    :param json_1_path: Path to one JSON
    :param json_2_path: Path to the second JSON
    :return: Merged JSON
    '''

    # Load JSON
    with open(json_1_path) as json_file:
        json_data_v1 = json.load(json_file)

    # Load JSON
    with open(json_2_path) as json_file:
        json_data_v2 = json.load(json_file)

    # Get keys
    v1_keys = list(json_data_v1.keys())
    v2_keys = list(json_data_v2.keys())

    # Add v2 keys to v1
    n = int(v1_keys[-1][3:]) + 1
    for v2_key in v2_keys:
        json_data_v1['img' + str(n)] = json_data_v2[v2_key]
        n += 1

    return json_data_v1

if __name__ == '__main__':

    # Load paths
    labels_path_v1 = 'PATH 1'
    labels_path_v2 = 'PATH 2'
    output_path = 'OUTPUT_PATH'

    # Merge jsons
    output_json = merge_labels(labels_path_v1, labels_path_v2)

    # Save output
    with open(output_path, 'w') as outfile:
        json.dump(output_json, outfile)
