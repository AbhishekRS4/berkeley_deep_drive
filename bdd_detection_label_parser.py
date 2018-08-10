# @author : Abhishek R S  
# convert berkeley deep drive bounding box labels to match the yolo label format

import os
import sys
import json
import argparse
import numpy as np

class_mapping_dict = {'bus' : 0, 'traffic light' : 1, 'traffic sign' : 2, 'person' : 3, 'bike' : 4, 'truck' : 5, 'motor' : 6, 'car' : 7, 'train' : 8, 'rider' : 9}

'''
class mapping

bus - 0
traffic light - 1
traffic sign - 2
person - 3
bike - 4
truck - 5
motor - 6
car - 7
train - 8
rider - 9
'''

# convert json to text file with yolo v2 format
def convert_json_to_yolo(src_labels_dir, tar_labels_dir, src_image_width = 1280, src_image_height = 720, tar_image_width = 416, tar_image_height = 416, delimiter = ' '):
    # list all files in the source label directory
    src_labels_list = os.listdir(src_labels_dir) 

    # create target directory if it doesn't exist
    if not os.path.exists(tar_labels_dir):
        os.makedirs(tar_labels_dir)

    # set the appropriate delimiter
    if delimiter == ',':
        tar_file_format = '.csv'
    else:
        tar_file_format = '.txt'

    # parse every json file and convert it to appropriate format
    for json_file in src_labels_list:
        label = json.load(open(os.path.join(src_labels_dir, json_file)))

        target_labels = list()
        for i in range(len(label['frames'][0]['objects'])):
            obj_dict = label['frames'][0]['objects'][i]

            # if box2d element is present then parse
            if 'box2d' in obj_dict.keys():
                tar_vertices = list()
                category = class_mapping_dict[obj_dict['category']]
                vertices = obj_dict['box2d']

                # compute x1, y1, x2, y2 for target image dimension 
                x1 = vertices['x1'] * tar_image_width / src_image_width
                y1 = vertices['y1'] * tar_image_height / src_image_height
                x2 = vertices['x2'] * tar_image_width / src_image_width
                y2 = vertices['y2'] * tar_image_height / src_image_height

                # compute center_x, center_y, width, height of the bounding box
                c_x = (x1 + x2) / 2.0
                c_y = (y1 + y2) / 2.0
                w = (x2 - x1)
                h = (y2 - y1)

                # compute center_x, center_y, width, height of the bounding box relative to target image dimension
                c_x *= (1.0 / tar_image_width)
                c_y *= (1.0 / tar_image_height)
                w *= (1.0 / tar_image_width)
                h *= (1.0 / tar_image_height) 

                # ignore very small objects
                if w < (1 / float(tar_image_width)) or h < (1 / float(tar_image_height)):
                    continue

                # add center_x, center_y, width, height of the bounding box to a temporary list
                tar_vertices.append(c_x)
                tar_vertices.append(c_y)
                tar_vertices.append(w)
                tar_vertices.append(h)

                # add one object to the main list
                target_labels.append([category] + tar_vertices)

        # convert and save the array in the specified format
        np.savetxt(os.path.join(tar_labels_dir, json_file.split('.')[0] + tar_file_format), np.array(target_labels), fmt = ['%d', '%.8f', '%.8f', '%.8f', '%.8f'],delimiter = delimiter)

def main():
    # default values
    src_image_width = 1280
    src_image_height = 720
    tar_image_width = 416
    tar_image_height = 416

    src_labels_dir = '/data/data/datasets/berkeley_deep_drive/object_detection/labels/val/'
    tar_labels_dir = '/data/data/datasets/berkeley_deep_drive/object_detection/labels/val_resized/'

    delimiter = ' '

    parser = argparse.ArgumentParser()
    parser.add_argument('-src_labels_dir', default = src_labels_dir, type = str, help = 'path to source label files')
    parser.add_argument('-tar_labels_dir', default = tar_labels_dir, type = str, help = 'path to target label files')
    parser.add_argument('-src_width', default = src_image_width, type = int, help = 'width of source image')
    parser.add_argument('-src_height', default = src_image_height, type = int, help = 'height of source image')
    parser.add_argument('-tar_width', default = tar_image_width, type = int, help = 'width of target image')
    parser.add_argument('-tar_height', default = tar_image_height, type = int, help = 'height of target image')
    parser.add_argument('-delimiter', default = delimiter, type = str, help = 'delimiter to be used')

    input_args = vars(parser.parse_args(sys.argv[1:]))

    print('Values used for converting source to target labels')
    for k in input_args.keys():
        print(k + ': ' + str(input_args[k]))
    print('')
    print('')

    print('Conversion Started.......')
    convert_json_to_yolo(input_args['src_labels_dir'], input_args['tar_labels_dir'], input_args['src_width'], input_args['src_height'], input_args['tar_width'], input_args['tar_height'], input_args['delimiter'])
    print('Conversion Completed')

if __name__ == '__main__':
    main()
