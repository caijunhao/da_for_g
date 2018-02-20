import argparse
import os

import cv2
from math import pi

parser = argparse.ArgumentParser(description='organize data collected by our own baxter.')
parser.add_argument('--data_path', required=True, type=str, help='path to the data set.')
parser.add_argument('--label_path', required=True, type=str, help='path to the label txt file.')
parser.add_argument('--output_path', required=True, type=str, help='path to the organized data.')
parser.add_argument('--argumented', default=False, type=str, help='whether to argument the data set.')
args = parser.parse_args()


def main():
    if not os.path.exists(args.data_path):
        raise ValueError('data path does not exist.')
    positive_data_path = os.path.join(args.output_path, 'positive', 'Images')
    negative_data_path = os.path.join(args.output_path, 'negative', 'Images')
    if not os.path.exists(args.output_path):
        os.makedirs(positive_data_path)
        os.makedirs(negative_data_path)
    f_info_positive = open(os.path.join(args.output_path, 'positive', 'dataInfo.txt'), 'a')
    f_info_negative = open(os.path.join(args.output_path, 'negative', 'dataInfo.txt'), 'a')
    num_pos = len(os.listdir(positive_data_path))
    num_neg = len(os.listdir(negative_data_path))
    num_data = num_pos + num_neg
    with open(args.label_path, 'r') as f:
        annotations = f.readlines()
    for i in xrange(len(annotations)):
        file_path, grasp_index, label = annotations[i].split(' ')
        if file_path.endswith('.jpg'):
            file_name = file_path.split('/')[-1]
        else:
            file_name = file_path + '.jpg'
        angle_index = int(grasp_index)
        label = int(label)
        img = cv2.imread(os.path.join(args.data_path, file_name))
        if args.argumented:
            for j in range(4):
                grasp_angle = (((angle_index + j * 9) % 18) * 10 - 90) * 1.0 / 180 * pi
                rotated_img = img if j == 0 else cv2.rotate(img, j - 1)
                if label == 0:
                    cv2.imwrite(positive_data_path + '/{:06d}.jpg'.format(i * 4 + j + num_data), rotated_img)
                    f_info_positive.write(
                        ','.join(['{:06d}.jpg'.format(i * 4 + j + num_data), '{}\n'.format(grasp_angle)]))
                else:
                    cv2.imwrite(negative_data_path + '/{:06d}.jpg'.format(i * 4 + j + num_data), rotated_img)
                    f_info_negative.write(
                        ','.join(['{:06d}.jpg'.format(i * 4 + j + num_data), '{}\n'.format(grasp_angle)]))
        else:
            grasp_angle = (angle_index * 10 - 90) * 1.0 / 180 * pi
            if label == 0:
                cv2.imwrite(positive_data_path + '/{:06d}.jpg'.format(i + num_data), img)
                f_info_positive.write(
                    ','.join(['{:06d}.jpg'.format(i + num_data), '{}\n'.format(grasp_angle)]))
            else:
                cv2.imwrite(negative_data_path + '/{:06d}.jpg'.format(i + num_data), img)
                f_info_negative.write(
                    ','.join(['{:06d}.jpg'.format(i + num_data), '{}\n'.format(grasp_angle)]))
    f_info_positive.close()
    f_info_negative.close()


if __name__ == '__main__':
    main()
