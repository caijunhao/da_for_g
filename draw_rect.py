import argparse
import copy
import os

import numpy as np
import cv2
from math import pi

parser = argparse.ArgumentParser(description='organize data collected by our own baxter.')
parser.add_argument('--data_path', required=True, type=str, help='path to the data set.')
parser.add_argument('--label_path', required=True, type=str, help='path to the label txt file.')
args = parser.parse_args()


def process_and_draw_rect(img, grasp_angle, sx, sy):
    img_temp = copy.deepcopy(img)
    grasp_l = 200 / 3.0
    grasp_w = 200 / 6.0
    points = np.array([[-grasp_l, -grasp_w],
                       [grasp_l, -grasp_w],
                       [grasp_l, grasp_w],
                       [-grasp_l, grasp_w]])
    rotate_matrix = np.array([[np.cos(grasp_angle), -np.sin(grasp_angle)],
                              [np.sin(grasp_angle), np.cos(grasp_angle)]])
    rot_points = np.dot(rotate_matrix, points.transpose()).transpose()
    temp = np.array([[sx, sy],
                     [sx, sy],
                     [sx, sy],
                     [sx, sy]])
    im_points = (rot_points + temp).astype(np.int)
    cv2.line(img_temp, tuple(im_points[0]), tuple(im_points[1]), color=(0, 255, 0), thickness=5)
    cv2.line(img_temp, tuple(im_points[1]), tuple(im_points[2]), color=(0, 0, 255), thickness=5)
    cv2.line(img_temp, tuple(im_points[2]), tuple(im_points[3]), color=(0, 255, 0), thickness=5)
    cv2.line(img_temp, tuple(im_points[3]), tuple(im_points[0]), color=(0, 0, 255), thickness=5)
    return img_temp


def main():
    if not os.path.exists(args.data_path):
        raise ValueError('data path does not exist.')
    with open(args.label_path, 'r') as f:
        annotations = f.readlines()
    for i in xrange(len(annotations)):
        file_path, grasp_index, label = annotations[i].split(' ')
        file_name = file_path.split('/')[-1]
        angle_index = int(grasp_index)
        grasp_angle = (angle_index * 10 - 90) * 1.0 / 180 * pi
        img = cv2.imread(os.path.join(args.data_path, file_name))
        h, w, _ = img.shape
        img_temp = process_and_draw_rect(img, grasp_angle, h/2, w/2)
        cv2.imwrite(args.data_path+'/'+file_name[:-4]+'_rect.jpg', img_temp)


if __name__ == '__main__':
    main()

