#!/usr/bin/env python
'''
Author: Sahil Badyal
This module contains the functions to view the  imagenet VRE dataset
'''
import json
import cv2
import numpy as np
import os
import sys

def showImage(x):
    '''
    Shows the image and waits for the user input infinitely
    '''
    cv2.imshow("image", x)
    if cv2.waitKey(0)==ord('e'):
        sys.exit()
    #closing all open windows
    cv2.destroyAllWindows()

def createBoundingBox(img , bb, color = (0, 255, 0)):
    '''
    Creates a green bounding box
    '''
    print(bb)
    start = (int(bb[0]), int(bb[1]))
    end = (int(bb[2]), int(bb[3]))
    print (start, end, color)
    cv2.rectangle(img, start, end, color, 2)


inp_dir = './light_test/full_images/'
input_exp_dir = './final_results_on_vrep/'
#input_exp_dir = './final_results_on_refcoco+_on_vrep/'
pred_file = 'results.json'
test_json = 'final_refer_testset.json'
path  = os.path.join(input_exp_dir, test_json)
with open(path, 'r') as f:
    dataset = json.load(f)
path  = os.path.join(input_exp_dir, pred_file)
with open(path, 'r') as f:
    preds = json.load(f)
count = 1
for i,datum in enumerate(preds):
    ref_id = str(datum['ref_id'])
    pred_bb = datum['box']
    if int(pred_bb[0]) == -2 or int(pred_bb[0]) == 0:
        count +=1
        continue
    print(count)
    gt_g = dataset[ref_id][1]
    gt_bb = [gt_g[0], gt_g[1], gt_g[0] + gt_g[2], gt_g[1] + gt_g[3]]
    image = dataset[ref_id][0]
    reference = dataset[ref_id][2]
    x = cv2.imread(os.path.join(inp_dir+image))
    createBoundingBox(x, gt_bb) ## Green
    createBoundingBox(x, pred_bb, (0,0,255)) ## Red is prediction
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(x, reference, (10,40), font, 1, (0, 0, 255), 3)
    #showImage(x)
    cv2.imwrite(f"{image}_{ref_id}.jpg", x)
