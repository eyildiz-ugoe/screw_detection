from __future__ import print_function
import argparse
import os,sys
import numpy as np
    

def voc_ap(rec, prec, use_07_metric=False):
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))
	
        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    #print (rec)
    return ap

def voc_eval(det_path,
             groundtruth_path,
             ovthresh=0.5,
             use_07_metric=False):

    #read groundtruth
    #groundtruth format: image+path xmin,ymin,xmax,ymax,0 xmin,ymin,xmax,ymax,0 .....
    # 1 line for 1 image
    npos = 0
    class_recs = {}
    with open(groundtruth_path, 'r') as groundtruth_file:
      while True:
        line = groundtruth_file.readline()
        if len(line.split(','))<4:
           break
        line_split = line.strip().split('.png ')
        image_id = line_split[0].split('/')[-1]
        boxes = line_split[1].split(' ')
        bbox = []
        for box in boxes:
            xmin, ymin, xmax, ymax, xx = box.split(',')
            bbox.append([int(xmin), int(ymin), int(xmax), int(ymax)])
        bbox = np.array([x for x in bbox])
        difficult = np.array([0 for x in bbox]).astype(np.bool)
        det = [False] * len(bbox)
        npos = npos + sum(~difficult)
        class_recs[image_id] = {'bbox': bbox,
                                 'difficult': difficult,
                                 'det': det}
    # read dets
    #detfile format: image_id score xmin ymin xmax ymax
    # 1 line for 1 bbox
    with open(det_path, 'r') as f:
        lines = f.readlines()

    splitlines = [x.strip().split(' ') for x in lines]
    image_ids = [x[0] for x in splitlines]
    confidence = np.array([float(x[1]) for x in splitlines])
    BB = np.array([[float(z) for z in x[2:]] for x in splitlines])

    # sort by confidence
    sorted_ind = np.argsort(-confidence)
    sorted_scores = np.sort(-confidence)
    BB = BB[sorted_ind, :]
    image_ids = [image_ids[x] for x in sorted_ind]

    # go down dets and mark TPs and FPs
    nd = len(image_ids)
    tp = np.zeros(nd)
    fp = np.zeros(nd)
    for d in range(nd):
        R = class_recs[image_ids[d]]
        bb = BB[d, :].astype(float)
        ovmax = -np.inf
        BBGT = R['bbox'].astype(float)

        if BBGT.size > 0:
            # compute overlaps
            # intersection
            ixmin = np.maximum(BBGT[:, 0], bb[0])
            iymin = np.maximum(BBGT[:, 1], bb[1])
            ixmax = np.minimum(BBGT[:, 2], bb[2])
            iymax = np.minimum(BBGT[:, 3], bb[3])
            iw = np.maximum(ixmax - ixmin + 1., 0.)
            ih = np.maximum(iymax - iymin + 1., 0.)
            inters = iw * ih

            # union
            uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
                   (BBGT[:, 2] - BBGT[:, 0] + 1.) *
                   (BBGT[:, 3] - BBGT[:, 1] + 1.) - inters)

            overlaps = inters / uni
            ovmax = np.max(overlaps)
            jmax = np.argmax(overlaps)

        if ovmax > ovthresh:
            if not R['difficult'][jmax]:
                if not R['det'][jmax]:
                    tp[d] = 1.
                    R['det'][jmax] = 1
                else:
                    fp[d] = 1.
        else:
            fp[d] = 1.

    # compute precision recall
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / float(npos)
    # avoid divide by zero in case the first detection matches a difficult
    # ground truth
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = voc_ap(rec, prec, use_07_metric)

    return rec, prec, ap
    
def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='mAP Calculation')

    parser.add_argument('--det_path', dest='det_path', help='The data path', default='det_2keras.txt', type=str)
    parser.add_argument('--gt_path', dest='gt_path', help='The data path', default='gt_test.txt', type=str)
    args = parser.parse_args()

    return args

import matplotlib.pyplot as plt

if __name__ == '__main__':
    args = parse_args()
    use_07_metric = False
    print('VOC07 metric? ' + ('Yes' if use_07_metric else 'No'))
    rec, prec, ap = voc_eval(args.det_path, args.gt_path, ovthresh=0.5, use_07_metric=use_07_metric)
    print('AP = {:.4f}'.format( ap))
    plt.plot(rec,prec)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision Recall curve')
    plt.show()
