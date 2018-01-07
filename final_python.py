#!/usr/bin/env python
import numpy as np
from template_0.slid_win import sliding_windows
from process import shi_tomasi
from refinement import refinement_axial

def template2(img):
    return shi_tomasi(img, maxCorners=10, qualityLevel=0.25)

def god_function(list_axial, list_coronal, list_sagittal): 
    length = len(list_axial)

    corners = []
    for z in range(63, length):
        if z in range(90, 111):
            continue
        shi = template2(list_axial[z])
        corners.extend([list(corn) + [z] for corn in shi])

    raw = refinement_axial(corners, list_axial.shape[::-1], mode='soft')
    print("Clustering...")
    clust = DBSCAN(eps=100, leaf_size=4, min_samples=1)
    predictions = clust.fit_predict(raw)
    labels = set(predictions)
    final = []
    for label in list(labels):
        centroid = [0, 0, 0]
        count = 0
        for i in range(len(raw)):
            if predictions[i] == label:
                count += 1
                centroid[0] += raw[i][0]
                centroid[1] += raw[i][1]
                centroid[2] += raw[i][2]
        centroid[0] /= count
        centroid[1] /= count
        centroid[2] /= count
        final.append(centroid)
    return final

