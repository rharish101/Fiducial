#!/usr/bin/env python
import os
from process import *

folder = './2016.06.27 PVC Skull Model/Sequential Scan/DICOM/PA1/ST1/SE5/'
images = []
for img in os.listdir(folder):
    images.append(import_dicom(folder + img))
