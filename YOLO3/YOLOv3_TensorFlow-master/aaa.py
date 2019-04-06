
import numpy as  np
from collections import Counter
import numpy as np
import tensorflow as tf
import  cv2 as cv
box_sizes = np.array([[10,10],[20,20]])
box_sizes = np.expand_dims(box_sizes, 1)
print(box_sizes.shape)
anchors = np.array([[13,13],[16,16],[14,14],[40,40]])
mins = np.maximum(- box_sizes / 2, - anchors / 2)  # anchors=  9个值
maxs = np.minimum(box_sizes / 2, anchors / 2)
whs = maxs - mins
print(mins)
print(maxs)
print(whs)