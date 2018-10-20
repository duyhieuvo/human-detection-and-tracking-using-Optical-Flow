import numpy as np
import cv2


#Function to capture the image to be used as background when displaying the result
def takeImage(cam):
    (grabbed, frame) = cam.read()
    return frame


# def getCenter(bbox):
#     x = (bbox[0] + bbox[2])/2
#     y = (bbox[1] + bbox[3])/2
#     return np.array([[np.float32(x)],[np.float32(y)]])

# def predict(old_pos,velocity):
#     new_pos = old_pos + velocity
#     return new_pos

# def getVelocity(p_old,p_new):
#     vec = p_new - p_old
#     return vec

# def getFeature(bbox):
#     area = (bbox[2]*bbox[3])
#     peri = (bbox[2]+bbox[3])*2
#     ratio = round(((float(bbox[2]))/(float(bbox[3]))),2) #x/y
#
#     return np.hstack((np.array([area,peri,ratio])))
#
# def Comparison(curfeature,feature):
#     ratios = np.divide(np.float32(curfeature),np.float32(feature))
#     indices = np.where(ratios>1)
#     ratios[indices]=1/ratios[indices]
#     comp = (np.sum(ratios)/3)**2
#     return comp

def overlapArea(bbox1,bbox2):
    xx1 = np.maximum(bbox1[0],bbox2[0])
    yy1 = np.maximum(bbox1[1],bbox2[1])
    xx2 = np.minimum(bbox1[2],bbox2[2])
    yy2 = np.minimum(bbox1[3],bbox2[3])

    w = np.maximum(0, xx2 - xx1 + 1)
    h = np.maximum(0, yy2 - yy1 + 1)
    area = (bbox2[2]-bbox2[0]+1)*(bbox2[3]-bbox2[1]+1)
    return w*h/area


# Malisiewicz et al.
# http://www.computervisionblog.com/2011/08/blazing-fast-nmsm-from-exemplar-svm.html
def non_max_suppression_fast(boxes, overlapThresh):
    if len(boxes) == 0:
        return []
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")
    pick = []

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)

    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        overlap = (w * h) / area[idxs[:last]]

        idxs = np.delete(idxs, np.concatenate(([last],
                                               np.where(overlap > overlapThresh)[0])))

    return boxes[pick].astype("int")