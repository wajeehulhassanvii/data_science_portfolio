from deeplearningai.course4.week3.e1.h00_utils import *


def iou(box1, box2):
    """Implement the intersection over union (IoU) between box1 and box2
    
    Arguments:
    box1 -- first box, list object with coordinates (x1, y1, x2, y2)
    box2 -- second box, list object with coordinates (x1, y1, x2, y2)
    """

    # Calculate the (y1, x1, y2, x2) coordinates of the intersection of box1 and box2. Calculate its Area.
    ### START CODE HERE ### (≈ 5 lines)
    xi1 = max(box1[0] ,box2[0])
    yi1 = max(box1[1] ,box2[1])
    xi2 = min(box1[2] ,box2[2])
    yi2 = min(box1[3] ,box2[3])
    inter_area = max(( xi2 -xi1) ,0 ) *max(( yi2 -yi1) ,0)
    ### END CODE HERE ###    

    # Calculate the Union area by using Formula: Union(A,B) = A + B - Inter(A,B)
    ### START CODE HERE ### (≈ 3 lines)
    box1_area = (box1[2 ] -box1[0] ) *(box1[3 ] -box1[1])
    box2_area = (box2[2 ] -box2[0] ) *(box2[3 ] -box2[1])
    union_area = box1_area + box2_area - inter_area
    ### END CODE HERE ###

    # compute the IoU
    ### START CODE HERE ### (≈ 1 line)
    iou = inter_area/ union_area
    ### END CODE HERE ###

    return iou


box1 = (2, 1, 4, 3)
box2 = (1, 2, 3, 4)
print("iou = " + str(iou(box1, box2)))
