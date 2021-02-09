import torch

## Recal that for images the origin is in the top-left corner.
def intersection_over_union(box_preds, boxes_labels, box_formats="midpoint"):
    """
    Calculates the intersection over union

    Parameters:
            boxes_preds (tensor): Predictions of Bounding Boxes (BATCH_SIZE,4)
            boxes_labels (tensor): Correct labels of Bounding Boxes (BATCH_SIZE,4)
            box_formats (str): midpoint/corners (x,y,w,h) or (x1,y1,x2,y2)

        Returns:
            tensor: Intersection over Union for all examples
    """
   
    # Slice [..., 0:1] instead of [...,1] to 
    # keep the dimensions as is (N,1) instead of (N)
    
    # Format BB [x_mid, y_mid, width, height]
    if box_formats == "midpoint":
        box1_x1 = box_preds[..., 0:1] - box_preds[..., 2:3] /2
        box1_y1 = box_preds[..., 1:2] - box_preds[..., 3:4] /2
        box1_x2 = box_preds[..., 2:3] + box_preds[..., 2:3] /2
        box1_y2 = box_preds[..., 3:4] + box_preds[..., 3:4] /2
        box2_x1 = boxes_labels[..., 0:1] - boxes_labels[..., 2:3] /2
        box2_y1 = boxes_labels[..., 1:2] - boxes_labels[..., 3:4] /2
        box2_x2 = boxes_labels[..., 2:3] + boxes_labels[..., 2:3] /2
        box2_y2 = boxes_labels[..., 3:4] + boxes_labels[..., 3:4] /2
    # Format BB [x1, y1, x2, y2]
    elif box_formats == "corners":
        box1_x1 = box_preds[..., 0:1]
        box1_y1 = box_preds[..., 1:2]
        box1_x2 = box_preds[..., 2:3]
        box1_y2 = box_preds[..., 3:4]
        box2_x1 = boxes_labels[..., 0:1]
        box2_y1 = boxes_labels[..., 1:2]
        box2_x2 = boxes_labels[..., 2:3]
        box2_y2 = boxes_labels[..., 3:4]

    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)
    
    # .clamp(0) for the case when there is no intersection
    intersection = ( x2-x1 ).clamp(0) * ( y2-y1 ).clamp(0)

    box1_area = abs((box1_x2-box1_x1) * (box1_y1 - box1_y2))
    box2_area = abs((box2_x2-box2_x1) * (box2_y1 - box2_y2))

    return intersection / (box1_area + box2_area - intersection + 1e-6)