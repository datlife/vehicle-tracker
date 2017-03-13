import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob


def find_matches(img, template_list):
    # Make a copy of the image to draw on
    # Define an empty list to take bbox coords
    bbox_list = []
    # Iterate through template list
    for template in template_list:
        # Use cv2.matchTemplate() to search the image using whichever of the OpenCV search methods you prefer
        # Other options include: cv2.TM_CCORR_NORMED', 'cv2.TM_CCOEFF', 'cv2.TM_CCORR',
        #         'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED'
        matches = cv2.matchTemplate(img, template, method=cv2.TM_CCOEFF_NORMED)
        # Use cv2.minMaxLoc() to extract the location of the best match
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(matches)
        # Determine a bounding box for the match
        w, h = (template.shape[1], template.shape[0])
        if cv2.TM_CCOEFF_NORMED in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
            top_left = min_loc
        else:
            top_left = max_loc
        bottom_right = (top_left[0] + w, top_left[1] + h)
        # Append bbox position to list
        bbox_list.append((top_left, bottom_right))
    # Determine bounding box corners for the match
    # Return the list of bounding boxes
    return bbox_list


def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    # Make a copy of the image
    draw_img = np.copy(img)
    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(draw_img, bbox[0], bbox[1], color, thick)
    # Return the image copy with boxes drawn
    return draw_img


image = mpimg.imread('bbox-example-image.jpg')
img_path = glob.glob('cutouts/cutout*.jpg')
templates = [cv2.imread(i) for i in img_path]
bboxes = find_matches(image, templates)
result = draw_boxes(image, bboxes)
plt.imshow(result)
plt.show()