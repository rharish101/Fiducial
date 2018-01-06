from dicom.errors import InvalidDicomError
from process import import_dicom, display
from template_0.slid_win import sliding_windows
from tkinter import TclError
import cv2

def display_slid_win(img_path, template=0, verbose=True, **kwargs):
    try:
        img = import_dicom(img_path)
    except InvalidDicomError:
        img = cv2.imread(img_path, 0)
    if img is None:
        raise Exception('Image is of an unsupported type')

    assert template in (0, 1)
    model = 'template_' + str(template) + '/template_' + str(template) +\
            '_detect.h5'
    if template == 0:
        shape = (1, 50, 50, 1)
    else:
        shape = (1, 40, 40, 1)
    try:
        for count, img in enumerate(sliding_windows(img, model, shape,
                0, return_images=True, verbose=verbose, **kwargs)):
            display(img, title=count, pause=1)
    except TclError:
        pass
    finally:
        display.blank = True
