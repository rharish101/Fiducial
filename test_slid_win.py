from dicom.errors import InvalidDicomError
from process import import_dicom, display
from template_0.slid_win import sliding_windows
from tkinter import TclError

def display_slid_win(img_path, template=0):
    try:
        img = import_dicom(img_path)
    except InvalidDicomError:
        img = cv2.imread(args.image, 0)
    if img is None:
        raise Exception('Image is of an unsupported type')

    assert template in (0, 1)
    model = 'template_' + str(template) + '/template_' + str(template) +\
            '_detect.h5'
    try:
        for count, img in enumerate(sliding_windows(img, model, (1, 50, 50, 1),
                                                    1):
            display(img, title=count, pause=1)
    except (KeyboardInterrupt, TclError):
        pass
    finally:
        display.blank = True
