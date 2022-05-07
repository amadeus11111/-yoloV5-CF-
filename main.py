from grabscreen import grab_screen
import cv2
import win32gui
import win32con
from CF_model import load_model
import torch
from utils.general import non_max_suppression , scale_coords , xyxy2xywh
from utils.augmentations import letterbox
import numpy as np
import pynput
from mouse_control import lock
from pynput import mouse

x , y = (3072 , 1920)
re_x , re_y = (3072 , 1920)
model = load_model()
device = 'cuda' if torch.cuda .is_available() else 'cpu'
half = device != 'cpu'
stride, names, pt = model.stride, model.names, model.pt

conf_thres = 0.4
iou_thres = 0.05
img_size = 640
agnostic_nms =False

mouse = pynput.mouse.Controller()

lock_mode = False

with pynput.mouse.Events() as events:
    while True:
        it = next(events)
        while it is not None and not isinstance(it , pynput.mouse.Events.Click):
            it = next(events)
        if it is not None and it.button == it.button.x2 and it.pressed:
            lock_mode = not lock_mode
            print('lock mode' , 'on' if lock_mode else 'off')

        img0 = grab_screen(region=(0 , 0 , x , y))
        img0 = cv2.resize(img0 , (re_x , re_y))

        img = letterbox(img0, img_size, stride=stride)[0]
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)

        im = torch.from_numpy(img).to('cpu')
        im = im.half() if half else im.float()
        im /= 255.
        if len(im.shape) == 3:
            im = im[None]
        pred = model(im, augment=False, visualize=False)
        pred = non_max_suppression(pred, conf_thres, iou_thres,  agnostic_nms)
        aims = []
        for i, det in enumerate(pred):
            s = ''
            s += '%gx%g ' % im.shape[2:]  # print string
            gn = torch.tensor(img0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], img0.shape).round()
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                for *xyxy, conf, cls in reversed(det):
                    # Write to file
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    line = cls, *xywh  # label format
                    aim = ('%g ' * len(line)).rstrip() % line
                    aim = aim.split(' ')
                    aims.append(aim)


            if len(aims):
                if lock_mode:
                    lock(aims , mouse , x , y)

                for i , det in enumerate(aims):
                    _ , x_center , y_center , width , height = det
                    x_center , width = re_x * float(x_center) , re_x * float(width)
                    y_center , height = re_y * float(y_center) , re_y * float(height)
                    top_left = (int(x_center - width / 2.) , int(y_center - height / 2.))
                    bottom_right = (int(x_center + width / 2 ), int(y_center + height / 2))
                    color = (0 , 255 , 0)
                    cv2.rectangle(img0 , top_left , bottom_right , color , 3)






        cv2.namedWindow('CF-detect' , cv2.WINDOW_NORMAL)
        cv2.resizeWindow('CF-detect' , re_x //3 , re_y // 3 )
        cv2.imshow('CF-detect' , img0)

        hwnd = win32gui.FindWindow(None,'CF-detect')
        CVRECT = cv2.getWindowImageRect('CF-detect')
        win32gui.SetWindowPos(hwnd , win32con.HWND_TOPMOST , 0 ,0 ,0 ,0 , win32con.SWP_NOMOVE | win32con.SWP_NOSIZE)



        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
