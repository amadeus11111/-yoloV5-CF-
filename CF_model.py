import torch
from models.common import DetectMultiBackend
from utils.torch_utils import select_device
from utils.general import check_img_size

weights = 'C:/Users/Mr.Q/PycharmProjects/cf/cs/yolov5-master/runs/train/exp5/weights/best.pt'
data = ''
bs = 1

def load_model():
    imgsz = (640, 640)
    device = ''
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=False, data=data, fp16=False)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size
    model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))

    return model