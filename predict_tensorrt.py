import tensorrt as trt
from utils import *
import cv2
import pycuda.gpuarray as gpuarray
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import json


class YOLOTRT(object):
    
    def __init__(self, engine_path, labels, anchors):
        logger = trt.Logger()
        runtime = trt.Runtime(logger)
        with open(engine_path, 'rb') as f:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        
        self.input_size = self.engine.get_binding_shape(0)[2]
        self.nb_class = len(labels)
        self.labels = labels
        self.anchors = anchors
        
    def normalize(self, image):
        raise NotImplementedError
        
    def predict(self, image):
        image_h, image_w, _ = image.shape
        image = cv2.resize(image, (self.input_size, self.input_size))
        image = self.normalize(image)

        input_image = image[:,:,::-1] # bgr -> rgb
        input_image = np.expand_dims(input_image, 0)
        output_shape = tuple(self.engine.get_binding_shape(1))
        output_gpu = gpuarray.empty(output_shape, np.float32)
        input_image_gpu = gpuarray.to_gpu(input_image.astype(np.float32))

        self.context.execute(1, [int(input_image_gpu.gpudata), int(output_gpu.gpudata)])
        netout = output_gpu.get()[0]
        boxes  = decode_netout(netout, self.anchors, self.nb_class)

        return boxes
        

class MobileNetYOLOTRT(YOLOTRT):
    
    def normalize(self, image):
        image = image / 255.
        image = image - 0.5
        image = image * 2.
        return image
    
    
if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        '-e',
        '--engine',
        help='path to tensorrt engine')
    
    parser.add_argument(
        '-c',
        '--conf',
        help='path to configuration file')

    parser.add_argument(
        '-i',
        '--input',
        help='path to an image or an video (mp4 format)')
    
    args = parser.parse_args()
    
    with open(args.conf) as config_buffer:
        config = json.load(config_buffer)
        
    yolo = MobileNetYOLOTRT(
        args.engine, 
        config['model']['labels'], 
        config['model']['anchors']
    )
    
    image = cv2.imread(args.input)
    boxes = yolo.predict(image)
    print(boxes)
    image = draw_boxes(image, boxes, config['model']['labels'])

    print(len(boxes), 'boxes are found')

    image_path = args.input
    cv2.imwrite(image_path[:-4] + '_detected' + image_path[-4:], image)