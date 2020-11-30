import keras2onnx
import argparse
import json
from frontend import YOLO


parser = argparse.ArgumentParser()
parser.add_argument(
    '-c',
    '--conf',
    help='path to configuration file')
parser.add_argument(
    '-w',
    '--weights',
    default='',
    help='path to pretrained weights')
parser.add_argument(
    '-o',
    '--output',
    help='path to output ONNX model'
)
args = parser.parse_args()

config_path = args.conf

with open(args.conf) as f:    
    config = json.load(f)

yolo = YOLO(
    backend=config['model']['backend'],
    input_size=config['model']['input_size'], 
    labels=config['model']['labels'], 
    max_box_per_image=config['model']['max_box_per_image'],
    anchors=config['model']['anchors']
)

if args.weights:
    yolo.load_weights(args.weights)
    
# export onnx
model_onnx = keras2onnx.convert_keras(yolo.model)

with open(args.output, 'wb') as f:
    f.write(model_onnx.SerializeToString())
