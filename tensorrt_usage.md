There are a few changes to keras-yolo2 to address tensorrt compatibility:

1. For Full Yolo model, the space_to_depth_x2 is re-implemented using primitive methods that are supported with TensorRT.  This doesn't apply to other backbones.
2. Tensorflow isn't actually used by the ``utils`` module, so we remove the import so we can use the post-processing functions without loading tensorflow

> Please note, currently I only implemented the pre-processing for MobileNet.  It is done in ``predict_tensorrt.py``.  It should be straightforward to implement the TensorRT wrapper for other backbones by referencing ``backend.py``.  

## Usage

1. Checkout tensorrt branch

    ```bash
    git clone https://github.com/jaybdub/keras-yolo2
    cd keras-yolo2
    git checkout tensorrt
    ```
    
2. Ensure ``config.json`` and model weights are in ``keras-yolo2/``.  We've tested MobileNet
3. Export trained model to ONNX.  

> Please note, for this step only, I used TensorFlow 2.x for compatibility with keras2onnx.  For all other steps, I used TensorFlow 1.x, since this seemed compatible with keras-yolo2 repository.
   
    ```bash
    python3 export_onnx.py -c config.json -w model_weights.h5 -o model.onnx
    ```
    
4. Optimize model using trtexec

    ```bash
    trtexec --onnx=model.onnx --explicitBatch --saveEngine=model.engine --fp16 # other TensorRT parameters as desired...
    ```
5. Run inference using TensorRT Python API

    ```bash
    python3 predict_tensorrt.py -c config.json -e model.engine -i test_image.jpg
    ```
    
Please reference ``predict_tensorrt.py`` for details on how to execute using TensorRT.  