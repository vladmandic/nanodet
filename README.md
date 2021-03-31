# NanoDet: Tiny Object Detection for TFJS and NodeJS

Models included in `/model-tfjs-graph-*` were converted to TFJS Graph model format from the original repository  
Models descriptors have been additionally parsed for readability

Actual model parsing implementation in `nanodet.js` does not follow original  
and is fully custom and optimized for JavaScript execution

Original model is internally using Int64 values, but TFJS does not support Int64 so there are some overflows due to Int32 casting,  
Most commonly around class 62, so that one is excluded from results  

Note that `NanoDet-G` variation is about 4x faster in Browser execution using `WebGL` backend than `NanoDet-M` variation  

<br><hr><br>

## Conversion Notes

Source: <https://github.com/RangiLyu/nanodet>  

### Requirements:

```shell
pip install torch onnx onnx-tf onnx-simplifier tensorflowjs
```

### Fixes:

- Error during conversion: `pytorch_half_pixel`  
  Edit `export.py` to `set opset_version=10` which forces onnx export to use older upscale instead of resize op

### Conversion:

- From PyTorch to ONNX to TensorFlow Saved model to TensorFlow/JS Graph model

```shell
python export.py --cfg_path config/nanodet-m-416.yml --model_path models/nanodet_m_416.pth --out_path models/nanodet_m_416.onnx
python -m onnxsim models/nanodet_m_416.onnx models/nanodet_m_416-simplified.onnx
onnx-tf convert --infile models/nanodet_m_416-simplified.onnx --outdir models/saved-m
tensorflowjs_converter --input_format tf_saved_model --output_format tfjs_graph_model --strip_debug_ops=* --weight_shard_size_bytes 8388608 models/saved-m models/graph-m
```

<br><hr><br>

## Tests

```shell
node nanodet.js car.jpg
```

```js
2021-03-16 12:03:39 INFO:  detector version 0.0.1
2021-03-16 12:03:39 INFO:  User: vlado Platform: linux Arch: x64 Node: v15.4.0
2021-03-16 12:03:39 INFO:  Loaded model { modelPath: 'file://models/nanodet/nanodet.json', minScore: 0.15, iouThreshold: 0.1, maxResults: 10, scaleBox: 2.5 } tensors: 524 bytes: 3771112
2021-03-16 12:03:39 INFO:  Loaded image: car.jpg inputShape: [ 2000, 1333, [length]: 2 ] outputShape: [ 1, 3, 416, 416, [length]: 4 ]
2021-03-16 12:03:39 DATA:  Results: [
  {
    score: 0.7859958410263062,
    strideSize: 1,
    class: 3,
    label: 'car',
    center: [ 1000, 1076, [length]: 2 ],
    centerRaw: [ 0.5, 0.8076923076923077, [length]: 2 ],
    box: [ 375, 868, 1625, 1284, [length]: 4 ],
    boxRaw: [ 0.1875, 0.6514423076923077, 0.8125, 0.9639423076923077, [length]: 4 ]
  },
  {
    score: 0.20603930950164795,
    strideSize: 1,
    class: 26,
    label: 'umbrella',
    center: [ 1615, 358, [length]: 2 ],
    centerRaw: [ 0.8076923076923077, 0.2692307692307692, [length]: 2 ],
    box: [ 1302, -57, 1927, 983, [length]: 4 ],
    boxRaw: [ 0.6514423076923077, -0.04326923076923078, 0.9639423076923077, 0.7379807692307692, [length]: 4 ]
  },
  {
    score: 0.16496318578720093,
    strideSize: 4,
    class: 59,
    label: 'potted plant',
    center: [ 865, 858, [length]: 2 ],
    centerRaw: [ 0.4326923076923077, 0.6442307692307693, [length]: 2 ],
    box: [ 748, 754, 943, 910, [length]: 4 ],
    boxRaw: [ 0.3740985576923077, 0.5661057692307693, 0.4717548076923077, 0.6832932692307693, [length]: 4 ]
  },
  {
    score: 0.15522807836532593,
    strideSize: 4,
    class: 14,
    label: 'bench',
    center: [ 1557, 858, [length]: 2 ],
    centerRaw: [ 0.7788461538461539, 0.6442307692307693, [length]: 2 ],
    box: [ 1362, 832, 1753, 884, [length]: 4 ],
    boxRaw: [ 0.6811899038461539, 0.6246995192307693, 0.8765024038461539, 0.6637620192307693, [length]: 4 ]
  },
]
2021-03-16 12:03:39 STATE:  Created output image: car-nanodet.jpg]
```

<br><hr><br>

## Notes

- BoxRaw and CenterRaw are normalized to range 0..1
- Box and Center are normalized to input image size in pixels

What's different about this model is that we get 3 different resultsets
and need to check each as different strides pick up different sized objects
