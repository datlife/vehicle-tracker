### YOLO V2
-----------

This library is based on thtrieu repo
```shell
https://github.com/thtrieu/darkflow
```

### Usage
```shell
import cv2
from yolo.net.build import TFNet

options = {"model": "cfg/yolo.cfg", "load": "bin/yolo.weights", "threshold": 0.1}

tfnet = TFNet(options)

imgcv = cv2.imread("./test/test.jpg")
result = tfnet.return_predict(imgcv)
print(result)

```
### Download Pre-trained weight files
```shell
# Tiny YOLO
http://pjreddie.com/media/files/tiny-yolo-voc.weights

# YOLO V2
http://pjreddie.com/media/files/yolo-voc.weights
```
