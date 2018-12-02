#Final Recognition
from darkflow.net.build import TFNet
from classify.FInal_Model import predict_pic
from classify.FInal_Model import Classify_train
import cv2 as cv
import os



options = {"model": "darkflow/cfg/tiny-yolo-voc.cfg","load": "darkflow/bin/tiny-yolo-voc.weights", "threshold": 0.1}



def get_picture(load_path,store_path):
    tfnet = TFNet(options)
    path = os.getcwd()
    imgcv = cv.imread(load_path)
    results = tfnet.return_predict(imgcv)
    for result in results:
        if (result['label']=='person'):
            features = result
            if (features['confidence']>=0.5):
                x_begin = features['topleft']['x']
                y_begin = features['topleft']['y']
                x_end =  features['bottomright']['x']
                y_end = features['bottomright']['y']
                img = imgcv[y_begin:y_end,x_begin:x_end]
                img_process = cv.resize(img,(48,48))
                cv.imwrite(store_path,img_process)



def ALL_WORK():
    now_path = os.path.abspath(os.getcwd())
    load_path = "darkflow/preview.png"
    store_pic_path = 'classify/1.jpg'
    store_path = 'classify'
    get_picture(load_path,store_pic_path)
    os.chdir(os.path.join(now_path,store_path))
    result = predict_pic(os.path.join(now_path,store_pic_path))
    os.chdir(now_path)
    print(result)

if __name__ == "__main__":
    ALL_WORK()

"""

[{'label': 'chair', 'confidence': 0.10067865, 'topleft': {'x': 421, 'y': 100}, 
'bottomright': {'x': 461, 'y': 155}}, {'label': 'person', 'confidence': 0.12787649, 
'topleft': {'x': 305, 'y': 84}, 'bottomright': {'x': 337, 'y': 133}}, 
{'label': 'person', 'confidence': 0.116214745, 'topleft': {'x': 339, 'y': 85}, 'bottomright': 
{'x': 368, 'y': 123}}, {'label': 'person', 'confidence': 0.515861, 'topleft': {'x': 170, 'y': 117}, 
'bottomright': {'x': 211, 'y': 186}}, {'label': 'person', 'confidence': 0.16828239, 'topleft': 
{'x': 155, 'y': 114}, 'bottomright': {'x': 219, 'y': 232}}, {'label': 'sofa', 'confidence': 0.43484208, 
'topleft': {'x': 220, 'y': 157}, 'bottomright': {'x': 497, 'y': 295}}]
"""
