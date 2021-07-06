from icepredictor import get_images, predict_mask, display, get_model
import numpy as np
from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import gc
import tensorflow as tf
import os, psutil

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory='templates')

#Define IoU metric as this is information is not stored in the saved model (by stack overflow user HuckleberryFinn)
class UpdatedMeanIoU(tf.keras.metrics.MeanIoU):
    def __init__(self,
               y_true=None,
               y_pred=None,
               num_classes=None,
               name=None,
               dtype=None):
        super(UpdatedMeanIoU, self).__init__(num_classes = num_classes,name=name, dtype=dtype)

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.math.argmax(y_pred, axis=-1)
        return super().update_state(y_true, y_pred, sample_weight)

model = get_model()

@app.get("/", response_class=HTMLResponse)
def form_get(request: Request):
    img_name = "default"
    long1 = -70.63
    lat1 = 62.18
    dateStart = '2021-06-22'
    return templates.TemplateResponse('index.html', context={'request': request, 'img_name': img_name,
                                                              'long1_f': long1, 'lat1_f': lat1, 
                                                              'imgDate_f':'2021-06-23 11:15:47', 'dateStart_f': dateStart})

@app.post("/", response_class=HTMLResponse)
def form_post(request: Request, long1: float = Form(...), lat1: float = Form(...), 
              dateStart: str = Form(...)):
    img_name = "download"
    tf.random.set_seed(1) #used to clearn kernel cache
    gc.collect()
    
    try:
        print(psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2)#print memory consumption of python
        img, imgDate = get_images(longCenter = long1, latCenter = lat1, time_start = dateStart)
        imgs = np.expand_dims(img, axis=0)#predict mask expects an array of images so add an additional dimension
        mask = predict_mask(imgs, model)[0]
        display([img, mask])
        
        imgDate = str(imgDate)[:-6]#remove the trailing '+00:00' from the date string
    except Exception as e:
        error_code= 'No Images'
        imgDate = 'No Images'
        img_name = "none"
        print(e)
    else:
        error_code='OK'
        print('success')
    
    return templates.TemplateResponse('index.html', context={'request': request, 'img_name': img_name, 
                                                             'long1_f': long1, 'lat1_f': lat1, 
                                                             'dateStart_f': dateStart, 'imgDate_f':imgDate,
                                                             'error_code_f':error_code})


