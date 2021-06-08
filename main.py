# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""


#if __name__ == "__main__":
#    uvicorn.run(app, host="0.0.0.0", port=8000)

from icepredictor import get_images, predict_mask, display
import numpy as np
import uvicorn
from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import gc
import tensorflow as tf

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

@app.get("/", response_class=HTMLResponse)
def form_get(request: Request):
    img_name = "default"
    long1 = -103.991
    lat1 = 68.520
    dateStart = '2020-07-22'
    return templates.TemplateResponse('index.html', context={'request': request, 'img_name': img_name,
                                                              'long1_f': long1, 'lat1_f': lat1, 
                                                              'imgDate_f':'2020-07-23 18:55:05', 'dateStart_f': dateStart})

@app.post("/", response_class=HTMLResponse)
def form_post(request: Request, long1: float = Form(...), lat1: float = Form(...), 
              dateStart: str = Form(...)):
    img_name = "download"
    tf.random.set_seed(1) #used to clearn kernel cache
    gc.collect()
    
    try:
        img, imgDate = get_images(longCenter = long1, latCenter = lat1, time_start = dateStart)
        imgs = np.expand_dims(img, axis=0)#predict mask expects an array of images so add an additional dimension
        mask = predict_mask(imgs)[0]
        display([img, mask])
    except Exception as e:
        error_code= 'no_imgs'
        imgDate = 'no_imgs'
        print(e)
    else:
        error_code='OK'
        print('success')
    
    return templates.TemplateResponse('index.html', context={'request': request, 'img_name': img_name, 
                                                             'long1_f': long1, 'lat1_f': lat1, 
                                                             'dateStart_f': dateStart, 'imgDate_f':imgDate,
                                                             'error_code_f':error_code})


