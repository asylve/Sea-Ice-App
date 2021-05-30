# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""


#if __name__ == "__main__":
#    uvicorn.run(app, host="0.0.0.0", port=8000)

from icepredictor import get_images, predict_mask, display
import uvicorn
from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory='templates')

@app.get("/", response_class=HTMLResponse)
def form_get(request: Request):
    img_name = "default"
    long1 = -103.991
    lat1 = 68.520
    dateStart = '2020-07-22'
    return templates.TemplateResponse('index.html', context={'request': request, 'img_name': img_name,
                                                              'long1_f': long1, 'lat1_f': lat1, 
                                                              'imgDate_f':'2020-07-23 18:55:05', 'dateStart_f': dateStart})

@app.post("/predict", response_class=HTMLResponse)
def form_post(request: Request, long1: float = Form(...), lat1: float = Form(...), 
              dateStart: str = Form(...)):
    img_name = "download"
    
    try:
        imgs, imgDates = get_images(longCenter = long1, latCenter = lat1, time_start = dateStart)
        imgDate = imgDates[0]
        masks = predict_mask(imgs)
        display([imgs[0], masks[0]])
    except:
        error_code='no_imgs'
        imgDate = 'no_imgs'
        print('error')
    else:
        error_code='OK'
        print('success')
    
    return templates.TemplateResponse('index.html', context={'request': request, 'img_name': img_name, 
                                                             'long1_f': long1, 'lat1_f': lat1, 
                                                             'dateStart_f': dateStart, 'imgDate_f':imgDate,
                                                             'error_code_f':error_code})


