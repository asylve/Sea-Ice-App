# -*- coding: utf-8 -*-
"""
Created on Sun May 30 14:28:10 2021

@author: DELL
"""
    
from fastapi import FastAPI

app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}