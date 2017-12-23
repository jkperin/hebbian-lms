import numpy as np
import random
from PIL import Image

import os
import csv
from os import listdir
from os.path import isfile, join

def crop_image(X_in, x, y):
    ''' Crop image
    Resulting image has dimensions (width - 2x) and (height - 2y)
    '''
    if x == 0 and y == 0:
        return X_in;
    
    X_crop = X_in[:, x:-x, y:-y, :];
    return X_crop;

def save_history(history, filename, verbose=True):       
    with open(filename, 'w') as csv_file:
        w = csv.writer(csv_file)
        for key in history.keys():
            if verbose:
                print(key, history[key]);
            
            w.writerow([key] + history[key])
    

def load_training_data():
    fids = open("tiny-imagenet-200/wnids.txt","r") 
    ids = fids.read().splitlines();
    fids.close();
    
    X_train = [];
    y_train = [];
    meanImage = np.zeros((64, 64, 3), dtype='float64');
    for i in range(len(ids)):
        path = "tiny-imagenet-200/train/"+ids[i]+"/images/";
        files = [f for f in listdir(path) if ".JPEG" in f]
        for f in files:
            imgObj = Image.open(path+f);
            imgObj.load();
            imgArray = np.asarray(imgObj, dtype="uint8");
            if imgArray.shape != (64, 64, 3):
                h, w = imgArray.shape;
                a = np.empty((h, w, 3), dtype="uint8")
                a[:, :, :] = imgArray[:, :, np.newaxis];
                imgArray = a;
                
            X_train.append(imgArray); 
            y_train.append(i);
            meanImage += imgArray;
    
    # Shuffle training data
    c = list(zip(X_train, y_train));
    random.shuffle(c)
    X_train, y_train = zip(*c)
    
    X_train = np.array(X_train, dtype="float64");
    y_train = np.array(y_train, dtype="uint8");
    meanImage /= len(y_train);
    X_train -= meanImage;
    X_train /= 128; # normalize 
    
    print("X_train shape:", X_train.shape)
    print("y_train shape:", y_train.shape)
        
    return X_train, y_train, meanImage, ids

def load_validation_data(ids, meanImage):
    filename = [];
    category = [];
    with open("tiny-imagenet-200/val/val_annotations.txt", 'r') as file:
        reader = csv.reader(file, delimiter='\t')
        for row in reader:
            filename.append(row[0]);
            category.append(row[1]);
            
    path = 'tiny-imagenet-200/val/images/';
    X_val = [];
    y_val = [];
    for f, i in zip(filename, category):
        imgObj = Image.open(path+f);
        imgObj.load();
        imgArray = np.asarray(imgObj, dtype="uint8");
        if imgArray.shape != (64, 64, 3):
            h, w = imgArray.shape;
            a = np.empty((h, w, 3), dtype="uint8")
            a[:, :, :] = imgArray[:, :, np.newaxis];
            imgArray = a;   
            
        X_val.append(imgArray); 
        y_val.append(ids.index(i));
        
    X_val = np.array(X_val, dtype="float64");
    y_val = np.array(y_val, dtype="uint8");
    X_val -= meanImage;
    X_val /= 128; # normalize 
    print("X_val shape:", X_val.shape)
    print("y_val shape:", y_val.shape)
    
    return X_val, y_val


def load_test_data(meanImage):    
    path = 'tiny-imagenet-200/test/images/';
    files = [f for f in listdir(path) if ".JPEG" in f]
    
    X_test = [];
    for f in files:
        imgObj = Image.open(path+f);
        imgObj.load();
        imgArray = np.asarray(imgObj, dtype="uint8");
        if imgArray.shape != (64, 64, 3):
            h, w = imgArray.shape;
            a = np.empty((h, w, 3), dtype="uint8")
            a[:, :, :] = imgArray[:, :, np.newaxis];
            imgArray = a;   
            
        X_test.append(imgArray); 
        
    X_test = np.array(X_test, dtype="float64");
    X_test -= meanImage;
    X_test /= 128; # normalize 
    print("X_test shape:", X_test.shape)
    
    return X_test, files


