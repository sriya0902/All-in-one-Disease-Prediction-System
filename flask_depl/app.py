# -*- coding: utf-8 -*-
"""
Created on Wed Nov  2 12:58:20 2022

@author: Kkira
"""

from flask import Flask, render_template, request, url_for
app = Flask(__name__)

import pickle
import numpy as np
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

#Model Loading

model = pickle.load(open('cancer.pkl','rb'))
ckd_model = pickle.load(open('ckd.pkl','rb'))
liver_model = pickle.load(open('liver.pkl','rb'))
diabetes_model = pickle.load(open('diabetes.pkl','rb'))
arr_model = load_model('ECG.h5')
malaria_model = load_model('Malaria.h5')
pneu_model = load_model('Pneumonia.h5')



#Initial Page Routing from Home Page

@app.route('/')
def open_func():
    return render_template("index.html")


@app.route("/home")
def home():
    return render_template("index.html")
 

@app.route("/about")
def about():
    return render_template("metadata_index_test.html")


@app.route("/c1")
def cancer():
    return render_template("cancer_index_test.html")


@app.route("/d1")
def diabetes():
    #if form.validate_on_submit():
    return render_template("diabetes_index_test.html")

@app.route("/a1")
def heart():
    return render_template("arrhythmia_index_test.html")


@app.route("/l1")
def liver():
    #if form.validate_on_submit():
    return render_template("liver_index_test.html")

@app.route("/k1")
def kidney():
    #if form.validate_on_submit():
    return render_template("kidney_index_test.html")

@app.route("/m1")
def Malaria():
    return render_template("malaria_index_test.html")

@app.route("/p1")
def Pneumonia():
    return render_template("pneumonia_index_test.html")


#Upload and Prediction Functionalities

@app.route('/cancer', methods=["POST"])
def cancer_func():
    r_mean = request.form["Radius_mean"]
    t_mean = request.form["Texture_mean"]
    p_mean = request.form["Perimeter_mean"]
    a_mean = request.form["Area-mean"]
    s_mean = request.form["Smoothness_mean"]
    c_mean = request.form["Compactness_mean"]
    conc_mean = request.form["Concavity_mean"]
    conc_points_mean = request.form["Concave_points_mean"]
    sym_mean = request.form["Symmetry_mean"]
    fd_mean = request.form["fractional_dimension_mean"]
    r_se = request.form["Radius_se"]
    t_se = request.form["texture_se"]
    p_se = request.form["Perimeter_se"]
    a_se = request.form["Area_se"]
    s_se = request.form["smoothness_se"]
    c_se = request.form["Compactness_se"]
    conc_se = request.form["Concavity_se"]
    conc_points_se = request.form["Concave_points_se"]
    sym_se = request.form["symmetry_se"]
    fd_se = request.form["Fractal_dimension_se"]
    r_w = request.form["Radius_worst"]
    t_w = request.form["Texture_worst"]
    p_w = request.form["Perimeter_worst"]
    a_w = request.form["Area_worst"]
    s_w = request.form["Smoothness_worst"]
    c_w = request.form["Compactness_worst"]
    conc_w = request.form["Concavity_worst"]
    conc_points_worst = request.form["Concave_points_worst"]
    sym_w = request.form["Symmetry_worst"]
    fd_w = request.form["Fractal_dimension_worst"]
    
    data = [[r_mean,t_mean,p_mean,a_mean,s_mean,c_mean,conc_mean,conc_points_mean, sym_mean,fd_mean,
             r_se,t_se, p_se,a_se,s_se, c_se,conc_se,conc_points_se,sym_se, fd_se,
             r_w,t_w,p_w,a_w,s_w,c_w,conc_w,conc_points_worst,sym_w, fd_w]]
    
    pred = model.predict(data)
    
    if pred==1:
       # print('M')
        inter = 'Well... I find something unusually "serious" from my analysis :(Pls visit doctor to receive aid asap!'
        return render_template("cancer_index_test.html", y='Malignant',z =pred, a = '96.50%', u= 'Logistic Regression', p= inter)
    else:
        # print('B')
        inter = 'Well... I find something "less serious" from my analysis :( But pls visit a doctor to find cure in initial stage!'
        return render_template("cancer_index_test.html", y='Benign',z =pred,a = '96.50%', u='Logistic Regression', p= inter)
    
@app.route('/ckd', methods = ['POST'])
def ckd_func():
    sg = request.form["sg"]
    htn = request.form["htn"]
    hemo = request.form["hemo"]
    dm = request.form["dm"]
    al = request.form["al"]
    appet = request.form["appet"]
    rc = request.form["rc"]
    pc = request.form["pc"]    
    
    data = [[sg,htn,hemo,dm,al,appet,rc,pc]]
    pred = ckd_model.predict(data)
    
    if pred==1:
        #print('ckd patient')
        inter = 'Well... I find something unusual from my analysis :(  Pls visit doctor to receive aid asap!'
        return render_template('kidney_index_test.html', y='Positive for Chronic Kidney Disease', z= pred,a='97.50%', u='Logistic Regression', p=inter )
    else:
        #print('Non CKD patient')
        inter = 'Hurray! You are in sound health..... Set your worries aside :)'
        return render_template('kidney_index_test.html', y='Negative for Chronic Kidney Disease', z= pred,a ='97.50%', u='Logistic Regression', p=inter)
    

@app.route('/liver', methods =['POST'])
def liver_func():  
    asp = request.form["asp"]
    ala = request.form["ala"]
    alk = request.form["alk"]
    tot_bil = request.form["tot_bil"]
    alb = request.form["alb"]
    direct_bil = request.form["direct_bil"]
    alb_glo = request.form["alb_glo"]
    tot_pro = request.form["tot_pro"]    
    
    data = [[asp,ala,alk,tot_bil,alb,direct_bil,alb_glo,tot_pro]]
    pred = liver_model.predict(data)
    
    if pred==0:
        #print('Liver Disease patient')
        inter = 'Well... I find something unusual from my analysis :(  Pls visit doctor to receive aid asap!'
        return render_template('liver_index_test.html', y='Positive for Liver Disease', z= pred,a =' 76.06%', u ='K-Nearest Neighbors', p=inter)
    
    else:
        #print('Non Liver Disease patient')
        inter = 'Hurray! You are in sound health..... Set your worries aside :)'
        return render_template('liver_index_test.html', y='Negative for Liver Disease', z= pred,a ='76.06%', u ='K-Nearest Neighbors', p=inter)

@app.route('/diabetes', methods= ['POST'])
def diab_func():
    preg = request.form["preg"]
    gluc = request.form["glucose"]
    bp = request.form["bp"]
    skin = request.form["skin"]
    insulin = request.form["insulin"]
    bmi = request.form["bmi"]
    ped_func = request.form["dia_ped_func"]
    age = request.form["age"]    
    
    data = [[preg, gluc, bp, skin, insulin, bmi, ped_func, age]]
    pred = diabetes_model.predict(data)
    
    if pred==1:
        
        inter ='Sorry! You are suffering from Diabetes :( Pls monitor your sugar levels regularly!'
        return render_template('diabetes_index_test.html', y='Diabetic', z= pred,a ='82.46%', u ='Logistic Regression', p = inter)
    else:
       
        inter = 'Hurray! You are in sound health..... Set your worries aside :)'
        return render_template('diabetes_index_test.html', y='Non-Diabetic', z= pred, a='82.46%', u ='Logistic Regression', p = inter)

@app.route('/arrhythmia', methods= ['POST'])
def arrhy_func():
        f = request.files["file"]
        basepath = os.path.dirname('__file__')
        filepath = os.path.join(basepath,"uploads",f.filename) #C:/Users/Kkira/OneDrive/Desktop/mini project/test_depl/uploads
        f.save(filepath) 
        
        img = image.load_img(filepath, target_size=(64,64))
        x = image.img_to_array(img)
        x = np.expand_dims(x,axis=0)
        
        pred = arr_model.predict(x)
        classes_x = np.argmax(pred,axis=1)
        
        index =['Left Bundle Branch Block','Normal','Premature Atrial Contraction',
       'Premature Ventricular Contractions', 'Right Bundle Branch Block','Ventricular Fibrillation']
        
        result=str(index[classes_x[0]])
        
        if classes_x==1:
            inter = 'Hurray! You are in sound health..... Set your worries aside :)'
        else:
            inter = 'Well... I find something unusual from my analysis :(  Pls visit a doctor to receive aid asap!'
            
        return render_template('arrhythmia_index_test.html', y= result, z= classes_x, a='95.89%', u='Convolution Neural Networks!', p =inter ) 

@app.route('/malaria', methods =['POST'])
def malarial_func():
        f = request.files["file1"]
        basepath = os.path.dirname('__file__')
        filepath = os.path.join(basepath,"uploads",f.filename) #C:/Users/Kkira/OneDrive/Desktop/mini project/test_depl/uploads
        f.save(filepath) 
        
        img = image.load_img(filepath,target_size= (130,130))#loading of the image
        x =  image.img_to_array(img)#image to array
        x = np.expand_dims(x,axis = 0)#changing the shape
        
        pred = malaria_model.predict(x)
        res= pred[0][0]
        
        index = ['Parasitized','Uninfected']
        
        if res>0.50:
            inter = 'Hurray! You are in sound health..... Set your worries aside :)'
            classes_x = index[1]
        else:
            inter ='Well... I find something unusual from my analysis :( Pls visit a doctor to receive aid asap! '
            classes_x = index[0]
        
        return render_template('malaria_index_test.html', y= classes_x, z= res, a='96.50%', u='Convolution Neural Networks!', p=inter )

@app.route('/pneumonia', methods =['POST'])
def pneu_func():
        f = request.files["file2"]
        basepath = os.path.dirname('__file__')
        filepath = os.path.join(basepath,"uploads",f.filename) #C:/Users/Kkira/OneDrive/Desktop/mini project/test_depl/uploads
        f.save(filepath) 
        
        img = image.load_img(filepath,target_size= (150,150))#loading of the image
        x = image.img_to_array(img)#image to array
        x = np.expand_dims(x,axis = 0)#changing the shape
        x = x * 1.0 / 255
        pred = pneu_model.predict(x)
        res= pred[0][0]
        
        index = ['Normal', 'pneumonia']
        
        if res>0.50:
            inter ='Well... I find something unusual from my analysis :( Pls visit a doctor to receive aid asap! '
            classes_x = index[1]
        else:
            inter = 'Hurray! You are in sound health..... Set your worries aside :)'
            classes_x = index[0]
        
        return render_template('pneumonia_index_test.html', y= classes_x, z= res, a='94.09%', u='Convolution Neural Networks!', p=inter )
    
      

app.run(debug = True)
    
    