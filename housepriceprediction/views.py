from django.shortcuts import render
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle
import numpy as np

# Create your views here.

def HouseModelTraining(request):
    context={}
    data=pd.read_csv("House_data_preprocessed.csv")
    context["samples"]=data.shape[0]

    if request.method == 'GET':
        context["score"]="-"

    if request.method == 'POST':
        Y= data["price"]
        X = data.drop("price", axis="columns")
        x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.2)
        model=LinearRegression()
        model.fit(x_train,y_train)
        score=model.score(x_test,y_test)
        context["score"]=score
        with open('house_model.pickle','wb') as f:
            pickle.dump(model,f)
    return render(request, 'housepriceprediction/HouseModelTraining.html', context) 

def HouseModelPrediction(request):
    context={}
    data=pd.read_csv("House_data_preprocessed.csv")
    context["locations"]=data.columns[4:]

    if request.method == 'GET':
        context['area'] ='1500'
        context['bathroom']='2'
        context['bhk']='3'
        context['location']=''
        context['price']="-"

    if request.method == 'POST':
        Y= data["price"]
        X = data.drop("price", axis="columns")
        area=int(request.POST.get('area',0))
        bathroom=int(request.POST.get('bathrooms',0))
        bhk=int(request.POST.get('bhk',0))
        location=(request.POST.get('location',0))

        context['area']= area
        context['bathroom']=bathroom
        context['bhk']=bhk
        context['location']=location

        with open('house_model.pickle','rb') as r:
            model=pickle.load(r)

        loc_index= np.where(X.columns==location)[0][0]

        input = np.zeros(len(X.columns))
        input[0]= area
        input[1]= bathroom
        input[2] = bhk
        if loc_index >= 0:
            input[loc_index] = 1

        price = model.predict([input])
        context['price']= "{0:.2f}".format(price[0])

    return render(request, 'housepriceprediction/HouseModelPrediction.html', context)     
