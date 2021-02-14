# House Price Prediction
## AIM:
To design a website to train the house price model and to predict the price.

## DESIGN STEPS:
### Step 1: 
Requirement collection.
### Step 2:
Creating the layout using HTML and CSS.
### Step 3:
Train the model using the given data set
### Step 4:
Save the trained model using pickle
### Step 5:
Get the input from the user
### Step 6:
Load the trained model using pickle
### Step 7:
Apply the given data to the model
### Step 8:
Display the result
### Step 9:
Publish the website in the given URL.

## PROGRAM:
### HouseModelPrediction.html
```
<!DOCTYPE html>
{% load static %}
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Document</title>
    <link rel="stylesheet" href="{% static 'css/house.css' %}">
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@4.6.0/dist/css/bootstrap.min.css" integrity="sha384-B0vP5xmATw1+K9KRQjQERJvTumQW0nPEzvF6L/Z6nronJ3oUOFUFpCjEUQouq2+l" crossorigin="anonymous">
</head>
<body>
<form class="p-5" method="POST" action="/HouseModelPrediction/">
    {% csrf_token %}
    <div class="container-fluid text-center bg p-5">
        <h1 class="display-1" style="padding-top: 65px; color: black;">TITAN COMPANY</h1>
    </div>
    <div class="form-group">
      <label for="total">Total</label>
      <input name="area" type="text" class="form-control" id="total" placeholder="{{area}}">
    </div>
    <div class="form-group">
      <label for="bathroom">Bathroom</label>
      <input name="bathroom" type="text" class="form-control" id="bathroom" placeholder="{{bathroom}}">
    </div>
    <div class="form-group">
      <label for="bhk">BHK</label>
      <input name="bhk" type="text" class="form-control" id="bhk" placeholder="{{bhk}}">
    </div>
    <div class="col-md-3 mb-3">
      <label for="validationTooltip04">Location</label>
      <select name="location" id="validationTooltip04" class="col-md-10 form-control">
          {% for loc in locations %}
            {% if loc == location %}
                <option selected value="{{loc}}">{{loc}}</option>
            {% else %}
                <option value="{{loc}}">{{loc}}</option>
            {% endif %}
          {% endfor %}
      </select>
    </div>
    <div class="form-group">
      <label for="price">Price</label>
      <input name="price" type="text" class="form-control" id="price" placeholder="{{price}}">
    </div>
    <button type="submit" class="btn btn-primary">Predict</button>
</form>
</body>
<script src="https://code.jquery.com/jquery-3.5.1.slim.min.js" integrity="sha384-DfXdz2htPH0lsSSs5nCTpuj/zy4C+OGpamoFVy38MVBnE+IbbVYUew+OrCXaRkfj" crossorigin="anonymous"></script>
<script src="https://cdn.jsdelivr.net/npm/bootstrap@4.6.0/dist/js/bootstrap.bundle.min.js" integrity="sha384-Piv4xVNRyMGpqkS2by6br4gNJ7DXjqk09RmUpJ8jgGtD7zP9yug3goQfGII0yAns" crossorigin="anonymous"></script>
</html>
```

### Housetraining.html
```
{% load static %}
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Document</title>
    <link rel="stylesheet" href="{% static 'css/house.css' %}">
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@4.6.0/dist/css/bootstrap.min.css" integrity="sha384-B0vP5xmATw1+K9KRQjQERJvTumQW0nPEzvF6L/Z6nronJ3oUOFUFpCjEUQouq2+l" crossorigin="anonymous">
</head>
<body>
    <div class="container-fluid text-center bg p-5">
        <h1 class="display-1" style="padding-top: 65px; color: black;">TRAIN</h1>
    </div>
    <div class="p-5">
        <form action="/HouseModelTraining/" method="POST">
            {% csrf_token %}
            <div class="form-group">
              <label for="sample">Total Samples</label>
              <input type="text" class="form-control" id="sample" placeholder="{{samples}}">
            </div>
            <div class="form-group">
              <label for="score">Score</label>
              <input type="text" class="form-control" id="score" placeholder="{{score}}">
            </div>
            <button type="submit" class="btn btn-primary">Train</button>
          </form>
    </div>
</body>
<script src="https://code.jquery.com/jquery-3.5.1.slim.min.js" integrity="sha384-DfXdz2htPH0lsSSs5nCTpuj/zy4C+OGpamoFVy38MVBnE+IbbVYUew+OrCXaRkfj" crossorigin="anonymous"></script>
<script src="https://cdn.jsdelivr.net/npm/bootstrap@4.6.0/dist/js/bootstrap.bundle.min.js" integrity="sha384-Piv4xVNRyMGpqkS2by6br4gNJ7DXjqk09RmUpJ8jgGtD7zP9yug3goQfGII0yAns" crossorigin="anonymous"></script>
</html>
```
### Views.py
```
from django.shortcuts import render
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle
import numpy as np


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
```


## OUTPUT:
![output](./static/img/Screenshot(184).jpg)

![output](./static/img/Screenshot(185).jpg)

![output](./static/img/Screenshot(186).jpg)

## RESULT:
Thus the housepricemodel Website is created and hosted on server
