# PredictionIO template: DecisionTree with feature importance

## Overview
An engine template is an almost-complete implementation of an engine. In this Engine Template, we have integrated Apache Spark MLlib's Decision Tree algorithm by default.
The default use case of this regression Engine Template is to predict the price of the [Boston Housing dataset](https://archive.ics.uci.edu/ml/datasets/Housing).
You can customize it easily to fit your specific use case and needs.

We are going to show you how to create your own regression engine for production use based on this template.

## Usage

### With docker:

The best way to get started with predictionio is by using docker. 

#### From our image

```
docker run -ti -p 9000:9000 -v /Users/vallette/ANTS/template-decision-tree-feature-importance:/MyRegression ants/predictionio:v0.9.1
```

#### Building the docker image

Following the steps of [mingfang](https://github.com/mingfang/docker-predictionio):

```
git clone https://github.com/mingfang/docker-predictionio.git
cd docker-predictionio
./build
./shell
```
#### Once in the predictionio container

```
runsvdir-start&
```

to start all the services.

You can check the status by running:
```
pio status
```

## Download Template
Clone the current repository by executing the following command in the directory where you want the code to reside:
    
```
git clone https://github.com/anthill/template-decision-tree-feature-importance.git MyRegression
cd MyRegression
```

## Generate an App ID and Access Key
Let's assume you want to use this engine in an application named "MyApp1". You will need to collect some training data for machine learning modeling. You can generate an App ID and Access Key that represent "MyApp1" on the Event Server easily:
```
pio app new MyApp1

```
You should find the following in the console output:
```
...
[INFO] [App$] Initialized Event Store for this app ID: 1.
[INFO] [App$] Created new app:
[INFO] [App$]       Name: MyApp1
[INFO] [App$]         ID: 1
[INFO] [App$] Access Key: ZON0FP6gdLeXxg1g7O1E9TPXIOxQMIngIr0LWQUC5Tv0utyyGwvs1AmG6DyDchLO
```

Take note of the Access Key and App ID. You will need the Access Key to refer to "MyApp1" when you collect data. At the same time, you will use App ID to refer to "MyApp1" in engine code.

`pio app list` will return a list of names and IDs of apps created in the Event Server.


## Collecting Data

The description of the dataset gives inforamtion about the features.

    1. CRIM      per capita crime rate by town
    2. ZN        proportion of residential land zoned for lots over 
                 25,000 sq.ft.
    3. INDUS     proportion of non-retail business acres per town
    4. CHAS      Charles River dummy variable (= 1 if tract bounds 
                 river; 0 otherwise)
    5. NOX       nitric oxides concentration (parts per 10 million)
    6. RM        average number of rooms per dwelling
    7. AGE       proportion of owner-occupied units built prior to 1940
    8. DIS       weighted distances to five Boston employment centres
    9. RAD       index of accessibility to radial highways
    10. TAX      full-value property-tax rate per $10,000
    11. PTRATIO  pupil-teacher ratio by town
    12. B        1000(Bk - 0.63)^2 where Bk is the proportion of blacks 
                 by town
    13. LSTAT    % lower status of the population

Most of the features ar continuous execpt CHAS and RAD. We can specify this in `data/learning_metadata.csv` which contains:

```
co,co,co,ca2,co,co,co,co,ca25,co,co,co,co
```

We've implemented a file reader in `DataSource` enabling you to lead big chunks of data at the initialisation of the algorithm.
You can do the learning with:

```
pio build --varbose
pio train
```

 
Next, let's collect some training data. By default, the Classification Engine Template reads 4 properties of a user record: attr0, attr1, attr2 and plan.

You can send these data to PredictionIO Event Server in real-time easily by making a HTTP request or through the EventClient of an SDK.

Although you can integrate your app with PredictionIO and collect training data in real-time, we are going to import a sample dataset with the provided scripts for demonstration purpose.

Execute the following command in the Engine directory(MyClassification) to get the sample dataset from MLlib repo:
```
curl https://raw.githubusercontent.com/apache/spark/master/data/mllib/sample_naive_bayes_data.txt --create-dirs -o data/sample_decision_trees.txt

```

A Python import script import_eventserver.py is provided in the template to import the data to Event Server using Python SDK.
Replace the value of access_key parameter by your Access Key and run:
```python
$ cd MyRecomendation
$ python data/import_eventserver.py --access_key 3mZWDzci2D5YsqAnqNnXH9SB6Rg3dsTBs8iHkK6X2i54IQsIZI1eEeQQyMfs7b3F
```
You should see the following output:
```
Importing data...
6 events are imported.
```
This python script converts the data file to proper events formats as needed by the event server.
Now the training data is stored as events inside the Event Store.

## Deploy the Engine as a Service
Now you can build, train, and deploy the engine. First, make sure you are under the MyClassification directory.

### Input Query
- array of features values ( 3 features)
```
{"features": [0, 2, 0]}
```

### Output Predicted Result
- the predicted label 
```
{"label":0.0}
```

### Engine.json

Under the directory, you should find an engine.json file; this is where you specify parameters for the engine.
Make sure the appId defined in the file match your App ID. (This links the template engine with the App)

Parameters for the Decision Tree model are to be set here. 

numClasses: Number of classes in the dataset

maxDepth: Max depth of the tree generated

maxBins: Max number of Bins

```
{
  "id": "default",
  "description": "Default settings",
  "engineFactory": "org.template.classification.ClassificationEngine",
  "datasource": {
    "params": {
      "appId": 1
    }
  },
  "algorithms": [
    {
      "name": "decisiontree",
      "params": {
        
        "numClasses": 3,
        "maxDepth": 5,
        "maxBins": 100      
      }
    }
  ]
}
```
### Build

Start with building your MyClassification engine.
```
$ pio build
```
This command should take few minutes for the first time; all subsequent builds should be less than a minute. You can also run it with --verbose to see all log messages.

Upon successful build, you should see a console message similar to the following.
```
[INFO] [Console$] Your engine is ready for training.
```

### Training the Predictive Model

Train your engine.

```
$ pio train
```
When your engine is trained successfully, you should see a console message similar to the following.

```
[INFO] [CoreWorkflow$] Training completed successfully.
```
### Deploying the Engine

Now your engine is ready to deploy.

```
$ pio deploy
```
This will deploy an engine that binds to http://localhost:8000. You can visit that page in your web browser to check its status.

## Use the Engine

Now, You can try to retrieve predicted results. For example, to predict the label (i.e. plan in this case) of a user with attr0=2, attr1=0 and attr2=0, you send this JSON { "features": [2, 0, 0] } to the deployed engine and it will return a JSON of the predicted plan. Simply send a query by making a HTTP request or through the EngineClient of an SDK:
```python
import predictionio
engine_client = predictionio.EngineClient(url="http://localhost:8000")
print engine_client.send_query({"features": [2, 0, 0]})
```
The following is sample JSON response:

```
{"label":0.0}
```

The sample quesry can be found in **test.py**


