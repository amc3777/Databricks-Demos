# Databricks notebook source
user = dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()
uc_prefix = user.replace(".", "").split("@")[0]
catalog = uc_prefix + "_catalog"

dbutils.widgets.text("catalog_name", catalog)
dbutils.widgets.text("schema_name", "models")
catalog_name = dbutils.widgets.get("catalog_name")
schema_name = dbutils.widgets.get("schema_name")

registered_model_name = f"{catalog_name}.{schema_name}.VectorAssembler_ExtendedIsolationForestModel"
dbutils.widgets.text("model_name", registered_model_name)
model_name = dbutils.widgets.get("model_name")

dbutils.widgets.text("endpoint_name", "VectorAssembler_ExtendedIsolationForestModel")
endpoint_name = dbutils.widgets.get("endpoint_name")

# COMMAND ----------

# Importing necessary libraries
import pandas  # Data manipulation library
import pyarrow  # Library used for working with Arrow format data
import numpy  # Numerical library
import pyspark  # Apache Spark library
from pyspark.sql.functions import col  # Importing function 'col' from 'pyspark.sql.functions'
import mlflow  # Open-source platform for managing the machine learning lifecycle
from pyspark.ml.feature import VectorAssembler  # Transforming data into feature vectors
import h2o  # Open-source, distributed machine learning platform
from pysparkling import *  # Library for using H2O's machine learning algorithms with Apache Spark
from pysparkling.ml import *  # Importing H2O's Sparkling Water algorithms for use with Apache Spark
from mlflow.tracking.client import MlflowClient  # Python client for MLflow's REST API
import requests  # Library for making HTTP requests
import time  # Library for time-related functions

# COMMAND ----------

# Creating an instance of H2OContext
hc = H2OContext.getOrCreate()

# COMMAND ----------

# Importing the "prostate.csv" dataset into an H2O dataframe
h2o_df = h2o.import_file("https://raw.github.com/h2oai/h2o/master/smalldata/logreg/prostate.csv")

# Converting the H2O dataframe to a Spark dataframe using H2OContext
sparkDF = hc.asSparkFrame(h2o_df)

# Splitting the Spark dataframe into training and testing dataframes, with a ratio of 0.8:0.2 respectively
[trainingDF, testingDF] = sparkDF.randomSplit([0.8, 0.2])

# COMMAND ----------

# Create an instance of VectorAssembler
vecAssembler = VectorAssembler()

# Set the output column name as "features"
vecAssembler.setOutputCol("features")

# Set the input columns of VectorAssembler as the columns of the trainingDF dataframe
vecAssembler.setInputCols(trainingDF.columns)

# Transform the trainingDF dataframe using the VectorAssembler, selecting only the "features" column
trainingDF = vecAssembler.transform(trainingDF)[['features']]

# Transform the testingDF dataframe using the VectorAssembler, selecting only the "features" column
testingDF = vecAssembler.transform(testingDF)[['features']]

# Display the trainingDF dataframe
display(trainingDF)

# COMMAND ----------

# Create an instance of H2OExtendedIsolationForest algorithm with the specified parameters:
# - sampleSize: It determines the size of the subsamples used for building isolation trees
# - ntrees: It defines the number of isolation trees to create
# - seed: It sets the random seed for reproducibility purposes
# - extensionLevel: It determines the level of extension for the extended Isolation Forest algorithm
algo = H2OExtendedIsolationForest(
                                  sampleSize=256,
                                  ntrees=100,
                                  seed=123,
                                  extensionLevel=8)

# Fit the H2OExtendedIsolationForest model on the training data
# and assign the resulting model to the variable 'model'
model = algo.fit(trainingDF)

# COMMAND ----------

# Save the model to a specific location
h2o_model_path = "file:/exported_mojo"
model.write().overwrite().save(h2o_model_path)

# COMMAND ----------

# Test loading the model for batch inference and returning results as a pandas dataframe
model = H2OMOJOModel.createFromMojo("file:///exported_mojo/mojo_model")
output = model.transform(testingDF)
output_example = output.select("prediction").toPandas() # This will be needed later when registering the model
display(output_example)

# COMMAND ----------

# Create an artifacts key-value pair for loading into the context of a pyfunc predict function
artifacts = {"h2o_model": "file:///exported_mojo/mojo_model"}

# COMMAND ----------

class ModelWrapper(mlflow.pyfunc.PythonModel):
    # Define a class ModelWrapper that inherits from mlflow.pyfunc.PythonModel

    def load_context(self, context):
        # Define a method load_context that takes self, context as parameters

        self.path = str(context.artifacts["h2o_model"])
        # Assign the value of "h2o_model" from context.artifacts to self.path

    def predict(self, context, model_input, params=None):
        # Define a method predict that takes self, context, model_input, and params as parameters

        from pyspark.sql import SparkSession
        import pysparkling
        import pysparkling.ml
        from pyspark.ml.feature import VectorAssembler
        # Import the required libraries

        spark = SparkSession.builder.master("local[*]").appName("VecAssembler").getOrCreate()
        # Create a SparkSession

        hc = H2OContext.getOrCreate()
        # Create or get an H2OContext

        h2o_model = H2OMOJOModel.createFromMojo(f"file://{self.path}")
        # Create an H2O MOJO model using the path stored in self.path

        model_input = spark.createDataFrame(model_input)
        # Create a Spark DataFrame from the model_input, the input will be a pandas dataframe

        vecAssembler = VectorAssembler(outputCol="features")
        vecAssembler.setInputCols(model_input.columns)
        assembled_vector = vecAssembler.transform(model_input)[[('features')]]
        # Create a vector assembler and transform the model_input DataFrame to a DataFrame with a features column

        output = h2o_model.transform(assembled_vector)
        output = output.select("prediction").toPandas()
        # Generate predictions using the h2o_model and select the "prediction" column as a Pandas DataFrame

        return output
        # Return the output

# COMMAND ----------

# Import the required libraries
import platform  # To get the Python version
import requests  # To make HTTP requests
import tabulate  # To format tabular data
import future  # To support Python 2/3 compatibility
import py4j # To support Python access to JVM objects

# Get the Python version and assign it to the Python_VERSION variable
python_VERSION = platform.python_version()

# COMMAND ----------

# Set the dependencies for the Conda environment
conda_env = {
    "channels": ["conda-forge"],
    "dependencies": [
        
        # Set the Python version
        "python={}".format(python_VERSION),
        
        "pip",
        
        # Add H2O dependency
        "h2o-py",
        
        # Add JDK dependency
        "openjdk=8",

        # Add H2O Sparkling Water dependency    
        "h2oai::h2o_pysparkling_3.4",
        
        {
            "pip": [
                # Add MLflow dependency
                "mlflow=={}".format(mlflow.__version__),
                
                # Add PySpark dependency
                "pyspark=={}".format(pyspark.__version__),
                
                # Add Pandas dependency
                "pandas=={}".format(pandas.__version__),
                
                # Add Py4j dependency
                "py4j=={}".format(py4j.__version__),
                
                # Add PyArrow dependency
                "pyarrow=={}".format(pyarrow.__version__),
                
                # Add NumPy dependency
                "numpy=={}".format(numpy.__version__),
                
                # Add Requests dependency
                "requests=={}".format(requests.__version__),
                
                # Add Tabulate dependency
                "tabulate=={}".format(tabulate.__version__),
                
                # Add Future dependency
                "future=={}".format(future.__version__),
            ],
        },
    ],
    
    # Set the name of the Conda environment
    "name": "pyspark_env",
}

# COMMAND ----------

# Create an input example to log with the model to MLflow
h2o_df = h2o.import_file("https://raw.github.com/h2oai/h2o/master/smalldata/logreg/prostate.csv")
sparkDF = hc.asSparkFrame(h2o_df)
[trainingDF, testingDF] = sparkDF.randomSplit([0.8, 0.2])
testingDF = testingDF.select([col(c).cast("long") for c in testingDF.columns])
input_example = testingDF.toPandas()[:5] # pandas dataframes is a compatible format for custom models endpoints
display(input_example)

# COMMAND ----------

# Save and log the custom model to MLflow
from mlflow.models.signature import infer_signature

mlflow.set_registry_uri("databricks-uc")

with mlflow.start_run(run_name="VectorAssembler_ExtendedIsolationForestModel") as run:
  mlflow.pyfunc.log_model(
      artifact_path="VectorAssembler_ExtendedIsolationForestModel",
      python_model=ModelWrapper(),
      artifacts=artifacts,
      conda_env=conda_env,
      registered_model_name=model_name, # This will be needed later for the model endpoint
      input_example=input_example,
      signature=infer_signature(input_example, output_example)
      )
  
  run_id = run.info.run_id

# COMMAND ----------

# Set important variables for model endpoint creation
model_name = model_name # This was set above with the registered model name
model_serving_endpoint_name = endpoint_name # Set this to name your model endpoint

client = MlflowClient()
results = client.search_model_versions(filter_string=f"name='{model_name}'")
model_version = results[0].version # Store latest model version for model in this variable

java_tags = dbutils.notebook.entry_point.getDbutils().notebook().getContext().tags()
tags = sc._jvm.scala.collection.JavaConversions.mapAsJavaMap(java_tags)
instance = tags["browserHostName"] # This ultimately returns the workspace host name for the API-based creation of an endpoint

# COMMAND ----------

# Create the JSON request payload for API-based Model Serving endpoint creation
token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().getOrElse(None)

headers = {
    "Authorization": f"Bearer {token}",
    "Content-Type": "application/json"
  }

my_json = {
  "name": model_serving_endpoint_name,
  "config": {
   "served_models": [{
     "model_name": model_name,
     "model_version": model_version,
     "workload_size": "Small",
     "scale_to_zero_enabled": True
   }]
 }
}

# COMMAND ----------

# A function to create or update Model Serving endpoints through the Python requests library
def func_create_endpoint(model_serving_endpoint_name):

  endpoint_url = f"https://{instance}/api/2.0/serving-endpoints"
  url = f"{endpoint_url}/{model_serving_endpoint_name}"
  r = requests.get(url, headers=headers)

  if "RESOURCE_DOES_NOT_EXIST" in r.text:

    print("Creating this new endpoint: ", f"https://{instance}/serving-endpoints/{model_serving_endpoint_name}/invocations")
    re = requests.post(endpoint_url, headers=headers, json=my_json)

  else:

    new_model_version = (my_json['config'])['served_models'][0]['model_version']
    print("This endpoint existed previously! We are updating it to a new config with new model version: ", new_model_version)
  
    url = f"{endpoint_url}/{model_serving_endpoint_name}/config"
    re = requests.put(url, headers=headers, json=my_json['config']) 

    import time,json

    url = f"https://{instance}/api/2.0/serving-endpoints/{model_serving_endpoint_name}"
    retry = True
    total_wait = 0

    while retry:

      r = requests.get(url, headers=headers)

      assert r.status_code == 200, f"Expected an HTTP 200 response when accessing endpoint info, received {r.status_code}"
      endpoint = json.loads(r.text)

      if "pending_config" in endpoint.keys():

        seconds = 10
        print("New config still pending")

        if total_wait < 6000:

          print(f"Wait for {seconds} seconds")
          print(f"Total waiting time so far: {total_wait} seconds")
          time.sleep(10)
          total_wait += seconds

        else:

          print(f"Stopping,  waited for {total_wait} seconds")
          retry = False  

      else:

        print("New config in place now!")
        retry = False

  assert re.status_code == 200, f"Expected an HTTP 200 response, received {re.status_code}"

# A function to delete Model Serving endpoints through the Python requests library
def func_delete_model_serving_endpoint(model_serving_endpoint_name):

  endpoint_url = f"https://{instance}/api/2.0/serving-endpoints"
  url =  f"{endpoint_url}/{model_serving_endpoint_name}" 
  response = requests.delete(url, headers=headers)

  if response.status_code != 200:

    raise Exception(f"Request failed with status {response.status_code}, {response.text}")

  else:
    
    print(model_serving_endpoint_name, "endpoint is deleted!")

# COMMAND ----------

# Create or update an endpoint
func_create_endpoint(model_serving_endpoint_name)

# COMMAND ----------

# Wait for the endpoint to finish deployment
def wait_for_endpoint():

    endpoint_url = f"https://{instance}/api/2.0/serving-endpoints"

    while True:

        url =  f"{endpoint_url}/{model_serving_endpoint_name}"
        response = requests.get(url, headers=headers)
        assert response.status_code == 200, f"Expected an HTTP 200 response, received {response.status_code}\n{response.text}"
 
        status = response.json().get("state", {}).get("ready", {})

        if status == "READY": print(status); print("-"*80); return

        else: print(f"Endpoint not ready ({status}), waiting 10 seconds"); time.sleep(10) # Wait 10 seconds
        
api_url = mlflow.utils.databricks_utils.get_webapp_url()
 
wait_for_endpoint()
 
time.sleep(5)

# COMMAND ----------

# Create a test example to send to the endpoint
input_df = input_example.to_dict(orient="split")

payload_json = {"dataframe_split": input_df}

display(payload_json)

# COMMAND ----------

# Use the model endpoint for inference
def score_model(data_json: dict):
    url =  f"https://{instance}/serving-endpoints/{model_serving_endpoint_name}/invocations"
    response = requests.request(method="POST", headers=headers, url=url, json=data_json)
    if response.status_code != 200:
        raise Exception(f"Request failed with status {response.status_code}, {response.text}")
    return response.json()
  
print(score_model(payload_json))

# COMMAND ----------

# Delete an endpoint
# func_delete_model_serving_endpoint(model_serving_endpoint_name)
