# Databricks notebook source
import pyspark.sql.functions as f

# Generate a DataFrame with 1000*100 records
df = (spark
      .range(1000*100)
      .select(f.col("id").alias("record_id"), (f.col("id")%10).alias("device_id"))
      .withColumn("feature_1", f.rand() * 1)                    # Add random values between 0 and 1 to column "feature_1"
      .withColumn("feature_2", f.rand() * 2)                    # Add random values between 0 and 2 to column "feature_2"
      .withColumn("feature_3", f.rand() * 3)                    # Add random values between 0 and 3 to column "feature_3"
      .withColumn("label", (f.col("feature_1") + f.col("feature_2") + f.col("feature_3")) + f.rand())  # Compute sum of "feature_1", "feature_2", and "feature_3" and add random value
     )

# Display the DataFrame
display(df)

# COMMAND ----------

train_return_schema = "device_id integer, n_used integer, model_path string, mse float"

# COMMAND ----------

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

def train_model(df_pandas: pd.DataFrame) -> pd.DataFrame:
    """
    Trains an sklearn model on grouped instances
    """
    # Pull metadata
    device_id = df_pandas["device_id"].iloc[0]
    n_used = df_pandas.shape[0]
    run_id = df_pandas["run_id"].iloc[0] # Pulls run ID to do a nested run

    # Train the model
    X = df_pandas[["feature_1", "feature_2", "feature_3"]]
    y = df_pandas["label"]
    rf = RandomForestRegressor()
    rf.fit(X, y)

    # Evaluate the model
    predictions = rf.predict(X)
    mse = mean_squared_error(y, predictions) # Note we could add a train/test split

    # Resume the top-level training
    with mlflow.start_run(run_id=run_id) as outer_run:
        # Small hack for running as a job
        experiment_id = outer_run.info.experiment_id
        print(f"Current experiment_id = {experiment_id}")

        # Create a nested run for the specific device
        with mlflow.start_run(run_name=str(device_id), nested=True, experiment_id=experiment_id) as run:
            mlflow.sklearn.log_model(rf, str(device_id))
            mlflow.log_metric("mse", mse)
            mlflow.set_tag("device", str(device_id))

            artifact_uri = f"runs:/{run.info.run_id}/{device_id}"
            # Create a return pandas DataFrame that matches the schema above
            return_df = pd.DataFrame([[device_id, n_used, artifact_uri, mse]], 
                                    columns=["device_id", "n_used", "model_path", "mse"])

    return return_df


# COMMAND ----------

# Start an MLflow run with the name "Training session for all devices" and get the run ID
with mlflow.start_run(run_name="Training session for all devices") as run:
    run_id = run.info.run_id

    # Add the run_id column to the DataFrame
    model_directories_df = (df
        .withColumn("run_id", f.lit(run_id))
        # Group the data by device_id and apply the train_model function to each group, using the specified schema
        .groupby("device_id")
        .applyInPandas(train_model, schema=train_return_schema)
        # Cache the DataFrame to improve performance
        .cache()
    )

# Join the original DataFrame with the model_directories_df DataFrame on the device_id column, using a left join
combined_df = df.join(model_directories_df, on="device_id", how="left")
# Display the combined DataFrame
display(combined_df)

# COMMAND ----------

# Define the schema for the return DataFrame
apply_return_schema = "record_id integer, prediction float"

def apply_model(df_pandas: pd.DataFrame) -> pd.DataFrame:
    """
    Applies model to data for a particular device, represented as a pandas DataFrame
    """
    # Get the model path for the device
    model_path = df_pandas["model_path"].iloc[0]

    # Select the input columns for the model
    input_columns = ["feature_1", "feature_2", "feature_3"]
    X = df_pandas[input_columns]

    # Load the pre-trained model
    model = mlflow.sklearn.load_model(model_path)

    # Make predictions using the model
    prediction = model.predict(X)

    # Create a new DataFrame with the record_id and prediction columns
    return_df = pd.DataFrame({
        "record_id": df_pandas["record_id"],
        "prediction": prediction
    })
    return return_df

# Apply the model to each device in the combined_df DataFrame
prediction_df = combined_df.groupby("device_id").applyInPandas(apply_model, schema=apply_return_schema)

# Display the prediction DataFrame
display(prediction_df)

# COMMAND ----------

# Get the experiment ID from the current run
experiment_id = run.info.experiment_id

# Read the model data from the MLflow experiment using the experiment ID
model_df = (spark.read.format("mlflow-experiment")
            .load(experiment_id)
            
            # Filter the data to exclude records where the device tag is missing
            .filter("tags.device IS NOT NULL")
            
            # Sort the data by end_time in descending order
            .orderBy("end_time", ascending=False)
            
            # Select only the device and run_id columns
            .select("tags.device", "run_id")
            
            # Limit the number of records to 10
            .limit(10))

# Display the model_df DataFrame
display(model_df)

# COMMAND ----------

# Create a dictionary called device_to_model
# The keys of the dictionary will be the device IDs
# The values of the dictionary will be the models loaded from MLflow using the run_id and device ID
# Iterate over each row in the model_df DataFrame using the collect() method
# For each row, load the model using mlflow.sklearn.load_model() with the run_id and device ID
# Assign the loaded model to the corresponding device ID in the device_to_model dictionary
device_to_model = {row["device"]: mlflow.sklearn.load_model(f"runs:/{row['run_id']}/{row['device']}") for row in model_df.collect()}

# Display the device_to_model dictionary
device_to_model

# COMMAND ----------

from mlflow.pyfunc import PythonModel

class OriginDelegatingModel(PythonModel):
    
    def __init__(self, device_to_model_map):
        '''
        This method initializes the OriginDelegatingModel with a device_to_model_map
        which maps device IDs to the appropriate ML models
        '''
        self.device_to_model_map = device_to_model_map
        
    def predict_for_device(self, row):
        '''
        This method applies to a single row of data by
        fetching the appropriate model from the device_to_model_map
        and generating predictions using the model for the given row of data
        '''
        model = self.device_to_model_map.get(str(row["device_id"]))
        
        # Extract the features from the row of data
        features = row[["feature_1", "feature_2", "feature_3"]].to_frame().T
        
        # Generate predictions using the model for the given row of data
        predictions = model.predict(features)[0]
        
        return predictions
    
    def predict(self, model_input):
        '''
        This method generates predictions for the given model_input
        by applying the predict_for_device method to each row of the input
        '''
        return model_input.apply(self.predict_for_device, axis=1)

# COMMAND ----------

# Create an instance of the OriginDelegatingModel class called example_model
# Pass the device_to_model dictionary as an argument to the constructor of OriginDelegatingModel
example_model = OriginDelegatingModel(device_to_model)

# Convert the combined_df Spark DataFrame to a Pandas DataFrame using the toPandas() method
# Retrieve the first 20 rows of the Pandas DataFrame using the head(20) method
# Apply the predict method of the example_model to the first 20 rows of the combined_df DataFrame
# This will generate predictions for each row using the appropriate ML model based on the device ID
# The result will be a Pandas series of predictions
# Note that this code assumes that the combined_df DataFrame has a similar structure as the df DataFrame with columns: record_id, device_id, feature_1, feature_2, feature_3, and label
example_model.predict(combined_df.toPandas().head(20))

# COMMAND ----------

input_example = combined_df.toPandas().head(5)

predictions = example_model.predict(input_example)

# COMMAND ----------

import cloudpickle

print(cloudpickle.__version__)

# COMMAND ----------

from mlflow.models import infer_signature
import sklearn

mlflow.set_registry_uri("databricks-uc")

model_name = "andrewcooley_catalog.models.groupedrf"

signature = infer_signature(input_example, predictions)

with mlflow.start_run():
    model = OriginDelegatingModel(device_to_model)
    mlflow.pyfunc.log_model("model", 
                            python_model=model,
                            registered_model_name=model_name,
                            signature=signature,
                            input_example=input_example,
                            extra_pip_requirements=["sklearn=={}".format(sklearn.__version__)],)

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


