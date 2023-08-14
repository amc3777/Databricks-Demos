# Databricks notebook source
import warnings
warnings.filterwarnings("ignore")

# COMMAND ----------

from databricks.feature_store.online_store_spec import AmazonDynamoDBSpec

# secret_access_key = dbutils.secrets.get(scope="scope", key="secret-key")
 
online_store_spec = AmazonDynamoDBSpec(
  region="us-west-2",
#   access_key_id="access-key_id",
#   secret_access_key=secret_access_key,
  table_name = "online_airbnb_sf_listings_features"
)

try:

  fs.publish_table(
    "andrewcooleycatalog.airbnb_data.airbnb_sf_listings_features", 
    online_store_spec
  )

except Exception:

  online_store = False
  print("Permissions on Amazon DynamoDB to publish to an online feature store are missing. Online feature store not created.")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Get model version by alias

# COMMAND ----------

import mlflow
from mlflow.tracking.client import MlflowClient

mlflow.set_registry_uri('databricks-uc')

client = MlflowClient()

if not online_store:

  model_name = 'andrewcooleycatalog.airbnb_data.airbnb_sf_listings_price_predictor'
  model_serving_endpoint_name ='ab_airbnb_sf_listings_price_predictor_endpoint'

else:

  model_name = 'andrewcooleycatalog.airbnb_data.fs_airbnb_sf_listings_price_predictor'
  model_serving_endpoint_name ='ab_fs_airbnb_sf_listings_price_predictor_endpoint'

def get_latest_model_version(model_name: str):
  alias_mv = client.get_model_version_by_alias(model_name, "candidate")
  return alias_mv.version

# COMMAND ----------

# MAGIC %md
# MAGIC ### Build config for API request to create Model Serving endpoint

# COMMAND ----------

import requests

token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().getOrElse(None)
 
headers = {
    "Authorization": f"Bearer {token}",
    "Content-Type": "application/json"
  }
 
java_tags = dbutils.notebook.entry_point.getDbutils().notebook().getContext().tags()
tags = sc._jvm.scala.collection.JavaConversions.mapAsJavaMap(java_tags)
instance = tags["browserHostName"]
 
my_json = {
    "name": "ab_airbnb_sf_listings_price_predictor_endpoint",
    "config": {
        "served_models": [
            {
                "name": "baseline",
                "model_name": "andrewcooleycatalog.airbnb_data.airbnb_sf_listings_price_predictor",
                "model_version": "7",
                "workload_size": "Small",
                "scale_to_zero_enabled":True
            },
            {
                "name": "candidate",
                "model_name": "andrewcooleycatalog.airbnb_data.airbnb_sf_listings_price_predictor",
                "model_version": "9",
                "workload_size": "Small",
                "scale_to_zero_enabled":True
            }
        ],
        "traffic_config": {
            "routes": [
                {
                    "served_model_name": "baseline",
                    "traffic_percentage": "90"
                },
                {
                    "served_model_name": "candidate",
                    "traffic_percentage": "10"
                }
            ]
        }
    }
}

# COMMAND ----------

# MAGIC %md
# MAGIC ### Build functions for CRUD operations on a Model Serving endpoint

# COMMAND ----------

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

def func_delete_model_serving_endpoint(model_serving_endpoint_name):

  endpoint_url = f"https://{instance}/api/2.0/serving-endpoints"
  url =  f"{endpoint_url}/{model_serving_endpoint_name}" 
  response = requests.delete(url, headers=headers)

  if response.status_code != 200:

    raise Exception(f"Request failed with status {response.status_code}, {response.text}")

  else:
    
    print(model_serving_endpoint_name, "endpoint is deleted!")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Create Model Serving endpoint

# COMMAND ----------

func_create_endpoint(model_serving_endpoint_name)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Wait for Model Serving endpoint to be up and ready

# COMMAND ----------

import time, mlflow
 
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

# MAGIC %md
# MAGIC ### Create Model Serving input in scoring request format

# COMMAND ----------

from databricks import feature_store

fs = feature_store.FeatureStoreClient()

input_df = fs.read_table(name="andrewcooleycatalog.airbnb_data.airbnb_sf_listings_features").limit(5)

input_df = input_df.toPandas().to_dict(orient="split")

payload_json = {"dataframe_split": input_df}

display(payload_json)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Score model using the Model Serving endpoint

# COMMAND ----------

def score_model(data_json: dict):
    url =  f"https://{instance}/serving-endpoints/{model_serving_endpoint_name}/invocations"
    response = requests.request(method="POST", headers=headers, url=url, json=data_json)
    if response.status_code != 200:
        raise Exception(f"Request failed with status {response.status_code}, {response.text}")
    return response.json()
  
print(score_model(payload_json))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Create DBSQL AI Function

# COMMAND ----------

# TODO

# COMMAND ----------

# MAGIC %md
# MAGIC ### (Optional) Delete Model Serving endpoint

# COMMAND ----------

# func_delete_model_serving_endpoint(model_serving_endpoint_name)
