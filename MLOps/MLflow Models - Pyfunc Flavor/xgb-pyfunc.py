# Databricks notebook source
import mlflow
mlflow.autolog(
    log_input_examples=False,
    log_model_signatures=False,
    log_models=False,
    disable=True,
    exclusive=True,
    disable_for_unsupported_versions=True,
    silent=False
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Train and save model to path

# COMMAND ----------

# Load training and test datasets
from sys import version_info
import xgboost as xgb
from sklearn import datasets
from sklearn.model_selection import train_test_split

PYTHON_VERSION = "{major}.{minor}.{micro}".format(
    major=version_info.major, minor=version_info.minor, micro=version_info.micro
)
iris = datasets.load_iris()
x = iris.data[:, 2:]
y = iris.target
x_train, x_test, y_train, _ = train_test_split(x, y, test_size=0.2, random_state=42)
dtrain = xgb.DMatrix(x_train, label=y_train)

# Train and save an XGBoost model
xgb_model = xgb.train(params={"max_depth": 10}, dtrain=dtrain, num_boost_round=10)
xgb_model_path = "xgb_model.pth"
xgb_model.save_model(xgb_model_path)

# Create an `artifacts` dictionary that assigns a unique name to the saved XGBoost model file.
# This dictionary will be passed to `mlflow.pyfunc.save_model`, which will copy the model file
# into the new MLflow Model's directory.
artifacts = {"xgb_model": xgb_model_path}

# COMMAND ----------

# MAGIC %md
# MAGIC ### Create custom pyfunc model and package dependencies into conda environment

# COMMAND ----------

# Define the model class
import mlflow.pyfunc


class XGBWrapper(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        import xgboost as xgb

        self.xgb_model = xgb.Booster()
        self.xgb_model.load_model(context.artifacts["xgb_model"])

    def predict(self, context, model_input):
        input_matrix = xgb.DMatrix(model_input.values)
        return self.xgb_model.predict(input_matrix)


# Create a Conda environment for the new MLflow Model that contains all necessary dependencies.
import cloudpickle

conda_env = {
    "channels": ["defaults"],
    "dependencies": [
        "python={}".format(PYTHON_VERSION),
        "pip",
        {
            "pip": [
                "mlflow=={}".format(mlflow.__version__),
                "xgboost=={}".format(xgb.__version__),
                "cloudpickle=={}".format(cloudpickle.__version__),
            ],
        },
    ],
    "name": "xgb_env",
}

# COMMAND ----------

# MAGIC %md
# MAGIC ### log custom pyfunc model to MLflow Tracking and load back for local inference

# COMMAND ----------

from mlflow.models.signature import infer_signature
import pandas as pd

y_pred = xgb_model.predict(xgb.DMatrix(pd.DataFrame(x_test)))

# Infer model signature
signature = infer_signature(pd.DataFrame(x_test), y_pred)

input_example = {
    "0": 1.0,
    "1": 1.5
}

# Save the MLflow Model
mlflow_pyfunc_model_path = "xgb_mlflow_pyfunc"

with mlflow.start_run():
    model_info = mlflow.pyfunc.log_model(artifact_path=mlflow_pyfunc_model_path, 
                                         python_model=XGBWrapper(), 
                                         conda_env=conda_env,
                                         artifacts=artifacts,
                                         signature=signature,
                                         input_example=input_example,
                                         registered_model_name="xgb-clf-pyfunc-model")

# Load the model in `python_function` format
loaded_model = mlflow.pyfunc.load_model(model_uri=model_info.model_uri)

# Evaluate the model
test_predictions = loaded_model.predict(pd.DataFrame(x_test))
print(test_predictions)
