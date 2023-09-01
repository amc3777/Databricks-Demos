from pyspark.sql import functions as F

volumes_path = dbutils.jobs.taskValues.get(taskKey = "01-save-raw-data-task", key = "volumes_path")

ID_COL = "ID"
LABEL_COL = "price"
CATALOG = "andrewcooleycatalog"
SCHEMA = "andrewcooleyschema"
TRAINING = f"{CATALOG}.{SCHEMA}.airbnb_price_prediction_training"
TEST = f"{CATALOG}.{SCHEMA}.airbnb_price_prediction_test"
INFERENCE = f"{CATALOG}.{SCHEMA}.airbnb_price_prediction_inference"

raw_df = (spark.read.format("parquet")
  .load(volumes_path))

features_list = ["bedrooms", "neighbourhood_cleansed", "accommodates", "cancellation_policy", "beds", "host_is_superhost", "property_type", "minimum_nights", "bathrooms", "host_total_listings_count", "number_of_reviews", "review_scores_value", "review_scores_cleanliness"]

train_df, test_df, inference_df = raw_df.select(*features_list+[ID_COL, LABEL_COL]).randomSplit(weights=[0.6, 0.2, 0.2], seed=42)

(train_df
  .write
  .format("delta")
  .mode("overwrite")
  .option("overwriteSchema",True)
  .saveAsTable(TRAINING)
)

(test_df
  .write
  .format("delta")
  .mode("overwrite")
  .option("overwriteSchema",True)
  .option("delta.enableChangeDataFeed", "true")
  .saveAsTable(TEST)
)

(inference_df
  .write
  .format("delta")
  .mode("overwrite")
  .option("overwriteSchema",True)
  .saveAsTable(INFERENCE)
)

dbutils.jobs.taskValues.set(key = 'training_path', value = TRAINING)
dbutils.jobs.taskValues.set(key = 'test_path', value = TEST)
dbutils.jobs.taskValues.set(key = 'inference_path', value = INFERENCE)