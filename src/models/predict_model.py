# Databricks notebook source
# MAGIC %run ../config_env

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC # Batch Inference

# COMMAND ----------

import mlflow
from databricks.feature_store import FeatureStoreClient
import datetime

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Data

# COMMAND ----------

fs = FeatureStoreClient()

retrieve_date = datetime.date.today()

#customer_features_df = fs.read_table(name=feature_store_db_name_and_table, as_of_delta_timestamp=str(retrieve_date))
customer_features_df = fs.read_table(name=feature_store_db_name_and_table)

# COMMAND ----------

_drop = ['Churn']
features_spark_df = customer_features_df.drop(*_drop)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Inference
# MAGIC 
# MAGIC ### Load from Model Registry
# MAGIC 
# MAGIC ```
# MAGIC logged_model = 'models:/churn_prediction/production'
# MAGIC loaded_model = mlflow.pyfunc.load_model(logged_model)
# MAGIC ```

# COMMAND ----------

prediction = fs.score_batch(model_uri="models:/churn_prediction/production", df=features_spark_df)
display(prediction)
