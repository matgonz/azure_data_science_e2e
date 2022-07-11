# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC # Batch Inference

# COMMAND ----------

# MAGIC %md
# MAGIC # 1. Load Data

# COMMAND ----------

from databricks.feature_store import FeatureStoreClient
import datetime

fs = FeatureStoreClient()

retrieve_date = datetime.date.today()
customer_features_df = fs.read_table(name='fs_ecommerce.churn', as_of_delta_timestamp=str(retrieve_date))

#customer_features_df = fs.read_table(name='fs_ecommerce.churn')

customer_features_pd = customer_features_df.toPandas()

# COMMAND ----------

# MAGIC %md
# MAGIC - customer_features_df: PySpark DataFrame
# MAGIC - customer_features_pd: Pandas DataFrame

# COMMAND ----------

_drop = ['CustomerID','Churn']
features_spark_df = customer_features_df.drop(*_drop)
features_pandas_df = customer_features_pd.drop(_drop, axis=1)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Inference
# MAGIC 
# MAGIC ### Load from Model Registry

# COMMAND ----------

# MAGIC %md
# MAGIC Predict on a Spark DataFrame:

# COMMAND ----------

import mlflow
logged_model = 'runs:/9caf0c05424c42e99dee6519d2ec1b74/model'

# Load model as a Spark UDF. Override result_type if the model does not return double values.
loaded_model = mlflow.pyfunc.spark_udf(spark, model_uri=logged_model, result_type='double')

# Predict on a Spark DataFrame.
columns = list(features_spark_df.columns)
customer_features_df = customer_features_df.withColumn('predictions', loaded_model(*columns))

display(customer_features_df)

# COMMAND ----------

# MAGIC %md
# MAGIC Predict on a Pandas DataFrame:

# COMMAND ----------

import mlflow
logged_model = 'runs:/9caf0c05424c42e99dee6519d2ec1b74/model'

# Load model as a PyFuncModel.
loaded_model = mlflow.pyfunc.load_model(logged_model)

# Predict on a Pandas DataFrame.
import pandas as pd
customer_features_pd['predictions'] = loaded_model.predict(features_pandas_df)

display(customer_features_pd)