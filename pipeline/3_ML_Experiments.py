# Databricks notebook source
# MAGIC %md
# MAGIC # ML Experiments
# MAGIC 
# MAGIC So, the last we used AutoML to create a baseline model. Now, we'll to analyse metrics, the bias variance tradeoff and propose the best solution for this problem.
# MAGIC 
# MAGIC **AutoML**
# MAGIC - [MLflow experiments](#mlflow/experiments/713584551057122/s?orderByKey=metrics.%60val_f1_score%60&orderByAsc=false)
# MAGIC - [Best Model](#mlflow/experiments/713584551057122/runs/de7019603a78477ab6391adf466f1a80)

# COMMAND ----------

# MAGIC %md
# MAGIC # 1. Libraries

# COMMAND ----------

import os
import uuid
import shutil
import pandas as pd
import collections

import mlflow
from mlflow.tracking import MlflowClient
from databricks.automl_runtime.sklearn.column_selector import ColumnSelector
import databricks.automl_runtime

import sklearn
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn import set_config

set_config(display="diagram")

# COMMAND ----------

# MAGIC %md
# MAGIC # 2. Load Data
# MAGIC 
# MAGIC We'll download the artifacts created by AutoML Experiments to use here.

# COMMAND ----------

# Create temp directory to download input data from MLflow
input_temp_dir = os.path.join(os.environ["SPARK_LOCAL_DIRS"], "tmp", str(uuid.uuid4())[:8])
os.makedirs(input_temp_dir)

# Download the artifact and read it into a pandas DataFrame
input_client = MlflowClient()
input_data_path = input_client.download_artifacts("8a47aae0f21648beaa8df758e517c597", "data", input_temp_dir)

df_loaded = pd.read_parquet(os.path.join(input_data_path, "training_data"))
# Delete the temp data
shutil.rmtree(input_temp_dir)

# Preview data
df_loaded.head(5)

# COMMAND ----------

# MAGIC %md
# MAGIC # 3. Train - Validation - Test Split
# MAGIC 
# MAGIC Split the input data into 3 sets:
# MAGIC - Train (60% of the dataset used to train the model)
# MAGIC - Validation (20% of the dataset used to tune the hyperparameters of the model)
# MAGIC - Test (20% of the dataset used to report the true performance of the model on an unseen dataset)

# COMMAND ----------

target_col = "Churn"

split_X = df_loaded.drop([target_col], axis=1)
split_y = df_loaded[target_col]

# Split out train data
X_train, split_X_rem, y_train, split_y_rem = train_test_split(split_X, split_y, train_size=0.6, random_state=591423154, stratify=split_y)

# Split remaining data equally for validation and test
X_val, X_test, y_val, y_test = train_test_split(split_X_rem, split_y_rem, test_size=0.5, random_state=591423154, stratify=split_y_rem)

# COMMAND ----------

# Check distribution
dist_arr = []
arr_y = [('train', y_train), ('val', y_val), ('test', y_test)]

for dataset_name, dist in arr_y:
    
    c = collections.Counter(dist)
    total = c[0] + c[1]
    dist_arr.append({'dataset':dataset_name, '0': c[0] / total, '1': c[1] / total})

pd.DataFrame(dist_arr)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC # 3. Preprocessors
# MAGIC 
# MAGIC Defining preprocessors of pipeline

# COMMAND ----------

# Select supported columns
supported_cols = ['Tenure', 'WarehouseToHome', 'HourSpendOnApp',
       'NumberOfDeviceRegistered', 'SatisfactionScore', 'NumberOfAddress',
       'Complain', 'OrderAmountHikeFromlastYear', 'CouponUsed', 'OrderCount',
       'DaySinceLastOrder', 'CashbackAmount', 'PreferredLoginDevice_Computer',
       'PreferredLoginDevice_Mobile_Phone', 'PreferredLoginDevice_Phone',
       'CityTier_1', 'CityTier_2', 'CityTier_3', 'PreferredPaymentMode_CC',
       'PreferredPaymentMode_COD', 'PreferredPaymentMode_Cash_on_Delivery',
       'PreferredPaymentMode_Credit_Card', 'PreferredPaymentMode_Debit_Card',
       'PreferredPaymentMode_E_wallet', 'PreferredPaymentMode_UPI',
       'Gender_Female', 'Gender_Male', 'PreferedOrderCat_Fashion',
       'PreferedOrderCat_Grocery', 'PreferedOrderCat_Laptop_&_Accessory',
       'PreferedOrderCat_Mobile', 'PreferedOrderCat_Mobile_Phone',
       'PreferedOrderCat_Others', 'MaritalStatus_Divorced',
       'MaritalStatus_Married', 'MaritalStatus_Single']
col_selector = ColumnSelector(supported_cols)

# Transformers
transformers = []
# Numerical Pipeline
numerical_pipeline = Pipeline(steps=[
        # Cast to numeric values
        ("converter", FunctionTransformer(lambda df: df.apply(pd.to_numeric, errors="coerce"))),
        # Mean and Standard deviation
        ("standardizer", StandardScaler())])
# Transformers
transformers.append(("numerical", numerical_pipeline, 
                     ["WarehouseToHome", "CashbackAmount", "DaySinceLastOrder", "Tenure", "OrderAmountHikeFromlastYear"]))
# Preprocessor
preprocessor = ColumnTransformer(transformers, remainder="passthrough", sparse_threshold=0)

# COMMAND ----------

# MAGIC %md
# MAGIC **Pipeline**

# COMMAND ----------

# Create a separate pipeline to transform the validation dataset. This is used for early stopping.
pipeline = Pipeline([
    ("column_selector", col_selector),
    ("preprocessor", preprocessor)])

mlflow.sklearn.autolog(disable=True)
pipeline.fit(X_train, y_train)
X_val_processed = pipeline.transform(X_val)
pipeline

# COMMAND ----------

# MAGIC %md
# MAGIC ## Train classification model
# MAGIC 
# MAGIC - Read about the model LGBMClassifier and the parameters used by AutoML
# MAGIC - Which others algorithms can be used for a binary classification?
# MAGIC - Plot metrics without log mlflot
# MAGIC - The bias variance tradeoff analysis (course)

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

from lightgbm import LGBMClassifier

#help(LGBMClassifier)

# COMMAND ----------

import mlflow
import sklearn
from sklearn import set_config
from sklearn.pipeline import Pipeline

set_config(display="diagram")

lgbmc_classifier = LGBMClassifier(
  colsample_bytree=0.5438762801993433,
  lambda_l1=2.80383773385808,
  lambda_l2=5.401710572056857,
  learning_rate=0.03280149487316273,
  max_bin=415,
  max_depth=9,
  min_child_samples=80,
  n_estimators=1499,
  num_leaves=205,
  path_smooth=16.164096261021655,
  subsample=0.7728650113993107,
  random_state=591423154,
)

model = Pipeline([
    ("column_selector", col_selector),
    ("preprocessor", preprocessor),
    ("classifier", lgbmc_classifier),
])

# Create a separate pipeline to transform the validation dataset. This is used for early stopping.
pipeline = Pipeline([
    ("column_selector", col_selector),
    ("preprocessor", preprocessor),
])

mlflow.sklearn.autolog(disable=True)
pipeline.fit(X_train, y_train)
X_val_processed = pipeline.transform(X_val)

model

# COMMAND ----------

# Enable automatic logging of input samples, metrics, parameters, and models
mlflow.sklearn.autolog(log_input_examples=True, silent=True)

with mlflow.start_run(experiment_id="713584551057122", run_name="lightgbm") as mlflow_run:
    model.fit(X_train, y_train, classifier__early_stopping_rounds=5, classifier__eval_set=[(X_val_processed,y_val)], classifier__verbose=False)
    
    # Training metrics are logged by MLflow autologging
    # Log metrics for the validation set
    lgbmc_val_metrics = mlflow.sklearn.eval_and_log_metrics(model, X_val, y_val, prefix="val_")

    # Log metrics for the test set
    lgbmc_test_metrics = mlflow.sklearn.eval_and_log_metrics(model, X_test, y_test, prefix="test_")

    # Display the logged metrics
    lgbmc_val_metrics = {k.replace("val_", ""): v for k, v in lgbmc_val_metrics.items()}
    lgbmc_test_metrics = {k.replace("test_", ""): v for k, v in lgbmc_test_metrics.items()}
    display(pd.DataFrame([lgbmc_val_metrics, lgbmc_test_metrics], index=["validation", "test"]))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Feature importance
# MAGIC 
# MAGIC SHAP is a game-theoretic approach to explain machine learning models, providing a summary plot
# MAGIC of the relationship between features and model output. Features are ranked in descending order of
# MAGIC importance, and impact/color describe the correlation between the feature and the target variable.
# MAGIC - Generating SHAP feature importance is a very memory intensive operation, so to ensure that AutoML can run trials without
# MAGIC   running out of memory, we disable SHAP by default.<br />
# MAGIC   You can set the flag defined below to `shap_enabled = True` and re-run this notebook to see the SHAP plots.
# MAGIC - To reduce the computational overhead of each trial, a single example is sampled from the validation set to explain.<br />
# MAGIC   For more thorough results, increase the sample size of explanations, or provide your own examples to explain.
# MAGIC - SHAP cannot explain models using data with nulls; if your dataset has any, both the background data and
# MAGIC   examples to explain will be imputed using the mode (most frequent values). This affects the computed
# MAGIC   SHAP values, as the imputed samples may not match the actual data distribution.
# MAGIC 
# MAGIC For more information on how to read Shapley values, see the [SHAP documentation](https://shap.readthedocs.io/en/latest/example_notebooks/overviews/An%20introduction%20to%20explainable%20AI%20with%20Shapley%20values.html).

# COMMAND ----------

# Set this flag to True and re-run the notebook to see the SHAP plots
shap_enabled = False

# COMMAND ----------

if shap_enabled:
    from shap import KernelExplainer, summary_plot
    # Sample background data for SHAP Explainer. Increase the sample size to reduce variance.
    train_sample = X_train.sample(n=min(100, X_train.shape[0]))

    # Sample some rows from the validation set to explain. Increase the sample size for more thorough results.
    example = X_val.sample(n=min(10, X_val.shape[0]))

    # Use Kernel SHAP to explain feature importance on the example from the validation set.
    predict = lambda x: model.predict(pd.DataFrame(x, columns=X_train.columns))
    explainer = KernelExplainer(predict, train_sample, link="identity")
    shap_values = explainer.shap_values(example, l1_reg=False)
    summary_plot(shap_values, example, class_names=model.classes_)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Inference
# MAGIC [The MLflow Model Registry](https://docs.databricks.com/applications/mlflow/model-registry.html) is a collaborative hub where teams can share ML models, work together from experimentation to online testing and production, integrate with approval and governance workflows, and monitor ML deployments and their performance. The snippets below show how to add the model trained in this notebook to the model registry and to retrieve it later for inference.
# MAGIC 
# MAGIC > **NOTE:** The `model_uri` for the model already trained in this notebook can be found in the cell below
# MAGIC 
# MAGIC ### Register to Model Registry
# MAGIC ```
# MAGIC model_name = "Example"
# MAGIC 
# MAGIC model_uri = f"runs:/{ mlflow_run.info.run_id }/model"
# MAGIC registered_model_version = mlflow.register_model(model_uri, model_name)
# MAGIC ```
# MAGIC 
# MAGIC ### Load from Model Registry
# MAGIC ```
# MAGIC model_name = "Example"
# MAGIC model_version = registered_model_version.version
# MAGIC 
# MAGIC model = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/{model_version}")
# MAGIC model.predict(input_X)
# MAGIC ```
# MAGIC 
# MAGIC ### Load model without registering
# MAGIC ```
# MAGIC model_uri = f"runs:/{ mlflow_run.info.run_id }/model"
# MAGIC 
# MAGIC model = mlflow.pyfunc.load_model(model_uri)
# MAGIC model.predict(input_X)
# MAGIC ```

# COMMAND ----------

# model_uri for the generated model
print(f"runs:/{ mlflow_run.info.run_id }/model")
