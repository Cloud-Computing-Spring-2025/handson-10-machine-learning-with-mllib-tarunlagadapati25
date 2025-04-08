from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, ChiSqSelector
from pyspark.ml.classification import LogisticRegression, DecisionTreeClassifier, RandomForestClassifier, GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml import Pipeline

# Initialize Spark session
spark = SparkSession.builder.appName("CustomerChurnMLlib").getOrCreate()

# Load dataset
data_path = "customer_churn.csv"
df = spark.read.csv(data_path, header=True, inferSchema=True)

# Task 1: Data Preprocessing and Feature Engineering
def preprocess_data(df):
    df = df.fillna({"TotalCharges": 0})

    categorical_cols = ["gender", "PhoneService", "InternetService"]
    indexers = [StringIndexer(inputCol=col, outputCol=col + "_Index", handleInvalid="keep") for col in categorical_cols]
    encoders = [OneHotEncoder(inputCol=col + "_Index", outputCol=col + "_Vec") for col in categorical_cols]

    feature_cols = ["tenure", "MonthlyCharges", "TotalCharges"] + [col + "_Vec" for col in categorical_cols]
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")

    pipeline = Pipeline(stages=indexers + encoders + [assembler])
    model = pipeline.fit(df)
    final_df = model.transform(df)

    final_df.select("features", "Churn").write.mode("overwrite").parquet("outputs/task1/output")
    return final_df

# Task 2: Splitting Data and Building a Logistic Regression Model
def train_logistic_regression_model(df):
    indexer = StringIndexer(inputCol="Churn", outputCol="label")
    df = indexer.fit(df).transform(df)

    train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)

    lr = LogisticRegression(featuresCol="features", labelCol="label")
    model = lr.fit(train_df)
    predictions = model.transform(test_df)

    evaluator = BinaryClassificationEvaluator()
    auc = evaluator.evaluate(predictions)

    predictions.write.mode("overwrite").parquet("outputs/task2/output")
    print(f"Logistic Regression AUC: {auc}")

# Task 3: Feature Selection Using Chi-Square Test
def feature_selection(df):
    indexer = StringIndexer(inputCol="Churn", outputCol="label")
    df = indexer.fit(df).transform(df)

    selector = ChiSqSelector(numTopFeatures=5, featuresCol="features", labelCol="label", outputCol="selectedFeatures")
    result = selector.fit(df).transform(df)

    result.select("selectedFeatures", "label").write.mode("overwrite").parquet("outputs/task3/output")

# Task 4: Hyperparameter Tuning with Cross-Validation for Multiple Models
def tune_and_compare_models(df):
    indexer = StringIndexer(inputCol="Churn", outputCol="label")
    df = indexer.fit(df).transform(df)

    train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)

    models = {
        "LogisticRegression": LogisticRegression(),
        "DecisionTree": DecisionTreeClassifier(),
        "RandomForest": RandomForestClassifier(),
        "GBT": GBTClassifier()
    }

    param_grids = {
        "LogisticRegression": ParamGridBuilder().addGrid(models["LogisticRegression"].regParam, [0.01, 0.1]).build(),
        "DecisionTree": ParamGridBuilder().addGrid(models["DecisionTree"].maxDepth, [3, 5, 7]).build(),
        "RandomForest": ParamGridBuilder().addGrid(models["RandomForest"].numTrees, [10, 20]).build(),
        "GBT": ParamGridBuilder().addGrid(models["GBT"].maxIter, [10, 20]).build(),
    }

    best_model = None
    best_auc = 0
    best_name = ""

    evaluator = BinaryClassificationEvaluator()

    for name, model in models.items():
        grid = param_grids[name]
        cv = CrossValidator(estimator=model, estimatorParamMaps=grid, evaluator=evaluator, numFolds=5)
        cv_model = cv.fit(train_df)
        predictions = cv_model.transform(test_df)
        auc = evaluator.evaluate(predictions)
        print(f"{name} AUC: {auc}")

        predictions.write.mode("overwrite").parquet(f"outputs/task4/{name}_predictions")

        if auc > best_auc:
            best_auc = auc
            best_model = cv_model.bestModel
            best_name = name

    print(f"Best Model: {best_name} with AUC: {best_auc}")

# Execute tasks
preprocessed_df = preprocess_data(df)
train_logistic_regression_model(preprocessed_df)
feature_selection(preprocessed_df)
tune_and_compare_models(preprocessed_df)

# Stop Spark session
spark.stop()