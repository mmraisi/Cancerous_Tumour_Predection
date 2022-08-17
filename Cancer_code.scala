// ------Logistic Regression ML -----

// first let's import all the required libraries

import org.apache.spark.sql.functions._
import org.apache.spark.ml.feature.{VectorAssembler}
import org.apache.spark.ml.Pipeline
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.tuning.{CrossValidator, CrossValidatorModel, ParamGridBuilder}
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.sql.types.{IntegerType, DoubleType}

import org.apache.spark.sql.functions.{expr, col, column, min, max, desc, avg, when, count}
import org.apache.spark.sql.types._

//let's define a schema for our dataset 
//-- we will be changing the column names and remove the spaces between the words in the columns by the schema as follows

val Cancer_schema_Mahmoud = StructType(Array(
		
		StructField("Id", IntegerType, true),
		StructField("ClumpThickness", IntegerType, true),
		StructField("UofCSize", IntegerType,true),
		StructField("UofCShape", IntegerType,true),
		StructField("MarginalAdhesion", IntegerType,true),
		StructField("SECSize", IntegerType,true),
		StructField("BareNuclei", IntegerType,true),
		StructField("BlandChromatin", IntegerType,true),
		StructField("NormalNucleoli", IntegerType,true),
        StructField("Mitoses", IntegerType,true),
        StructField("Class", IntegerType,true)
        ))

// let load the dataset

val Cancer_Mahmoud = spark
  .read.format("csv")
  .option("header", "true")
  .schema(Cancer_schema_Mahmoud)
  .load("hdfs://10.142.0.4:8020/MidTerm/cancer.csv")

// next let's print the first 10 records to check the dataset


Cancer_Mahmoud.show(10)

/*
+-------+--------------+--------+---------+----------------+-------+----------+--------------+--------------+-------+-----+
|     Id|ClumpThickness|UofCSize|UofCShape|MarginalAdhesion|SECSize|BareNuclei|BlandChromatin|NormalNucleoli|Mitoses|Class|
+-------+--------------+--------+---------+----------------+-------+----------+--------------+--------------+-------+-----+
|1000025|             5|       1|        1|               1|      2|         1|             3|             1|      1|    2|
|1002945|             5|       4|        4|               5|      7|        10|             3|             2|      1|    2|
|1015425|             3|       1|        1|               1|      2|         2|             3|             1|      1|    2|
|1016277|             6|       8|        8|               1|      3|         4|             3|             7|      1|    2|
|1017023|             4|       1|        1|               3|      2|         1|             3|             1|      1|    2|
|1017122|             8|      10|       10|               8|      7|        10|             9|             7|      1|    4|
|1018099|             1|       1|        1|               1|      2|        10|             3|             1|      1|    2|
|1018561|             2|       1|        2|               1|      2|         1|             3|             1|      1|    2|
|1033078|             2|       1|        1|               1|      2|         1|             1|             1|      5|    2|
|1033078|             4|       2|        1|               1|      2|         1|             2|             1|      1|    2|
+-------+--------------+--------+---------+----------------+-------+----------+--------------+--------------+-------+-----+
*/

// let's check the schema of the dataset and make sure that we have the right type for each column

Cancer_Mahmoud.printSchema()

/*
root
 |-- Id: integer (nullable = true)
 |-- ClumpThickness: integer (nullable = true)
 |-- UofCSize: integer (nullable = true)
 |-- UofCShape: integer (nullable = true)
 |-- MarginalAdhesion: integer (nullable = true)
 |-- SECSize: integer (nullable = true)
 |-- BareNuclei: integer (nullable = true)
 |-- BlandChromatin: integer (nullable = true)
 |-- NormalNucleoli: integer (nullable = true)
 |-- Mitoses: integer (nullable = true)
 |-- Class: integer (nullable = true)
*/

// feature engineering part

// checking for missing values
import org.apache.spark.sql.Column

def countMissingValues_Mahmoud(columns:Array[String]):Array[Column]={
    columns.map(columnName=>{
      count(when(col(columnName).isNull, columnName)).alias(columnName)
    })
}
Cancer_Mahmoud.select(countMissingValues_Mahmoud(Cancer_Mahmoud.columns):_*).show()

/*
+---+--------------+--------+---------+----------------+-------+----------+--------------+--------------+-------+-----+
| Id|ClumpThickness|UofCSize|UofCShape|MarginalAdhesion|SECSize|BareNuclei|BlandChromatin|NormalNucleoli|Mitoses|Class|
+---+--------------+--------+---------+----------------+-------+----------+--------------+--------------+-------+-----+
|  0|             0|       0|        0|               0|      0|         0|             0|             0|      0|    0|
+---+--------------+--------+---------+----------------+-------+----------+--------------+--------------+-------+-----+
*/

// first let's remove the identifier column for the data. 
//This will not be much help in feeding it into a machine learning model, so we will be removing that.

val Cancer_Mahmoud_no_Id =  Cancer_Mahmoud.drop("Id")

// next we will be changing the class column to 0/1 instead of 2/4 values
// 2 is for benign which will be 0, and 4 for malignant which will be 1
val Cancer_Mahmoud_labeled =
  Cancer_Mahmoud_no_Id.withColumn(
    "Class",
    when(col("Class") === 2, 0
	).otherwise(1)
  );

// let's look at the dataset before starting to build the ML model
/*
+--------------+--------+---------+----------------+-------+----------+--------------+--------------+-------+-----+
|ClumpThickness|UofCSize|UofCShape|MarginalAdhesion|SECSize|BareNuclei|BlandChromatin|NormalNucleoli|Mitoses|Class|
+--------------+--------+---------+----------------+-------+----------+--------------+--------------+-------+-----+
|             5|       1|        1|               1|      2|         1|             3|             1|      1|    0|
|             5|       4|        4|               5|      7|        10|             3|             2|      1|    0|
|             3|       1|        1|               1|      2|         2|             3|             1|      1|    0|
|             6|       8|        8|               1|      3|         4|             3|             7|      1|    0|
|             4|       1|        1|               3|      2|         1|             3|             1|      1|    0|
+--------------+--------+---------+----------------+-------+----------+--------------+--------------+-------+-----+
*/

// no we will start building the Logestic Regression machine learning model
// let's cretae an array that has all the features we have in the dataset

val cols_mahmoud = Array("ClumpThickness", "UofCSize", "UofCShape", "MarginalAdhesion", "SECSize", "BareNuclei", "BlandChromatin", "NormalNucleoli", "Mitoses")

// now let's do the VectorAssembler to add feature column - we are going to name the output of it as features which will contain all the features values at once to be used in the machine learning model
val vAssembler_mahmoud = new VectorAssembler()
  .setInputCols(cols_mahmoud)
  .setOutputCol("features")


// now let's split our dataset into training and testing
// we will be defining the seed number: which is a random selection of the data, adn this number insures a consistant number of data on each run
// we will be spliting the data into 80% for training and 20% for testing
val Array(train_data, test_data) = Cancer_Mahmoud_labeled.randomSplit(Array(0.8, 0.2), 455) // the seed number is 455
// now creating the logistec regression model
val lr_mahmoud = new LogisticRegression()
 .setFeaturesCol("features")// this is the assembeled dataset features
 .setLabelCol("Class") // that's our target

// creating pipline: the pipline is used to set stages and tell it where to start wiht and end, in our example (assembler, then logistec regresssion model)

val pipeline_mahmoud = new Pipeline()
  .setStages(Array(vAssembler_mahmoud, lr_mahmoud))

// We use a ParamGridBuilder to construct a grid of parameters to search over.
//// elasticNet: The elastic net method overcomes the limitations of the LASSO (least absolute shrinkage and selection operator) method
/// The elastic net method improves lasso’s limitations, i.e., where lasso takes a few samples for high dimensional data
val paramGrid_mahmoud = new ParamGridBuilder()
  .addGrid(lr_mahmoud.elasticNetParam, Array(0.0, 0.5, 1.0)) // explained above
  .addGrid(lr_mahmoud.regParam, Array(0.0, 0.1, 0.01))// Set the regularization parameter -- make the data a little bit wider - to avoid overfitting
  .build()

// let's build the evaluater: here we are using the BinaryClassificationEvaluator
// the Receiver Operator Characteristic (ROC) is a metric used to evaluate binary classifications, since the logistec predicts binary events, 
//we are going to use this one. Also notting that this metric measures the area under teh curve to get the  probability.
val evaluator_mahmoud = new BinaryClassificationEvaluator()
  .setLabelCol("Class")
  .setMetricName("areaUnderROC") 

// here is the CrossValidator  which will combine everything together in one object, 
//where the pipline has the stages we did above, the evaluter, and the hyper parameter from the paramGrid
// note that we gave the cros validater a number of folds of 3, meaning it will try the hyper parameter 3 times against each other
val cross_validator_mahmoud = new CrossValidator()
  .setEstimator(pipeline_mahmoud)
  .setEvaluator(evaluator_mahmoud)
  .setEstimatorParamMaps(paramGrid_mahmoud)
  .setNumFolds(3)  // Use 3 folds
  .setParallelism(2)  // Evaluate up to 2 parameter settings in parallel

// next feeding the trainting data into the cross validater object to train teh model

val cvModel_mahmoud = cross_validator_mahmoud.fit(train_data)

// follows making teh model to predict the test_data

val predictions_mahmoud = cvModel_mahmoud.transform(test_data)

// finally evaluating the model usin gthe evaluator object, to get teh areaUnderROC

val accuracy_mahmoud = evaluator_mahmoud.evaluate(predictions_mahmoud)

println("areaUnderROC on test data = " + accuracy_mahmoud)

// after evaluating the model, we ended up with the following areaUnderROC:

// areaUnderROC on test data = 0.996  
