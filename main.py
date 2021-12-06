#packages used for streaming
import time
import json
import pyspark
from pyspark import SparkContext
from pyspark.streaming import StreamingContext
from pyspark.sql import SparkSession
import pandas as pd
from pyspark.sql.types import StructType
from pyspark.sql.types import StructField
from pyspark.sql.types import StringType
from pprint import pprint
import socket

#packages used for preprocessing
from pyspark.sql.functions import length
from pyspark.ml.feature import Tokenizer,StopWordsRemover, CountVectorizer,IDF,StringIndexer
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.linalg import Vector
from pyspark.ml import Pipeline

#packages used for modeling
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.classification import NaiveBayes
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from sklearn.svm import SVC
from sklearn.svm import LinearSVC

sc = SparkContext(appName="SpamAnalysis")
ssc = StreamingContext(sc,1)
spark = SparkSession.builder.appName(
  "pandas to spark").getOrCreate()
lines = ssc.socketTextStream('localhost',6100)

def convtoDF(x):
	spark = SparkSession(x.context)
	schema1 = StructType([StructField("feature0",StringType(),True),StructField("feature1",StringType(),True),StructField("feature2",StringType(),True)])
	if not x.isEmpty():
		k = json.loads(x.collect()[0])
		df = pd.DataFrame.from_dict(k,orient="index")
		df_spark = spark.createDataFrame(df,schema = schema1)
		preprocess(df_spark)
counts = lines.flatMap(lambda line: line.split("\n"))
counts.foreachRDD(lambda rdd: convtoDF(rdd))
print(counts) 


def preprocess(data):

	data = data.withColumnRenamed('feature0','subject').withColumnRenamed('feature1','body').withColumnRenamed('feature2','class')
	data = data.withColumn('length',length(data['body']))
	data.show()
	data.groupby('class').mean().show()
	
	#modeling functions are also called here after pipeling 
	tokenizer = Tokenizer(inputCol="body", outputCol="token_body")
	stopremove = StopWordsRemover(inputCol='token_body',outputCol='stop_tokens')
	count_vec = CountVectorizer(inputCol='stop_tokens',outputCol='c_vec')
	idf = IDF(inputCol="c_vec", outputCol="tf_idf")
	ham_spam_to_num = StringIndexer(inputCol='class',outputCol='label')
	clean_up = VectorAssembler(inputCols=['tf_idf','length'],outputCol='features')
	lr = LogisticRegression(maxIter=10, regParam=0.001)
	
	data_prep_pipe = Pipeline(stages=[ham_spam_to_num,tokenizer,stopremove,count_vec,idf,clean_up])
	cleaner = data_prep_pipe.fit(data)
	clean_data = cleaner.transform(data)
	clean_data = clean_data.select(['label','features'])
	clean_data.show()
	
	logisticClassifier(clean_data)
	naiveBayesClassifier(clean_data)
	svmClassifier(clean_data)

def logisticClassifier(clean_data):
	lr = LogisticRegression(maxIter=10, regParam=0.001)
	data_prep_pipe = Pipeline(stages=[lr])
	(training,testing) = clean_data.randomSplit([0.7,0.3])
	spam_predictor = lr.fit(training)
	test_results = spam_predictor.transform(testing)
	print("\ntest results of logistic classifier")
	test_results.show()
	acc_eval = MulticlassClassificationEvaluator()
	acc = acc_eval.evaluate(test_results)
	print("Accuracy of logistic classifier at predicting spam was: {}".format(acc))
	print("-------------------1----------------------\n")

def naiveBayesClassifier(clean_data):
	nb = NaiveBayes()
	(training,testing) = clean_data.randomSplit([0.7,0.3])
	spam_predictor = nb.fit(training)
	test_results = spam_predictor.transform(testing)
	print("\ntest results of naive bayes classifier")
	test_results.show()
	acc_eval = MulticlassClassificationEvaluator()
	acc = acc_eval.evaluate(test_results)
	print("Accuracy of naive bayes classifier at predicting spam was: {}".format(acc))
	print("--------------------2---------------------\n")

'''def svmClassifier(clean_data):
	(training,testing) = clean_data.randomSplit([0.7,0.3])
	lsvc = LinearSVC()
	# Fit the model
	lsvcModel = lsvc.fit(training,testing)
	print("\ntest results of svm classifier")
	# Print the coefficients and intercept for linear SVC
	print("Coefficients: " + str(lsvcModel.coefficients))
	print("Intercept: " + str(lsvcModel.intercept))
	acc_eval = MulticlassClassificationEvaluator()
	acc = acc_eval.evaluate(classifier)
	print("Accuracy of SVM classifier at predicting spam was: {}".format(acc))
	print("--------------------3---------------------\n")'''
	
ssc.start()
ssc.awaitTermination(500)
ssc.stop()
