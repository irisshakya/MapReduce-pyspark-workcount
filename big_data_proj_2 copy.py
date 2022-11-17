
from pyspark import SparkConf, SparkContext
from pyspark.mllib.feature import HashingTF, IDF
import string
import numpy as np
import pandas as pd
from pyspark import SparkContext
from pyspark.sql import SparkSession
import gc
from pyspark.sql.functions import monotonically_increasing_id

sc = SparkContext(master = 'local')
spark = SparkSession.builder.appName('pySpark word-count').config('spark.some.config.option', 'some-value')             .getOrCreate()

data = spark.sparkContext.textFile("/Users/iris/project2_demo.txt")

my_para = data.map(lambda x: x.split('\n'))

split_list = my_para.map(lambda x: list(x[0].split()))
split_list2 = split_list.filter(lambda x: x[0] is not None).filter(lambda x: x[0] !="")


split_list2.take(2)

map1 = split_list2.map(lambda x: 1)
map1.take(2)


