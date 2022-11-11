

from pyspark import SparkConf, SparkContext
from pyspark.mllib.feature import HashingTF
from pyspark.mllib.feature import IDF
import string
import numpy as np
import pandas as pd
from pyspark import SparkContext
from pyspark.sql import SparkSession
import gc



sc = SparkContext(master = 'local')
spark = SparkSession.builder.appName('pySpark word-count').config('spark.some.config.option', 'some-value').getOrCreate()



data = spark.sparkContext.textFile("/Users/iris/data.txt")


# #PRE-PROCESSING



#first map to split each term
res = data.flatMap(lambda x: x.replace("\n", '').split(" "))

#second mapper to append integer to word
res = res.map(lambda x: (x,1))

#reducer to do the count
count = res.reduceByKey(lambda x,y: x+y)

print(count.take(3))


# In[5]:


#convert to dataframe
df = spark.createDataFrame(count, ['term', 'count'])
df = df.filter(df["term"] != '')
df.show()


# In[6]:


from pyspark.sql.functions import col, lit
from functools import reduce

# we can edit any prefixes we require 
prefixes = ['dis_','gene_']

# reducer to filter with prefix
# starts_with = reduce(
#     lambda x, y: x | y,
#     [col("term").startswith(s) for s in prefixes], 
#     lit(False))


# In[10]:


h_df=df.where(reduce(
    lambda x, y: x | y,
    [col("term").startswith(s) for s in prefixes], 
    lit(False))).show()


# In[8]:


h_df.sort(col("count").desc()).show(truncate=False)

