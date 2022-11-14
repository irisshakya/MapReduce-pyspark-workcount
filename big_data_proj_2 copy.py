#!/usr/bin/env python
# coding: utf-8

# In[1]:


from pyspark import SparkConf, SparkContext
from pyspark.mllib.feature import HashingTF, IDF
import string
import numpy as np
import pandas as pd
from pyspark import SparkContext
from pyspark.sql import SparkSession
import gc


# In[2]:


from pyspark.sql import functions as f
from pyspark.sql.window import Window
from pyspark.sql.functions import row_number


# In[3]:


sc = SparkContext(master = 'local')
spark = SparkSession.builder.appName('pySpark word-count')             .config('spark.some.config.option', 'some-value')             .getOrCreate()


# In[4]:


data = spark.sparkContext.textFile("/Users/iris/data.txt")


# In[43]:


# SEPARATE EACH NEWLINE AS A list
my_para = data.flatMap(lambda x: x.split('\n'))


# In[62]:


my_list = my_para.flatMap(lambda x: list(x.split(" ")))


# In[72]:


# fil_list = my_list.filter(lambda x: x is not None).filter(lambda x: x != ' ')
# to_df = spark.createDataFrame(my_list, ['doc']).show(truncate=True)
# df = to_df.filter(to_df["doc"] != '')


# In[80]:


from pyspark.sql import Row

row = Row("doc") # Or some other column name
my_df = my_para.flatMap(row).toDF(schema=StringType())
#my_df = spark.createDataFrame(my_para, ['doc'])
my_df.show(truncate=True)


# In[81]:


from pyspark.sql.types import StringType
# df = my_df.filter(my_df["doc"] != '')
# df.show(truncate=True)
# df.count()


# In[85]:


df_vocab = my_df.select('value').rdd.map(lambda x: x[0]).toDF(schema=StringType()).toDF('doc')
df_vocab.show(truncate=True)


# In[86]:


vocab_freq = df_vocab.rdd.countByValue()
pdf = pd.DataFrame({
        'doc': list(vocab_freq.keys()),
        'frequency': list(vocab_freq.values())
    })
pdf
tf = spark.createDataFrame(pdf).orderBy('frequency', ascending=False)
tf.show()


# In[90]:


from pyspark.sql.functions import monotonically_increasing_id

#windowSpec  = Window.partitionBy("doc").orderBy("doc")

rank_df = tf.withColumn("doc_id", monotonically_increasing_id())
rank_df.show(4)

#first map to split each term
res = data.flatMap(lambda x: x.replace("\n", '').split(" "))

#MOD
#filter empty
my_filtered = res.filter(lambda x: len(x)!=0)

#second mapper to append integer to word
my_fil_map1 = my_filtered.map(lambda x: (x,1))

#reducer to do the count
count = my_fil_map1.reduceByKey(lambda x,y: x+y)

print(count.take(10))

#convert to dataframe
    #df = spark.createDataFrame(count, ['term', 'count'])
    #df = df.filter(df["term"] != '')
    #df.show()
    
from pyspark.sql.functions import col, lit
from functools import reduce

# we can edit any prefixes we require 
prefixes = ['dis_','gene_']

h_df=df.where(reduce(
    lambda x, y: x | y,
    [col("term").startswith(s) for s in prefixes], 
    lit(False))).show()
# In[ ]:




