// Databricks notebook source
// MAGIC %md ![](https://github.com/dataresults-zz/notebooks/blob/master/resources/dr_logo.jpg?raw=true=50x25) 
// MAGIC  # Essential statistics for machine learning with Apache Spark / Scala
// MAGIC 
// MAGIC 
// MAGIC  ### prepared by [Rolf-Dieter Kaschke](http://www.w2technology.de)
// MAGIC  
// MAGIC  *supported by* [![](https://github.com/dataresults-zz/notebooks/blob/master/resources/databricks_logoTM_200px.png?raw=true)](https://databricks.com/)
// MAGIC  
// MAGIC  *slides:*
// MAGIC  
// MAGIC  [![accompnying slides](https://github.com/dataresults-zz/notebooks/blob/master/statistics/resources/statistics.jpg?raw=true=100x20)](https://github.com/dataresults-zz/notebooks/blob/master/statistics/resources/statistics.pdf)

// COMMAND ----------

// MAGIC %md ###### Purpose of this notebook
// MAGIC This notebook is intended to provide essential statistics for machine learning applications. More statistical or mathematical information can be found in the slides. The approach is to wrap up statistics for machine learning experts with focus on practice rather than on exact mathematical proofs.
// MAGIC 
// MAGIC Also it is intended to show how to use the Apache Spark statistics methods for (almost) dataframes from different libraries, but also to develop own statistical methods. 
// MAGIC 
// MAGIC It forms a introduction to the practical aspects of exploratory data analysis.
// MAGIC 
// MAGIC Start practicing.

// COMMAND ----------

// MAGIC %md ###### Statistics with Apache Spark
// MAGIC Apache Spark’s ability to support data statistics with DataFrames contains methods of fundamental and advanced statistics. 
// MAGIC 
// MAGIC <b>This Notebook contains:</b> 
// MAGIC    * *descriptive statistics*
// MAGIC    * *independence of variables*
// MAGIC    * *Inference statistics*
// MAGIC    * *hypotheses testing*
// MAGIC    * *bootstrap technique*

// COMMAND ----------

// MAGIC %md #### Organize all imports

// COMMAND ----------

// organize imports
import org.apache.spark.sql._
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._
import org.apache.spark.sql.expressions.{MutableAggregationBuffer, UserDefinedAggregateFunction}

import org.apache.spark.mllib.stat.Statistics
import org.apache.spark.mllib.linalg.{Vectors, Vector, Matrices, Matrix => LM, DenseMatrix => DM, SparseMatrix => SM}
import org.apache.spark.ml.feature.{Bucketizer,StandardScaler,VectorAssembler}
import spark.implicits._
import breeze.linalg.{CSCMatrix => BSM, DenseMatrix => BDM, Matrix => BM}

import scala.collection.mutable._
import scala.util.Sorting.quickSort


// COMMAND ----------

// MAGIC %md ### Read data for statistical analysis

// COMMAND ----------

// MAGIC %md ##### 1 \- example data donated by kaggle challenge Rossmann
// MAGIC 
// MAGIC File is available at https://www.kaggle.com/c/rossmann-store-sales/download/test.csv.zip

// COMMAND ----------

// MAGIC %md ###### Create schema from string, read raw data, preprocess them and apply schema
// MAGIC 
// MAGIC First create a schema which will be used to read and name data correctly.

// COMMAND ----------

// create the schema from string
val schemaString = "Store DayOfWeek Date Sales Customer Open Promo StateHoliday SchoolHoliday"
val schema = StructType(schemaString.split(" ").map(fieldName ⇒ StructField(fieldName, StringType, true)))

// COMMAND ----------

// input is the example dataset from kaggle Rossmann challenge
val inputPath = "/FileStore/tables/ppibh5ja1485270759318/train.csv"

// COMMAND ----------

// read and preprocess data
val rawData = sc.textFile(inputPath)
// skip headers as we want to use our own
val header = rawData.first()
val rawDataWithoutHeader = rawData.filter(row => row != header)
// create fields (as Row)
val storeSalesData = rawDataWithoutHeader.map(_.split(",")).map(e ⇒ Row(e(0), e(1), e(2), e(3), e(4), e(5), e(6), e(7), e(8)))
var storeSalesDataDf = sqlContext.createDataFrame(storeSalesData, schema)
display(storeSalesDataDf)

// COMMAND ----------

// MAGIC %md ######cast values into required numerical formats 
// MAGIC 
// MAGIC - Remark - if values need to be available in different formats, duplicate them

// COMMAND ----------

// cast columns from strings to numeric values 
storeSalesDataDf = storeSalesDataDf.selectExpr(
  "Date",
  "cast(Store as int) Store",
  "cast(DayOfWeek as int) DayOfWeek",
  "cast(Customer as int) CustomerInt",
  "cast(Promo as int) Promo",
  "cast(Open as int) Open",
  "StateHoliday",
  "SchoolHoliday",
  "cast(Sales as Double) Sales",
  "Customer"
)
storeSalesDataDf.createOrReplaceTempView("storeSales")

// COMMAND ----------

// MAGIC %md ##### 2 \- example data taken from yahoo stocks

// COMMAND ----------

// MAGIC %md ###### load helper functions from another notebook and read data fron netflix and ibm to analyze further

// COMMAND ----------

// MAGIC %run /Users/kaschke@w2technology.de/HelperFunctions

// COMMAND ----------

val nflxDf = readCSVStd(sqlContext, "/FileStore/tables/1apgmvk71485626475073/NFLX_yahoo_stock_prize-8f902.csv", header=true)
val ibmDf = readCSVStd(sqlContext, "/FileStore/tables/fqys4bui1485626771735/IBM_yahoo_stock_prize-a7a3c.csv", header=true)
var stockDf = ibmDf.withColumn("IBM",ibmDf("Adj Close")).select("Date", "IBM")
  .join(nflxDf.withColumn("NFLX",nflxDf("Adj Close")).select("Date", "NFLX"), Seq("Date"),"left")
stockDf.createOrReplaceTempView("stocks")
stockDf.show(3)

// COMMAND ----------

// MAGIC %md ###Statistics \- Summarizing Data
// MAGIC 
// MAGIC Summarize values of one column

// COMMAND ----------

// MAGIC %md ###### Summarize Data
// MAGIC 
// MAGIC Apache Spark offers a describe() method to get fundamental summarized data. This includes
// MAGIC * count
// MAGIC * mean 
// MAGIC * minimum
// MAGIC * maximum
// MAGIC * standard deviation
// MAGIC * variance.
// MAGIC 
// MAGIC Using this method is good for visualization of the results. If you need the values for further processing it is somewhat complicated to extract the right value from the DataFrame that was created by this method.<p>
// MAGIC 
// MAGIC ######Mean
// MAGIC 
// MAGIC Mean is the average over all values (here from a column).<br>
// MAGIC 
// MAGIC Calculation:<br>
// MAGIC ![](https://github.com/dataresults-zz/notebooks/blob/master/statistics/formulas/mean.bmp?raw=true) 
// MAGIC 
// MAGIC ######Standard deviation
// MAGIC 
// MAGIC In statistics, the standard deviation (represented by the Greek letter σ) is a measure that is used to quantify the amount of variation or dispersion of a set of data values. A low standard deviation indicates that the data points tend to be close to the mean (also called the expected value) of the set, while a high standard deviation indicates that the data points are spread out over a wider range of values.
// MAGIC 
// MAGIC Calculation:<br>
// MAGIC ![](https://github.com/dataresults-zz/notebooks/blob/master/statistics/formulas/stdDev.bmp?raw=true) 
// MAGIC  
// MAGIC ######Variance
// MAGIC 
// MAGIC The variance is the squared standard deviation:<br>
// MAGIC 
// MAGIC Calculation:<br>
// MAGIC ![](https://github.com/dataresults-zz/notebooks/blob/master/statistics/formulas/variance.bmp?raw=true) 

// COMMAND ----------

val describedData =  storeSalesDataDf.describe()
display(describedData) 

// COMMAND ----------

// MAGIC %md ###### Additional methods of column statistic for one column
// MAGIC Some basic statistics can be found as aggregate functions in `org.apache.spark.sql.functions`. <br>
// MAGIC These methods allow easier access to the values. So these methods can be used if values need to be processed further.
// MAGIC <br>
// MAGIC The aggregate method is applied to single columns, so if you need statistics of multiple columns you need to iterate over the columns (e.g. by column names). <br>
// MAGIC 
// MAGIC Functions included: 
// MAGIC  - min
// MAGIC  - max
// MAGIC  - avg
// MAGIC  - mean
// MAGIC  - counts (`count(Column e)`, `countDistinct(Column e)`, `approxCountDistinct(Column e)`,)
// MAGIC  - sum
// MAGIC  - standard deviation (for populations and sampled data)
// MAGIC  - variance (for populations and sampled data)
// MAGIC  
// MAGIC  
// MAGIC  

// COMMAND ----------

def getSummaryStatistics(df:DataFrame, column:String):DataFrame = {
  df.agg(min(column) as "min", max(column) as "max", avg(column) as "avg", mean(column) as "mean",  
    count(column) as "count", countDistinct(column) as "distinct", approxCountDistinct(column) as "approx_distinct", 
    sum(column) as "sum", stddev(column) as "stddev", stddev_samp(column) as "stdev_samp", stddev_pop(column) as "stddev_pop",
    variance(column) as "variance", var_samp(column) as "var_samp", var_pop(column) as "var_pop")
}

// COMMAND ----------

// single column functions 
val colName = "Sales"
val statisticObj = getSummaryStatistics(storeSalesDataDf, colName)
display(statisticObj)

// COMMAND ----------

// get single values
//statisticObj.printSchema
val minimum = statisticObj.first.getDouble(0)
val maximum = statisticObj.first.getDouble(1)
val mean = statisticObj.first.getDouble(2)
val total_count = statisticObj.first.getLong(4)
val distinct_count = statisticObj.first.getLong(5)
val total_sum = statisticObj.first.getDouble(7)
val stdDev = statisticObj.first.getDouble(8)
val variance = statisticObj.first.getDouble(11)

// COMMAND ----------

// MAGIC %md ###### Additional methods of column statistic for one column
// MAGIC Some basic statistics can not be found in `org.apache.spark.sql.functions`. <br>
// MAGIC These functions are (e.g. ): 
// MAGIC  * <b>percentiles</b> (q1, q2, q3)
// MAGIC  * <b>median</b> (the middle value of the ordered data \- which is in fact the q2 percentile) 
// MAGIC  * <b>mode</b> (the value that occurs most often in the data)
// MAGIC  * <b>rank</b> (top n)  
// MAGIC  <p>
// MAGIC These methods must be developed on your own. As all these methods needs sorted values, they are very consuming. So make sure that only a fraction of data (when using high volume production data) are used (by sampling).
// MAGIC <br>The methods can be implemented in different ways:
// MAGIC 1. as an extension of the DataFrame itself - see getMode,
// MAGIC 2. as sql queries
// MAGIC 3. as a simple method definition

// COMMAND ----------

object DataFrameExtensions {
  implicit def extendedDataFrame(dataFrame: DataFrame): ExtendedDataFrame = new ExtendedDataFrame(dataFrame: DataFrame)
  class ExtendedDataFrame(dataFrame: DataFrame) {
      def getMode(column:String):DataFrame = {
         dataFrame.filter(col(column).isNotNull).groupBy(col(column)).agg(
            count("*").alias("cnt")).sort($"cnt".desc)
      }
   }
}

// COMMAND ----------

import DataFrameExtensions._
val mode = storeSalesDataDf.getMode(colName).first.getDouble(0)

// COMMAND ----------

// get top n values in column
def getTopN(table:String, column:String, context:SQLContext, limit:Int = 5):DataFrame = {
    sqlContext.sql(s"SELECT $column, COUNT(*) AS count FROM $table where $colName is NOT NULL GROUP BY $column ORDER BY count DESC LIMIT   5")
}

// COMMAND ----------

display(getTopN("storeSales",colName,sqlContext))

// COMMAND ----------

// get most frequent items of column

def getFreqItems(df:DataFrame, columns:Array[String], percent:Int = 40):DataFrame = {
    df.stat.freqItems(columns, percent/100.0)
}

// COMMAND ----------

val freqItems = getFreqItems(storeSalesDataDf, Array("Sales","Customer"))
display(freqItems)

// COMMAND ----------

def getPercentiles(table:String, column:String, context:SQLContext, limit:Int = 5):Array[Double] = {
    val row = sqlContext.sql(s"SELECT percentile_approx($column, 0.25) as q1 , percentile_approx($column, 0.5) as median , percentile_approx($column, 0.75) as q3 FROM storeSales").first
  Array(row.getDouble(0), row.getDouble(1), row.getDouble(2))
}

// COMMAND ----------

val percentiles = getPercentiles("storeSales",colName,sqlContext)

// COMMAND ----------

// MAGIC %md ###### Quantiles
// MAGIC Some basic statistics can not be also found in `org.apache.spark.sql.DataFrameStatFunctions`. <br>
// MAGIC One example is calculating the quantiles between 0 and 1<p>
// MAGIC Quantiles are cutpoints dividing the range of a probability distribution into contiguous intervals with equal probabilities.<br>
// MAGIC 
// MAGIC The p\-quantil x_p defines the real number for which at least p * 100% values are less or equal x_p and at least (1-p) * 100% are bigger or equal x_p.
// MAGIC <br>
// MAGIC 
// MAGIC This method calculates the approximate quantiles of a numerical column of a DataFrame. So indeed we dont calculate exact percentiles as usual with a count method, but by quantile probabilities p (a parameter of that method). Also we have to define the error (relative target error).<br>
// MAGIC 
// MAGIC The result of this algorithm has the following deterministic bound: If the DataFrame has N elements and if we request the quantile at probability p up to error err, then the algorithm will return a sample x from the DataFrame so that the *exact* rank of x is close to (p * N). <br>
// MAGIC 
// MAGIC This method implements a variation of the Greenwald-Khanna algorithm (with some speed optimizations). The algorithm was first present in Space-efficient Online Computation of Quantile Summaries by Greenwald and Khanna.

// COMMAND ----------

// value min-> 0.0, median-> 0.5, max-> 1
val relativeError = 0.05
val apprQuantile = storeSalesDataDf.stat.approxQuantile(colName, Array(0.0, 0.25, 0.5, 0.75, 1.0), relativeError)
val minimum =apprQuantile(0)
val quantile25 = apprQuantile(1)
val quantile50 = apprQuantile(2)
val quantile75 = apprQuantile(3)
val maximum = apprQuantile(4)
val iqr = quantile75 - quantile25

// COMMAND ----------

// MAGIC %md ###### Detect outliers
// MAGIC There are different methods to identify outlier in statistics. One method uses the interquartile range IQR. If a value is outside quartile plus/minus 1.5 times IQR this value is assumed to be an outlier.
// MAGIC 
// MAGIC <b>Calculation:</b><br>
// MAGIC ![](https://github.com/dataresults-zz/notebooks/blob/master/statistics/formulas/outlier.bmp?raw=true) 
// MAGIC 
// MAGIC This method is known as Box and Whisker method. Any value smaller than Q1 - 1.5 \* IQR or any value greater than Q3 + 1.5 \* IQR will be categorised as an outlier.
// MAGIC 
// MAGIC The same can be done with quantiles from above, here the range can be adjusted by parameter p for the quantiles.

// COMMAND ----------

val iqr = percentiles(2) -  percentiles(0)
val lowerRange = percentiles(0) - 1.5 * iqr
val upperRange = percentiles(2) + 1.5 * iqr
val outliers = storeSalesDataDf.select(colName).filter(col(colName) < lowerRange or col(colName) > upperRange)
println(storeSalesDataDf.count)
println(outliers.count)
outliers.show(5)

// COMMAND ----------

// MAGIC %md ###### Histogram
// MAGIC A histogram shows how the data is distributed between different range of the values. A histogram contains the count of values where the values are grouped into buckets.
// MAGIC 
// MAGIC Histograms are visual representation of the shape/distribution of the data. This visual representation is heavily used in statistical data exploration.

// COMMAND ----------

case class HistogramRow(startPoint:Double,count:Long)
def createHistDf(df:DataFrame, colName:String, noOfBuckets:Int=5, sqlContext: SQLContext):Dataset[HistogramRow] = {
  val (startValues:Array[Double], counts:Array[Long]) = df.select(colName).map(value => value.getDouble(0)).rdd.histogram(noOfBuckets)
  val zippedValues = startValues.zip(counts)  
  val rowRDD = zippedValues.map( value => HistogramRow(value._1,value._2))
  sqlContext.createDataFrame(rowRDD).as[HistogramRow]  
}

// COMMAND ----------

val colName = "Sales"
val histDf = createHistDf(storeSalesDataDf, colName, 10, sqlContext)
histDf.createOrReplaceTempView("histogramTable")
display(histDf)

// COMMAND ----------

// MAGIC %sql
// MAGIC select * from histogramTable

// COMMAND ----------

// MAGIC %md ###### Count of distinct values
// MAGIC A count of distinct values. Normally it is not necessary to sort the values. Sorting should be avoided as it is very costly.

// COMMAND ----------

def createDistinctCountDf(df:DataFrame, colName:String, newColName:String):DataFrame = {
   df.groupBy(col(colName)).count().withColumnRenamed("count", newColName)//.sort(col(newColName).desc)
}

// COMMAND ----------

val newColName = colName + "_distCount"
val distinctCountDf = createDistinctCountDf(storeSalesDataDf, colName, newColName)
display(distinctCountDf)

// COMMAND ----------

def addDistinctCountToDf(df:DataFrame, colName:String, newColName:String):DataFrame = {
  val schemaNames = df.schema.fieldNames
  val dfTemp = if (schemaNames.contains(newColName)) df.drop(newColName) else df
    val dc = createDistinctCountDf(dfTemp, colName, newColName)
    val tmp = dc.withColumnRenamed(colName, "key")
    dfTemp.join(tmp, dfTemp(colName) === tmp("key"), "inner").drop(col("key"))
}

// COMMAND ----------

storeSalesDataDf = addDistinctCountToDf(storeSalesDataDf, colName, newColName)
storeSalesDataDf.show(3)

// COMMAND ----------

def getCountPerVal(distinctCount:DataFrame, colName:String, newColName:String, value:String):Long = {
  distinctCount.select(colName, newColName).filter(distinctCount(colName)===value).first.getAs(newColName)
}

// COMMAND ----------

val newColName = colName + "_dc1"
val dcDf = createDistinctCountDf(storeSalesDataDf, colName, newColName)
val dbl = getCountPerVal(dcDf, colName, newColName, "934.0")
println(dbl)

// COMMAND ----------

// MAGIC %md ###### Create own aggregate functions 
// MAGIC With UDAF own methods can be implemented.
// MAGIC 
// MAGIC UserDefinedAggregateFunction can be extended to write custom aggregate function, like a quadratic mean function.

// COMMAND ----------

package de.web2technology
import org.apache.spark.sql.Row
import org.apache.spark.sql.expressions.{MutableAggregationBuffer, UserDefinedAggregateFunction}
import org.apache.spark.sql.types._

class QuadraticMean() extends UserDefinedAggregateFunction {
 
  // Input Data Type Schema
  def inputSchema: StructType = StructType(Array(StructField("item", DoubleType))) 
  // Intermediate Schema
  def bufferSchema = StructType(Array(
    StructField("sum", DoubleType),
    StructField("cnt", LongType)
  )) 
  // Returned Data Type .
  def dataType: DataType = DoubleType 
  // Self-explaining
  def deterministic = true 
  // This function is called whenever key changes
  def initialize(buffer: MutableAggregationBuffer) = {
    buffer(0) = 0.toDouble // set sum to zero
    buffer(1) = 0L // set number of items to 0
  } 
  // Iterate over each entry of a group
  def update(buffer: MutableAggregationBuffer, input: Row) = {
    //buffer(0) = buffer.getDouble(0) + input.getDouble(0)
    buffer(0) = buffer.getDouble(0) + (input.getDouble(0) * input.getDouble(0))
    buffer(1) = buffer.getLong(1) + 1
  } 
  // Merge two partial aggregates
  def merge(buffer1: MutableAggregationBuffer, buffer2: Row) = {
    buffer1(0) = buffer1.getDouble(0) + buffer2.getDouble(0)
    buffer1(1) = buffer1.getLong(1) + buffer2.getLong(1)
  } 
  // Called after all the entries are exhausted.
  def evaluate(buffer: Row) = {
    Math.sqrt(buffer.getDouble(0)/buffer.getLong(1).toDouble)
  } 
}

// COMMAND ----------

// initialize UDAF
val quadraticMean = new de.web2technology.QuadraticMean()
println(colName)
// Calculate average value for each group
val qMeanStat = storeSalesDataDf.agg(quadraticMean(col(colName)).as("quadratic_mean"),
  avg(colName).as("avg")
)
qMeanStat.show()

// COMMAND ----------

// MAGIC %md ###### Probability mass functions
// MAGIC To represent a distribution a probability mass function (PMF) can be used, which maps from each value to its probability. A probability is a
// MAGIC frequency expressed as a fraction of the sample size, n. 
// MAGIC 
// MAGIC To get from frequencies to probabilities, divide the frequencies (how often a value oocur) through the total number of values, which is called normalization.
// MAGIC 
// MAGIC The PMF is a probability measure that gives us probabilities of the possible values for a random variable.
// MAGIC 
// MAGIC Calculation:<br>
// MAGIC ![](https://github.com/dataresults-zz/notebooks/blob/master/statistics/formulas/pmf.bmp?raw=true) 

// COMMAND ----------

def calculatePMF(df:DataFrame, colName:String, newColName:String):DataFrame ={
  val schemaNames = df.schema.fieldNames
  val tmpColName = if (schemaNames.contains(colName + "_tmp")) colName + "_tmp1" else colName + "_tmp"
  val dfTmp = if (schemaNames.contains(newColName)) df.drop(newColName) else df
  val total = dfTmp.select(colName).count
  val df1 = addDistinctCountToDf(dfTmp, colName, tmpColName)
  df1.withColumn(colName + "_pmf", df1(tmpColName) / total).drop(tmpColName)
}

// COMMAND ----------

storeSalesDataDf = calculatePMF(storeSalesDataDf, colName, colName + "_PMF")
storeSalesDataDf.show(3)

// COMMAND ----------

// MAGIC %md ###### Cumulative distribution function (CDF)
// MAGIC The Cumulative distribution function (CDF) is the function that maps from a value to its percentile rank.
// MAGIC It is a function of x, where x is any value that might appear in the distribution. 
// MAGIC 
// MAGIC To evaluate CDF(x) for a particular value of x, compute the fraction of values in the distribution less than or equal to x.
// MAGIC 
// MAGIC Calculation:<br>
// MAGIC ![](https://github.com/dataresults-zz/notebooks/blob/master/statistics/formulas/cdf.bmp?raw=true) 

// COMMAND ----------

package de.web2technology
import org.apache.spark.sql.Row
import org.apache.spark.sql.expressions.{MutableAggregationBuffer, UserDefinedAggregateFunction}
import org.apache.spark.sql.types._

class CDF() extends UserDefinedAggregateFunction {
 
  // Input Data Type Schema
  def inputSchema: StructType = StructType(Array(StructField("item", DoubleType), StructField("limit", DoubleType))) 
  // Intermediate Schema
  def bufferSchema = StructType(Array(
    StructField("cnt", LongType),
    StructField("total", LongType)
  )) 
  // Returned Data Type .
  def dataType: DataType = DoubleType 
  // Self-explaining
  def deterministic = true 
  // This function is called whenever key changes
  def initialize(buffer: MutableAggregationBuffer) = {
    buffer(0) = 0L // set cnt to zero
    buffer(1) = 0L // set total of items to 0
  } 
  // Iterate over each entry of a group
  def update(buffer: MutableAggregationBuffer, input: Row) = {
    if (input.getDouble(0) < input.getDouble(1)) buffer(0) = buffer.getLong(0) + 1
    buffer(1) = buffer.getLong(1) + 1
  } 
  // Merge two partial aggregates
  def merge(buffer1: MutableAggregationBuffer, buffer2: Row) = {
    buffer1(0) = buffer1.getLong(0) + buffer2.getLong(0)
    buffer1(1) = buffer1.getLong(1) + buffer2.getLong(1)
  } 
  // Called after all the entries are exhausted.
  def evaluate(buffer: Row) = {
    buffer.getLong(0)/buffer.getLong(1).toDouble
  } 
}

// COMMAND ----------

val colName = "Sales"
val limit = 8000.0 
// initialize UDAF
val cdf = new de.web2technology.CDF()
val cdfVal = storeSalesDataDf.agg(cdf(col(colName),lit(limit))).first.getDouble(0)
println(cdfVal)


// COMMAND ----------

// MAGIC %md ### Relationships between two (or more) variables

// COMMAND ----------

// MAGIC %md #### Correlation and covariance
// MAGIC 
// MAGIC Relations beween different variables (samples)

// COMMAND ----------

// MAGIC %md ###### Correlation and dependence of variables
// MAGIC 
// MAGIC In statistics, dependence is any statistical relationship, whether causal or not, between two random variables or two sets of data. <br>
// MAGIC 
// MAGIC <b>Correlation</b> in common usage it most often refers to the extent to which two variables have a *linear* relationship with each other.
// MAGIC 
// MAGIC Correlations are useful because they can indicate a predictive relationship that can be exploited in practice. In some cases there is a causal relationship; however, correlation is not sufficient to demonstrate the presence of any a causal relationship (i.e., correlation does not imply causation).
// MAGIC 
// MAGIC Formally, dependence refers to any situation in which random variables do not satisfy a mathematical condition of probabilistic independence. Technically it refers to any of several more specialized types of relationship between mean values. 
// MAGIC 
// MAGIC There are several correlation coefficients, often denoted ρ, measuring the degree of correlation. The most common of these is the <b>*Pearson correlation coefficient*</b>, which is sensitive only to a linear relationship between two variables.
// MAGIC 
// MAGIC ###### Apache Spark column statistic for two or more column
// MAGIC Statistics can be foud in `org.apache.spark.sql.DataFrameStatFunctions`<br>
// MAGIC 
// MAGIC Functions:
// MAGIC  - correlation
// MAGIC  - covariance

// COMMAND ----------

// MAGIC %md ######Pearson's correlation coefficient
// MAGIC To describe the strength of a <b>linear</b> relationship between two variables the Paerson's correlation coefficient provides a quantitative measure. 
// MAGIC The <b>*Pearson correlation coefficient*</b> is a measure of the <b>linear</b> dependence (correlation) between two variables X and describe the strength of the relationship with a quantitative measure. 
// MAGIC 
// MAGIC The coefficient has values between +1 and −1 inclusive, where 1 is total positive linear correlation, 0 is no linear correlation, and −1 is total negative linear correlation.
// MAGIC Note: Pearsons correlation coefficient only describe linear dependencies. Also a coefficient of +1 or -1 implies a linear relationship, it doesn't need to be a causal relationship. The relation can also be indirect by some other (causal) related facts.
// MAGIC 
// MAGIC The strength of a linear relationship is an indication of how closely the points in a scatter diagram fit a straight line, so a scatter plot should be performed to check manually if such a realtionship exists.
// MAGIC 
// MAGIC ######Calculation<br>
// MAGIC <b>For a population</b>
// MAGIC 
// MAGIC Pearson's correlation coefficient is commonly represented by the Greek letter ρ (rho) for populations. 
// MAGIC The formula for ρ is:
// MAGIC 
// MAGIC    ![](https://github.com/dataresults-zz/notebooks/blob/master/statistics/formulas/pearsond-rho.bmp?raw=true) 
// MAGIC 
// MAGIC             where  cov is the covariance
// MAGIC             σX is the standard deviation of X
// MAGIC                 
// MAGIC <b>For samples</b><br>
// MAGIC For samples the Pearson's correlation coefficient is represented by r. The formula for r is:
// MAGIC 
// MAGIC ![](https://github.com/dataresults-zz/notebooks/blob/master/statistics/formulas/pearsons-r.bmp?raw=true) 
// MAGIC 
// MAGIC             where σx is the standard deviation

// COMMAND ----------

// calculate pearson's correlation coefficient of two columns of a DataFrame.
// the data are aken from Yahoos archived stock data
val colName1 = "IBM"
val colName2 = "NFLX"
val corr = stockDf.stat.corr(colName1, colName2)
println("correlation between columns " + colName1 + " & " + colName2 + " is " + corr)


// COMMAND ----------

// MAGIC %md ###### Covariance
// MAGIC In statistics, <b>covariance</b> is a measure of the joint variability of two random variables.
// MAGIC 
// MAGIC If the greater values of one variable mainly correspond with the greater values of the other variable, and the same holds for the lesser values, the variables tend to show similar behavior and the covariance is positive. In contrast, when the greater values of one variable mainly correspond to the lesser values of the other, the covariance is negative. 
// MAGIC 
// MAGIC The magnitude of the covariance is not easy to interpret. The normalized version of the covariance, the correlation coefficient, however, shows by its magnitude the strength of the linear relation.
// MAGIC 
// MAGIC ######Calculation:<br>
// MAGIC The covariance between two jointly distributed real-valued random variables X and Y with finite second moments is:<br>
// MAGIC 
// MAGIC ![](https://github.com/dataresults-zz/notebooks/blob/master/statistics/formulas/covariance.bmp?raw=true)
// MAGIC 
// MAGIC             where E[X] is the expected value of X, also known as the mean of X. 

// COMMAND ----------

val cov = stockDf.stat.cov(colName1, colName2)
println("covariance between columns " + colName1 + " & " + colName2 + " is " + cov)


// COMMAND ----------

// MAGIC %md 
// MAGIC - A distinction must be made between the covariance of two random variables, which is a population parameter that can be seen as a property of the joint probability distribution:<br>
// MAGIC   `covar_pop()`  
// MAGIC 
// MAGIC - and the sample covariance, which in addition to serving as a descriptor of the sample, also serves as an estimated value of the population parameter:<br>
// MAGIC   `covar_samp()`

// COMMAND ----------

display(stockDf.agg(covar_pop(colName1, colName2) as "covar_pop", covar_samp(colName1, colName2) as "covar_samp"))

// COMMAND ----------

// MAGIC %md ######Spearman's rank correlation coefficient
// MAGIC <b>Spearman's rank correlation coefficient</b> or Spearman's rho, denoted by the Greek letter ρ, is a nonparametric measure of rank correlation (statistical dependence between the ranking of two variables). It assesses how well the relationship between two variables can be described using a monotonic function.
// MAGIC 
// MAGIC The <b>Spearman correlation</b> between two variables is equal to the <b>Pearson correlation</b> between the rank values of those two variables; while Pearson's correlation assesses <b>linear</b> relationships, Spearman's correlation assesses <b>monotonic</b> relationships (whether linear or not). If there are no repeated data values, a perfect Spearman correlation of +1 or −1 occurs when each of the variables is a perfect monotone function of the other.
// MAGIC 
// MAGIC Intuitively, the Spearman correlation between two variables will be high when observations have a similar (or identical for a correlation of 1) rank (i.e. relative position label of the observations within the variable: 1st, 2nd, 3rd, etc.) between the two variables, and low when observations have a dissimilar (or fully opposed for a correlation of -1) rank between the two variables.
// MAGIC 
// MAGIC Spearman's coefficient is appropriate for both continuous and discrete variables, including ordinal variables.
// MAGIC 
// MAGIC ###### Calculation
// MAGIC 
// MAGIC For a sample of size n, the n raw scores Xi , Yi are converted to ranks rgXi and rgYi, and rs is computed from:
// MAGIC 
// MAGIC ![](https://github.com/dataresults-zz/notebooks/blob/master/statistics/formulas/spearman.bmp?raw=true)
// MAGIC    
// MAGIC         where
// MAGIC             ρ denotes the usual Pearson correlation coefficient, but applied to the rank variables.
// MAGIC             cov ⁡ (rgX , rgY) is the covariance of the rank variables.
// MAGIC             σrgX is the standard deviations of the rank variables.

// COMMAND ----------

// MAGIC %md ######Preprocess data
// MAGIC 
// MAGIC The calculation of *Spearman's rank correlation coefficient* is only available for rdd's in Apache Spark.<br>
// MAGIC First all null values are dropped and then the DataFrame is converted into two rdd's (each rdd contains one column under test).

// COMMAND ----------

// drop empty rows
val stockClearedDf = stockDf.na.drop()
println(stockClearedDf.count)

// Select columns and extract values
val rddCol1 = stockClearedDf.select(colName1).rdd.map(_.getDouble(0))
val rddCol2 = stockClearedDf.select(colName2).rdd.map(_.getDouble(0))

// COMMAND ----------

// MAGIC %md ###### Calculate Spearman's rank correlation coefficient 

// COMMAND ----------

// calculate correlation
val correlation: Double = Statistics.corr(rddCol1, rddCol2, "spearman")

// COMMAND ----------

// MAGIC %md ###Hypothesis testing
// MAGIC 
// MAGIC The major purpose of hypothesis testing is to choose between two <b>competing </b>hypotheses about the value of a population parameter.<br>
// MAGIC In general we don't know the true value of population parameters, so they must be estimated.<br>
// MAGIC 
// MAGIC The hypothesis to be tested is denoted <b>H0</b> and is called null hypothesis. The null hypothesis is assumed to be true until there is a strong evidence to the contrary. The alternative hypothesis is denoted as <b>H1.</b><p>
// MAGIC <b>Decisions:</b><br>
// MAGIC - try to determine whether there is sufficient evidence to declare H0 false
// MAGIC - decisions are based on probability rather then certainty errors can occur
// MAGIC 
// MAGIC <br><b>Errors:</b><br>
// MAGIC - type I error: rejection of a null hypothesis while it is true. The probability of type I error is `α=P(rejecting H0 | H0 is true)`, typical values chosen for α are 0.05 (5%) or 0.01 (1%)
// MAGIC - type II error: acception of a null hypothesis when it is not true. The probability of type I error is `β=P(accepting H0 | H0 is false)`, 

// COMMAND ----------

// MAGIC %md ###### Bivariate statistics
// MAGIC 
// MAGIC Bivariate statistics is used to compare two study groups to see if they are similar. To provide strong evidence that comparing groups are different, a conservative threshold of p < 0.05 to determine statistical significance is used.
// MAGIC It is also used to identify covariates for global explanatory models. When characteristics in different groups is different, so the characteristic is associated with the outcome. This is not causal (not necessarily causal). Rather a characteristic tends to be present when the outcome is present. If a variable is independently associated with the outcome, it might continue to explain the outcom once other factors are taken into account.
// MAGIC To filter out potential covariates in multivariate analysis a generous threshold of p > 0.1 to determine statistical significance is used to ensure that potentially useful variables aren't filtered out from analysis.

// COMMAND ----------

// MAGIC %md ######Pearson’s chi-squared tests for independence
// MAGIC 
// MAGIC The chi-square distribution is used in tests of hypotheses concerning the independence of two random variables and concerning whether a discrete random variable follows a specified distribution.<br>
// MAGIC 
// MAGIC The chi-square test for independence is used to determine whether there is a significant association between two variables. <br>
// MAGIC Two random variables x and y are called independent if the probability distribution of one variable is not affected by the presence of another.<br>A chi-square random variable is a random variable that assumes only positive values and follows a chi-square distribution.<br>
// MAGIC <br>
// MAGIC <b>Hypothesis:</b><br>
// MAGIC H0: variables are indpendent<br>
// MAGIC H1: variables are <b>not</b> independent.<p>
// MAGIC <b>Contingency table:</b>
// MAGIC To form a contingency table, values from variables under test are associated to categories and form a nxm matrix (which is called contingency table). A sample contingency table is:<br><code>
// MAGIC income|&nbsp;Male|&nbsp;Female<br>
// MAGIC &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;0|&nbsp;&nbsp;&nbsp;104|&nbsp;&nbsp;&nbsp;862<br> 
// MAGIC &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;1|&nbsp;&nbsp;&nbsp;945|&nbsp;&nbsp;2366 <br>
// MAGIC &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;2|&nbsp;&nbsp;2880|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;0<br>
// MAGIC &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;3|&nbsp;&nbsp;1350|&nbsp;&nbsp;5110 <br>
// MAGIC &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;4|&nbsp;&nbsp;3840|&nbsp;&nbsp;4710 <br>
// MAGIC &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;5|&nbsp;&nbsp;8260|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;0</code><p>
// MAGIC <b>Calculation:</b><br>
// MAGIC Total of row: ![](https://github.com/dataresults-zz/notebooks/blob/master/statistics/formulas/chi_sq_rowtotal.bmp?raw=true)<br>
// MAGIC Total of column: ![](https://github.com/dataresults-zz/notebooks/blob/master/statistics/formulas/chi_sq_coltotal.bmp?raw=true)<br>
// MAGIC Total: ![](https://github.com/dataresults-zz/notebooks/blob/master/statistics/formulas/chi_sq_total.bmp?raw=true)<br>
// MAGIC Expectation: ![](https://github.com/dataresults-zz/notebooks/blob/master/statistics/formulas/chi_sq_expectation.bmp?raw=true)<br>
// MAGIC Observed values: `oij, the observed real values`
// MAGIC <br>
// MAGIC A measure of how the values deviate from what would be expected (eij) is:<br>
// MAGIC    ![](https://github.com/dataresults-zz/notebooks/blob/master/statistics/formulas/chi-square.bmp?raw=true)<br>
// MAGIC   <br>
// MAGIC The null hypothesis of the independence assumption is to be rejected if the p-value of the following Chi-squared test statistics is less than a given significance level α.<br>
// MAGIC 
// MAGIC Apache Spark: The input data type determine which test is conducted. The independence test requires a Matrix as input.<br>
// MAGIC degree of freedom: `df = (i-1)(j-1)`<br>
// MAGIC 
// MAGIC The chi-square test is only applicable if
// MAGIC - the values under test are independent
// MAGIC - the observed count oij of each cell is at least 5

// COMMAND ----------

// MAGIC %md ###### Data for Chi-squared independence test
// MAGIC Data are taken from the GOV.uk - National Statistics
// MAGIC <br><b>Distribution of median and mean income and tax by age range and gender
// MAGIC </b><br>
// MAGIC table: gender\_income\_relations<br>

// COMMAND ----------

// read data from Databricks Hive store
var genderIncomeDf = sqlContext.table("gender_income_relations").select("income","Male","Female")
display(genderIncomeDf.select("*"))

// COMMAND ----------

// MAGIC %md ###### Preprocess data
// MAGIC To calculate chi-squared independence test data must be preprocessed. At first income must be casted from string to <b>numerical</b>.<br>
// MAGIC To form the contingency table income is splitted into 6 categories with same range and associated to males and females.
// MAGIC 
// MAGIC To categorize income a <b>Bucketizer</b> is used.

// COMMAND ----------

// cast income to double
genderIncomeDf = genderIncomeDf.selectExpr(
  "cast(income as double) income",
  "Male",
  "Female"
  )
// calculate min & max
// to demonstrate different methods available
val minIncome = genderIncomeDf.agg(min(("income"))).head.getDouble(0)
val maxIncome = genderIncomeDf.stat.approxQuantile("income", Array(1.0), 0.25)(0)

// calculate splits
val nrOfSplits = 6
val splitSize = (maxIncome - minIncome) / nrOfSplits
val splits = List.tabulate(nrOfSplits + 1)(_ * splitSize + minIncome).toArray

val bucketizer = new Bucketizer()
  .setInputCol("income")
  .setOutputCol("bucketedIncome")
  .setSplits(splits)

// Transform original data into its bucket index.
var bucketedGenderIncomeDf = bucketizer.transform(genderIncomeDf)

display(bucketedGenderIncomeDf.sort("bucketedIncome"))
// register table
bucketedGenderIncomeDf.createOrReplaceTempView("bucketedGenderIncome")

// COMMAND ----------

// MAGIC %md
// MAGIC To calculate chi-squared independence we sum up the occurences in every bucket, separated by male and female. So we have only one reamining value for one bucket.

// COMMAND ----------

bucketedGenderIncomeDf = bucketedGenderIncomeDf.groupBy($"bucketedIncome").agg((sum($"Male").alias("Male")),(sum($"Female").alias("Female"))).select("Male","Female").na.fill(0)
display(bucketedGenderIncomeDf)


// COMMAND ----------

// MAGIC %md ###### Create a matrix from Dataframe
// MAGIC The method to calculate the chi-suared independence  from `org.apache.spark.mllib.stat.test.ChiSqTestResult` needs a matrix as input. So we have to convert the DataFrame to a DenseMatrix. This is done by two helper functions.
// MAGIC 
// MAGIC Although the functions seem very low level it is exact the same way Spark works inside (code is based on Spark source code). Code to convert a DataFrame to a breeze matrix is taken from [SPARK-7492]. A breeze matrix is used to process conversion as it is much easier (and hopefully faster) than with a `mllib.linalg.DenseMatrix`.
// MAGIC 
// MAGIC The second helper function casts from a breeze matrix to a `mllib.linalg.DenseMatrix` as requested by the method mentioned above. This code is taken from `mllib.linalg.Matrices`, but as this method is private we have to define our own.

// COMMAND ----------

def matrixFromDataFrame(df: DataFrame): BM[Double] = {
//   require(df.isLocal, "This method is used to transform a local DataFrame to a matrix. Please " +
//      "use the BlockMatrix.fromDataFrame to generate a Matrix from a distributed DataFrame.")
   val columns = new ArrayBuffer[Column](df.columns.length)
    // The cast will throw an error if the value can't be cast to a Double.
    df.schema.fields.foreach { field =>
      if (field.dataType.isInstanceOf[NumericType]) {
        columns += col(field.name).cast(DoubleType)
      }
    }
    // cast fields to DoubleType
    val doubleDF = df.select(columns: _*).collect()
    val numRows = doubleDF.length
    val numCols = columns.length
    require(numRows.toLong * numCols <= Int.MaxValue,
      s"$numRows x $numCols dense matrix is too large to allocate")
    val mat = BDM.zeros[Double](numRows, numCols)
    var i = 0
    while (i < numRows) {
      var j = 0
      val row = doubleDF(i)
      while (j < numCols) {
        if (!row.isNullAt(j)) mat.update(i, j, row.getDouble(j))
        j += 1
      }
      i += 1
    }
    mat
  }

// COMMAND ----------

  def matrixFromBreeze(breeze: BM[Double]): LM = {
    breeze match {
      case dm: BDM[Double] =>
        new DM(dm.rows, dm.cols, dm.data, dm.isTranspose)
      case sm: BSM[Double] =>
        // There is no isTranspose flag for sparse matrices in Breeze
        new SM(sm.rows, sm.cols, sm.colPtrs, sm.rowIndices, sm.data)
      case _ =>
        throw new UnsupportedOperationException(
          s"Do not support conversion from type ${breeze.getClass.getName}.")
    }
}

// COMMAND ----------

// MAGIC %md ###### Create the matrix
// MAGIC 
// MAGIC Type: `mllib.linalg.DenseMatrix`

// COMMAND ----------

val matrix = matrixFromBreeze(matrixFromDataFrame(bucketedGenderIncomeDf)) 


// COMMAND ----------

// MAGIC %md ###### Process chi-squared independence test

// COMMAND ----------

// conduct Pearson's independence test on the input contingency matrix
val independenceTestResult = Statistics.chiSqTest(matrix)

// COMMAND ----------

// MAGIC %md ###### Z-scores
// MAGIC 
// MAGIC Z-scores are useful to standardize the values of a dataset (e.g. to compare them).
// MAGIC 
// MAGIC The standard score is the signed number of standard deviations by which the value of an observation or data point is above the mean value of what is being observed or measured.
// MAGIC 
// MAGIC Observed values above the mean have positive standard scores, while values below the mean have negative standard scores. The standard score is a dimensionless quantity obtained by subtracting the population mean from an individual raw score and then dividing the difference by the population standard deviation. 
// MAGIC 
// MAGIC This conversion process is called standardizing or normalizing.
// MAGIC 
// MAGIC Calculation:<br>
// MAGIC ![](https://github.com/dataresults-zz/notebooks/blob/master/statistics/formulas/z-score.bmp?raw=true) 

// COMMAND ----------

def calculateZScore(df:DataFrame, colName:String):DataFrame ={
  // calculate average and standard deviation for column
  val aggRow = storeSalesDataDf.agg(avg(colName) as "avg", stddev_samp(colName) as "stddev").first
  val average = aggRow.getDouble(0)
  val sd = aggRow.getDouble(1)
  df.withColumn("z", ((col(colName) - average) / sd))  
}

// COMMAND ----------

val colName = "Sales"
storeSalesDataDf = calculateZScore(storeSalesDataDf, colName)
storeSalesDataDf.show(4)

// COMMAND ----------

// MAGIC %md ###### Z-scores calculation with Apache Spark Standard scaler
// MAGIC The standard scaler implements the methods to calculate the z-scores for every value. Unfortunately it is only applicable for vector columns, so a transfomation with an *VectorAssembler* might be necessary.

// COMMAND ----------

storeSalesDataDf = storeSalesDataDf.drop(colName + "_v")
val assembler = new VectorAssembler()
  .setInputCols(Array(colName))
  .setOutputCol(colName + "_v")
storeSalesDataDf = assembler.transform(storeSalesDataDf)

// COMMAND ----------

storeSalesDataDf = storeSalesDataDf.drop(colName + "_z")
val scaler = new StandardScaler()
  .setInputCol(colName + "_v")
  .setOutputCol(colName + "_z")
  .setWithStd(true)
  .setWithMean(true)

// Compute summary statistics by fitting the StandardScaler.
val scalerModel = scaler.fit(storeSalesDataDf)

// Normalize each feature to have unit standard deviation.
storeSalesDataDf = scalerModel.transform(storeSalesDataDf)
storeSalesDataDf.select(colName, colName + "_z", "z").show(3) 

// COMMAND ----------

// MAGIC %md ###### Cohen's d
// MAGIC An effect size is a quantitative measure of the strength of a phenomenon. Examples of effect sizes are the correlation between two variables, the regression coefficient in a regression, the mean difference, or even the risk with which something happens.
// MAGIC 
// MAGIC To convey the size of an effect compare the difference between groups to the variability within groups. Cohen's d is a statistic intended to do that. 
// MAGIC 
// MAGIC Cohen's d is an effect size used to indicate the standardised difference between two means. It can be used, for example, to accompany reporting of t-test and ANOVA results. It is also widely used in meta-analysis.
// MAGIC 
// MAGIC Cohen's d is an appropriate effect size for the comparison between two means. 
// MAGIC 
// MAGIC Cohen's d can be calculated as the difference between the means divided by the pooled standard deviation:
// MAGIC 
// MAGIC Calculation:<br>
// MAGIC ![](https://github.com/dataresults-zz/notebooks/blob/master/statistics/formulas/cohens_d.bmp?raw=true) 

// COMMAND ----------

def cohenEffectSize(df1:DataFrame, colName1:String, df2:DataFrame, colName2:String):Double = {
  val aggRow1 = df1.agg(avg(colName1) as "g1avg", var_samp(colName1) as "g1var", count(colName1) as "g1cnt").first
  val aggRow2 = df2.agg(avg(colName1) as "g2avg", var_samp(colName1) as "g2var", count(colName1) as "g1cnt").first
  val g1avg = aggRow1.getDouble(0)
  val g1var = aggRow1.getDouble(1)
  val g1cnt = aggRow1.getDouble(2)
  val g2avg = aggRow2.getDouble(0)
  val g2var = aggRow2.getDouble(1) 
  val g2cnt = aggRow1.getDouble(2)
  val diff = g1avg - g2avg
  val pooled_var = (g1cnt * g1var + g2cnt * g2var) / (g1cnt + g2cnt)
  val d = diff / Math.sqrt(pooled_var)
  return d
}

// COMMAND ----------

// MAGIC %md ####Bootstrapping
// MAGIC 
// MAGIC Bootstrapping is a statistical technique of resampling (random sampling with replacement). Bootstrapping is used if the theoretical distribution of a statistic is not known. With bootstrapping repeatedly statistics are calculated based on only a single sample with resampling.
// MAGIC 
// MAGIC Bootstrapping technique allows estimation of the sampling distribution of almost any statistic using random sampling methods.
// MAGIC 
// MAGIC Bootstrapping is the practice of estimating properties of an estimator (such as its variance) by measuring those properties when sampling from an approximating distribution. One standard choice for an approximating distribution is the empirical distribution function of the observed data. In the case where a set of observations can be assumed to be from an independent and identically distributed population, this can be implemented by constructing a number of resamples with replacement, of the observed dataset (and of equal size to the observed dataset).
// MAGIC 
// MAGIC The basic idea of bootstrapping is that inference about a population from sample data, can be modeled by resampling the sample data and performing inference about a sample from resampled data. As the population is unknown, the true error in a sample statistic against its population value is unknowable. In bootstrap-resamples, the 'population' is in fact the sample, and this is known; hence the quality of inference of the 'true' sample from resampled data, is measurable.

// COMMAND ----------

// MAGIC %md ####Application of bootstrap techniques
// MAGIC The application of bootstrap techniques in Apache Spark is based on an article of DMITRY PETROV in https://fullstackml.com/2016/01/19/how-to-check-hypotheses-with-boots....
// MAGIC <br>
// MAGIC 
// MAGIC The most common application with bootstrapping is calculating confidence intervals and use of these confidence intervals as a part of the hypotheses checking process. 
// MAGIC 
// MAGIC <b>Idea behind bootstrap:</b><p>
// MAGIC Sample a data set of size N for hundreds or even thousands times with <b>replacement</b> (this is important) and calculate the estimated metrics for each of the sampled subsets. This process gives you a histogram which is approximately the distribution of the data. This actual distribution can further used for hypothesis testing.
// MAGIC 
// MAGIC <b>Distribution:</b><p>
// MAGIC In general the distribution of a population is unknown, so an approximate must be used.<br>
// MAGIC In a classical statistical approach, an approximation of a distribution of data is done by normal distribution and calculation of  z-scores or student-scores based on <b>theoretical</b> distributions. With <b>bootstrap</b> the distribution of the sample is easy to calculate 2.5% percentile and 97.5% percentiles and approximates the population distribution well.<br>

// COMMAND ----------

// MAGIC %md ######Hypotheses H0 and H1
// MAGIC 
// MAGIC Hypotheses testing is part of the analytical process and isn’t usual for machine learning experts. 
// MAGIC 
// MAGIC In the analytics process, knowing the correlation is not enough, you should know if the hypothesis is correct and what is your level of confidence.
// MAGIC 
// MAGIC If there is a null hypotheses H0 it is easy to check the hypothesis based on the bootstrapping approach. 

// COMMAND ----------

// MAGIC %md ######Checking hypotheses - how to do with Bootstrap and Apache Spark
// MAGIC 
// MAGIC For hypotheses checking the confidence interval for dataset A is calculated by sampling and calculating the p-% confidence interval. If the interval does not contain the value under test (e.g. the mean) then the hypotheses H0 is rejected.
// MAGIC 
// MAGIC If p=0.025 the confident interval starts with 2.5% and ends 97.5% which gives 95% of the items between this interval. <br>
// MAGIC If qt1 <= testvalue <= qt3, then H0 can't be rejected. <br>
// MAGIC “Failed to reject H0” does not man “proof H0”. A failing to reject a hypothesis gives you a pretty strong level of evidence that the hypothesis is correct and you can use this information in your decision making process . But this is not an actual proof.

// COMMAND ----------

// MAGIC %md ######Framework for Bootstrap
// MAGIC 
// MAGIC The bootstrap process is always the same. Resample dates n-times and apply a function to the samples. The results of the function is then sorted and the percentiles at a given point (as parameter) are searched.

// COMMAND ----------

def getConfIntervalByBootstrap(f:(DataFrame, String)  => Double, input:DataFrame, sampleSize:Int, confidence:Double, column:String):(Double, Double) = {
  // with replacement must be true, else bootstrap doesn't wrk
  val withReplacement = true
  val fraction = 1.0
  val r = scala.util.Random
  // upper bound
  val confidenceRight = 1.0 - confidence
  
  // Simulate by sampling and calculating averages for each of subsamples
  val hist = Array.fill(sampleSize){0.0}
    for (i <- 0 to sampleSize-1) {
        hist(i) = f(input.sample(withReplacement, fraction, r.nextLong), column)
       // hist(i) = input.sample(withReplacement, fraction, seed).agg(avg(column)).first.getDouble(0) // mean
    }
  quickSort(hist)
  val left_quantile  = hist((sampleSize*confidence).toInt)
  val right_quantile = hist((sampleSize*confidenceRight).toInt)
  return (left_quantile, right_quantile)
}

// COMMAND ----------

// the user function that should be applied
def userFunc(df:DataFrame, column:String):Double = df.agg(avg(column)).first.getDouble(0) // mean

// COMMAND ----------

val colName = "values"
// read the data from HIVE table, cast to double
val skewdataDf = sqlContext.table("skewdata").selectExpr(
  "cast(" + colName + " as Double) " + colName
)
val (qt1, qt3) = getConfIntervalByBootstrap(userFunc, skewdataDf, 1000, 0.025, colName)


// COMMAND ----------

// MAGIC %md ######Hypotheses checking

// COMMAND ----------

val H0_mean = 30

if (qt1 < H0_mean && H0_mean < qt3) {
    println("We failed to reject H0. It seems like H0 is correct. H0 is: " + H0_mean + " and confidence interval is ["+ qt1 + "," + qt3 + "]")
} else {
    println("We rejected H0. H0 is: " + H0_mean + " and confidence interval is ["+ qt1 + "," + qt3 + "]")
}

// COMMAND ----------

// MAGIC 
// MAGIC %md ###### Checking two columns vith Bootstrap
// MAGIC 
// MAGIC Another important application of bootstrapis to compare two columns of a dataset. Check e.g. if the means of the two datasets are different. This leads us to the usual design of experiment questions – has a change a significant effect?

// COMMAND ----------

def getConfIntervalByBootstrap2Values(f:(DataFrame, String)  => Double, input:DataFrame, sampleSize:Int, confidence:Double, columns:Array[String]):(Double, Double) = {
  // with replacement must be true, else bootstrap doesn't wrk
  val withReplacement = true
  val fraction = 1.0
  val r = scala.util.Random
  // upper bound
  val confidenceRight = 1.0 - confidence
  
  // Simulate by sampling and calculating averages for each of subsamples
  val hist = Array.fill(sampleSize){0.0}
    for (i <- 0 to sampleSize-1) {
      val mean1 = f(input.sample(withReplacement, fraction, r.nextLong), columns(0))
      val mean2 = f(input.sample(withReplacement, fraction, r.nextLong), columns(1))
      hist(i) = mean2 - mean1
    }
  quickSort(hist)
  val left_quantile  = hist((sampleSize*confidence).toInt)
  val right_quantile = hist((sampleSize*confidenceRight).toInt)
  return (left_quantile, right_quantile)
}

// COMMAND ----------

// MAGIC %md ###### Confidence interval for one-tailed hypothesis testing
// MAGIC 
// MAGIC Change the 2.5% and 97.5% percentiles of the interval to 5% percentile at the left side only because of one-side (one-tailed) hypothesis testing. 

// COMMAND ----------

// Let's try to check the same dataset with itself
val columns = Array("values", "values")
val (qt1, qt3) = getConfIntervalByBootstrap2Values(userFunc, skewdataDf, 1000, 0.025, columns)

// A condition was changed because of one-tailed test
if (qt1 > 0) {
    println("We failed to reject H0. It seems like H0 is correct.")
} else {
    println("We rejected H0")
}
