from pyspark.sql import SparkSession
from pyspark.sql.functions import col
import sys
import numpy as np
from scipy import stats
import time
from pyspark.sql.functions import lit

# Initialize SparkSession with reduced parallelism
spark = SparkSession.builder.appName("TaxiDataAnalysis").config("spark.sql.shuffle.partitions", 2).getOrCreate()

# Load the CSV file directly without using inferSchema (expensive operation)
df = spark.read.csv(sys.argv[1], header=False)

# Perform necessary transformations without unnecessary UDF
corrected_df = df.select(
    col("_c0").cast("string").alias("medallion"),
    col("_c1").cast("string").alias("hack_license"),
    col("_c2").cast("timestamp").alias("pickup_datetime"),
    col("_c3").cast("timestamp").alias("dropoff_datetime"),
    col("_c4").cast("integer").alias("trip_time_in_secs"),
    col("_c5").cast("float").alias("trip_distance"),
    col("_c6").cast("float").alias("pickup_longitude"),
    col("_c7").cast("float").alias("pickup_latitude"),
    col("_c8").cast("float").alias("dropoff_longitude"),
    col("_c9").cast("float").alias("dropoff_latitude"),
    col("_c10").cast("string").alias("payment_type"),
    col("_c11").cast("float").alias("fare_amount"),
    col("_c12").cast("float").alias("surcharge"),
    col("_c13").cast("float").alias("mta_tax"),
    col("_c14").cast("float").alias("tip_amount"),
    col("_c15").cast("float").alias("tolls_amount"),
    col("_c16").cast("float").alias("total_amount")
)

# Filter data without using collect() and unnecessary UDF
corrected_df = corrected_df.filter(
    (col("trip_distance") >= 1) & (col("trip_distance") <= 50) &
    (col("fare_amount") >= 3) & (col("fare_amount") <= 200) &
    (col("tolls_amount") >= 3) &
    ((col("dropoff_datetime").cast("long") - col("pickup_datetime").cast("long")) >= 120) &
    ((col("dropoff_datetime").cast("long") - col("pickup_datetime").cast("long")) <= 3600)
)

# Perform linear regression on a sample of the data
trip_distance = corrected_df.select("trip_distance").rdd.flatMap(lambda x: x).collect()
fare_amount = corrected_df.select("fare_amount").rdd.flatMap(lambda x: x).collect()

slope, intercept, r_value, p_value, std_err = stats.linregress(trip_distance, fare_amount)

print("Slope (m):", slope)
print("Intercept (b):", intercept)

start_time = time.time()  # Record start time
computation_time = time.time() - start_time
print("Computation Time:", computation_time, "seconds")

# Add a bias term to the features
corrected_df = corrected_df.withColumn("bias", lit(1.0))

# Select the relevant columns
data_df = corrected_df.select("bias", "trip_distance", "fare_amount")

# Convert DataFrame to RDD of tuples
data_rdd = data_df.rdd.map(lambda row: (row["bias"], row["trip_distance"], row["fare_amount"]))

learning_rate = 0.0001
num_iterations = 20
m, b = 0.1, 0.1

for iteration in range(num_iterations):
  predictions = data_rdd.map(lambda x: (x[0] * m + x[1] * b, x[2]))
  errors = predictions.map(lambda x: x[0] - x[1])

  gradient_m = errors.mean()
  gradient_b = errors.sum()

  m -= learning_rate * gradient_m
  b -= learning_rate * gradient_b

  cost = errors.map(lambda x: x ** 2).mean()
  print(f"Iteration {iteration + 1}: Cost={cost}, Parameters (m, b)=({m}, {b})")

print("Final Model Parameters (m, b):", m, b)

# Stop the SparkSession
spark.stop()