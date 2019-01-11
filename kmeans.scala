import org.apache.spark.mllib.clustering.{KMeans, KMeansModel}
import org.apache.spark.mllib.linalg.Vectors

val r_csv = sc.textFile("gs://bigdata-iris/movies_processed.csv")
val header = r_csv.first()
val r_csv2 = r_csv.filter(row => row != header)
val t = r_csv2.map(line=> line.split(',')(1) + "," + line.split(',')(2) + "," + line.split(',')(3) + "," + line.split(',')(4))
val convData = t.map(s => Vectors.dense(s.split(',').map(_.toDouble))).cache()

val numClusters = 0
val numIterations = 300
val clusters = None

// Elbow method
for (numClusters <- 1 to 10) {
    val clusters = KMeans.train(convData, numClusters, numIterations)
    val WSSSE = clusters.computeCost(convData)
}


//val foo = clusters.predict(convData)

// Save labels
//foo.saveAsTextFile("gs://bigdata-iris/labels")

// Save original features
//val convDataArr = convData.map(arr => arr.toArray)
//convDataArr.map(arr => arr(0) + "," + arr(1) + "," + arr(2) + "," + arr(3)).saveAsTextFile("gs://bigdata-iris/data")