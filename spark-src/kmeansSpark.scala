import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf
import org.apache.spark.mllib.clustering.{KMeans, KMeansModel}
import org.apache.spark.mllib.linalg.Vectors

object Main {
  def main(args: Array[String]) {
 
    val conf = new SparkConf().setAppName("KMeans")
    val sc = new SparkContext(conf)
    
    //Read data
    val features_csv = sc.textFile("gs://bigdata-iris/creditcard_features.csv")
    val labels_csv = sc.textFile("gs://bigdata-iris/creditcard_labels.csv")

    // Remove headers
    val featuresHeader = features_csv.first()
    val features = features_csv.filter(row => row != featuresHeader)

    val labelsHeader = labels_csv.first()
    val labels = labels_csv.filter(row => row != labelsHeader).map(s => s.toInt)

    //.map(line=> line.split(',')(1) + "," + line.split(',')(2) + "," + line.split(',')(3) + "," + line.split(',')(4))
    val convData = features.map(s => Vectors.dense(s.split(',').map(_.toDouble))).cache()

    val numClusters = 0
    val numIterations = 10000

    // Elbow method
    for (numClusters <- 1 to 10) {
        val clusters = KMeans.train(convData, numClusters, numIterations)
        val WSSSE = clusters.computeCost(convData)
        println(WSSSE)
    }

    // Use model with 2 clusters
    val clusters = KMeans.train(convData, 2, numIterations)

    // Prepare to eval model
    val pred = clusters.predict(convData)
    val aux1 = pred.zipWithIndex.map((x) =>(x._2, x._1))
    val aux2 = labels.zipWithIndex.map((x) =>(x._2, x._1))
    val combined = aux1.join(aux2).map(x => x._2)

    println("acc:")
    println(combined.filter(p => p._1 == p._2).count().toFloat / combined.count())

    // Save labels
    //pred.saveAsTextFile("gs://bigdata-iris/labels")

    // Save original features
    //val convDataArr = convData.map(arr => arr.toArray)
    //convDataArr.map(arr => arr(0) + "," + arr(1)).saveAsTextFile("gs://bigdata-iris/data")

    println("Done")
    sc.stop()
    
  }
}
