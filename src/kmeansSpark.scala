import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf
import org.apache.spark.mllib.clustering.{KMeans, KMeansModel}
import org.apache.spark.mllib.linalg.Vectors

object Main {
  def main(args: Array[String]) {
 
    val conf = new SparkConf().setAppName("KMeans")
    val sc = new SparkContext(conf)
    
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

    println("Done")
    sc.stop()
    
  }
}