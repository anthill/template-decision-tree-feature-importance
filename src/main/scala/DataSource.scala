package org.template.regression

import io.prediction.controller.PDataSource
import io.prediction.controller.EmptyEvaluationInfo
import io.prediction.controller.EmptyActualResult
import io.prediction.controller.Params
import io.prediction.data.storage.Event
import io.prediction.data.storage.Storage

import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg.Vectors

import grizzled.slf4j.Logger

case class DataSourceParams(
  appId: Int,
  evalK: Option[Int]  // define the k-fold parameter.
) extends Params

class DataSource(val dsp: DataSourceParams)
  extends PDataSource[TrainingData, EmptyEvaluationInfo, Query, ActualResult] {

  @transient lazy val logger = Logger[this.type]

  def readFromDbAndFile(sc: SparkContext, filename: String, metadata: String): TrainingData = {
    val eventsDb = Storage.getPEvents()
    val labeledPoints: RDD[LabeledPoint] = eventsDb.aggregateProperties(
      appId = dsp.appId,
      entityType = "row",
      // only keep entities with these required properties defined
      required = Some(List(
        "target", "attr0", "attr1", "attr2", "attr3", 
        "attr4", "attr5", "attr6", "attr7", "attr8",
        "attr9", "attr10", "attr11", "attr12")))(sc)
      // aggregateProperties() returns RDD pair of
      // entity ID and its aggregated properties
      .map { case (entityId, properties) =>
        try {
          LabeledPoint(properties.get[Double]("target"),
            Vectors.dense(Array(
              properties.get[Double]("attr0"),
              properties.get[Double]("attr1"),
              properties.get[Double]("attr2"),
              properties.get[Double]("attr3"),
              properties.get[Double]("attr4"),
              properties.get[Double]("attr5"),
              properties.get[Double]("attr6"),
              properties.get[Double]("attr7"),
              properties.get[Double]("attr8"),
              properties.get[Double]("attr9"),
              properties.get[Double]("attr10"),
              properties.get[Double]("attr11"),
              properties.get[Double]("attr12")
            ))
          )
        } catch {
          case e: Exception => {
            logger.error(s"Failed to get properties ${properties} of" +
              s" ${entityId}. Exception: ${e}.")
            throw e
          }
        }
      }.cache()
      println(s"Read ${labeledPoints.count} labeled Points from db.")

      // read from file
      val textFile = sc.textFile(filename)
      val metadataFile = sc.textFile(metadata)

      val labeledPoints2 = textFile
        .map { line =>
          val linesplit = line.split(",")
          val targetStr = linesplit.head
          val features = linesplit.drop(1).map(_.toDouble).toArray

          LabeledPoint(targetStr.toDouble, Vectors.dense(features))
        }.cache()
      println(s"Read ${labeledPoints2.count} labeled Points from file.")

      // read metadata
      // metadata contains the number of categories with "ca" prepend if the
      // feature is categorical and "co" if the the feature is continuous
      // for example: target,ca13,co,ca3,co,co
      val categoricalFeaturesInfo = metadataFile.first.split(",").drop(1).zipWithIndex
      .filter{case (value, index) => value != "co"}
      .map{case (value, index) => index -> value.drop(2).toInt }.toMap

      val labeledPointsAll = labeledPoints.union(labeledPoints2)
      val Array(trainingPoints, testingPoints) = labeledPointsAll.randomSplit(Array(0.8, 0.2))
      println(s"Read ${trainingPoints.count} for training.")
      println(s"Read ${testingPoints.count} for testing.")

      new TrainingData(trainingPoints, testingPoints, categoricalFeaturesInfo)

  }


  override
  def readTraining(sc: SparkContext): TrainingData = {

    readFromDbAndFile(sc, "data/learning.csv", "data/learning_metadata.csv")

  }

  override
  def readEval(sc: SparkContext): Seq[(TrainingData, EmptyEvaluationInfo, RDD[(Query, ActualResult)])] = {
    require(!dsp.evalK.isEmpty, "DataSourceParams.evalK must not be None")

    val trainingData = readFromDbAndFile(sc, "data/learning.csv", "data/learning_metadata.csv")

    // K-fold splitting
    val evalK = dsp.evalK.get
    val indexedPoints: RDD[(LabeledPoint, Long)] = trainingData.trainingPoints.zipWithIndex

    (0 until evalK).map { idx => 
      val trainingPoints = indexedPoints.filter(_._2 % evalK != idx).map(_._1)
      val testingPoints = indexedPoints.filter(_._2 % evalK == idx).map(_._1)

      (
        new TrainingData(trainingPoints, trainingPoints, Map[Int, Int]()),
        new EmptyEvaluationInfo(),
        testingPoints.map { 
          p => (new Query(p.features.toArray), new ActualResult(p.label)) 
        }
      )
    }
  }

}

class TrainingData(
  val trainingPoints: RDD[LabeledPoint],
  val testingPoints: RDD[LabeledPoint],
  val categoricalFeaturesInfo: Map[Int, Int]
) extends Serializable
