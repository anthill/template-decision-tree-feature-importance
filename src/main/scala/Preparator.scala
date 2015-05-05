package org.template.regression

import io.prediction.controller.PPreparator

import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.regression.LabeledPoint

class PreparedData(
  val trainingPoints: RDD[LabeledPoint],
  val testingPoints: RDD[LabeledPoint],
  val categoricalFeaturesInfo: Map[Int, Int]
) extends Serializable

class Preparator extends PPreparator[TrainingData, PreparedData] {

  def prepare(sc: SparkContext, trainingData: TrainingData): PreparedData = {
    new PreparedData(trainingData.trainingPoints, trainingData.testingPoints, trainingData.categoricalFeaturesInfo)
  }
}
