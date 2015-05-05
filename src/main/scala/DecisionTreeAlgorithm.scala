package org.template.regression

import io.prediction.controller.P2LAlgorithm
import io.prediction.controller.Params

import org.apache.spark.mllib.tree.DecisionTree
import org.apache.spark.mllib.tree.model.DecisionTreeModel
import org.apache.spark.mllib.tree.model.Node
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.mllib.stat.Statistics

import grizzled.slf4j.Logger
import spray.json._
import java.io._
import scala.collection.mutable.{ Map => MutableMap }
import scala.collection.immutable.ListMap

case class AlgorithmParams(
  impurity: String,
  maxDepth: Int,
  maxBins: Int
) extends Params


// extends P2LAlgorithm because the MLlib's DecisionTreeModel doesn't contain RDD.
class DecisionTreeAlgorithm(val ap: AlgorithmParams)
  extends P2LAlgorithm[PreparedData, DecisionTreeModel, Query, PredictedResult] {

  @transient lazy val logger = Logger[this.type]

  def train(sc: SparkContext, data: PreparedData): DecisionTreeModel = {
    // MLLib DecisionTree cannot handle empty training data.
    require(!data.trainingPoints.take(1).isEmpty,
      s"RDD[labeldPoints] in PreparedData cannot be empty." +
      " Please check if DataSource generates TrainingData" +
      " and Preprator generates PreparedData correctly.")

    val model = DecisionTree.trainRegressor(data.trainingPoints, data.categoricalFeaturesInfo, ap.impurity, ap.maxDepth, ap.maxBins)

    //eval the model
    val labelsAndPredictions = data.testingPoints.map { point =>
      val prediction = model.predict(point.features)
      (point.label, prediction)
    }

    val trainMSE = Math.sqrt(labelsAndPredictions.map{ case(v, p) => math.pow((v - p), 2)}.mean())
    println("Training Mean Squared Error = " + trainMSE)

    val seriesX = labelsAndPredictions.map{case(v, p) => v}
    val seriesY = labelsAndPredictions.map{case(v, p) => p}
    val pearson = Statistics.corr(seriesX, seriesY, "pearson")
    println("Pearson correlation:\n" + pearson)

    // show feature importance
    def computeFeatureImportance(model: DecisionTreeModel): MutableMap[Int, Double] = {
        // same indices than otherFeatures
        val importance = MutableMap[Int, Double]();

        def processNode(n : Node): Unit = {

            n.split match{
                case Some(split) =>
                    val featureIndex = split.feature
                    //val featureName = otherFeatures(featureIndex).name;
                    //print(s"${featureName}/")

                    if(!importance.contains(featureIndex))
                        importance.put(featureIndex, 0);

                    importance(featureIndex) += n.impurity

                    n.leftNode match{
                        case Some(n) => importance(featureIndex) -= n.impurity
                        case None => 
                    }

                    n.rightNode match{
                        case Some(n) => importance(featureIndex) -= n.impurity
                        case None => 
                    }

                case None => 
            }

            //println(s"${n.impurity}");

            // process children
            n.leftNode match{
                case Some(node) => processNode(node)
                case None => 
            }

            n.rightNode match{
                case Some(node) => processNode(node)
                case None => 
            }

        }

        processNode(model.topNode);

        return importance;

    }

    val importances = computeFeatureImportance(model)
    var sumImportances = 0.0
    importances foreach {case (featureIndex, value) => sumImportances += value}

    val featuresIndexes = (0 until data.trainingPoints.first.features.size)toList

    val normalizedFeatureImportancesList = featuresIndexes.zipWithIndex.map({
        case (feature, i) => 
            if(!importances.contains(i))
                importances.put(i, 0);
            (feature, importances(i)/sumImportances);
    })

    // http://stackoverflow.com/a/9966265
    val sortedFeatures = new ListMap() ++ normalizedFeatureImportancesList.sortBy(_._2).reverse

    sortedFeatures foreach {
        case (name, importance) => 
            println (name + "\t-->\t" + importance )
    }

    // persist the tree
    object NodeJsonProtocol extends DefaultJsonProtocol {
      implicit object NodeJsonFormat extends RootJsonFormat[Node] {
        def write(c: Node) = {
          c.split match {
            case Some(split) =>
              JsObject(
                "feature" -> JsString(split.feature.toString),
                "type" -> JsString(split.featureType.toString),
                "ruleCategorical" -> split.categories.toJson,
                "ruleRegression" -> JsNumber(split.threshold),
                "impurity" -> JsNumber(c.impurity),
                "trueNode" -> c.leftNode.toJson,
                "falseNode" -> c.rightNode.toJson
              )
            case None =>
              JsObject(
                "leafValue" -> JsNumber(c.predict.predict),
                "impurity" -> JsNumber(c.impurity)
              )
            }
          }
          def read(value: JsValue) = {
            throw new DeserializationException("Color expected")
          }
        
      }
    }
    import NodeJsonProtocol._

    val writer = new PrintWriter(new File("data/decisionTree.json" ))
    writer.write(model.topNode.toJson.toString)
    writer.close()
    model
  }

  def predict(model: DecisionTreeModel, query: Query): PredictedResult = {
    val label = model.predict(Vectors.dense(query.features))
    new PredictedResult(label)
  }

}
