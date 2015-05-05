package org.template.regression

import io.prediction.controller.AverageMetric
import io.prediction.controller.EmptyEvaluationInfo
import io.prediction.controller.EngineParams
import io.prediction.controller.EngineParamsGenerator
import io.prediction.controller.Evaluation
import io.prediction.controller.Workflow
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.rdd.RDD

case class Precision 
  extends AverageMetric[EmptyEvaluationInfo, Query, PredictedResult, ActualResult] {
  def calculate(query: Query, predicted: PredictedResult, actual: ActualResult)
  : Double = (predicted.value - actual.value)
}

object PrecisionEvaluation extends Evaluation {
  // Define Engine and Metric used in Evaluation
  engineMetric = (RegressionTreeEngine(), new Precision())
}

object EngineParamsList extends EngineParamsGenerator {
  // Define list of EngineParams used in Evaluation

  // First, we define the base engine params. It specifies the appId from which
  // the data is read, and a evalK parameter is used to define the
  // cross-validation.
  private[this] val baseEP = EngineParams(
    dataSourceParams = DataSourceParams(appId = 2, evalK = Some(5)))

  // Second, we specify the engine params list by explicitly listing all
  // algorithm parameters. In this case, we evaluate 2 engine params, each with
  // a different algorithm params value.
  engineParamsList = Seq(
    baseEP.copy(algorithmParamsList = Seq(("regressiontree", AlgorithmParams("variance", 15, 500)))),
    baseEP.copy(algorithmParamsList = Seq(("regressiontree", AlgorithmParams("variance", 20, 500))))
  )
}
