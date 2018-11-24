package edu.tulane.cs.hetml.nlp.sprl.Helpers

import java.io.{FileOutputStream, PrintStream, PrintWriter}
import java.math.BigInteger

import edu.illinois.cs.cogcomp.core.utilities.XmlModel
import edu.illinois.cs.cogcomp.saul.classifier.Results
import edu.illinois.cs.cogcomp.saulexamples.nlp.SpatialRoleLabeling.SpRL2013._
import edu.illinois.cs.cogcomp.saulexamples.nlp.SpatialRoleLabeling.SpRLAnnotation
import edu.tulane.cs.hetml.nlp.sprl.Eval._
import edu.tulane.cs.hetml.nlp.BaseTypes._
import edu.tulane.cs.hetml.nlp.sprl.TextOnly.MultiModalSpRLDataModel._
//import edu.tulane.cs.hetml.nlp.sprl.Pairs.MultiModalSpRLPairClassifiers
import edu.tulane.cs.hetml.nlp.sprl.SpRL2017.{Scene, SpRL2017Document}
import edu.tulane.cs.hetml.vision.ImageTriplet
import org.h2.store.fs.FilePath

import scala.collection.JavaConversions._
import scala.collection.mutable.ListBuffer
import scala.util.control.Breaks.{break, breakable}

/** Created by taher on 2017-02-28.
  */
object ReportHelper {

  def saveImageTripletErrorTypes(rels: List[Relation],
                                 imTripletToRel: Relation => Iterable[ImageTriplet],
                                 resultsDir: String,
                                 isTrain: Boolean,
                                 groundTruth: Relation => String,
                                 typeClassifier: Relation => String
                              ) = {
    val errorWriter = new PrintStream(s"$resultsDir/region_error_analysis_${if (isTrain) "train" else "test"}.txt")
    rels.foreach {
      r =>
        val aligned = imTripletToRel(r)
        if (aligned.nonEmpty) {
          val x = aligned.head
          val features = List(
            x.getTrBox.getX, x.getTrBox.getY, x.getTrBox.getWidth, x.getTrBox.getHeight,
            x.getLmBox.getX, x.getLmBox.getY, x.getLmBox.getWidth, x.getLmBox.getHeight,
            x.getEuclideanDistance, x.getIou, x.getLmAreaBbox, x.getLmAreaImage, x.getLmAspectRatio,
            x.getTrAreaBbox, x.getTrAreaImage, x.getTrAreawrtLM, x.getTrAspectRatio, x.getAbove, x.getBelow,
            x.getLeft, x.getRight, x.getIntersectionArea, x.getUnionArea).mkString("\t\t")

          errorWriter.println(
            x.getImageId + "\t\t" + r.getProperty("ActualId") + "\t\t" + x.getFirstSegId + "\t\t" + x.getTrajector +
              "\t\t" + x.getSecondSegId + "\t\t" + x.getLandmark + "\t\t" + x.getSp + "\t\t" +
              typeClassifier(r) + "\t\t" + groundTruth(r) + "\t\t" + features)
        }
    }
    errorWriter.close()
  }

  def saveEvalResultsFromXmlFile(actualFile: String, predictedFile: String, output: String):Unit = {

    val stream = new FileOutputStream(output)
    val xmlEvaluator = new XmlSpRLEvaluator(actualFile, predictedFile, new OverlapComparer)
    val evaluator = new SpRLEvaluator

    ReportHelper.saveEvalResults(stream, s"Roles", xmlEvaluator.evaluateRoles())

    val relationsEval = xmlEvaluator.evaluateRelations()
    val relEval = relationsEval.head
    ReportHelper.saveEvalResults(stream, s"Relations", relationsEval)

    ReportHelper.saveEvalResults(stream, "General Type", evaluator.evaluateRelationGeneralType(relEval))

    ReportHelper.saveEvalResults(stream, "Specific Type", evaluator.evaluateRelationSpecificType(relEval))

    ReportHelper.saveEvalResults(stream, "Specific Value", evaluator.evaluateRelationRCC8(relEval))

    ReportHelper.saveEvalResults(stream, "FoR", evaluator.evaluateRelationFoR(relEval))

    stream.close()
  }

  def saveAsXmlbio(relations: List[Relation],
                trajectors: List[Phrase],
                indicators: List[Phrase],
                landmarks: List[Phrase],
                generalTypeClassifier: Relation => String,
                specificTypeClassifier: Relation => String,
                RCC8ValueClassifier: Relation => String,
                DirectionClassifier: Relation => String,
                filePath: String): SpRL2017Document = {
    val doc = new SpRL2017Document()
    val trPerSentence = trajectors.filter(_ != dummyPhrase).groupBy(_.getSentence)
    val lmPerSentence = landmarks.filter(_ != dummyPhrase).groupBy(_.getSentence)
    val spPerSentence = indicators.filter(_ != dummyPhrase).groupBy(_.getSentence)
    val relationPerSentence = relations.groupBy(_.getParent.asInstanceOf[Sentence])
    val sentences = trPerSentence.keys.toSet.union(lmPerSentence.keys.toSet).union(spPerSentence.keys.toSet)
      .union(relationPerSentence.keys.toSet).toList.sortBy(_.getId)
    val sceneIds = sentences.map(_.getDocument.getId)
    sentences.groupBy(_.getDocument.getId).foreach { case (sId, sentenceList) =>
      val scene = new Scene()
      scene.setDocNo(sId)
      sentenceList.foreach(s => {
        val sent = new edu.tulane.cs.hetml.nlp.sprl.SpRL2017.Sentence()
        sent.setStart(s.getStart)
        sent.setEnd(s.getEnd)
        sent.setText(s.getText)
        sent.setId(s.getId)

        val rel = if (relationPerSentence.containsKey(s)) relationPerSentence(s) else List()

        val tr = (if (trPerSentence.containsKey(s)) trPerSentence(s) else List()).toSet
          .union(rel.map(_.getArgument(0).asInstanceOf[Phrase]).toSet)
          .map(x => setXmlRoleValues(s, x, new TRAJECTOR)).toList.sortBy(_.getStart)

        val sp = (if (spPerSentence.containsKey(s)) spPerSentence(s) else List()).toSet
          .union(rel.map(_.getArgument(1).asInstanceOf[Phrase]).toSet)
          .map(x => setXmlRoleValues(s, x, new SPATIALINDICATOR)).toList.sortBy(_.getStart)

        val lm = (if (lmPerSentence.containsKey(s)) lmPerSentence(s) else List()).toSet
          .union(rel.map(_.getArgument(2).asInstanceOf[Phrase]).toSet)
          .map(x => setXmlRoleValues(s, x, new LANDMARK)).toList
          .sortBy(_.getStart)

        sent.setTrajectors(tr)
        sent.setLandmarks(lm)
        sent.setSpatialindicators(sp)
        sent.setRelations(
          getXmlRelations(rel, generalTypeClassifier, specificTypeClassifier, RCC8ValueClassifier, DirectionClassifier))
        scene.getSentences.add(sent)
      })
      doc.getScenes.add(scene)
    }
    XmlModel.write(doc, filePath)
    doc
  }
  def saveAsXml(relations: List[Relation],
                trajectors: List[Phrase],
                indicators: List[Phrase],
                landmarks: List[Phrase],
                generalTypeClassifier: Relation => String,
                specificTypeClassifier: Relation => String,
                RCC8ValueClassifier: Relation => String,
                DirectionClassifier: Relation => String,
                filePath: String): SpRL2017Document = {
    val doc = new SpRL2017Document()
    val trPerSentence = trajectors.filter(_ != dummyPhrase).groupBy(_.getSentence)
    val lmPerSentence = landmarks.filter(_ != dummyPhrase).groupBy(_.getSentence)
    val spPerSentence = indicators.filter(_ != dummyPhrase).groupBy(_.getSentence)
    val relationPerSentence = relations.groupBy(_.getParent.asInstanceOf[Sentence])
    val sentences = trPerSentence.keys.toSet.union(lmPerSentence.keys.toSet).union(spPerSentence.keys.toSet)
      .union(relationPerSentence.keys.toSet).toList.sortBy(_.getId)
    val sceneIds = sentences.map(_.getDocument.getId)
    sentences.groupBy(_.getDocument.getId).foreach { case (sId, sentenceList) =>
      val scene = new Scene()
      scene.setDocNo(sId)
      sentenceList.foreach(s => {
        val sent = new edu.tulane.cs.hetml.nlp.sprl.SpRL2017.Sentence()
        sent.setStart(s.getStart)
        sent.setEnd(s.getEnd)
        sent.setText(s.getText)
        sent.setId(s.getId)

        val rel = if (relationPerSentence.containsKey(s)) relationPerSentence(s) else List()

        val tr = (if (trPerSentence.containsKey(s)) trPerSentence(s) else List()).toSet
          .union(rel.map(_.getArgument(0).asInstanceOf[Phrase]).toSet)
          .map(x => setXmlRoleValues(s, x, new TRAJECTOR)).toList.sortBy(_.getStart)

        val sp = (if (spPerSentence.containsKey(s)) spPerSentence(s) else List()).toSet
          .union(rel.map(_.getArgument(1).asInstanceOf[Phrase]).toSet)
          .map(x => setXmlRoleValues(s, x, new SPATIALINDICATOR)).toList.sortBy(_.getStart)

        val lm = (if (lmPerSentence.containsKey(s)) lmPerSentence(s) else List()).toSet
          .union(rel.map(_.getArgument(2).asInstanceOf[Phrase]).toSet)
          .map(x => setXmlRoleValues(s, x, new LANDMARK)).toList
          .sortBy(_.getStart)

        sent.setTrajectors(tr)
        sent.setLandmarks(lm)
        sent.setSpatialindicators(sp)
        sent.setRelations(
          getXmlRelations(rel, generalTypeClassifier, specificTypeClassifier, RCC8ValueClassifier, DirectionClassifier))
        scene.getSentences.add(sent)
      })
      doc.getScenes.add(scene)
    }
    XmlModel.write(doc, filePath)
    doc
  }

  private def getXmlRelations(
                               rel: List[Relation],
                               generalTypeClassifier: Relation => String,
                               specificTypeClassifier: Relation => String,
                               RCC8ValueClassifier: Relation => String,
                               DirectionClassifier: Relation => String
                             ): List[RELATION] = {
    rel.map(x => {
      val r = new RELATION
      //r.setId(x.getId)
      r.setTrajectorId("T_" + getArgId(x, 0))
      r.setSpatialIndicatorId("SP_" + getArgId(x, 1))
      r.setLandmarkId("L_" + getArgId(x, 2))
      r.setGeneralType(generalTypeClassifier(x))
      r.setSpecificType(specificTypeClassifier(x))
      val rcc8 = RCC8ValueClassifier(x)
      if("None".equalsIgnoreCase(rcc8))
        r.setRCC8Value(DirectionClassifier(x))
      else
        r.setRCC8Value(rcc8)
      r
    })
  }

  private def getArgId(x: Relation, arg: Int) = {
    if (x.getArgumentId(arg) == dummyPhrase.getId)
      x.getParent.asInstanceOf[Sentence].getId + "_null" else x.getArgumentId(arg)
  }

  private def setXmlRoleValues[T](s: Sentence, x: Phrase, t: T): T = {
    (t match {
      case y: TRAJECTOR =>
        val prefix = "T_"
        if (x == dummyPhrase) {
          y.setId(prefix + s.getId + "_null")
          y.setStart(-1)
          y.setEnd(-1)
          return t
        }
        y.setStart(x.getStart)
        y.setEnd(x.getEnd)
        y.setText(x.getText)
        y.setId(prefix + x.getId)
        y
      case y: LANDMARK =>
        val prefix = "L_"
        if (x == dummyPhrase) {
          y.setId(prefix + s.getId + "_null")
          y.setStart(-1)
          y.setEnd(-1)
          return t
        }
        y.setStart(x.getStart)
        y.setEnd(x.getEnd)
        y.setText(x.getText)
        y.setId(prefix + x.getId)
        y
      case y: SPATIALINDICATOR =>
        val prefix = "SP_"
        if (x == dummyPhrase) {
          y.setId(prefix + s.getId + "_null")
          y.setStart(-1)
          y.setEnd(-1)
          return t
        }
        y.setStart(x.getStart)
        y.setEnd(x.getEnd)
        y.setText(x.getText)
        y.setId(prefix + x.getId)
        y
    }).asInstanceOf[T]
  }

  def reportTripletResults(
                             actualFile: String,
                             resultsDir: String,
                             resultFilePrefix: String,
                             predicted: List[Relation],
                             globalSpan: Boolean
                           )={
    val actual = new SpRLXmlReader(actualFile, globalSpan).getTripletsWithArguments()
    reportRelationResults(resultsDir, resultFilePrefix, actual, predicted, new OverlapComparer, 3)
  }
  def reportRelationResults(
                             resultsDir: String,
                             resultFilePrefix: String,
                             a: List[Relation],
                             p: List[Relation],
                             comparer: EvalComparer,
                             argumentCount: Int
                           ) = {
    val actual = a.map(r => (r, getRelationEval(r)))
    val predicted = p.map(r => (r, getRelationEval(r)))

    val tp = ListBuffer[(Relation, Relation)]()
    actual.foreach { a =>
      breakable {
        predicted.foreach { p =>
          if (comparer.isEqual(a._2, p._2)) {
            tp += ((a._1, p._1))
            break()
          }
        }
      }
    }
    val fp = predicted.filterNot(x => tp.exists(_._2 == x._1))
    val fn = actual.filterNot(x => tp.exists(_._1 == x._1))

    var writer = new PrintWriter(s"$resultsDir/${resultFilePrefix}-fp.txt")
    fp.groupBy(x => getDocumentId(x._1.getArgument(1))).toList.sortBy(_._1).foreach {
      case (key, list) => {
        writer.println(s"===================================== ${key} ==================================")
        list.foreach {
          case (r, _) =>
            writer.println(argumentsString(r, argumentCount))
        }
      }
    }
    writer.close()

    writer = new PrintWriter(s"$resultsDir/${resultFilePrefix}-fn.txt")
    fn.groupBy(x => getDocumentId(x._1.getArgument(1))).toList.sortBy(_._1).foreach {
      case (key, list) => {
        writer.println(s"===================================== ${key} ==================================")
        list.foreach {
          case (r, _) =>
            writer.println(s"${r.getId} : ${argumentsString(r, argumentCount)}")
        }
      }
    }
    writer.close()

    writer = new PrintWriter(s"$resultsDir/${resultFilePrefix}-tp.txt")
    tp.groupBy(x => getDocumentId(x._1.getArgument(1))).toList.sortBy(_._1).foreach {
      case (key, list) => {
        writer.println(s"===================================== ${key} ==================================")
        list.foreach {
          case (a, p) =>
            val actualArgs = a.getArguments.toList
            val predictedArgs = p.getArguments.toList
            writer.println(s"${a.getId} : ${argumentsString(a, argumentCount)} == ${argumentsString(p, argumentCount)}")
        }
      }
    }
    writer.close()

    val evaluator = new SpRLEvaluator()
    val actualEval = new RelationsEvalDocument(actual.map(_._2))
    val predictedEval = new RelationsEvalDocument(predicted.map(_._2))
    val results = evaluator.evaluateRelations(actualEval, predictedEval, comparer)
    SpRLEvaluator.printEvaluation(results)
    results
  }

  def saveCandidateList(isTrain: Boolean, candidateRelations: List[Relation]): Unit = {

    def getArg(i: Int, r: Relation) = r.getArgument(i).getText.toLowerCase

//    def print(r: Relation) = {
//      MultiModalSpRLPairClassifiers.pairFeatures(FeatureSets.BaseLine)
//        .map(prop => printVal(prop(r))).mkString(" | ")
//    }

    def printVal(v: Any) = {
      v match {
        case x: List[_] => x.mkString(", ")
        case _ => v.toString
      }
    }

    val name = if (isTrain) "Train" else "Test"
    val writer = new PrintWriter(s"data/mSprl/results/RoleCandidates-${name}.txt")
    candidateRelations.foreach(x =>
      writer.println(s"(${getArg(0, x)}, ${getArg(1, x)})[${print(x)}] -> ${x.getProperty("RelationType")}"))
    writer.close()
  }

  def saveEvalResults(stream: FileOutputStream, caption: String, results: Results): Unit =
    saveEvalResults(stream, caption, convertToEval(results))

  def saveEvalResults(stream: FileOutputStream, caption: String, results: Seq[SpRLEvaluation], ignore:Seq[String] = Seq("none")): Unit = {
    val writer = new PrintStream(stream, true)
    writer.println("===========================================================================")
    writer.println(s" ${caption}")
    writer.println("---------------------------------------------------------------------------")
    SpRLEvaluator.printEvaluation(stream, results.filterNot(x => ignore.exists(i => x.getLabel.equalsIgnoreCase(i))))
    writer.println()
  }

  def reportRoleStats(instances: List[NlpBaseElement], candidates: List[NlpBaseElement], tagName: String): Unit = {

    val roleInstances = instances.filter(_.containsProperty(s"${tagName}_id"))
    val actual = roleInstances.map(_.getPropertyValues(s"${tagName}_id").size()).sum
    val missingTokens = roleInstances.toSet.diff(candidates.toSet).toList.map(_.getText.toLowerCase())
    val missing = actual - candidates.map(_.getPropertyValues(s"${tagName}_id").size()).sum

    println(s"Candidate ${tagName}: ${candidates.size}")
    println(s"Actual ${tagName}: $actual")
    println(s"Missing ${tagName} in the candidates: $missing (${missingTokens.mkString(", ")})")
  }

  def reportRelationStats(candidateRelations: List[Relation], goldTrajectorRelations: List[Relation],
                          goldLandmarkRelations: List[Relation]): Unit = {

    val missedTrSp = goldTrajectorRelations.size - candidateRelations.count(_.getProperty("RelationType") == "TR-SP")
    println(s"actual TR-SP: ${goldTrajectorRelations.size}")
    println(s"Missing TR-SP in the candidates: $missedTrSp")
    val missingTrRelations = goldTrajectorRelations
      .filterNot(r => candidateRelations.exists(x => x.getProperty("RelationType") == "TR-SP" && x.getProperty("ActualId") == r.getId))
      .map(_.getId)
    println(s"missing relations from TR-SP: (${missingTrRelations.mkString(", ")})")

    val missedLmSp = goldLandmarkRelations.size - candidateRelations.count(_.getProperty("RelationType") == "LM-SP")
    println(s"actual LM-SP: ${goldLandmarkRelations.size}")
    println(s"Missing LM-SP in the candidates: $missedLmSp")
    val missingLmRelations = goldLandmarkRelations
      .filterNot(r => candidateRelations.exists(x => x.getProperty("RelationType") == "LM-SP" && x.getProperty("ActualId") == r.getId))
      .map(_.getId)
    println(s"missing relations from LM-SP: (${missingLmRelations.mkString(", ")})")
  }

  private def argumentsString(r: Relation, count: Int) = {
    Range(0, count)
      .map(i => r.getArgument(i).getText + "[" + r.getArgument(i).getStart + ":" + r.getArgument(i).getEnd + "]")
      .mkString(" -> ")
  }

  def convertToEval(r: Results): Seq[SpRLEvaluation] = r.perLabel
    .map(x => {
      var p = if (x.predictedSize == 0) 1.0 else x.precision
      var r = if (x.labeledSize == 0) 1.0 else x.recall
      var f1 = if (x.predictedSize == 0) if (x.labeledSize == 0) 1.0 else 0.0 else x.f1
      if(p.isNaN)
        p = 0
      if(r.isNaN)
        r = 0
      if(f1.isNaN)
        f1 = 0
      val result = new SpRLEvaluation(x.label, p * 100, r * 100, f1 * 100, x.labeledSize, x.predictedSize)
      result
    })

  def getRelationEval(r: Relation): RelationEval = {
    val tr = r.getArgument(0)
    val sp = r.getArgument(1)
    val lm = r.getArgument(2)
    new RelationEval(tr.getGlobalStart, tr.getGlobalEnd, sp.getGlobalStart, sp.getGlobalEnd,
      lm.getGlobalStart, lm.getGlobalEnd)
  }

  private def notNull(t: NlpBaseElement) = {
    t != null && t.getId != dummyPhrase.getId && t.getStart >= 0
  }

  private def getDocumentId(e: NlpBaseElement) = {
    e match {
      case x: Token => x.getDocument.getId
      case x: Phrase => x.getDocument.getId
      case _ => e.asInstanceOf[Token].getDocument.getId
    }
  }
}
