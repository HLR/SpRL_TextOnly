package edu.tulane.cs.hetml.nlp.sprl.TextOnly

import edu.illinois.cs.cogcomp.saul.util.Logging
import edu.tulane.cs.hetml.nlp.BaseTypes.Document
import edu.tulane.cs.hetml.nlp.sprl.Helpers.{FeatureSets, ReportHelper}
import MultiModalPopulateData._
import MultiModalSpRLDataModel.{phrases, triplets}
import edu.tulane.cs.hetml.nlp.sprl.TextOnly.MultiModalSpRLTripletClassifiers._
import edu.tulane.cs.hetml.nlp.sprl.TextOnly.SpRLApp.expName
import edu.tulane.cs.hetml.nlp.sprl.TextOnly.TripletSentenceLevelConstraintClassifiers._
import tripletConfigurator.{resultsDir, populateImages}

import scala.io.Source

object PlainTextApp extends App with Logging {
  populateImages = false
  featureSet = FeatureSets.BaseLine
  val classifiers = List(
    IndicatorRoleClassifier,
    TrajectorRoleClassifier,
    LandmarkRoleClassifier,
    TripletRelationClassifier,
    TripletGeneralTypeClassifier,
    TripletRegionClassifier,
    TripletDirectionClassifier
  )
  classifiers.foreach {
    c =>
      c.modelDir = s"models/mSpRL/triplet/$featureSet/"
      c.load()
  }

  val doc = new Document("doc1")
  val lines = Source.fromFile("data/TESTnew.txt", enc = "UTF-8").getLines()
    .toList.filter(x => x.trim.nonEmpty).map(x=> x + ".")
  doc.setText(lines.mkString("\n"))
  populateDataFromPlainTextDocuments(List(doc), x => IndicatorRoleClassifier(x) == "true")

  val trajectors = phrases.getTestingInstances.filter(x => TRConstraintClassifier(x) == "Trajector").toList
  val landmarks = phrases.getTestingInstances.filter(x => LMConstraintClassifier(x) == "Landmark").toList
  val indicators = phrases.getTestingInstances.filter(x => IndicatorRoleClassifier(x) == "Indicator").toList

  val tripletList = triplets.getTestingInstances
    .filter(x => TripletRelationConstraintClassifier(x) == "true").toList


  ReportHelper.saveAsXml(tripletList, trajectors, indicators, landmarks,
    x => TripletGeneralTypeConstraintClassifier(x),
    x => "",
    x => TripletRegionConstraintClassifier(x),
    x => TripletDirectionConstraintClassifier(x),
    s"$resultsDir/test.xml")

}
