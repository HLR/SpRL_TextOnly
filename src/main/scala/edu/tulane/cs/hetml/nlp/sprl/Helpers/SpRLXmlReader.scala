package edu.tulane.cs.hetml.nlp.sprl.Helpers

import edu.tulane.cs.hetml.nlp.BaseTypes._
import edu.tulane.cs.hetml.nlp.sprl.TextOnly.MultiModalSpRLDataModel.dummyPhrase
import edu.tulane.cs.hetml.nlp.Xml.NlpXmlReader
import edu.tulane.cs.hetml.nlp.XmlMatchings
import edu.tulane.cs.hetml.nlp.sprl.Eval.OverlapComparer

import scala.collection.JavaConversions._
import scala.collection.mutable.ListBuffer

/** Created by taher on 2017-02-28.
  */
class SpRLXmlReader(dataPath: String, useGlobalSpan: Boolean) {

  val trTag = "TRAJECTOR"
  val lmTag = "LANDMARK"
  val spTag = "SPATIALINDICATOR"
  val relationTag = "RELATION"

  lazy val reader = createXmlReader()

  def setPairTypes(candidateRelations: List[Relation], populateNullPairs: Boolean): Unit = {

    val goldTrajectorRelations = getGoldTrajectorPairs(populateNullPairs)
    val goldLandmarkRelations = getGoldLandmarkPairs(populateNullPairs)

    candidateRelations.foreach(_.setProperty("RelationType", "None"))

    setLmSpPairTypes(goldLandmarkRelations, candidateRelations)
    setTrSpPairTypes(goldTrajectorRelations, candidateRelations)

    ReportHelper.reportRelationStats(candidateRelations, goldTrajectorRelations, goldLandmarkRelations)
  }

  def setTrSpPairTypes(goldTrajectorRelations: List[Relation], candidateRelations: List[Relation]): Unit = {

    goldTrajectorRelations.foreach(r => {
      val c = candidateRelations
        .find(x =>
          x.getArgument(0).getPropertyValues(s"${trTag}_id").contains(r.getArgumentId(0)) &&
            x.getArgument(1).getPropertyValues(s"${spTag}_id").contains(r.getArgumentId(1)))

      if (c.nonEmpty) {
        if (c.get.getProperty("RelationType") == "TR-SP") {
          println(s"warning: candidate already marked as TR-SP via ${c.get.getId}. duplicate relation: ${r.getId}")
        } else {
          if (c.get.getProperty("RelationType") == "LM-SP") {
            println(s"warning: overriding LM-SP relation ${c.get.getId} by TR-SP relation: ${r.getId}")
          }
          c.get.setProperty("RelationType", "TR-SP")
          c.get.setProperty("ActualId", r.getId)
        }
      } else {
        println(s"cannot find TR-SP candidate relation for ${r.getId}")
      }
    })
  }

  def setLmSpPairTypes(goldLandmarkRelations: List[Relation], candidateRelations: List[Relation]): Unit = {

    goldLandmarkRelations.foreach(r => {
      val c = candidateRelations
        .find(x =>
          x.getArgument(0).getPropertyValues(s"${lmTag}_id").contains(r.getArgumentId(0)) &&
            x.getArgument(1).getPropertyValues(s"${spTag}_id").contains(r.getArgumentId(1)))

      if (c.nonEmpty) {
        if (c.get.getProperty("RelationType") == "LM-SP") {
          println(s"warning: candidate already marked as LM-SP via ${c.get.getId}. duplicate relation: ${r.getId}")
        } else {
          if (c.get.getProperty("RelationType") == "TR-SP") {
            println(s"warning: overriding TR-SP relation ${c.get.getId} by LM-SP relation: ${r.getId}")
          }
          c.get.setProperty("RelationType", "LM-SP")
          c.get.setProperty("ActualId", r.getId)
        }
      } else {
        println(s"cannot find LM-SP candidate relation for ${r.getId}")
      }
    })
  }

  def setTripletRelationTypes(triplets: List[Relation]): Unit = {

    val actualTriplets = getTripletsWithArguments()
    val missed = new ListBuffer[String]()
    actualTriplets.foreach(a => {
      val predicted = triplets.filter(x => isEqual(x, a))
      if (predicted.nonEmpty) {
        predicted.foreach(x => copyRelationProperties(a, x))
      }
      else {
        missed.add(a.getId)
      }
    })
    println(s"Warning: Missed relations ${missed.size}")
  }

  def getTripletsWithArguments(): List[Relation] = {

    val relations = reader.getRelations(relationTag, "trajector_id", "spatial_indicator_id", "landmark_id")

    reader.setPhraseTagName(trTag)
    val trajectors = reader.getPhrases().map(x => x.getId -> x).toMap

    reader.setPhraseTagName(lmTag)
    val landmarks = reader.getPhrases().map(x => x.getId -> x).toMap

    reader.setPhraseTagName(spTag)
    val indicators = reader.getPhrases().map(x => x.getId -> x).toMap

    relations.map(r => {

      if (!trajectors.containsKey(r.getArgumentId(0))) {
        println(s"Warning: cannot find trajector ${r.getArgumentId(0)} for relation ${r.getId}. Relation skipped.")
        null
      }
      else if (!indicators.containsKey(r.getArgumentId(1))) {
        println(s"Warning: cannot find indicator ${r.getArgumentId(2)} for relation ${r.getId}. Relation skipped.")
        null
      }
      else if (!landmarks.containsKey(r.getArgumentId(2))) {
        println(s"Warning: cannot find landmark ${r.getArgumentId(2)} for relation ${r.getId}. Relation skipped.")
        null
      }
      else {
        val tr = trajectors(r.getArgumentId(0))
        tr.setGlobalSpan(useGlobalSpan)
        val sp = indicators(r.getArgumentId(1))
        sp.setGlobalSpan(useGlobalSpan)
        val lm = landmarks(r.getArgumentId(2))
        lm.setGlobalSpan(useGlobalSpan)
        r.setArgument(0, tr)
        r.setArgument(1, sp)
        r.setArgument(2, lm)
        r
      }
    }).filter(x => x != null).toList
  }

  def getTrSpPairsWithArguments(): List[Relation] = {

    val relations = reader.getRelations(relationTag, "trajector_id", "spatial_indicator_id")

    reader.setPhraseTagName(trTag)
    val trajectors = reader.getPhrases().map(x => x.getId -> x).toMap

    reader.setPhraseTagName(spTag)
    val indicators = reader.getPhrases().map(x => x.getId -> x).toMap

    relations.map(r => {
      val tr = trajectors(r.getArgumentId(0))
      val sp = indicators(r.getArgumentId(1))
      r.setArgument(0, tr)
      r.setArgument(1, sp)
      r
    }).groupBy(x => x.getArgumentIds.mkString(",")).map(_._2.head).toList // remove duplicates
  }

  def getLmSpPairsWithArguments(): List[Relation] = {

    val relations = reader.getRelations(relationTag, "landmark_id", "spatial_indicator_id")

    reader.setPhraseTagName(lmTag)
    val landmarks = reader.getPhrases().map(x => x.getId -> x).toMap

    reader.setPhraseTagName(spTag)
    val indicators = reader.getPhrases().map(x => x.getId -> x).toMap

    relations.map(r => {
      val lm = landmarks(r.getArgumentId(0))
      val sp = indicators(r.getArgumentId(1))
      r.setArgument(0, lm)
      r.setArgument(1, sp)
      r
    }).groupBy(x => x.getArgumentIds.mkString(",")).map(_._2.head).toList // remove duplicates
  }

  def setRoles(instances: List[NlpBaseElement]): Unit = {
    if (instances.isEmpty)
      return

    reader.addPropertiesFromTag(trTag, instances, XmlMatchings.elementContainsXmlHeadwordMatching)
    reader.addPropertiesFromTag(lmTag, instances, XmlMatchings.elementContainsXmlHeadwordMatching)
    reader.addPropertiesFromTag(spTag, instances, XmlMatchings.elementContainsXmlPrepositionMatching)
  }

  def getSentences: List[Sentence] = {
    reader.getSentences().toList
  }

  def getDocuments: List[Document] = {
    reader.getDocuments().toList
  }

  private def copyRelationProperties(from: Relation, to: Relation) = {
    to.setProperty("ActualId", from.getId)
    to.setProperty("GeneralType", from.getProperty("general_type"))
    to.setProperty("SpecificType", from.getProperty("specific_type"))
    to.setProperty("RCC8", from.getProperty("RCC8_value"))
    to.setProperty("FoR", from.getProperty("FoR"))
    to.setProperty("Relation", "true")
  }

  private def isEqual(r1: Relation, r2: Relation): Boolean = {
    new OverlapComparer().isEqual(ReportHelper.getRelationEval(r1), ReportHelper.getRelationEval(r2))
  }

  private def getGoldLandmarkPairs(populateNullPairs: Boolean): List[Relation] = {

    // create pairs which first argument is landmark and second is indicator, and remove duplicates
    val nullLandmarkIds = getTags(lmTag).filter(_.getStart == -1).map(_.getId)
    val relations = getRelations("landmark_id", "spatial_indicator_id")
      .groupBy(x => x.getArgumentId(0) + "_" + x.getArgumentId(1))
      .map { case (_, list) => list.head }
      .toList
    if (populateNullPairs) {
      relations.foreach(r => if (nullLandmarkIds.contains(r.getArgumentId(0))) r.setArgumentId(0, dummyPhrase.getId))
      relations
    } else {
      relations.filterNot(r => nullLandmarkIds.contains(r.getArgumentId(0)))
    }
  }

  private def getGoldTrajectorPairs(populateNullPairs: Boolean): List[Relation] = {

    // create pairs which first argument is trajector and second is indicator, and remove duplicates
    val nullTrajectorIds = getTags(trTag).filter(_.getStart == -1).map(_.getId)
    val relations = getRelations("trajector_id", "spatial_indicator_id")
      .groupBy(x => x.getArgumentId(0) + "_" + x.getArgumentId(1))
      .map { case (_, list) => list.head }
      .toList
    if (populateNullPairs) {
      relations.foreach(r => if (nullTrajectorIds.contains(r.getArgumentId(0))) r.setArgumentId(0, dummyPhrase.getId))
      relations
    } else {
      relations.filterNot(r => nullTrajectorIds.contains(r.getArgumentId(0)))
    }
  }

  private def getRelations(firstArgId: String, secondArgId: String): List[Relation] = {
    reader.getRelations(relationTag, firstArgId, secondArgId).toList
  }

  private def getTags(tag: String): List[NlpBaseElement] = {
    reader.getTagAsNlpBaseElement(tag).toList
  }

  private def createXmlReader(): NlpXmlReader = {
    val reader = new NlpXmlReader(dataPath, "SCENE", "SENTENCE", null, null)
    reader.setIdUsingAnotherProperty("SCENE", "DOCNO")
    reader.setUseGlobalSpans(useGlobalSpan)
    reader
  }

}
