package edu.tulane.cs.hetml.nlp.sprl.TextOnly

import edu.illinois.cs.cogcomp.saul.util.Logging
import edu.tulane.cs.hetml.nlp.BaseTypes._
import edu.tulane.cs.hetml.nlp.LanguageBaseTypeSensors.documentToSentenceGenerating
import edu.tulane.cs.hetml.nlp.Xml.NlpXmlReader
import edu.tulane.cs.hetml.nlp.XmlMatchings
import edu.tulane.cs.hetml.nlp.sprl.Helpers._
import edu.tulane.cs.hetml.nlp.sprl.TextOnly.MultiModalSpRLDataModel._
import edu.tulane.cs.hetml.nlp.sprl.TextOnly.tripletConfigurator.{isTrain, _}

import scala.collection.JavaConversions._
object MultiModalPopulateData extends Logging {

  LexiconHelper.path = spatialIndicatorLex
  lazy val xmlTestReader = new SpRLXmlReader(testFile, globalSpans)
  lazy val xmlTrainReader = new SpRLXmlReader(trainFile, globalSpans)

 // lazy val tripsTestReader = new NlpXmlReader("data/mSpRL/trips_cleaned_parses/manuallyCollectedTripsParse_test.xml", "SCENE", "SENTENCE", "PHRASE", null)
  // lazy val tripsTrainReader = new NlpXmlReader("data/mSpRL/trips_cleaned_parses/manuallyCollectedTripsParse_train.xml", "SCENE", "SENTENCE", "PHRASE", null)
  def xmlReader = if (isTrain) xmlTrainReader else xmlTestReader
//  def readerTrips = if (isTrain) tripsTrainReader else tripsTestReader

// lazy val tripsRelationList = readerTrips.getRelations("RELATION","head", "res")
// lazy val tripsPhraseList = readerTrips.getPhrases()



  def populateRoleDataFromAnnotatedCorpus(populateNullPairs: Boolean = true): Unit = {
    logger.info("Role population started ...")
    if (isTrain && onTheFlyLexicon) {
      LexiconHelper.createSpatialIndicatorLexicon(xmlReader)
    }


    documents.populate(xmlReader.getDocuments, isTrain)
    sentences.populate(xmlReader.getSentences, isTrain)

    if (populateNullPairs) {
      phrases.populate(List(dummyPhrase), isTrain)
    }

    val phraseInstances = (if (isTrain) phrases.getTrainingInstances.toList else phrases.getTestingInstances.toList)
      .filter(_.getId != dummyPhrase.getId)

    if (globalSpans) {
      phraseInstances.foreach {
        p =>
          p.setStart(p.getSentence.getStart + p.getStart)
          p.setEnd(p.getSentence.getStart + p.getEnd)
          p.setGlobalSpan(globalSpans)
      }
    }

    xmlReader.setRoles(phraseInstances)
   // readerTrips.addPropertiesFromTag("PHRASE", phraseInstances, XmlMatchings.elementContainsXmlHeadwordMatching)
    logger.info("Role population finished.")
  }

  def populateTripletDataFromAnnotatedCorpus(
                                              trFilter: (Phrase) => Boolean,
                                              spFilter: (Phrase) => Boolean,
                                              lmFilter: (Phrase) => Boolean
                                            ): Unit = {

    logger.info("Triplet population started ...")
    val candidateRelations = TripletCandidateGenerator.generateAllTripletCandidates(
      trFilter,
      spFilter,
      lmFilter,
      isTrain
    )
    xmlReader.setTripletRelationTypes(candidateRelations)

//    val res = readerTrips.getPhrases().map(x => x.getId -> x).toMap
//    val heads = readerTrips.getPhrases().map(x => x.getId -> x).toMap
//
//    val rel= tripsRelationList.map(r => {
//
//      if (!res.containsKey(r.getArgumentId(0))) {
//        println(s"Warning: cannot find trajector ${r.getArgumentId(0)} for relation ${r.getId}. Relation skipped.")
//        null
//      }
//     else if (!heads.containsKey(r.getArgumentId(1))) {
//        println(s"Warning: cannot find landmark ${r.getArgumentId(1)} for relation ${r.getId}. Relation skipped.")
//        null
//      }
//      else {
//        val tr = res(r.getArgumentId(0))
//        tr.setGlobalSpan(false)
//        val lm = heads(r.getArgumentId(1))
//        lm.setGlobalSpan(false)
//        r.setArgument(0, tr)
//        r.setArgument(1, lm)
//        r
//      }
//    }).filter(x => x != null).toList


//rel is the trips binary relation (r) and candidateRelations are SpRL triplet candidates (x)

//    rel.foreach(r => {
//
//       candidateRelations.foreach(x =>{
//        val temp=  tripsOverlap(r,x)
//        if (temp._2)
//          x.setProperty("tripsRelationLabel", r.getProperty("label"))})})

    triplets.populate(candidateRelations, isTrain)

    logger.info("Triplet population finished.")
  }

  def populateDataFromPlainTextDocuments(documentList: List[Document],
                                         indicatorClassifier: Phrase => Boolean,
                                         populateNullPairs: Boolean = true
                                        ): Unit = {

    logger.info("Data population started ...")
    val isTrain = false

    documents.populate(documentList, isTrain)
    sentences.populate(documentList.flatMap(d => documentToSentenceGenerating(d)), isTrain)
    if (populateNullPairs) {
      phrases.populate(List(dummyPhrase), isTrain)
    }
    val spCandidatesTrain = TripletCandidateGenerator.getIndicatorCandidates(phrases().toList)
    val trCandidatesTrain = TripletCandidateGenerator.getTrajectorCandidates(phrases().toList)
      .filterNot(x => spCandidatesTrain.contains(x))
    val lmCandidatesTrain = TripletCandidateGenerator.getLandmarkCandidates(phrases().toList)
      .filterNot(x => spCandidatesTrain.contains(x))


    logger.info("Triplet population started ...")
    val candidateRelations = TripletCandidateGenerator.generateAllTripletCandidates(
      x => trCandidatesTrain.exists(_.getId == x.getId),
      x => indicatorClassifier(x),
      x => lmCandidatesTrain.exists(_.getId == x.getId),
      isTrain
    )

    triplets.populate(candidateRelations, isTrain)


    logger.info("Data population finished.")
  }

def tripsOverlap(trips: Relation, sprl: Relation): (String,Boolean)=  {
  // This function finds the overlap between the sprl's triplet candidates and the trips binary relations.
  // The overlap should exist between both trips arguments and two of the roles in sprl triplets. We record
  // that the trips relation holds between which argument positions (tr/lm/sp) and add later this will be added
  // to the properties of the triplets.

  var p =List("","")
  for (i <- 0 to (trips.getArgumentsCount-1)) {
   for (j <- 0 to sprl.getArgumentsCount - 1) {
     if (((trips.getArgument(i).getGlobalStart <= sprl.getArgument(j).getGlobalStart) &&
       (sprl.getArgument(j).getGlobalStart <= trips.getArgument(i).getGlobalEnd)) ||
       ((sprl.getArgument(j).getGlobalStart <= trips.getArgument(i).getGlobalStart) &&
         (trips.getArgument(i).getGlobalStart <= sprl.getArgument(j).getGlobalEnd))) {
       if (j==0) p=p.updated(i,"tr")// p(i)= "tr"
       if (j==1) p=p.updated(i,"sp")
       if (j==2) p=p.updated(i,"lm")
     }
   }
 }
if (!p(0).isEmpty && !p(1).isEmpty)
  return (p(0)+p(1), true)
  else
  return ("",false)
 }
}



//    if (populateImages) {
//      alignmentReader.setAlignments(phraseInstances)
//      images.populate(imageReader.getImageList, isTrain)
//      val segs = getAdjustedSegments(imageReader.getSegmentList)
//      segments.populate(segs, isTrain)
//      imageSegmentsDic = getImageSegmentsDic()
//      if (alignmentMethod != "topN") {
//        setBestAlignment()
//      }
//      else {
//        val ws = segmentPhrasePairs().map {
//          pair =>
//            val s = (segmentPhrasePairs(pair) ~> -segmentToSegmentPhrasePair).head
//            val p = (segmentPhrasePairs(pair) ~> segmentPhrasePairToPhrase).head
//            val segs = (segments(s) ~> -imageToSegment ~> imageToSegment).toList
//            val lemma = headWordLemma(p)
//            val wordSegs = segs.map(x => new WordSegment(lemma, x, false))
//            val topIds = alignmentHelper.predictTopSegmentIds(wordSegs, tripletConfigurator.topAlignmentCount)
//            if (topIds.contains(s.getSegmentId)) {
//              val wordSegment = new WordSegment(lemma, s, false)
//              wordSegment.setPhrase(p)
//              wordSegment
//            }
//            else
//              null
//        }.filter(x => x != null)
//        wordSegments.populate(ws)
//      }
//
//    }

//  def populateVisualTripletsFromExternalData(): Unit = {
//    val flickerTripletReader = new ImageTripletReader("data/mSprl/saiapr_tc-12/imageTriplets", "Flickr30k.majorityhead")
//    val msCocoTripletReader = new ImageTripletReader("data/mSprl/saiapr_tc-12/imageTriplets", "MSCOCO.originalterm")
//
//    val externalTrainTriplets = flickerTripletReader.trainImageTriplets ++ msCocoTripletReader.trainImageTriplets
//
//    if (trainPrepositionClassifier && isTrain) {
//      println("Populating Visual Triplets from External Dataset...")
//      visualTriplets.populate(externalTrainTriplets, isTrain)
//    }
//  }

//  def getAdjustedSegments(segments: List[Segment]): List[Segment] = {
//
//    val alignedPhrases = phrases().filter(_.containsProperty("goldAlignment"))
//    val update = alignedPhrases
//      .filter(p => segments.exists(s => s.getAssociatedImageID == p.getPropertyFirstValue("imageId") &&
//        p.getPropertyValues("segId").exists(_.toInt == s.getSegmentId)))
//
//    update.foreach {
//      p =>
//        segments.filter(x =>
//          x.getAssociatedImageID == p.getPropertyFirstValue("imageId") &&
//            p.getPropertyValues("segId").exists(_.toInt == x.getSegmentId)
//        ).foreach {
//          seg =>
//            val im = images().find(_.getId == seg.getAssociatedImageID).get
//            val x = Math.min(im.getWidth, Math.max(0, p.getPropertyFirstValue("segX").toDouble))
//            val y = Math.min(im.getHeight, Math.max(0, p.getPropertyFirstValue("segY").toDouble))
//            val w = Math.min(im.getWidth - x, p.getPropertyFirstValue("segWidth").toDouble)
//            val h = Math.min(im.getHeight - y, p.getPropertyFirstValue("segHeight").toDouble)
//            if (seg.getBoxDimensions == null)
//              seg.setBoxDimensions(new Rectangle2D.Double(x, y, w, h))
//            else {
//              seg.getBoxDimensions.setRect(x, y, w, h)
//            }
//        }
//    }
//
//    segments
//  }

//  private def setBestAlignment() = {
//    sentences().foreach(s => {
//      val phraseSegments = (sentences(s) ~> sentenceToPhrase)
//        .toList.flatMap(p => (phrases(p) ~> -segmentPhrasePairToPhrase).toList)
//        .sortBy(x => x.getProperty("similarity").toDouble).reverse
//      val usedSegments = ListBuffer[String]()
//      val usedPhrases = ListBuffer[String]()
//      phraseSegments.foreach(pair => {
//        if (!usedPhrases.contains(pair.getArgumentId(0)) && !usedSegments.contains(pair.getArgumentId(1))) {
//          usedPhrases.add(pair.getArgumentId(0))
//          usedSegments.add(pair.getArgumentId(1))
//          val p = (segmentPhrasePairs(pair) ~> segmentPhrasePairToPhrase).head
//          if (pair.getProperty("similarity").toDouble > 0.30 || alignmentMethod == "classifier") {
//            p.addPropertyValue("bestAlignment", pair.getArgumentId(1))
//            p.addPropertyValue("bestAlignmentScore", pair.getProperty("similarity"))
//          }
//        }
//      }
//      )
//    })
//  }

//val missed = new ListBuffer[String]()
//    rel.foreach(a => {
//
//val predicted = candidateRelations.filter(x => new OverlapComparer().isEqual(ReportHelper.getRelationEval(x), ReportHelper.getRelationEval(a)))
//      //if a.arg1 equals x.arg1 => p1="tr"  if equals arg2=> p1="sp" if equals arg3 p1="lm"
//      //if a.arg1 equals  x.arg1 => p2="tr"  if equals arg2=> p2="sp" if equals arg3 p2="lm"
//
//    val tr = r.getArgument(0)
//    val sp = r.getArgument(1)
//    val lm = r.getArgument(2)
//    new RelationEval(tr.getGlobalStart, tr.getGlobalEnd, sp.getGlobalStart, sp.getGlobalEnd,
//      lm.getGlobalStart, lm.getGlobalEnd)
//

//      if (predicted.nonEmpty) {
//        predicted.foreach{x =>
//
//          x.setProperty("ActualId",a.getId)
//          x.setProperty(p1+"_"+p2, a.getProperty("label"))
////        to.setProperty("SpecificType", from.getProperty("specific_type"))
////        to.setProperty("RCC8", from.getProperty("RCC8_value"))
////        to.setProperty("FoR", from.getProperty("FoR"))
////        to.setProperty("Relation", "true")
////          copyRelationProperties(a, x)
//
//
//      }
//      }
//      else {
//        missed.add(a.getId)
//      }
//    })
