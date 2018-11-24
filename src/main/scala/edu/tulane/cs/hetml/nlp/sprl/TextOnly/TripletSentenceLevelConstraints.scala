package edu.tulane.cs.hetml.nlp.sprl.TextOnly

import edu.illinois.cs.cogcomp.lbjava.infer.{FirstOrderConstant, FirstOrderConstraint}
import edu.illinois.cs.cogcomp.saul.classifier.ConstrainedClassifier
import edu.illinois.cs.cogcomp.saul.constraint.ConstraintTypeConversion._
import edu.tulane.cs.hetml.nlp.BaseTypes._
import MultiModalSpRLDataModel._
import edu.tulane.cs.hetml.nlp.sprl.TextOnly.MultiModalSpRLTripletClassifiers._
import edu.tulane.cs.hetml.vision.{ImageTriplet, WordSegment}

import scala.collection.JavaConversions._

object TripletSentenceLevelConstraints {

  val roleShouldHaveRel = ConstrainedClassifier.constraint[Sentence] {
    var a: FirstOrderConstraint = null
    s: Sentence =>
      a = new FirstOrderConstant(true)
      val sentTriplets = (sentences(s) ~> sentenceToTriplets).toList
      sentTriplets.foreach {
        x =>
          val trRel = sentTriplets.filter(r => r.getArgumentId(0) == x.getArgumentId(0))
            ._exists(x => (TripletRelationClassifier on x) is "true")

          val lmRel = sentTriplets.filter(r => r.getArgumentId(2) == x.getArgumentId(2))
            ._exists(x => (TripletRelationClassifier on x) is "true")

          val tr = TrajectorRoleClassifier on (triplets(x) ~> tripletToTr).head is "true"
          val lm = LandmarkRoleClassifier on (triplets(x) ~> tripletToLm).head is "true"

          a = a and (tr ==> trRel) and (lm ==> lmRel)
      }
      a
  }

  val roleIntegrity = ConstrainedClassifier.constraint[Sentence] {
    var a: FirstOrderConstraint = null
    s: Sentence =>
      a = new FirstOrderConstant(true)
      (sentences(s) ~> sentenceToTriplets).foreach {
        x =>
          a = a and (
            (
              (TripletRelationClassifier on x) is "true") ==>
              (
                (TrajectorRoleClassifier on (triplets(x) ~> tripletToTr).head is "true") and
                  (IndicatorRoleClassifier on (triplets(x) ~> tripletToSp).head is "true") and
                  (LandmarkRoleClassifier on (triplets(x) ~> tripletToLm).head is "true")
                )
            )
      }
      a
  }

  val relationsShouldNotHaveCommonRoles = ConstrainedClassifier.constraint[Sentence] {
    var a: FirstOrderConstraint = null
    s: Sentence =>
      a = new FirstOrderConstant(true)
      val sentTriplets = (sentences(s) ~> sentenceToTriplets).toList
      sentTriplets.foreach {
        r =>
          val tr = (triplets(r) ~> tripletToTr).head
          val sp = (triplets(r) ~> tripletToSp).head
          val lm = (triplets(r) ~> tripletToLm).head
          var othersFalse: FirstOrderConstraint = new FirstOrderConstant(true)
          val others = sentTriplets.filter(x => x != r && (triplets(x) ~> tripletToSp).head == sp)
          if (others.nonEmpty) {
            others.foreach(x => othersFalse = othersFalse and ((TripletRelationClassifier on x) is "false"))
            a = a and (((TripletRelationClassifier on r) is "true") ==> othersFalse)
          }
      }
      a
  }

  val boostTrajector = ConstrainedClassifier.constraint[Sentence] {
    s: Sentence =>
      (
        (sentences(s) ~> sentenceToPhrase).toList._exists { x: Phrase => IndicatorRoleClassifier on x is "true" }
          ==>
          (sentences(s) ~> sentenceToPhrase).toList._exists { x: Phrase => TrajectorRoleClassifier on x is "true" }
        )
  }

  val boostLandmark = ConstrainedClassifier.constraint[Sentence] {
    s: Sentence =>
      (
        (sentences(s) ~> sentenceToPhrase).toList._exists { x: Phrase => IndicatorRoleClassifier on x is "true" }
          ==>
          (sentences(s) ~> sentenceToPhrase).toList._exists { x: Phrase => LandmarkRoleClassifier on x is "true" }
        )
  }

  val boostTripletByGeneralType = ConstrainedClassifier.constraint[Sentence] {
    var a: FirstOrderConstraint = null
    s: Sentence =>
      a = new FirstOrderConstant(true)
      (sentences(s) ~> sentenceToTriplets).foreach {
        x =>
          a = a and (
            (
              (TripletGeneralTypeClassifier on x) isNot "None"
              ) <==>
              (TripletRelationClassifier on x is "true")
            )
      }
      a
  }


  val boostGeneralByDirectionMulti = ConstrainedClassifier.constraint[Sentence] {
    var a: FirstOrderConstraint = null
    s: Sentence =>
      a = new FirstOrderConstant(true)
      (sentences(s) ~> sentenceToTriplets).foreach {
        x =>
          a = a and
            ((TripletDirectionClassifier on x isNot "None") <==> (TripletGeneralTypeClassifier on x is "direction"))
      }
      a
  }


  val boostGeneralByRegionMulti = ConstrainedClassifier.constraint[Sentence] {
    var a: FirstOrderConstraint = null
    s: Sentence =>
      a = new FirstOrderConstant(true)
      (sentences(s) ~> sentenceToTriplets).foreach {
        x =>
          a = a and
            ((TripletRegionClassifier on x isNot "None") <==> (TripletGeneralTypeClassifier on x is "region"))
      }
      a
  }

  val regionShouldHaveLandmark = ConstrainedClassifier.constraint[Sentence] {
    var a: FirstOrderConstraint = null
    s: Sentence =>
      a = new FirstOrderConstant(true)
      (sentences(s) ~> sentenceToTriplets).foreach {
        x =>
          val lmIsNull = (triplets(x) ~> tripletToLm).head == dummyPhrase
          if (lmIsNull) {
            a = a and (TripletRegionClassifier on x is "None")
          }
      }
      a
  }

//  lazy val sentWordSegs = wordSegments().groupBy(_.getPhrase.getSentence.getId)
//  val alignmentConsistency = ConstrainedClassifier.constraint[Sentence] {
//    var a: FirstOrderConstraint = null
//    s: Sentence =>
//      a = new FirstOrderConstant(true)
//      if(sentWordSegs.contains(s.getId)) {
//        val pairs = sentWordSegs(s.getId)
//        //wordSegments().filter(x => x.getPhrase.getSentence.getId == s.getId).toList
//        val perSeg = pairs.groupBy(_.getSegment)
//        val perPhrase = pairs.groupBy(_.getPhrase)
//
//        perPhrase.foreach {
//          z =>
//            val w = z._2.head.getWord
//            val c = wordAsClassifierHelper.trainedWordClassifier(w)
//            z._2.foreach {
//              x =>
//                var b: FirstOrderConstraint = new FirstOrderConstant(true)
//                z._2.filter(y => y != x).foreach {
//                  y =>
//                    b = b and (c on y is "false")
//                }
//                a = a and ((c on x is "true") ==> b)
//            }
//        }
//        perSeg.foreach {
//          z =>
//            z._2.foreach {
//              x =>
//                val w = x.getWord
//                val c = wordAsClassifierHelper.trainedWordClassifier(w)
//                var b: FirstOrderConstraint = new FirstOrderConstant(true)
//                z._2.filter(y => y != x).foreach {
//                  y =>
//                    b = b and (c on y is "false")
//                }
//                a = a and ((c on x is "true") ==> b)
//            }
//        }
//      }
//      a
//  }

//  val approveRelationByImage2 = ConstrainedClassifier.constraint[Sentence] {
//    var a: FirstOrderConstraint = null
//    s: Sentence =>
//      a = new FirstOrderConstant(true)
//      if(sentWordSegs.contains(s.getId)) {
//        val candidateAlignments = sentWordSegs(s.getId)
//        //wordSegments().filter(x => x.getPhrase.getSentence.getId == s.getId).toList
//        val candidatePhraseIds = candidateAlignments.map(_.getPhrase.getId)
//        val img = (sentences(s) ~> -documentToSentence ~> documentToImage).head
//        (sentences(s) ~> sentenceToTriplets)
//          .filter(x => candidatePhraseIds.contains(x.getArgumentId(0)) && candidatePhraseIds.contains(x.getArgumentId(2)))
//          .foreach {
//            r =>
//              val tr = headWordLemma((triplets(r) ~> tripletToTr).head)
//              val lm = headWordLemma((triplets(r) ~> tripletToLm).head)
//              val sp = (triplets(r) ~> tripletToSp).head.getText.toLowerCase.replace(" ", "_")
//              val trPairs = candidateAlignments.filter(y => y.getPhrase.getId == r.getArgumentId(0))
//              val lmPairs = candidateAlignments.filter(y => y.getPhrase.getId == r.getArgumentId(2))
//              trPairs.foreach {
//                x =>
//                  val trSeg = x.getSegment
//                  val t = new WordSegment(tr, trSeg, false)
//                  val trAlignment = wordAsClassifierHelper.trainedWordClassifier(x.getWord)
//                  lmPairs.foreach {
//                    y =>
//                      val lmAlignment = wordAsClassifierHelper.trainedWordClassifier(y.getWord)
//                      val lmSeg = y.getSegment
//                      val vt = new ImageTriplet(sp, tr, lm, trSeg.getBoxDimensions, lmSeg.getBoxDimensions,
//                        img.getWidth, img.getHeight)
//                      val l = new WordSegment(lm, lmSeg, false)
//                      a = a and
//                        (((imageSupportsSp2 on vt is "true")
//                          and (trAlignment on t is "true")
//                          and (lmAlignment on l is "true")) ==>
//                          (TripletRelationClassifier on r is "true"))
//                  }
//              }
//          }
//      }
//      a
//  }

//  val approveRelationByImage = ConstrainedClassifier.constraint[Sentence] {
//    var a: FirstOrderConstraint = null
//    s: Sentence =>
//      a = new FirstOrderConstant(true)
//      (sentences(s) ~> sentenceToTriplets).foreach {
//        x =>
//          a = a and ((imageSupportsSp on x is "true") ==>
//            (TripletRelationClassifier on x is "true"))
//      }
//      a
//  }
//
//  val discardRelationByImage = ConstrainedClassifier.constraint[Sentence] {
//    var a: FirstOrderConstraint = null
//    s: Sentence =>
//      a = new FirstOrderConstant(true)
//      (sentences(s) ~> sentenceToTriplets).foreach {
//        x =>
//          a = a and ((imageSupportsSp on x is "false") ==>
//            (TripletRelationClassifier on x is "false"))
//      }
//      a
//  }

  val tripletConstraints = ConstrainedClassifier.constraint[Sentence] {

    x: Sentence =>
      var a: FirstOrderConstraint = null
      a =
        roleShouldHaveRel(x) and
          boostTrajector(x) and
          boostLandmark(x) and
          boostTripletByGeneralType(x) and
          boostGeneralByDirectionMulti(x) and
          boostGeneralByRegionMulti(x)

//      if (tripletConfigurator.usePrepositions) {
//        if(tripletConfigurator.alignmentMethod == "topN"){
//          a = a and
//            alignmentConsistency(x) and
//            approveRelationByImage2(x)
//        }
//        else{
//          a = a and
//            discardRelationByImage(x) //and
//            //approveRelationByImage(x)
//        }
//      }
      a
  }

}
