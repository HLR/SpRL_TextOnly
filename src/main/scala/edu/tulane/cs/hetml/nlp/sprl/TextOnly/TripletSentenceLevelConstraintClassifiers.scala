package edu.tulane.cs.hetml.nlp.sprl.TextOnly

import edu.illinois.cs.cogcomp.infer.ilp.GurobiHook
import edu.illinois.cs.cogcomp.saul.classifier.ConstrainedClassifier
import edu.tulane.cs.hetml.nlp.BaseTypes.{Phrase, Relation, Sentence}
import MultiModalSpRLDataModel._
import edu.tulane.cs.hetml.nlp.sprl.TextOnly.MultiModalSpRLTripletClassifiers._
import edu.tulane.cs.hetml.nlp.sprl.TextOnly.TripletSentenceLevelConstraints._
import edu.tulane.cs.hetml.vision.{ImageTriplet, WordSegment}

object TripletSentenceLevelConstraintClassifiers {

  val erSolver = new GurobiHook()

  object LMConstraintClassifier extends ConstrainedClassifier[Phrase, Sentence](LandmarkRoleClassifier) {
    def subjectTo = tripletConstraints

    override val solver = erSolver
    override val pathToHead = Some(-sentenceToPhrase)
  }

  object TRConstraintClassifier extends ConstrainedClassifier[Phrase, Sentence](TrajectorRoleClassifier) {
    def subjectTo = tripletConstraints

    override val solver = erSolver
    override val pathToHead = Some(-sentenceToPhrase)
  }

  object IndicatorConstraintClassifier extends ConstrainedClassifier[Phrase, Sentence](IndicatorRoleClassifier) {
    def subjectTo = tripletConstraints

    override val solver = erSolver
    override val pathToHead = Some(-sentenceToPhrase)
  }

  object TripletRelationConstraintClassifier extends ConstrainedClassifier[Relation, Sentence](TripletRelationClassifier) {
    def subjectTo = tripletConstraints

    override val solver = erSolver
    override val pathToHead = Some(-sentenceToTriplets)
  }

  object TripletGeneralTypeConstraintClassifier extends ConstrainedClassifier[Relation, Sentence](TripletGeneralTypeClassifier) {
    def subjectTo = tripletConstraints

    override val solver = erSolver
    override val pathToHead = Some(-sentenceToTriplets)
  }

  object TripletDirectionConstraintClassifier extends ConstrainedClassifier[Relation, Sentence](TripletDirectionClassifier) {
    def subjectTo = tripletConstraints

    override val solver = erSolver
    override val pathToHead = Some(-sentenceToTriplets)
  }

  object TripletRegionConstraintClassifier extends ConstrainedClassifier[Relation, Sentence](TripletRegionClassifier) {
    def subjectTo = tripletConstraints

    override val solver = erSolver
    override val pathToHead = Some(-sentenceToTriplets)
  }

//  object PrepositionConstraintClassifier extends ConstrainedClassifier[ImageTriplet, Sentence](PrepositionClassifier) {
//    def subjectTo = tripletConstraints
//
//    override val solver = erSolver
//    override val pathToHead = Some(-sentenceToVisualTriplet)
//  }
//  class ConstrainedSingleWordAsClassifier(w: String)
//    extends ConstrainedClassifier[WordSegment, Sentence](alignmentHelper.trainedWordClassifier(w)){
//
//    def subjectTo = tripletConstraints
//
//    override val solver = erSolver
//    override val pathToHead = Some(-sentenceToWordSegments)
//  }
}
