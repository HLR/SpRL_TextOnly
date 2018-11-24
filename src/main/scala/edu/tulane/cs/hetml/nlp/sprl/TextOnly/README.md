# Combining text and image using word-as-classifier and proposition classifier
In this project we include the implementation codes of [1] paper.

## Dataset
Please download the dataset from here (??) and place it in the root directory of the application. The layout of the data should be as follows:
```bash
Data/
|-- saiapr_tc-12
|   |-- TrainWords
|   |   |-- TrainedWords.txt  
|   |   |-- MissedWords.txt
|   |-- VGData
|   |   |-- image_data.json  
|   |   |-- relationships.json
|   |-- SegmentCNNFeatures
|   |   |-- ImageSegmentsFeaturesNewTrain.txt  
|   |   |-- ImageSegmentsFeaturesNewTest.txt
|   |-- PhraseSegmentPairs
|   |   |-- SegmentsPhraseText_train_head.txt  
|   |   |-- SegmentsPhraseText_test_head.txt
|   |-- Alignments
|   |   |-- phrases_train.txt  
|   |   |-- phrases_test.txt
|   `-- SegmentBoxes
|       |-- segmentBoxes.txt
|-- newSprl2017_train.xml
|-- newSprl2017_gold.xml
|-- training.mat
|-- testing.mat
|-- validation.mat
`-- ReferGames.txt
```
 
## Preprocessing Step:
- `Train - Word-as-classifiers`
Before, running the main application, you need to train the word-as-classifiers. It is a separate application [`WordasClassifier`](WordasClassifierApp.scala) and has its own configuration file
[`WordasClassifierConfigurator`](WordasClassifierConfigurator.scala). To train the classifiers, set `isTrain` and `preprocessReferitExp` to `true` and run the application using the following command.

```
sbt "project saulExamples" "runMain edu.tulane.cs.hetml.nlp.sprl.WordasClassifier.WordasClassifierApp"
``` 
## MultiModelApp
All configurations needed to run the main application are placed in 
[`tripletConfigurator`](tripletConfigurator.scala). In order to run the application for training, set `IsTraining` to `Configurator.TRUE` and for testing set it to `Configurator.FALSE`. 


## Running

To run the main app with default properties:

```
sbt "project saulExamples" "runMain edu.tulane.cs.hetml.nlp.sprl.Triplets.MultiModalTripletApp"
```

results will be saved in `data/mSprL/results` folder as text files corresponding to the model selected in the config file. 

## Results on CLEF 2017 dataset
Here are the summarized results of relation classifier for different models
```
label                           Precision  Recall     F1         LCount     PCount    
-----------------------------------------------------------------------------------
BM                              65.640     60.226     62.817     885        812
BM+C                            70.036     66.554     68.250     885        841       
BM+C+I_gold_align               66.367     75.141     70.482     885        1002      
BM+C+I_gold_align_prep          67.140     74.802     70.764     885        986
BM+C+I_classifier_align         71.394     66.554     68.889     885        825
BM+C+I_classifier_align_perp    71.691     66.102     68.783     885        816
```

### Setting models
You can use the `Triplets/tripletConfigurator.scala` to set the configuration of the train and test models. 

 [BM] is the baseline model and is set using `val model = FeatureSets.BaseLine` and disable using images and preposition classifiers
 `val usePrepositions = false` and `var populateImages = false`

\TODO: explain the parameters in the configuration file here or in the code itself.

## References
[1] Taher Rahgooy, Umar Manzoor, and Parisa Kordjamshidi. "??". In preparation