
This is a text-based baseline model for spatial role labeling task.

## Prerequisites:

 * [Scala](http://www.scala-lang.org/) 2.11.7
 * [Simple Build Tool(sbt)](http://www.scala-sbt.org/)

1) HetSaul Library
2) Gurobi optimization tool

--This project uses `HetSaul` library as a dependency. Before running project you need to clone to  `https://github.com/HLR/HetSaul.git`
and assuming you have sbt, do `sbt publish-local`, this should be all you need to do to be able to use the HetSaul project as a dependency. 

--To download and install Gurobi visit [Gurobi Website](http://www.gurobi.com/)
  Make sure to include Gurobi in your PATH and LD_LIBRARY variables
  
  ```bash
  export GUROBI_HOME="PATH-TO-GUROBI/linux64"
  export PATH="${PATH}:${GUROBI_HOME}/bin"
  export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${GUROBI_HOME}/lib"
  ```

## Dataset
The dataset is rather small and is copied at `data/mSpRL/saiapr_tc-12`, make sure to download google vectors and have it in `/data/GoogleNews-vectors-negative300.bin`.
 if you want to use `FeatureSets.WordEmbedding` in the configuration. 
 
 The layout of the data should be as follows:
```bash
Data/
|-- mSpRL      
|   |-- saiapr_tc-12
|   |  |   
|   |  |-- newSprl2017_train.xml
`------`-- newSprl2017_gold.xml

```

``` 
## SpRLApp

All configurations needed to run the main application are placed in 
[`tripletConfigurator`](tripletConfigurator.scala). In order to run the application for training, set `IsTraining` to `true` and for testing set it to `false`. 

### Setting models
 [BM] is the baseline model and is set using `val model = FeatureSets.BaseLine` and disable using images and preposition classifiers
 `val usePrepositions = false` and `var populateImages = false`. 
 You need to set the `isTrain= True` and run the SpRLApp to train models and when training finished change to `isTrain= False` 
 and rerun the SpRLApp. You can see the results on screen as well as in `SpRL_TextOnly/data/mSpRL/results/triplet_BM.txt` file.

## Running

To run the main app with default properties:

```sbt "project SpRL_textOnly" "runMain edu.tulane.cs.hetml.nlp.sprl.TextOnly.SpRLApp"
```

results will be saved in `data/mSprL/results` folder as text files corresponding to the model selected in the config file. 

## Results on CLEF 2017 dataset
Here are the summarized results of relation classifier for a Baseline model and the baseline plus constraints:

```
label                           Precision  Recall     F1         LCount     PCount    
-----------------------------------------------------------------------------------
BM                              65.640     60.226     62.817     885        812
BM+Constraints                  70.036     66.554     68.250     885        841       

```

