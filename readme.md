# Multi-modal Spatial Role Labeling
In this project we consider different settings and implementations for [Multi-Modal Spatial Role Labeling task](http://www.cs.tulane.edu/~pkordjam/mSpRL_CLEF_lab.htm).
We use [HetSaul](https://github.com/HetML/HetSaul) as the base framework to model and implement these settings.

## Implementations

- [**EMNLP paper**](src/main/scala/edu/tulane/cs/hetml/nlp/sprl/Pairs/README.md): in this setting we generated pair candidates for trajector-indicator and landmark-indicators and then 
trained classifiers to learn these relations. Finally, we combined the results of these two classifiers to generate the 
final  relations. Global inference over text used to adjust the predictions and also a binary feature is used to determine 
if the current word is connected to any region in the image. This connection is identified by word embedding similarity of 
the word with the label of the segment.


- [**Combining text and image using word-as-classifier and proposition classifier**](src/main/scala/edu/tulane/cs/hetml/nlp/sprl/TextOnly/README.md) : in this setting we generated candidate triplets for 
relations and trained classifiers to learn the relations and relation types in the text. 
In addition spatial relation in the image side are detected through `PrepositionClassifier`. 
A collection of pre-trained word localization classifiers is used to connect text and image modalities.
Finally, we used global inference over the two modalities to adjust final predictions.