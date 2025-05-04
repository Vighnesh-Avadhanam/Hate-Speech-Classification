# Hate Speech Classification

* Find Dataset to train, ideal if hate is labeled
* Test with "ambiguous data," (Twitter, Reddit, Craigslist, Quora)
* Each comes up with a model to use
    - Nick: Logistsic
    - Ben: Naive Baye's
    - Juan: XGBoost
    - Chris: Random Forest
    - Vighnesh: KNN
    - Misc.: Ridge, Lasso
    - Batch Gradient Descent: Randomly take batch (subset of row) and compute gradient on subset of data
 
* Use Hate BERT for each of our models.
* Write code that brings all ROCs together from regular model and Hate Bert model. (so two graphs each showing multiple ROCs)
* Then created stacked models, one for regular and one for hate BERT
* Then get confusion matrix for each stacked model in order to give a final comparison between Hate Bert and regular

# Problem
In this project, we compare various classification models in their ability to detect hate speech. Broadly, we compare their abilities with a $all-MiniLM-L6-v2$ sentence transformer and a hateBERT (Bidirectional Encoder Representations from Transformers) model.
Specifically, we will train the models with phrases that are already classified as hate speech or not. We then test the models on phrases that are not classified to see how well they can identify hate speech.

# Data Sources and Collection Methods
The source for training data comes from https://huggingface.co/datasets/tasksource/dynahate, created by Vidgen et.al (2021). The source for test data comes from the ETHOS Hate Speech Dectection dataset (https://github.com/intelligence-csd-auth-gr/Ethos-Hate-Speech-Dataset/blob/master/ethos/ethos_data/Ethos_Dataset_Binary.csv), which scrapes comments from YouTube and Reddit with various levels of hate speech recognition.

# Instructions for Reproducibility


# Models Used
## all-MiniLM-L6-v2


## hateBERT


# Model Selection


# Data Limitations


### Citation

Vidgen, B., Thrush, T., Waseem, Z., & Kiela, D. (2021).  
**Learning from the Worst: Dynamically Generated Datasets to Improve Online Hate Detection**.  
*Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics (ACL)*.  
[https://doi.org/10.18653/v1/2021.acl-long.132](https://doi.org/10.18653/v1/2021.acl-long.132)

