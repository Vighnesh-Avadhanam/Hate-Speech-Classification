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
Our training dataset was created by Vidgen et.al (2021), found [here](https://huggingface.co/datasets/tasksource/dynahate).The source for the test data comes from the ETHOS Hate Speech Dectection dataset, found [here](https://github.com/intelligence-csd-auth-gr/Ethos-Hate-Speech-Dataset/blob/master/ethos/ethos_data/Ethos_Dataset_Binary.csv). This dataset scrapes comments from YouTube and Reddit and contains the likelihood of whether a comment is hate speech or not.

# Instructions for Reproducibility


# Models Used
## all-MiniLM-L6-v2


## hateBERT


# Model Selection


# Data Limitations
Our major limitation is that we are unable to detect any possible hate speech that contains sarcasm, as our models are not specifically trained on that. Furthermore, we are unable to capture the full context of the comment. For example, we may have replies for a certain tweet but may not necessarily have information of the tweet itself so that context is ignored.

### Citation

Vidgen, B., Thrush, T., Waseem, Z., & Kiela, D. (2021).  
**Learning from the Worst: Dynamically Generated Datasets to Improve Online Hate Detection**.  
*Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics (ACL)*.  
[https://doi.org/10.18653/v1/2021.acl-long.132](https://doi.org/10.18653/v1/2021.acl-long.132)

