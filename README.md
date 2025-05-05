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
In this project, we compare various classification models in their ability to detect hate speech. Specifically, we will train the models with phrases that are already classified as hate speech or not. We then test the models on phrases that are not classified to see how well they can identify hate speech. Broadly, we compare their detection abilities using a $all-MiniLM-L6-v2$ sentence transformer model, and later find an optimal stacking of the models.

# Data Sources and Collection Methods
Our training dataset was created by Vidgen et.al (2021), found [here](https://huggingface.co/datasets/tasksource/dynahate). This consists of over 41,000 comments that are defined as either "hate" or "not hate". The source for the test data comes from the ETHOS Hate Speech Dectection dataset, found [here](https://github.com/intelligence-csd-auth-gr/Ethos-Hate-Speech-Dataset/blob/master/ethos/ethos_data/Ethos_Dataset_Binary.csv). This dataset scrapes 998 comments from YouTube and Reddit and contains the likelihood of whether a comment is hate speech or not.

# Instructions for Reproducibility
To obtain the training data, download `hugging_face_data.py` file from the Hugging Face dataset, place into the code folder, and run the `csv_creator.py` file in the code folder. To obtain the test data, download the $Ethos_Dataset_Binary.csv$ file from the attached test repository, place into the data folder, then run the `clean_data.py` code in the code folder to change the format and file name. To run the models, their respective codes can be ran in the code folder to obtain the necessary information on each model.

# Models Used
## all-MiniLM-L6-v2
![image](https://github.com/user-attachments/assets/d22a1f52-a94d-4e7f-a4ad-f9749ccc18d0)


## hateBERT


# Model Selection


# Data Limitations
Our major limitation is that we are unable to detect any possible hate speech that contains sarcasm, as our models are not specifically trained on that. Furthermore, we are unable to capture the full context of the comment. For example, we may have replies for a certain tweet but may not necessarily have information of the tweet itself so that context is ignored. Similarly, much of the hate speech online nowadays is heavily coded and abstract to the point where the data we tested on would not be able to pick up on the subtleties that could be found.

# Model Limitations
A major limitation for the models is the computation time, as some of these models run for at least 20 minutes to obtain output.

### Citation

Vidgen, B., Thrush, T., Waseem, Z., & Kiela, D. (2021).  
**Learning from the Worst: Dynamically Generated Datasets to Improve Online Hate Detection**.  
*Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics (ACL)*.  
[https://doi.org/10.18653/v1/2021.acl-long.132](https://doi.org/10.18653/v1/2021.acl-long.132)

