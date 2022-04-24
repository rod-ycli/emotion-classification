VU 2021 Introduction to Human Language Technology
Final Assignment
Roderick YC Li 2740992

This assignment looks at how two different vocabulary filters would have an effect on emotion classifiers for different data sets using different vectorization models. This zip contains files for the assignment where 12 systems were trained and tested on 2 vector models, 3 data sets and 2 vocabulary filters.
These files include:
(A)  7 jupyter notebooks and pdfs containing the source codes for an data overview and for each data set;
(B)  1 pdf report, and;
(C)  1 python file to record the functions used throughout the notebooks.

    (A) The Source Codes

    Training and test on each vector model and data set is documented on each notebook:
    * 0: Overview on MELD and Tweets data
    * 1: MELD, BoW
    * 2: Tweets, BoW
    * 3: MELD+Tweets, BoW
    * 4: MELD, word-embeddings
    * 5: Tweets, word-embeddings
    * 6: MELD+Tweets,  word-embeddings

        Notebook 1-6 are constructed with the same structure:
        I Data
            a. Preprocessing
            b. Filtering:
                - Filter A
                - Filter B
        II Training
            a. Vectorization of training data (1-3 BoW+tfidf; 4-6 Embeddings)
                - Filter A
                - Filter B
            b. Encoding the training labels
            c. Building and training the classifier
                - Filter A
                - Filter B
        III Predictions
            a. Encoding the test labels
            b. Filtering and vectorization of the test data
                - Filter A
                - Filter B
            c. Prediction and results
                - Filter A
                    + Classification report;
                    + Confustion matrix;
                    + Class probability;
                    (+ Important features [BoW])
                - Filter B
                    + Classification report;
                    + Confustion matrix;
                    + Class probability;
                    (+ Important features [BoW])            

    (B) Report
    This report details the motivation for the two filters, the specification of the systems, the analysis of the results and the reflections.
    
    (C) utils.py
    This python file contains the two filters and other important functions used in the notebooks.
