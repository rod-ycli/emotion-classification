def low_high_mid_df(min_df, max_df, texts):
    """
    This function separates texts into three iterables according to their document frequencies.
    :min_df: int, words occuring in documents less than this number will be put into low_df.
    :max_df: int, words occuring in documents more than this number will be put into high_df.
    :texts: a list of lists with sentences of spaCy-tokenized words.
    :return: low_df - set of words of which df is too low.
        high_df - set of words of which df is too high.
        mid_df_texts - list of texts in which the dfs of their words are in the middle.
    """
    new_texts = []
    alltokens = set()
    for text in texts:
        sent = []
        for token in text:
            token = token.lemma_.lower()  # Lemmatize the input and make them lowercase.
            sent.append(token)
            alltokens.add(token)
        new_texts.append(sent)
    
    kw_count = dict.fromkeys(alltokens, 0)
    for text in new_texts:
        for key in kw_count:
            if key in text:  # If a word is in a document,
                kw_count[key] += 1  # Df += 1

    low_df = set()
    high_df = set()
    for word, count in kw_count.items():
        if count>max_df:
            high_df.add(word)
        elif count<min_df:
            low_df.add(word)
            
    mid_df_texts = []
    for text in new_texts:
        mid_df_texts.append([tok for tok in text if (tok not in high_df) and (tok not in low_df)])
    
    print('Min_df', min_df)
    print('Max_df', max_df)
    return low_df, high_df, mid_df_texts

def remove_DT_PRP(min_df, texts):
    """
    This function 1) removes determiners and pronouns in texts; 2) separates rare words with low df.
    :min_df: int, words occuring in documents less than this number will be put into low_df.
    :texts: a list of lists with sentences of spaCy-tokenized, POS-tagged words.
    :return: low_df - list of words of which df is too low.
        clean_texts - list of sentences with no determiners, pronouns and low-df words.
    """
    DTandPRP_tag = ["DT", "PRP", "PRP$"]
    DTandPRP_tok = set()
    vocab = set()
    new_texts = []
    for text in texts:
        sent = []
        for token in text:
            if token.tag_ in DTandPRP_tag:
                DTandPRP_tok.add(token.lemma_.lower())
            else:
                token = token.lemma_.lower()
                sent.append(token)
                vocab.add(token)  # Create a set a vocab without the DTs and PRPs
        new_texts.append(sent)
    
    kw_count = dict.fromkeys(vocab, 0)
    for text in new_texts:
        for key in kw_count:
            if key in text:  # If a word is in a document,
                kw_count[key] += 1  # Df += 1
    
    low_df = set()
    for word, count in kw_count.items():
        if count<min_df:
            low_df.add(word)
    
    clean_texts = []
    for text in new_texts:
        # Keep words if they are not DTs nor PRPs, and not lower than min_df
        clean_texts.append([tok for tok in text if (tok in vocab) and (tok not in low_df)])
    
    print('Determiner and pronouns', DTandPRP_tok)
    print('Min_df', min_df)
    return low_df, DTandPRP_tok, clean_texts

def dummy(x):
    return x

# Function to average all word vectors in a paragraph
def featureVecMethod(list_of_words_in_the_utterance, # Tokenized list of tokens from an utterance
                     model,
                     modelword_index, 
                     num_embedding_dimensions 
                    ):
    
    import numpy as np
    featureVec = np.zeros(num_embedding_dimensions,dtype="float32")
    
    nwords = 0
    embedding_words= []
    no_embedding_words= []
    
    for word in  list_of_words_in_the_utterance:
        if word in modelword_index:
            featureVec = np.add(featureVec,model[word]/np.linalg.norm(model[word]))

            embedding_words.append(word)
            nwords = nwords + 1
        else:
            no_embedding_words.append(word)
         
    if nwords>0:
        featureVec = np.divide(featureVec, nwords)
    
    return featureVec, embedding_words, no_embedding_words


# Function for calculating the average feature vector
def getAvgFeatureVecs(texts, ### List of texts, in our case tokenized utterances 
                      model, 
                      modelword_index, 
                      num_embedding_dimensions
                     ):
    counter = 0
    embedding_words=[]
    no_embedding_words=[]
    
    import numpy as np
    textFeatureVecs = np.zeros((len(texts),num_embedding_dimensions),dtype="float32")
    print('Shape of our matrix is:',textFeatureVecs.shape)
    
    for text in texts:
        if counter%1000 == 0:
            print("Review %d of %d"%(counter,len(texts)))
        
        textFeatureVecs[counter], emb_words, nemb_words = featureVecMethod(text,
                                                                           model, 
                                                                           modelword_index,
                                                                           num_embedding_dimensions)
        
        counter = counter+1
        embedding_words.extend(emb_words)
        no_embedding_words.extend(nemb_words)
        
    textFeatureVecs = np.nan_to_num(textFeatureVecs) 

    return textFeatureVecs, embedding_words, no_embedding_words