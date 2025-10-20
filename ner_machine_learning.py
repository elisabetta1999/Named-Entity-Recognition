from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction import DictVectorizer
import pandas as pd
import sys
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
import csv
import numpy as np
from skopt import BayesSearchCV
from scipy.stats import uniform
from skopt.space import Real

def extract_embeddings_as_features_and_gold(conllfile,word_embedding_model):
    '''
    Function that extracts features and gold labels using word embeddings
    
    :param conllfile: path to conll file
    :param word_embedding_model: a pretrained word embedding model
    :type conllfile: string
    :type word_embedding_model: gensim.models.keyedvectors.Word2VecKeyedVectors
    
    :return features: list of vector representation of tokens
    :return labels: list of gold labels
    '''
    labels = []
    features = []
    
    conllinput = open(conllfile, 'r')
    csvreader = csv.reader(conllinput, delimiter='\t',quotechar='|')
    features_dicts = extract_features(conllfile)
    for dic, row in zip(features_dicts, csvreader):
        if not row or len(row) < 2:
            continue
        for key, value in dic.items():
            if dic['token'] in word_embedding_model:
                token_vector = word_embedding_model[dic['token']]
            else:
                token_vector = [0]*300
            if dic['surrounding_tokens'][0] in word_embedding_model:
                pt_vector = word_embedding_model[dic['surrounding_tokens'][0]]
            else:
                pt_vector = [0]*300
                
            features.append(np.concatenate((token_vector,pt_vector)))
            labels.append(row[-1])
    return features, labels


def extract_features_and_labels(trainingfile):
    
    data = []
    targets = []
    with open(trainingfile, 'r', encoding='utf8') as infile:
        for line in infile:
            components = line.rstrip('\n').split()
            if len(components) > 0:
                token = components[0]
                feature_dict = {'token':token}
                data.append(feature_dict)
                #gold is in the last column
                targets.append(components[-1])
    return data, targets
    
def extract_features(inputfile):
    """Takes as input the path to a conll file and returns a list of dictionaries
    with features for each token. Those features are the token itself, the POS tag,
    the capitalization type ('upper', 'lower', 'title', or 'other'), and the surrounding tokens."""   
    data = []
    with open(inputfile, 'r', encoding='utf8') as infile:
        lines = infile.readlines()
        for line_index, line in enumerate(lines):
            components = line.rstrip('\n').split()
            if len(components) > 0:
                token = components[0] # token itself
                pos = components[1] # pos tag
                chunk = components[2]
                pt = components[3]
                nt = components[4]
                capitalization = components[5]
                feature_dict = {'Token': token, 'Pos': pos, 'Chunklabel': chunk, 
                                'Capitalization': capitalization,'Prev_token': pt, 'Next_token':nt}
                data.append(feature_dict)
    return data
    
def extract_dict_gold(inputfile):
    """Takes as input the path to a conll file and returns a list of dictionaries
    with features for each token. Those features are the token itself, the POS tag,
    the capitalization type ('upper', 'lower', 'title', or 'other'), and the surrounding tokens."""   
    data = []
    with open(inputfile, 'r', encoding='utf8') as infile:
        lines = infile.readlines()
        for line_index, line in enumerate(lines):
            components = line.rstrip('\n').split()
            if len(components) > 0:
                token = components[0] # token itself
                pos = components[1] # pos tag
                chunk = components[2] # chunk
                gold_label = components[-1] 
                if token.isupper(): #capitalization pattern
                    capitalization = 'uppercase'
                elif token.islower():
                    capitalization = 'lowercase'
                elif token.istitle():
                    capitalization = 'first_letter_capitalization'
                else:
                    capitalization = 'other'                
                if line_index > 0:  # if previous line
                    prev_line = lines[line_index-1].rstrip('\n')  # get previous line
                    if prev_line:  # if empty
                        prev_token = prev_line.split()[0] # get previous token
                    else:
                        prev_token = "START!!"  # no tokens before
                else:
                    prev_token = "START!!"  # no tokens before
                if line_index < len(lines) - 1:  # if next line
                    next_line = lines[line_index+1].rstrip('\n')  # get next line
                    if next_line:  # if empty
                        next_token = next_line.split()[0]  # get next token
                    else:
                        next_token = "END!!"  # no tokens after
                else:
                    next_token = "END!!"  # no tokens after
                feature_dict = {'Token': token, 'Pos': pos, 'Chunklabel': chunk, 
                                'Capitalization': capitalization, 'Prev_token': prev_token, 
                                'Next_token': next_token,'Gold_label':gold_label}
                data.append(feature_dict)
    return data

def write_features_to_conll(train_features, outputfile):
    '''takes dict with features and writes them to conll file'''
    with open(outputfile, 'w', encoding='utf8') as outfile:
        for feature_dict in train_features:
            token = feature_dict['Token']
            pos = feature_dict['Pos']
            chunk = feature_dict['Chunklabel']
            capitalization = feature_dict['Capitalization']
            gold_label = feature_dict['Gold_label']
            prev_token = feature_dict['Prev_token']
            next_token = feature_dict['Next_token']
            line = f"{token}\t{pos}\t{chunk}\t{prev_token}\t{next_token}\t{capitalization}\t{gold_label}\n"
            outfile.write(line)

        outfile.write("\n")
    
def create_classifier(train_file, train_targets, modelname = 'logreg', embedded = False, 
                      language_model = None):
    """If 'embedded' is False, it trains either a logistic regression, or Naive Bayes, or SVM classifier 
    by taking as inputs a conll file from which it extracts selected features as a dictionary, and a list of 
    labels corresponding to each instance.It returns a trained logistic regression/naive bayes/SVM model
    and the vectoriser fitted on the selected features, which will be used for predictions.
    If 'embedded' is True, it takes as imput a conll file from which it extracts features and labels, and
    a 'language_model'. it returns the trained logistic regression model fitted on word embeddings. 
    """
    
    if embedded == True:
        feature_vectors, vec, gold_labels = extract_traditional_features_and_embeddings_plus_gold_labels(train_file, language_model)
        model = LinearSVC()
        model.fit(feature_vectors, gold_labels)
        return model, vec
    
    else:
        train_features = extract_features(train_file)
        vec = DictVectorizer()
        features_vectorized = vec.fit_transform(train_features)
        
        if modelname == 'logreg':
            model = LogisticRegression(max_iter=1000)
            model.fit(features_vectorized, train_targets)
            return model,vec
            
        elif modelname == 'NB':
            model = MultinomialNB()
            param_grid = {'alpha':[0.01, 0.05, 0.08],'force_alpha':[False], 'fit_prior':[True], 'class_prior':[None]}
            model.fit(features_vectorized, train_targets)
            grid_search = GridSearchCV(model, param_grid, cv=5, scoring='f1_macro', n_jobs=-1)
            grid_search.fit(features_vectorized, train_targets)
            model = grid_search.best_estimator_
            return model, vec
        
        elif modelname == 'SVM':
            param_space = {'C': Real(0.1, 50, 'uniform')}
            print('Searching parameters...')
            bayes_search = BayesSearchCV(
                LinearSVC(max_iter=5000, tol=0.01),
                param_space,
                n_iter=10,
                cv=5,
                random_state=42
            )
            bayes_search.fit(features_vectorized, train_targets)
            print("Best parameters for SVC:", bayes_search.best_params_)
            print("Best score during tuning:", bayes_search.best_score_)
            
            best_model = bayes_search.best_estimator_
            return best_model, vec

    
def classify_data(model, inputdata, outputfile, vec, embedded = False, language_model = None):
    """Taken an inputdata, an output path, a logistic regression model, a vectorizer(if needed),
    and a language_model(if needed). The function extracts features from the input and writes each 
    line to the output adding the predicted labels"""
    if embedded == False:
        features = extract_features(inputdata)
        features = vec.transform(features)
        predictions = model.predict(features)
          
    if embedded == True:
        predictions, gold_labels = label_data_with_combined_features(inputdata, model, vec, language_model)
    
    outfile = open(outputfile, 'w')
    counter = 0
    for line in open(inputdata, 'r'):
        if len(line.rstrip('\n').split()) > 0:
            outfile.write(line.rstrip('\n') + '\t' + predictions[counter] + '\n')
            counter += 1
    outfile.close()


def main(argv=None):
    
    #a very basic way for picking up commandline arguments
    if argv is None:
        argv = sys.argv
        
    #Note 1: argv[0] is the name of the python program if you run your program as: python program1.py arg1 arg2 arg3
    #Note 2: sys.argv is simple, but gets messy if you need it for anything else than basic scenarios with few arguments
    #you'll want to move to something better. e.g. argparse (easy to find online)
    
    
    #you can replace the values for these with paths to the appropriate files for now, e.g. by specifying values in argv
    #argv = ['mypython_program','','','']
    trainingfile = argv[1]
    inputfile = argv[2]
    outputfile = argv[3]
    
    ## for the word_embedding_model used in the `extract_embeddings_as_features_and_gold' you can either choose to use a statement like this:
    # language_model = gensim.models.KeyedVectors.load_word2vec_format('../../models/GoogleNews-vectors-negative300.bin.gz', binary=True)
    ## and make sure the path works correctly, or you can add an argument to the commandline that allows users to specify the location of the language model.
    
    training_features, gold_labels = extract_features_and_labels(trainingfile)
    for modelname in ['logreg', 'NB', 'SVM']:
        ml_model, vec = create_classifier(training_features, gold_labels, 'logreg')
        classify_data(ml_model, vec, inputfile, outputfile.replace('.conll','.' + modelname + '.conll'))
    
    
if __name__ == '__main__':
    main()


def extract_word_embedding(token, word_embedding_model):
    '''
    Function that returns the word embedding for a given token out of a distributional semantic model and a 300-dimension vector of 0s otherwise
    
    :param token: the token
    :param word_embedding_model: the distributional semantic model
    :type token: string
    :type word_embedding_model: gensim.models.keyedvectors.Word2VecKeyedVectors
    
    :returns a vector representation of the token
    '''
    if token in word_embedding_model:
        vector = word_embedding_model[token]
    else:
        vector = [0]*300
    return vector


def extract_feature_values(row, selected_features):
    '''
    Function that extracts feature value pairs from row
    
    :param row: row from conll file
    :param selected_features: list of selected features
    :type row: string
    :type selected_features: list of strings
    
    :returns: dictionary of feature value pairs
    '''
    feature_to_index = {'Token': 0, 'Pos': 1, 'Chunklabel': 2,'Prev_token': 3, 
                        'Next_token': 4,'Capitalization': 5, 'Gold': 6}
    feature_values = {}
    for feature_name in selected_features:
        r_index = feature_to_index.get(feature_name)
        feature_values[feature_name] = row[r_index]
        
    return feature_values
 
def create_vectorizer_traditional_features(feature_values):
    '''
    Function that creates vectorizer for set of feature values
    
    :param feature_values: list of dictionaries containing feature-value pairs
    :type feature_values: list of dictionairies (key and values are strings)
    
    :returns: vectorizer with feature values fitted
    '''
    vectorizer = DictVectorizer()
    vectorizer.fit(feature_values)
    
    return vectorizer
    
def combine_sparse_and_dense_features(dense_vectors, sparse_features):
    '''
    Function that takes sparse and dense feature representations and appends their vector representation
    
    :param dense_vectors: list of dense vector representations
    :param sparse_features: list of sparse vector representations
    :type dense_vector: list of arrays
    :type sparse_features: list of lists
    
    :returns: list of arrays in which sparse and dense vectors are concatenated
    '''
    
    combined_vectors = []
    sparse_vectors = np.array(sparse_features.toarray())
    
    for index, vector in enumerate(sparse_vectors):
        combined_vector = np.concatenate((vector,dense_vectors[index]))
        combined_vectors.append(combined_vector)
    return combined_vectors
    
def extract_traditional_features_and_embeddings_plus_gold_labels(conllfile, word_embedding_model, vectorizer=None):
    '''
    Function that extracts traditional features as well as embeddings and gold labels using word embeddings for current and preceding token
    
    :param conllfile: path to conll file
    :param word_embedding_model: a pretrained word embedding model
    :type conllfile: string
    :type word_embedding_model: gensim.models.keyedvectors.Word2VecKeyedVectors
    
    :return features: list of vector representation of tokens
    :return labels: list of gold labels
    '''
    labels = []
    dense_vectors = []
    traditional_features = []
    
    conllinput = open(conllfile, 'r')
    csvreader = csv.reader(conllinput, delimiter='\t',quotechar='|')
    for row in csvreader:
        if len(row) == 7:
            token_vector = extract_word_embedding(row[0], word_embedding_model)
            pt_vector = extract_word_embedding(row[3], word_embedding_model)
            nt_vector = extract_word_embedding(row[4], word_embedding_model)
            dense_vectors.append(np.concatenate((token_vector,pt_vector,nt_vector)))
            other_features = extract_feature_values(row, ['Pos', 'Chunklabel','Capitalization'])
            traditional_features.append(other_features)
            labels.append(row[-1])
    if vectorizer is None:
        vectorizer = create_vectorizer_traditional_features(traditional_features)
    sparse_features = vectorizer.transform(traditional_features)
    combined_vectors = combine_sparse_and_dense_features(dense_vectors, sparse_features)
    
    return combined_vectors, vectorizer, labels

def label_data_with_combined_features(testfile, classifier, vectorizer, word_embedding_model):
    '''
    Function that labels data with model using both sparse and dense features
    '''
    feature_vectors, vectorizer, goldlabels = extract_traditional_features_and_embeddings_plus_gold_labels(testfile, word_embedding_model, vectorizer)
    predictions = classifier.predict(feature_vectors)
    
    return predictions, goldlabels