from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction import DictVectorizer
import pandas as pd
import sys
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import utils
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, classification_report, ConfusionMatrixDisplay
import numpy as np
import sklearn
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from utils import calculate_metrix_from_output
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
import csv
import gensim
import argparse

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
                chunk = components[2] # syntax
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
                                'Capitalization': capitalization, 'Prev_token': prev_token, 'Next_token': next_token,'Gold_label':gold_label}
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
                # feature_dict = {'Token': token, 'Pos': pos, 
                #                 'Capitalization': capitalization}
                data.append(feature_dict)
    return data

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

def create_classifier(train_file = None, model_name = 'logreg', embedded = False, 
                      language_model_path = None):
    """If 'embedded' is False, it trains either a logistic regression, or Naive Bayes, or SVM classifier 
    by taking as inputs a conll file from which it extracts selected features as a dictionary, and a list of 
    labels corresponding to each instance.It returns a trained logistic regression/naive bayes/SVM model
    and the vectoriser fitted on the selected features, which will be used for predictions.
    If 'embedded' is True, it takes as imput a conll file from which it extracts features and labels, and
    a 'language_model'. it returns the trained logistic regression model fitted on word embeddings including one-hot-vectors for
    for traditional features, such as POS tags,chunk labels and capitalization. 
    """
    
    if embedded == True:
        word_embedding_model = gensim.models.KeyedVectors.load_word2vec_format(language_model_path, binary=True)
        feature_vectors, vec, gold_labels = extract_traditional_features_and_embeddings_plus_gold_labels(train_file, language_model)
        model = SVC(kernel='rbf',gamma='scale',C=1.0)
        
        model.fit(feature_vectors[:90000], gold_labels[:90000])
        return model, vec, word_embedding_model
    
    else:
        train_features = extract_features(train_file)
        train_data = extract_dict_gold(train_file)
        train_targets = [dic['Gold_label'] for dic in train_data]
        
        vec = DictVectorizer()
        features_vectorized = vec.fit_transform(train_features)
        
        if model_name == 'logreg':
            model = LogisticRegression(max_iter=1000)
            model.fit(features_vectorized, train_targets)
            return model,vec
            
        elif model_name == 'NB':
            model = MultinomialNB(alpha=0.08)
            model.fit(features_vectorized, train_targets)
            return model, vec
        
        elif model_name == 'SVM':
            model = LinearSVC(max_iter=2000, C=1, dual=False, tol=0.001)
            vec = DictVectorizer()
            features_vectorized = vec.fit_transform(train_features)
            model.fit(features_vectorized, train_targets)
            return model, vec

def classify_data(inputdata, outputfile, model = None, vec = None, embedded = False, language_model = None):
    """Taken an inputdata, an output path, a logistic regression model, a vectorizer,
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


def metrics_and_matrix(file):
    '''
    Takes a conll file as input, with its last two columns being the gold labels and the predicted labels.
    Prints precision, recall, and F1-score metrics and a numeric confusion matrix (without colors).
    :param file: Path to the CoNLL file
    :type file: str
    '''
    new_gold_labels = []
    new_pred_labels = []

    with open(file, 'r') as infile:
        for line in infile:
            if line.strip():
                parts = line.strip().split()
                new_gold_labels.append(parts[-2])
                new_pred_labels.append(parts[-1])

    labels = ['B-LOC', 'B-MISC', 'B-ORG', 'B-PER', 'I-LOC', 'I-MISC', 'I-ORG', 'I-PER', 'O']

    precision = metrics.precision_score(new_gold_labels, new_pred_labels, labels=labels, average='macro')
    recall = metrics.recall_score(new_gold_labels, new_pred_labels, labels=labels, average='macro')
    fscore = metrics.f1_score(new_gold_labels, new_pred_labels, labels=labels, average='macro')

    confusion = confusion_matrix(new_gold_labels, new_pred_labels, labels=labels)
    
    report = classification_report(new_gold_labels, new_pred_labels, digits=3)
    print(report)
    print("Confusion Matrix:")
    print(confusion)

def main(trainingfile, inputfile, outputfile, model_name, embedded, language_model_path):
    """
    Main function to train a model and classify data.
    
    :param trainingfile: Path to the training file
    :param inputfile: Path to the input file for classification
    :param outputfile: Path to save the classification results
    :param model_name: Name of the model to use for training
    :param embedded: Whether to use word embeddings (True or False)
    :param language_model_path: Path to the language model (e.g., Word2Vec)
    """
    if embedded:
        train_features = extract_dict_gold(trainingfile)
        write_features_to_conll(train_features, 'all_train.conll')
        input_features = extract_dict_gold(inputfile)
        write_features_to_conll(input_features, 'all_input.conll')
        ml_model, vec, word_embedding_model = create_classifier(train_file='all_train.conll', embedded=True, language_model_path=language_model_path)
        classify_data('all_input.conll', outputfile, model=ml_model, vec=vec, language_model=word_embedding_model, embedded=True)
    else:
        ml_model, vec = create_classifier(train_file=trainingfile, model_name=model_name)
        classify_data(inputfile, outputfile, model=ml_model, vec=vec)

    metrics_and_matrix(outputfile)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a classifier and classify data.")
    
    # required arguments
    parser.add_argument('trainingfile', type=str, help="Path to the training file")
    parser.add_argument('inputfile', type=str, help="Path to the input file for classification")
    parser.add_argument('outputfile', type=str, help="Path to save the classification results")
    
    # optional arguments
    parser.add_argument('--model_name', type=str, default='logreg', help="Name of the model to use (default: logreg)")
    parser.add_argument('--embedded', action='store_true', help="Use word embeddings (default: False)")
    parser.add_argument('--language_model_path', type=str, help="Path to the pre-trained language model (e.g., Word2Vec binary file)")
    
    args = parser.parse_args()

    main(args.trainingfile, args.inputfile, args.outputfile, args.model_name, args.embedded, args.language_model_path)
