from collections import defaultdict, Counter
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, classification_report, ConfusionMatrixDisplay
from itertools import combinations

def inspect_data(data, tags, targets):
    """takes as input three lists of data and combines them together in a list of tuples"""
    list_tup = []
    for dt, tags, target in zip(data, tags, targets):
        if target != 'O':
            list_tup.append((dt, tags, target))
    return list_tup

def most_common_words(a_list):
    """takes a list of tuples as input and returns a list with the most 50 common words """
    words_list=[]
    for tuple in a_list:
        word = tuple[0]['token']
        words_list.append(word)
    counts_words = Counter(words_list).most_common(50)
    return counts_words

def most_common_pos(a_list):
    """takes a list of tuples as input and returns a list that includes the most common POS"""
    words_list=[]
    for tuple in a_list:
        word = tuple[1]
        words_list.append(word)
    counts_words = Counter(words_list).most_common(50)
    return counts_words

def analyze_token_length(a_list):
    lengths = [len(tuple[0]['token']) for tuple in a_list]
    avg_length = sum(lengths) / len(lengths)
    return avg_length, max(lengths), min(lengths)

def capitalization_patterns(a_list):
    """takes a list of tuples as input and analyses the capitalization features of the first element of each tuple.
    It counts the occurrences of each capitalization pattern. It returns a dictionary with the number of occurrences
    for each capitalization feature: 'uppercase', 'lowercase', 'first_letter_capitalization', 'other'"""
    capitalization = {"uppercase": 0, "lowercase": 0, "first_letter_capitalization": 0, "other": 0}
    for tuple in a_list:
        token = tuple[0]['token']
        if token.isupper():
            capitalization["uppercase"] += 1
        elif token.islower():
            capitalization["lowercase"] += 1
        elif token.istitle():
            capitalization["first_letter_capitalization"] += 1
        else:
            capitalization["other"] += 1
    return capitalization

def find_labels(inputfile):
    """Takes as input the path to a conll file and returns a list of NER labels"""   
    labels =[]
    with open(inputfile, 'r', encoding='utf8') as infile:
        for line in infile:
            components = line.rstrip('\n').split()
            if len(components) > 0:
                label = components[1]
                labels.append(label)
    return labels

def calculate_metrics(ground_truth_file, prediction_sample_file):
    """takes as  input twto filepaths and returns precision, recall, f_score, confusion matrix and a report"""
    labels = ['B-LOC','B-MISC','B-ORG','B-PER','I-LOC','I-MISC','I-ORG','I-PER','O']
    gt_labels = find_labels(ground_truth_file)
    pred_labels = find_labels(prediction_sample_file)
    precision = precision_score(gt_labels, pred_labels, average='micro')
    recall = recall_score(gt_labels, pred_labels, average='micro')
    f_score = f1_score(gt_labels, pred_labels, average='micro')
    cm = confusion_matrix(gt_labels, pred_labels,labels=labels)
    o_label = labels.index("O")
    cm[o_label, o_label] /= 100
    report = classification_report(gt_labels,pred_labels,digits = 3)

    return precision, recall, f_score, cm, report


def calculate_metrix_from_output (outputfile):
    """it takes a conll file as input, reads it and extrapolates the last two columns of labels (gold and prediction).
    afterwards it creates a report"""
    labels = ['B-LOC','B-MISC','B-ORG','B-PER','I-LOC','I-MISC','I-ORG','I-PER','O']
    conll_gold_labels = []
    conll_pred_labels = []
    with open(outputfile, 'r') as infile:
        for line in infile:
            if line.strip():
                parts = line.strip().split()
                conll_gold_labels.append(parts[-2])
                conll_pred_labels.append(parts[-1])
        
        conll_report= classification_report(conll_gold_labels, conll_pred_labels, digits=3, zero_division = 1)
        print(conll_report)
        conll_confusion_matrix= confusion_matrix(conll_gold_labels, conll_pred_labels, labels = labels)
        o_label = labels.index("O")
        conll_confusion_matrix[o_label, o_label] /= 100
        display_matrix = ConfusionMatrixDisplay(confusion_matrix=conll_confusion_matrix, display_labels=labels)
        display_matrix.plot()

def error_analysis(output_gold_pred_file):
    all_errors = []
    with open(output_gold_pred_file, 'r', encoding='utf8') as infile:
            lines = infile.readlines()
            for line_index, line in enumerate(lines):
                components = line.rstrip('\n').split()
                if len(components) > 0:
                    token = components[0]
                    gold_label = components[-2]
                    prediction = components[-1]
                    if gold_label != prediction:
                        all_errors.append(f'{token}: {gold_label}, {prediction}')
    return all_errors     


def feature_ablation(train_file, dev_file, all_features):
    all_features = ['Token','Pos','Chunklabel','Capitalization','Prev_token','Next_token']
    score = 0
    feat = []
    for r in range(1, len(all_features) + 1):
        for selected_features in combinations(all_features, r):

            selected_features = list(selected_features)
            
            print(f"\033[1mTesting features: {selected_features}\033[0m")

            train_features = extract_features(train_file)
            dev_features = extract_features(dev_file)
            
            train_features = [
                {feature: feature_dict[feature] for feature in selected_features if feature in feature_dict}
                for feature_dict in train_features
            ]
            dev_features = [
                {feature: feature_dict[feature] for feature in selected_features if feature in feature_dict}
                for feature_dict in dev_features
            ]

            train_data = extract_dict_gold(train_file)
            train_labels = [dic['Gold_label'] for dic in train_data]
            dev_data = extract_dict_gold(dev_file)
            dev_labels = [dic['Gold_label'] for dic in dev_data]
            
            vec = DictVectorizer()

            train_features = vec.fit_transform(train_features)
            dev_features = vec.transform(dev_features)

            model = LogisticRegression(max_iter=1000)
            model.fit(train_features, train_labels)
            
            predictions = model.predict(dev_features)

            f1 = sklearn.metrics.f1_score(dev_labels, predictions, average='macro', zero_division=0)

            print(f1)
            if f1> score:
                score = f1
                feat = selected_features

    return score,feat