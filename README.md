# Named Entity Recognition (NER) - CoNLL-2003

## What is Named Entity Recognition?

Named Entity Recognition is a Natural Language Processing task that identifies and classifies named entities in text into predefined categories. Common entity types include:

- **Persons**: Names of people (e.g., "John Smith")
- **Organizations**: Companies, institutions, groups (e.g., "Google", "the UN")
- **Locations**: Geographic places (e.g., "Paris", "United States")
- **Miscellaneous**: Other named entities like products, events, etc.

NER is fundamental to many NLP applications: information extraction, question answering, knowledge graph construction, and search engines all rely on understanding *what* entities appear in text and *what they are*.

### Why is this useful?

Instead of just recognizing "John Smith works at Google," NER allows systems to understand that "John Smith" is a person and "Google" is an organization. This structured understanding enables downstream tasks like building knowledge databases, entity linking, and relationship extraction.

## About This Project

This project implements machine learning approaches to NER on the CoNLL-2003 benchmark dataset. Rather than using modern neural networks, it focuses on traditional feature engineering combined with classical machine learning classifiers, demonstrating how linguistic knowledge can be leveraged for entity recognition.

## Approach

### Feature Engineering

The code extracts the following linguistic features for each token:

- **Token itself**: The word being classified
- **POS (Part-of-Speech) tag**: Grammatical category (noun, verb, etc.)
- **Chunk label**: Phrase-level syntactic information
- **Capitalization pattern**: Whether the token is uppercase, lowercase, title-case, or mixed
- **Context**: The previous and next tokens in the sentence

These features are particularly informative for NER because named entities often have distinctive capitalization patterns and grammatical properties.

### Feature Representation

Two approaches are compared:

1. **Traditional features only**: Categorical features (POS tags, capitalization, etc.) converted to sparse vectors via one-hot encoding
2. **Combined embeddings + traditional features**: Dense word embeddings (e.g., Word2Vec) concatenated with sparse traditional features

## Machine Learning Models Tested

The project compares three classical machine learning classifiers:

### 1. Logistic Regression

- **What it does**: Linear classifier that learns a decision boundary between entity classes
- **Advantages**: Fast training, interpretable, works well with moderate feature sets
- **Hyperparameters**: `max_iter=1000`
- **Best for**: When you need quick results and interpretability

### 2. Naive Bayes (Multinomial)

- **What it does**: Probabilistic classifier assuming features are conditionally independent given the class
- **Advantages**: Very fast, works well with text data, handles high-dimensional sparse vectors naturally
- **Hyperparameters**: `alpha=0.08` (smoothing parameter, optimized via GridSearch)
- **Best for**: Sparse, high-dimensional feature spaces; fast baseline

### 3. Support Vector Machine (SVM)

- **What it does**: Finds the optimal hyperplane to separate classes in high-dimensional space
- **Advantages**: Powerful at finding non-linear decision boundaries, handles both sparse and dense features
- **Hyperparameters**: `C=1, max_iter=2000, dual=False, tol=0.001` (optimized via BayesSearch)
- **Best for**: Complex patterns, small to medium-sized datasets

## Model Comparison Summary

| Aspect | Logistic Regression | Naive Bayes | SVM |
|--------|-------------------|-------------|-----|
| Speed | Fast | Very Fast | Moderate |
| Interpretability | High | Medium | Low |
| Scalability | Good | Excellent | Good |
| Non-linear patterns | Limited | Limited | Excellent |
| Hyperparameter tuning | Minimal | Minimal | Significant |

## Requirements

```
scikit-learn>=1.0.0
numpy>=1.20.0
pandas>=1.3.0
gensim>=4.0.0
matplotlib>=3.3.0
```

Install dependencies:
```bash
pip install -r requirements.txt
```

**Optional**: For word embedding features, download pre-trained Word2Vec vectors:
- [Google News Word2Vec (3GB)](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?resourcekey=0-wjGZdNAUop6WykTtMip30g)

### Running the classifier

**Basic usage (traditional features only):**
```bash
python final_script.py <training_file> <test_file> <output_file> --model_name <model>
```

**Available models**: `logreg`, `NB`, `SVM`

**Examples:**

Train Logistic Regression (default):
```bash
python final_script.py data/train.conll data/test.conll results/predictions_logreg.conll
```

Train Naive Bayes:
```bash
python final_script.py data/train.conll data/test.conll results/predictions_nb.conll --model_name NB
```

Train SVM:
```bash
python final_script.py data/train.conll data/test.conll results/predictions_svm.conll --model_name SVM
```

**With word embeddings (combined features):**
```bash
python final_script.py data/train.conll data/test.conll results/predictions_embed.conll --embedded --language_model_path path/to/GoogleNews-vectors-negative300.bin
```

The script automatically evaluates the model and outputs:
- Classification report (precision, recall, F1-score per entity type)
- Confusion matrix
- Predictions written to the output file

### Input Format (CoNLL-2003)

**Original CoNLL-2003 format** (tab-separated, 4 columns):
```
Token	POS	Chunk	NER_Label
EU	NNP	B-NP	B-ORG
rejects	VBZ	B-VP	O
German	JJ	B-NP	B-MISC
call	NN	I-NP	O
to	TO	B-VP	O
boycott	VB	I-VP	O
British	JJ	B-NP	B-MISC
lamb	NN	I-NP	O
.	.	O	O
```

The script automatically extracts additional features (capitalization patterns, previous/next tokens) during processing.

**Entity labels use BIO format:**
- `B-X`: Beginning of entity type X
- `I-X`: Inside/continuation of entity type X  
- `O`: Outside any entity (not an entity)

**Entity types in CoNLL-2003:**
- `PER`: Person names
- `ORG`: Organizations
- `LOC`: Locations
- `MISC`: Miscellaneous entities

## Key Implementation Details

- **Out-of-vocabulary tokens**: Unknown words represented as zero vectors (300 dimensions)
- **Context window**: Single-token context (previous and next tokens only)
- **Feature combination**: Sparse and dense features concatenated using NumPy
- **Training efficiency**: SVM trained on first 50,000 instances for performance

## Limitations

- **No sequence modeling**: Classifies tokens independently, ignoring relationships between consecutive entity labels. A CRF (Conditional Random Field) or BiLSTM would be better.
- **Limited context**: Only considers immediate previous/next tokens; longer context could improve performance
- **No transformer models**: Modern pre-trained models (BERT, RoBERTa) would likely outperform these approaches significantly
- **Feature engineering overhead**: Requires manual feature extraction; end-to-end neural approaches learn features automatically

## Modern Alternatives

For production NER systems, consider:

- **Fine-tuned transformers**: BERT, RoBERTa, or multilingual BERT fine-tuned on NER tasks
- **Specialized libraries**: spaCy, Stanza, or Hugging Face transformers with pre-trained NER models
- **Sequence models**: BiLSTM-CRF architectures that model entity sequences explicitly

## References

- Tjong Kim Sang, E.F., & De Meulder, F. (2003). [Introduction to the CoNLL-2003 Shared Task: Language-Independent Named Entity Recognition](https://arxiv.org/abs/cs/0306050)
- CoNLL-2003 benchmark dataset

---

**Author**: Developed as part of Master's coursework in Natural Language Processing
