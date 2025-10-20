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

## Usage

### Installation

```bash
pip install -r requirements.txt
```

### Running the classifier

```bash
python ner_classifier.py <training_file> <test_file> <output_file>
```

**Example:**
```bash
python ner_classifier.py data/train.conll data/test.conll results/predictions.conll
```

This trains all three models and outputs predictions in CoNLL format.

### Input Format (CoNLL-2003)

Expected tab-separated format:
```
Token	POS	Chunk	PrevToken	NextToken	Capitalization	GoldLabel
John	NNP	B-NP	START!!	Smith	first_letter_capitalization	B-PER
Smith	NNP	B-NP	John	works	first_letter_capitalization	I-PER
works	VBZ	B-VP	Smith	at	lowercase	O
at	IN	B-PP	works	Google	lowercase	O
Google	NNP	B-NP	at	.	first_letter_capitalization	B-ORG
```

The last column contains entity labels in BIO format:
- `B-X`: Beginning of entity type X
- `I-X`: Inside/continuation of entity type X
- `O`: Outside any entity (not an entity)

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

**Author**: Developed as part of Master's coursework in Natural Language Processing# Named-Entity-Recognition
