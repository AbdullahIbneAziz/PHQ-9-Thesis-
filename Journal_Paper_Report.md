# Journal Paper Report: A Comparative Study of Mental Health Detection Using PHQ-9 and Social Media Analysis in Bangla with Traditional ML and NLP Techniques

## 1. Overview and Flow of Content

The journal paper will be structured with the following flow:

```
1. Abstract
2. Introduction
3. Literature Review
4. Methodology
   4.1 PHQ-9 Based Analysis (Traditional ML)
   4.2 Bengali Social Media Analysis (NLP/BERT)
5. Experimental Setup
6. Results and Discussion
   6.1 PHQ-9 Model Results
   6.2 Bengali NLP Model Results
   6.3 Comparative Analysis
7. Clinical Implications
8. Limitations and Future Work
9. Conclusion
10. References
```

---

## 2. Detailed Content for Each Section

### Section 1: Abstract

**Content:**

- Brief problem statement: Mental health detection, particularly depression, is a critical healthcare challenge. This study presents a comparative analysis of two approaches for binary depression classification.
- Methodology overview: The research employs two complementary approaches - (1) Traditional Machine Learning on structured PHQ-9 questionnaire data and (2) NLP-based BERT classification on Bengali social media text.
- Key findings:
  - PHQ-9 approach: XGBoost achieved 96.35% accuracy with AUC-ROC of 0.9944
  - Bengali NLP approach: Multilingual BERT achieved ~79% accuracy
- Contribution: First comparative study combining PHQ-9 structured data analysis with Bengali social media text analysis for depression detection

---

### Section 2: Introduction

**Content:**

1. **Background and Motivation:**

   - Global mental health crisis statistics
   - Depression as a leading cause of disability worldwide
   - The need for automated depression detection systems
   - Gap in research for low-resource languages like Bengali
2. **Problem Statement:**

   - Limited automated tools for depression screening
   - Lack of Bengali language NLP tools for mental health
   - Need for comparative analysis between structured questionnaire-based and unstructured text-based approaches
3. **Research Objectives:**

   - Develop binary depression classification using PHQ-9 responses
   - Build Bengali text-based depression classifier using BERT
   - Compare performance of traditional ML vs. deep learning NLP approaches
   - Evaluate clinical applicability of both methods
4. **Contributions:**

   - Novel feature engineering for PHQ-9 data
   - First Bengali social media depression detection using BERT
   - Comprehensive comparison of structured vs. unstructured data approaches
   - Clinically interpretable probability calibration

---

### Section 3: Literature Review

**Content:**

1. **Depression Detection Methods:**

   - Clinical screening tools (PHQ-9, BDI, etc.)
   - Machine learning approaches for mental health
   - Deep learning in mental health detection
2. **PHQ-9 Based Studies:**

   - Previous work on PHQ-9 automated scoring
   - Machine learning applications in questionnaire analysis
   - Feature engineering techniques for clinical data
3. **Social Media Mental Health Analysis:**

   - Text-based depression detection methods
   - Sentiment analysis for mental health
   - Transformer models (BERT) in healthcare NLP
4. **Bengali Language NLP:**

   - Challenges in Bengali text processing
   - Previous work in Bengali sentiment analysis
   - Gap in Bengali mental health NLP research
5. **Research Gap:**

   - No comparative study combining PHQ-9 and social media analysis
   - Limited Bengali depression detection tools
   - Need for clinically deployable solutions

---

### Section 4: Methodology

#### 4.1 PHQ-9 Based Analysis (Traditional ML)

**4.1.1 Dataset Description:**

- Source: `PHQ-9_Dataset_5th Edition.csv`
- Features: Age, Gender, and 9 PHQ-9 questions
- PHQ-9 Categories:
  - Interest/Pleasure (Q1)
  - Feeling Down (Q2)
  - Sleep Trouble (Q3)
  - Tired/Low Energy (Q4)
  - Appetite Issues (Q5)
  - Feeling Bad About Self (Q6)
  - Concentration Trouble (Q7)
  - Moving/Speaking Issues (Q8)
  - Thoughts of Self-Harm (Q9)
- Additional features: Sleep Quality, Study Pressure, Financial Pressure

**4.1.2 Data Preprocessing:**

- Column renaming for clarity
- Label Encoding for categorical variables:
  ```python
  categorical_columns = ['Gender', 'Interest_Pleasure', 'Feeling_Down', 
                        'Sleep_Trouble', 'Tired_Low_Energy', 'Appetite_Issues',
                        'Feeling_Bad_About_Self', 'Concentration_Trouble', 
                        'Moving_Speaking_Issues', 'Thoughts_Self_Harm',
                        'PHQ_Severity', 'Sleep_Quality', 'Study_Pressure', 
                        'Financial_Pressure']
  ```

**4.1.3 Feature Engineering:**

- Binary Target Variable: `Depression_Binary = (PHQ_Total >= 10)`
- Symptom Grouping:
  ```python
  Core_Symptoms = Interest_Pleasure + Feeling_Down
  Physical_Symptoms = Sleep_Trouble + Tired_Low_Energy + Appetite_Issues
  Cognitive_Symptoms = Concentration_Trouble + Feeling_Bad_About_Self
  Risk_Symptoms = Thoughts_Self_Harm + Moving_Speaking_Issues
  Total_Symptom_Score = Core + Physical + Cognitive + Risk
  ```
- Risk Indicators:
  ```python
  High_Risk = (PHQ_Total >= 15)
  Suicidal_Risk = (Thoughts_Self_Harm >= 2)
  ```

**4.1.4 Machine Learning Models:**

1. Random Forest Classifier
2. XGBoost Classifier
3. Logistic Regression
4. Decision Tree Classifier
5. K-Nearest Neighbors (KNN)

**4.1.5 Training Pipeline:**

- Train-Test Split: 80-20 with stratification
- Feature Scaling: StandardScaler
- Cross-Validation: 10-fold StratifiedKFold

**4.1.6 Hyperparameter Optimization:**

- GridSearchCV for XGBoost:
  ```python
  param_grid = {
      'n_estimators': [100, 150],
      'max_depth': [5, 7],
      'learning_rate': [0.1],
      'subsample': [0.8],
      'colsample_bytree': [0.8],
      'min_child_weight': [1, 3]
  }
  ```

**4.1.7 Feature Selection Methods:**

1. XGBoost Feature Importance
2. Recursive Feature Elimination (RFE)
3. Mutual Information Scores

**4.1.8 Probability Calibration:**

- Isotonic Calibration
- Sigmoid (Platt Scaling) Calibration

---

#### 4.2 Bengali Social Media Analysis (NLP/BERT)

**4.2.1 Dataset Description:**

- Source: `PHQ-9 NLP Dataset Collection.csv`
- Total Samples: 2,841 Bengali sentences
- Columns: Category, Sentence, Severity Level
- Categories aligned with PHQ-9 questions
- Binary Labels: "Depressed" vs "Not Depressed"

**4.2.2 Text Preprocessing:**

```python
def clean_mixed_text(text):
    # Lowercase conversion
    # URL removal
    # Mentions and hashtags removal
    # Numbers removal
    # Punctuation removal
    # Emoji and symbol removal (keep Bangla + English letters)
    # Extra whitespace removal
    # Stopword removal (Bengali + English)
```

**Bengali Stopwords:**

- Custom comprehensive Bengali stopword list
- Combined with NLTK English stopwords

**4.2.3 BERT Model Architecture:**

```python
class BERTClassifier(nn.Module):
    def __init__(self, dropout=0.3):
        self.bert = BertModel.from_pretrained('bert-base-multilingual-cased')
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(bert.config.hidden_size, num_classes)
  
    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.bert(input_ids, attention_mask, token_type_ids)
        cls_output = outputs.last_hidden_state[:, 0, :]  # CLS token
        return self.linear(self.dropout(cls_output))
```

**4.2.4 Training Configuration:**

- Tokenizer: `bert-base-multilingual-cased`
- Max Length: 64 tokens
- Batch Size: 8
- Learning Rate: 2e-5
- Optimizer: AdamW
- Loss Function: CrossEntropyLoss
- Epochs: 10
- Train-Validation Split: 80-20

---

### Section 5: Experimental Setup

**Content:**

1. **Hardware Configuration:**

   - GPU: CUDA-enabled device
   - Runtime environment specifications
2. **Software Stack:**

   - Python 3.x
   - Libraries: scikit-learn, XGBoost, PyTorch, Transformers, NLTK, pandas, numpy, matplotlib, seaborn
3. **Evaluation Metrics:**

   - Classification: Accuracy, Precision, Recall, F1-Score, AUC-ROC
   - Calibration: Brier Score
   - Cross-Validation: Mean ± Std for all metrics
4. **Reproducibility:**

   - Random seeds set (42)
   - Stratified sampling
   - Model saving and loading procedures

---

### Section 6: Results and Discussion

#### 6.1 PHQ-9 Model Results

**6.1.1 Model Comparison Table:**

| Model               | Accuracy         | AUC-ROC          | Precision | Recall | F1-Score |
| ------------------- | ---------------- | ---------------- | --------- | ------ | -------- |
| Random Forest       | 0.8905           | 0.9743           | -         | -      | -        |
| **XGBoost**   | **0.9635** | **0.9944** | -         | -      | -        |
| Logistic Regression | 0.8394           | 0.9360           | -         | -      | -        |
| Decision Tree       | 0.8613           | 0.9325           | -         | -      | -        |
| KNN                 | 0.8686           | 0.9384           | -         | -      | -        |

**6.1.2 Cross-Validation Results (XGBoost):**

- Mean Accuracy: 0.9457 ± 0.0137
- Mean AUC-ROC: 0.9904 ± 0.0068
- Mean F1-Score: 0.9414 ± 0.0163
- Minimal overfitting detected

**6.1.3 Optimized XGBoost Results:**

- Best Parameters: `learning_rate=0.1, max_depth=7, n_estimators=150`
- Accuracy: 0.9635
- AUC-ROC: 0.9957

**6.1.4 Feature Importance Analysis:**

- Top Features:
  1. Total_Symptom_Score
  2. Physical_Symptoms
  3. Tired_Low_Energy
  4. Core_Symptoms
  5. Cognitive_Symptoms

**6.1.5 Probability Calibration:**

| Method             | Brier Score      |
| ------------------ | ---------------- |
| Uncalibrated       | 0.0302           |
| Sigmoid            | 0.0283           |
| **Isotonic** | **0.0276** |

---

#### 6.2 Bengali NLP Model Results

**6.2.1 Training Progress:**

- Loss decreased from 162.31 (Epoch 1) to 23.97 (Epoch 10)
- Validation Accuracy stabilized around 79-80%

**6.2.2 Classification Report:**

```
              precision    recall  f1-score   support
   Depressed       0.74      0.85      0.79       273
Not Depressed      0.84      0.73      0.78       296

     accuracy                           0.79       569
    macro avg      0.79      0.79      0.79       569
 weighted avg      0.79      0.79      0.79       569
```

**6.2.3 Confusion Matrix Analysis:**

- Depressed: 231 TP, 42 FP
- Not Depressed: 216 TP, 80 FP

**6.2.4 ROC-AUC Analysis:**

- Depressed class AUC
- Not Depressed class AUC

---

#### 6.3 Comparative Analysis

**Comparison Table:**

| Aspect                 | PHQ-9 (XGBoost)           | Bengali NLP (BERT)  |
| ---------------------- | ------------------------- | ------------------- |
| Data Type              | Structured                | Unstructured Text   |
| Accuracy               | 96.35%                    | ~79%                |
| AUC-ROC                | 0.9944                    | Variable            |
| Interpretability       | High (Feature Importance) | Lower (Black-box)   |
| Clinical Applicability | Immediate                 | Requires Validation |
| Language Support       | Universal                 | Bengali Specific    |
| Data Collection        | Questionnaire             | Social Media        |

**Key Findings:**

1. PHQ-9 structured approach significantly outperforms text-based method
2. Feature engineering crucial for PHQ-9 model success
3. BERT shows promise for low-resource language mental health NLP
4. Combination of both approaches could provide comprehensive screening

---

### Section 7: Clinical Implications

**Content:**

1. **PHQ-9 Model Deployment:**

   - High accuracy suitable for clinical screening
   - Probability calibration enables risk stratification
   - Clinical interpretation guide:
     - <20%: Low risk
     - 20-50%: Monitor
     - 50-70%: Elevated risk
     - > 70%: High priority
       >
2. **Bengali NLP Model Applications:**

   - Social media mental health monitoring
   - Early warning system for at-risk individuals
   - Community mental health screening
3. **Combined Approach Benefits:**

   - PHQ-9 for formal clinical assessment
   - NLP for passive monitoring between assessments
   - Comprehensive mental health surveillance

---

### Section 8: Limitations and Future Work

**Limitations:**

1. **PHQ-9 Model:**

   - Dataset size limitations
   - Potential demographic biases
   - Self-reported data reliability
2. **Bengali NLP Model:**

   - Limited dataset (2,841 samples)
   - Binary classification only
   - Social media text differs from clinical language

**Future Work:**

1. Multi-class severity classification
2. Larger Bengali mental health corpus creation
3. Ensemble methods combining both approaches
4. Longitudinal depression tracking
5. Multi-modal analysis (text + questionnaire)
6. Explainable AI (SHAP, LIME) for NLP model
7. Real-time social media monitoring system

---

### Section 9: Conclusion

**Content:**

1. **Summary of Contributions:**

   - Comprehensive binary depression classification study
   - Novel feature engineering for PHQ-9 data
   - First Bengali social media depression detection using BERT
   - Comparative analysis of structured vs. unstructured approaches
2. **Key Takeaways:**

   - XGBoost on PHQ-9 data achieves 96.35% accuracy (best for clinical use)
   - BERT on Bengali text achieves ~79% accuracy (promising for social media screening)
   - Both approaches complement each other for comprehensive mental health detection
3. **Impact:**

   - Foundation for automated depression screening tools
   - Advancement in Bengali mental health NLP research
   - Clinically deployable models with probability calibration

---

### Section 10: References

**Categories to include:**

1. Mental health and depression statistics (WHO, etc.)
2. PHQ-9 questionnaire validation studies
3. Machine learning in mental health
4. BERT and transformer models
5. Bengali NLP research
6. Social media mental health detection
7. Probability calibration methods
8. Feature engineering techniques

---

## 3. Summary of Key Results

### PHQ-9 Traditional ML Approach:

- **Best Model:** XGBoost Classifier
- **Accuracy:** 96.35%
- **AUC-ROC:** 0.9944 (optimized: 0.9957)
- **Key Features:** Total_Symptom_Score, Physical_Symptoms, Tired_Low_Energy
- **Calibration:** Isotonic calibration recommended (Brier Score: 0.0276)

### Bengali NLP (BERT) Approach:

- **Model:** Multilingual BERT (bert-base-multilingual-cased)
- **Accuracy:** ~79%
- **Precision/Recall:** Balanced performance (0.74-0.85)
- **Dataset:** 2,841 Bengali sentences

### Comparative Insight:

The PHQ-9 structured approach significantly outperforms the text-based NLP method in accuracy (96.35% vs 79%). However, both approaches serve different purposes:

- PHQ-9: Formal clinical assessment
- Bengali NLP: Passive social media monitoring for early detection

---

## 4. Recommended Paper Structure

**Title:** A Comparative Study of Mental Health Detection Using PHQ-9 and Social Media Analysis in Bangla with Traditional ML and NLP Techniques

**Keywords:** Depression Detection, PHQ-9, Bengali NLP, BERT, Machine Learning, Mental Health, XGBoost, Binary Classification, Social Media Analysis

**Estimated Length:** 12-15 pages

**Figures to Include:**

1. Methodology flowchart (both approaches)
2. PHQ-9 feature correlation heatmap
3. ROC curves comparison (all ML models)
4. Bengali text sentiment distribution
5. Confusion matrices for both approaches
6. Feature importance visualization
7. Probability calibration curves

**Tables to Include:**

1. Dataset statistics
2. Model comparison results
3. Cross-validation results
4. Feature importance rankings
5. Calibration comparison
6. Bengali NLP classification report
