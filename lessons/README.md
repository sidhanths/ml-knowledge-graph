# FAANG MLE Curriculum

Carved subgraph of 106 concepts from the 2,081-node graph, targeted at Classic MLE
(ranking / reco / ads / search). Each lesson: runnable code + plots you can view
on GitHub from a phone.

## Run any lesson locally

```bash
pip install scikit-learn numpy pandas matplotlib shap xgboost
python3 lessons/01_regression/regression_lesson.py
```

Plots land in `lessons/XX_topic/plots/`.

## Curriculum

| # | Lesson | Graph nodes covered |
|---|---|---|
| 01 | **Regression family** | OLS, Ridge, Lasso, Elastic Net, Huber, Quantile, Logistic, SHAP, + split conformal |
| 02 | Probability & expectations | probability, expectation, variance, Bernoulli, Normal, MLE, KL, cross-entropy |
| 03 | Trees & boosting | CART, Random Forest, Gradient Boosting, XGBoost, LightGBM, CatBoost |
| 04 | Optimization | gradient descent, SGD, momentum, AdaGrad, RMSProp, Adam, AdamW, backprop |
| 05 | Neural nets from scratch | perceptron, MLP, activation, dropout, batch/layer norm, softmax |
| 06 | Sequence models | RNN, LSTM, GRU, seq2seq, Bahdanau/Luong attention |
| 07 | Transformers | Transformer, BERT, GPT, T5, ViT |
| 08 | Embeddings + retrieval | TF-IDF, BM25, word2vec, GloVe, FastText, Sentence-BERT, LSH |
| 09 | Matrix factorization reco | MF, ALS, SVD++, FM, NCF |
| 10 | Deep reco | Wide&Deep, DeepFM, DIN, DIEN |
| 11 | Two-tower & retrieval reco | DSSM, Two-Tower, ColBERT, Poly/Cross-Encoder |
| 12 | Learning to rank | RankNet, LambdaRank, LambdaMART, ListNet, ListMLE, BPR |
| 13 | Sequential reco | SASRec, BERT4Rec, GRU4Rec |
| 14 | Training at scale | knowledge distillation, QAT, ZeRO, mixup |
| 15 | Interpretability & uncertainty | SHAP (deep), conformal prediction, calibration |
| 16 | RAG + agents | Retrieval-Augmented Generation, agent isolation patterns |

Ask in chat to advance a lesson.
