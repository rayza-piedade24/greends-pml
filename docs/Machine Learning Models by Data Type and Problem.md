<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" class="logo" width="120"/>

## Machine Learning Models by Data Type and Problem

| Data Type | Problem Type | Example Applications | Main ML Models/Algorithms |
| :-- | :-- | :-- | :-- |
| Tabular | Classification | Fraud detection, churn prediction | Logistic Regression, Decision Tree, Random Forest, XGBoost, SVM, Gradient Boosting[^1][^4][^7] |
| Tabular | Regression | Price prediction, demand forecasting | Linear Regression, Random Forest, XGBoost, Neural Networks[^1][^7] |
| Tabular | Forecasting | Sales forecasting, inventory | ARIMA, Prophet, LSTM, Gradient Boosting, Random Forest[^7] |
| Images | Classification | Disease diagnosis, object recognition | Convolutional Neural Networks (CNNs), ResNet, VGG, SVM[^2][^4] |
| Images | Object Detection | Autonomous vehicles, surveillance | YOLO, Faster R-CNN, RetinaNet, EfficientDet[^5] |
| Images | Segmentation | Medical imaging, satellite analysis | U-Net, CNNs, K-Means, DBSCAN, SVM, Random Forest[^2][^6] |
| Images | Clustering | Unlabeled image grouping | K-Means, Hierarchical Clustering, DBSCAN[^2][^6] |
| Text | Classification | Spam detection, sentiment analysis | SVM, Naive Bayes, fastText, XLNet, BERT[^3][^4] |
| Text | Sequence Labeling | Named entity recognition | LSTM, CRF, BERT, XLNet[^3] |
| Text | Generation | Chatbots, summarization | GPT, BERT, XLNet[^3] |

**Notes:**

- Many problems can be addressed with both traditional ML (e.g., Random Forest, SVM) and deep learning (e.g., CNNs for images, LSTM for sequences) depending on data size and complexity[^1][^2][^3][^4][^6].
- Some models are specialized: U-Net for image segmentation, YOLO for object detection, XLNet and fastText for text classification[^3][^5][^6].
- Unsupervised methods like K-Means and DBSCAN are common for clustering and exploratory segmentation, especially with images and tabular data[^2][^6].

This table summarizes the most relevant data types, problem types, and the main machine learning models applied to each, based on current best practices and recent literature.

<div style="text-align: center">⁂</div>

[^1]: https://github.com/lmassaron/Machine-Learning-on-Tabular-Data

[^2]: https://www.datacamp.com/tutorial/seeing-like-a-machine-a-beginners-guide-to-image-analysis-in-machine-learning

[^3]: https://www.projectpro.io/article/machine-learning-nlp-text-classification-algorithms-and-models/523

[^4]: https://www.geeksforgeeks.org/top-6-machine-learning-algorithms-for-classification/

[^5]: https://www.hitechbpo.com/blog/top-object-detection-models.php

[^6]: https://labelyourdata.com/articles/segmentation-machine-learning

[^7]: https://cloud.google.com/vertex-ai/docs/tabular-data/overview

[^8]: https://www.manning.com/books/machine-learning-for-tabular-data

[^9]: https://www.reddit.com/r/MLQuestions/comments/1hzr2s2/what_is_the_best_ml_model_to_use_for_large/

[^10]: https://developers.arcgis.com/python/latest/guide/ml-and-dl-on-tabular-data/

