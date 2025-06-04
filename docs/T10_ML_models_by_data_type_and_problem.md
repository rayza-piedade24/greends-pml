## ML approaches and foundation models

When we need to apply ML to solve a problem, it is necessary to (1) identify the problem, (2) choose an adequate model, and possibly, (3) adapt and fine-tune an available foundation model.

Using a foundation model instead of creating a model from scratch offers several significant advantages:
- Time and Cost Efficiency: Foundation models are pre-trained on massive datasets, which means organizations can skip the resource-intensive and expensive pretraining phase. **Fine-tuning** a foundation model for a specific task is much faster and requires significantly less data and computational power than building a model from the ground up
- Higher Baseline Performance: Foundation models typically deliver strong baseline accuracy and performance because they have already learned general patterns and representations from diverse data sources. This provides a high-quality starting point for many applications
- Adaptability and Versatility: These models are designed to be general-purpose and can be adapted to a wide range of tasks—including those not originally envisioned during their pretraining—by leveraging **transfer learning** and fine-tuning
- Lower Data Requirements for New Tasks: Because foundation models have already learned broad features, they require much **smaller labeled datasets** to adapt to new domains or tasks, reducing the need for costly data collection and annotation
- Faster Deployment and Scalability: Organizations can deploy AI solutions more rapidly by building on top of foundation models, accelerating time to value and enabling quick scaling across different use cases
- Reduced Technical Barriers: Adapting a foundation model typically requires **less specialized machine learning expertise** than developing a new model from scratch, making advanced AI more accessible to a broader range of teams and organizations

## Some ML models that are relevant for some types of data and problems to address

| Data Type | Problem Type | Agrarian Example | Main ML Models/Algorithms | Foundation Models to Consider |
| :-- | :-- | :-- | :-- | :-- |
| Tabular | Yield Prediction | Predicting wheat yield from weather and soil data | Linear Regression, Random Forest, XGBoost, Neural Networks | Tabular FMs, LLMs for tabular data, AutoML FMs |
| Tabular | Market Forecasting | Forecasting crop prices for corn or soybeans | ARIMA, Prophet, Gradient Boosting, Random Forest | Time-series FMs (TimeGPT, TabPFN), LLMs for economic data |
| Tabular | Resource Optimization | Optimizing fertilizer use based on field data | Decision Tree, Random Forest, Bayesian Models | Multimodal FMs (sensor, weather, management data) |
| Images | Disease Detection | Identifying leaf diseases in tomatoes from photos | CNNs, ResNet, VGG | Large Vision Models (SAM, CLIP, ViT), crop-specific vision FMs |
| Images | Weed Detection | Detecting weeds in drone images of fields | YOLO, Faster R-CNN, Deep Learning CNNs | Vision FMs (YOLOv8, SAM), multimodal FMs for image-text tasks |
| Images | Crop Segmentation | Mapping field boundaries from satellite images | U-Net, CNNs, K-Means, DBSCAN | Remote Sensing FMs (SatMAE, SAM), ViT, multimodal FMs |
| Images | Livestock Monitoring | Monitoring cattle health via camera feeds | CNNs, LSTM for video, Object Detection models | Vision FMs, multimodal FMs (video, audio, sensor data) |
| Text | Pest Alert Classification | Classifying pest risk from farmer reports | Naive Bayes, SVM, BERT, fastText | LLMs (GPT-4, BERT), domain-adapted LLMs |
| Time Series | Irrigation Scheduling | Predicting optimal irrigation times from sensor data | LSTM, Random Forest, Regression Models | Time-series FMs (TimeGPT), Multimodal FMs for sensor fusion |
| Sensor Data | Soil Moisture Prediction | Predicting soil moisture for precision irrigation | LSTM, Regression, Random Forest | Multimodal FMs (sensor, weather, imagery), Time-series FMs |

- **FMs:** Foundation Models
- **LLMs:** Large Language Models
- **SAM:** Segment Anything Model
- **ViT:** Vision Transformer
- **TimeGPT, TabPFN, SatMAE:** Examples of specialized foundation models for time-series, tabular, and satellite data

