# Practical Machine Learning, Green Data Science, 2nd semester 2024/2025

---
Instructor: Manuel Campagnolo, ISA/ULisboa (mlc@isa.ulisboa.pt)

Teaching assistant: Dominic Welsh (djwelsh@edu.ulisboa.pt)

The course will follow a mixed flipped classroom model, where students are supposed to work on suggested topics autonomously before classes. Work outside class will be based on a range of Machine Learning resources including the book *Sebastian Raschka, Yuxi (Hayden) Liu, and Vahid Mirjalili. Machine Learning with PyTorch and Scikit-Learn. Packt Publishing, 2022*. During classes, Python notebooks will be run on Google Colab.

Links for class resources:
  - [Fenix webpage](https://fenix.isa.ulisboa.pt/courses/aaapl-283463546572013). Course official page, where final results will be posted.
  - [Moodle ULisboa](https://elearning.ulisboa.pt/). Evaluation: assignments. The course is called [Practical Machine Learning](https://elearning.ulisboa.pt/course/view.php?id=10469). Students need to self-register in the Moodle page for the course.
  - [Kaggle](https://www.kaggle.com/). Access to data; candidate problems for the final project.

<!---
[Overview notebook](https://github.com/isa-ulisboa/greends-pml/blob/main/ML_overview_with_examples.ipynb) This notebook provides an overview of the full course and contains pointers for other sources of relevant information and Python scripts.
--->

---

# Sessions
Each description below includes the summary of the topics covered in the session, as well as the description of assignments and links to videos or other materials that students should work through.

<details markdown="block">
<summary> Introduction (Feb 21, 2025) </summary>

The goal of the first class is to give an introduction to ML and also to show some of the problems that can be addressed with the techniques and tools that will be discussed during the semester. The examples will be run on Colab.

- See (Raschka et al, 2022), Chapter 1: Giving Computers the Ability to Learn from Data
- Types of machine learning problems: supervised learning, unsupervised learning, reinforcement learning. Suggestion: check video [Types of machine learning](https://www.youtube.com/watch?v=gh6mNF2BGvk)
- Supervised learning: classification vs regression 
- Examples of input data for machine learning problems: tabular data, images, text. See *Iris data set* example with the notebook [iris_regression_classification.ipynb](https://github.com/isa-ulisboa/greends-pml/blob/main/notebooks/iris_regression_classification.ipynb)
- [Example of inference for regression over the Iris data set](https://github.com/isa-ulisboa/greends-pml/blob/main/notebooks/iris_LM_inference.ipynb)
- Statistics vs Machine Learning: Check video: [When to use stats vs. ML?](https://www.youtube.com/watch?v=xUsm34qnE30)
- An example of a prediction task for time series: check the notebook [modeling ground water levels](https://www.kaggle.com/code/andreshg/timeseries-analysis-a-complete-guide/) for the Kaggle competition [Acea Smart Water Analytics](https://www.kaggle.com/competitions/acea-water-prediction/). Try to download the data and run the notebook to reproduce the results. 
</details>

<details markdown="block">
<summary> Basic concepts (Feb 28, 2025): model, loss, fit, learning rate, perceptron, ... </summary>

The goal of the following classes is to understand how ML models can be trained in and used to solve regression and classification problems. We start by applying the machine learning approach to well-known statistical problems like linear regression to illustrate the stepwise approach followed in ML. We use synthetic data generated from a linear or quadratic regression, where one can control the underlying model and the amout of noise. Then, we consider the  `Iris` tabular data set with 4 explanatory variables and categorical label that can be one of three species.

- See (Raschka et al, 2022), Chapter 2: Training Simple Machine Learning Algorithms for Classification
- Video on the Perceptron and early times of AI [The First Neural Networks](https://www.youtube.com/watch?v=e5dVSygXbAE&t=88s)
- Basic concepts in Machine learning: *model*, *fit*, *epochs*, *loss*, *learning rate*, *perceptron*, parameters *weights*, for a simple regression problem. See [Basic concepts notes](https://github.com/isa-ulisboa/greends-pml/blob/main/docs/T1_basic_concepts.md).
- Exercise: pseudo-code to train a simple Linear Regression model. See [Basic concepts notes](https://github.com/isa-ulisboa/greends-pml/blob/main/docs/T1_basic_concepts.md). 
</details>

<details markdown="block">
<summary> Backpropagation (Mar 7, 2025): SGD, forward pass, backward pass, PyTorch, optimizer, ... </summary>

- See (Raschka et al, 2022), Chapter 2: Training Simple Machine Learning Algorithms for Classification
- See [Basic concepts notes](https://github.com/isa-ulisboa/greends-pml/blob/main/docs/T1_basic_concepts.md). 
- Revise solutions for the problems listed in the previous class.
- Backpropagation and computation graph
- `PyTorch` pipeline: loss, optimizer
- The following table illustrates the changes from a basic Python script which is dependent on the model, loss, etc,  to a PyTorch higher-level script that can easily generalized to other models, loss functions or optimizer strategies.

| Basic Python | PyTorch 
|---|---
| Define model explicitly | Use a pre-defined model
|`def predict(x):`|`torch.nn.Linear(in_size,out_size)`
| Define loss explicitly | Use a pre-defined loss function
|`def loss(y,y_pred):`|`loss=torch.nn.MSEloss(y,y_pred)`
| Loss optimization strategy | Use a pre-defined optimizer
| Code explicitly| `optimizer=torch.optim.SGD(params, learn_rate)`
| Compute *ad hoc* gradient | **Use built-in backpropagation mechanism**
|`def gradient(x,y,y_pred):`|`loss.backward()`
|Update weights explicitly| `optimizer.step()`

- Description of assignment #1

</details>

<details markdown="block">
<summary> Decision trees (Mar 14, 2025): entropy, over-fitting, train and development </summary>

- See (Raschka et al, 2022), Chapter 3: Decision tree learning (pg 86-98)
- See [Decision tree notes](https://github.com/isa-ulisboa/greends-pml/blob/main/docs/T2_decision_trees_overfitting_train_test.md)
- How to grow a decision tree
- What is entropy and how does it help us to find the best model? Check  the Princeton video on [Information Theory Basics](https://www.youtube.com/watch?v=bkLHszLlH34).
- The risk of over-fitting: train and development sets
- Decision tree hyper-parameters
- Exercise: create a decision tree for the [Soil detection for cotton crop problem](https://www.kaggle.com/datasets/zohasohail/soil-detection-for-cotton-crop) and determine the best values for hyper-parameters Maximum depth and Minimum leaf size.
- Comparing  last session (perceptron) with this session (decision tree):

| Class | Mar 7 | Mar 14
|--- |--- |---
| Model | Perceptron | Decision tree
| Problem | regression | classification
| Data set | train only | train and development
| Hyper-parameters | learning rate, number iterations | tree depth, leaf size, ...
| Risk of over-fitting | very low | very high
| Loss function | $MSE=\frac{1}{n}\sum_{i=1}^n \left(y_i-\hat{y}_i\right)^2$ | entropy:  $H({\rm \bf p})=-\sum_{i=1}^n \hat{p}_i \log_2 \hat{p}_i$
| Optimization | backpropagation (SGD) | brute force (try all features and all thresholds)
| Python package | PyTorch | scikit learn

  
</details>

<details markdown="block">
<summary> Data preprocessing (Mar 21, 2025): pipelines, missing data, categorical features, scaling, train and test </summary>

- See (Raschka et al, 2022), Chapter 4 (Data Preprocessing) and Chapter 6 (Streamlining workflows with pipelines)
- Supervised learning flowchart
  <details markdown="block">
  <summary>Figure 1.9 (Raschka et al, 2022) </summary>
  <img src="https://github.com/isa-ulisboa/greends-pml/blob/main/docs/supervised_learning_flowchart_raschka_2022.png" alt="Alt Text" width="600" >
  </details>
- The Titanic data set example: See [Pre-processing notes](https://github.com/isa-ulisboa/greends-pml/blob/main/docs/T3_missing_data_categorical_scaling.md)
- Dealing with missing data;
- Handling categorical data;
- Bringing features onto the same scale;
- Partitioning a dataset into separate training and test datasets;
- Scikit learn pipeline: `.transform`, `.fit` and `.predict` methods.
  <details markdown="block">
  <summary>Figure 6.1 (Raschka et al, 2022) </summary>
  <img src="https://github.com/isa-ulisboa/greends-pml/blob/main/docs/pipeline_fig_6_1.png" alt="Alt Text" width="500">
  </details>
- Exercise: apply the principles and code discussed above to the Montesinho burned area data set. You can convert the problem into a classification problem by categorizing the original response variable (burned area). See [Pre-processing notes](https://github.com/isa-ulisboa/greends-pml/blob/main/docs/T3_missing_data_categorical_scaling.md)
</details>

<details markdown="block">
<summary>Model Evaluation and hyper-parameter Tuning (Mar 28, 2025): cross-validation, strata and groups, grid-search </summary>

- See (Raschka et al, 2022), Chapter 6: Learning Best Practices for Model Evaluation and hyper-parameter Tuning
- See [Cross-validation and hyper-parameter tuning notes](https://github.com/isa-ulisboa/greends-pml/blob/main/docs/T4_cross_validation.md)
- Streamlining workflows with pipelines
- Using k-fold cross-validation to assess model performance
- Debugging algorithms with learning and validation curves
- Fine-tuning machine learning models via grid search
- Moodle quiz on basic concepts for ML: [Practical Machine Learning](https://elearning.ulisboa.pt/course/view.php?id=10469)
</details>

<details markdown="block">
<summary>  Evaluation metrics (Apr 4, 2025): confusion matrix, precision, recall, F1-score, ROC curve, AUC </summary>

- See (Raschka et al, 2022), Chapter 6: Learning Best Practices for Model Evaluation and hyper-parameter Tuning
- See [Cross-validation and hyper-parameter tuning notes](https://github.com/isa-ulisboa/greends-pml/blob/main/docs/T4_cross_validation.md)
- Looking at different performance evaluation metrics
- Reading a confusion matrix 
- Optimizing the precision and recall of a classification model
- Plotting a receiver operating characteristic (ROC)
- Scoring metrics for multiclass classification 
- Dealing with class imbalance
- Discussion of assignment \#2

</details>

<details markdown="block">
<summary>  Combining Different Models for Ensemble Learning (May 2, 2025): random forest, gradient boosting, variable importance </summary>

- See (Raschka et al, 2022), Chapter 7:  Combining Different Models for Ensemble Learning
- See [Notes on ensemble learning and variable importance](https://github.com/isa-ulisboa/greends-pml/blob/main/docs/T5_ensemble_methods.md)
- Ensemble classifiers
- Random Forests
- Gradient boosting
- Exercise: adapt the classification pipeline to apply the XGBoost classifier (Montesinho burned area data set)
- Variable importance: MDI (Gini importance) and MDA (permutation importance)
- Pipeline that includes feature selection, followed by hyperparameter search: https://github.com/isa-ulisboa/greends-pml/blob/main/notebooks/wine_region_pipeline_XGB_CV_gridsearch_featselection.ipynb

</details>

<details markdown="block">
<summary> Data pipeline for deep learning  (May 9, 2025):  PyTorch, datasets, dataloaders</summary>

- See (Raschka et al, 2022), Chapter 12:   Parallelizing Neural Network Training with PyTorch
- See [Notebook on introduction to data pipelines for deep learning](https://github.com/isa-ulisboa/greends-pml/blob/main/notebooks/T6_pytorch_dataset_dataloader.ipynb). With deep learning (DL), it is possible to solve problems that involve complex input data like images, text and audio. The first step in order to apply DL is to organize the input data. PyTorch provides some key tools like `Dataset` and `DataLoader` that allow the creation of robust pipelines for DL.
- See [Veritasium video (3'42 to 14'50)](https://www.youtube.com/watch?v=GVsUOuSjvcg) for an historic introduction to multilayer neural networks  for deep learning.
- Run an interpret the code in pages 386-388 with an example of a dataset (`CelebA`) with several labels.
  
</details>


<details markdown="block">
<summary> Pipeline for deep learning with PyTorch (May 16, 2025):  data, model, model training and validation</summary>

- See (Raschka et al, 2022), Chapter 12: pp 389 to the end,  and Chapter 13: Going Deeper – The Mechanics of PyTorch, namely the MNIST project (ppp 436-439)
- See [Notebook the typical pipeline for deep learning with (non-convolutional) neural networks](https://github.com/isa-ulisboa/greends-pml/blob/main/notebooks/T7_torch_NN_pipeline.ipynb). In particular, we explore the MNIST dataset.
- Assignment #3 available on Moodle
- Suggestions of videos:
  - [PyTorch Course (2022), Part 4: Image Classification (MNIST)](https://www.youtube.com/watch?v=gBw0u_5u0qU)
  - [PyTorch Crash Course - Getting Started with Deep Learning](https://www.youtube.com/watch?v=OIenNRt2bjg)
  - [Build Your First Pytorch Model In Minutes! [Tutorial + Code](https://www.youtube.com/watch?v=tHL5STNJKag)
  - [MIT Introduction to Deep Learning 2025 (1:09)](https://www.youtube.com/watch?v=alfdI7S6wCY); Introduction up to "What is Deep Learning" (10'57); Why deep learning and why now (15'06); Building Neural Networks with Perceptrons (27'13); Applying NNs (35'30); Training NNs (41'21); NN in practice: Optimization (48'05).
    
</details>

<details markdown="block">
<summary> Deep convolutional neural networks  (May 23, 2025): input preparation, convolution, model architecture, receptive field </summary>

- See (Raschka et al, 2022), Chapter 14: Classifying Images with Deep Convolutional Neural Networks
- Check introductory video [What are CNNs?, by IBM (6'20)](https://www.youtube.com/watch?v=QzY57FaENXg)
- See [Notebook on introduction convolutional neural networks](https://github.com/isa-ulisboa/greends-pml/blob/main/notebooks/T9_CNNs_for_image_classification.ipynb). 
- Application of CNNs to the MNIST problem.
- Suggestions of videos:
  - [MIT 6.S191: Convolutional Neural Networks 2025 (1:01)](https://www.youtube.com/watch?v=oGpzWAlP5p0)
  
</details>

<details markdown="block">
<summary> Model deployment  (May 30, 2025):  saving and loading ML model, Gradio, Hugging Face places</summary>

- Saving and loading a PyTorch model: (1) Saving and Loading the Entire Model (Pickle-Based); (2) TorchScript Export (`jit`); (3) Saving Only State Dict (Most Flexible, Requires Architecture)
- Deploying models with HF spaces. Choose a simple image classification app on Hugging Face spaces (e.g. https://huggingface.co/spaces/ByTixty1/Date_fruit-image-Classification/blob/main/app.py) and test it. Check the files `app.py`, `requirements.py`, `model.pth`. Try to understand the contents of `app.py` which runs Gradio and defines the interface.
- Build a interface with Gradio from scratch
- Create your app in Hugging Face places
- Suggestions of videos:
  - [How to deploy a gradio app on huggingface (43')](https://www.youtube.com/watch?v=bN9WTxzLBRE&t=1845s)
  
</details>

--- 

# Other resources

<details markdown="block">
<summary> Basic resources </summary>
  
- Sebastian Raschka, Yuxi (Hayden) Liu, and Vahid Mirjalili. Machine Learning with PyTorch and Scikit-Learn. Packt Publishing, 2022. See the presentation [webpage](https://sebastianraschka.com/blog/2022/ml-pytorch-book.html) and [GitHub repository](https://github.com/rasbt/machine-learning-book)
- [https://pytorch.org/tutorials/](https://pytorch.org/tutorials/)

</details>

<details markdown="block">
<summary> Tutorials </summary>
  
- [Machine Learning for Beginners (Microsoft)](https://microsoft.github.io/ML-For-Beginners/); [youtube channel](https://www.youtube.com/playlist?list=PLlrxD0HtieHjNnGcZ1TWzPjKYWgfXSiWG)
- [AI for Beginners (Microsoft)](https://microsoft.github.io/AI-For-Beginners/)
- [NYU course: Data Science for Everyone](https://www.youtube.com/@jonesrooy)
- [MIT 6.S191: Introduction to Deep Learning (2024)](https://www.youtube.com/watch?v=ErnWZxJovaM)
- [PyTorch tutorial by Patrick Loeber](https://www.youtube.com/playlist?list=PLqnslRFeH2UrcDBWF5mfPGpqQDSta6VK4). [Github repo](https://github.com/patrickloeber/pytorchTutorial)
- [Stanford Lecture Collection  Convolutional Neural Networks for Visual Recognition (2017)](https://www.youtube.com/playlist?list=PL3FW7Lu3i5JvHM8ljYj-zLfQRF3EO8sYv) and [Notes for the Stanford course on Convolutional Neural Networks for Visual Recognition](https://cs231n.github.io/)
- [Stanford Machine Learning Full Course led by Andrew Ng (2020)](https://www.youtube.com/playlist?list=PLoROMvodv4rMiGQp3WXShtMGgzqpfVfbU). Led by Andrew Ng, this course provides a broad introduction to machine learning and statistical pattern recognition. Topics include: supervised learning (generative/discriminative learning, parametric/non-parametric learning, neural networks, support vector machines); unsupervised learning (clustering, dimensionality reduction, kernel methods); learning theory (bias/variance tradeoffs, practical advice); reinforcement learning and adaptive control.
- [Broderick: Machine Learning, MIT 6.036 Fall 2020](https://www.youtube.com/watch?v=ZOiBe-nrmc4); [Full lecture information and slides](http://tamarabroderick.com/ml.html)
  
</details>
 



