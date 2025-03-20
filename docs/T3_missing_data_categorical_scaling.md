[NIve example of pipeline for the Titanic data set](https://medium.com/analytics-vidhya/how-to-apply-preprocessing-steps-in-a-pipeline-only-to-specific-features-4e91fe45dfb8)



## Exercise: Montesinho burned area data set (with numerical and categorical variables)

Consider the dataset that described 517 fires from the Montesinho natural park in Portugal. For each incident weekday, month, coordinates, and the burnt area are recorded, as well as several meteorological data such as rain, temperature, humidity, and wind (https://www.kaggle.com/datasets/vikasukani/forest-firearea-datasets). For reference, a copy of the file is available [forestfires.csv](forestfires.csv). The variables are:

- X - x-axis spatial coordinate within the Montesinho park map: 1 to 9
- Y - y-axis spatial coordinate within the Montesinho park map: 2 to 9
- month - month of the year: "jan" to "dec"
- day - day of the week: "mon" to "sun"
- FFMC - FFMC index from the FWI system: 18.7 to 96.20
- DMC - DMC index from the FWI system: 1.1 to 291.3
- DC - DC index from the FWI system: 7.9 to 860.6
- ISI - ISI index from the FWI system: 0.0 to 56.10
- temp - the temperature in Celsius degrees: 2.2 to 33.30
- RH - relative humidity in %: 15.0 to 100
- wind - wind speed in km/h: 0.40 to 9.40
- rain - outside rain in mm/m2 : 0.0 to 6.4
- area - the burned area of the forest (in ha): 0.00 to 1090.84. IN fact we are going to convert this into a binary variable by asking if fires are larger than 5 ha

## Read data

- Read the data and create arrays `X`and `y`. You can discard the original grid coordinates `X,Y` and just keep attributes 'month', 'day', 'FFMC', 'DMC', 'DC', 'ISI', 'temp', 'RH', 'wind', 'rain' and the response variable 'area'.
- Try to fit a regression tree to the data. Do you get the error message `could not convert string to float: 'mar'`?

## Categorical variables and `get.dummies`
- Solve that problem with `pd.get_dummies`. What does this do?
- What happens if you use the `drop_first=True` option?
- And the `dtype=float` option?
  



- [sklearn Estimators that handle NaN values](https://scikit-learn.org/stable/modules/impute.html#estimators-that-handle-nan-values)

- [sklearn missing values support](https://scikit-learn.org/stable/modules/tree.html#missing-values-support)

- [pipelines](https://scikit-learn.org/stable/modules/compose.html#pipeline)

- [transforming targets: example](https://scikit-learn.org/stable/auto_examples/compose/plot_transformed_target.html#sphx-glr-auto-examples-compose-plot-transformed-target-py)

<details markdown="block">
<summary> Example with the Iris data set </summary>

```
import pandas as pd
# pre-processing, missing values
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, Normalizer

# pipeline
from sklearn.pipeline import Pipeline

# some models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# partition data
from sklearn.model_selection import train_test_split, GridSearchCV, RepeatedStratifiedKFold

# precision metrics
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, confusion_matrix
import matplotlib.pyplot as plt 

# read data
titanic = pd.read_csv('titanic.csv', delimiter=',',  )
titanic.columns=[x.lower() for x in titanic.columns]

categorical_features = ['pclass', 'sex', 'embarked']
categorical_transformer = Pipeline(
    [
        # ('imputer_cat', SimpleImputer(strategy = 'constant', fill_value = 'missing')),
        ('imputer_cat', SimpleImputer(strategy = 'most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown = 'ignore'))
    ]
)

numeric_features = ['age', 'sibsp', 'parch', 'fare']
numeric_transformer = Pipeline(
    [
        ('imputer_num', SimpleImputer(strategy = 'median')),
        #('scaler', StandardScaler())
        ('normalizer', Normalizer())
    ]
)

preprocessor = ColumnTransformer(
    [
        ('categoricals', categorical_transformer, categorical_features),
        ('numericals', numeric_transformer, numeric_features)
    ],
    remainder = 'drop' # By default, only the specified columns in transformers are transformed and combined in the output, and the non-specified columns are dropped.
)

pipeline = Pipeline(
    [
        ('preprocessing', preprocessor),
        ('clf', RandomForestClassifier(n_estimators=10)) #tree.DecisionTreeClassifier()) # LogisticRegression())
    ]
)

X = titanic.drop('survived', axis = 1)
y = titanic.survived

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, stratify=y)

pipeline.fit(X_train, y_train)

y_pred=pipeline.predict(X_test)

# confusion matrix
confmat = confusion_matrix(y_true=y_test, y_pred=y_pred) #actual, predicted
cm_display = ConfusionMatrixDisplay(confusion_matrix = confmat, display_labels = ['do not survive', 'do survive']) 
cm_display.plot()
plt.show() 
```

</details>
