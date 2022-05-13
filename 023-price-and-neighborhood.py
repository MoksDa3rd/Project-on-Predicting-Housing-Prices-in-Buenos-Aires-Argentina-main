#!/usr/bin/env python
# coding: utf-8

# <font size="+3"><strong>Predicting Price with Neighborhood</strong></font>

# In[20]:


import warnings
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wqet_grader
from category_encoders import OneHotEncoder
from IPython.display import VimeoVideo
from sklearn.linear_model import LinearRegression, Ridge  # noqa F401
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import make_pipeline
from sklearn.utils.validation import check_is_fitted

warnings.simplefilter(action="ignore", category=FutureWarning)
wqet_grader.init("Project 2 Assessment")


# In the last lesson, we created a model that used location — represented by latitude and longitude — to predict price. In this lesson, we're going to use a different representation for location: neighborhood. 

# In[21]:


VimeoVideo("656790491", h="6325554e55", width=600)


# # Prepare Data

# ## Import

# In[22]:


def wrangle(filepath):
    # Read CSV file
    df = pd.read_csv(filepath)

    # Subset data: Apartments in "Capital Federal", less than 400,000
    mask_ba = df["place_with_parent_names"].str.contains("Capital Federal")
    mask_apt = df["property_type"] == "apartment"
    mask_price = df["price_aprox_usd"] < 400_000
    df = df[mask_ba & mask_apt & mask_price]

    # Subset data: Remove outliers for "surface_covered_in_m2"
    low, high = df["surface_covered_in_m2"].quantile([0.1, 0.9])
    mask_area = df["surface_covered_in_m2"].between(low, high)
    df = df[mask_area]

    # Split "lat-lon" column
    df[["lat", "lon"]] = df["lat-lon"].str.split(",", expand=True).astype(float)
    df.drop(columns="lat-lon", inplace=True)

    # Extract "Neighborhood"
    df["neighborhood"] = df["place_with_parent_names"].str.split("|", expand=True)[3]
    
    df.drop(columns="place_with_parent_names", inplace=True)

    return df


# In the last lesson, we used our `wrangle` function to import two CSV files as DataFrames. But what if we had hundreds of CSV files to import? Wrangling them one-by-one wouldn't be an option. So let's start with a technique for reading several CSV files into a single DataFrame. 
# 
# The first step is to gather the names of all the files we want to import. We can do this using pattern matching. 

# In[23]:


VimeoVideo("656790237", h="1502e3765a", width=600)


# **Task 2.3.1:** Use [`glob`](https://docs.python.org/3/library/glob.html#glob.glob) to create a list that contains the filenames for all the Buenos Aires real estate CSV files in the `data` directory. Assign this list to the variable name `files`.
# 
# - [<span id='technique'>Assemble a list of path names that match a pattern in <span id='tool'>glob.](../%40textbook/02-python-advanced.ipynb#Working-with-strings-)

# In[24]:


files = glob("data/buenos-aires-real-estate-*.csv")
files = ['data/buenos-aires-real-estate-1.csv',
 'data/buenos-aires-real-estate-2.csv',
 'data/buenos-aires-real-estate-3.csv',
 'data/buenos-aires-real-estate-4.csv',
 'data/buenos-aires-real-estate-5.csv']
files


# In[25]:


# Check your work
assert len(files) == 5, f"`files` should contain 5 items, not {len(files)}"


# The next step is to read each of the CSVs in `files` into a DataFrame, and put all of those DataFrames into a list. What's a good way to iterate through `files` so we can do this? A `for` loop!

# In[26]:


VimeoVideo("656789768", h="3b8f3bca0b", width=600)


# **Task 2.3.2:** Use your `wrangle` function in a `for` loop to create a list named `frames`. The list should the cleaned DataFrames created from the CSV filenames your collected in `files`.
# 
# - [What's a <span id='term'>for loop</span>?](../%40textbook/01-python-getting-started.ipynb#Python-for-Loops)
# - [<span id='technique'>Write a for loop in <span id='tool'>Python.](../%40textbook/01-python-getting-started.ipynb#Working-with-for-Loops)

# In[27]:


frames = []
for file in files:
    df = wrangle(file)
    frames.append(df)


# In[28]:


# Check your work
assert len(frames) == 5, f"`frames` should contain 5 items, not {len(frames)}"
assert all(
    [isinstance(frame, pd.DataFrame) for frame in frames]
), "The items in `frames` should all be DataFrames."


# The final step is to use pandas to combine all the DataFrames in `frames`. 

# In[29]:


VimeoVideo("656789700", h="57adef4afe", width=600)


# **Task 2.3.3:** Use [`pd.concat`](https://pandas.pydata.org/docs/reference/api/pandas.concat.html) to concatenate the items in `frames` into a single DataFrame `df`. Make sure you set the `ignore_index` argument to `True`.
# 
# - [<span id='technique'>Concatenate two or more DataFrames using <span id='tool'>pandas.](../%40textbook/03-pandas-getting-started.ipynb#Concatenating-DataFrames)

# In[30]:


df = pd.concat(frames, ignore_index=True)
df.head()


# In[31]:


# Check your work
assert len(df) == 6582, f"`df` is the wrong size: {len(df)}."


# Excellent work! You can now clean and combine as many CSV files as your computer can handle. You're well on your way to working with big data. 📈

# ## Explore

# Looking through the output from the `df.head()` call above, there's a little bit more cleaning we need to do before we can work with the neighborhood information in this dataset. The good news is that, because we're using a `wrangle` function, we only need to change the function to re-clean all of our CSV files. This is why functions are so useful.

# In[32]:


VimeoVideo("656791659", h="581201dc92", width=600)


# **Task 2.3.4:** Modify your `wrangle` function to create a new feature `"neighborhood"`. You can find the neighborhood for each property in the `"place_with_parent_names"` column. For example, a property with the place name `"|Argentina|Capital Federal|Palermo|"` is located in the neighborhood is `"Palermo"`. Also, your function should drop the `"place_with_parent_names"` column.
# 
# Be sure to rerun all the cells above before you continue.
# 
# - [<span id='technique'>Split the strings in one column to create another using <span id='tool'>pandas.](../%40textbook/03-pandas-getting-started.ipynb#Splitting-Strings)

# In[33]:


# Check your work
assert df.shape == (6582, 17), f"`df` is the wrong size: {df.shape}."
assert (
    "place_with_parent_names" not in df
), 'Remember to remove the `"place_with_parent_names"` column.'


# ## Split

# At this point, you should feel more comfortable with the splitting data, so we're going to condense the whole process down to one task. 

# In[34]:


VimeoVideo("656791577", h="0ceb5341f8", width=600)


# **Task 2.3.5:** Create your feature matrix `X_train` and target vector `y_train`. `X_train` should contain one feature: `"neighborhood"`. Your target is `"price_aprox_usd"`. 
# 
# - [What's a <span id='term'>feature matrix?](../%40textbook/15-ml-regression.ipynb#Linear-Regression)
# - [What's a <span id='term'>target vector?](../%40textbook/15-ml-regression.ipynb#Linear-Regression)
# - [<span id='technique'>Subset a DataFrame by selecting one or more columns in <span id='tool'>pandas.](../%40textbook/04-pandas-advanced.ipynb#Subset-a-DataFrame-by-Selecting-One-or-More-Columns) 
# - [<span id='technique'>Select a Series from a DataFrame in <span id='tool'>pandas.](../%40textbook/04-pandas-advanced.ipynb#Select-a-Series-from-a-DataFrame) 

# In[35]:


target = "price_aprox_usd"
features = ["neighborhood"]
X_train = df[features]
y_train = df[target]


# In[36]:


# Check your work
assert X_train.shape == (6582, 1), f"`X_train` is the wrong size: {X_train.shape}."
assert y_train.shape == (6582,), f"`y_train` is the wrong size: {y_train.shape}."


# # Build Model

# ## Baseline

# Let's also condense the code we use to establish our baseline. 

# In[37]:


VimeoVideo("656791443", h="120a740cc3", width=600)


# **Task 2.3.6:** Calculate the baseline mean absolute error for your model.
# 
# - [<span id='term'>What's a performance metric?](../%40textbook/12-ml-core.ipynb#Performance-Metrics)
# - [<span id='term'>What's mean absolute error?](../%40textbook/12-ml-core.ipynb#Performance-Metrics)
# - [<span id='technique'>Calculate summary statistics for a DataFrame or Series in <span id='tool'>pandas.](../%40textbook/05-pandas-summary-statistics.ipynb#Working-with-Summary-Statistics)
# - [<span id='technique'>Calculate the mean absolute error for a list of predictions in <span id='tool'>scikit-learn.](../%40textbook/15-ml-regression.ipynb#Calculating-the-Mean-Absolute-Error-for-a-List-of-Predictions)

# In[38]:


y_mean = y_train.mean()
y_pred_baseline = [y_mean] * len(y_train)

print("Mean apt price:", mean_absolute_error(y_train, y_pred_baseline))

print("Baseline MAE:", y_mean)


# The mean apartment price and baseline MAE should be similar but not identical to last lesson. The numbers will change since we're working with more data.

# ## Iterate

# If you try to fit a `LinearRegression` predictor to your training data at this point, you'll get an error that looks like this:
# 
# ```
# ValueError: could not convert string to float
# ```
# 
# What does this mean? When you fit a linear regression model, you're asking scikit-learn to perform a mathematical operation. The problem is that our training set contains neighborhood information in non-numerical form. In order to create our model we need to **encode** that information so that it's represented numerically. The good news is that there are lots of transformers that can do this. Here, we'll use the one from the [Category Encoders](https://contrib.scikit-learn.org/category_encoders/index.html) library, called a [`OneHotEncoder`](https://contrib.scikit-learn.org/category_encoders/onehot.html).
# 
# Before we build include this transformer in our pipeline, let's explore how it works. 

# In[39]:


VimeoVideo("656792790", h="4097efb40d", width=600)


# **Task 2.3.7:** First, instantiate a `OneHotEncoder` named `ohe`. Make sure to set the `use_cat_names` argument to `True`. Next, fit your transformer to the feature matrix `X_train`. Finally, use your encoder to transform the feature matrix `X_train`, and assign the transformed data to the variable `XT_train`.
# 
# - [What's <span id='term'>one-hot encoding?](../%40textbook/13-ml-data-pre-processing-and-production.ipynb#One-Hot-Encoding)
# - [<span id='technique'>Instantiate a transformer in <span id='tool'>scikit-learn.](../%40textbook/13-ml-data-pre-processing-and-production.ipynb#One-Hot-Encoding)
# - [<span id='technique'>Fit a transformer to training data in <span id='tool'>scikit-learn.](../%40textbook/13-ml-data-pre-processing-and-production.ipynb#One-Hot-Encoding)
# - [<span id='technique'>Transform data using a transformer in <span id='tool'>scikit-learn.](../%40textbook/13-ml-data-pre-processing-and-production.ipynb#One-Hot-Encoding)

# In[40]:


ohe = OneHotEncoder(use_cat_names=True)
ohe.fit(X_train)
XT_train = ohe.transform(X_train)
print(XT_train.shape)
XT_train.head()


# In[41]:


# Check your work
assert XT_train.shape == (6582, 57), f"`XT_train` is the wrong shape: {XT_train.shape}"


# Now that we have an idea for how the `OneHotEncoder` works, let's bring it into our pipeline.

# In[42]:


VimeoVideo("656792622", h="0b9d189e8f", width=600)


# **Task 2.3.8:** Create a pipeline named `model` that contains a `OneHotEncoder` transformer and a `LinearRegression` predictor. Then fit your model to the training data. 
# 
# - [What's a <span id='term'>pipeline?](../%40textbook/13-ml-data-pre-processing-and-production.ipynb#scikit-learn-in-Production)
# - [<span id='technique'>Create a pipeline in <span id='tool'>scikit-learn.](../%40textbook/13-ml-data-pre-processing-and-production.ipynb#Creating-a-Pipeline-in-scikit-learn)

# In[50]:


model = make_pipeline(
    OneHotEncoder(use_cat_names=True),
    Ridge()
)
model.fit(X_train, y_train)


# In[44]:


X_train.head()


# In[45]:


# Check your work
check_is_fitted(model[-1])


# Wow, you just built a model with two transformers and a predictor! When you started this course, did you think you'd be able to do something like that? 😁

# ## Evaluate

# Regardless of how you build your model, the evaluation step stays the same. Let's see how our model performs with the training set.

# In[46]:


VimeoVideo("656792525", h="09edc1c3d6", width=600)


# **Task 2.3.9:** First, create a list of predictions for the observations in your feature matrix `X_train`. Name this list `y_pred_training`. Then calculate the training mean absolute error for your predictions in `y_pred_training` as compared to the true targets in `y_train`.
# 
# - [<span id='technique'>Generate predictions using a trained model in <span id='tool'>scikit-learn.](../%40textbook/15-ml-regression.ipynb#Generating-Predictions-Using-a-Trained-Model)
# - [<span id='technique'>Calculate the mean absolute error for a list of predictions in <span id='tool'>scikit-learn.](../%40textbook/15-ml-regression.ipynb#Calculating-the-Mean-Absolute-Error-for-a-List-of-Predictions)

# In[47]:


y_pred_training = model.predict(X_train)
mae_training = mean_absolute_error(y_train, y_pred_training)
print("Training MAE:", round(mae_training, 2))


# Now let's check our test performance. 

# **Task 2.3.10:** Run the code below to import your test data `buenos-aires-test-features.csv` into a DataFrame and generate a Series of predictions using your model. Then run the following cell to submit your predictions to the grader.
# 
# - [What's generalizability?](../%40textbook/12-ml-core.ipynb#Generalization)
# - [<span id='technique'>Generate predictions using a trained model in <span id='tool'>scikit-learn.](../%40textbook/15-ml-regression.ipynb#Generating-Predictions-Using-a-Trained-Model)
# - [<span id='technique'>Calculate the mean absolute error for a list of predictions in <span id='tool'>scikit-learn.](../%40textbook/15-ml-regression.ipynb#Calculating-the-Mean-Absolute-Error-for-a-List-of-Predictions)

# In[51]:


X_test = pd.read_csv("data/buenos-aires-test-features.csv")[features]
y_pred_test = pd.Series(model.predict(X_test))
y_pred_test.head()


# In[52]:


wqet_grader.grade("Project 2 Assessment", "Task 2.3.10", y_pred_test)


# # Communicate Results

# If we write out the equation for our model, it'll be too big to fit on the screen. That's because, when we used the `OneHotEncoder` to encode the neighborhood data, we created a much wider DataFrame, and each column/feature has it's own coefficient in our model's equation.
# 
# <center><img src="../images/proj-2.006.png" alt="Equation: y = β0 + β1 x1 + β2 x2 + ... + β59 x59 + β60 x60 " style="width: 800px;"/></center>
# 
# This is important to keep in mind for two reasons. First, it means that this is a **high-dimensional** model. Instead of a 2D or 3D plot, we'd need a 58-dimensional plot to represent it, which is impossible! Second, it means that we'll need to extract and represent the information for our equation a little differently than before. Let's start by getting our intercept and coefficient.

# In[12]:


VimeoVideo("656793909", h="fca67856b4", width=600)


# **Task 2.3.11:** Extract the intercept and coefficients for your model. 
# 
# - [What's an <span id='term'>intercept</span> in a linear model?](../%40textbook/12-ml-core.ipynb#Model-Types)
# - [What's a <span id='term'>coefficient</span> in a linear model?](../%40textbook/12-ml-core.ipynb#Model-Types)
# - [<span id='technique'>Access an object in a pipeline in <span id='tool'>scikit-learn.](../%40textbook/13-ml-data-pre-processing-and-production.ipynb#Accessing-an-Object-in-a-Pipeline)

# In[54]:


intercept = model.named_steps["ridge"].intercept_.round()
coefficients = model.named_steps["ridge"].coef_.round()
print("coefficients len:", len(coefficients))
print(coefficients[:5])  # First five coefficients


# In[55]:


# Check your work
assert isinstance(
    intercept, float
), f"`intercept` should be a `float`, not {type(intercept)}."
assert isinstance(
    coefficients, np.ndarray
), f"`coefficients` should be a `float`, not {type(coefficients)}."
assert coefficients.shape == (
    57,
), f"`coefficients` is wrong shape: {coefficients.shape}."


# We have the values of our coefficients, but how do we know which features they belong to? We'll need to get that information by going into the part of our pipeline that did the encoding.

# In[13]:


VimeoVideo("656793812", h="810161b84e", width=600)


# **Task 2.3.12:** Extract the feature names of your encoded data from the `OneHotEncoder` in your model.
# 
# - [Access an object in a pipeline in scikit-learn.](../%40textbook/13-ml-data-pre-processing-and-production.ipynb#Accessing-an-Object-in-a-Pipeline)

# In[56]:


feature_names = model.named_steps["onehotencoder"].get_feature_names()
print("features len:", len(feature_names))
print(feature_names[:5])  # First five feature names


# In[57]:


# Check your work
assert isinstance(
    feature_names, list
), f"`features` should be a `list`, not {type(features)}."
assert len(feature_names) == len(
    coefficients
), "You should have the same number of features and coefficients."


# We have coefficients and feature names, and now we need to put them together. For that, we'll use a Series.

# In[14]:


VimeoVideo("656793718", h="1e2a1e1de8", width=600)


# **Task 2.3.13:** Create a pandas Series named `feat_imp` where the index is your `features` and the values are your `coefficients`.
# 
# - [<span id='technique'>Create a Series in <span id='tool'>pandas.](../%40textbook/03-pandas-getting-started.ipynb#Working-with-Columns)

# In[60]:


feat_imp = pd.Series(coefficients, index=feature_names)
feat_imp.head()


# In[61]:


# Check your work
assert isinstance(
    feat_imp, pd.Series
), f"`feat_imp` should be a `float`, not {type(feat_imp)}."
assert feat_imp.shape == (57,), f"`feat_imp` is wrong shape: {feat_imp.shape}."
assert all(
    a == b for a, b in zip(sorted(feature_names), sorted(feat_imp.index))
), "The index of `feat_imp` should be identical to `features`."


# To be clear, it's definitely not a good idea to show this long equation to an audience, but let's print it out just to check our work. Since there are so many terms to print, we'll use a `for` loop.

# In[15]:


VimeoVideo("656797021", h="dc90e6dac3", width=600)


# **Task 2.3.14:** Run the cell below to print the equation that your model has determined for predicting apartment price based on longitude and latitude.
# 
# - [What's an f-string?](../%40textbook/02-python-advanced.ipynb#Working-with-f-strings-)

# In[62]:


print(f"price = {intercept.round(2)}")
for f, c in feat_imp.items():
    print(f"+ ({round(c, 2)} * {f})")


# <div class="alert alert-block alert-warning">
# <b>Warning:</b> In the first lesson for this project, we said that you shouldn't make any changes to your model after you see your test metrics. That's still true. However, we're breaking that rule here so that we can discuss overfitting. In future lessons, you'll learn how to protect against overfitting without checking your test set.
# </div>

# In[16]:


VimeoVideo("656799309", h="a7130deb64", width=600)


# **Task 2.3.15:** Scroll up, change the predictor in your model to `Ridge`, and retrain it. Then evaluate the model's training and test performance. Do you still have an overfitting problem? If not, extract the intercept and coefficients again (you'll need to change your code a little bit) and regenerate the model's equation. Does it look different than before?
# 
# - What's <span id='term'>overfitting?
# - What's <span id='term'>regularization?
# - What's <span id='term'>ridge regression?

# In[63]:


# Check your work
assert isinstance(
    model[-1], Ridge
), "Did you retrain your model using a `Ridge` predictor?"


# We're back on track with our model, so let's create a visualization that will help a non-technical audience understand what the most important features for our model in predicting apartment price. 

# In[17]:


VimeoVideo("656798530", h="9a9350eff1", width=600)


# **Task 2.3.16:** Create a horizontal bar chart that shows the top 15 coefficients for your model, based on their absolute value.
# 
# - [What's a <span id='term'>bar chart</span>?](../%40textbook/07-visualization-pandas.ipynb#Bar-Charts)
# - [<span id='technique'>Create a bar chart using <span id='tool'>pandas</span></span>.](../%40textbook/07-visualization-pandas.ipynb#Bar-Charts)

# In[67]:


feat_imp.sort_values(key=abs).tail(15).plot(
    kind="barh",
    xlabel="Importance [USD]",
    ylabel="Features",
    title="Feature Importance for Apaetment Price");


# Looking at this bar chart, we can see that the poshest neighborhoods in Buenos Aires like [Puerto Madero](https://en.wikipedia.org/wiki/Puerto_Madero) and [Recoleta](https://en.wikipedia.org/wiki/Recoleta,_Buenos_Aires) increase the predicted price of an apartment, while more working-class neighborhoods like [Villa Soldati](https://en.wikipedia.org/wiki/Villa_Soldati) and [Villa Lugano](https://en.wikipedia.org/wiki/Villa_Lugano) decrease the predicted price. 
# 
# Just for fun, check out [this song](https://www.youtube.com/watch?v=RGlunBDvsaw) by Kevin Johansen about Puerto Madero. 🎶

# ---
# Copyright © 2022 WorldQuant University. This
# content is licensed solely for personal use. Redistribution or
# publication of this material is strictly prohibited.
# 
