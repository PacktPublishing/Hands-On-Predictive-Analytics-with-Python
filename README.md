# Hands-On Predictive Analytics with Python

<a href="https://www.packtpub.com/big-data-and-business-intelligence/hands-predictive-analytics-python?utm_source=github&utm_medium=repository&utm_campaign=9781789138719"><img src="https://www.packtpub.com/sites/default/files/9781789138719_cover.png" height="256px" align="right"></a>

This is the code repository for [Hands-On Predictive Analytics with Python](https://www.packtpub.com/big-data-and-business-intelligence/hands-predictive-analytics-python?utm_source=github&utm_medium=repository&utm_campaign=9781789138719), published by Packt.

**Master the complete predictive analytics process, from problem definition to model deployment**

## What is this book about?
This book will teach you all the processes you need to build a predictive analytics solution: understanding the problem, preparing datasets, exploring relationships, model building, tuning, evaluation, and deployment. You'll earn to use Python and its data analytics ecosystem to implement the main techniques used in real-world projects.

This book covers the following exciting features: 
* Get to grips with the main concepts and principles of predictive analytics
* Learn about the stages involved in producing complete predictive analytics solutions
* Understand how to define a problem, propose a solution, and prepare a dataset
* Use visualizations to explore relationships and gain insights into the dataset
* Learn to build regression and classification models using scikit-learn
* Use Keras to build powerful neural network models that produce accurate predictions
* Learn to serve a model's predictions as a web application

If you feel this book is for you, get your [copy](https://www.amazon.com/dp/178913871X) today!

<a href="https://www.packtpub.com/?utm_source=github&utm_medium=banner&utm_campaign=GitHubBanner"><img src="https://raw.githubusercontent.com/PacktPublishing/GitHub/master/GitHub.png" 
alt="https://www.packtpub.com/" border="5" /></a>

## Instructions and Navigations

### Installation
To be able to run the code of the book without any problems, please do the following:
1. Download the Anaconda distribution for your system, you can find the installers [here](https://www.anaconda.com)
1. Once you have installed the Anaconda distribution, create a new Python 3.6 environment with the packages you will need.
To create the environment (named `ho-pawp`, but you can use any other name you like) run the following command
in the Anaconda Prompt terminal `conda create --name ho-pawp --file requirements.txt `

For a quick guide on conda refer to the conda-cheatsheet.pdf in this repo.
### Using the code files

All of the code is organized into folders. Most of the code consists of Jupyter Notebooks. For example, Chapter02.

The code will look like the following:
```
carat_values = np.arange(0.5, 5.5, 0.5)
preds = first_ml_model(carat_values)
pd.DataFrame({"Carat": carat_values, "Predicted price":preds})
```

**Following is what you need for this book:**
This book is aimed at data scientists, data engineers, software engineers, and business analysts. Also, students and professionals who are constantly working with data in quantitative fields such as finance, economics, and business, among others, who would like to build models to make predictions will find this book useful. In general, this book is aimed at all professionals who would like to focus on the practical implementation of predictive analytics with Python.

With the following software and hardware list you can run all code files present in the book (Chapter 1-13).
### Software and Hardware List
| Chapter | Software required                     | OS required                         |
| ------- | ------------------------------------  | ----------------------------------- |
| 1-9     | Python 3.6 or higher, Jupyter Notebook, Recent versions of the following Python libraries: NumPy, pandas, and matplotlib, Seaborn, scikit-learn, Recent installations of TensorFlow and Keras, Basic libraries for Dash | Windows, Mac OS X, and Linux (Any) |

We also provide a PDF file that has color images of the screenshots/diagrams used in this book. [Click here to download it]().

### Related products
* TensorFlow: Powerful Predictive Analytics with TensorFlow [[Packt]](https://www.packtpub.com/big-data-and-business-intelligence/tensorflow-powerful-predictive-analytics-tensorflow?utm_source=github&utm_medium=repository&utm_campaign=9781789136913) [[Amazon]](https://www.amazon.com/dp/1789136911)

* Building Machine Learning Systems with Python - Third Edition [[Packt]](https://www.packtpub.com/big-data-and-business-intelligence/building-machine-learning-systems-python-third-edition?utm_source=github&utm_medium=repository&utm_campaign=9781788623223) [[Amazon]](https://www.amazon.com/dp/1788623223)


## Get to Know the Author
**Alvaro Fuentes** is a Senior Data Scientist with more than 13 years of experience in analytical roles.
He holds an M.S. in applied mathematics and an M.S. in quantitative economics. He has been working for one of the top global
management consulting firms solving analytical and AI problems in different industries like Banking, Telco, Mining and others.
He worked for many years in the Central Bank of Guatemala as an economic analyst, building models for economic and financial data.
He is a big Python fan and has been using it routinely for 5+ years to analyzing data and building and deploying analytical models that transform data into intelligence.


### Suggestions and Feedback
[Click here](https://docs.google.com/forms/d/e/1FAIpQLSdy7dATC6QmEL81FIUuymZ0Wy9vH1jHkvpY57OiMeKGqib_Ow/viewform) if you have any feedback or suggestions.


