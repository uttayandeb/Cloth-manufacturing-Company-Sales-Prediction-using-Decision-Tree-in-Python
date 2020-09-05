#A cloth manufacturing company is interested to know about the segment or attributes causes high sale. 
#A decision tree can be built with target variable Sale 



#### Importing packages and loading dataset ############
import pandas as pd
import matplotlib.pyplot as plt
Company_Data = pd.read_csv("C:\\Users\\home\\Desktop\\Data Science Assignments\\Python_codes\\Desicion_tree\\Company_Data.csv")
Company_Data.head()



Company_Data.isnull().any()#to check if we have null valuesin dataset or not
#so there are no null values in the dataset


Company_Data.dtypes# to check the data types


Company_Data.describe()### to check the summary of the dataset

Company_Data.info()


Company_Data['ShelveLoc'],class_names = pd.factorize(Company_Data['ShelveLoc'])
Company_Data.ShelveLoc
print(class_names)

Company_Data['Urban'],class_names2=pd.factorize(Company_Data['Urban'])
print(class_names2)

Company_Data['US'].replace(to_replace=['Yes', 'No'],value= ['0', '1'], inplace=True)
Company_Data.US


print(Company_Data['US'].unique())
Company_Data.info()

Company_Data['Sales'].plot.hist()
plt.show()

Company_Data.Sales.describe()#count    400.000000
                             #mean       7.496325
                             #std        2.824115
                             #min        0.000000
                             #25%        5.390000
                             #50%        7.490000
                             #75%        9.320000
                             #max       16.270000
                             #Name: Sales, dtype: float64


#Converting the Sales column which is continuous into categorical
category = pd.cut(Company_Data.Sales,bins=[0,5.39,9.32,17],labels=['low','moderate','high'])
Company_Data.insert(0,'Sales_Group',category)

Company_Data.drop(['Sales'],axis = 1, inplace = True)

import seaborn as sns
sns.pairplot(Company_Data)


Company_Data['Sales_Group'].unique()
Company_Data.Sales_Group.value_counts()



######### Features Selection ##########


colnames = list(Company_Data.columns)
colnames
predictors = colnames[1:]#excluding 1st column all
predictors        #['CompPrice',
                  #'Income',
                  #'Advertising',
                  #'Population',
                  # 'Price',
                  # 'ShelveLoc',
                  # 'Age',
                  # 'Education',
                  # 'Urban',
                  # 'US'],           feature variables

target = colnames[0]#only 1st column
target            #'Sales_Group',           target variable


########## Splitting data into training and testing data set #############

import numpy as np

# np.random.uniform(start,stop,size) will generate array of real numbers with size = size
#Company_Data['is_train'] = np.random.uniform(0, 1, len(Company_Data))<= 0.75
#Company_Data['is_train']
#train,test = Company_Data[Company_Data['is_train'] == True],Company_Data[Company_Data['is_train']==False]
#train
#train.shape #(302, 12)
#test
#test.shape  #(98, 12)

#### another way to split data into train and test

from sklearn.model_selection import train_test_split
train,test = train_test_split(Company_Data,test_size = 0.2)
test
test.shape # (80, 11)
train
train.shape #(320, 11)







##############  Decision Tree Model building ###############

from sklearn.tree import  DecisionTreeClassifier
help(DecisionTreeClassifier)

model = DecisionTreeClassifier(criterion = 'entropy')

model.fit(train[predictors],train[target])

preds = model.predict(test[predictors])

preds

from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
print("Accuracy:",metrics.accuracy_score(test[target], preds))



pd.Series(preds).value_counts()
#moderate    43
#low         19
#high        18
#dtype: int64

pd.crosstab(test[target],preds)
#col_0        high  low  moderate
#Sales_Group                     
#low             0    7        12
#moderate        6   11        24
#high           12    1         6


# Accuracy = train 
np.mean(train.Sales_Group == model.predict(train[predictors]))#1.0

# Accuracy = Test
np.mean(preds==test.Sales_Group) #  0.5375
 
model.score(test[predictors],test[target])








from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO  #:pip install --upgrade scikit-learn==0.23.1
from IPython.display import Image  
import pydotplus
import io



dot_data = StringIO()
export_graphviz(model, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True,feature_names = predictors,class_names=['low','moderate','high'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png('Sales.png')
Image(graph.create_png())






