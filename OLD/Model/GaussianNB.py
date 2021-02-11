import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

iris=pd.read_csv("iris.csv")
X = iris['Sepal.Length'].values.reshape(-1,1)
Y = iris['Sepal.Width'].values.reshape(-1,1)
train, test = train_test_split(iris, test_size = 0.25)

train_X = train[['Sepal.Length', 'Sepal.Width', 'Petal.Length','Petal.Width']]
train_y = train.Species
test_X = test[['Sepal.Length', 'Sepal.Width', 'Petal.Length','Petal.Width']]
test_y = test.Species

model = GaussianNB()
model.fit(train_X,train_y)


inputt=[int(x) for x in "45 32 60".split(' ')]
final=[np.array(inputt)]

b = log_reg.predict_proba(final)


pickle.dump(log_reg,open('model.pkl','wb'))
model=pickle.load(open('model.pkl','rb'))


# flower_example = {"Sepal_Length":5.9,"Sepal_Width":3,"Petal_Length":5.1,"Petal_Width":1.8}
# s_len=flower_example["Sepal_Length"]
# s_wid=flower_example["Sepal_Width"]
# p_len=flower_example["Petal_Length"]
# p_wid=flower_example["Petal_Width"]
# flower=[[s_len,s_wid,p_len,p_wid]]

# model.predict(flower)
# print(model.predict(flower))
