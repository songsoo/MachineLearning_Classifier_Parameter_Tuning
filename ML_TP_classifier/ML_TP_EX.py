import pandas as pd
from sklearn.model_selection import train_test_split

import ML_TP_auto
pd.set_option('display.max_columns',None)

data = pd.read_csv("online_shoppers_intention.csv")

#numerical_columns = [['Administrative','Administrative_Duration','Informational','Informational_Duration','ProductRelated','ProductRelated_Duration','BounceRates','ExitRates','PageValues','SpecialDay']]
#categorical_columns = [['Month','OperatingSystems','Browser','Region','TrafficType','VisitorType','Weekend','Revenue']]

numerical_columns = [['Administrative','Administrative_Duration'],['Administrative','Administrative_Duration','ProductRelated','ProductRelated_Duration'],['Administrative','Administrative_Duration','ProductRelated','ProductRelated_Duration','PageValues']]
categorical_columns = [['Month','OperatingSystems'],['Month','OperatingSystems','Weekend'],['Month','OperatingSystems','Weekend']]

y = data.loc[:,'Revenue']
x = data.loc[:,['Administrative','Administrative_Duration','Informational','Informational_Duration','ProductRelated','ProductRelated_Duration','BounceRates','ExitRates','PageValues','SpecialDay','Month','OperatingSystems','Browser','Region','TrafficType','VisitorType','Weekend','Revenue']]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=1 / 100, shuffle=True)

result = ML_TP_auto.get_Result(x_train,y_train,x_test,y_test,numerical_columns,categorical_columns)
print("Score :" ,result)

