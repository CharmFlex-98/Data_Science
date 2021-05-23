import numpy as np
import pandas as pd

data = pd.read_csv('test.csv')
print(data)

print(data.columns)

# slice index
print(data.iloc[0])
print(data.iloc[0:2])

# slice columns
print(data['PassengerId'])
print(data[['PassengerId', 'Pclass']])
print(data.loc[:, ['PassengerId', 'Pclass']])

# set index
data.set_index('Age', inplace=True)
data.sort_index(inplace=True, ascending=False)
print(data)

# reset index
data.reset_index(inplace=True)
print(data)

# filtering / masking
data = data.loc[(data['Age'] < 30) & (data['Pclass'] < 2), 'Fare']
print(data)

data = pd.read_csv('test.csv')

passengerid = [1200, 1201]
data = data.loc[data['PassengerId'].isin(passengerid), 'PassengerId']
print(data)

data = pd.read_csv('test.csv')

data = data.loc[data['Embarked'].str.contains('C', na=False), 'Embarked']
print(data)

# reset----------------------
data = pd.read_csv('test.csv')
# ----------------------

# change columns name
data.rename(lambda x: x.lower(), axis=1, inplace=True)
print(data)

data.columns = [x.upper() for x in data.columns]
print(data)

data.rename(columns={'PASSENGERID': 'PassengerId'}, inplace=True)
print(data)

# reset----------------------
data = pd.read_csv('test.csv')
# ----------------------

# change values in columns
data['Embarked'] = data['Embarked'].apply(lambda x: x.lower())
print(data)


def my_function(x):
    return x.upper()


data['Embarked'] = data['Embarked'].apply(my_function)
print(data)

data['Embarked'] = data['Embarked'].str.lower()
print(data)

data['Embarked'] = data['Embarked'].replace({'c': 'o'})
print(data)

data['PassengerId'] = data['PassengerId'].astype(str)
data['PassengerId'] = data['PassengerId'].str.extract('(\w)*')
print(data)
# drop columns
data.drop(columns=['Embarked', 'PassengerId'], inplace=True)
print(data)

# append dataframe
data = data.append(data, ignore_index=True)
print(data)

# drop index
data = data.drop(index=range(0, 600))
print(data)

data = data.drop(index=data[data['Pclass'] == 3].index)
print(data)

# sort
data.sort_values(['Pclass', 'Fare'], ascending=[True, False], inplace=True)
print(data)

# calculation
print(data.describe())
print(data.mean())
print(data.median())

print(data['Fare'].median())
print(data['Fare'].count())


def makenana(x):
    if x <= 20:
        x = None
    return x


data['Fare'] = data['Fare'].apply(makenana)
print(data['Fare'].count())
print(data['Fare'].value_counts())
print(data['Fare'].value_counts(normalize=True))
print(data['Fare'].agg(['mean', 'median']))

# reset----------------------
data = pd.read_csv('test.csv')
# ----------------------
print(data)
# multi_index
data.set_index(['Pclass', 'Fare'], inplace=True)
data.sort_index(ascending=[True, True], inplace=True)
print(data.head(50))

# group by method
data = pd.read_csv('test.csv')

data = data.groupby('Pclass').get_group(3)
print(data)

data = pd.read_csv('test.csv')

data = data.groupby('Pclass')['Fare'].agg(['median', 'mean'])
print(data)

data = pd.read_csv('test.csv')


def cal_fare_mean(group):
    avg = group['Fare'].mean()
    group['avg'] = avg

    return group


data = data.groupby('Pclass').apply(cal_fare_mean)
print(data.head(10))

data = data.groupby('Pclass').filter(lambda x: x['PassengerId'].mean() > 1100)
print(data)

# cleaning data
data = pd.read_csv('test.csv')
data.dropna(inplace=True)
print(data)

data = pd.read_csv('test.csv')
data.dropna(axis=0, how='any', inplace=True)
print(data)

data = pd.read_csv('test.csv')
data.dropna(axis=0, how='all', inplace=True)
print(data)

data = pd.read_csv('test.csv')
data.dropna(axis=0, how='all', subset=['Embarked', 'Cabin'], inplace=True)

print(data.isna())
print(data.fillna('fck u'))

data = pd.read_csv('test.csv', na_values=['Q', 'S'])
print(data)
data['Passengerid'] = data['PassengerId'].astype(int)
print(data)

# date time conversion
data['PassengerId'] = pd.to_datetime(data['PassengerId'], format='')
print(data)
data['day_in_week'] = data['PassengerId'].dt.day_name()
print(data)

# date time resample (something like group by) (must set date time as index)
data.set_index('PassengerId', inplace=True)
gg = data['Pclass'].resample('D').agg(['mean', 'median'])
print(gg)
ff = data.resample('D').agg({'Pclass': 'mean', 'Fare': 'median'})
print(ff)

# data.mean(), data.median combination
print(data.agg({'Fare': 'median', 'Pclass': 'mean'}))

# write to csv
data.to_csv('my_final.csv')

# merging
data = pd.merge(data, data, on='PassengerId', how='outer')
print(data.head(10))

data = pd.merge(data, data, how='inner', left_index=True, right_index=True)
print(data.head(10))

data = pd.concat([data, data], ignore_index=True)
print(data)

# categorial data
data = pd.read_csv('test.csv')
data['Embarked'] = data['Embarked'].astype('category')
my_cate = pd.CategoricalDtype(['C', 'S', 'Q'], ordered=True)
embarked = data['Embarked'].astype(my_cate)
print(embarked.head(10))

# pivot table
data = pd.read_csv('test.csv')
pt = data.pivot_table(values='Fare', index='Pclass', columns='Embarked', aggfunc=[np.mean])
print(pt)

my_data = pd.DataFrame([1, 2, None], index=['Ming', 'Soon', 'Chin'], columns=['Grade'])
print(my_data)

# groupby and transform

def my_f(x):
    x = 15
    return x


print(data.groupby('Pclass').agg({'Fare': my_f}))

print(data[['Pclass', 'Fare']].groupby('Pclass').transform(my_f))

print(data['PassengerId'].sort_values(ascending=False))