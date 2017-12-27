# https://www.youtube.com/watch?v=T0Myf8B0Dj8&feature=youtu.be
import requests
# import pickle
import pandas as pd
from bs4 import BeautifulSoup

from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from subprocess import check_call


def salary_currency(x):
    return x['currency'] if type(x) == dict else None


def salary_from(x):
    return x['from'] if type(x) == dict else None


def salary_gross(x):
    return x['gross'] if type(x) == dict else None


def salary_to(x):
    return x['to'] if type(x) == dict else None


def profarea_id(x):
    return x[0]['profarea_id'] if type(x) == list else None


def area_id(x):
    return x['id'] if type(x) == dict else None


def type_id(x):
    return x['id'] if type(x) == dict else None


def billing_type_id(x):
    return x['id'] if type(x) == dict else None


def employment_id(x):
    return x['id'] if type(x) == dict else None


def schedule_id(x):
    return x['id'] if type(x) == dict else None


def experience_id(x):
    return x['id'] if type(x) == dict else None


vac_url = 'https://api.hh.ru/vacancies/{}'
max_vac_id = 20000000
min_vac_id = 19998000

vac = []

for i in range(max_vac_id, min_vac_id, -1):
    print(vac_url.format(i))
    vac.append(requests.get(vac_url.format(i)).json())

df = pd.DataFrame(vac)

df['salary_currency'] = df['salary'].map(salary_currency)
df['salary_from'] = df['salary'].map(salary_from)
df['salary_gross'] = df['salary'].map(salary_gross)
df['salary_to'] = df['salary'].map(salary_to)
del df['salary']

print(set(df['salary_currency']))

df['profarea_id'] = df['specializations'].map(profarea_id)
del df['specializations']

print(set(df['profarea_id']))

# Удаляем ненужное
del df['accept_handicapped']
del df['accept_kids']
del df['alternate_url']

del df['apply_alternate_url']
del df['archived']

df['area_id'] = df['area'].map(area_id)
del df['area']

df['type_id'] = df['type'].map(area_id)
del df['type']

del df['address']

df['billing_type_id'] = df['billing_type'].map(area_id)
del df['billing_type']

del df['contacts']
del df['employer']

del df['driver_license_types']

del df['id']

del df['negotiations_url']
del df['relations']

df['employment_id'] = df['employment'].map(area_id)
del df['employment']

del df['site']

del df['created_at']
del df['published_at']

df['schedule_id'] = df['schedule'].map(area_id)
del df['schedule']

df['experience_id'] = df['experience'].map(area_id)
del df['experience']

del df['key_skills']
del df['test']
del df['suitable_resumes_url']

del df['errors']
del df['response_url']
del df['code']

df['all_text'] = [
    (x if type(x) == str else '  ') +
    (y if type(y) == str else '  ')
    for x, y in zip(
        df['name'],
        df['description']
    )
]
del df['name']
del df['description']

df['all_text'] = df['all_text'].map(
    lambda x: BeautifulSoup(x, "lxml").text
)

# Указана ли зарплата
df['with_salary'] = [
    1 if (x > 0) or (y > 0) else 0 for x, y in zip(
        df['salary_from'], df['salary_to']
    )
]

# Удаляем больше не нужные столбцы
del df['salary_currency'], df['salary_from'], df['salary_gross'], df['salary_to']
del df['branded_description']
del df['department']

texts = df['all_text']
del df['all_text']

df = pd.get_dummies(df)

y = df['with_salary']
del df['with_salary']

X_train, x_test, Y_train, y_test = train_test_split(df, y, test_size=0.3)

dtc = DecisionTreeClassifier(max_depth=10)
dtc.fit(X_train, Y_train)

export_graphviz(dtc, feature_names=X_train.columns, out_file='c:\\temp\\test.dot', filled=True)

check_call(
    ['C:\\Program Files (x86)\\Graphviz2.38\\bin\\dot.exe', '-Tpng', 'c:\\temp\\test.dot', '-o', 'c:\\temp\\test.png'])

rfc = RandomForestClassifier(n_estimators=100)
rfc.fit(X_train, Y_train)

predict = rfc.predict(x_test)

print(accuracy_score(y_test, predict))

# print(X_train)
# print(df.info())
# print(df.shape)
