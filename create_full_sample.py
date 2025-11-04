import pandas as pd
import numpy as np

print("Creating larger Titanic sample dataset...")

# Create a more substantial sample dataset (50 passengers)
np.random.seed(42)

n_passengers = 50

data = {
    'PassengerId': range(1, n_passengers + 1),
    'Survived': np.random.randint(0, 2, n_passengers),
    'Pclass': np.random.choice([1, 2, 3], n_passengers, p=[0.2, 0.3, 0.5]),
    'Sex': np.random.choice(['male', 'female'], n_passengers, p=[0.6, 0.4]),
    'Age': np.random.normal(30, 15, n_passengers).astype(int),
    'SibSp': np.random.poisson(0.5, n_passengers),
    'Parch': np.random.poisson(0.4, n_passengers),
    'Fare': np.random.exponential(50, n_passengers),
    'Embarked': np.random.choice(['S', 'C', 'Q'], n_passengers, p=[0.7, 0.2, 0.1])
}

# Adjust survival based on realistic patterns
df = pd.DataFrame(data)

# Women survive more
df.loc[df['Sex'] == 'female', 'Survived'] = np.random.choice([0, 1], len(df[df['Sex'] == 'female']), p=[0.3, 0.7])

# First class survives more  
df.loc[df['Pclass'] == 1, 'Survived'] = np.random.choice([0, 1], len(df[df['Pclass'] == 1]), p=[0.4, 0.6])

# Children survive more
df.loc[df['Age'] < 18, 'Survived'] = np.random.choice([0, 1], len(df[df['Age'] < 18]), p=[0.2, 0.8])

# Add some missing values
df.loc[np.random.choice(df.index, 5), 'Age'] = np.nan
df['Cabin'] = np.random.choice([None, 'C123', 'B45', 'D56', 'E12'], n_passengers, p=[0.7, 0.1, 0.1, 0.05, 0.05])

# Create names
titles = ['Mr.', 'Mrs.', 'Miss.', 'Master.', 'Dr.']
first_names = ['John', 'Mary', 'James', 'Elizabeth', 'William', 'Anna', 'Robert', 'Emily']
last_names = ['Smith', 'Johnson', 'Williams', 'Brown', 'Jones', 'Miller', 'Davis', 'Wilson']

names = []
for i in range(n_passengers):
    title = np.random.choice(titles, p=[0.4, 0.3, 0.2, 0.05, 0.05])
    first = np.random.choice(first_names)
    last = np.random.choice(last_names)
    names.append(f"{last}, {title} {first}")

df['Name'] = names

# Create tickets
df['Ticket'] = ['TKT' + str(i).zfill(6) for i in range(1000, 1000 + n_passengers)]

# Reorder columns
cols = ['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']
df = df[cols]

df.to_csv('titanic_large_sample.csv', index=False)
print(f"Created 'titanic_large_sample.csv' with {n_passengers} passengers")
print("\nRealistic survival patterns:")
print(f"- Overall survival rate: {df.Survived.mean():.1%}")
print(f"- Female survival rate: {df[df.Sex=='female'].Survived.mean():.1%}")
print(f"- 1st class survival rate: {df[df.Pclass==1].Survived.mean():.1%}")
print(f"- Child survival rate: {df[df.Age<18].Survived.mean():.1%}")