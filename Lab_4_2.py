import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re


def cost(df, path):
    res = 0
    for i in range(len(path) - 1):
        res += int(df.loc[path[i], path[i + 1]])

    res += int(df.loc[path[-1], path[0]])

    return res


df_dist = pd.read_csv('ha30_dist.txt', sep='\t', header=None)
df_dist = pd.DataFrame([x.strip().replace('    ', ' ').replace('   ', ' ').replace('  ', ' ').split(' ') for x in df_dist[0]])

df_coords = pd.read_csv('ha30_xyz.txt', sep='\t', decimal='.', header=None)
df_coords = pd.DataFrame([x.strip().replace('       ', ' ').replace('  ', ' ').split() for x in df_coords[0]])

xs = []
ys = []

for index, row in df_coords.iterrows():
    xs.append(float(row[0]))
    ys.append(float(row[1]))

plt.plot(xs, ys, 'o-')
plt.plot(xs[0], ys[0], 'ro')
plt.grid()
plt.show()

with open('ha30_name.txt') as f:
    names = f.read()

names = names.split('\n')

df_dist.columns = names
df_dist.index = names

df_test = df_dist.copy()

cities = list(df_test.columns)
start_city = cities[0]

df_coords.index = cities

cities = list(np.random.choice(cities, size=len(cities), replace=False))

cost_initial = cost(df_test, cities)
p = 0.05

best_path = ','.join(cities)
best_cost = cost_initial

for i in range(100000):

    pair_of_cities = np.random.choice(cities, size=2, replace=False)
    a, b = cities.index(pair_of_cities[0]), cities.index(pair_of_cities[1])
    cities[b], cities[a] = cities[a], cities[b]

    new_cost = cost(df_test, cities)

    if new_cost < cost_initial:
        cost_initial = new_cost
        if cost_initial < best_cost:
            best_path = ','.join(cities)
            best_cost = cost_initial

        continue
    else:
        p2 = np.random.uniform()
        if p2 < p:
            cost_initial = new_cost
            continue
        else:
            cities[a], cities[b] = cities[b], cities[a]

print('best path - ', best_path)
print('with cost - ', best_cost)

xs1 = []
ys1 = []

for city in best_path.split(","):
    xs1.append(float(df_coords.loc[city][0]))
    ys1.append(float(df_coords.loc[city][1]))

plt.plot(xs1, ys1, 'o-')
plt.plot(xs1[0], ys1[0], 'ro')
plt.grid()
plt.show()

cost_stats = []
cost_max = 0
max_path = ','.join(cities)

for i in range(100000):
    cities = list(np.random.choice(cities, size=len(cities), replace=False))
    cost_i = cost(df_test, cities)
    cost_stats.append(cost_i)

    if cost_i > cost_max:
        cost_max = cost_i
        max_path = ','.join(cities)

print('mean cost - ', np.mean(cost_stats))
print()
print('worst path - ', max_path)
print('with cost - ', cost_max)


np.median(cost_stats)

plt.figure(figsize=(12, 5))
plt.hist(cost_stats, bins=100)
plt.grid()
plt.xlabel('path cost')
plt.ylabel('number of paths')
plt.show()

np.quantile(cost_stats, 0.001)
np.min(cost_stats)

best_cost
best_path
cities
cost(df_test, cities)

i = ['title', 'email', 'password2', 'password1', 'first_name',
     'last_name', 'next', 'newsletter']
a, b = i.index('password2'), i.index('password1')
i[b], i[a] = i[a], i[b]

paths = {}

k = 0

while k != 15:
    k += 1
    cities = np.random.choice(cities, size=len(cities), replace=False)
    if ','.join(list(cities)) in paths:
        continue

    cost_i = cost(df_test, cities)

    if len(paths) == 0:
        paths[','.join(list(cities))] = cost(df_test, cities)

    if cost_i < max([x for x in paths.values()]):
        paths[','.join(list(cities))] = cost(df_test, cities)
        k = 0


max(paths, key=paths.get)

paths[min(paths, key=paths.get)]

np.random.choice(cities, size=len(cities), replace=False)
cost(df_test, cities)

start_city
df_test.loc['Azores', 'Berlin']

df_dist
