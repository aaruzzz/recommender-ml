from math import sin, cos, sqrt, atan2, radians
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import csv
import numpy as np

visited_users = []
place = []
xc = []
yc = []
dist = []

# Read csv file and store data as list
with open('final_dataset.csv') as csvfile:
    csvReader = csv.reader(csvfile)
    next(csvReader)
    for row in csvReader:
        visited_users.append(row[6])
        place.append(row[1])
        xc.append(row[2])
        yc.append(row[3])
array = np.stack((visited_users, place), axis=1)

# USER INPUTS
gps = (27.713490563573057, 85.31674854544939)
place_user_likes = "Charumati Vihara Stupa"
user_age = 25

# Sort and print places acc. to visited_users
x = sorted(array, key=lambda x: int(x[0]), reverse=True)
print("Recommendations based on popularity:")
for i in x[:5]:
    print(i)

# Calculating distance to sort and print places acc. to incremental order
R = 6373.0
lat2 = radians(gps[0])
lon2 = radians(gps[1])
array1 = np.stack((xc, yc), axis=1)

for j, k in array1:
    lat1 = radians(float(j))
    lon1 = radians(float(k))
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = (sin(dlat/2))**2 + cos(lat1) * cos(lat2) * (sin(dlon/2))**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    distance = float("{:.2f}".format(R * c))
    dist.append(distance)

array2 = np.stack((dist, place), axis=1)
y = sorted(array2, key=lambda y: float(y[0]))
print("Recommendations based by distance:")
for j in y[:5]:
    print(j)

# Special Recommendation based on User Category and age

# helper functions


def get_place_from_index(index):
    return df[df.index == index]["place"].values[0]


def get_index_from_place(place):
    return df[df.place == place]["index"].values[0]


# Fetch csv
df = pd.read_csv("final_dataset.csv")

# Select features
# Categories: Historic, Architectural Landmark, Religious, Point of Interest, Fun, Monumental. Explore, Natural, Adventure, Peace
features = ['difficulty', 'category']

# Column is created in DF which combines all selected features
for feature in features:
    df[feature] = df[feature].fillna('')


def combine_features(row):
    try:
        return row['difficulty'] + " "+row['category']
    except:
        print("Error:", row)


df["combined_features"] = df.apply(combine_features, axis=1)
# print(df["combined_features"].head())
# print(df.column)

cv = CountVectorizer()  # count matrix created
count_matrix = cv.fit_transform(df["combined_features"])

# Cosine Similarity calculates(based on the count_matrix)
cosine_sim = cosine_similarity(count_matrix)

#Preferences = Easiest, Easy, Intermediate, Hard, Hardest, Godlevel
if ((user_age < 10) or (user_age >= 60)):
    user_preference = "Easiest"
elif (((user_age >= 10) and (user_age < 20)) or ((user_age >= 50) and (user_age < 60))):
    user_preference = "Easy"
elif ((user_age >= 20) and (user_age < 30)):
    user_preference = "Godlevel"
elif ((user_age >= 30) and (user_age < 35)):
    user_preference = "Hardest"
elif ((user_age >= 35) and (user_age < 45)):
    user_preference = "Hard"
elif ((user_age >= 45) and (user_age < 50)):
    user_preference = "Intermediate"
else:
    user_preference = "Easy"

# Index of the selected place is taken
place_index = get_index_from_place(place_user_likes)
similar_places = list(enumerate(cosine_sim[place_index]))
# print(user_preference)

# List of similar places is sorted in descending order of similarity score
sorted_similar_places = sorted(
    similar_places, key=lambda x: x[1], reverse=True)

print("Recommendations for you:")
i = 0
for x in sorted_similar_places:
    print(get_place_from_index(x[0]))
    i = i+1
    if i > 4:
        break
