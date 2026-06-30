# ----------------------------------------------------------------------------------- K Means Clustering Project --------------------------------------------------------------------------------------------
# This project is based on Unsupervised Machine Learning which uses K-Means Clustering Algorithm to access a given dataset and create a target of its own and access them in the form of graphs and outputs.
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# 1) We now import our required CSV Dataset from our Storage and use it for further access
from google.colab import files
uploaded = files.upload()

# --------------------------------------------------------------------------------------------
# Output for (1)
#student_activity_dataset.csv
#student_activity_dataset.csv(text/csv) - 39865 bytes, last modified: 6/30/2026 - 100% done
#Saving student_activity_dataset.csv to student_activity_dataset.csv
# --------------------------------------------------------------------------------------------

# 2) Now using Python Pandas, we read the contents in the Dataset which displays top 5 rows along with column names and details of each column
import pandas as pd

df = pd.read_csv("student_activity_dataset.csv")
print(df.head())
print(df.info())

# ---------------------------------------------------------------------------------------------------------------------
# Output for (2)
#  StudentID  Age  Gender  Maths  Physics  Chemistry  English  \
#0      S001   16    Male     60       51         48       37   
#1      S002   17    Male     28       31         47       49   
#2      S003   17    Male     82       95         55       20   
#3      S004   16    Male     68       33         31       68   
#4      S005   16  Female     93       35         68       30   
#   Computer Science  Attendance(%)  Study Hours per Day  \
#0                33             88                    9   
#1                84             83                    1   
#2                40             89                    7   
#3                32             67                    6   
#4                90             63                   10   
#   Assignments Completed  Sports Hours per Week  
#0                      2                      4  
#1                     17                      1  
#2                     10                      2  
#3                     19                      2  
#4                     11                      4  
#<class 'pandas.core.frame.DataFrame'>
#RangeIndex: 1000 entries, 0 to 999
#Data columns (total 12 columns):
# #   Column                 Non-Null Count  Dtype 
#---  ------                 --------------  ----- 
# 0   StudentID              1000 non-null   object
# 1   Age                    1000 non-null   int64 
# 2   Gender                 1000 non-null   object
# 3   Maths                  1000 non-null   int64 
# 4   Physics                1000 non-null   int64 
# 5   Chemistry              1000 non-null   int64 
# 6   English                1000 non-null   int64 
# 7   Computer Science       1000 non-null   int64 
# 8   Attendance(%)          1000 non-null   int64 
# 9   Study Hours per Day    1000 non-null   int64 
# 10  Assignments Completed  1000 non-null   int64 
# 11  Sports Hours per Week  1000 non-null   int64 
#dtypes: int64(10), object(2)
#memory usage: 93.9+ KB
#None
# ---------------------------------------------------------------------------------------------------------------------

# 3) Creating variable to store features (In Unsupervised ML, there is no dedicated target to be fixed)
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder, StandardScaler

encoder = LabelEncoder()
df["Gender"] = encoder.fit_transform(df["Gender"])

df = df.drop("StudentID", axis=1)

X = df[[
    "Age",
    "Gender",
    "Maths",
    "Physics",
    "Chemistry",
    "English",
    "Computer Science",
    "Attendance(%)",
    "Study Hours per Day",
    "Assignments Completed",
    "Sports Hours per Week"
]]

# 4) Now we initialize K-Means Clustering and train our model to access the dataset for our means
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

wcss = []

for i in range(1,11):
    kmeans = KMeans(
        n_clusters=i,
        random_state=42,
        n_init=10
    )
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

# 5) Using Matplotlib Python Library, we plot graphs and calculate the required equations to access our data efficiently
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import seaborn as sns

  # 5.1) Using Elbow Method, we create a plot graph for WCSS v/s Clusters, to check for the Elbow Curve
plt.figure(figsize=(6,3))

plt.plot(
    range(1,11),
    wcss,
    marker='o'
)

plt.title("Elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")

plt.grid(True)
plt.show()

  # 5.2) This code is to calculate the Silhouette Scores for different values of K
scores = []

for k in range(2, 11):
    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    clusters = kmeans.fit_predict(X_scaled)
    labels = kmeans.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels)
    scores.append(score)
    print(f"K = {k}, Silhouette Score = {score:.4f}")

  # 5.3) This code is to plot a graph for PCA 2D Cluster Visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

    # 5.3.1) Plotting the clusters
plt.figure(figsize=(6,3))

plt.scatter(
    X_pca[:,0],
    X_pca[:,1],
    c=clusters,
    cmap='viridis',
    s=50,
    alpha=0.8
)

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA 2D Cluster Visualization")

plt.colorbar(label="Cluster")
plt.grid(True)
plt.show()

    # 5.3.2) Plotting the Cluster centroids
centroids_pca = pca.transform(kmeans.cluster_centers_)
plt.figure(figsize=(6,3))

plt.scatter(
    X_pca[:,0],
    X_pca[:,1],
    c=clusters,
    cmap='viridis',
    s=50,
    alpha=0.8
)

plt.scatter(
    centroids_pca[:,0],
    centroids_pca[:,1],
    c='red',
    marker='X',
    s=300,
    label='Centroids'
)

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA 2D Cluster Visualization with Centroids")

plt.legend()
plt.grid(True)
plt.show()

  # 5.4) This code is to calculate the Explained Variance Ration and Total Explained Variance
print("Explained Variance Ratio:")
print(pca.explained_variance_ratio_)

print("Total Explained Variance:")
print(sum(pca.explained_variance_ratio_))

  # 5.5) We create a Scatter Plot Graph for Study Hours v/s Average Marks
df["Cluster"] = clusters

df["Average Marks"] = (
    df["Maths"] +
    df["Physics"] +
    df["Chemistry"] +
    df["English"] +
    df["Computer Science"]
) / 5

plt.figure(figsize=(6,3))

plt.scatter(
    df["Study Hours per Day"],
    df["Average Marks"],
    c=df["Cluster"],
    cmap="viridis",
    s=60,
    alpha=0.8
)

plt.xlabel("Study Hours per Day")
plt.ylabel("Average Marks")
plt.title("Study Hours vs Average Marks (Cluster-wise)")

plt.colorbar(label="Cluster")
plt.grid(True)
plt.show()

  # 5.6) A Bar Graph to count the number of students in each cluster
cluster_counts = df["Cluster"].value_counts().sort_index()

plt.figure(figsize=(6,3))
plt.bar(
    cluster_counts.index,
    cluster_counts.values
)

for i, count in enumerate(cluster_counts.values):
    plt.text(i, count + 5, str(count), ha='center', fontsize=10)

plt.xlabel("Cluster")
plt.ylabel("Number of Students")
plt.title("Cluster Size Distribution")

plt.xticks(cluster_counts.index)
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.show()

  # 5.7) Calculating Average Marks for each scattered plots in the clusters
cluster_summary = df.groupby("Cluster")[[
    "Maths",
    "Physics",
    "Chemistry",
    "English",
    "Computer Science"
]].mean()
print(cluster_summary)

cluster_summary.plot(
    kind="bar",
    figsize=(6,3)
)

plt.title("Average Subject Marks for Each Cluster")
plt.xlabel("Cluster")
plt.ylabel("Average Marks")

plt.xticks(rotation=0)
plt.legend(title="Subjects")
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.show()

  # 5.8) Heatmap of Cluster Centers
original_centers = scaler.inverse_transform(kmeans.cluster_centers_)
cluster_centers = pd.DataFrame(
    original_centers,
    columns=X.columns
)

plt.figure(figsize=(6,3))

sns.heatmap(
    cluster_centers,
    annot=True,
    fmt=".1f",
    cmap="YlGnBu",
    linewidths=0.5
)

plt.title("Heatmap of Cluster Centers (Original Values)")
plt.xlabel("Features")
plt.ylabel("Clusters")
plt.show()

# The output for (5) is depicted as graphs which are stored in the same repository in a separate folder under the name 'Graphs'

# ---------------------------------------------------------------------------------------------- THANK YOU !!!! --------------------------------------------------------------------------------------------
