import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import missingno as msno
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
from matplotlib import pyplot as plt

from matplotlib import pyplot as plt
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.metrics import adjusted_rand_score, calinski_harabasz_score, silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler,OneHotEncoder
from datetime import datetime

from kneed import KneeLocator
from sklearn.neighbors import NearestNeighbors

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.datasets import mnist
from keras import layers, models, callbacks
from sklearn.metrics import f1_score

data = pd.read_csv("ergasia/data/ObesityDataSet_raw_and_data_sinthetic.csv")

#============================================[STEP 1]============================================

#return the first 10 records
print(data.head(10))

#describe each columns data 
#print(data.describe().round(2))

#describe more about one column
#print(data.MTRANS.describe())

#show if they are null values
print(data.info())

#============================================[A]============================================

numeric_feats = [
    "Age",
    "Height",
    "Weight",
    "FCVC",
    "NCP",
    "CH2O",
    "FAF",
    "TUE",
]

#============================================Check for the min/max values of each numeric_feat============================================
for i in numeric_feats:
    print(data[i].describe())
    print("\n")

#============================================All Distribution plots============================================

#for every numeric feat create a Distribution_histplot
for i in range(len(numeric_feats)):
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    # Custom color for better visibility
    color = sns.color_palette("deep")[i]

    sns.histplot(
        data=data,
        x=numeric_feats[i],
        bins=20,
        kde=True,
        ax=ax,
        color=color,
        alpha=0.6,
    )
    # Set labels with improved clarity
    ax.set_xlabel(numeric_feats[i], fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    # Adding skewness annotation
    skewness = data[numeric_feats[i]].skew()
    ax.text(
        0.95,
        0.85,
        f"Skewness: {skewness:.2f}",
        transform=ax.transAxes,
        ha="right",
        color="black",
        weight="bold",
        fontsize=10,
        bbox=dict(facecolor="white", edgecolor="black", boxstyle="round,pad=0.3"),
    )
    # Add title and subtitle
    ax.set_title(
        numeric_feats[i]+" Distribution",
        fontsize=16,
        fontweight="bold",
        loc="left",
        pad=20,
    )
    plt.figtext(
        0.1,
        0.9,
        "Histogram with KDE overlay",
        fontsize=10,
        ha="left",
    )

    # Grid and despine for a cleaner look
    ax.grid(axis="y", linestyle="--", alpha=0.6)
    sns.despine(left=True)
    plt.tight_layout()
    plt.savefig("ergasia/plots/step1/"+numeric_feats[i]+"_Distribution_histplot.png", dpi=300, bbox_inches="tight")

#============================================[Density Plots of Numeric Features]============================================

fig, ax = plt.subplots(4, 2, figsize=(20, 40))
    
# Loop over numeric features to plot density curves
for idx, (ax_i, feat) in enumerate(zip(ax.flatten(), numeric_feats)):    
    data[feat].plot.density(
        ax=ax_i, alpha=0.7, color=f"C{idx}"
    )  # Different color for each plot
    ax_i.set_title(f"Distribution of {feat.replace('_', ' ').title()}", fontsize=14)
    ax_i.set_xlabel("Value", fontsize=12)
    ax_i.set_ylabel("Density", fontsize=12)
    ax_i.grid(True, linestyle="--", alpha=0.5)

# Adjust spacing and add a figure title
fig.suptitle("Density Plots of Numeric Features", fontsize=18, fontweight="bold")
plt.tight_layout(rect=(0, 0, 1, 0.96))  # Ensure the title doesn't overlap with subplots

plt.savefig("ergasia/plots/step1/Density_Plots_of_Numeric_Features.png", dpi=300, bbox_inches="tight")


#============================================[boxplot]============================================

fig, ax = plt.subplots(figsize=(8, 6))

# Box plot
data.select_dtypes(include=["number"]).plot.box(
    ax=ax,
    rot=30,  # Rotate labels for readability
    showmeans=True,  # Show mean indicator
    meanprops={"marker": "o", "markerfacecolor": "red", "markeredgecolor": "black"},
    patch_artist=True,  # Fill boxes with color
)

ax.set_yscale("symlog", linthresh=100)

# Grid & Layout
ax.grid(True, linestyle="--", alpha=0.6)
ax.set_ylim(0, 400)

# Labels and Title
ax.set_ylabel("Count (Log Scale)", fontsize=12)
ax.set_title(
    "Distribution of Features",
    fontsize=14,
    fontweight="bold",
    loc="left",
    pad=30,
)
plt.figtext(
    0.1,
    0.92,
    "Box plot representing statistics with log scaling",
    ha="left",
    fontsize=10,
)

# Seaborn Style
sns.despine()
plt.tight_layout()

plt.savefig("ergasia/plots/step1/Box_Plot.png", dpi=300, bbox_inches="tight")


#============================================[B]============================================

# print(data.head())

#reduced the number of different values 
data["FCVC"] = data["FCVC"].round(2)
#normalize all numeric_feats
scaler = MinMaxScaler()
data[numeric_feats]  = scaler.fit_transform(data[numeric_feats])

#============================================[C]============================================

#normalize binary options to 1/0
data["Gender"] = data["Gender"].map({"Male": 1, "Female": 0})
data["family_history_with_overweight"] = data["family_history_with_overweight"].map({"yes": 1, "no": 0})
data["FAVC"] = data["FAVC"].map({"yes": 1, "no": 0})
data["SMOKE"] = data["SMOKE"].map({"yes": 1, "no": 0})
data["SCC"] = data["SCC"].map({"yes": 1, "no": 0})

#normalize using LabelEncoder for Categorical feats
columns_to_encode = ["CAEC", "CALC", "MTRANS", "NObeyesdad"]

encoder = LabelEncoder()
for column in columns_to_encode:
    data[column] = encoder.fit_transform(data[column])

# print(data.head())
# print(data.columns)

#============================================[D]============================================

#find correlation of feats to keep only one of other if correlation is big enough

# Calculate the correlation matrix
corr = data.select_dtypes(include=["number"]).corr(method="pearson").round(2)

# Create the mask for the upper triangle
mask = np.triu(np.ones_like(corr, dtype=bool))

# Create the heatmap with the mask
plt.figure(figsize=(10, 10))
sns.heatmap(
    corr,
    cmap="coolwarm",
    annot=True,
    fmt=".2f",
    linewidths=0.5,
    vmin=-1,
    vmax=1,  # Ensure that color scaling is consistent
    cbar_kws={"label": "Correlation Coefficient"},
    annot_kws={"size": 10},  # Adjust annotation size
    mask=mask,  # Apply the mask to hide the upper triangle
)

# Title and labels for context
plt.title("Correlation Matrix of Numerical Features", fontsize=16, fontweight="bold")
plt.xlabel("Features", fontsize=12)
plt.ylabel("Features", fontsize=12)

# Rotate the axis labels for better readability
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0, ha="right")

plt.tight_layout()
plt.savefig("ergasia/plots/step1/Correlation_Plot.png", dpi=300, bbox_inches="tight")

#There is not enough correlation between two features to keep one from the other!!!!<-----


#============================================[STEP 2]============================================


#selecting non binary features related to food/drinks

features = [
    "FCVC",#Do you eat high caloric food frequently?
    "NCP",#How many main meals do you have daily?
    "CAEC",#Do you eat any food between meals?
    "CH2O",#How much water do you drink daily?
    "CALC",#How often do you drink alcohol?
]

new_data = data[features].copy()


#============================================[CLUSTERING]============================================

pca = PCA(n_components=0.95)
X_pca = pca.fit_transform(new_data)

k = data["NObeyesdad"].nunique()

start_time = datetime.now()
kmeans = KMeans(n_clusters=k, random_state=42)
clusters = kmeans.fit_predict(X_pca)
stop_time = datetime.now()
print("KMeans time :"+ str(stop_time-start_time))
silhouette = silhouette_score(X_pca, clusters)
print("KMeans Silhouette Score:", silhouette)

#create plot
_, ax = plt.subplots(1, 1, figsize=(6, 6))

# Note: seaborn expects "hue" for coloring categories, not "c"
sns.scatterplot(
    x=X_pca[:, 0],
    y=X_pca[:, 1],
    hue=kmeans.fit_predict(X_pca),
    palette="Paired",
    ax=ax,
    s=40,  # optional: adjust marker size
    edgecolor="k",  # optional: outline for better visibility
)

ax.set_title("Cluster Assignments", fontsize=14, fontweight="bold")
ax.set_xlabel("Component 1")
ax.set_ylabel("Component 2")
sns.despine(ax=ax)
plt.legend(title="Cluster", loc="upper right", bbox_to_anchor=(1.15, 1))
plt.tight_layout()

plt.savefig("ergasia/plots/step2/KMeans_Plot.png", dpi=300, bbox_inches="tight")

#============================================[DBSCAN]============================================

#find maximum distances
number_of_neighbors = 20

nearest_neighbors = NearestNeighbors(n_neighbors = number_of_neighbors)
neighbors = nearest_neighbors.fit(X_pca)

distances, indices = neighbors.kneighbors(X_pca)
distances = np.sort(distances[:, number_of_neighbors - 1], axis=0)

#create plot
i = np.arange(len(distances))
knee = KneeLocator(
    i, distances, S=1, curve="convex", direction="increasing", interp_method="polynomial"
)

fig = plt.figure(figsize=(5, 5))
knee.plot_knee()
plt.xlabel("Points")
plt.ylabel("Distance")
plt.axhline(y=distances[knee.knee], color="red", linestyle="--", linewidth=1.5)

plt.savefig("ergasia/plots/step2/KneeLocator_Plot.png", dpi=300, bbox_inches="tight")


#============================================[Create and run DBSCAN]============================================

start_time = datetime.now()
dbscan = DBSCAN(eps=distances[knee.knee], n_jobs=-1)
dbscan.fit(X_pca)
stop_time = datetime.now()
print("DBSCAN time :"+ str(stop_time-start_time))

labels = dbscan.labels_
silhouette = silhouette_score(X_pca, labels)
print("DBSCAN Silhouette Score:", silhouette)

#create plot
fig, ax = plt.subplots(1, 1, figsize=(7, 7), layout="constrained")

sns.scatterplot(
    x=X_pca[:, 0],
    y=X_pca[:, 1],
    hue=dbscan.labels_,
    palette="Set1",
    ax=ax,
    legend="full",
    s=50,
)
ax.set_title("DBSCAN Clustering", fontsize=14, fontweight="bold")
sns.despine()
plt.savefig("ergasia/plots/step2/DBSCAN_Plot.png", dpi=300, bbox_inches="tight")

#create plot showing each one class alone
fig, ax = plt.subplots(4, 3, figsize=(10, 10), sharex=True, sharey=True)
ax_flat = ax.flatten()

for i, cluster_id in enumerate(set(dbscan.labels_)):
    mask = dbscan.labels_ == cluster_id
    ax_flat[i].scatter(
        X_pca[mask, 0],
        X_pca[mask, 1],
        color=plt.cm.tab20(cluster_id) if cluster_id != -1 else "white",
    )
    ax_flat[i].set_title(f"Cluster ID: {cluster_id}")

plt.savefig("ergasia/plots/step2/DBSCAN2_Plot.png", dpi=300, bbox_inches="tight")


#============================================[STEP 3]============================================

#set X and y
X = data.drop(columns=["NObeyesdad"]).copy()
y = data["NObeyesdad"]

ix = list(range(len(X)))
dev_size, test_size = 0.2, 0.1

dev_slice = dev_size + test_size
test_slice = test_size / dev_size

ix_train, ix_dev = train_test_split(ix, test_size=dev_slice, random_state=42, stratify=y)
ix_dev, ix_test = train_test_split(ix_dev, test_size=test_slice, random_state=42, stratify=y[ix_dev])

#============================================[Show Class Distribution Across Train, Dev, and Test Sets]============================================

# Plot bar chart with value counts for train, dev, and test
fig, ax = plt.subplots(1, 1, figsize=(13, 7))

# Calculate the value counts for each split and concatenate them
df = pd.concat(
    (
        pd.Series(y[ix_train]).value_counts(normalize=True).sort_index().rename("train"),
        pd.Series(y[ix_dev]).value_counts(normalize=True).sort_index().rename("dev"),
        pd.Series(y[ix_test]).value_counts(normalize=True).sort_index().rename("test"),
    ),
    axis=1,
)

# Plot the bar chart
df.plot.bar(ax=ax, color=["#0eff8f", "#ff0e0e", "#cb0eff"])

# Add title and labels
ax.set_title(
    "Class Distribution Across Train, Dev, and Test Sets",
    fontsize=18,
    fontweight="bold",
)
ax.set_xlabel("Class", fontsize=14)
ax.set_ylabel("Proportion", fontsize=14)

# Annotate bars with the actual percentage values
for container in ax.containers:
    ax.bar_label(
        container,
        label_type="edge",
        fontsize=8,
        color="black",
        fmt="%.1f",
    )


# Adjust legend to be horizontal and place it below the plot
ax.legend(
    labels=["Train", "Dev", "Test"],
    fontsize=10,
    loc="upper center",
    bbox_to_anchor=(0.5, -0.05),
    ncol=3,
)


# Customize gridlines for readability
ax.grid(True, axis="y", linestyle="--", alpha=0.6)

# Display the plot
plt.tight_layout()
plt.savefig("ergasia/plots/step3/Class_Distribution_Across_Train_Dev_and_Test_Sets_Plot.png", dpi=300, bbox_inches="tight")


#============================================[Classifying using RandomForestClassifier]============================================

#make forest with 500 trees with each tree having unlimited depth  
rf = RandomForestClassifier(
    n_estimators = 500, 
    criterion = "entropy",
    max_depth = None,
    random_state = 42,
)
rf.fit(X.iloc[ix_train], y.iloc[ix_train])

print(f"Classifying Accuracy using RandomForestClassifier: {rf.score(X.iloc[ix_test], y.iloc[ix_test]):.3f}")

#============================================[f1_score]============================================

score = f1_score(
    y[ix_test],
    y_pred = rf.predict(X.iloc[ix_test]),
    average="weighted",
    labels=np.arange(k),
)
print(f"Weighted F1 score: {score:.3f}")

#============================================[confusion_matrix_for_RandomForestClassifier_Plot]============================================

y_pred = rf.predict(X.iloc[ix_test])

cm = confusion_matrix(y.iloc[ix_test], y_pred)

cmd_obj = ConfusionMatrixDisplay(cm ,display_labels = encoder.classes_)
cmd_obj.plot()
cmd_obj.ax_.set(
    title = "Confusion matrix for RandomForestClassifier",
    xlabel = "Predictive categories",
    ylabel = "Actual categories",
)
plt.xticks(rotation=90)  

plt.savefig("ergasia/plots/step3/Classification_confusion_matrix_for_RandomForestClassifier_Plot.png", dpi=300, bbox_inches="tight")


#============================================[Using Neural Networks with early stopping]============================================

#128-16-7
class FC_MNIST(models.Model):
    def __init__(self):
        super(FC_MNIST, self).__init__()
        # Creating layers in the initializer
        self.fc2 = layers.Dense(units=128 , activation="relu")  # Hidden layer
        #self.fc3 = layers.Dense(units=64, activation="relu")  # Hidden layer
        #self.fc4 = layers.Dense(units=64, activation="relu")  # Hidden layer
        self.fc5 = layers.Dense(units=16, activation="relu")  # Hidden layer
        self.fc6 = layers.Dense(units=7, activation="softmax")  # Output layer

    def call(self, input_tensor):
        # Pass input_tensor through the layers sequentially
        x = self.fc2(input_tensor)
        #x = self.fc3(x)
        #x = self.fc4(x)
        x = self.fc5(x)
        return self.fc6(x)



# Create the input layer
input_layer = layers.Input(shape=(16,))

# Instantiate the custom model and use it on the input layer
model = FC_MNIST()(input_layer)

# Create a complete Keras model by specifying the inputs and outputs
model = models.Model(inputs=input_layer, outputs=model)

# Model summary
model.summary(expand_nested=True)

#encode y
y_onehot = pd.get_dummies(pd.Series(y[ix_train])).values 
y_dev_onehot = pd.get_dummies(pd.Series(y[ix_dev])).values

callback = callbacks.EarlyStopping(monitor="val_loss", patience=20)
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

model_hist = model.fit(
    X.iloc[ix_train],
    y_onehot,
    epochs=200,
    validation_data=(X.iloc[ix_dev], y_dev_onehot),
    callbacks=[callback],
)
model_hist

#============================================[f1_score]============================================

score = f1_score(
    y[ix_test],
    model.predict(X.iloc[ix_test]).argmax(axis=1),
    average="weighted",
    labels=np.arange(k),
)
print(f"Weighted F1 score: {score:.3f}")


#============================================[Training vs. Validation Loss plot]============================================

#Create DataFrame from model history
df = pd.DataFrame(
    {
        "Training Loss": model_hist.history["loss"],
        "Validation Loss": model_hist.history["val_loss"],
    }
)

# Plot with customizations
fig, ax = plt.subplots(figsize=(8, 6))

df.plot(ax=ax, style=["-o", "--s"], color=["#1f77b4", "#ff7f0e"])

# Add title and labels
ax.set_title("Training vs. Validation Loss", fontsize=16, fontweight="bold")
ax.set_xlabel("Epoch", fontsize=14)
ax.set_ylabel("Loss", fontsize=14)

# Adjust legend
ax.legend(title="Loss Type", loc="upper right", fontsize=12)

# Improve readability with gridlines
ax.grid(True, linestyle="--", alpha=0.6)

# Show plot
plt.tight_layout()

plt.savefig("ergasia/plots/step3/Classification_Training_vs_Validation_Loss_plot.png", dpi=300, bbox_inches="tight")

#============================================[confusion_matrix_for_Neural Network_Plot]============================================

y_pred = model.predict(X.iloc[ix_test]).argmax(axis=1)

cm = confusion_matrix(y.iloc[ix_test], y_pred)

cmd_obj = ConfusionMatrixDisplay(cm ,display_labels = encoder.classes_)
cmd_obj.plot()
cmd_obj.ax_.set(
    title = "Confusion matrix for Neural Network",
    xlabel = "Predictive categories",
    ylabel = "Actual categories",
)
plt.xticks(rotation=90)  

plt.savefig("ergasia/plots/step3/Classification_confusion_matrix_for_Neural Network_Plot.png", dpi=300, bbox_inches="tight")


#============================================[Doing Regression for bmi]============================================

data = pd.read_csv("ergasia/data/ObesityDataSet_raw_and_data_sinthetic.csv")
#calculate bmi and create a new calumn
data["BMI"] = data["Weight"]/ (data["Height"] **2)

numeric_feats.append("BMI") 
#print(data.head())

#============================================[Normalization]============================================
data["FCVC"] = data["FCVC"].round(2)

scaler = MinMaxScaler()
data[numeric_feats]  = scaler.fit_transform(data[numeric_feats])

#make binary options to 1/0
data["Gender"] = data["Gender"].map({"Male": 1, "Female": 0})
data["family_history_with_overweight"] = data["family_history_with_overweight"].map({"yes": 1, "no": 0})
data["FAVC"] = data["FAVC"].map({"yes": 1, "no": 0})
data["SMOKE"] = data["SMOKE"].map({"yes": 1, "no": 0})
data["SCC"] = data["SCC"].map({"yes": 1, "no": 0})

#use LabelEncoder
columns_to_encode = ["CAEC", "CALC", "MTRANS", "NObeyesdad"]

encoder = LabelEncoder()
for column in columns_to_encode:
    data[column] = encoder.fit_transform(data[column])

#============================================[spliting dataset for training]============================================
X = data.drop(columns=["BMI"]).copy()
y = data["BMI"].copy()

ix = list(range(len(X)))
dev_size, test_size = 0.2, 0.1

dev_slice = dev_size + test_size
test_slice = test_size / dev_size

ix_train, ix_dev = train_test_split(ix, test_size=dev_slice, random_state=42)
ix_dev, ix_test = train_test_split(ix_dev, test_size=test_slice, random_state=42)


#============================================[Using Neural Networks with early stopping]============================================
class FC_MNIST(models.Model):
    def __init__(self):
        super(FC_MNIST, self).__init__()
        # Creating layers in the initializer
        self.fc1 = layers.Dense(units=64 , activation="relu")  # Hidden layer
        self.fc2 = layers.Dense(units=32, activation="relu")  # Hidden layer
        self.fc3 = layers.Dense(units=8, activation="relu")  # Hidden layer
        self.fc4 = layers.Dense(units=1, activation="linear")  # Output layer

    def call(self, input_tensor):
        # Pass input_tensor through the layers sequentially
        x = self.fc1(input_tensor)
        x = self.fc2(x)
        x = self.fc3(x)
        return self.fc4(x)

# Create the input layer
input_layer = layers.Input(shape=(17,))

# Instantiate the custom model and use it on the input layer
model = FC_MNIST()(input_layer)
# Create a complete Keras model by specifying the inputs and outputs
model = models.Model(inputs=input_layer, outputs=model)

# Model summary
model.summary(expand_nested=True)

y_train = pd.Series(y[ix_train]).values  
y_dev = pd.Series(y[ix_dev]).values

callback = callbacks.EarlyStopping(monitor="val_loss", patience=20)
model.compile(optimizer="adam", loss="mean_squared_error", metrics=["mean_absolute_error"])

model_hist = model.fit(
    X.iloc[ix_train],
    y_train,
    epochs=200,
    validation_data=(X.iloc[ix_dev], y_dev),
    callbacks=[callback],
)
model_hist

#============================================[Training vs. Validation Loss plot]============================================

#Create DataFrame from model history
df = pd.DataFrame(
    {
        "Training Loss": model_hist.history["loss"],
        "Validation Loss": model_hist.history["val_loss"],
    }
)

# Plot with customizations
fig, ax = plt.subplots(figsize=(10, 10))

df.plot(ax=ax, style=["-o", "--s"], color=["#1f77b4", "#ff7f0e"])

# Add title and labels
ax.set_title("Training vs. Validation Loss", fontsize=16, fontweight="bold")
ax.set_xlabel("Epoch", fontsize=14)
ax.set_ylabel("Loss", fontsize=14)

# Adjust legend
ax.legend(title="Loss Type", loc="upper right", fontsize=12)

# Improve readability with gridlines
ax.grid(True, linestyle="--", alpha=0.6)

# Show plot
plt.tight_layout()

plt.savefig("ergasia/plots/step3/Regression_Training_vs_Validation_Loss_plot.png", dpi=300, bbox_inches="tight")


#============================================[MAPE]============================================

y_pred = model.predict(X.iloc[ix_test]).flatten()
mape = np.mean(np.abs((y.loc[ix_test] - y_pred) / y.loc[ix_test])) * 100

print(f"MAPE: {mape:.2f}%")