

import pandas as pd
from shiny import App, ui, render
import seaborn as sns
from sklearn.cluster import KMeans
import matplotlib.cm as cm

# from sklearn.model_selection import train_test_split
# from sklearn.tree import plot_tree
# from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
    

app_ui = ui.page_fluid(

    ui.h1("K Means Clustering Algorithm"),
    ui.h6("The Iris dataset comprises 150 samples, with 50 samples for each of the three species. Each sample is characterized by four features: sepal length, sepal width, petal length, and petal width."),
    ui.h6("cluster centers are :"), 
    ui.h6(ui.output_text("ModelBuilding")),

    ui.row(
        ui.output_plot("elbow")      
    ),
    ui.row(
        ui.output_plot("cluster")
    ),
)

def server(input, output, session):


    
    @render.text 
    def ModelBuilding():

       # Load the Iris dataset
        df = sns.load_dataset("iris")
        x = df.iloc[:, :-1] 
        # Perform K-means clustering
        kmeans = KMeans(n_clusters=2, random_state=42)  # Assuming we want to find 3 clusters
        kmeans.fit(x)

        # Get cluster centers and labels
        centers = kmeans.cluster_centers_
        labels = kmeans.labels_
        
        return centers
    

    
    @render.plot  
    def elbow():
        # Load the Iris dataset
        df = sns.load_dataset("iris")
        x = df.iloc[:, :-1] 
        
        # Initialize an empty list to store inertia values for different k values
        inertia = []

        # Loop through k values from 1 to 10
        for k in range(1, 11):
            # Create a KMeans instance with current k value
            kmeans = KMeans(n_clusters=k, random_state=42)
            
            # Fit the KMeans model to the data
            kmeans.fit(x)
            
            # Append the inertia value (sum of squared distances of samples to their closest cluster center) to the list
            inertia.append(kmeans.inertia_)
                # Set labels and title
        fig, ax = plt.subplots()
        ax.set_xlabel("Number of clusters (k)")
        ax.set_ylabel("Sum Squared Error")
        ax.set_title("Elbow Curve")
        ax.plot(range(1, 11), inertia, marker='o', linestyle='-', color='b')

        return fig    
    
    @render.plot  
    def cluster():
        # Load the Iris dataset
        df = sns.load_dataset("iris")
        x = df.iloc[:, :-1] 

        kmeans = KMeans(n_clusters = 2, random_state = 2)
        kmeans.fit(x)
        kmeans.cluster_centers_
        pred = kmeans.fit_predict(x)

        # Get cluster centers and labels
        centers = kmeans.cluster_centers_
        labels = kmeans.labels_

        fig, ax = plt.subplots()

        # Plotting the first subplot
        ax.scatter(x['petal_length'], x['petal_width'], c=pred, cmap=cm.Accent)
        ax.grid(True)
        for center in kmeans.cluster_centers_:
            ax.scatter(center[2], center[3], marker='^', c='red')  
        ax.set_xlabel("petal length (cm)")
        ax.set_ylabel("petal width (cm)")
        return fig    
             
app = App(app_ui, server)
