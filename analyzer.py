import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")

#  1. Load and Clean Data
def load_and_clean_data(file_path):
    df = pd.read_csv(file_path)
    print("First 5 records:")
    print(df.head())
    
    print("\n Checking Missing Data:")
    print(df.isnull().sum())

    df.dropna(inplace=True)
    return df

#  2. Analyze Scores
def compute_averages(df):
    scores = df[["Math", "Science", "English"]].values
    df["Average"] = np.mean(scores, axis=1)
    return df

def get_top_students(df, top_n=5):
    top = df.sort_values(by="Total", ascending=False).head(top_n)
    print("\n Top Scorers:")
    print(top[["Name", "Total", "Grade"]])
    return top

#  3. Visualization Functions
def plot_subject_averages(df):
    subject_avgs = df[["Math", "Science", "English"]].mean()
    subject_avgs.plot(kind="bar", color="skyblue")
    plt.title(" Subject-Wise Average Scores")
    plt.ylabel("Average Marks")
    plt.xlabel("Subjects")
    plt.tight_layout()
    plt.show()

def plot_heatmap(df):
    plt.figure(figsize=(6, 4))
    sns.heatmap(df[["Math", "Science", "English", "Total", "Average"]].corr(), annot=True, cmap="YlGnBu")
    plt.title(" Feature Correlation Heatmap")
    plt.show()

def plot_boxplot(df):
    plt.figure(figsize=(6, 4))
    sns.boxplot(data=df[["Math", "Science", "English"]])
    plt.title(" Outlier Detection by Subject")
    plt.show()

def plot_grade_distribution(df):
    sns.countplot(x="Grade", hue="Grade", data=df, palette="Set2", legend=False)
    plt.title(" Grade Distribution")
    plt.show()

def plot_total_distribution(df):
    sns.histplot(df["Total"], bins=10, kde=True, color="purple")
    plt.title(" Total Marks Distribution")
    plt.show()

def plot_pairplot(df):
    sns.pairplot(df[["Math", "Science", "English", "Total"]])
    plt.suptitle(" Feature Relationships", y=1.02)
    plt.show()

def plot_scatter_math_science(df):
    sns.scatterplot(x="Math", y="Science", data=df)
    plt.title(" Math vs Science Scores")
    plt.show()

#  4. Run Entire Analysis
def run_analysis(file_path):
    df = load_and_clean_data(file_path)
    df = compute_averages(df)
    get_top_students(df)

    plot_subject_averages(df)
    plot_heatmap(df)
    plot_boxplot(df)
    plot_grade_distribution(df)
    plot_total_distribution(df)
    plot_pairplot(df)
    plot_scatter_math_science(df)

#  Run the script
if __name__ == "__main__":
    run_analysis("E:\AI\AI BASIC PROJECTS\main_projects\project1\marks.csv")
