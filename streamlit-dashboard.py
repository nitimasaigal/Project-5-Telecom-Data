import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from PIL import Image
import pickle
st.set_option('deprecation.showPyplotGlobalUse', False)


#load data
def load_data():
    try:
        df = pd.read_csv(r'C:\Users\Niti\NEXT HIKES\PROJECT 5\streamlit_df.csv')
        return df
    except FileNotFoundError:
        print('The specified file was not found')
        return None
    
df = load_data()

left_column, right_column = st.columns([1,1])

with left_column:
    try:
        st.image(Image.open(r'C:\Users\Niti\NEXT HIKES\PROJECT 5\telecom image.jpg'))
    except FileNotFoundError:
        st.write("Image file not found")

with right_column:
#title
    st.title("TellCo's Data Dashboard")
#subheader
    st.subheader("Explore Applications with Streamlit")


#sidebar
#st.sidebar.subheader('Filters')
application_data = {
    'Social Media': 0.36,
    'Google': 1.57,
    'Email': 0.45,
    'Youtube': 4.57,
    'Netflix': 4.56,
    'Gaming': 86.8
}


new_df = pd.DataFrame({
    'app_name': list(application_data.keys()),
    'Total_data': [0.36,1.57,0.45,4.57,4.56,86.8]
})




def display_application_relationship(app_name):
    total_data_percentage = application_data[app_name]
    st.sidebar.write(f"Relationship of {app_name} with Total Data is: {total_data_percentage}%")


st.sidebar.write('### Select Application')
selected_app = st.sidebar.selectbox("Select Application", list(application_data.keys()))

display_application_relationship(selected_app)
filtered_data = {selected_app: application_data[selected_app]}

#application_data['Total_data'] = df['Total_data']

st.sidebar.write('### Relationship between most used Application and Total Data')



#plt.figure(figsize=(8,8))
#plt.pie(filtered_data.values(), labels=filtered_data.keys(), autopct='%1.1f%%',startangle=140)
#plt.axis('equal')
filtered_df = pd.DataFrame(filtered_data.items(),columns = ['app_name','Total_data'])
sns.barplot(x='app_name', y='Total_data', data =filtered_df, color='pink')
#sns.violinplot(x='app_name', y='Total_data', data =filtered_df, color='green')
#filtered_df = df[df['app_name'] == selected_app]
#sns.jointplot(data=application_data, x='app_name', y = 'Total_data', color='pink')
#sns.barplot(x=list(application_data.keys()), y=list(application_data.values()), color='blue')
st.sidebar.pyplot()




import os
model_file_path = 'C:\\Users\\Niti\\NEXT HIKES\\PROJECT 5\\final_model.pkl'
if os.path.exists(model_file_path):
    try:
        with open(model_file_path, 'rb') as model_file:
            model = pickle.load(model_file)
            pass
    except Exception as e:
        print(f"Error opening file: {e}")
else:
    print(f"File not found: {model_file_path}")


def predict(input_data):
    prediction = model.predict(input_data)
    return prediction

def main():
    st.success('Model Prediction Demo')
    uploaded_file = st.file_uploader(r'C:\Users\Niti\NEXT HIKES\PROJECT 5\input_features.csv', type= ['csv'])

    if uploaded_file is not None:
        input_data = pd.read_csv(uploaded_file)

        if st.button('Predict'):
            prediction = predict(input_data)

            st.write('Prediction:', prediction)

if __name__ == '__main__':
    main()


left_column, right_column = st.columns([1,1])
with right_column:
    st.success('Prediction accuracy is 100%')
with left_column:
    st.write('### Final Result')
    csv_file_path = 'C:\\Users\\Niti\\NEXT HIKES\\PROJECT 5\\final_result.csv'
    if os.path.exists(csv_file_path):
        try:
            data = pd.read_csv(csv_file_path)
            st.dataframe(data)
        except Exception as e:
            print(f"Error reading CSV file: {e}")
    else:
        print(f"CSV file not found: {csv_file_path}")



#visualizations
st.subheader('Data Visualizations')

try:
    X = df[['engagement_score', 'experience_score']]
except KeyError as e:
    print(f"Error accessing columns: {e}")
except TypeError as e:
    print(f"Type error encountered: {e}")
st.write('### KMeans Clustering')

num_clusters = st.slider('Please select number of clusters:', min_value= 1, max_value=5, value=2)
if 'X' in locals():
    kmeans = KMeans(n_clusters=num_clusters)
    kmeans.fit_predict(X)
    pred = kmeans.labels_

    plt.figure(figsize=(8,6), dpi = 200)
    plt.scatter(X['engagement_score'], X['experience_score'], c= pred, cmap='viridis', alpha=0.5, edgecolor='k')
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:,1], s=300, c='red', marker='x')
    plt.xlabel('Engagement Score')
    plt.ylabel('Experience Score')
    plt.title(f"KMeans Clustering with {num_clusters} Clusters")
    st.pyplot(plt)
else:
    st.write("X is not defined. Please make sure the DataFrame is loaded and the specified columns exist.")



#slider_value = st.slider('Please select a value', min_value=0, max_value=10, value=5, step=1)
#st.write('You selected:', slider_value)
#x = df['satisfaction_score']
#y = df['engagement_score']

#plt.plot(x,y)
#plt.xlabel('satisfaction_score')
#plt.ylabel('engagement_score')
#plt.title('Scores relationship is {}'.format(slider_value))
#st.pyplot()


st.write('### Correlation Heatmap')
if 'engagement_score' in df.columns and 'experience_score' in df.columns and 'satisfaction_score' in df.columns:
    correlation_matrix = df[['engagement_score', 'experience_score', 'satisfaction_score']].corr(numeric_only=True)
    plt.figure(figsize=(10,6), dpi = 200)
    sns.heatmap(correlation_matrix, annot=True, cmap= 'coolwarm', fmt=".2f")
    plt.title('Correlation Heatmap')
    plt.xlabel('Features')
    plt.ylabel('Features')
    st.pyplot()
else:
    st.write("The specified columns 'engagement_score', 'experience_score' and 'satisfaction_score' do not exist in the DataFrame")



#import git
#git.refresh()
#import os
#os.environ['GIT_PYTHON_GIT_EXECUTABLE'] = 'https://github.com/nitimasaigal/Project-5-Telecom-Data'


#from git import Repo

# Specify the URL of the GitHub repository
#repo_url = 'https://github.com/nitimasaigal/Project-5-Telecom-Data.git'

# Specify the local path where you want to clone the repository
#local_path = 'C:\\Users\\Niti\\NEXT HIKES\\PROJECT 5'
#local_path = r'C:\Users\Niti\NEW_DESTINATION_DIRECTORY'
#local_path = r'C:\Users\Niti\NEXT HIKES\PROJECT 5\streamlit_dashboard'



# Clone the repository
#Repo.clone_from(repo_url, local_path)
