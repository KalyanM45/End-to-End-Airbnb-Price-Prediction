# End to End Airbnb-Price-Prediction 

## Introduction
In today's fast-paced world, the way we travel and seek accommodations has undergone a remarkable transformation, thanks to platforms like Airbnb. This dynamic marketplace has empowered property owners and travellers, offering a diverse range of lodging options. However, one enduring challenge is setting the right price for a listing. Hosts aspire to optimize their earnings while ensuring competitive pricing, while guests seek value for their money. Balancing these interests can be intricate, and that's where the motivation for Airbnb price prediction comes in.

## Motivation 
To harness the power of data science and machine learning to provide more accurate and data-driven pricing strategies for Airbnb hosts and guests. By developing predictive models that factor in myriad variables such as location, property type, and market dynamics, the objective is to help hosts maximize their income and guests find fair deals. In this exploration of Airbnb price prediction, we will delve into methodologies, data sources, and emerging trends, shedding light on how technology is enhancing the overall Airbnb experience for both hosts and travellers.


# Installation Guide

This guide provides step-by-step instructions on how to install and set up the Airbnb Price Prediction project. You can choose to install it either directly from GitHub or using a Docker container from DockerHub.

## Prerequisites

Before you begin, make sure you have the following prerequisites installed on your system:

 - Numpy
 - Pandas
 - Seaborn
 - Matplotlib
 - Scikit-learn
 - xgboost
 - Flask
 - Pillow
 - Catboost
 - DVC

## Installation Steps

### Option 1: Installation from GitHub

Follow these steps to install and set up the project directly from the GitHub repository:

1. **Clone the Repository**
   - Open your terminal or command prompt.
   - Navigate to the directory where you want to install the project.
   - Run the following command to clone the GitHub repository:
     ```
     git clone https://github.com/KalyanMurapaka45/Airbnb-Price-Prediction.git
     ```

2. **Create a Virtual Environment** (Optional but recommended)
   - It's a good practice to create a virtual environment to manage project dependencies. Run the following command:
     ```
     conda create -p <Environment_Name> python==<python version> -y
     ```

3. **Activate the Virtual Environment** (Optional)
   - Activate the virtual environment based on your operating system:
       ```
       conda activate <Environment_Name>/
       ```

4. **Install Dependencies**
   - Navigate to the project directory:
     ```
     cd [project_directory]
     ```
   - Run the following command to install project dependencies:
     ```
     pip install -r requirements.txt
     ```

5. **Run the Project**
   - Start the project by running the appropriate command.
     ```
     python app.py
     ```

6. **Access the Project**
   - Open a web browser or the appropriate client to access the project.
  
<br><br>
### Option 2: Installation from DockerHub

If you prefer to use Docker, you can install and run the project using a Docker container from DockerHub:

1. **Pull the Docker Image**
   - Open your terminal or command prompt.
   - Run the following command to pull the Docker image from DockerHub:
     ```
     docker pull kalyan45/airbnb-app
     ```

2. **Run the Docker Container**
   - Start the Docker container by running the following command, mapping any necessary ports:
     ```
     docker run -p 5000:5000 kalyan45/airbnb-app
     ```

3. **Access the Project**
   - Open a web browser or the appropriate client to access the project.

## Troubleshooting

- If you encounter any issues during the installation process, Contact me at ```kalyanmurapaka274@gmail.com```


# Contributing

We welcome contributions from the community! If you have any ideas or suggestions for improving the project, please feel free to create an issue or submit a pull request.

# Acknowledgements

This project was inspired by the Kaggle dataset on AirBNB Price Prediction and the corresponding competition. We also acknowledge the open-source Python libraries used in this project and their contributors.
