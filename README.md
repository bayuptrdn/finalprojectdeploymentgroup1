# Food Delivery Time Prediction

![Python](https://img.shields.io/badge/Python-3.9-blue?logo=python)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Regression-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red?logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-green)

---

# English Version

## Project Overview

**Food Delivery Time Prediction** is a Machine Learning project that develops a regression model to estimate **food delivery time (Estimated Time of Arrival / ETA)** based on operational and environmental delivery conditions.

Accurate ETA prediction is essential for food delivery platforms because it can support operational efficiency, improve delivery planning, and provide customers with more realistic delivery expectations.

The project applies an end-to-end Machine Learning workflow, covering data preprocessing, exploratory analysis, feature engineering, model development, evaluation, hyperparameter tuning, and deployment.

---

## Business Context

Accurate delivery-time estimation is a critical component of food delivery operations. A significant difference between the estimated and actual delivery time can negatively affect customer satisfaction and operational efficiency.

Delivery duration can be influenced by multiple factors, including:

* Delivery distance
* Traffic conditions
* Weather conditions
* Food preparation time
* Courier experience
* Time of day
* Vehicle type

This project addresses the problem using a **Supervised Learning – Regression** approach to estimate delivery duration from these operational variables.

---

## Objectives

The main objectives of this project are to:

* Analyze historical food delivery data
* Identify factors associated with delivery duration
* Perform data cleaning and exploratory data analysis
* Develop and compare multiple regression models
* Apply feature engineering and preprocessing techniques
* Optimize the selected model through hyperparameter tuning
* Deploy the trained model through an interactive web application

---

## Dataset

### Dataset

`Food_Delivery_Times.csv`

### Dataset Overview

The dataset contains **1,000 observations** and **9 columns**, with `Delivery_Time_min` as the target variable.

### Key Features

#### Numerical Features

* `Distance_km`
* `Preparation_Time_min`
* `Courier_Experience_yrs`

#### Categorical Features

* `Weather`
* `Traffic_Level`
* `Time_of_Day`
* `Vehicle_Type`

Some features contain missing values, which are addressed during the preprocessing stage.

---

## Methodology

The project follows a structured Machine Learning regression workflow.

### 1. Data Cleaning & Exploratory Data Analysis

The initial stage focuses on understanding the dataset, identifying data quality issues, examining distributions, and exploring relationships between features and delivery duration.

### 2. Feature Engineering & Preprocessing

The preprocessing pipeline includes:

* Handling missing values
* Encoding categorical variables
* Feature scaling
* Preparing the dataset for model training

### 3. Model Development

Several regression algorithms are evaluated:

* Linear Regression
* K-Nearest Neighbors (KNN)
* Support Vector Regression (SVR)
* Decision Tree
* Random Forest
* XGBoost

### 4. Model Evaluation

The models are compared based on their predictive performance and generalization capability.

Key evaluation metrics include:

* **Mean Absolute Error (MAE)**
* **R² Score**

### 5. Hyperparameter Tuning

The selected model is further optimized through hyperparameter tuning to improve predictive performance and generalization.

### 6. Deployment

The final model is integrated into an interactive web application, allowing users to generate delivery-time predictions based on new delivery conditions.

---

## Best Performing Model

The selected model is a **tuned XGBoost Regressor**.

### Performance

| Metric       |                  Result |
| ------------ | ----------------------: |
| **MAE**      | Approximately 7 minutes |
| **R² Score** |      Approximately 0.70 |

The model demonstrates a reasonable balance between predictive accuracy and generalization, making it suitable for estimating delivery duration based on the available features.

The trained model is serialized in `.pkl` format and can be used for inference through the deployment application.

---

## Business Insights

The analysis identifies several factors that have an important relationship with delivery duration:

* **Delivery distance** is one of the most influential factors affecting delivery time.
* **Food preparation time** and **courier experience** also contribute significantly to delivery duration.
* **Poor weather conditions** and **heavy traffic** are consistently associated with longer delivery times.

These findings can support several operational use cases, including:

* More realistic ETA estimation
* Courier scheduling and allocation
* Operational capacity planning
* Delivery performance monitoring
* Improved customer communication

---

## Technology Stack

### Tools & Platforms

* **Python 3.9** – Programming language and data processing
* **Jupyter Notebook** – Data analysis and model development
* **Streamlit** – Interactive Machine Learning application
* **Apache Airflow** – Data and Machine Learning pipeline orchestration

### Python Libraries

* `pandas`
* `numpy`
* `matplotlib`
* `seaborn`
* `scikit-learn`
* `xgboost`
* `pickle`

---

## Repository Structure

```text
Food_Delivery_Time_Prediction/
│
├── README.md
├── FinalProject.ipynb
├── FinalProject_inference.ipynb
├── Food_Delivery_Times.csv
├── airflow_ml_pipeline.py
├── requirements.txt
│
└── deployment/
    ├── Food_Delivery_Times_Clean.csv
    ├── app.py
    ├── best_xgboost_delivery_time_tuned.pkl
    ├── eda.py
    └── prediction.py
```

---

## Deployment

The **Food Delivery Time Prediction** application is deployed as an interactive web application.

### Live Application

[Open Food Delivery Time Prediction](https://huggingface.co/spaces/bputradana/deploygroup1)

The application allows users to:

* Generate food delivery-time predictions interactively
* Use the trained Machine Learning model directly
* Simulate different delivery scenarios based on distance, weather, traffic, and other conditions

---

## Future Development

Potential improvements include:

* Integration with real-time weather and traffic APIs
* End-to-end delivery tracking integration
* Expansion of the training dataset
* Continuous model retraining using new delivery data
* Model performance monitoring in production
* Deployment through scalable cloud infrastructure

---

## References

* [scikit-learn Documentation](https://scikit-learn.org/stable/)
* [XGBoost Documentation](https://xgboost.readthedocs.io/)
* Industry use cases and research related to ETA prediction in food delivery and ride-hailing platforms

---

## Author

**Bayu Putradana**
Data Analyst | Machine Learning Enthusiast

---

> **Food Delivery Time Prediction**
> Transforming delivery data into actionable insights for more accurate ETA estimation and more efficient operations.

---

# Versi Bahasa Indonesia

## Gambaran Umum Proyek

**Food Delivery Time Prediction** merupakan proyek Machine Learning yang mengembangkan model regresi untuk memperkirakan **durasi pengantaran makanan (Estimated Time of Arrival / ETA)** berdasarkan kondisi operasional dan lingkungan selama proses pengantaran.

Prediksi ETA yang akurat merupakan salah satu komponen penting dalam layanan food delivery karena dapat membantu meningkatkan efisiensi operasional, mendukung perencanaan pengantaran, serta memberikan estimasi waktu yang lebih realistis kepada pelanggan.

Proyek ini menerapkan workflow Machine Learning secara end-to-end yang mencakup preprocessing data, exploratory data analysis, feature engineering, pengembangan model, evaluasi, hyperparameter tuning, hingga deployment.

---

## Konteks Bisnis

Estimasi waktu pengantaran yang akurat merupakan bagian penting dalam operasional layanan food delivery. Perbedaan yang signifikan antara estimasi waktu dan waktu pengantaran aktual dapat berdampak negatif terhadap kepuasan pelanggan maupun efisiensi operasional.

Durasi pengantaran dapat dipengaruhi oleh berbagai faktor, antara lain:

* Jarak pengantaran
* Kondisi lalu lintas
* Kondisi cuaca
* Waktu persiapan makanan
* Pengalaman kurir
* Waktu pengantaran
* Jenis kendaraan

Proyek ini menggunakan pendekatan **Supervised Learning – Regression** untuk memperkirakan durasi pengantaran berdasarkan berbagai faktor tersebut.

---

## Tujuan

Tujuan utama dari proyek ini adalah:

* Menganalisis data historis pengantaran makanan
* Mengidentifikasi faktor yang berkaitan dengan durasi pengantaran
* Melakukan data cleaning dan exploratory data analysis
* Mengembangkan dan membandingkan beberapa model regresi
* Menerapkan teknik feature engineering dan preprocessing
* Mengoptimalkan model melalui hyperparameter tuning
* Mengimplementasikan model melalui aplikasi web interaktif

---

## Dataset

### Dataset

`Food_Delivery_Times.csv`

### Ringkasan Dataset

Dataset terdiri dari **1.000 observasi** dan **9 kolom**, dengan `Delivery_Time_min` sebagai target variable.

### Fitur Utama

#### Fitur Numerik

* `Distance_km`
* `Preparation_Time_min`
* `Courier_Experience_yrs`

#### Fitur Kategorikal

* `Weather`
* `Traffic_Level`
* `Time_of_Day`
* `Vehicle_Type`

Beberapa fitur mengandung missing values yang ditangani pada tahap preprocessing.

---

## Metodologi

Proyek ini menggunakan workflow Machine Learning berbasis regresi secara terstruktur.

### 1. Data Cleaning & Exploratory Data Analysis

Tahap awal berfokus pada pemahaman dataset, identifikasi masalah kualitas data, analisis distribusi variabel, serta eksplorasi hubungan antara fitur dan durasi pengantaran.

### 2. Feature Engineering & Preprocessing

Tahap preprocessing mencakup:

* Penanganan missing values
* Encoding variabel kategorikal
* Feature scaling
* Persiapan dataset untuk proses training

### 3. Pengembangan Model

Beberapa algoritma regresi yang dibandingkan meliputi:

* Linear Regression
* K-Nearest Neighbors (KNN)
* Support Vector Regression (SVR)
* Decision Tree
* Random Forest
* XGBoost

### 4. Evaluasi Model

Model dibandingkan berdasarkan performa prediksi dan kemampuan generalisasi.

Metrik evaluasi utama yang digunakan meliputi:

* **Mean Absolute Error (MAE)**
* **R² Score**

### 5. Hyperparameter Tuning

Model yang terpilih kemudian dioptimalkan melalui hyperparameter tuning untuk meningkatkan performa prediksi dan kemampuan generalisasi.

### 6. Deployment

Model akhir diintegrasikan ke dalam aplikasi web interaktif sehingga pengguna dapat menghasilkan prediksi durasi pengantaran berdasarkan kondisi pengantaran baru.

---

## Model dengan Performa Terbaik

Model yang dipilih adalah **XGBoost Regressor dengan hyperparameter tuning**.

### Performa

| Metrik       |           Hasil |
| ------------ | --------------: |
| **MAE**      | Sekitar 7 menit |
| **R² Score** |    Sekitar 0.70 |

Model menunjukkan keseimbangan yang cukup baik antara akurasi prediksi dan kemampuan generalisasi, sehingga dapat digunakan untuk memperkirakan durasi pengantaran berdasarkan fitur yang tersedia.

Model yang telah dilatih disimpan dalam format `.pkl` dan dapat digunakan untuk proses inference melalui aplikasi deployment.

---

## Business Insights

Hasil analisis menunjukkan beberapa faktor yang memiliki hubungan penting dengan durasi pengantaran:

* **Jarak pengantaran** merupakan salah satu faktor yang paling berpengaruh terhadap waktu pengiriman.
* **Waktu persiapan makanan** dan **pengalaman kurir** juga memberikan kontribusi yang signifikan terhadap durasi pengantaran.
* **Kondisi cuaca buruk** dan **tingkat lalu lintas yang tinggi** secara konsisten berkaitan dengan waktu pengantaran yang lebih lama.

Temuan tersebut dapat dimanfaatkan untuk berbagai kebutuhan operasional, seperti:

* Estimasi ETA yang lebih realistis
* Penjadwalan dan alokasi kurir
* Perencanaan kapasitas operasional
* Pemantauan performa pengiriman
* Peningkatan komunikasi kepada pelanggan

---

## Technology Stack

### Tools & Platforms

* **Python 3.9** – Bahasa pemrograman dan pengolahan data
* **Jupyter Notebook** – Analisis data dan pengembangan model
* **Streamlit** – Aplikasi Machine Learning interaktif
* **Apache Airflow** – Orkestrasi pipeline data dan Machine Learning

### Python Libraries

* `pandas`
* `numpy`
* `matplotlib`
* `seaborn`
* `scikit-learn`
* `xgboost`
* `pickle`

---

## Struktur Repository

```text
Food_Delivery_Time_Prediction/
│
├── README.md
├── FinalProject.ipynb
├── FinalProject_inference.ipynb
├── Food_Delivery_Times.csv
├── airflow_ml_pipeline.py
├── requirements.txt
│
└── deployment/
    ├── Food_Delivery_Times_Clean.csv
    ├── app.py
    ├── best_xgboost_delivery_time_tuned.pkl
    ├── eda.py
    └── prediction.py
```

---

## Deployment

Aplikasi **Food Delivery Time Prediction** telah diimplementasikan sebagai aplikasi web interaktif.

### Live Application

[Open Food Delivery Time Prediction](https://huggingface.co/spaces/bputradana/deploygroup1)

Aplikasi memungkinkan pengguna untuk:

* Menghasilkan prediksi waktu pengantaran secara interaktif
* Menggunakan model Machine Learning yang telah dilatih
* Mensimulasikan berbagai skenario pengantaran berdasarkan jarak, cuaca, lalu lintas, dan kondisi lainnya

---

## Pengembangan Selanjutnya

Beberapa pengembangan yang dapat dilakukan di masa mendatang meliputi:

* Integrasi dengan API cuaca dan lalu lintas secara real-time
* Integrasi sistem tracking pengantaran secara end-to-end
* Penambahan data training untuk meningkatkan kemampuan model
* Continuous model retraining menggunakan data pengantaran terbaru
* Monitoring performa model setelah deployment
* Implementasi pada cloud infrastructure yang lebih scalable

---

## Referensi

* [scikit-learn Documentation](https://scikit-learn.org/stable/)
* [XGBoost Documentation](https://xgboost.readthedocs.io/)
* Berbagai studi dan implementasi industri terkait ETA prediction pada layanan food delivery dan ride-hailing

---

## Author

**Bayu Putradana**
Data Analyst | Machine Learning Enthusiast

---

> **Food Delivery Time Prediction**
> Mengubah data pengantaran menjadi insight yang dapat digunakan untuk menghasilkan estimasi ETA yang lebih akurat dan mendukung operasional yang lebih efisien.




