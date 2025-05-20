<div>
<img src="https://github.com/user-attachments/assets/8a66510d-1bfa-49a2-97bc-811e84d24a31" width=1100>
<div>


<div align="center" style="font-family: 'Georgia', serif;" color: #3498db;">
  <h1 style="font-size: 10em; margin-bottom: 180px;">
    Classifying Stars and Galaxies with Deep Learning :<br>
    "IMAGE"
  </h1>
</div>

<div align="right" style="font-family: 'Georgia', serif;"
                          color: #8e44ad;
                          font-style: italic;
                          font-weight: bold;
                          margin-right: 15%;
                          margin-top: 10px;
                          font-size: 20em;">
   Realised by : AFAF Hajar & EZZERROUTI Salwa
</div>

&nbsp;


## 📌 Table of Contents  
1. 🌀 [Project Overview](#project-overview)  
2. 🌌 [Key Features](#key-features)  
3. 🚀 [Installation: Launch Sequence](#installation-launch-sequence)  
4. 🧑‍💻 [Data Collection](#data-collection)  
5. 🧠 [ Model Architecture](#model-architecture)  
6. 📡 [Data Constellation](#data-constellation)  
7. 📊 [Interstellar Results](#interstellar-results)  
8. 🔮 [Future Lightyears](#future-lightyears)  
9. 👽 [Join the Cosmic Crew](#join-the-cosmic-crew)
10. ⚖️ [Universal License](#universal-license)

<div align="center">
  <img src="https://github.com/user-attachments/assets/34f96a50-0427-409a-ae31-843c71c7ae0c"width=200>
</div>

## 🔍 Project Overview <a name="project-overview"></a>
This project leverages **Convolutional Neural Networks (CNNs)** to classify astronomical images into **stars** or **galaxies**. The model is trained on a dataset of labeled images and deployed via a **Streamlit web interface** for easy predictions.  
**End-to-End Deep Learning Solution** for classifying astronomical objects, developed for Université Mohammed V's Master IT program. This project demonstrates:
- Data collection from Kaggle's astronomy dataset
- CNN model training with TensorFlow
- Deployment via Streamlit web interface
- Ngrok tunneling for public access

**Modality**: Image Classification  
**Use Case**: Distinguishing stars from galaxies in telescope images

[↑ Back to Top](#table-of-contents)

---

## ✨ Key Features <a name="key-features"></a>
✅ **Interactive Web Interface** (Streamlit) 
✅ **CNN Model** with TensorFlow/Keras : 2 Conv layers + MaxPooling	TensorFlow/Keras 
✅ **Real-Time Predictions** with Confidence Scores  
✅ **Data  Pipeline** for improved generalization : Automated Kaggle download + preprocessing	KaggleHub, TensorFlow
✅ **Visualization** of training metrics (Accuracy/Loss)  
✅ **Deployment** Public URL generation	Ngrok

[↑ Back to Top](#table-of-contents)

 ![Space Banner](https://images.unsplash.com/photo-1462331940025-496dfbfc7564?ixlib=rb-1.2.1&auto=format&fit=crop&w=1600&h=400)

## ⚙️ Installation <a name="installation"></a>

### Prerequisites  
- Python 3.8+  
- pip (Python package manager)  

### Installation et importation de bibliothèques Python  
!pip install tensorflow kagglehub matplotlib
!pip install streamlit pyngrok
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os
from pathlib import Path

## 🧑‍💻Data Collection

data_path = Path(kagglehub.dataset_download("divyansh22/dummy-astronomy-data"))
print(f"🌟 Stars: {len(stars)} | 🌌 Galaxies: {len(galaxies)}")

## 🧠 Model Architecture
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(64,64,3)),
    MaxPooling2D(2,2), 
    Conv2D(64, (3,3), activation='relu'), 
    Flatten(),
    Dense(64, activation='relu'), 
    Dense(2, activation='softmax') 
])
