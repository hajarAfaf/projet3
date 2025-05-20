
%%writefile app.py

import streamlit as st
import tempfile
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import kagglehub



st.set_page_config(
    page_title="Classification Astronomique",
    page_icon="🔭",
    layout="wide"
)

# Fonction pour charger le modèle (avec cache pour ne pas recharger à chaque interaction)
@st.cache_resource
def download_and_prepare_data():
    data_path = Path(kagglehub.dataset_download("divyansh22/dummy-astronomy-data"))
    target_dir = Path("data")
    target_dir.mkdir(exist_ok=True)
    os.system(f'rsync -a "{data_path}/" "{target_dir}/"')
    cutout_path = os.path.join(target_dir, "Cutout Files")
    star_path = os.path.join(cutout_path, "star")
    galaxy_path = os.path.join(cutout_path, "galaxy")

    def load_images_from_folder(folder, label):
        images = []
        labels = []
        for filename in os.listdir(folder):
            img_path = os.path.join(folder, filename)
            img = tf.keras.preprocessing.image.load_img(img_path, target_size=IMG_SIZE)
            img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
            images.append(img_array)
            labels.append(label)
        return images, labels

    stars, star_labels = load_images_from_folder(star_path, 0)
    galaxies, galaxy_labels = load_images_from_folder(galaxy_path, 1)

    X = np.array(stars + galaxies)
    y = np.array(star_labels + galaxy_labels)
    return X, y

st.title("🌌 Classification d'Objets Astronomiques")
st.write("Ce projet permet de classifier des étoiles et galaxies à partir d'images.")

if st.button("Télécharger et préparer les données"):
    X, y = download_and_prepare_data()
    st.success(f"✅ {len(X)} images chargées.")

    # Split
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Image generators
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2,
        rotation_range=20,
        zoom_range=0.2,
        horizontal_flip=True
    )

    train_generator = train_datagen.flow(X_train_val, y_train_val, batch_size=BATCH_SIZE, subset='training')
    val_generator = train_datagen.flow(X_train_val, y_train_val, batch_size=BATCH_SIZE, subset='validation')

    test_datagen = ImageDataGenerator(rescale=1./255)
    test_generator = test_datagen.flow(X_test, y_test, batch_size=BATCH_SIZE)

    def create_model(input_shape=(*IMG_SIZE, 3), num_classes=2):
        model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=input_shape),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model

    model = create_model()
    st.session_state.model = model
    st.session_state.history = history.history
    with st.spinner("Entraînement du modèle..."):
        history = model.fit(train_generator, epochs=5, validation_data=val_generator)

    st.success("✅ Entraînement terminé !")
    model.save("saved_model.keras")
    model = tf.keras.models.load_model("saved_model.keras")


    st.subheader("📤 Charger une image pour prédiction")
    uploaded_file = st.file_uploader("Choisissez une image d'étoile ou de galaxie...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
    # Charger l'image
       img = tf.keras.preprocessing.image.load_img(uploaded_file, target_size=IMG_SIZE)
       img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
       img_expanded = np.expand_dims(img_array, axis=0)

    st.image(img, caption="Image chargée", use_column_width=True)

    # Charger le modèle
    if os.path.exists("saved_model.keras"):
        model = tf.keras.models.load_model("saved_model.keras")
        pred = model.predict(img_expanded)
        class_names = ['Star', 'Galaxy']
        predicted_label = class_names[np.argmax(pred)]

        st.markdown(f"### 🔍 Prédiction : **{predicted_label}**")
    else:
        st.warning("⚠️ Le modèle n'est pas encore entraîné. Cliquez sur le bouton ci-dessus pour l'entraîner.")
    # === Affichage des performances ===
    st.subheader("📈 Performance du modèle")
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(history.history['accuracy'], label='Train Acc')
    ax[0].plot(history.history['val_accuracy'], label='Val Acc')
    ax[0].set_title('Accuracy')
    ax[0].legend()

    ax[1].plot(history.history['loss'], label='Train Loss')
    ax[1].plot(history.history['val_loss'], label='Val Loss')
    ax[1].set_title('Loss')
    ax[1].legend()
    st.pyplot(fig)

    # === Test final ===
    test_loss, test_acc = model.evaluate(test_generator)
    st.write(f"🎯 **Test Accuracy : {test_acc:.2f}**")

    # Prédiction d'image aléatoire
    st.subheader("🔍 Prédiction sur une image aléatoire du test set")
    idx = np.random.randint(len(X_test))
    img = X_test[idx]
    label = y_test[idx]
    pred = np.argmax(model.predict(img.reshape(1, *IMG_SIZE, 3)))

    class_names = ['Star', 'Galaxy']
    # Afficher la classe prédite
    if idx == 0:
            st.success("## Prédiction: ÉTOILE 🌟")
            st.markdown("""
            **Caractéristiques détectées:**
            - Source ponctuelle de lumière
            - Profil lumineux symétrique
            - Pas de structure étendue
            """)
    else:
            st.success("## Prédiction: GALAXIE 🌌")
            st.markdown("""
            **Caractéristiques détectées:**
            - Structure étendue
            - Forme spirale ou elliptique
            - Multiple sources lumineuses
            """)
    st.image(img, caption=f"Vraie étiquette : {class_names[label]} | Prédiction : {class_names[pred]}", width=256)

