Larry Jack Gomez Casalins
Le code est conçu pour implémenter un système de recherche d'images basé sur le contenu (CBIR, pour son sigle en anglais) en utilisant la bibliothèque Streamlit pour créer une interface utilisateur interactive. Ce système permet de télécharger une image, de choisir un descripteur de caractéristiques et une métrique de distance, puis de rechercher et d'afficher les images les plus similaires à partir d'un ensemble prétraité. Voici un rapport détaillé sur le fonctionnement du code :
1. Importation des Bibliothèques et Définition des Fonctions
Importation
import numpy as np
import streamlit as st
from skimage.io import imread
from skimage.color import rgb2gray, rgba2rgb
from CalculoMetrics import euclidean_distance, manhattan_distance, chebyshev_distance, canberra_distance
from Ex_glcm_features import extract_glcm_features
from Ex_bit_features import extract_bit_features
Le code importe les bibliothèques nécessaires pour le traitement des images (skimage), la gestion des données (numpy), et la création de l'interface utilisateur (streamlit). De plus, il importe des fonctions personnalisées pour calculer des distances et extraire des caractéristiques.
Chargement des Caractéristiques
def load_features(feature_file):
    return np.load(feature_file, allow_pickle=True)
Cette fonction charge les fichiers de caractéristiques prétraités stockés au format .npy.
2. Configuration de l'Interface Utilisateur
Style Personnalisé

st.markdown(
    """
    <style>
    .title-container {
        display: flex;
        align-items: center;
    }
    .title {
        font-size: 24px;
        color: red;
        margin-left: 10px;
    }
    .icon {
        width: 50px;
        height: auto;
    }
    </style>
    """,
    unsafe_allow_html=True
)
Un style personnalisé est défini pour le titre de l'application, incluant l'ajout d'une icône.
Titre avec Icône
st.markdown(
    """
    <div class="title-container">
        <img src="download.png" class="icon">
        <h1 class="title">Recherche d'Images Basée sur le Contenu (CBIR)</h1>
    </div>
    """,
    unsafe_allow_html=True
)
Le titre de l'application est configuré avec une icône en utilisant HTML et CSS.
Options de la Barre Latérale
st.sidebar.title("Options")
uploaded_file = st.sidebar.file_uploader("Choisissez une image...", type=["jpg", "jpeg", "png", "bmp", "tiff"])
descriptor = st.sidebar.selectbox("Choisissez le descripteur", ["GLCM", "BiT"])
distance_metric = st.sidebar.selectbox("Choisissez la mesure de distance", ["Euclidean", "Manhattan", "Chebyshev", "Canberra"])
number_of_images = st.sidebar.slider("Nombre d'images similaires à afficher", 1, 10, 5)
Des widgets interactifs sont ajoutés dans la barre latérale pour télécharger une image, sélectionner le descripteur de caractéristiques, la métrique de distance, et le nombre d'images similaires à afficher.
3. Traitement de l'Image Téléversée
Lecture et Conversion de l'Image
if uploaded_file is not None:
    try:
        image = imread(uploaded_file)
        st.image(image, caption='Image Téléversée', use_column_width=True, width=300)
        
        if image.shape[-1] == 4:
            image = rgba2rgb(image)
        gray_image = rgb2gray(image)
    except Exception as e:
        st.write("Error: No se pudo cargar la imagen.")
        st.write(str(e))
L'image téléchargée est lue et convertie en niveaux de gris si nécessaire. En cas d'erreur, un message est affiché.
Extraction des Caractéristiques
    else:
        if descriptor == "GLCM":
            features = extract_glcm_features(uploaded_file)
            stored_features = load_features('glcm_features.npy')
            stored_paths = load_features('glcm_features_paths.npy')
        elif descriptor == "BiT":
            features = extract_bit_features(uploaded_file)
            stored_features = load_features('bit_features.npy')
            stored_paths = load_features('bit_features_paths.npy')
        
        if stored_features is None or stored_paths is None:
            st.write("Error: No features or paths loaded.")
        elif features is None:
            st.write("Error: Unable to extract features from the uploaded image.")
En fonction du descripteur sélectionné, les caractéristiques GLCM ou BiT sont extraites de l'image téléchargée. Ensuite, les caractéristiques et les chemins des images stockées précédemment sont chargés.
4. Calcul des Distances et Visualisation des Résultats
Fonctions de Distance
        else:
            distance_functions = {
                "Euclidean": euclidean_distance,
                "Manhattan": manhattan_distance,
                "Chebyshev": chebyshev_distance,
                "Canberra": canberra_distance
            }
            
            distances = [distance_functions[distance_metric](features, f) for f in stored_features]
            sorted_indices = np.argsort(distances)
La fonction de distance appropriée est sélectionnée et les distances entre les caractéristiques de l'image téléchargée et les caractéristiques stockées sont calculées. Les indices des distances sont triés pour trouver les images les plus similaires.


Affichage des Images Similaires

            cols = st.columns(4)
            for i, idx in enumerate(sorted_indices[:number_of_images]):
                col = cols[i % 4]
                try:
                    image_path = stored_paths[idx]
                    col.image(imread(image_path), caption=f"Similar Image {i+1}", width=150)
                except Exception as e:
                    col.write(f"Error al cargar la imagen {i+1}: {e}")
Les images les plus similaires sont affichées en colonnes, jusqu'au nombre spécifié par l'utilisateur.
5. Traitement des Images pour les Caractéristiques GLCM et BiT
Extraction des Caractéristiques BiT
import numpy as np
from skimage.feature import local_binary_pattern
from skimage.io import imread
from skimage.color import rgb2gray
import os

def extract_bit_features(image_path):
    try:
        image = imread(image_path)
        if image.ndim == 3 and image.shape[2] == 4:
            image = image[..., :3]
        gray_image = rgb2gray(image)
        gray_image = (gray_image * 255).astype(np.uint8)
        radius = 1
        n_points = 8 * radius
        lbp = local_binary_pattern(gray_image, n_points, radius, method='uniform')
        hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
        hist = hist.astype(float)
        hist /= hist.sum()
        return hist
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

def get_all_image_paths(input_dir):
    image_paths = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
                image_paths.append(os.path.join(root, file))
    return image_paths

def process_images(input_dir, output_file):
    feature_list = []
    image_files = get_all_image_paths(input_dir)
    for idx, image_file in enumerate(image_files):
        features = extract_bit_features(image_file)
        if features is not None:
            feature_list.append(features)
    if feature_list:
        np.save(output_file, np.array(feature_list))
        np.save(output_file.replace('.npy', '_paths.npy'), np.array(image_files))
    else:
        print("No features to save.")
Extraction des Caractéristiques GLCM
import numpy as np
from skimage.feature import graycomatrix, graycoprops
from skimage.io import imread
from skimage.color import rgb2gray
import os

def extract_glcm_features(image_path):
    try:
        image = imread(image_path)
        if image.ndim == 3 and image.shape[2] == 4:
            image = image[..., :3]
        gray_image = rgb2gray(image)
        gray_image = (gray_image * 255).astype(np.uint8)
        glcm = graycomatrix(gray_image, [1], [0], 256, symmetric=True, normed=True)
        features = [
            graycoprops(glcm, 'contrast')[0, 0],
            graycoprops(glcm, 'dissimilarity')[0, 0],
            graycoprops(glcm, 'homogeneity')[0, 0],
            graycoprops(glcm, 'energy')[0, 0],
            graycoprops(glcm, 'correlation')[0, 0],
            graycoprops(glcm, 'ASM')[0, 0]
        ]
        return features
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

def get_all_image_paths(input_dir):
    image_paths = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
                image_paths.append(os.path.join(root, file))
    return image_paths

def process_images(input_dir, output_file):
    feature_list = []
    image_files = get_all_image_paths(input_dir)
    for idx, image_file in enumerate(image_files):
        features = extract_glcm_features(image_file)
        if features is not None:
            feature_list.append(features)
    if feature_list:
        np.save(output_file, np.array(feature_list))
        np.save(output_file.replace('.npy', '_paths.npy'), np.array(image_files))
    else:
        print("No features to save.")
Ces sections du code fournissent les fonctions pour extraire les caractéristiques en utilisant les méthodes GLCM et BiT et enregistrer ces caractéristiques ainsi que les chemins des images.
En résumé, ce système CBIR permet aux utilisateurs de télécharger une image, de sélectionner différentes options pour rechercher des images similaires dans un ensemble de données prétraité, et d'afficher les résultats dans une interface interactive créée avec Streamlit.
