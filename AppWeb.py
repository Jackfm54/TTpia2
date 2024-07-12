import numpy as np
import streamlit as st
from skimage.io import imread
from skimage.color import rgb2gray, rgba2rgb
from CalculoMetrics import euclidean_distance, manhattan_distance, chebyshev_distance, canberra_distance
from Ex_glcm_features import extract_glcm_features
from Ex_bit_features import extract_bit_features

def load_features(feature_file):
    return np.load(feature_file, allow_pickle=True)

# Estilo personalizado para el título
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
        width: 50px; /* Ajusta el tamaño del icono */
        height: auto;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Título con icono y estilo personalizado
st.markdown(
    """
    <div class="title-container">
        <img src="download.png" class="icon">
        <h1 class="title">Recherche d'Images Basée sur le Contenu (CBIR)</h1>
    </div>
    """,
    unsafe_allow_html=True
    
)

# Pestaña del lado izquierdo con las opciones
st.sidebar.title("Options")
uploaded_file = st.sidebar.file_uploader("Choisissez une image...", type=["jpg", "jpeg", "png", "bmp", "tiff"])
descriptor = st.sidebar.selectbox("Choisissez le descripteur", ["GLCM", "BiT"])
distance_metric = st.sidebar.selectbox("Choisissez la mesure de distance", ["Euclidean", "Manhattan", "Chebyshev", "Canberra"])
number_of_images = st.sidebar.slider("Nombre d'images similaires à afficher", 1, 10, 5)

if uploaded_file is not None:
    try:
        image = imread(uploaded_file)
        st.image(image, caption='Image Téléversée', use_column_width=True, width=300)  # Ajustar el ancho de la imagen cargada
        
        # Asegurarse de que la imagen sea RGB
        if image.shape[-1] == 4:  # RGBA
            image = rgba2rgb(image)
        gray_image = rgb2gray(image)
    except Exception as e:
        st.write("Error: No se pudo cargar la imagen.")
        st.write(str(e))
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
        else:
            distance_functions = {
                "Euclidean": euclidean_distance,
                "Manhattan": manhattan_distance,
                "Chebyshev": chebyshev_distance,
                "Canberra": canberra_distance
            }
            
            distances = [distance_functions[distance_metric](features, f) for f in stored_features]
            sorted_indices = np.argsort(distances)

            # Mostrar las imágenes más similares
            cols = st.columns(4)
            for i, idx in enumerate(sorted_indices[:number_of_images]):
                col = cols[i % 4]
                try:
                    image_path = stored_paths[idx]
                    col.image(imread(image_path), caption=f"Similar Image {i+1}", width=150)  # Ajustar el ancho de las imágenes similares
                except Exception as e:
                    col.write(f"Error al cargar la imagen {i+1}: {e}")
