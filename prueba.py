"""Entry point for launching an IPython kernel.

This is separate from the ipykernel package so we can avoid doing imports until
after removing the cwd from sys.path.

correr el archivo----
Streamlit run prueba.py --server.enableXsrfProtection false
--------

"""

import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import stanza


@st.cache(allow_output_mutation=True)
def cargar_modelo_de_idioma():
    return stanza.Pipeline(lang='es', processors='tokenize, mwt, pos, lemma, depparse')



def calcular_similitud(corpus, documento, O1, O2, tipo):
    if tipo == 0:
        Corpus = corpus['Título']
        tipo_corpus = 'Título'
    elif tipo == 1:
        Corpus = corpus['Contenido']
        tipo_corpus = 'Contenido'
    elif tipo == 2:
        Corpus = corpus['Título'] + ' ' + corpus['Contenido']
        tipo_corpus = 'Título + Contenido'
    else:
        raise ValueError("El tipo de corpus seleccionado no es válido.")

    if O2 == 1:
        tipo_vectorizado = 'CountVectorizer'
        if O1 == 0:
            vectorizer = CountVectorizer()
        elif O1 == 1:
            vectorizer = CountVectorizer(ngram_range=(2, 2))
    elif O2 == 0:
        tipo_vectorizado = 'CountVectorizer (Binary)'
        if O1 == 0:
            vectorizer = CountVectorizer(binary=True)
        elif O1 == 1:
            vectorizer = CountVectorizer(binary=True, ngram_range=(2, 2))
    elif O2 == 2:
        tipo_vectorizado = 'TfidfVectorizer'
        if O1 == 0:
            vectorizer = TfidfVectorizer()
        elif O1 == 1:
            vectorizer = TfidfVectorizer(ngram_range=(2, 2))
    else:
        raise ValueError("La representación seleccionada no es válida.")

    corpus_vec = vectorizer.fit_transform(Corpus)
    documento_vec = vectorizer.transform([documento])
    similarity = cosine_similarity(corpus_vec, documento_vec)

    # Crear DataFrame con información sobre la similitud
    df_similitud = pd.DataFrame({
        'Tipo_Vectorizado': [tipo_vectorizado] * len(corpus),
        'Tipo_Corpus': [tipo_corpus] * len(corpus),
        'Contenido_Corpus': Corpus,
        'Similitud': similarity.flatten()
    })

    return df_similitud


with st.sidebar:
    st.title("Cargar archivos de texto (.txt)")

    uploaded_file = st.file_uploader("Selecciona un archivo .txt", type="txt")

    if uploaded_file is not None:
        file_contents = uploaded_file.getvalue()
        st.text("Contenido del archivo:")
        st.write(file_contents.decode())
        st.header("subir archivo")
   
   
    st.title("Menú de opciones")

        # Opciones de tipo
    tipo = ["Titulo", "Descripcion", "Ambas","Exahustivo"]
    clasificacion = ["Unigramas", "Bigramas"]
    conteo= ["binario", "frecuencia", "Tf-Df"]

          # Mostrar el menú desplegable
    seleccion = st.selectbox("Tipo:", tipo)
       # Dependiendo de la opción seleccionada, muestra un mensaje diferente
    if seleccion == "Titulo":
        tipo = 0
    elif seleccion == "Descripcion":
        tipo = 1
    elif seleccion == "Ambas":
        tipo = 2
    elif seleccion == "Exahustivo":
        tipo = 3
        
    Op1 = st.selectbox("Clasificacion:", clasificacion)
       # Dependiendo de la opción seleccionada, muestra un mensaje diferente
    if Op1 == "Unigramas":
        O1 = 0
    elif Op1 == "Bigramas":
        O1 = 1

    Op2 = st.selectbox("Conteo", conteo)
       # Dependiendo de la opción seleccionada, muestra un mensaje diferente
    if Op2 == "binario":
        O2 = 0
    elif Op2 == "frecuencia":
        O2 = 1
    elif Op2 == "Tf-Df":
        O2 = 2
        # Obtener los identificadores de inquilinos utilizando la función

    if uploaded_file and tipo is not None:
       # Llama a la función inquilinos_compatibles con los parámetros correspondientes
        corpus=pd.read_csv('normalized_data_corpus_lcd_293.csv')
        nlp = cargar_modelo_de_idioma()
        doc = nlp(file_contents.decode())
        stop = ['DET', 'ADP', 'CONJ', 'PRON','SCONJ']
        nlp = cargar_modelo_de_idioma()
        contenido_normalizado = ""
        for sent in doc.sentences:
            for token in sent.words:
                contenido_normalizado += token.lemma + " "

        st.write("Texto normalizado:")
        st.write(contenido_normalizado)
        if tipo == 3 :
            resultados=[]
            for i in range (3):
                tipo_iter = i
                for j in range (2):
                    O1_iter = j
                    for k in range (3):
                        O2_iter = k
                        resultado = calcular_similitud(corpus,contenido_normalizado,O1_iter,O2_iter,tipo_iter)
                        similitud_ordenado = resultado.sort_values(by='Similitud', ascending=False)
                        similitud_top10 = similitud_ordenado.head(10)
                        resultados.append(similitud_top10)
            similitud_ordenado = resultado.sort_values(by='Similitud', ascending=False)
            similitud_top10 = similitud_ordenado.head(10)
        else :
            resultado = calcular_similitud(corpus,contenido_normalizado,O1,O2,tipo)
            st.write('exito')
            similitud_ordenado = resultado.sort_values(by='Similitud', ascending=False)
            similitud_top10 = similitud_ordenado.head(10)
            


fig_table = go.Figure(data=[go.Table(
    columnwidth=[20] + [10] * (len(similitud_top10.columns) - 1),  # Ajustar el ancho de las columnas
    header=dict(values=list(similitud_top10.columns),
                fill_color='paleturquoise',
                align='left'),
    cells=dict(values=[similitud_top10[col] for col in similitud_top10.columns],
               fill_color='lavender',
               align='left'))
])

# Mostrar la tabla
st.write("Tabla:")
st.plotly_chart(fig_table, use_container_width=True)