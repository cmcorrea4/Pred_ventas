import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings('ignore')

# Configuración de la página
st.set_page_config(
    page_title="Predicción de Compras con IA",
    page_icon="🔮",
    layout="wide"
)

st.title("🔮 Predicción de Compras con Machine Learning")
st.markdown("Predice si un cliente comprará usando algoritmos de aprendizaje supervisado")

# Función para cargar datos
@st.cache_data
def cargar_datos(archivo):
    return pd.read_csv(archivo)

# Función para preprocesar datos
@st.cache_data
def preprocesar_datos(df):
    df_proc = df.copy()
    
    # Codificar variables categóricas
    categorical_columns = ['genero', 'region', 'programa_lealtad', 'categoria_favorita', 
                          'canal_preferido', 'estacion_ultima_compra']
    
    for col in categorical_columns:
        if col in df_proc.columns:
            le = LabelEncoder()
            df_proc[f'{col}_encoded'] = le.fit_transform(df_proc[col].astype(str))
    
    return df_proc

# Función para entrenar modelo
def entrenar_modelo(X_train, X_test, y_train, y_test, algoritmo, parametros):
    try:
        # Escalar datos para algunos algoritmos
        if algoritmo in ["Logistic Regression", "SVM"]:
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
        else:
            X_train_scaled = X_train
            X_test_scaled = X_test
            scaler = None
        
        # Inicializar modelo según algoritmo
        if algoritmo == "Random Forest":
            modelo = RandomForestClassifier(
                n_estimators=parametros['n_estimators'],
                max_depth=parametros.get('max_depth'),
                random_state=42
            )
        elif algoritmo == "Logistic Regression":
            modelo = LogisticRegression(
                C=parametros['C'],
                random_state=42,
                max_iter=1000
            )
        elif algoritmo == "Gradient Boosting":
            modelo = GradientBoostingClassifier(
                n_estimators=parametros['n_estimators'],
                learning_rate=parametros['learning_rate'],
                random_state=42
            )
        elif algoritmo == "SVM":
            modelo = SVC(
                C=parametros['C'],
                kernel=parametros['kernel'],
                random_state=42,
                probability=True
            )
        
        # Entrenar modelo
        modelo.fit(X_train_scaled, y_train)
        
        # Hacer predicciones
        y_pred = modelo.predict(X_test_scaled)
        y_pred_proba = modelo.predict_proba(X_test_scaled)[:, 1]  # Probabilidad de clase positiva
        
        # Calcular métricas
        metricas = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred)
        }
        
        # Validación cruzada
        if algoritmo in ["Logistic Regression", "SVM"]:
            cv_scores = cross_val_score(modelo, X_train_scaled, y_train, cv=5)
        else:
            cv_scores = cross_val_score(modelo, X_train, y_train, cv=5)
        
        metricas['cv_mean'] = cv_scores.mean()
        metricas['cv_std'] = cv_scores.std()
        
        return modelo, y_pred, y_pred_proba, metricas, scaler
        
    except Exception as e:
        st.error(f"Error al entrenar {algoritmo}: {str(e)}")
        return None, None, None, None, None

# Función para mostrar importancia de características (solo para algunos modelos)
def mostrar_importancia_caracteristicas(modelo, nombres_caracteristicas, algoritmo):
    if algoritmo in ["Random Forest", "Gradient Boosting"]:
        importancias = modelo.feature_importances_
        indices = np.argsort(importancias)[::-1][:10]  # Top 10
        
        st.subheader("🔍 Top 10 Variables Más Importantes")
        
        df_importancia = pd.DataFrame({
            'Variable': [nombres_caracteristicas[i] for i in indices],
            'Importancia': [importancias[i] for i in indices]
        })
        
        st.dataframe(df_importancia)

# Generar dataset de ejemplo
@st.cache_data
def generar_dataset_ejemplo():
    np.random.seed(42)
    n_samples = 300
    
    data = []
    for i in range(n_samples):
        # Variables simuladas
        edad = np.random.randint(18, 70)
        ingresos = np.random.randint(20000, 100000)
        compras_previas = np.random.randint(0, 20)
        dias_ultima_compra = np.random.randint(1, 365)
        visitas_web = np.random.randint(0, 30)
        emails_abiertos = np.random.randint(0, 10)
        productos_wishlist = np.random.randint(0, 10)
        
        # Lógica simple para variable objetivo
        prob_compra = 0.1
        prob_compra += compras_previas * 0.05
        prob_compra += max(0, (30 - dias_ultima_compra) / 100)
        prob_compra += visitas_web * 0.02
        prob_compra += emails_abiertos * 0.05
        prob_compra += productos_wishlist * 0.03
        
        prob_compra = min(0.9, max(0.1, prob_compra))
        comprara = 1 if np.random.random() < prob_compra else 0
        
        data.append({
            'cliente_id': f'C{i+1:03d}',
            'edad': edad,
            'ingresos_anuales': ingresos,
            'compras_previas': compras_previas,
            'dias_ultima_compra': dias_ultima_compra,
            'visitas_web_ultimo_mes': visitas_web,
            'emails_abiertos_ultimo_mes': emails_abiertos,
            'productos_en_wishlist': productos_wishlist,
            'genero': np.random.choice(['M', 'F']),
            'programa_lealtad': np.random.choice(['Si', 'No']),
            'comprara': comprara
        })
    
    return pd.DataFrame(data)

# Cargar datos
st.header("📁 Cargar Dataset")

opcion_datos = st.radio(
    "Selecciona el origen de los datos:",
    ["Usar dataset de ejemplo", "Cargar archivo CSV"]
)

df = None

if opcion_datos == "Usar dataset de ejemplo":
    if st.button("Generar Dataset de Ejemplo"):
        with st.spinner("Generando dataset..."):
            df = generar_dataset_ejemplo()
            st.success("Dataset generado exitosamente!")
else:
    archivo_cargado = st.file_uploader("Selecciona archivo CSV", type=['csv'])
    
    if archivo_cargado is not None:
        df = cargar_datos(archivo_cargado)
        st.success(f"Dataset cargado: {len(df)} filas, {len(df.columns)} columnas")

if df is not None:
    # Mostrar información del dataset
    st.subheader("Vista previa de datos")
    st.dataframe(df.head())
    
    # Verificar si existe variable objetivo
    if 'comprara' not in df.columns:
        st.error("⚠️ El dataset debe tener una columna llamada 'comprara' (0=No comprará, 1=Comprará)")
        st.stop()
    
    # Mostrar distribución de variable objetivo
    distribucion = df['comprara'].value_counts()
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Clientes", len(df))
    with col2:
        st.metric("Comprarán", distribucion.get(1, 0))
    with col3:
        st.metric("No Comprarán", distribucion.get(0, 0))
    
    # Preprocesar datos
    df_procesado = preprocesar_datos(df)
    
    # Configuración del modelo
    st.header("🤖 Configuración del Modelo")
    
    # Selección de variables predictoras
    columnas_numericas = df_procesado.select_dtypes(include=[np.number]).columns.tolist()
    # Excluir variable objetivo y ID
    columnas_numericas = [col for col in columnas_numericas if col not in ['comprara'] and 'id' not in col.lower()]
    
    col1, col2 = st.columns(2)
    
    with col1:
        variables_predictoras = st.multiselect(
            "Variables predictoras:",
            columnas_numericas,
            default=columnas_numericas[:6] if len(columnas_numericas) >= 6 else columnas_numericas
        )
    
    with col2:
        algoritmo = st.selectbox(
            "Algoritmo de Machine Learning:",
            ["Random Forest", "Logistic Regression", "Gradient Boosting", "SVM"]
        )
    
    # Parámetros específicos por algoritmo
    st.subheader("⚙️ Parámetros del Algoritmo")
    parametros = {}
    
    if algoritmo == "Random Forest":
        col1, col2 = st.columns(2)
        with col1:
            parametros['n_estimators'] = st.slider("Número de árboles:", 10, 200, 100)
        with col2:
            parametros['max_depth'] = st.slider("Profundidad máxima:", 3, 20, 10)
            
    elif algoritmo == "Logistic Regression":
        parametros['C'] = st.slider("Regularización (C):", 0.01, 10.0, 1.0, 0.01)
        
    elif algoritmo == "Gradient Boosting":
        col1, col2 = st.columns(2)
        with col1:
            parametros['n_estimators'] = st.slider("Número de estimadores:", 50, 300, 100)
        with col2:
            parametros['learning_rate'] = st.slider("Tasa de aprendizaje:", 0.01, 0.3, 0.1, 0.01)
            
    elif algoritmo == "SVM":
        col1, col2 = st.columns(2)
        with col1:
            parametros['C'] = st.slider("Parámetro C:", 0.1, 10.0, 1.0, 0.1)
        with col2:
            parametros['kernel'] = st.selectbox("Kernel:", ['rbf', 'linear', 'poly'])
    
    # División de datos
    test_size = st.slider("Porcentaje para prueba:", 0.1, 0.4, 0.2, 0.05)
    
    # Entrenar modelo
    if st.button("🚀 Entrenar Modelo") and variables_predictoras:
        with st.spinner("Entrenando modelo..."):
            # Preparar datos
            X = df_procesado[variables_predictoras]
            y = df_procesado['comprara']
            
            # Dividir datos
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=y
            )
            
            # Entrenar modelo
            modelo, y_pred, y_pred_proba, metricas, scaler = entrenar_modelo(
                X_train, X_test, y_train, y_test, algoritmo, parametros
            )
            
            if modelo is not None:
                # Mostrar resultados
                st.header("📊 Resultados del Modelo")
                
                # Métricas principales
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Exactitud", f"{metricas['accuracy']:.3f}")
                with col2:
                    st.metric("Precisión", f"{metricas['precision']:.3f}")
                with col3:
                    st.metric("Recall", f"{metricas['recall']:.3f}")
                with col4:
                    st.metric("F1-Score", f"{metricas['f1']:.3f}")
                
                # Validación cruzada
                st.subheader("📈 Validación Cruzada")
                st.write(f"**Exactitud promedio:** {metricas['cv_mean']:.3f} ± {metricas['cv_std']:.3f}")
                
                # Matriz de confusión
                st.subheader("🔍 Matriz de Confusión")
                cm = confusion_matrix(y_test, y_pred)
                
                cm_df = pd.DataFrame(cm, 
                                   index=['No Comprará', 'Comprará'], 
                                   columns=['Predicho: No', 'Predicho: Sí'])
                st.dataframe(cm_df)
                
                # Importancia de características
                mostrar_importancia_caracteristicas(modelo, variables_predictoras, algoritmo)
                
                # Predicciones con probabilidades
                st.subheader("🎯 Muestra de Predicciones")
                
                # Crear DataFrame con resultados
                df_resultados = pd.DataFrame({
                    'Cliente_ID': df.iloc[X_test.index]['cliente_id'].values if 'cliente_id' in df.columns else X_test.index,
                    'Real': y_test.values,
                    'Predicción': y_pred,
                    'Probabilidad_Compra': y_pred_proba,
                    'Correcto': y_test.values == y_pred
                })
                
                # Mostrar solo una muestra
                st.dataframe(df_resultados.head(10))
                
                # Interpretación de resultados
                st.subheader("💡 Interpretación")
                
                exactitud_pct = metricas['accuracy'] * 100
                precision_pct = metricas['precision'] * 100
                recall_pct = metricas['recall'] * 100
                
                if exactitud_pct >= 80:
                    st.success(f"✅ Excelente modelo: {exactitud_pct:.1f}% de exactitud")
                elif exactitud_pct >= 70:
                    st.info(f"📊 Buen modelo: {exactitud_pct:.1f}% de exactitud")
                else:
                    st.warning(f"⚠️ Modelo mejorable: {exactitud_pct:.1f}% de exactitud")
                
                st.write(f"""
                **Interpretación de métricas:**
                - **Precisión ({precision_pct:.1f}%)**: De los clientes que predijo que comprarían, {precision_pct:.1f}% realmente compraron
                - **Recall ({recall_pct:.1f}%)**: De todos los clientes que realmente compraron, el modelo identificó {recall_pct:.1f}%
                """)
                
                # Descargar predicciones
                st.subheader("💾 Descargar Predicciones")
                
                # Crear dataset completo con predicciones
                if len(df) == len(X):  # Si no se dividieron los datos
                    df_completo = df.copy()
                    if algoritmo in ["Logistic Regression", "SVM"] and scaler:
                        X_scaled = scaler.transform(X)
                        predicciones_completas = modelo.predict(X_scaled)
                        probabilidades_completas = modelo.predict_proba(X_scaled)[:, 1]
                    else:
                        predicciones_completas = modelo.predict(X)
                        probabilidades_completas = modelo.predict_proba(X)[:, 1]
                    
                    df_completo['prediccion_comprara'] = predicciones_completas
                    df_completo['probabilidad_compra'] = probabilidades_completas
                else:
                    df_completo = df_resultados
                
                csv_predicciones = df_completo.to_csv(index=False)
                st.download_button(
                    label="Descargar predicciones completas",
                    data=csv_predicciones,
                    file_name="predicciones_compra.csv",
                    mime="text/csv"
                )
                
else:
    st.info("Carga un dataset para comenzar el análisis predictivo.")
