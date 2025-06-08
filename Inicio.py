import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings('ignore')

# Configuración de la página
st.set_page_config(
    page_title="Predicción de Compras con IA",
    page_icon="🔮",
    layout="wide"
)
st.title("🔮 Predicción de Compras con IA")
st.markdown("Predice si un cliente comprará usando Machine Learning")
#st.image("logo_sume_blanco.png",width=150)

from PIL import Image

# Carga tu logo
logo = Image.open("logo_sume_blanco.png")

# Opción A: Mostrarlo en la cabecera
#st.image(logo, width=200)

# Opción B: Mostrarlo en la barra lateral
st.sidebar.image(logo, width=200)



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
def entrenar_modelo(X_train, X_test, y_train, y_test, algoritmo):
    try:
        if algoritmo == "Random Forest":
            modelo = RandomForestClassifier(n_estimators=100, random_state=42)
        elif algoritmo == "Logistic Regression":
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            modelo = LogisticRegression(random_state=42, max_iter=1000)
        
        # Entrenar modelo
        modelo.fit(X_train, y_train)
        
        # Hacer predicciones
        y_pred = modelo.predict(X_test)
        
        # Calcular métricas
        metricas = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred)
        }
        
        if algoritmo == "Logistic Regression":
            return modelo, metricas, scaler
        else:
            return modelo, metricas, None
        
    except Exception as e:
        st.error(f"Error al entrenar {algoritmo}: {str(e)}")
        return None, None, None

# Generar dataset de ejemplo
@st.cache_data
def generar_dataset_ejemplo():
    np.random.seed(42)
    n_samples = 300
    
    data = []
    for i in range(n_samples):
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

# 1. CARGAR DATOS
st.header("📁 1. Cargar Dataset")

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
    # Mostrar información básica
    st.subheader("Vista previa de datos")
    st.dataframe(df.head())
    
    # Verificar variable objetivo
    if 'comprara' not in df.columns:
        st.error("⚠️ El dataset debe tener una columna llamada 'comprara' (0=No comprará, 1=Comprará)")
        st.stop()
    
    # Mostrar distribución
    distribucion = df['comprara'].value_counts()
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Clientes", len(df))
    with col2:
        st.metric("Comprarán", distribucion.get(1, 0))
    with col3:
        st.metric("No Comprarán", distribucion.get(0, 0))
    
    # 2. ENTRENAR MODELO
    st.header("🤖 2. Entrenar Modelo")
    
    # Preprocesar datos
    df_procesado = preprocesar_datos(df)
    
    # Selección de variables y algoritmo
    col1, col2 = st.columns(2)
    
    with col1:
        columnas_numericas = df_procesado.select_dtypes(include=[np.number]).columns.tolist()
        columnas_numericas = [col for col in columnas_numericas if col not in ['comprara'] and 'id' not in col.lower()]
        
        variables_predictoras = st.multiselect(
            "Variables predictoras:",
            columnas_numericas,
            default=columnas_numericas[:6] if len(columnas_numericas) >= 6 else columnas_numericas
        )
    
    with col2:
        algoritmo = st.selectbox(
            "Algoritmo:",
            ["Random Forest", "Logistic Regression"]
        )
    
    # Entrenar modelo
    if st.button("🚀 Entrenar Modelo") and variables_predictoras:
        with st.spinner("Entrenando modelo..."):
            # Preparar datos
            X = df_procesado[variables_predictoras]
            y = df_procesado['comprara']
            
            # Dividir datos
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Entrenar modelo
            modelo, metricas, scaler = entrenar_modelo(
                X_train, X_test, y_train, y_test, algoritmo
            )
            
            if modelo is not None:
                # Guardar en session_state
                st.session_state.modelo = modelo
                st.session_state.metricas = metricas
                st.session_state.scaler = scaler
                st.session_state.variables_predictoras = variables_predictoras
                st.session_state.algoritmo = algoritmo
                
                # Mostrar métricas
                st.subheader("📊 Resultados del Entrenamiento")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Exactitud", f"{metricas['accuracy']:.3f}")
                with col2:
                    st.metric("Precisión", f"{metricas['precision']:.3f}")
                with col3:
                    st.metric("Recall", f"{metricas['recall']:.3f}")
                with col4:
                    st.metric("F1-Score", f"{metricas['f1']:.3f}")
                
                exactitud_pct = metricas['accuracy'] * 100
                if exactitud_pct >= 80:
                    st.success(f"✅ Excelente modelo: {exactitud_pct:.1f}% de exactitud")
                elif exactitud_pct >= 70:
                    st.info(f"📊 Buen modelo: {exactitud_pct:.1f}% de exactitud")
                else:
                    st.warning(f"⚠️ Modelo mejorable: {exactitud_pct:.1f}% de exactitud")
    
    # 3. PREDICCIÓN INDIVIDUAL
    if 'modelo' in st.session_state:
        st.header("🎯 3. Predicción para Nuevo Cliente")
        
        # Inicializar session_state para mantener valores
        if 'nuevo_cliente_data' not in st.session_state:
            st.session_state.nuevo_cliente_data = {}
        
        # Crear formulario
        nuevo_cliente = {}
        
        # Variables principales
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Datos Básicos")
            
            if 'edad' in st.session_state.variables_predictoras:
                edad_min, edad_max = int(df['edad'].min()), int(df['edad'].max())
                default_edad = st.session_state.nuevo_cliente_data.get('edad', int(df['edad'].median()))
                nuevo_cliente['edad'] = st.slider(
                    "Edad:", edad_min, edad_max, 
                    value=default_edad,
                    key="edad_slider"
                )
                st.session_state.nuevo_cliente_data['edad'] = nuevo_cliente['edad']
            
            if 'ingresos_anuales' in st.session_state.variables_predictoras:
                ingresos_min = int(df['ingresos_anuales'].min())
                ingresos_max = int(df['ingresos_anuales'].max())
                default_ingresos = st.session_state.nuevo_cliente_data.get('ingresos_anuales', int(df['ingresos_anuales'].median()))
                nuevo_cliente['ingresos_anuales'] = st.number_input(
                    "Ingresos anuales:", 
                    min_value=ingresos_min, 
                    max_value=ingresos_max,
                    value=default_ingresos,
                    key="ingresos_input"
                )
                st.session_state.nuevo_cliente_data['ingresos_anuales'] = nuevo_cliente['ingresos_anuales']
            
            if 'compras_previas' in st.session_state.variables_predictoras:
                compras_min, compras_max = int(df['compras_previas'].min()), int(df['compras_previas'].max())
                default_compras = st.session_state.nuevo_cliente_data.get('compras_previas', int(df['compras_previas'].median()))
                nuevo_cliente['compras_previas'] = st.slider(
                    "Compras previas:", compras_min, compras_max,
                    value=default_compras,
                    key="compras_slider"
                )
                st.session_state.nuevo_cliente_data['compras_previas'] = nuevo_cliente['compras_previas']
        
        with col2:
            st.subheader("Comportamiento")
            
            if 'dias_ultima_compra' in st.session_state.variables_predictoras:
                dias_min, dias_max = int(df['dias_ultima_compra'].min()), int(df['dias_ultima_compra'].max())
                default_dias = st.session_state.nuevo_cliente_data.get('dias_ultima_compra', int(df['dias_ultima_compra'].median()))
                nuevo_cliente['dias_ultima_compra'] = st.slider(
                    "Días desde última compra:", dias_min, dias_max,
                    value=default_dias,
                    key="dias_slider"
                )
                st.session_state.nuevo_cliente_data['dias_ultima_compra'] = nuevo_cliente['dias_ultima_compra']
            
            if 'visitas_web_ultimo_mes' in st.session_state.variables_predictoras:
                visitas_min, visitas_max = int(df['visitas_web_ultimo_mes'].min()), int(df['visitas_web_ultimo_mes'].max())
                default_visitas = st.session_state.nuevo_cliente_data.get('visitas_web_ultimo_mes', int(df['visitas_web_ultimo_mes'].median()))
                nuevo_cliente['visitas_web_ultimo_mes'] = st.slider(
                    "Visitas web último mes:", visitas_min, visitas_max,
                    value=default_visitas,
                    key="visitas_slider"
                )
                st.session_state.nuevo_cliente_data['visitas_web_ultimo_mes'] = nuevo_cliente['visitas_web_ultimo_mes']
            
            if 'productos_en_wishlist' in st.session_state.variables_predictoras:
                wishlist_min, wishlist_max = int(df['productos_en_wishlist'].min()), int(df['productos_en_wishlist'].max())
                default_wishlist = st.session_state.nuevo_cliente_data.get('productos_en_wishlist', int(df['productos_en_wishlist'].median()))
                nuevo_cliente['productos_en_wishlist'] = st.slider(
                    "Productos en wishlist:", wishlist_min, wishlist_max,
                    value=default_wishlist,
                    key="wishlist_slider"
                )
                st.session_state.nuevo_cliente_data['productos_en_wishlist'] = nuevo_cliente['productos_en_wishlist']
        
        # Variables restantes
        variables_restantes = [var for var in st.session_state.variables_predictoras if var not in nuevo_cliente.keys()]
        if variables_restantes:
            st.subheader("Otras Variables")
            cols = st.columns(3)
            for i, var in enumerate(variables_restantes):
                with cols[i % 3]:
                    if var in df.columns:
                        var_min, var_max = df[var].min(), df[var].max()
                        default_val = st.session_state.nuevo_cliente_data.get(var, df[var].median())
                        
                        if df[var].dtype in ['int64', 'int32']:
                            nuevo_cliente[var] = st.slider(
                                f"{var}:", int(var_min), int(var_max),
                                value=int(default_val),
                                key=f"{var}_slider"
                            )
                        else:
                            nuevo_cliente[var] = st.number_input(
                                f"{var}:", min_value=float(var_min), max_value=float(var_max),
                                value=float(default_val),
                                key=f"{var}_input"
                            )
                        st.session_state.nuevo_cliente_data[var] = nuevo_cliente[var]
        
        # Botones
        col_btn1, col_btn2 = st.columns(2)
        
        with col_btn1:
            if st.button("🔄 Resetear Valores"):
                st.session_state.nuevo_cliente_data = {}
                st.rerun()
        
        with col_btn2:
            if st.button("🔮 Predecir Compra"):
                # Crear DataFrame con los datos del nuevo cliente
                df_nuevo_cliente = pd.DataFrame([nuevo_cliente])
                
                # Asegurar que todas las variables estén presentes
                for var in st.session_state.variables_predictoras:
                    if var not in df_nuevo_cliente.columns:
                        df_nuevo_cliente[var] = 0
                
                # Reordenar columnas
                df_nuevo_cliente = df_nuevo_cliente[st.session_state.variables_predictoras]
                
                # Hacer predicción
                try:
                    if st.session_state.algoritmo == "Logistic Regression" and st.session_state.scaler:
                        X_nuevo_scaled = st.session_state.scaler.transform(df_nuevo_cliente)
                        prediccion = st.session_state.modelo.predict(X_nuevo_scaled)[0]
                        probabilidad = st.session_state.modelo.predict_proba(X_nuevo_scaled)[0, 1]
                    else:
                        prediccion = st.session_state.modelo.predict(df_nuevo_cliente)[0]
                        probabilidad = st.session_state.modelo.predict_proba(df_nuevo_cliente)[0, 1]
                    
                    # Mostrar resultado
                    st.subheader("🎯 Resultado de la Predicción")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if prediccion == 1:
                            st.success("✅ **EL CLIENTE COMPRARÁ**")
                        else:
                            st.error("❌ **EL CLIENTE NO COMPRARÁ**")
                    
                    with col2:
                        st.metric("Probabilidad de Compra", f"{probabilidad*100:.1f}%")
                    
                    # Interpretación
                    if probabilidad >= 0.8:
                        st.success(f"🎯 **Muy Alta Probabilidad ({probabilidad*100:.1f}%)**\n\n"
                                 "**Recomendación:** Cliente ideal para campañas premium.")
                    elif probabilidad >= 0.6:
                        st.info(f"📈 **Alta Probabilidad ({probabilidad*100:.1f}%)**\n\n"
                               "**Recomendación:** Buen candidato para ofertas especiales.")
                    elif probabilidad >= 0.4:
                        st.warning(f"⚖️ **Probabilidad Moderada ({probabilidad*100:.1f}%)**\n\n"
                                  "**Recomendación:** Considera descuentos atractivos.")
                    else:
                        st.error(f"📉 **Baja Probabilidad ({probabilidad*100:.1f}%)**\n\n"
                                "**Recomendación:** Enfócate en otros clientes.")
                
                except Exception as e:
                    st.error(f"Error al hacer la predicción: {str(e)}")
    
    else:
        st.info("👆 Primero entrena un modelo para poder hacer predicciones.")

else:
    st.info("👆 Carga un dataset para comenzar.")
