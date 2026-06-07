# Análisis Detallado del Repositorio txebas/Finance

---

## Visión General del Proyecto

Este repositorio es un **laboratorio personal de análisis financiero con Python**, centrado en tres áreas principales:

1. Extracción y análisis de documentos regulatorios del SEC (formularios 10-K) de empresas cotizadas en EE.UU.
2. Procesamiento de lenguaje natural (NLP) sobre texto financiero.
3. Construcción de pipelines RAG (Retrieval-Augmented Generation) para responder preguntas sobre documentos financieros.

Todos los notebooks están diseñados para ejecutarse en **Google Colab** (gratuito), usando principalmente la librería `edgartools` para acceder a datos del SEC de forma gratuita.

---

## Ficheros de Código

---

### 1. `Techstocksstudy.ipynb` — Estudio Comparativo de Empresas Tecnológicas

**Tamaño:** 45 KB · 40 celdas · Tecnología principal: `edgartools`, `pandas`, `matplotlib`, `seaborn`

Este notebook es el más completo y sirve como estudio de análisis cuantitativo de divulgación corporativa entre empresas tecnológicas del mercado americano.

**Paso a paso:**

**Paso 1 — Instalación y configuración del entorno**
Se instala `edgartools` con `pip install -U edgartools` y se registra la identidad del usuario ante la SEC (requisito legal para el acceso automatizado al sistema EDGAR).

```python
from edgar import *
set_identity("your.name@example.com")
```

**Paso 2 — Obtención del listado de tickers tecnológicos (dos métodos alternativos)**

- *Método 1 (recomendado):* Llama directamente a la API del screener de NASDAQ sin necesidad de API key, filtrando por sector "Technology" y obteniendo hasta 1.000 tickers.
- *Método 2 (alternativo):* Usa `yfinance` para filtrar por sector, más lento pero más preciso.

**Paso 3 — Filtrado de tickers válidos**
Por cada ticker obtenido, intenta acceder a sus formularios 10-K en EDGAR. Solo se mantienen en la lista final aquellos tickers que tienen al menos un 10-K registrado. Esto filtra empresas sin cobertura en la SEC. Los resultados se persisten en un archivo `datos_lista.pkl`.

**Paso 4 — Extracción de conteo de palabras por sección**
Para cada empresa válida, descarga el 10-K más reciente y extrae el número de palabras de cada sección (Item 1, Item 1A, Item 7, etc.). Se construye un DataFrame `df_item_comparison` con los resultados.

**Paso 5 — Análisis estadístico de variabilidad**
Se calculan las métricas estadísticas clave para cada sección:
- Media y Mediana
- Desviación Estándar
- Rango Intercuartílico (IQR)
- Coeficiente de Variación (CV = σ/μ)

El notebook explica explícitamente por qué los valores `0` deben reemplazarse por `NaN` antes de calcular estadísticos, para no sesgar artificialmente los resultados.

**Paso 6 — Visualización**
Genera tres tipos de gráficas con `matplotlib` y `seaborn`:
- *Boxplot (Gráfica A):* Muestra la distribución del número de palabras por sección, incluyendo outliers.
- *Coeficiente de Variación (Gráfica B):* Permite comparar la variabilidad relativa entre secciones de distinto tamaño.
- *Curva de densidad (Gráfica C):* Muestra la distribución de los datos y posible asimetría.

También genera gráficas de barras agrupadas por empresa y por las 4 partes estructurales del formulario 10-K (Parte I, II, III y IV), con ordenación natural de los ítems (resuelve el problema clásico de que "Item 10" se ordene antes que "Item 2").

**Paso 7 — Funcionalidades adicionales de edgartools**
Demuestra cómo buscar términos específicos dentro de un filing (`filing.search("artificial intelligence")`) y cómo obtener el documento en formato de chunks estructurados, listos para alimentar pipelines RAG.

**Paso 8 — Comparativa entre empresas**
Extrae y compara el volumen total de texto de los 10-K de NVDA, MSFT, AAPL y GOOG, diferenciando entre el texto completo y el de las secciones más relevantes para NLP.

---

### 2. `Copy_of_sec_filing_text_nlp_python.ipynb` — Extracción de Texto SEC para NLP

**Tamaño:** 49 KB · 39 celdas · Es una bifurcación del notebook original de edgartools con extensiones propias.

Este notebook comparte mucha estructura con `Techstocksstudy.ipynb` (es una versión previa o base), pero incluye algunas diferencias notables.

**Paso a paso:**

**Paso 1 — Instalación y setup**
Idéntico al notebook anterior: instala `edgartools` y configura la identidad SEC.

**Paso 2 — Extracción básica de texto en 3 líneas**
Demuestra la capacidad central de `edgartools`:
```python
filing = Company("NVDA").get_filings(form="10-K")[0]
text = filing.text()      # Texto plano
md   = filing.markdown()  # En Markdown
html = filing.html()      # En HTML
```

**Paso 3 — Filtrado de empresas tech por código SIC**
A diferencia del notebook anterior (que usa la API de NASDAQ), aquí se prueba un método alternativo: obtener el listado completo de empresas registradas en la SEC (`edgar.get_companies()`) y filtrarlas por rangos de código SIC (Standard Industrial Classification) correspondientes al sector tecnológico. Los rangos incluidos son: equipos informáticos (3570-3579), equipos de comunicaciones (3660-3679), telecomunicaciones (4812-4813), software y datos (7370-7379) y distribución de hardware (5045).

**Paso 4 — Obtención de tickers válidos y extracción masiva**
Mismo proceso que en el notebook anterior: bucle sobre todos los tickers, acceso a sus 10-K, guardado de la lista válida en `datos_lista.pkl`.

**Paso 5 — Construcción del DataFrame comparativo**
Recorre todos los tickers válidos y extrae el recuento de palabras de cada ítem del 10-K, construyendo `df_item_comparison`.

**Paso 6 — Estadísticas y visualizaciones**
Calcula las mismas métricas estadísticas (Media, Mediana, IQR, CV) y genera los mismos tipos de gráficas, con las explicaciones en español sobre cómo interpretar cada una.

**Paso 7 — Chunking para LLM**
Muestra cómo segmentar el texto del filing en fragmentos estructurados (`chunked_document`) para alimentar ventanas de contexto de LLMs.

---

### 3. `Pruebas_con_edgartools.ipynb` — Exploración y Construcción del Pipeline RAG

**Tamaño:** 2,7 MB · 45 celdas** · Notebook de experimentación iterativa. Contiene múltiples versiones (borrador → versión final) de los mismos componentes.

Este notebook documenta el proceso de exploración y construcción progresiva de un pipeline RAG completo sobre documentos financieros.

**Paso a paso:**

**Fase 1 — Exploración básica de edgartools (celdas 2-16)**
Se accede a los 10-K de NVIDIA, Apple y Google de forma manual, probando distintas formas de obtener el texto:
- `.text()` — texto plano
- `.markdown()` — representación en Markdown
- `.sections` — listado de secciones disponibles
- `.financials.income_statement()` — datos financieros estructurados

Se detecta un problema: el texto en Markdown generado por edgartools contiene artefactos HTML (`<div align='center'>`) con números de página y numeración de líneas del PDF original, que contaminan el texto útil.

**Fase 2 — Limpieza de texto con Regex (celdas 25-28)**
Se diseña una función `limpiar_numeros_linea()` que usa expresiones regulares para eliminar los números de página y contadores de línea del Markdown:
```python
patron = r"<div align='center'>(?:\d+\.?|\w+\sInc\.\s\|\s\d{4}\sForm\s10-K\s\|\s\d+)</div>"
```
La función filtra todas las líneas que encajan con ese patrón, dejando solo el texto semánticamente relevante.

**Fase 3 — Estrategias de chunking con LangChain (celdas 29-37)**
Se instalan `langchain_text_splitters`, `langchain_experimental` y `langchain_community`. Se define una arquitectura orientada a objetos con un patrón de diseño Strategy (interfaz abstracta `ChunkingStrategy`), implementando múltiples estrategias intercambiables:

- `RecursiveStrategy` — divide el texto por jerarquía de caracteres (párrafos → líneas → espacios).
- `TokenStrategy` — divide según el número de tokens reales (medido con `tiktoken`), útil para controlar el costo en APIs de LLM.
- `SentenceStrategy` — divide por oraciones usando NLTK.
- `MarkdownStrategy` — divide respetando los encabezados Markdown.
- `SemanticChunker` (de LangChain Experimental) — divide por similitud semántica usando embeddings.
- `ParentChildChunking` — crea chunks pequeños para recuperación precisa y chunks grandes para contexto completo.

El código evoluciona en varias versiones iterativas (celdas 32-37), refinando los parámetros y añadiendo manejo de errores en cada iteración.

**Fase 4 — Embeddings y búsqueda semántica (celdas 38-45)**
Se experimenta con modelos de embeddings especializados en finanzas:
- `FinanceMTEB/FinE5` — modelo de embeddings entrenado específicamente en textos financieros, requiere prefijos `"query:"` y `"passage:"`.
- `BAAI/bge-large-en-v1.5` — modelo general de alta calidad.
- `all-MiniLM-L6-v2` — modelo ligero y rápido.

Se implementa `EmbeddingProcessor`, una clase reutilizable que encapsula la lógica de codificación de queries y documentos, y el cálculo de similitud coseno para recuperar los fragmentos más relevantes.

---

### 4. `pipelinepruebasconedgartools.ipynb` — Pipeline RAG con LangChain (Versión Refinada)

**Tamaño:** 3,1 MB · 41 celdas · Es una versión muy similar a `Pruebas_con_edgartools.ipynb` pero con la refactorización final del pipeline integrada al final.

**Paso a paso:**

**Fases 1-3 — Idénticas al notebook anterior**
Las primeras 31 celdas replican exactamente la exploración, limpieza de texto y experimentación con chunking del notebook anterior. Esto sugiere que es una rama paralela o una copia de trabajo que se desarrolló simultáneamente.

**Fase 4 — Integración del pipeline RAG completo con LangChain (celdas 32-37)**
Es la parte más avanzada y diferencial. Se construye un orquestador `RAGPipelineOrchestrator` que automatiza la experimentación en **matriz completa**: prueba todas las combinaciones posibles de las variables del pipeline.

Las dimensiones de la matriz de experimentos son:

- **Estrategia de chunking:** RecursiveStrategy, TokenStrategy, MarkdownStrategy, ParentChildChunking...
- **Modelo de embeddings:** all-MiniLM-L6-v2, BAAI/bge-large-en-v1.5...
- **Estrategia de recuperación:** Vector similarity (FAISS), BM25 (ranking léxico), RRF — Reciprocal Rank Fusion (híbrido que combina ambos)
- **Modelo de reranking:** modelos de reordenación semántica post-recuperación
- **Modelo generador (SLM):** Pequeños modelos de lenguaje locales

El componente de almacenamiento vectorial usa **FAISS** (Facebook AI Similarity Search), una librería de búsqueda eficiente en espacios de alta dimensión. La búsqueda híbrida combina FAISS (semántico) con **BM25** (léxico/keyword), fusionándolos mediante **RRF** para obtener resultados más robustos que cualquier método por separado.

**Fase 5 — Evaluación y comparativa de resultados (celdas 14-16)**
Los resultados de todos los experimentos se recopilan en la clase `ExperimentoResultado` y se presentan en tablas comparativas usando `tabulate`. Las métricas registradas por experimento son: tiempo de ejecución, longitud de la respuesta generada y número de tokens de contexto recuperados.

**Fase 6 — Integración con datos reales de Apple (celdas 17-27)**
Se conecta el pipeline completo con un 10-K real de Apple (`AAPL`), aplicando la limpieza de texto, el chunking y el pipeline RAG para responder preguntas concretas sobre el documento.

---

### 5. `LIGHTPIPELINERAG.ipynb` — Pipeline RAG Ligero y Autónomo

**Tamaño:** 3,6 MB · 27 celdas · El notebook más avanzado y autocontenido del repositorio.

Este notebook es una versión consolidada y refactorizada del pipeline RAG, diseñada para ser más limpia y ejecutable de principio a fin. Recibe el nombre "Light" porque está orientado a ejecutarse con modelos pequeños (SLM — Small Language Models) de forma local.

**Paso a paso:**

**Paso 1 — Instalación de dependencias**
Instala las tres librerías clave del pipeline:
- `rank_bm25` — para búsqueda léxica BM25
- `sentence-transformers` — para embeddings semánticos
- `faiss-cpu` (o `faiss-gpu-cu12` para GPU) — para búsqueda vectorial eficiente

**Paso 2 — Definición de estructuras de datos (dataclasses)**
Define dos clases de datos con `@dataclass`:
- `Chunk`: representa un fragmento de texto con su ID, contenido y metadatos.
- `ExperimentoResultado`: captura todos los parámetros y resultados de un experimento: estrategia de chunking, modelo de embeddings, estrategia de recuperación, modelo de reranking, modelo SLM, respuesta generada y tiempo de ejecución.

**Paso 3 — Primera versión del orquestador (celdas 5-9)**
Se implementa una primera versión de `RAGPipelineOrchestrator` que ejecuta la matriz de experimentos y presenta los resultados en tabla. En este punto se usa `texto_limpio` (el texto del 10-K ya procesado) como corpus del sistema RAG.

**Paso 4 — Refactorización con LangChain (celdas 13-16)**
Se reescribe el pipeline usando componentes nativos de LangChain:
- `HuggingFaceEmbeddings` para los vectores semánticos
- `FAISS` de LangChain como vector store
- `RecursiveCharacterTextSplitter` y `MarkdownTextSplitter` para chunking
- `BM25Okapi` para búsqueda léxica
- `ChatPromptTemplate` de `langchain_core` para formatear los prompts enviados al SLM

La lógica de fusión híbrida con RRF se implementa a mano, combinando los rankings de FAISS y BM25 para producir una lista de fragmentos ordenada por relevancia combinada.

**Paso 5 — Integración con datos reales (celdas 17-23)**
Se conecta al 10-K real de Apple (`AAPL`) via `edgartools`, se extrae en Markdown y se aplica la función de limpieza `limpiar_numeros_linea()` para eliminar artefactos del PDF.

**Paso 6 — Ejecución final del pipeline completo (celdas 24-27)**
Con el corpus limpio, se ejecuta el `RAGPipelineOrchestrator` completo sobre el 10-K de Apple. El pipeline:
1. Fragmenta el documento según cada estrategia de chunking.
2. Genera embeddings de cada chunk con el modelo configurado.
3. Indexa los embeddings en FAISS.
4. Dado un query (p.ej. "What are the risks?"), recupera los fragmentos más relevantes por similitud vectorial y/o BM25.
5. Aplica reranking semántico sobre los candidatos recuperados.
6. Alimenta los fragmentos como contexto a un SLM para generar la respuesta final.
7. Registra todos los resultados para la comparativa.

---

### 6. `datos_lista.pkl` — Datos Serializados

**Tamaño:** pequeño · Formato: Python Pickle

No es un fichero de código ejecutable. Es el resultado persistido del proceso de filtrado de tickers descrito en los notebooks de SEC. Contiene la lista de tickers tecnológicos para los que se encontró al menos un formulario 10-K válido en EDGAR. Se usa para evitar repetir el proceso de filtrado (que puede tardar horas) en ejecuciones posteriores.

---

### 7. `notebooks/` — Carpeta de Notebooks Adicionales

Subcarpeta del repositorio. No se han podido listar sus contenidos completos desde el listado principal, pero presumiblemente contiene versiones alternativas o experimentos secundarios de los notebooks principales.

---

## Resumen del Flujo de Trabajo Global

El repositorio implementa un flujo de trabajo de extremo a extremo:

```
SEC EDGAR (10-K filings)
        ↓
   edgartools          ← Descarga y parsing de documentos regulatorios
        ↓
  Limpieza Regex       ← Eliminación de artefactos del PDF
        ↓
  Chunking Strategy    ← Fragmentación en chunks (Recursive / Token / Markdown / Semantic)
        ↓
  Embeddings           ← Vectorización (FinE5 / BGE / MiniLM)
        ↓
  FAISS + BM25         ← Índice vectorial + búsqueda léxica
        ↓
  RRF Hybrid Search    ← Fusión de rankings
        ↓
  Reranking            ← Reordenación semántica
        ↓
  SLM Generation       ← Respuesta en lenguaje natural
        ↓
  Comparativa          ← Tabla de resultados por configuración
```

---

## Librerías Utilizadas

| Librería | Propósito |
|---|---|
| `edgartools` | Acceso gratuito a SEC EDGAR |
| `pandas` | Manipulación de datos tabulares |
| `matplotlib` / `seaborn` | Visualización estadística |
| `langchain` / `langchain-community` | Orquestación del pipeline RAG |
| `faiss-cpu` | Búsqueda vectorial eficiente |
| `rank_bm25` | Búsqueda léxica BM25 |
| `sentence-transformers` | Modelos de embeddings |
| `tiktoken` | Conteo de tokens (OpenAI tokenizer) |
| `nltk` | Tokenización por oraciones |
| `numpy` / `scipy` | Cálculos numéricos y estadísticos |
| `pickle` | Serialización de objetos Python |
| `re` (regex) | Limpieza de texto |
