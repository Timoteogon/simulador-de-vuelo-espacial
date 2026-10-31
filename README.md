\documentclass[11pt]{article}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage[spanish,es-noshorthands]{babel}
\usepackage[margin=2.4cm]{geometry}
\usepackage{amsmath}
\usepackage{booktabs}
\usepackage{longtable}
\usepackage{array}
\usepackage{enumitem}
\usepackage{url}
\usepackage[hidelinks]{hyperref}

\setlist{itemsep=2pt,topsep=4pt}
\urlstyle{tt}

\title{\textbf{Simulador de cohetes 2D}\\[4pt]\large Versión 12.13.6}
\author{Mateo Chavarria Gonzalez}
\date{2026}

\begin{document}
\maketitle

\begin{quote}\itshape
\textbf{Estado del proyecto:} herramienta de simulación y análisis preliminar. No es software de vuelo, no está certificado y sus resultados no deben utilizarse como solución operacional de navegación o guiado de una misión real.
\end{quote}

\section{Características principales}

\begin{itemize}
\item Cohete genérico de dos etapas, con un conjunto de parámetros iniciales inspirado en un lanzador tipo Falcon~9.
\item Modelo de movimiento del vehículo en \textbf{2D}, en el plano orbital adoptado por la simulación.
\item Gravedad central terrestre con variación de la aceleración gravitatoria con la distancia.
\item Atmósfera exponencial y arrastre aerodinámico; el coeficiente de arrastre puede depender del número de Mach.
\item Empuje dependiente de la presión ambiente, con área de salida de tobera por etapa.
\item Masa y delta-v propulsivo integrados como variables de estado durante las fases con empuje.
\item Gravity turn, separación de etapas, regulación de empuje y límite de aceleración para el perfil tripulado.
\item Integración híbrida con \path{solve_ivp} (RK45) y kernels RK4/Verlet compilados con Numba.
\item Localización de eventos de corte e impacto, interpolación de Hermite y control explícito de fallos numéricos.
\item Historial de vuelo con almacenamiento temporal en disco, exportación a NPZ/CSV y gráficas.
\item Interfaz Tkinter con visualización de trayectoria, telemetría, fases y eventos.
\item Cálculo independiente de ventanas interplanetarias mediante elementos keplerianos aproximados y Lambert.
\end{itemize}

\section{Modos de misión}

\begin{longtable}{@{}l p{4.6cm} p{7.4cm}@{}}
\caption{Modos de misión disponibles}\\
\toprule
Modo & Descripción & Criterio / alcance importante \\
\midrule
\endfirsthead
\toprule
Modo & Descripción & Criterio / alcance importante \\
\midrule
\endhead
\bottomrule
\endfoot
\path{satelital} & Inserción orbital mediante secuencia \emph{burn--coast--burn}. & Busca el apogeo configurado y realiza la circularización; no aplica límite fisiológico de aceleración. \\
\path{tripulada} & Inserción orbital con límite de carga g. & Perfiles \path{bcb} y \path{directo}; la órbita se verifica con periapsis seguro y tolerancia de excentricidad. \\
\path{suborbital} & Vuelo balístico de un cohete sonda. & Puede cortar por apogeo objetivo o agotar las etapas; continúa hasta apogeo/impacto o el límite temporal configurado. \\
\path{gto} & Transferencia a órbita geoestacionaria. & Primer tramo hasta el apogeo objetivo y segunda quema para elevar el periapsis y reducir la excentricidad. \\
\path{tli} & Transferencia de alta energía hacia la distancia lunar. & Es un modo simplificado de transferencia; alcanzar el apogeo objetivo no garantiza por sí mismo un encuentro lunar controlado. \\
\path{escape} & Trayectoria hiperbólica de escape. & El corte se define por energía específica no negativa. \\
\path{deorbit} & Desórbita y reentrada. & Permite activar la quema por órbita, por tiempo o mediante planificación para aproximar el impacto al sitio de lanzamiento. No incluye modelo térmico de reentrada. \\
\path{lunar} & Ascenso, aparcamiento, planificación TLI y propagación hacia la esfera de influencia lunar. & Utiliza efemérides consultables en cualquier instante. La captura LOI es opcional y sigue siendo un modelo 2D provisional. \\
\end{longtable}

En el modo \path{tripulada}, \path{perfil_tripulada} admite \path{bcb} y \path{directo}.

\section{Modelo físico y numérico}

\subsection{Dinámica del cohete}

El vehículo se propaga en un plano 2D. En el marco terrestre se consideran la gravedad central, el empuje, el arrastre y, cuando se activan, las aceleraciones diferenciales debidas a terceros cuerpos.

La perturbación de un cuerpo externo se expresa como aceleración de marea en el marco geocéntrico:
\begin{equation}
\vec a_{\mathrm{pert},i}=\mu_i\left(
\frac{\vec r_i-\vec r}{|\vec r_i-\vec r|^3}
-\frac{\vec r_i}{|\vec r_i|^3}
\right),
\end{equation}
donde $\vec r$ es la posición geocéntrica del cohete y $\vec r_i$ la del cuerpo perturbador. El segundo término resta la aceleración común del origen terrestre.

La rotación terrestre, si se activa, contribuye a la velocidad inicial de la plataforma y a la velocidad del aire usada para calcular el arrastre. La simulación no añade términos de Coriolis y centrífugo al marco inercial, para evitar contar dos veces esos efectos.

\subsection{Integradores}

\begin{itemize}
\item \textbf{RK45} (\path|scipy.integrate.solve_ivp|) durante el empuje, cerca de la atmósfera y en fases que requieren integración adaptativa.
\item \textbf{RK4} en coast libre y condiciones suaves, con subpasos acotados.
\item \textbf{Velocity-Verlet 3D} para generar las tablas de efemérides de Sol y Luna en la ruta \path|Horizons+Verlet|.
\item \textbf{Hermite cúbico} para consultar las tablas entre nodos, usando posición y velocidad.
\item Eventos y bisección/interpolación para localizar cortes, cruces y contactos con precisión temporal y espacial controlada.
\end{itemize}

La dinámica de masa variable está integrada durante el empuje. El delta-v propulsivo se acumula a partir de la aceleración del motor, y no se confunde con el cambio neto de rapidez causado también por gravedad y arrastre.

\subsection{Efemérides y alcance del modelo}

El registro del código contiene Sol, Luna y planetas. Sin embargo, \textbf{la presencia de un cuerpo en el registro no significa que su perturbación esté activa en la simulación}. El cohete es una partícula de prueba: no modifica las trayectorias de los cuerpos celestes. La configuración general mantiene desactivadas por defecto las perturbaciones opcionales de terceros cuerpos.

La ruta \path|Horizons+Verlet| usa un acoplamiento simplificado y unilateral: el Sol se propaga con el modelo Sol--Tierra de dos cuerpos, mientras que la Luna incluye el término de marea solar; no es un integrador N-cuerpos mutuo.

Para el modo lunar:
\begin{itemize}
\item \path|efemerides_lunar = "horizons_verlet"| es la opción predeterminada. Consulta condiciones iniciales en JPL Horizons y propaga Sol y Luna con un modelo simplificado de Verlet 3D.
\item \path|efemerides_lunar = "spice"| usa la ruta SPICE mediante PyMCel.
\item El modo lunar no acepta silenciosamente una efeméride kepleriana de respaldo como sustituto de una fuente válida.
\item La trayectoria del cohete continúa siendo 2D. La coordenada fuera del plano de la Luna se utiliza para advertir cuando un contacto 2D no representa un contacto físico 3D.
\item La captura LOI está desactivada por defecto (\path|LUNAR_LOI = False|). Cuando se activa, la captura y verificación orbital siguen siendo provisionales y 2D.
\item En \path|Horizons+Verlet|, \path|T_MAX_LUNAR| está limitado a 30 días por el coste de preintegrar la tabla.
\end{itemize}

El modelo no incluye perturbaciones planetarias en la propagación Verlet Sol--Tierra--Luna, ni J2 terrestre ni relatividad. Esa es una delimitación deliberada del modelo, no una afirmación de que esos efectos sean siempre despreciables.

\section{Validación de Horizons+Verlet (estado medido)}

La versión 12.13.6 incorpora un aviso en la interfaz cuando no existe un informe de validación que corresponda \textbf{a la misma versión del motor y a la misma época inicial}.

Se ejecutó la comparación para la época \path|2026-01-01T00:00:00+00:00|, con paso interno Verlet de 20 s, tabla cada 600 s y muestras en 1, 2, 5 y 10 días. Los errores observados fueron:

\begin{table}[h]
\centering
\caption{Errores de posición y velocidad respecto de JPL Horizons}
\begin{tabular}{@{}r r r r r@{}}
\toprule
Tiempo desde $T_0$ & Luna: pos. (km) & Luna: vel. (m/s) & Sol: pos. (km) & Sol: vel. (m/s) \\
\midrule
1 día  & 0{,}004952 & 0{,}000115 & 140{,}790     & 3{,}258389 \\
2 días & 0{,}019642 & 0{,}000223 & 560{,}394     & 6{,}461563 \\
5 días & 0{,}110845 & 0{,}000532 & 3.348{,}822300 & 14{,}997956 \\
10 días& 0{,}439146 & 0{,}000850 & 11.537{,}269652& 23{,}677544 \\
\bottomrule
\end{tabular}
\end{table}

El validador utiliza umbrales de 5 km para la posición lunar y 50 km para la posición solar. En esta ejecución:
\begin{itemize}
\item \textbf{Luna:} queda dentro del umbral configurado; error máximo de posición de 0{,}439146 km a 10 días.
\item \textbf{Sol:} supera ampliamente el umbral configurado; error máximo de posición de 11.537{,}269652 km a 10 días.
\item \textbf{Resultado global:} \texttt{NO\ ACEPTADA} según los umbrales actuales.
\end{itemize}

Por tanto, no debe describirse la ruta completa \path|Horizons+Verlet| como validada globalmente. La buena coincidencia lunar de esta época no elimina la discrepancia solar ni demuestra que el resultado se mantenga para otras fechas. La causa de la discrepancia solar está en investigación; no se ha modificado la física para ocultarla.

El informe leído por el motor se llama \path|validacion_horizons_verlet_v12.13.6.json| y debe estar junto al archivo del motor. El motor comprueba el hash SHA-256, la época, el esquema, la configuración del integrador y los umbrales antes de aceptar el informe.

\section{Requisitos}

\begin{itemize}
\item Python 3.9 o superior (por el uso de anotaciones genéricas integradas como \path|list[str]| y \path|tuple[...]|).
\item NumPy, SciPy, Matplotlib, Numba, Astropy, SpiceyPy, PyMCel.
\item Tkinter, para la interfaz gráfica (normalmente incluido en la distribución de Python para Windows).
\item Plotly, opcional: se utiliza para las gráficas finales interactivas.
\item Pytest, opcional: necesario para ejecutar la suite de pruebas.
\end{itemize}

\begin{quote}\small
PyMCel se importa como dependencia obligatoria para obtener constantes físicas. Aunque la ruta lunar \path|Horizons+Verlet| parte de consultas a Horizons, el módulo actual también importa PyMCel, Astropy y SpiceyPy al cargar el motor.
\end{quote}

\section{Instalación y ejecución}

Clona el repositorio e instala las dependencias desde PowerShell o una terminal:

\begin{verbatim}
git clone https://github.com/Timoteogon/simulador-de-vuelo-espacial.git
cd simulador-de-vuelo-espacial
python -m pip install -r requirements.txt
\end{verbatim}

En Windows también se puede usar \path|py| en lugar de \path|python|.

La instalación como paquete permite ejecutar el simulador desde cualquier carpeta:

\begin{verbatim}
python -m pip install .
py -m simulador
\end{verbatim}

Esto abre la interfaz gráfica Tkinter. En ella se puede configurar el modo de misión, el perfil tripulado, los parámetros de lanzamiento, los objetivos y las opciones de visualización; después se inicia o reinicia la simulación desde la propia interfaz.

Si Tkinter no está disponible, el módulo puede cargarse para el modo de consola de ventanas interplanetarias, pero no podrá abrir la GUI.

\subsection{Cálculo de ventanas interplanetarias por consola}

El subcomando \path|--ventanas| calcula ventanas mediante elementos planetarios keplerianos aproximados de Standish y un método de Lambert. \textbf{No propaga el vuelo del cohete ni verifica una misión completa}: estima fechas y condiciones de salida/llegada.

Ejemplo Tierra--Marte:

\begin{verbatim}
py -m simulador --ventanas tierra marte 2026-01-01 2029-01-01
\end{verbatim}

Opciones disponibles:

\begin{verbatim}
--c3-max           Límite de C3 (km2/s2)
--dv-salida-max    Límite de delta-v de salida (km/s)
--vinf-llegada-max Límite de velocidad hiperbólica de llegada (km/s)
--tof-min          Tiempo de vuelo mínimo (días)
--tof-max          Tiempo de vuelo máximo (días)
--criterio         dv_total | c3
--sin-captura      No incluir delta-v de captura en el cálculo
--paso             Paso de la malla de fechas (días)
--png               Guardar el porkchop de la ventana elegida
--ventana          Índice de ventana para el porkchop (base 1)
\end{verbatim}

\section{Interfaz y exportación de datos}

\begin{itemize}
\item La interfaz organiza la configuración de misión, lanzamiento, opciones y eventos en pestañas.
\item La visualización incluye la trayectoria, cuerpos de referencia, fases y telemetría.
\item La escala logarítmica y el marco de vista afectan solo al dibujo; no modifican la dinámica ni los datos exportados.
\item La tecla \textbf{E} exporta el historial completo a \path|.npz|.
\item \textbf{Shift + E} exporta el historial completo a \path|.csv|.
\item La exportación incluye los datos que se hayan resumido o podado de la memoria durante vuelos orbitales.
\item La tecla \textbf{M} ejecuta mediciones de rendimiento del render; esta operación bloquea temporalmente la GUI mientras mide.
\end{itemize}

Los archivos exportados se guardan en la carpeta del proyecto y no sobrescriben archivos existentes: se añade un sufijo numérico si el nombre ya existe.

\section{Pruebas}

La suite focalizada se incorporará al repositorio en un commit próximo (archivo \path|test_cohete.py|). Desde la carpeta del repositorio, una vez presente:

\begin{verbatim}
py -m pytest -q test_cohete.py
\end{verbatim}

La suite selecciona por defecto el archivo \path|cohete_vX.Y.Z.py| de mayor versión numérica que encuentre en la carpeta. Para fijar explícitamente la versión 12.13.6 en PowerShell:

\begin{verbatim}
$env:COHETE_ARCHIVO = "cohete_v12.13.6.py"
py -m pytest -q test_cohete.py
Remove-Item Env:COHETE_ARCHIVO
\end{verbatim}

Estas pruebas son focalizadas y no sustituyen una regresión de misiones completas ni una validación física externa.

Si se dispone del ejecutor y de un JSON de referencia compatible, se puede hacer una comparación de regresión. El JSON de referencia debe corresponder a los escenarios y criterios definidos por ese ejecutor. Una comparación correcta significa coincidencia con esa referencia, no que todos los escenarios hayan completado una misión real ni que el modelo físico esté validado externamente.

\section{Validador de efemérides}

El validador (\path|validador_horizons.py|) se incorporará al repositorio en un commit próximo. Requiere conexión a Internet. Una vez presente, junto al archivo del motor, ejecutar:

\begin{verbatim}
py .\validador_horizons.py .\cohete_v12.13.6.py --timeout 60
\end{verbatim}

El informe se guarda junto al motor como:

\begin{verbatim}
validacion_horizons_verlet_v12.13.6.json
\end{verbatim}

El validador consulta JPL Horizons en $T_0$ y en $T_0 + 1, 2, 5$ y $10$ días, compara posición y velocidad de Sol y Luna con la propagación Verlet y actualiza el JSON después de cada comparación. Si una consulta falla, conserva un informe parcial con el diagnóstico; un informe parcial nunca se considera validación aprobada.

La validación externa requiere red. Los resultados dependen de la época, del marco, de las unidades y de las opciones de consulta; deben conservarse junto con la versión exacta del motor que los produjo.

\section{Cambios destacados de la versión 12.13.6}

\begin{itemize}
\item \path|_k_quema| diferencia no convergencia, falta de combustible, ausencia de empuje neto y estancamiento de energía; el estancamiento se detecta tras 100 pasos sin progreso resoluble.
\item El planificador TLI rechaza una predicción cuando no se confirma que se haya pasado el periselene dentro del horizonte más la extensión configurada.
\item El proveedor \path|Horizons+Verlet| conserva e interpola la tabla XYZ 3D original con Hermite, sin construir una tabla auxiliar artificial para z.
\item Se valida que \path|T_MAX_LUNAR| supere el horizonte de transferencia más la extensión al periselene; \path|Horizons+Verlet| limita la preintegración a 30 días.
\item La GUI advierte cuando no hay validación empírica para el hash exacto del motor y la época seleccionada, o cuando el informe no supera los umbrales.
\item \path|Simulacion.__deepcopy__| omite el proceso externo y el generador de planificación, que no son copiables.
\item La caché de efemérides SPICE utiliza política LRU y conserva únicamente los lotes necesarios para interpolar.
\item Se separan los diagnósticos de \path|advance_to| para tiempos no finitos y saltos excesivos; la derivada rechaza masas no positivas o no finitas.
\item Se unifica el transporte y el parseo de las consultas 2D/3D a Horizons; errores de programación no se convierten silenciosamente en respaldo kepleriano.
\item Se documentan y prueban el criterio de corte asimétrico entre escape y TLI.
\item Se habilita \path|nogil=True| en kernels seleccionados y se ajustan operaciones de distancia en bucles calientes tras un benchmark local; no se paraleliza el planificador.
\end{itemize}

\section{Estructura conceptual del código}

El archivo principal es monolítico, pero mantiene capas funcionales diferenciadas:

\begin{enumerate}
\item Constantes físicas, registro de cuerpos, efemérides y marcos de referencia.
\item Física pura: atmósfera, arrastre, mareas, derivadas, elementos orbitales, guiado e integradores.
\item Datos y configuración: \path|Config|, etapas, estado del cohete y registro de eventos.
\item Simulación: propagación, fases y máquina de estados de misión.
\item Interfaz gráfica Tkinter/Matplotlib.
\item Gráficas finales, exportación y resúmenes.
\end{enumerate}

Estructura del repositorio:

\begin{verbatim}
simulador-de-vuelo-espacial/
+-- src/simulador/
|   +-- cohete.py      # Simulador completo (v12.13.6)
|   +-- __init__.py
|   +-- __main__.py
+-- requirements.txt
+-- requirements-dev.txt
+-- pyproject.toml
+-- LICENSE
+-- README.md
\end{verbatim}

\section{Limitaciones conocidas y trabajo futuro}

\begin{itemize}
\item La nave se modela en 2D; no se representa toda la geometría tridimensional de una misión real.
\item La reentrada no incluye aerotermodinámica ni ablación: se confirma por cruce de altitud.
\item Los modos \path|tli| y \path|escape| son modelos simplificados de alta energía y no equivalen a una navegación translunar completa con correcciones de trayectoria.
\item La captura lunar LOI es opcional y provisional en 2D.
\item El modelo \path|Horizons+Verlet| tiene una discrepancia solar medida que supera el umbral configurado; la causa está siendo investigada. No se debe presentar la ruta completa como validada.
\item La comparación Tierra--Luna A/B para estudiar el acoplamiento gravitatorio Luna $\rightarrow$ movimiento relativo Sol--Tierra es un experimento separado y no forma parte de las ecuaciones de la versión 12.13.6.
\item La arquitectura sigue concentrada en un archivo principal; la modularización es una posible evolución futura, pero no se ha realizado en esta versión.
\end{itemize}

\section{Licencia y contribuciones}

Este proyecto se distribuye bajo la \textbf{licencia MIT} (ver el archivo \path|LICENSE| del repositorio). Copyright \copyright{} 2026 Mateo Chavarria Gonzalez.

Las contribuciones deberían incluir pruebas reproducibles, unidades explícitas, descripción del modelo físico afectado y comparación contra una referencia cuando se modifique la dinámica.

\end{document}
