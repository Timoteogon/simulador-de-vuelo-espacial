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

\title{\textbf{Simulador de cohetes 2D}\\[4pt]\large VersiÃ³n 12.13.6}
\author{Mateo Chavarria Gonzalez}
\date{2026}

\begin{document}
\maketitle

\begin{quote}\itshape
\textbf{Estado del proyecto:} herramienta de simulaciÃ³n y anÃ¡lisis preliminar. No es software de vuelo, no estÃ¡ certificado y sus resultados no deben utilizarse como soluciÃ³n operacional de navegaciÃ³n o guiado de una misiÃ³n real.
\end{quote}

\section{CaracterÃ­sticas principales}

\begin{itemize}
\item Cohete genÃ©rico de dos etapas, con un conjunto de parÃ¡metros iniciales inspirado en un lanzador tipo Falcon~9.
\item Modelo de movimiento del vehÃ­culo en \textbf{2D}, en el plano orbital adoptado por la simulaciÃ³n.
\item Gravedad central terrestre con variaciÃ³n de la aceleraciÃ³n gravitatoria con la distancia.
\item AtmÃ³sfera exponencial y arrastre aerodinÃ¡mico; el coeficiente de arrastre puede depender del nÃºmero de Mach.
\item Empuje dependiente de la presiÃ³n ambiente, con Ã¡rea de salida de tobera por etapa.
\item Masa y delta-v propulsivo integrados como variables de estado durante las fases con empuje.
\item Gravity turn, separaciÃ³n de etapas, regulaciÃ³n de empuje y lÃ­mite de aceleraciÃ³n para el perfil tripulado.
\item IntegraciÃ³n hÃ­brida con \path{solve_ivp} (RK45) y kernels RK4/Verlet compilados con Numba.
\item LocalizaciÃ³n de eventos de corte e impacto, interpolaciÃ³n de Hermite y control explÃ­cito de fallos numÃ©ricos.
\item Historial de vuelo con almacenamiento temporal en disco, exportaciÃ³n a NPZ/CSV y grÃ¡ficas.
\item Interfaz Tkinter con visualizaciÃ³n de trayectoria, telemetrÃ­a, fases y eventos.
\item CÃ¡lculo independiente de ventanas interplanetarias mediante elementos keplerianos aproximados y Lambert.
\end{itemize}

\section{Modos de misiÃ³n}

\begin{longtable}{@{}l p{4.6cm} p{7.4cm}@{}}
\caption{Modos de misiÃ³n disponibles}\\
\toprule
Modo & DescripciÃ³n & Criterio / alcance importante \\
\midrule
\endfirsthead
\toprule
Modo & DescripciÃ³n & Criterio / alcance importante \\
\midrule
\endhead
\bottomrule
\endfoot
\path{satelital} & InserciÃ³n orbital mediante secuencia \emph{burn--coast--burn}. & Busca el apogeo configurado y realiza la circularizaciÃ³n; no aplica lÃ­mite fisiolÃ³gico de aceleraciÃ³n. \\
\path{tripulada} & InserciÃ³n orbital con lÃ­mite de carga g. & Perfiles \path{bcb} y \path{directo}; la Ã³rbita se verifica con periapsis seguro y tolerancia de excentricidad. \\
\path{suborbital} & Vuelo balÃ­stico de un cohete sonda. & Puede cortar por apogeo objetivo o agotar las etapas; continÃºa hasta apogeo/impacto o el lÃ­mite temporal configurado. \\
\path{gto} & Transferencia a Ã³rbita geoestacionaria. & Primer tramo hasta el apogeo objetivo y segunda quema para elevar el periapsis y reducir la excentricidad. \\
\path{tli} & Transferencia de alta energÃ­a hacia la distancia lunar. & Es un modo simplificado de transferencia; alcanzar el apogeo objetivo no garantiza por sÃ­ mismo un encuentro lunar controlado. \\
\path{escape} & Trayectoria hiperbÃ³lica de escape. & El corte se define por energÃ­a especÃ­fica no negativa. \\
\path{deorbit} & DesÃ³rbita y reentrada. & Permite activar la quema por Ã³rbita, por tiempo o mediante planificaciÃ³n para aproximar el impacto al sitio de lanzamiento. No incluye modelo tÃ©rmico de reentrada. \\
\path{lunar} & Ascenso, aparcamiento, planificaciÃ³n TLI y propagaciÃ³n hacia la esfera de influencia lunar. & Utiliza efemÃ©rides consultables en cualquier instante. La captura LOI es opcional y sigue siendo un modelo 2D provisional. \\
\end{longtable}

En el modo \path{tripulada}, \path{perfil_tripulada} admite \path{bcb} y \path{directo}.

\section{Modelo fÃ­sico y numÃ©rico}

\subsection{DinÃ¡mica del cohete}

El vehÃ­culo se propaga en un plano 2D. En el marco terrestre se consideran la gravedad central, el empuje, el arrastre y, cuando se activan, las aceleraciones diferenciales debidas a terceros cuerpos.

La perturbaciÃ³n de un cuerpo externo se expresa como aceleraciÃ³n de marea en el marco geocÃ©ntrico:
\begin{equation}
\vec a_{\mathrm{pert},i}=\mu_i\left(
\frac{\vec r_i-\vec r}{|\vec r_i-\vec r|^3}
-\frac{\vec r_i}{|\vec r_i|^3}
\right),
\end{equation}
donde $\vec r$ es la posiciÃ³n geocÃ©ntrica del cohete y $\vec r_i$ la del cuerpo perturbador. El segundo tÃ©rmino resta la aceleraciÃ³n comÃºn del origen terrestre.

La rotaciÃ³n terrestre, si se activa, contribuye a la velocidad inicial de la plataforma y a la velocidad del aire usada para calcular el arrastre. La simulaciÃ³n no aÃ±ade tÃ©rminos de Coriolis y centrÃ­fugo al marco inercial, para evitar contar dos veces esos efectos.

\subsection{Integradores}

\begin{itemize}
\item \textbf{RK45} (\path|scipy.integrate.solve_ivp|) durante el empuje, cerca de la atmÃ³sfera y en fases que requieren integraciÃ³n adaptativa.
\item \textbf{RK4} en coast libre y condiciones suaves, con subpasos acotados.
\item \textbf{Velocity-Verlet 3D} para generar las tablas de efemÃ©rides de Sol y Luna en la ruta \path|Horizons+Verlet|.
\item \textbf{Hermite cÃºbico} para consultar las tablas entre nodos, usando posiciÃ³n y velocidad.
\item Eventos y bisecciÃ³n/interpolaciÃ³n para localizar cortes, cruces y contactos con precisiÃ³n temporal y espacial controlada.
\end{itemize}

La dinÃ¡mica de masa variable estÃ¡ integrada durante el empuje. El delta-v propulsivo se acumula a partir de la aceleraciÃ³n del motor, y no se confunde con el cambio neto de rapidez causado tambiÃ©n por gravedad y arrastre.

\subsection{EfemÃ©rides y alcance del modelo}

El registro del cÃ³digo contiene Sol, Luna y planetas. Sin embargo, \textbf{la presencia de un cuerpo en el registro no significa que su perturbaciÃ³n estÃ© activa en la simulaciÃ³n}. El cohete es una partÃ­cula de prueba: no modifica las trayectorias de los cuerpos celestes. La configuraciÃ³n general mantiene desactivadas por defecto las perturbaciones opcionales de terceros cuerpos.

La ruta \path|Horizons+Verlet| usa un acoplamiento simplificado y unilateral: el Sol se propaga con el modelo Sol--Tierra de dos cuerpos, mientras que la Luna incluye el tÃ©rmino de marea solar; no es un integrador N-cuerpos mutuo.

Para el modo lunar:
\begin{itemize}
\item \path|efemerides_lunar = "horizons_verlet"| es la opciÃ³n predeterminada. Consulta condiciones iniciales en JPL Horizons y propaga Sol y Luna con un modelo simplificado de Verlet 3D.
\item \path|efemerides_lunar = "spice"| usa la ruta SPICE mediante PyMCel.
\item El modo lunar no acepta silenciosamente una efemÃ©ride kepleriana de respaldo como sustituto de una fuente vÃ¡lida.
\item La trayectoria del cohete continÃºa siendo 2D. La coordenada fuera del plano de la Luna se utiliza para advertir cuando un contacto 2D no representa un contacto fÃ­sico 3D.
\item La captura LOI estÃ¡ desactivada por defecto (\path|LUNAR_LOI = False|). Cuando se activa, la captura y verificaciÃ³n orbital siguen siendo provisionales y 2D.
\item En \path|Horizons+Verlet|, \path|T_MAX_LUNAR| estÃ¡ limitado a 30 dÃ­as por el coste de preintegrar la tabla.
\end{itemize}

El modelo no incluye perturbaciones planetarias en la propagaciÃ³n Verlet Sol--Tierra--Luna, ni J2 terrestre ni relatividad. Esa es una delimitaciÃ³n deliberada del modelo, no una afirmaciÃ³n de que esos efectos sean siempre despreciables.

\section{ValidaciÃ³n de Horizons+Verlet (estado medido)}

La versiÃ³n 12.13.6 incorpora un aviso en la interfaz cuando no existe un informe de validaciÃ³n que corresponda \textbf{a la misma versiÃ³n del motor y a la misma Ã©poca inicial}.

Se ejecutÃ³ la comparaciÃ³n para la Ã©poca \path|2026-01-01T00:00:00 UTC|, con paso interno Verlet de 20 s, tabla cada 600 s y muestras en 1, 2, 5 y 10 dÃ­as. Los errores observados fueron:

\begin{table}[h]
\centering
\caption{Errores de posiciÃ³n y velocidad respecto de JPL Horizons}
\begin{tabular}{@{}r r r r r@{}}
\toprule
Tiempo desde $T_0$ & Luna: pos. (km) & Luna: vel. (m/s) & Sol: pos. (km) & Sol: vel. (m/s) \\
\midrule
1 dÃ­a  & 0{,}004952 & 0{,}000115 & 140{,}790     & 3{,}258389 \\
2 dÃ­as & 0{,}019642 & 0{,}000223 & 560{,}394     & 6{,}461563 \\
5 dÃ­as & 0{,}110845 & 0{,}000532 & 3{,}348{,}822 & 14{,}997956 \\
10 dÃ­as& 0{,}439146 & 0{,}000850 & 11{,}537{,}270& 23{,}677544 \\
\bottomrule
\end{tabular}
\end{table}

El validador utiliza umbrales de 5 km para la posiciÃ³n lunar y 50 km para la posiciÃ³n solar. En esta ejecuciÃ³n:
\begin{itemize}
\item \textbf{Luna:} queda dentro del umbral configurado; error mÃ¡ximo de posiciÃ³n de 0{,}439146 km a 10 dÃ­as.
\item \textbf{Sol:} supera ampliamente el umbral configurado; error mÃ¡ximo de posiciÃ³n de 11{,}537{,}270 km a 10 dÃ­as.
\item \textbf{Resultado global:} \path|NO ACEPTADA| segÃºn los umbrales actuales.
\end{itemize}

Por tanto, no debe describirse la ruta completa \path|Horizons+Verlet| como validada globalmente. La buena coincidencia lunar de esta Ã©poca no elimina la discrepancia solar ni demuestra que el resultado se mantenga para otras fechas. La causa de la discrepancia solar estÃ¡ en investigaciÃ³n; no se ha modificado la fÃ­sica para ocultarla.

El informe leÃ­do por el motor se llama \path|validacion_horizons_verlet_v12.13.6.json| y debe estar junto al archivo del motor. El motor comprueba el hash SHA-256, la Ã©poca, el esquema, la configuraciÃ³n del integrador y los umbrales antes de aceptar el informe.

\section{Requisitos}

\begin{itemize}
\item Python 3.9 o superior (por el uso de anotaciones genÃ©ricas integradas como \path|list[str]| y \path|tuple[...]|).
\item NumPy, SciPy, Matplotlib, Numba, Astropy, SpiceyPy, PyMCel.
\item Tkinter, para la interfaz grÃ¡fica (normalmente incluido en la distribuciÃ³n de Python para Windows).
\item Plotly, opcional: se utiliza para las grÃ¡ficas finales interactivas.
\item Pytest, opcional: necesario para ejecutar la suite de pruebas.
\end{itemize}

> PyMCel se importa como dependencia obligatoria para obtener constantes fÃ­sicas. Aunque la ruta lunar \path|Horizons+Verlet| parte de consultas a Horizons, el mÃ³dulo actual tambiÃ©n importa PyMCel, Astropy y SpiceyPy al cargar el motor.

\section{InstalaciÃ³n y ejecuciÃ³n}

Clona el repositorio e instala las dependencias desde PowerShell o una terminal:

\begin{verbatim}
git clone https://github.com/Timoteogon/simulador-de-vuelo-espacial.git
cd simulador-de-vuelo-espacial
python -m pip install -r requirements.txt
\end{verbatim}

En Windows tambiÃ©n se puede usar \path|py| en lugar de \path|python|.

La instalaciÃ³n como paquete permite ejecutar el simulador desde cualquier carpeta:

\begin{verbatim}
python -m pip install .
py -m simulador
\end{verbatim}

Esto abre la interfaz grÃ¡fica Tkinter. En ella se puede configurar el modo de misiÃ³n, el perfil tripulado, los parÃ¡metros de lanzamiento, los objetivos y las opciones de visualizaciÃ³n; despuÃ©s se inicia o reinicia la simulaciÃ³n desde la propia interfaz.

Si Tkinter no estÃ¡ disponible, el mÃ³dulo puede cargarse para el modo de consola de ventanas interplanetarias, pero no podrÃ¡ abrir la GUI.

\subsection{CÃ¡lculo de ventanas interplanetarias por consola}

El subcomando \path|--ventanas| calcula ventanas mediante elementos planetarios keplerianos aproximados de Standish y un mÃ©todo de Lambert. \textbf{No propaga el vuelo del cohete ni verifica una misiÃ³n completa}: estima fechas y condiciones de salida/llegada.

Ejemplo Tierra--Marte:

\begin{verbatim}
py -m simulador --ventanas tierra marte 2026-01-01 2029-01-01
\end{verbatim}

Opciones disponibles:

\begin{verbatim}
--c3-max           LÃ­mite de C3 (km2/s2)
--dv-salida-max    LÃ­mite de delta-v de salida (km/s)
--vinf-llegada-max LÃ­mite de velocidad hiperbÃ³lica de llegada (km/s)
--tof-min          Tiempo de vuelo mÃ­nimo (dÃ­as)
--tof-max          Tiempo de vuelo mÃ¡ximo (dÃ­as)
--criterio         dv_total | c3
--sin-captura      No incluir delta-v de captura en el cÃ¡lculo
--paso             Paso de la malla de fechas (dÃ­as)
--png               Guardar el porkchop de la ventana elegida
--ventana          Ãndice de ventana para el porkchop (base 1)
\end{verbatim}

\section{Interfaz y exportaciÃ³n de datos}

\begin{itemize}
\item La interfaz organiza la configuraciÃ³n de misiÃ³n, lanzamiento, opciones y eventos en pestaÃ±as.
\item La visualizaciÃ³n incluye la trayectoria, cuerpos de referencia, fases y telemetrÃ­a.
\item La escala logarÃ­tmica y el marco de vista afectan solo al dibujo; no modifican la dinÃ¡mica ni los datos exportados.
\item La tecla \textbf{E} exporta el historial completo a \path|.npz|.
\item \textbf{Shift + E} exporta el historial completo a \path|.csv|.
\item La exportaciÃ³n incluye los datos que se hayan resumido o podado de la memoria durante vuelos orbitales.
\item La tecla \textbf{M} ejecuta mediciones de rendimiento del render; esta operaciÃ³n bloquea temporalmente la GUI mientras mide.
\end{itemize}

Los archivos exportados se guardan en la carpeta del proyecto y no sobrescriben archivos existentes: se aÃ±ade un sufijo numÃ©rico si el nombre ya existe.

\section{Pruebas}

La suite focalizada se incorporarÃ¡ al repositorio en un commit prÃ³ximo (archivo \path|test_cohete.py|). Desde la carpeta del repositorio, una vez presente:

\begin{verbatim}
py -m pytest -q test_cohete.py
\end{verbatim}

La suite selecciona por defecto el archivo \path|cohete_vX.Y.Z.py| de mayor versiÃ³n numÃ©rica que encuentre en la carpeta. Para fijar explÃ­citamente la versiÃ³n 12.13.6 en PowerShell:

\begin{verbatim}
$env:COHETE_ARCHIVO = "cohete_v12.13.6.py"
py -m pytest -q test_cohete.py
Remove-Item Env:COHETE_ARCHIVO
\end{verbatim}

Estas pruebas son focalizadas y no sustituyen una regresiÃ³n de misiones completas ni una validaciÃ³n fÃ­sica externa.

Si se dispone del ejecutor y de un JSON de referencia compatible, se puede hacer una comparaciÃ³n de regresiÃ³n. El JSON de referencia debe corresponder a los escenarios y criterios definidos por ese ejecutor. Una comparaciÃ³n correcta significa coincidencia con esa referencia, no que todos los escenarios hayan completado una misiÃ³n real ni que el modelo fÃ­sico estÃ© validado externamente.

\section{Validador de efemÃ©rides}

El validador (\path|validador_horizons.py|) se incorporarÃ¡ al repositorio en un commit prÃ³ximo. Requiere conexiÃ³n a Internet. Una vez presente, junto al archivo del motor, ejecutar:

\begin{verbatim}
py .\validador_horizons.py .\cohete_v12.13.6.py --timeout 60
\end{verbatim}

El informe se guarda junto al motor como:

\begin{verbatim}
validacion_horizons_verlet_v12.13.6.json
\end{verbatim}

El validador consulta JPL Horizons en $T_0$ y en $T_0 + 1, 2, 5$ y $10$ dÃ­as, compara posiciÃ³n y velocidad de Sol y Luna con la propagaciÃ³n Verlet y actualiza el JSON despuÃ©s de cada comparaciÃ³n. Si una consulta falla, conserva un informe parcial con el diagnÃ³stico; un informe parcial nunca se considera validaciÃ³n aprobada.

La validaciÃ³n externa requiere red. Los resultados dependen de la Ã©poca, del marco, de las unidades y de las opciones de consulta; deben conservarse junto con la versiÃ³n exacta del motor que los produjo.

\section{Cambios destacados de la versiÃ³n 12.13.6}

\begin{itemize}
\item \path|_k_quema| diferencia no convergencia, falta de combustible, ausencia de empuje neto y estancamiento de energÃ­a; el estancamiento se detecta tras 100 pasos sin progreso resoluble.
\item El planificador TLI rechaza una predicciÃ³n cuando no se confirma que se haya pasado el periselene dentro del horizonte mÃ¡s la extensiÃ³n configurada.
\item El proveedor \path|Horizons+Verlet| conserva e interpola la tabla XYZ 3D original con Hermite, sin construir una tabla auxiliar artificial para z.
\item Se valida que \path|T_MAX_LUNAR| supere el horizonte de transferencia mÃ¡s la extensiÃ³n al periselene; \path|Horizons+Verlet| limita la preintegraciÃ³n a 30 dÃ­as.
\item La GUI advierte cuando no hay validaciÃ³n empÃ­rica para el hash exacto del motor y la Ã©poca seleccionada, o cuando el informe no supera los umbrales.
\item \path|Simulacion.__deepcopy__| omite el proceso externo y el generador de planificaciÃ³n, que no son copiables.
\item La cachÃ© de efemÃ©rides SPICE utiliza polÃ­tica LRU y conserva Ãºnicamente los lotes necesarios para interpolar.
\item Se separan los diagnÃ³sticos de \path|advance_to| para tiempos no finitos y saltos excesivos; la derivada rechaza masas no positivas o no finitas.
\item Se unifica el transporte y el parseo de las consultas 2D/3D a Horizons; errores de programaciÃ³n no se convierten silenciosamente en respaldo kepleriano.
\item Se documentan y prueban el criterio de corte asimÃ©trico entre escape y TLI.
\item Se habilita \path|nogil=True| en kernels seleccionados y se ajustan operaciones de distancia en bucles calientes tras un benchmark local; no se paraleliza el planificador.
\end{itemize}

\section{Estructura conceptual del cÃ³digo}

El archivo principal es monolÃ­tico, pero mantiene capas funcionales diferenciadas:

\begin{enumerate}
\item Constantes fÃ­sicas, registro de cuerpos, efemÃ©rides y marcos de referencia.
\item FÃ­sica pura: atmÃ³sfera, arrastre, mareas, derivadas, elementos orbitales, guiado e integradores.
\item Datos y configuraciÃ³n: \path|Config|, etapas, estado del cohete y registro de eventos.
\item SimulaciÃ³n: propagaciÃ³n, fases y mÃ¡quina de estados de misiÃ³n.
\item Interfaz grÃ¡fica Tkinter/Matplotlib.
\item GrÃ¡ficas finales, exportaciÃ³n y resÃºmenes.
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
\item La nave se modela en 2D; no se representa toda la geometrÃ­a tridimensional de una misiÃ³n real.
\item La reentrada no incluye aerotermodinÃ¡mica ni ablaciÃ³n: se confirma por cruce de altitud.
\item Los modos \path|tli| y \path|escape| son modelos simplificados de alta energÃ­a y no equivalen a una navegaciÃ³n translunar completa con correcciones de trayectoria.
\item La captura lunar LOI es opcional y provisional en 2D.
\item El modelo \path|Horizons+Verlet| tiene una discrepancia solar medida que supera el umbral configurado; la causa estÃ¡ siendo investigada. No se debe presentar la ruta completa como validada.
\item La comparaciÃ³n Tierra--Luna A/B para estudiar el acoplamiento gravitatorio Luna $\rightarrow$ movimiento relativo Sol--Tierra es un experimento separado y no forma parte de las ecuaciones de la versiÃ³n 12.13.6.
\item La arquitectura sigue concentrada en un archivo principal; la modularizaciÃ³n es una posible evoluciÃ³n futura, pero no se ha realizado en esta versiÃ³n.
\end{itemize}

\section{Licencia y contribuciones}

Este proyecto se distribuye bajo la \textbf{licencia MIT} (ver el archivo \path|LICENSE| del repositorio). Copyright \copyright{} 2026 Mateo Chavarria Gonzalez.

Las contribuciones deberÃ­an incluir pruebas reproducibles, unidades explÃ­citas, descripciÃ³n del modelo fÃ­sico afectado y comparaciÃ³n contra una referencia cuando se modifique la dinÃ¡mica.

\end{document}
