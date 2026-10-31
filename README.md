# Simulador de cohetes 2D

**Autor:** Mateo C. Gonzalez  
*Universidad de Antioquia — Medellín, Colombia*  
**GitHub:** [@Timoteogon](https://github.com/Timoteogon)  
**Contacto:** `mateo.chavarriag@udea.edu.co`

Simulador numérico de cohetes de dos etapas desarrollado en Python para estudiar el ascenso, la inserción orbital, las transferencias de alta energía, la desórbita y los encuentros con la Luna. El proyecto combina dinámica orbital, propulsión, aerodinámica, integración numérica, efemérides y una interfaz gráfica interactiva.

> **Estado del proyecto:** herramienta de simulación y análisis preliminar. No es software de vuelo, no está certificado y sus resultados no deben utilizarse como solución operacional de navegación o guiado de una misión real.

## Etapas del proyecto

- **Etapa I:** Modelo de ascenso atmosférico, propulsión de dos etapas, gravedad central, arrastre aerodinámico y guiado tipo *gravity turn*.
- **Etapa II:** Inserción y evolución orbital; perfiles satelitales y tripulados, transferencias GTO, TLI, escape y maniobras de desórbita.
- **Etapa III:** Dinámica de encuentros lunares, propagación de efemérides mediante Horizons+Verlet o SPICE y validación de las trayectorias frente a JPL Horizons.

## Requisitos

El proyecto utiliza librerías de Python para cálculo científico, integración numérica y visualización:

- `numpy`
- `scipy`
- `matplotlib`
- `numba`
- `astropy`
- `spiceypy`
- `pymcel`
- `tkinter` (normalmente incluido con Python en Windows)

Dependencias opcionales:

- `plotly`: gráficas finales interactivas.
- `pytest`: ejecución de las pruebas automatizadas.

```markdown
Instalación de las dependencias principales:
```

```bash
python -m pip install -r requirements.txt
```

```markdown
Para las herramientas de pruebas y gráficas opcionales:
```

```bash
python -m pip install -r requirements-dev.txt

```

## Modelo Físico y Ecuaciones

### 1. Dinámica traslacional del cohete

El vehículo se propaga en dos dimensiones. En el marco geocéntrico, la aceleración incluye la gravedad terrestre, el empuje, el arrastre atmosférico y, cuando están habilitadas, las perturbaciones de terceros cuerpos:

$$
\ddot{\vec r} = -\frac{\mu_T}{r^3}\vec r
+ \frac{T}{m}\hat{u}_T
- \frac{\rho C_D A}{2m}\,|\vec v_{\mathrm{rel}}|\vec v_{\mathrm{rel}}
+ \vec a_{\mathrm{pert}}.
$$

Aquí, $\vec r$ es la posición geocéntrica, $r=|\vec r|$, $\mu_T$ es el parámetro gravitacional terrestre, $T$ es el empuje, $m$ es la masa instantánea, $\hat{u}_T$ es la dirección del empuje, $\rho$ es la densidad atmosférica, $C_D$ es el coeficiente de arrastre y $A$ es el área de referencia. La velocidad $\vec v_{\mathrm{rel}}$ se mide respecto al aire.

La contribución de un cuerpo perturbador $i$, cuando se activa, se expresa como aceleración diferencial o de marea:

$$
\vec a_{\mathrm{pert},i}=\mu_i\left[
\frac{\vec r_i-\vec r}{|\vec r_i-\vec r|^3}
-\frac{\vec r_i}{|\vec r_i|^3}
\right].
$$

### 2. Atmósfera y propulsión

La densidad atmosférica se aproxima mediante un perfil exponencial:

$$
\rho(h)=\rho_0\exp\left(-\frac{h}{H}\right),
$$

donde $h$ es la altitud, $\rho_0$ es la densidad de referencia y $H$ es la escala de altura.

El empuje puede depender de la presión ambiente:

$$
T(h)=T_{\mathrm{vac}}-A_e p_{\mathrm{amb}}(h),
$$

donde $T_{\mathrm{vac}}$ es el empuje en vacío y $A_e$ es el área de salida de la tobera. Durante una quema, la masa disminuye según el flujo másico de propelente. El simulador integra la masa como variable de estado durante el empuje y acumula el delta-v propulsivo a partir de:

$$
\Delta v_{\mathrm{prop}}=\int_{t_0}^{t_1}\frac{T(t)}{m(t)}\,dt.
$$

### 3. Integración numérica

Para las fases con empuje, atmósfera o transitorios se utiliza integración adaptativa con `solve_ivp` (RK45). En fases de coast libre se emplea RK4 con subpasos limitados. Los kernels numéricos seleccionados se compilan con Numba.

Para la propagación de efemérides tridimensionales con Verlet se utiliza el esquema *velocity-Verlet*:

$$
\vec r_{n+1}=\vec r_n+\vec v_n\Delta t+\frac{1}{2}\vec a_n\Delta t^2,
$$

$$
\vec v_{n+1}=\vec v_n+\frac{1}{2}(\vec a_n+\vec a_{n+1})\Delta t.
$$

Entre nodos de las tablas de efemérides se utiliza interpolación cúbica de Hermite con posición y velocidad.

### 4. Energía orbital y criterios de misión

La energía específica de una trayectoria geocéntrica se calcula como:

$$
\varepsilon=\frac{v^2}{2}-\frac{\mu_T}{r}.
$$

En el modo de escape, el criterio energético de corte es $\varepsilon\geq 0$. Los demás modos utilizan sus propios criterios de misión: apogeo proyectado, periapsis seguro, excentricidad, agotamiento de propelente, eventos de impacto o entrada a la esfera de influencia lunar, según corresponda.

## Modos de misión

- **`satelital`:** inserción orbital mediante secuencia *burn–coast–burn* y circularización.
- **`tripulada`:** inserción orbital con límite de aceleración; admite perfiles `bcb` y `directo`.
- **`suborbital`:** vuelo balístico de un cohete sonda hasta apogeo e impacto, sujeto a límites temporales.
- **`gto`:** transferencia a órbita geoestacionaria mediante quema inicial, coast hasta el apogeo y circularización.
- **`tli`:** transferencia de alta energía hacia la distancia lunar. No representa por sí sola un encuentro lunar controlado.
- **`escape`:** trayectoria hiperbólica cuyo corte se determina mediante energía específica no negativa.
- **`deorbit`:** desórbita y reentrada; permite activar la quema por órbita, tiempo o planificación para aproximar el impacto al sitio de lanzamiento.
- **`lunar`:** ascenso a órbita de aparcamiento, planificación TLI y propagación hacia la esfera de influencia lunar. La captura LOI es opcional y provisional en 2D.

## Estructura del Repositorio

```text
simulador-de-vuelo-espacial/
├── src/simulador/
│   ├── cohete.py        # Motor principal del simulador (v12.13.6)
│   ├── __init__.py
│   └── __main__.py
├── requirements.txt     # Dependencias directas
├── requirements-dev.txt # Herramientas de desarrollo (pytest)
├── pyproject.toml       # Configuración del paquete
├── LICENSE              # Licencia MIT
└── README.md            # Documentación principal

```

Los nombres de archivos auxiliares son orientativos: incluye en el repositorio únicamente los que realmente distribuyas junto con el motor.

---

## Cómo ejecutar las simulaciones (Uso)

Clona el repositorio y entra en la carpeta del proyecto:

```bash
git clone https://github.com/Timoteogon/simulador-de-vuelo-espacial.git
cd simulador-de-vuelo-espacial
```

Instala las dependencias y ejecuta el motor:

```powershell
py -m pip install -r requirements.txt
py -m pip install -e .
py -m simulador
Alternativamente, sin instalar el paquete: `py .\src\simulador\cohete.py`
```

La ejecución abre la interfaz gráfica Tkinter. Desde ella se selecciona el modo de misión, el perfil, los parámetros de lanzamiento y las opciones de visualización.

### Ejecución de pruebas

```powershell
py -m pip install pytest
py -m pytest -q test_cohete.py
```

Para fijar explícitamente el motor que deben cargar las pruebas:

```powershell
$env:COHETE_ARCHIVO = "cohete_v12.13.6.py"
py -m pytest -q test_cohete.py
Remove-Item Env:COHETE_ARCHIVO
```

### Regresión contra una referencia

Si se dispone del ejecutor y del JSON de referencia compatibles:

```powershell
py regresion_cohete_v12.12.2.py cohete_v12.13.6.py --ref regresion_ref_v12.13.3.json
```

Una comparación satisfactoria indica coincidencia con los escenarios y tolerancias de esa referencia. No significa que todas las misiones hayan completado un vuelo ni sustituye la validación física externa.

## Consideraciones y Dependencias Externas

### JPL Horizons y SPICE

La ruta `Horizons+Verlet` consulta JPL Horizons para obtener condiciones iniciales y después propaga Sol y Luna con un modelo simplificado de Verlet 3D. La ruta SPICE se encuentra disponible mediante PyMCel/SpiceyPy. Las consultas externas requieren conexión a Internet y pueden fallar temporalmente por disponibilidad del servicio.

### Resultado medido de Horizons+Verlet

Para la época `2026-01-01T00:00:00 UTC`, con paso Verlet de 20 s, tabla cada 600 s y muestras en 1, 2, 5 y 10 días, se obtuvo:

| Tiempo | Error de posición Luna | Error de velocidad Luna | Error de posición Sol | Error de velocidad Sol |
|---:|---:|---:|---:|---:|
| 1 día | 0,004952 km | 0,000115 m/s | 140,790094 km | 3,258389 m/s |
| 2 días | 0,019642 km | 0,000223 m/s | 560,394295 km | 6,461563 m/s |
| 5 días | 0,110845 km | 0,000532 m/s | 3.348,822300 km | 14,997956 m/s |
| 10 días | 0,439146 km | 0,000850 m/s | 11.537,269652 km | 23,677544 m/s |

El validador utiliza umbrales de 5 km para la posición lunar y 50 km para la posición solar. La Luna quedó dentro del umbral, pero el Sol lo superó ampliamente; por tanto, el resultado global fue **NO ACEPTADA**. Esta medición corresponde a una época y a un horizonte concretos, y no demuestra la misma precisión para otras fechas.

Para repetir la validación, coloca `validador_horizons.py` junto al motor y ejecuta:

```powershell
py .\validador_horizons.py .\cohete_v12.13.6.py --timeout 60
```

El informe se guarda como `validacion_horizons_verlet_v12.13.6.json`, junto al motor. Si una consulta falla, el validador conserva un JSON parcial para diagnóstico; un informe parcial no es una validación aprobada.

### Alcance y simplificaciones

- El cohete se modela en 2D; las efemérides de Sol y Luna de la ruta `Horizons+Verlet` se propagan en 3D.
- El modelo de efemérides es simplificado y no incorpora perturbaciones planetarias, J2 terrestre ni relatividad.
- En la ruta `Horizons+Verlet`, Sol–Tierra se trata como un problema de dos cuerpos; la Luna incluye la marea solar. No es un integrador N-cuerpos mutuo.
- El vehículo no modifica las trayectorias de Sol o Luna: se trata como partícula de prueba.
- La reentrada se confirma mediante cruce de altitud; no incluye modelo térmico ni ablación.
- TLI es una transferencia simplificada y no equivale a navegación translunar con correcciones de trayectoria.
- La captura LOI es opcional, 2D y provisional.
- La discrepancia solar medida está siendo investigada. No se debe presentar la ruta completa `Horizons+Verlet` como validada globalmente.

## Licencia

El proyecto declara licencia MIT. Consulta el archivo `LICENSE` del repositorio para conocer los términos completos de uso, modificación y distribución.

Las contribuciones deben incluir pruebas reproducibles, unidades explícitas, descripción del modelo físico afectado y comparación contra una referencia cuando se modifique la dinámica.
