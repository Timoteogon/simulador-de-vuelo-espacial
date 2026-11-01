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
- **Etapa III:** Dinámica de encuentros lunares, propagación de efemérides con REBOUND (por defecto), SPICE o Horizons+Verlet y validación frente a SPICE y JPL Horizons.

## Requisitos

El proyecto utiliza librerías de Python para cálculo científico, integración numérica y visualización:

- `numpy`
- `scipy`
- `matplotlib`
- `numba`
- `astropy`
- `spiceypy`
- `pymcel`
- `rebound` (efemérides lunares por defecto)
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

## 1. Dinámica traslacional del cohete

El vehículo se propaga en dos dimensiones. En el marco geocéntrico, la aceleración incluye la gravedad terrestre, el empuje, el arrastre atmosférico y, cuando están habilitadas, las perturbaciones de terceros cuerpos:

$$
\ddot{\mathbf r} =
-\frac{\mu_T}{r^3}\mathbf r
+\frac{T}{m}\hat{\mathbf u}_T
-\frac{\rho C_D A}{2m}
\|\mathbf v_{\mathrm{rel}}\|\mathbf v_{\mathrm{rel}}
+\mathbf a_{\mathrm{pert}}.
$$

Aquí, $\mathbf r$ es la posición geocéntrica, $r=\|\mathbf r\|$, $\mu_T$ es el parámetro gravitacional terrestre, $T$ es el empuje, $m$ es la masa instantánea, $\hat{\mathbf u}_T$ es la dirección del empuje, $\rho$ es la densidad atmosférica, $C_D$ es el coeficiente de arrastre y $A$ es el área de referencia. La velocidad $\mathbf v_{\mathrm{rel}}$ se mide respecto al aire.


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

Las efemérides de Sol y Luna del modo lunar se propagan por defecto con REBOUND (IAS15, N cuerpos). La ruta alternativa Horizons+Verlet usa el esquema *velocity-Verlet*:

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
│   ├── cohete.py        # Motor principal del simulador
│   ├── __init__.py
│   └── __main__.py
├── tests/
│   ├── test_cohete.py       # Pruebas unitarias (pytest)
│   ├── regresion_cohete.py  # Regresión contra regresion_ref*.json
│   └── regresion_ref*.json  # Referencias (actual y de motores históricos)
├── herramientas/
│   └── validar_rebound.py   # Valida Verlet y REBOUND contra SPICE
├── docs/NOTAS.md        # Historial y notas de desarrollo
├── requirements.txt     # Dependencias directas
├── requirements-dev.txt # Herramientas de desarrollo (pytest)
├── pyproject.toml       # Configuración del paquete
├── LICENSE              # Licencia MIT
└── README.md            # Documentación principal

```

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
py -m pip install -r requirements-dev.txt
py -m pytest -q tests/test_cohete.py
```

Para probar otro motor: `$env:COHETE_ARCHIVO = "ruta\otro_cohete.py"`. La prueba lenta del escape completo se activa con `COHETE_LENTOS=1`.

### Regresión contra una referencia

```powershell
py tests\regresion_cohete.py
```

Por defecto prueba `src/simulador/cohete.py` contra `tests/regresion_ref.json`. Para regenerar la referencia usa `--guardar`; las referencias de otras plataformas o versiones de bibliotecas pueden diferir en el redondeo.

Una comparación satisfactoria indica coincidencia con los escenarios y tolerancias de esa referencia. No significa que todas las misiones hayan completado un vuelo ni sustituye la validación física externa.

## Consideraciones y Dependencias Externas

### Efemérides: REBOUND, SPICE y Horizons+Verlet

- **`rebound` (por defecto):** integra Sol, Tierra, Luna y los baricentros planetarios con REBOUND/IAS15, con condiciones iniciales de SPICE. Sin SPICE usa Horizons para Sol y Luna, sin planetas.
- **`spice`:** consulta directa a los kernels (PyMCel/SpiceyPy). La primera vez descarga ~100 MB.
- **`horizons_verlet`:** condiciones iniciales de JPL Horizons y Verlet 3D simplificado.

Las consultas externas requieren Internet y pueden fallar temporalmente.

### Validación frente a SPICE

Época `2026-01-01T00:00:00 UTC`, 10 días, posición 3D (umbrales del motor: Luna 5 km, Sol 50 km):

| | Verlet | REBOUND Sol-Tierra-Luna | REBOUND + planetas |
|---|---:|---:|---:|
| Luna | 0,44 km | 0,21 km | 0,33 km |
| Sol | 11 537 km | 49,5 km | 0,07 km |

El Verlet supera el umbral solar porque trata Sol-Tierra como dos cuerpos y omite el movimiento de la Tierra alrededor del baricentro Tierra-Luna (±4 670 km); REBOUND corrige ese modelo. Es una sola época y horizonte: repite con otras fechas.

```powershell
py herramientas\validar_rebound.py src\simulador\cohete.py 2026-06-15T00:00:00 10
```

### Alcance y simplificaciones

- El cohete se modela en 2D; las efemérides de Sol y Luna de la ruta `Horizons+Verlet` se propagan en 3D.
- Las efemérides con REBOUND incluyen masas puntuales (y planetas si hay SPICE), sin J2 terrestre, figura de la Luna ni relatividad; la ruta Verlet no incorpora planetas.
- En la ruta `Horizons+Verlet`, Sol–Tierra se trata como un problema de dos cuerpos; la Luna incluye la marea solar. No es N-cuerpos mutuo.
- El vehículo no modifica las trayectorias de Sol o Luna: se trata como partícula de prueba.
- La reentrada se confirma mediante cruce de altitud; no incluye modelo térmico ni ablación.
- TLI es una transferencia simplificada y no equivale a navegación translunar con correcciones de trayectoria.
- La captura LOI es opcional, 2D y provisional.
- La discrepancia solar de `Horizons+Verlet` está explicada (modelo a dos cuerpos); esa ruta no debe presentarse como validada. Ver `docs/NOTAS.md`.

## Licencia

El proyecto declara licencia MIT. Consulta el archivo `LICENSE` del repositorio para conocer los términos completos de uso, modificación y distribución.

Las contribuciones deben incluir pruebas reproducibles, unidades explícitas, descripción del modelo físico afectado y comparación contra una referencia cuando se modifique la dinámica.
