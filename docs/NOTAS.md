# Notas de desarrollo

Desde 12.13.8 el motor vive en un único archivo, `src/simulador/cohete.py`, sin versión en el nombre.
La versión se lleva en `pyproject.toml`, `simulador.__version__`, el encabezado de cambios de `cohete.py`
y los mensajes de commit. Las notas por versión anteriores (hasta 12.13.6) no se incluyen aquí.

## 12.13.9 — robustez ante estados no finitos
Revisión externa de 12.13.8 evaluada contra el código; se aplicó lo verificado:
- Estado no finito tras integrar → fallo controlado del integrador (antes `NaN < 0` era falso y el impacto nunca se detectaba).
- `_elementos_orbitales` rechaza estados no finitos; las funciones de evento de `solve_ivp` devuelven "sin corte".
- `_z_luna` ya no oculta errores de programación; `_error_refresco_visto` se inicializa en el constructor.

Revisado y descartado (con motivo): `_kepler_E` y `_estado_pymcel` (el respaldo Kepler de Sol/Luna tiene e < 0,06 y siempre converge;
añadir `RuntimeError` al `except` ocultaría la guarda de la caché); copia «solo física» en `_error_base_gen` (copiar el historial cuesta
~0,4 ms, no 40 ms); tolerancia de `_k_quema` (el error de Δv es ~1e-7 m/s, no 0,08 m/s); validación de booleanos (ya existe).
Pendientes sin urgencia: constantes con nombre, `logging`, partir `Config`, dividir el monolito en módulos.

## 12.13.8 — carga de SPICE y efemérides lunares con REBOUND
La física del vuelo no cambia.

### Carga de SPICE
pymcel 0.9.25 solo descarga y carga (`furnsh`) los kernels dentro de `consulta_spice` si
`pymcel/data/kernels.txt` NO existe, y `_preparar_datos_pymcel` lo crea antes. En un proceso nuevo los kernels
nunca se cargaban y SPICE fallaba con `SpiceMISSINGTIMEINFO` (el motor caía a Kepler con un aviso).
`_asegurar_kernels_spice()` copia los datos de pymcel, descarga los kernels grandes si no hay ningún `.bsp`
(~100 MB, una vez) y hace `furnsh` una sola vez por proceso; `_eph_cargar_lote` la llama antes de consultar.

Sin verificar con el motor completo y kernels reales en el entorno de desarrollo (sin red): se probó con dobles
de prueba y con `herramientas/validar_rebound.py`, que usa la misma secuencia y en la máquina del autor sí cargó SPICE.

### Proveedor REBOUND (`efemerides_lunar="rebound"`, por defecto)
Mismo contrato que `_ProveedorHorizonsVerlet` (tabla de 600 s + Hermite 3D); cambia cómo se calcula la tabla:
REBOUND/IAS15 con masas puntuales (G=1, masas = GM).
- **Con SPICE:** Sol, Tierra, Luna y los 7 baricentros planetarios (ids 1, 2, 4 a 8) en el marco baricéntrico.
- **Sin SPICE:** Sol y Luna geocéntricos de Horizons, Tierra en reposo en el origen, sin planetas (con aviso).
- Los GM de los planetas son valores tipo DE440 escritos a mano en `_GM_PLANETAS_BARICENTROS`.

Validación contra SPICE (`herramientas/validar_rebound.py`, 2026-01-01, 10 días, posición 3D):

| | Verlet | REBOUND Sol-Tierra-Luna | REBOUND + planetas |
|---|---|---|---|
| Luna | 0,44 km | 0,21 km | 0,33 km |
| Sol | 11 537 km | 49,5 km | 0,07 km |

Umbrales del motor: Luna 5 km, Sol 50 km. El Verlet falla en el Sol porque modela Sol-Tierra a dos cuerpos y
omite el vaivén de la Tierra por la Luna (±4 670 km): con la Luna sin masa coincide con REBOUND a 1e-4 km, así
que el integrador es exacto y lo que falta es el modelo. Es una sola fecha y 10 días; conviene repetirlo con otras.

Se mantienen `"spice"` y `"horizons_verlet"`. El límite de 30 días de `T_MAX_LUNAR` aplica también a `rebound`.
La casilla de la GUI «Lunar: usar SPICE» marcada elige `spice`; desmarcada elige `rebound`.

## 12.13.7
- `_estado_pymcel` poda la caché solo al cargar un lote (antes, en cada consulta): 13,5 µs → 4,9 µs por llamada.
- `_k_quema` con energía objetivo no finita devuelve estado 1 (antes caía en «sin empuje neto», diagnóstico erróneo).
- Harness: mock de `verificar_gui_bucle`, `rp/ra` de órbitas casi parabólicas con rtol 1e-5, `ATOL` 1e-5 m,
  legacy con rtol 1e-10 (redondeo hypot → sqrt), `--solo-ref` para generar referencias con motores históricos.
- Referencias legacy (`tests/regresion_ref_v11.21.2.json`, `v12.0.1`, `v12.2.0`) generadas con los motores
  históricos reales.
- El GTO no necesita cambios de `dt`: la fase durante el coast es `coasting` y cuesta ~4 s. El escape completo
  (122 914 pasos, ~56 s) queda como test opcional (`COHETE_LENTOS=1`).
- Sin cambiar: el barrido TLI de un vuelo lunar nominal imprime «3/8 instantes omitidos por fallo numérico» aunque
  sea esperable (periselene no alcanzado); separar ese caso del fallo numérico real queda pendiente.

## Pruebas
`pytest -q tests/test_cohete.py`: 38 pruebas + 1 opcional. `python tests/regresion_cohete.py` compara con
`tests/regresion_ref.json` (solo comparable entre la misma plataforma y versiones de bibliotecas; la referencia
actual se generó en Windows).

## Notación (alineada con los apuntes del curso de Mecánica Celeste)
La física y las fórmulas (Laplace, M.A.R.E., E.R.E., Kepler, CRTBP, Standish) ya coincidían con los apuntes; se unificaron los nombres:
- `h` queda solo para el M.A.R.E. (`Lz` → `h` en `_elementos_orbitales`; `_conica_selenocentrica` ya usaba `h`).
  La altitud pasa a `alt` (`fuerza_arrastre`, `fuerzas_congeladas`, `derivadas_v12`), el paso de integración a `dt`
  (`_rk4_kernel`, `_herm_*`, `_k_coast`, `_k_quema`, `_estado_pymcel`, ...) y el subpaso de la quema LOI a `dt_sub`.
  Los `h00, h10, h01, h11` de Hermite se conservan (notación estándar de la base).
- `omega` (ω, minúscula) es solo el argumento/longitud del periapsis (`_elementos_orbitales`, 2D con Ω=0).
  La velocidad angular pasa a `Omega` (Ω): `crucero_omega` → `crucero_Omega`, `_omega_rotacion` → `_Omega_tierra`,
  `LeyV12.Omega` y los parámetros de `derivadas_crtbp_rotante`, `_cross_z`, etc. `OMEGA_T` (Ω_⊕) no cambia.
- `MU_MASA_TL` → `RAZON_MASA_TL` (adimensional), `MU_TOTAL_TL` → `MU_SISTEMA_TL`, se borró `M_T`.
- `_ELEM_PLANETAS` usa la convención de Standish (a, e, i, L, ϖ, Ω) con ϖ = Ω + ω; no es intercambiable con `omega` de 2D.
- Si se añade el virial, usar `E_total` para no chocar con la anomalía excéntrica `E`.

Verificado: pytest (38 + 1 omitida) y la regresión completa (EXIT=0, mismos resultados contra la referencia).
