"""
Simulación de cohete 2D — cohete genérico de 2 etapas (Falcon 9 por defecto)

Física: gravedad central, arrastre atmosférico exponencial,
gravity turn clásico, integración numérica con solve_ivp (RK45).

Modos de misión:
  - "satelital": Burn-Coast-Burn clásico, prioriza Δv. Apogeo
    alto (por defecto 1300 km). Sin límite fisiológico.
  - "tripulada": límite de carga g activo (4 g por defecto),
    corredor LEO 180-500 km por debajo de Van Allen. Dos perfiles:
      · "bcb": Burn-Coast-Burn tripulado, igual que el satelital
        pero con g-limit. Corta el motor al alcanzar el apogeo
        objetivo, preserva combustible, coastea hasta el apogeo
        físico y ahí hace la quema de circularización real
        (tolerancia estricta de excentricidad, 0.01).
      · "directo": ascenso continuo sin corte intermedio; la 2ª
        etapa quema sin parar hasta que el periapsis alcanza
        ALTURA_SEGURIDAD (tolerancia laxa de excentricidad, 0.08,
        porque el empuje con v_radial > 0 deja la órbita más
        elíptica).

  El corte de la quema de circularización, en AMBOS perfiles y
  también en satelital, se decide únicamente por periapsis real
  (rp >= ALTURA_SEGURIDAD). No existe ni debe reintroducirse un
  atajo que corte por velocidad/energía sin mirar el periapsis:
  con velocidad radial residual, "rapidez ~ v_circ local" no
  implica periapsis seguro, y cortar por eso puede dejar a la
  tripulación en una órbita que reentra en pocas vueltas.
  Por la misma razón, `_verificar_orbita` exige, en tripulada,
  periapsis >= ALTURA_SEGURIDAD (no solo > 0) antes de declarar
  la órbita alcanzada.

  - "suborbital": cohete sonda / vuelo balístico. Ascenso casi
    vertical (angulo_kick 1.5°), sin circularización ni periapsis
    seguro (ALTURA_SEGURIDAD = 0). Todas las etapas queman en
    secuencia; al agotarse (o, con suborbital_cutoff_por_apogeo,
    al proyectar el apogeo objetivo) la nave pasa a coasting hasta
    el apogeo físico (v_radial < 0) y cae hasta el impacto. Éxito
    si alt_max >= APOGEO_OBJETIVO (menos TOL_APOGEO_SUBORBITAL, 100 m) y
    alt_max > 100 km (Kármán).

  - "gto": transferencia a órbita geoestacionaria. B–C–B con
    apogeo objetivo = 35 786 km: la 2ª etapa enciende al separar y
    se corta cuando el apogeo proyectado alcanza el objetivo; coast
    hasta el apogeo físico (horas) y ahí una segunda quema prógrada
    circulariza. El corte de esa quema se decide por periapsis real
    (rp >= GTO_RP_FINAL, 35 000 km) y excentricidad (< ECC_MAX_GTO,
    0.02), que es también el criterio de éxito. Ojo: el arco tras el
    primer corte es balístico (periapsis bajo el suelo, ecc > 0.8),
    no una GTO con perigeo a 200 km: el guiado no lleva el perigeo
    a esa altura.

  - "tli" / "escape": vuelo de alta energía ("translunar toy"). Mismo
    ascenso que gto, pero una sola quema: corta cuando el apogeo
    proyectado llega a la órbita lunar (tli, distancia al centro de
    384 400 km, es decir APOGEO_OBJETIVO = 384 400 km - R_T de altitud)
    o cuando la energía específica es >= 0 (escape,
    ecc >= 1). Sin circularización, sin phasing, sin correcciones:
    NO es un encuentro lunar controlado. Resultado mutuamente
    excluyente: "llego_luna" (ra >= objetivo y no escape),
    "escape" (ecc >= 1 al corte) o "insuficiente" (sin combustible
    antes de alcanzar el objetivo). Éxito: llego_luna + SOI en tli,
    escape en escape. Desde 11.21.0 el éxito de tli exige además ENTRAR a
    la esfera de influencia lunar (llego_luna es el resultado del
    corte; sin phasing la SOI casi nunca se alcanza y la misión queda
    «NO CUMPLIDA» por T_MAX o por impacto). Usa marco móvil (crucero CRTBP / zona lunar);
    r_max y distancia mínima a la Luna se miden en ECI. La Luna de
    la efeméride y la del CRTBP (circular, congelada al entrar en
    crucero) se separan ~14 000 km a los 3 días y ~29 000 km a los 5,
    por eso la distancia a la Luna es solo informativa.

  - "deorbit": desórbita / reentrada. Sube a órbita como "satelital"
    (apogeo 420 km, periapsis >= 200 km) y, ya en órbita, gira el
    empuje 180° (retrógrado) en el instante que da deorbit_activacion:
    "sitio" (por defecto: la hora de quema se busca para que la caída
    coincida con la base de lanzamiento, que la rotación terrestre va
    moviendo), "orbita" (apenas se detecta la órbita) o "t" (a un
    tiempo dado, para abortar durante el ascenso). Quema hasta que el
    periapsis cae a DEORBIT_RP_OBJETIVO (40 km), con el empuje
    regulado por el Δv que falta. Luego coast y caída hasta el
    impacto. Éxito = periapsis <= 40 km al corte Y reentrada
    confirmada (descenso por DEORBIT_ALT_REENTRADA, 120 km). No hay
    modelo térmico: "reentrada confirmada" es solo el cruce de altitud.
    En esta fase no actúa el corrector de altitud crítica.
    En la GUI, la hora de quema del modo "sitio" se calcula en un
    PROCESO aparte, en paralelo al ascenso (el ascenso es determinista:
    el proceso hijo recorre el mismo camino y devuelve el mismo plan,
    bit a bit). Si el plan no llegó cuando el principal detecta la
    órbita, la simulación espera sin avanzar (el HUD lo indica); si el
    hijo falla, no coincide o tarda demasiado, se planifica como antes.

Cambios de 11.21.1 (endurecimiento; sin cambio de arquitectura):
  - Config._validar_config(): ValueError descriptivo (todos los problemas
    juntos) antes de crear la simulación; la GUI lo muestra en pantalla.
  - _kepler_E: lanza RuntimeError si no converge y ValueError si e o M no
    son válidos (antes devolvía un valor sin avisar).
  - _localizar_corte: parada por tolerancia temporal (1e-7 s) con tope de
    25 iteraciones (antes 40 fijas). Frente a 11.21.0 el instante de corte
    cambia < 6e-7 s y el estado final < 0.1 m / 1e-4 m/s en los 8 modos
    probados: equivalente, no idéntico bit a bit.
  - _msg_fallo_integrador declarado en __init__; _EPH_PASO y
    _EPH_NODOS_LOTE documentados con unidades.
  - TLI al llegar a T_MAX sin SOI: «NO CUMPLIDA (T_MAX sin SOI)» explícito.
  - Esta sección: limitaciones heredadas frente a las de la localización de
    eventos. Nada de la física cambia salvo lo del corte indicado arriba.

Cambios de 12.13.8 (efemérides lunares; el vuelo no cambia):
  - Los archivos del informe de validación Horizons+Verlet ya no llevan versión en el nombre:
    `validacion_horizons_verlet.json` y `validar_horizons_verlet.py` (el informe sigue atado al
    SHA-256 del motor, así que cualquier cambio del archivo lo invalida).
  - Carga de SPICE: pymcel 0.9.25 solo hace `furnsh` en `consulta_spice` si `pymcel/data/kernels.txt`
    NO existe, y `_preparar_datos_pymcel` lo crea antes, así que en un proceso nuevo SPICE fallaba con
    MISSINGTIMEINFO y se caía a Kepler. `_asegurar_kernels_spice()` descarga los kernels grandes si faltan
    y los carga una vez por proceso; `_eph_cargar_lote` la llama antes de consultar.
  - Nuevo `_ProveedorRebound` (`efemerides_lunar="rebound"`, ahora por defecto): tabla de Sol y Luna
    geocéntricos con REBOUND/IAS15 y planetas (con SPICE; sin SPICE usa Horizons y no incluye planetas).
    Contra SPICE a 10 d (2026-01-01): Luna 0,33 km, Sol 0,07 km (Verlet: 0,44 km y 11 500 km, por no
    incluir el vaivén de la Tierra). "spice" y "horizons_verlet" siguen disponibles. Exige `pip install rebound`.
  - La casilla de la GUI desmarcada elige "rebound" (antes "horizons_verlet").

Cambios de 12.13.7 (rendimiento y diagnóstico; la física no cambia):
  - _k_quema: una energía objetivo no finita devuelve 1 (no convergió/no finito). En 12.13.6
    caía en el código 3 y el planificador TLI lo contaba como "sin empuje neto".
  - _estado_pymcel poda la caché de nodos solo cuando carga un lote (la tabla solo crece ahí).
    La poda de 12.13.5 recorría ~144-288 nodos en cada consulta: 13.5 µs frente a 3.6 µs por
    llamada medido en 12.13.4, que en vuelo lunar (varias consultas por paso) pesaba tanto
    como el propio paso.

Cambios de 12.13.6 (auditoría de dinámica, efemérides y robustez; sin ampliar el modelo físico):
  - _k_quema distingue no convergencia (1), combustible (2), sin empuje neto (3) y estancamiento
    de energía (4); el detector de estancamiento requiere 100 pasos sin progreso resoluble. Se
    añade un test Numba compilado con dt=0.25 s, 980 kN y masa inicial de 5000 kg.
  - El planificador TLI rechaza una predicción si al final de horizonte + extensión la velocidad
    radial selenocéntrica aún no es positiva; ya no optimiza usando un mínimo provisional.
  - _ProveedorHorizonsVerlet conserva la tabla XYZ real y la interpola con Hermite 3D; elimina
    la tabla auxiliar de z con columnas artificiales.
  - Config lunar exige T_MAX_LUNAR > horizonte + extensión; Horizons+Verlet limita la preintegración
    a 30 días por rendimiento. La ruta SPICE no tiene ese límite específico.
  - Horizons+Verlet muestra advertencia persistente mientras no exista una validación de la MISMA
    versión del motor y la MISMA época. Se añade validar_horizons_verlet_v12.13.6.py para medir
    posición y velocidad contra consultas JPL Horizons sin añadir planetas ni cambiar la física.
  - Simulacion.__deepcopy__ omite procesos/generadores no copiables; el historial reinicia también
    sus resúmenes. La caché SPICE pasa a LRU y conserva el límite de nodos por clave.
  - Mensajes de advance_to separan época no finita de salto excesivo; derivadas_v12 rechaza masa
    no positiva/no finita, sin sustituirla silenciosamente por 1 kg.
  - Se documentan el criterio asimétrico escape/TLI y el contrato de ContextoMision; el proveedor
    3D expone estado_3d y elimina la tabla z construida mediante columnas artificiales.
  - _fetch_horizons_state y _fetch_horizons_state_3d comparten transporte y parseo; los fallos
    de programación (TypeError/AttributeError) ya no se convierten en respaldo Kepler silencioso.
  - Benchmark local Numba (5 millones de radios): sqrt(x*x+y*y) fue más rápido que hypot en este
    entorno; se usa solo en bucles calientes con escalas orbitales acotadas. _k_coast/_k_quema
    habilitan nogil=True, sin paralelizar ni cambiar el planificador.

Cambios de 12.13.5 (correcciones de auditoría; la física nominal no cambia):
  - APOGEO_OBJETIVO debe ser finito en TODOS los modos salvo "escape"; antes tli y lunar
    aceptaban inf aunque la documentación reservaba ese valor para escape.
  - La caché SPICE mantiene solo los lotes necesarios para interpolar los nodos k y k+1
    (máximo 2 lotes/clave = 288 nodos), además del tope global de 16 claves. Los lotes
    descartados pueden recuperarse desde la caché en disco.
  - El timeout de planificación (900 s) registra un fallo terminal de misión y libera
    el estado de espera; la GUI ya no queda con una misión aparentemente activa.

Cambios de 12.13.4 (guardas defensivas; la física no cambia, referencia idéntica a la de 12.13.3):
  - _EphemerisVerlet.advance_to rechaza (ValueError) un t no finito o un salto mayor que
    _AVANCE_MAX_S (un año): un t absurdo ya no integra años de Verlet colgando la GUI.
  - _validar_config: APOGEO_OBJETIVO = inf en satelital/tripulada/gto/deorbit da un mensaje
    propio («no puede ser infinito») en vez de uno que habla de la altura de seguridad.
  - _paso_quema_loi_sim: si el primer paso de la quema da una energía no finita, falla con motivo antes
    de usarla en la decisión de corte.
  - Planificador (TLI y desórbita al sitio): _avanzar_plan_por_tramos lanza RuntimeError si supera
    PLAN_TIMEOUT_S (900 s) en vez de dejar la física esperando para siempre; el plan normal tarda 30-120 s.
  - _EPH_CACHE se acota a _EPH_CACHE_MAX_CLAVES fechas/cuerpos (se descarta la más antigua).
  - _limpiar_almacen reintenta el borrado tras gc.collect() si Windows aún tiene el archivo tomado.
  - _ProveedorHorizonsVerlet ya no guarda `tabla3d` (166 KB sin leer tras el constructor).
  - main() cierra el historial (borra el temporal) al salir, sin esperar al finalizador.
  - mostrar_graficas_finales avisa y sale si falta plotly (antes un ImportError tras la misión).
  - tkinter y el backend TkAgg se importan con tolerancia: sin Tk el módulo carga y `--ventanas`
    funciona en consola; la GUI avisa y termina. Imports redundantes (`re as _re`, `dataclasses`) fuera.

Cambios de 12.13.3 (guardia del paso lunar + prueba de equivalencia de corte):
  - _ajustar_dt_lunar solo calcula el instante de encendido LOI si t_burn es finito; si la quema
    estimada es inalcanzable (inf), conserva el paso de coast y deja que la lógica de misión
    registre el fallo de combustible, sin introducir un dt artificial de 1 microsegundo.
  - Se añade una prueba de propiedad determinista que compara _margen_corte(y) >= 0 con
    _corte_por_estado(y) en 10 000 estados sintéticos por modo de misión.

Cambios de 12.13.2 (robustez de cortes y LOI; caso nominal sin cambio):
  - Escape: el criterio documentado es ε >= 0; ya no exige además velocidad radial ascendente.
    El margen del evento y la lógica de misión usan el mismo criterio.
  - _k_quema: tolerancia de energía con suelo absoluto de 1e-6 J/kg, también válida si ε_obj = 0.
  - LOI: la estimación de duración devuelve infinito si el combustible no basta; el paso de quema
    termina con fallo explícito si no hay etapa/combustible/empuje o el paso no puede avanzar.
  - _t_drawer se inicializa explícitamente. El presupuesto de GUI se documenta como best-effort:
    40 ms más el coste del paso que ya está en curso.

Cambios de 12.13.1 (corrección; la física no cambia):
  - copy.deepcopy(Simulacion) fallaba ("cannot pickle 'BufferedWriter'") en cuanto el historial podado
    tenía su archivo abierto (a partir de la primera vuelta completa). Lo usa el planificador de
    desórbita al sitio, así que la planificación se rompía en misiones de varias vueltas.
    Ahora AlmacenHistorial define __deepcopy__: la copia recibe un almacén propio y vacío.

Cambios de 12.13.0 (fluidez de la interfaz; la física no cambia):
  - La trayectoria (y su glow) se dibujaba como ~3600 segmentos sueltos por cuadro: matplotlib crea
    un Path por segmento (~85 % de `_actualizar_animacion`) y los dibuja uno a uno. Ahora los
    segmentos contiguos del mismo color se unen en polilíneas (`agrupar_en_polilineas`): unas pocas
    por fase. Mismo dibujo, sin puntos brillantes en las uniones del glow.
  - La gráfica de vuelo se redibujaba en cada cuadro (~28 ms): ahora como mucho cada 0,4 s
    (`DRAWER_INTERVALO_S`), y siempre al abrirla o al terminar la misión.
  - Los lotes largos de física (`sim_speed` alto, coast lunar) se parten en rebanadas con un
    presupuesto de 40 ms (`PRESUPUESTO_FISICA_S`). Se cede el control a Tk al agotar ese presupuesto,
    más el coste del paso que ya esté en curso (best-effort); clics y repintados se atienden entre
    rebanadas y solo se refresca al completar el lote, así que los pasos por fotograma no cambian.
  - Marcas del mapa: `MaxNLocator(nbins=4)` en ambos ejes (~5 marcas): las etiquetas eran ~20 ms del
    dibujo y bajan un tercio.
  - Al cambiar de pestaña se fuerza el repintado pendiente (`update_idletasks`).

Cambios de 12.12.1 (petición tras probar 12.12.0; la física no cambia):
  - Tecla E: exporta sin diálogo, directo a la carpeta del proyecto (la del archivo .py), como
    `historial_<modo>_<AAAAMMDD_HHMMSS>.npz`; Shift+E lo guarda como .csv. Si el nombre ya existe se
    añade `_2`, `_3`… (nunca sobrescribe). La ruta completa se imprime en consola.

Cambios de 12.12.0 (historial sin pérdida; la física no cambia):
  - Lo que la poda orbital borraba de la RAM ahora se escribe antes a un archivo temporal en disco
    (`AlmacenHistorial`, 64 B por paso: t, altitud, velocidad, x, y, baricentro x/y y fase). La RAM
    sigue acotada (última vuelta) pero no se pierde ningún dato: `Simulacion.historial_completo()`
    lo recompone (ascenso + disco + última vuelta) y `leer_rango` lo carga por tramos.
  - Tecla E: exporta el historial completo a .npz o .csv (diálogo de guardado).
  - Si el disco falla, se avisa una vez y se sigue como antes (se descarta); el resultado se marca
    `incompleto`. El archivo temporal se borra al reiniciar, al recolectar la simulación o al salir.

Cambios de 12.11.1 (captura real de 12.11.0; la física no cambia):
  - Gráfica de vuelo: al podarse el historial orbital quedaba un hueco enorme entre el ascenso y la
    última vuelta (a las 8 h, de 2500 s a 20000 s). Ahora los puntos que se borran se resumen a
    ~60 por periodo orbital (`hist_resumen_*`, tope 20000 con diezmado a la mitad) y el drawer los
    empalma entre el ascenso y la última vuelta: línea continua, memoria acotada. El mapa sigue
    mostrando solo la última vuelta.

Cambios de 12.11.0 (interfaz; la física no cambia):
  - Línea de tiempo de eventos: barra de desplazamiento vertical, visible solo cuando el contenido
    no cabe. Un evento nuevo baja la vista al final solo si el usuario ya estaba abajo (antes
    siempre, y no se podía releer un evento anterior mientras llegaban otros).
  - Recuadro de la Tierra (esquina inferior izquierda del mapa): Tierra, Luna, nave y trayectoria
    centradas en la Tierra. Aparece cuando la cámara deja la Tierra fuera (o cortada) y se oculta
    cuando vuelve a verse entera (con histéresis, para que no parpadee en el borde).

Cambios de 12.10.3 (captura real de 12.10.2; la física no cambia):
  - Gráfica de vuelo (drawer): al podarse el historial orbital, la línea unía el final del
    ascenso con la última vuelta con un tramo recto falso. Ahora se inserta un NaN en el corte
    (`_serie_con_corte`), como ya hacía la trayectoria con la «cuerda».
  - Tecla M: avisa en la barra («Midiendo render…, la ventana no responde ~10 s») antes de bloquear
    la GUI.

Cambios de 12.10.2 (revisión de 12.10.1; los vuelos nominales no cambian):
  - Barrido de ventana TLI: un fallo numérico dentro del tramo con encuentro ya no lo trunca
    (`[hit, hit, FALLO, hit]` conserva el centro); el tramo solo termina con un instante sin
    encuentro o tras 3 fallos seguidos. Se imprime el primer fallo y un resumen al final (antes,
    una línea por instante omitido).
  - `_fin_suborbital_por_t_max` actualiza `datos_mision` (`update`) en vez de reemplazarlo.

Cambios de 12.10.1 (revisión de 12.10.0; los vuelos nominales no cambian):
  - Suborbital con `suborbital_cutoff_por_apogeo=False` y una trayectoria sin apogeo ni impacto
    corría sin fin (comprobado: 150 000 pasos, 275 000 km de altitud). Nuevo
    `Config.T_MAX_SUBORBITAL` (3 h): al alcanzarlo la misión queda NO CUMPLIDA con motivo explícito
    (si ya había veredicto, solo termina la simulación).
  - `_planificar_tli_gen`: un fallo numérico en un instante del barrido de ventana (corte sin
    converger, propagación no finita) ya no aborta la búsqueda: se salta ese instante con aviso y se
    sigue; sin combustible sigue siendo fatal (afecta a todos los instantes) y, si fallan TODOS, se
    reporta el último motivo.
  - `_k_quema`: sin empuje neto (`empuje <= 0`) no se gasta masa; termina con estado 1.
  - `_loi_estimar`: trayectoria casi parabólica (eps < 1e-6·μ/r) se rechaza con ValueError.
  - `_hermite_estado`: guarda para dt <= 0 (devuelve el estado final del paso).
  - Barra superior: `ABORTADA` (fallo del integrador) e `IMPACTO` en lugar de `FINALIZADA` cuando
    no hay veredicto.
  - Solo comentarios: `t_cutoff` es el instante de cierre del paso; `atol` en espacio profundo.

Cambios de 12.10.0 (canvas central; la física no cambia):
  - Glow de la trayectoria: una segunda `LineCollection` (`lc_tray_glow`, ancha y translúcida)
    con los segmentos y colores de la cola reciente (los últimos `GLOW_MAX_SEGMENTOS` = 600; el
    glow en toda la trayectoria sumaba ~70 % al coste del fotograma con 3 000 segmentos); se limpia al reiniciar y entra en la
    medición de la tecla M. Desactivable con `InterfazSimulacion.GLOW_TRAYECTORIA = False`.
  - Halo de la atmósfera: tres anillos concéntricos translúcidos (30/60/100 km) en lugar de un
    solo círculo; siguen a la Tierra en la vista baricéntrica.

Cambios de 12.9.1 (revisión de 12.9.0 y captura real; la física de los vuelos nominales no cambia):
  - `_k_quema`: si el paso completo agotaría el propelente, el último paso se acorta a lo que
    queda (hasta la masa seca) y, si ahí se alcanza la energía objetivo, el corte se refina como
    siempre; antes devolvía «sin combustible» sin intentarlo. El refinamiento comprueba que
    `eps_n - eps` sea > 0 (antes podía dividir por cero) y que el estado sea finito.
  - Verlet 3D (`_tabla_verlet3d_validada`): una distancia nula (ZeroDivisionError) o un estado no
    finito en la tabla lanzan `EfemeridesNoValidas` con diagnóstico, sin sustituir nada por un épsilon.
  - `_leer_config_desde_widgets` es transaccional: trabaja sobre una copia y solo sustituye
    `self.config` si todo convierte y valida. `sim_speed` rechaza valores no enteros (1.9).
  - Pestañas con `grid` + `tkraise` (un solo Frame visible sin repintar a mano), columna izquierda
    de 300 a 330 px y etiqueta «Long. ecl.» más corta (en la captura real los textos quedaban justos).

Cambios de 12.9.0 (interfaz, fase 2; la física no cambia):
  - Columna izquierda en pestañas: MISIÓN (modo, perfil, parámetros de vuelo), LANZ. (sitio,
    objetivos lunares, fecha, ventanas de lanzamiento), OPCIONES (velocidad y casillas) y
    EVENTOS. INICIAR/REINICIAR y el error de configuración quedan fijos al pie. Al iniciar
    una misión se pasa a EVENTOS; al reiniciar, a MISIÓN. Se van la barra de scroll y los
    paneles colapsables (`_crear_columna_izquierda_con_scroll`, `_crear_panel_colapsable`).
  - Eventos como línea de tiempo en un Canvas (punto de color, título y detalle, con scroll
    de rueda); `_agregar_evento(texto, tag)` conserva su firma.
  - Tarjetas: valores en 13 pt con fuente mono disponible (Consolas > DejaVu Sans Mono >
    Menlo > Courier New) y separador de miles («12,345 m»), solo al mostrar.
  - El perfil tripulado se inserta en su sitio al cambiar de modo (antes caía al final).

Cambios de 12.8.5 (revisión de 12.8.4; la física no cambia):
  - `_validar_config` (modo lunar): el tope que se compara con el horizonte de transferencia es
    `T_MAX_LUNAR` (el que consume el modo), no `T_MAX_ALTA_ENERGIA` (de TLI/escape).
  - `sim_speed` rechaza bool; GTO exige `GTO_RP_FINAL` <= `APOGEO_OBJETIVO`.
  - Plan TLI: `sin_comb` pasa a `fallo_plan`; `fallo[0]` se reinicia en cada `evaluar`.

Cambios de 12.8.4 (revisión de robustez; la física de los vuelos no cambia):
  - Efemérides: el fallo de SPICE (`_EPH_FALLO`) se guarda por Simulacion (`_eph_fallo`), no por
    proceso: una misión nueva con la misma fecha arranca limpia (antes un fallo transitorio
    envenenaba la sesión hasta reiniciar). `_estado_pymcel` y `_eph_descripcion` aceptan `fallos`.
  - Corte del TLI: `_k_quema` devuelve un estado (0 ok, 1 no convergió, 2 sin combustible) y el
    plan informa el motivo real; la propagación no finita (d_min, t_ent, ...) es fallo controlado.
  - `_validar_config` rechaza configuraciones imposibles (apogeo suborbital < Kármán, apogeo <=
    altura de seguridad, T_MAX_LUNAR <= horizonte de transferencia, órbita LOI fuera de la SOI)
    y limita `sim_speed` a [1, 10000].
  - GUI: el mensaje de campo inválido nombra el campo y el valor; `OverflowError` (1e400) se
    captura; la configuración de la GUI no queda a medio actualizar.

Cambios de 12.8.3 (auditoría de la interfaz de 12.8.2; la física no cambia):
  - Ventanas: `_ventana_de` ahora SÍ pasa `refinada` al constructor (en 12.8.1/12.8.2 la
    firma lo recibía pero se ignoraba: siempre True, y su prueba comparaba contra una lista vacía).
  - Barra: «MISIÓN CUMPLIDA · SEGUIMIENTO» mientras la simulación sigue tras cumplirse
    (p. ej. lunar sin LOI: SOI alcanzada, nave aún en vuelo); «MISIÓN CUMPLIDA» solo al terminar.
  - Tarjetas: una fila repetida (p. ej. «Dist. Luna:» en lunar con efemérides) se muestra una vez.
  - «Excentric.» sin el umbral «(≤1.000)» cuando no hay umbral real (ecc_max >= 1).

Cambios de 12.8.2 (retoques visuales tras ver la ventana; la física no cambia):
  - Tarjetas con filas en reposo («—») antes de iniciar, en vez de tarjetas vacías.
  - La barra superior muestra el modo elegido mientras no hay misión.
  - La vista de trayectoria usa más área (márgenes de la figura 0.07–0.98).

Cambios de 12.8.1 (auditoría de 12.8.0; la física de los vuelos no cambia):
  - Ventanas: `refinada` ya no sale True con `ventanas(refinar=False)` (mínimo de la malla gruesa).
  - `_k_quema` (corte del TLI): el factor del paso final se acota a (0, 1], se rechazan estados
    no finitos y se exige residuo de energía <= 1e-9·|ε_obj| (nominal: ~1e-14) o devuelve ok=False;
    un corte sin converger es un fallo controlado, como en la LOI.

Cambios de 12.8.0 (interfaz, fase 1; la física no cambia):
  - Barra superior estilo control de misión: reloj MET, fase en una píldora de color y
    estado global (EN CURSO / PLANIFICANDO / MISIÓN CUMPLIDA / NO CUMPLIDA / FINALIZADA,
    con el motivo del fallo).
  - Los «DATOS SÍNCRONOS» pasan de un cuadro de texto a tarjetas de telemetría en una
    columna derecha fija (TRAYECTORIA, PROPULSIÓN, MISIÓN); las filas son las mismas
    (`_filas_estado_general` y compañeras), solo cambia cómo se presentan.
  - Funciones puras `_formatear_met`, `_estado_global`, `_agrupar_filas_en_tarjetas`.

Cambios de 12.7.3 (limpieza de duplicados + ventanas; la física de los vuelos no cambia):
  - Ventanas: `_CAT_VENTANAS` se deriva de `CUERPOS` (mu y radio con una sola fuente;
    Marte ya divergía: 4.2828e13 vs 4.282837e13) y solo guarda las altitudes de parqueo.
  - Ventanas: el Kepler de `EfemeridesPlanetas.estado` usa `_kepler_E` (tol 1e-14).
  - Ventanas: `_EPOCA_J2000` (2 usos), `_periodo_orbital(a)` (3 usos) y semiejes una
    sola vez en `__init__`.
  - `VentanaLanzamiento.refinada`: False si la malla local no tenía celdas válidas y
    se devolvió el mínimo grueso; la tabla lo marca con «*».
  - `calcular_ventanas` valida los límites C3/Δv/v∞ (no NaN, positivos; inf = sin límite).

Cambios de 12.7.2 (correcciones de la auditoría de 12.7.1; la física de los vuelos no cambia):
  - Ventanas: el refinamiento local se recorta al intervalo pedido (salida) y al rango
    de las efemérides (llegada); antes un mínimo en el borde podía salir de ellos.
  - Ventanas: `calcular`, `ventanas` y `calcular_ventanas` validan paso, TOF, fechas y
    criterio ('dv_total' o 'c3'; antes cualquier otro caía en C3 sin avisar).
  - Porkchop: la estrella se etiqueta con el criterio realmente optimizado.

Cambios de 12.7.1 (correcciones de la auditoría de 12.7.0):
  - GUI: la simulación recibe una COPIA de la configuración (antes compartía el
    objeto con los widgets y `_cambiar_modo` podía cambiar `modo_mision` y los
    presets en pleno vuelo). Además el selector de modo no cambia mientras hay
    una misión activa (se pide REINICIAR).
  - LOI: tres niveles explícitos en `datos_mision["loi"]`: `captura_ligada` (ε<0),
    `captura_segura` (periselene sobre la superficie durante la verificación) y
    `orbita_objetivo` (|rp−objetivo| y |ra−objetivo| <= `LUNAR_ORBITA_TOL_ALT`
    y ecc <= `LUNAR_ORBITA_ECC_MAX`). La misión solo se cumple con la órbita
    objetivo; ligada y segura pero fuera de tolerancia termina «NO CUMPLIDA»
    (resultado `captura_fuera_de_objetivo`).
  - Ventanas: `_jd_de_fecha` convierte a UTC las fechas con zona horaria (antes la
    sobrescribía); cada ventana se REFINA con una malla local fina (paso/16) en
    torno al mínimo de la malla gruesa; límites de TOF en `calcular_ventanas`, el
    diálogo y la consola; el porkchop rechaza mallas sin valores válidos; el rango
    de efemérides (1800–2050) se comprueba antes de calcular y se muestra en el
    diálogo. La docstring de Lambert dice lo que hace: una revolución, sentido
    prógrado (+z), arco corto o largo según la geometría.

Cambios de 12.7.0 (ventanas de lanzamiento; la física de los vuelos NO cambia):
  - Nuevo planificador de ventanas interplanetarias, independiente de `Simulacion`:
    efemérides keplerianas de Standish (JPL, 1800-2050), Lambert (variables
    universales, una revolución, sentido prógrado) y malla salida x llegada. Da una
    ventana por ciclo sinódico con fechas, TOF, C3, v∞ de llegada y Δv de salida
    (desde una órbita circular de estacionamiento) y de captura. Planetas como origen
    y destino; lunas y asteroides, no. Es patched conics: elige fechas, no propaga
    ni verifica el vuelo (eso lo hacen los modos de misión).
  - Límites del usuario: C3 máx, Δv de salida máx, v∞ de llegada máx, TOF; no se
    deducen del cohete.
  - GUI: botón «VENTANAS DE LANZAMIENTO…» (diálogo con cálculo en un hilo, tabla y
    porkchop de la ventana elegida). Consola: `python cohete_v12.7.0.py --ventanas
    tierra marte 2026-01-01 2029-01-01 [--c3-max 12] [--png porkchop.png]`.
  - Origen del código: `ventanas_v0.1.0.py` (módulo suelto con su test).

Cambios de 12.6.4 (refactor sin cambio de comportamiento; resultados idénticos a 12.6.3):
  - Hermite cúbico: una sola fuente de coeficientes `_hermite_coefs(s)` (njit)
    para `_estado_pymcel`, `_herm_pos`, `_herm_vel` y `Simulacion._hermite_estado`
    (antes 4 copias de las 8 fórmulas). `_cruce_suelo` usa `_hermite_estado`.
  - Bisección de cruces: una sola `_biseccion_s(antes, n_iter, tol)` en
    `_cruce_suelo`, `_cruce_de_marco`, `_detectar_soi_lunar` e
    `_detectar_impacto_lunar` (mismas iteraciones y tolerancias que antes).
  - `_posicion_cuerpo` = `_estado_cuerpo(...)[0]` (antes repetía todo el despacho
    de efemérides); se borra `_posicion_kepler_simplificada`.
  - `_paso_quema_loi_sim`: el coast del resto del paso, una sola vez.
  - `_fase_orbita_lunar`: mínimos/máximos de la verificación en un bucle.
  - No se hizo: unificar los elementos que `_fase_quema_loi` y
    `_fase_orbita_lunar` calculan (3.7 del análisis; sin ahorro real) ni
    las dos rutas de Horizons (2D/3D, distinto formato).

Cambios de 12.6.3 (corte LOI sin aceptación silenciosa):
  - Si el corte de la LOI no converge (|ε − ε_obj| > tolerancia tras
    LOI_MAX_ITER_CORTE iteraciones) la misión termina con fallo controlado
    «LOI: corte sin converger» en vez de continuar con un corte impreciso.
    `datos_mision["loi"]["iter_corte"]` registra las iteraciones usadas.
  - La actualización h·(ε_obj − ε₀)/(ε_h − ε₀) es la cuerda desde el punto fijo
    (0, ε₀) (regula falsi con extremo fijo); no cambia. Un test comprueba que
    converge en <= 6 iteraciones en las geometrías nominales.

Cambios de 12.6.2 (endurecimiento de la LOI; los vuelos nominales no cambian):
  - Corte de la quema LOI: secante con intervalo [0, dt0] protegido (si el
    denominador no es finito o la corrección sale del intervalo, bisección),
    hasta |ε − ε_obj| <= LOI_TOL_EPS_REL·|ε_obj| o 12 iteraciones; el residuo y
    si convergió quedan en datos_mision["loi"] (`eps_error_corte`,
    `corte_convergido`). Antes: 4 correcciones fijas sin comprobación final.
  - `_conica_selenocentrica` lanza ValueError con estado no válido (r <= 0, no
    finito, μ no válido) y la LOI lo convierte en un fallo controlado y
    diagnosticable en vez de una excepción; `_loi_estimar` calcula el tiempo al
    periselene siempre que ε > 0 (ya no depende de `ecc > 1` en coma flotante,
    caso casi parabólico) y rechaza momento angular nulo.
  - La captura lleva `captura_validada=False` y `modelo_captura="2D"` en
    datos_mision y la deriva (ε, periodo) de la verificación.
  - GUI: casilla «Lunar: captura LOI (2D, provisional)» y campo «Altitud órbita
    LOI (km)»; antes LUNAR_LOI solo se podía activar desde código.
  - La física de los modos existentes y de la LOI nominal no cambia (dentro de
    la tolerancia del corte, ver regresión).

Cambios de 12.6.1 (refactor sin cambio de comportamiento; resultados idénticos a 12.6.0):
  - Un solo RK4 de quema: `_paso_quema_comun(..., seleno)`; `_paso_quema` (TLI,
    prógrado geocéntrico) y `_paso_quema_loi` (retrógrado respecto de la Luna) son
    envoltorios de una línea (antes ~60 líneas duplicadas).
  - Una sola cónica selenocéntrica: `_conica_selenocentrica`, usada por la entrada
    a la SOI y por la LOI (`_elementos_selenocentricos`).
  - No se unifica `_duracion_quema_loi` con `_k_quema` (objetivos distintos).

Cambios de 12.6.0 (captura lunar, LOI; solo con `LUNAR_LOI=True`, resultados 2D PROVISIONALES):
  - Nuevo: tras el encuentro, quema retrógrada respecto de la Luna en el
    periselene (fases `quema_loi` y `orbita_lunar`). Con `LUNAR_LOI=False` (por
    defecto) todo es idéntico a 12.5.12.
  - Con LOI el planificador apunta a un periselene de R_LUNA + `LUNAR_ORBITA_ALT`
    (100 km) en vez de `LUNAR_DIST_OBJETIVO`, y comprueba que el Δv restante de la
    etapa alcanza para la LOI (Δv_LOI ≈ v_periselene − v_circular).
  - Encendido centrado en el periselene: se estima el tiempo al periselene con
    la cónica selenocéntrica y la duración de la quema, y el paso se acorta para
    caer exacto en el encendido. Empuje en sentido contrario a la velocidad
    RELATIVA a la Luna, con la misma regulación por Δv faltante de la TLI; corte
    por energía selenocéntrica de la órbita circular (−μ_L/(2 r_periselene)),
    localizado dentro del paso.
  - Captura = órbita selenocéntrica ligada (ε<0) con periselene sobre la
    superficie durante `LUNAR_PERIODOS_VERIF` (3) periodos. Antes de eso la
    misión queda PENDIENTE; impacto, escape, sin combustible o T_MAX la dan por
    NO CUMPLIDA («sin captura»). Con LOI el encuentro SOI por sí solo ya no cumple.
  - La nave sigue siendo 2D: la captura es en el plano de la eclíptica.
    Provisional hasta la dinámica 3D (13.0.0).
  - `LUNAR_EXTENSION_PERIJOVE_S` se renombra `LUNAR_EXTENSION_PERISELENE_S`.
  - `_coast_hasta_periselene`: desempaqueta el segundo tramo con nombres.

Cambios de 12.5.12 (refactor sin cambio de comportamiento; resultados idénticos a 12.5.11):
  - Una sola construcción de las tablas de efemérides Luna/Sol para numba:
    `Simulacion._tabla_efemerides_lunar(t0, n)`, usada por el planificador y por
    el vuelo translunar (antes dos copias; ambos deben remuestrear igual).
  - La extensión del horizonte (12.5.11) sale de `evaluar` a un helper de módulo
    `_coast_hasta_periselene`, sin reconstruir la tupla de `_k_coast` por índices.

Cambios de 12.5.11 (apuntado lunar: horizonte de predicción; la física no cambia):
  - Hallazgo: el planificador propagaba la transferencia solo
    LUNAR_HORIZONTE_TRANSFERENCIA (6 d) tras el corte. Con encuentros lentos el
    periselene ocurre DESPUÉS (p. ej. 6.1 d) y la «distancia mínima» predicha era
    la del último instante propagado: plan 15 019 km, vuelo real 14 002 km (error
    de 1013 km, 6.7 %; el vuelo coincidía con DOP853 desde el estado de corte). La
    quema y el estado de corte coincidían con el plan a < 1 mm: el error era solo
    de la predicción.
  - `evaluar` continúa ahora la propagación (hasta LUNAR_EXTENSION_PERISELENE_S,
    2 d) cuando llega al horizonte ya dentro de la SOI, hasta pasar el periselene.
    La tabla de efemérides del planificador se alarga en la misma extensión.
  - Resumen lunar: línea «Paso real vs plan» en los sobrevuelos.

Cambios de 12.5.10 (convergencia del vuelo translunar; LA FÍSICA CAMBIA en modo lunar):
  - Las pruebas de convergencia con pasos 20/10/5/2.5/1.25 s mostraron que el
    vuelo translunar convergía solo a orden 1: la marea de la Luna y del Sol se
    evaluaba UNA vez por paso (congelada al inicio) aunque la Luna se mueve
    ~1 km/s. Con paso 5 s cerca de la Luna: d_min +5 km, instante de impacto
    ~+6 s (≈15 km a lo largo de la trayectoria) respecto del límite dt→0.
  - En la fase `translunar` el paso se hace ahora con el mismo RK4 del
    planificador (`_k_coast`: Luna y Sol interpolados en cada subpaso), así el
    vuelo y el plan usan la misma dinámica. Espera, quema TLI y el resto de
    modos no cambian (los escenarios no lunares reproducen 12.5.9 bit a bit).

Cambios de 12.5.9 (auditoría de 12.5.8; la física del sobrevuelo/impacto no cambia):
  - Apuntado: el plan guarda `error_d_min_m` y `tolerancia_cumplida`. Si el
    d_min predicho no llega a la tolerancia max(1 km, 0.2 %) se avisa (⚠); si el
    error supera el 10 % de LUNAR_DIST_OBJETIVO la misión se rechaza antes de
    quemar (antes se aceptaba en silencio).
  - Aviso 3D: la nave y el detector son 2D (plano de la eclíptica) pero la Luna
    real está inclinada ~5.1°. Con el proveedor Horizons+Verlet se registra
    `z_luna` en la entrada a la SOI y en el impacto; si |z| > LUNAR_Z_MAX_FISICO
    (100 km) el impacto se marca «solo proyección 2D (no físico en 3D)».
    Conviene una fecha con la Luna cerca del nodo. Con otros proveedores z es
    desconocido (None) y no se marca nada.
  - Resultados separados: encuentro SOI (sobrevuelo, sin captura) / impacto
    tras el encuentro / fallo. Resumen, HUD y mensaje de la GUI lo distinguen.

Cambios de 12.5.8 (solo GUI; la física no cambia):
  - La columna izquierda (Configuración / Datos síncronos / Eventos) tiene
    barra de desplazamiento vertical y rueda del ratón (solo con el puntero
    sobre la columna; sobre la animación la rueda sigue siendo zoom). Antes,
    con el panel de configuración abierto, los botones INICIAR/REINICIAR y los
    otros paneles quedaban fuera de la ventana.

Cambios de 12.5.7 (apuntado lunar sobre la distancia mínima REAL):
  - El planificador TLI resolvía el periselene con una cónica de 2 cuerpos
    desde la entrada a la SOI; la marea terrestre durante la aproximación
    desplaza el paso real ~500 km. Con 15 000 km da igual, pero con un
    objetivo de impacto (R_LUNA - 200 km) la nave rozaba la Luna sin tocarla.
  - Nueva etapa 3: secante sobre d_min(variable) - LUNAR_DIST_OBJETIVO con la
    propagación completa (t_encendido y/o apogeo TLI; el encendido nunca se
    fija antes de la hora de planificación). Tolerancia max(1 km, 0.2 %).
  - El contrato no cambia: éxito = entrada a la SOI. Solo mejora el apuntado.

Cambios de 12.5.6 (GUI: se reincorporan los arreglos de 12.5.1; la física no cambia):
  - Base: 12.5.5 (impacto lunar, seguimiento post-SOI) bajo regresión.
  - Bucle de la GUI a prueba de excepciones: un error al dibujar ya no mata
    el bucle de Tk en silencio (traceback una vez + aviso, la simulación
    sigue); un error de la física detiene la corrida con el motivo a la vista.
  - Modo lunar: en `espera_tli` y `translunar` se avanzan al menos
    `LUNAR_PASOS_MIN_COAST` (10) pasos por fotograma aunque la velocidad sea 1.
  - `[gui_bucle]` en la regresión; las pruebas lunares se reescriben para la
    física de 12.5.4/12.5.5 (la SOI ya no termina la simulación; impacto).

Cambios de 12.5.5 (impacto lunar y seguimiento post-SOI):
  - Detecta el cruce de la superficie lunar usando la trayectoria Hermite de
    la nave y las efemérides Luna(t); registra instante y velocidad relativa.
  - El objetivo lunar predeterminado queda a 200 km bajo la superficie para
    probar un impacto. En la GUI se puede editar «Perilunio objetivo lunar
    (km)»: valores menores que 1737.4 km apuntan al interior de la Luna; los
    mayores producen un sobrevuelo. El radio se mide desde el centro lunar.
  - El paso fino (5 s) se usa solo en las proximidades del encuentro, no en
    toda la región de 3 radios de la SOI; evita que el seguimiento post-SOI
    parezca congelado por un exceso de pasos diminutos.
  - Cada 6 h simuladas se informa del avance post-SOI en consola.
  - La física continúa después de la SOI y termina al detectar un impacto,
    llegar a T_MAX_LUNAR o fallar el integrador. No se simula captura/LOI.

Cambios de 12.5.4 (continuar después de entrar en la SOI lunar):
  - La entrada a la SOI registra el encuentro y marca la misión como
    cumplida, pero NO termina la simulación. La trayectoria continúa
    con la dinámica geocéntrica actual hasta T_MAX_LUNAR, un impacto
    u otro evento físico terminal. No se simula captura lunar: LOI
    sigue fuera del alcance actual.

Cambios de 12.5.3 (entrega terminal de GUI; física sin cambios):
  - Bloquea un segundo inicio si ya hay una misión activa; cuando termina,
    permite iniciar otra.
  - En el fin terminal cancela los draw_idle pendientes de Matplotlib para
    evitar que una reconstrucción gráfica pesada deje la ventana sin responder.
  - Muestra en la interfaz el estado final y mantiene abierta la ventana Tk.

Cambios de 12.5.2 (cierre terminal de GUI; física sin cambios):
  - La versión 12.5.1 todavía hacía un refresco gráfico completo antes de
    habilitar el botón; ese refresco podía mantener ocupada la GUI al terminar.
  - Al detectar un estado terminal, se actualizan primero los paneles de texto
    y los eventos, se omite el redibujado completo de la trayectoria y se libera
    el bucle de Tk. Se añaden marcas de consola para localizar cualquier bloqueo
    restante dentro del paso físico o de la interfaz.

Cambios de 12.5.0 (modo lunar: Horizons + Verlet por defecto; la física no cambia):
  - Config.efemerides_lunar = "horizons_verlet" por defecto (antes "spice"):
    el modo lunar ya no necesita pymcel/SPICE ni kernels; basta una consulta a
    Horizons con la fecha de lanzamiento. "spice" sigue disponible.
  - La fecha de lanzamiento (UTC, ISO) se escribe en la GUI («Fecha UTC»).
  - La casilla de la GUI pasa a ser «Lunar: usar SPICE (kernels)» (desmarcada).

Cambios de 12.4.0 (modo lunar: fuente de efemérides alternativa):
  - Config.efemerides_lunar = "spice" (por defecto, comportamiento de 12.3.x) |
    "horizons_verlet": una consulta a JPL Horizons (estado 3D de Luna y Sol en
    la fecha de lanzamiento), integración Verlet 3D de ambos cuerpos
    (compilada con numba) sobre todo el horizonte de la misión y tabla con
    interpolación de Hermite, consultable en cualquier orden. No necesita
    kernels de SPICE. Se etiqueta «Horizons + Verlet 3D (modelo simplificado,
    no es SPICE)»; la validación estricta del modo lunar sigue igual (sin
    respaldo Kepler: si Horizons falla, la misión se detiene).
  - Modelo: Luna con Tierra+Luna (mu_T+mu_L) más la marea directa del Sol;
    Sol en dos cuerpos con la Tierra. SIN planetas, J2 terrestre ni
    relatividad. Se integra en 3D (la Luna está inclinada ~5° respecto a la
    eclíptica; en 2D se perdía ~1 % de aceleración) y se proyecta a la
    eclíptica al final, igual que la ruta SPICE.
  - `validar_horizons_verlet_v12.4.0.py` mide el error de la tabla contra
    Horizons (hay que ejecutarlo con red; desde el entorno de desarrollo no
    se pudo: el error real queda SIN medir).
  - Casilla en la GUI «Lunar: Horizons+Verlet (sin SPICE)».
  - La física de los demás modos y de la ruta SPICE no cambia.

Cambios de 12.3.2 (la física no cambia):
  - `_preparar_datos_pymcel()`: copia los archivos de datos de pymcel a
    `pymcel/data/` con `shutil` antes de consultar SPICE. pymcel 0.9.25 usa
    `cp` (inexistente en Windows) y `kernels.txt` no llegaba a su sitio
    (SPICE NOSUCHFILE). La validación estricta del modo lunar no cambia.

Cambios de 12.3.1 (GUI del modo lunar; la física no cambia):
  - Campo «Long. eclíptica» en el panel; si `Simulacion` no puede arrancar
    (EfemeridesNoValidas / configuración inválida) la GUI muestra el motivo y
    deja el botón listo, en vez de dejar caer el bucle de Tk.
  - Filas del HUD del modo lunar: aparcamiento, cuenta atrás al encendido TLI,
    SOI prevista, distancia a la Luna, resultado de la SOI o motivo del fallo;
    «planificando TLI...» durante el apuntado.

Cambios de 12.3.0 (modo "lunar": encuentro con la Luna):
  - Nuevo modo_mision="lunar". Secuencia: B-C-B a una órbita de aparcamiento
    casi circular (rp >= 210 km, ecc < ECC_MAX_APARCAMIENTO) -> verificación
    del aparcamiento -> planificación de la ventana TLI -> espera -> quema
    TLI regulada por energía -> vuelo translunar -> ENTRADA A LA SOI como
    evento. «Cumplida» significa ENCUENTRO (SOI confirmada), no captura: LOI,
    MCC y órbita lunar objetivo quedan para versiones posteriores, igual que
    el 3D. tli, escape, el CRTBP y los puntos de Lagrange no cambian; el modo
    lunar NO usa el marco móvil (propaga en inercial geocéntrico con la Luna y
    el Sol de las efemérides como mareas).
  - Config.longitud_lanzamiento_ecliptica (90° por defecto): fija la fase del
    sitio respecto de la Luna con la fecha. Solo la usa el modo lunar (en los
    demás se rechaza otro valor). La simulación sigue en el marco de
    lanzamiento de siempre (sitio en +y, vuelo hacia +x) y se transforman las
    EFEMÉRIDES (isometría reflexión+giro); con 90° hay un caso explícito (sin
    cos(π/2)). El estado inicial de los siete modos es idéntico bit a bit.
  - Efemérides estrictas: el modo lunar solo corre con SPICE real (sin
    respaldo Kepler) o con un proveedor inyectado
    (Simulacion(config, proveedor_efemerides=f)) que cumpla el contrato; los
    datos de todo el horizonte se validan antes de despegar. Si no, lanza
    EfemeridesNoValidas.
  - Apuntado numérico (Simulacion._planificar_tli_gen): barrido de la hora de
    encendido y secante sobre el momento angular selenocéntrico a la entrada
    (variable más sensible: hora de encendido o energía) hasta un periselene
    previsto de LUNAR_DIST_OBJETIVO; documenta variables ajustadas y fijas en
    datos_mision["plan_tli"]. Se planifica por tramos en la GUI
    (presupuesto_plan_s) con el mismo resultado que de una vez.
  - SOI: cruce de R_SOI - |r - r_Luna(t)| localizado dentro del paso (Hermite
    de la nave + Luna de la efeméride, bisección); el estado queda en el cruce.
    La SOI registra ENCUENTRO -> CUMPLIDA y la propagación continúa; T_MAX_LUNAR
    sin SOI, Δv insuficiente, sin ventana, impacto o efemérides inválidas -> NO CUMPLIDA
    con motivo.
  - Los escenarios de siempre son idénticos a 12.2.0 (rtol 1e-12).

Cambios de 12.2.0 (transiciones de marco en el instante exacto del cruce):
  - Con usar_marco_movil, dinamica="variable" y eventos="integrador", los
    cambios Tierra→crucero, crucero→Luna (entrada a la SOI), crucero→Tierra y
    Luna→crucero se hacen en el INSTANTE del cruce, no al inicio del paso
    siguiente. El cruce se localiza dentro del paso con la interpolación
    cúbica de Hermite de posición y velocidad de los extremos (la misma que
    el impacto en 11.21.0; error < 1e-9 m en estas órbitas suaves), el
    estado se transforma de marco en ese instante y el resto del paso se
    integra en el marco nuevo. Antes el cruce se detectaba hasta un paso
    tarde (decenas de km con dt de 5 s en coast y 30 s en crucero) y la
    transformación se aplicaba a un estado ya fuera de la frontera.
  - Solo en vuelo sin empuje (los cruces ocurren a >300 000 km en coast); un
    paso con empuje que cruzara sigue con la comprobación al inicio del
    paso siguiente. `eventos="paso"` o `dinamica="congelada"` = 12.1.2.
  - NO incluido (queda para después, ver notas): apogeo/perigeo como eventos.
    La lógica de misión actúa por pasos (la 2ª quema arranca al detectar
    vr < 0 en el paso); encenderla a mitad de paso exige partir el paso con
    el motor, un mecanismo distinto.

Cambios de 12.1.2 (planificación de la desórbita sin congelar la GUI):
  - La búsqueda de la hora de quema ya no se ejecuta de un tirón en el hilo
    de la GUI en NINGUNA ruta de respaldo (sin proceso aparte, proceso
    descartado por estado discordante/error/timeout). Es un generador que
    cede cada _PLAN_PASOS_POR_TRAMO pasos de la copia de prueba; con
    `Simulacion.presupuesto_plan_s` (la GUI lo fija) cada fotograma trabaja
    solo ese tiempo y la física espera. Sin presupuesto (scripts, regresión,
    proceso hijo) se ejecuta completo, con el MISMO resultado bit a bit.
  - Mensajes de consola con tiempos en todas las rutas (antes la ruta sin
    planificador externo no decía que estaba calculando).
  - Aclaración de fuentes SPICE: posiciones de efemérides de de430.bsp
    (`pymcel/data/kernels.desc`; es lo que declara la caché, "spk": "de430");
    parámetros gravitacionales de pymcel.constantes, que cita DE441. No se
    mezclan dentro de la caché (solo guarda posiciones/velocidades).

Cambios de 12.1.1:
  - Sin cambio de física (referencia idéntica a la de 12.1.0). Se quita una
    rama inalcanzable de `_cierre_paso_eventos`, se documenta la frontera
    estricta de ecc en `_margen_corte` y el resumen de consola imprime cuántos
    cortes se resolvieron por evento y cuántos por la bisección de respaldo
    (distinto de cero = señal de que el desfase inicial es corto).

Cambios de 12.1.0 (eventos dentro del integrador, marco Tierra):
  - Config.eventos = "integrador" (por defecto) | "paso". Con "integrador" y
    dinamica="variable", el corte del motor y el cruce del suelo son EVENTOS
    de solve_ivp (terminales, con raíz sobre la salida densa): el estado en
    el instante del evento sale del propio integrador, con masa y Δv
    incluidos. Reemplaza la bisección sobre el paso (corte) y el Hermite
    sobre posición (suelo). Con "paso" (o dinamica="congelada") queda el
    comportamiento de 12.0.1 / 11.21.x.
  - Corte: `_margen_corte(y)` es una función escalar continua con
    margen >= 0 <=> `_corte_por_estado(y)` (min = Y, max = O de las
    condiciones); el evento es su cruce de negativo a positivo. Tras el corte
    el resto del paso sigue en coast con el estado exacto del corte. La
    condición se mira DURANTE el empuje, no solo al final del paso (un doble
    cruce dentro del paso ya no se pierde).
  - Impacto: el estado en el cruce (posición, velocidad, MASA y Δv) sale del
    evento; el consumo de combustible y el Δv hasta el impacto son los del
    integrador (ya no una proporción lineal).
  - Siguen igual: agotamiento (analítico, flujo constante en el paso), coast
    libre con RK4 (sin eventos; no puede cruzar el suelo, h > 200 km), marcos
    crucero y Luna (transiciones y SOI como eventos: 12.2.0).

Cambios de 12.0.1:
  - derivadas_v12: el propelente solo se consume mientras hay empuje neto
    (T > 0); con T <= 0 por presión ambiente ya no baja la masa. Coast libre
    en `_integrar_tramo_v12` simplificado (masa constante). Los escenarios de
    regresión no cambian (referencia idéntica a la de 12.0.0).

Cambios de 12.0.0 (rediseño de la dinámica del marco Tierra):
  - Config.dinamica = "variable" (por defecto) | "congelada" (= 11.21.x, bit a
    bit). En "variable" el integrador (RK45) lleva 6 estados: px, py, vx, vy,
    MASA y Δv propulsivo. Empuje (con pérdida por presión ambiente),
    dirección de guiado, arrastre (densidad, Mach, velocidad relativa al aire)
    y gravedad se evalúan en CADA evaluación del ODE con el estado en curso,
    no una vez al inicio del paso. La masa baja con el flujo mientras la
    etapa empuja; no se consume combustible sin empuje; el agotamiento sigue
    siendo exacto (flujo constante en el paso => t_comb = m_comb/flujo).
  - Δv propulsivo = estado integrado ∫T/m dt (ya no la masa media del tramo).
  - El consumo de combustible de la etapa sale de la masa integrada
    (balance de masa verificado en la regresión).
  - Siguen CONSTANTES dentro del paso de 0.25 s (retención de orden cero):
    el throttle (g-limit, circularización, ventanas, desórbita), el modo de
    guiado y la marea de terceros cuerpos. Coast libre sigue con RK4
    (empuje 0, fuera de la atmósfera densa).
  - NO cambia: marcos crucero/Luna, el modelo de Luna circular congelada
    (se evalúa aparte), eventos de corte por bisección y cruce de suelo por
    Hermite (los eventos con solve_ivp quedan para 12.1.0).

Cambios de 11.21.2:
  - `delta_v` del burnout era una diferencia de rapidez (incluye gravedad y
    arrastre), no el Δv del motor. Ahora cada etapa acumula `dv_prop` =
    ∫ T/m dt (masa media del tramo; solo mientras empuja) y el registro de
    burnout trae `dv_prop` (Δv del motor) y `delta_rapidez` (el valor de
    antes); la clave `dv` desaparece. Lo mismo en desórbita
    (`datos_mision["dv_prop"]` y `["delta_rapidez"]`). La física no cambia.

Limitaciones del integrador (v11.21.1). Se separan las aproximaciones
HEREDADAS del diseño por pasos de las limitaciones que introduce la
localización de eventos de 11.21.0; ninguna es un cambio de 11.21.1, que
solo las documenta:
  Heredadas (diseño por pasos de dt fijo; se resolverán en v12.0.0):
    - La masa NO es una variable de estado: es constante dentro de cada
      tramo integrado (con corte a mitad de paso, el tramo en coast usa la
      masa menos lo quemado). Las fuerzas que dependen del estado (empuje,
      arrastre, marea, dirección de guiado) se evalúan UNA vez al inicio del
      paso y se mantienen constantes durante él. Con dt = 0.25 s el error es
      pequeño en vuelo suave y mayor en transitorios (Max-Q, corte, cambio
      de etapa).
    - El paso grande de coast (DT_COAST_ALTA_ENERGIA) solo se usa sin empuje
      y fuera de la atmósfera densa; la dinámica sigue siendo RK45/RK4.
  Introducidas por la localización de eventos (11.21.0):
    - El corte se localiza por bisección sobre el paso con las MISMAS fuerzas
      congeladas; solo se busca si la condición era falsa al inicio del paso
      y verdadera al final (un doble cruce dentro de un paso no se detecta).
      La parada es por tolerancia temporal (_CORTE_TOL_T = 1e-7 s), con un
      tope de _CORTE_MAX_ITER = 25 iteraciones como guarda.
    - El cruce del suelo se interpola con un Hermite cúbico entre el estado
      inicial y el final del paso. PRECONDICIÓN: no coincide con un corte en
      el mismo paso; en los modos operativos el corte ocurre a cientos de km
      de altitud, así que esa combinación no se da.
    - Un fallo de solve_ivp detiene la simulación en el último estado válido
      (campo `fallo_integrador`); no se reintenta con otros parámetros.
  El Verlet de respaldo de las efemérides no está validado cuantitativamente
  contra SPICE (no hay datos de referencia todavía).

Sin límite de tiempo simulado: corre hasta el impacto o el cierre
de la ventana. Al detectarse órbita estable se toma un snapshot de
la primera vuelta completa (para gráficas finales) y el historial
en vivo se poda para conservar solo la última vuelta orbital.
Organización del archivo (de abajo hacia arriba, cada capa solo usa las
anteriores; la física NO depende de la GUI):
  1. Constantes, registro de cuerpos, efemérides y marcos de referencia.
  2. Física pura (sin estado): atmósfera y arrastre, mareas de terceros
     cuerpos, derivadas por marco, elementos orbitales, guiado, RK4.
  3. Datos: Config, EtapaConfig/Etapa, EstadoCohete, RegistroEventos.
  4. Simulación: bucle de pasos y máquina de estados de la misión.
  5. Interfaz gráfica (tkinter + matplotlib).
  6. Gráficas finales (plotly) y resumen en consola.
"""

from __future__ import annotations

import contextlib
import copy
import gc
import hashlib
import io
import json
import math
import multiprocessing
import argparse
import glob
import os
import re
import sys
import shutil
import tempfile
import threading
import weakref
from array import array
from bisect import bisect_right
from collections import OrderedDict

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Optional
from datetime import datetime, timezone, timedelta
from time import monotonic as _monotonic

import numpy as np
import matplotlib
try:                                   # (12.13.4) sin Tk el módulo carga igual (solo consola: --ventanas)
    import tkinter as tk
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    matplotlib.use("TkAgg")
except ImportError:
    tk = None
    FigureCanvasTkAgg = None
    matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.collections import LineCollection
from scipy.integrate import solve_ivp
from numba import njit as _njit

Vec2 = tuple[float, float]
Estado4 = tuple[float, float, float, float]   # (px, py, vx, vy)


# ============================================================
# CONFIGURACIÓN GLOBAL DE MATPLOTLIB
# ============================================================
mpl.rcParams['animation.embed_limit'] = 500.0
plt.rcParams["figure.facecolor"] = "#050a12"
plt.rcParams["axes.facecolor"]   = "#050a12"
plt.rcParams["text.color"]       = "#7fa8c9"
plt.rcParams["axes.edgecolor"]   = "#1e3246"
plt.rcParams["xtick.color"]      = "#5a8aa8"
plt.rcParams["ytick.color"]      = "#5a8aa8"
plt.rcParams["font.family"]      = "monospace"


# ============================================================
# CONSTANTES FÍSICAS
# ============================================================
# mu_earth (= G*M_tierra) se toma de pymcel.constantes, que a su
# vez lo resuelve de los kernels SPICE DE441 (JPL). Preferimos el
# mu ya resuelto en vez de multiplicar G y M por separado: G es de
# las constantes físicas menos precisas (~4-5 cifras confiables),
# mientras que el mu de cada cuerpo del sistema solar se mide
# directo por dinámica orbital real (sondas, satélites), con mucho
# más precisión. Además, tomar mu_earth y R_earth de la misma
# fuente que usaremos más adelante para Sol/Luna/planetas (SPICE)
# evita mezclar valores de referencias distintas cuando agreguemos
# más cuerpos.
#
# pymcel es dependencia obligatoria: sin valores de respaldo.
# El proceso hijo del planificador de desórbita ("spawn") reimporta este
# archivo y pymcel imprime su banner al importarse: se silencia solo en el
# hijo (el padre lo muestra una vez).
with (contextlib.redirect_stdout(io.StringIO())
      if multiprocessing.parent_process() is not None
      else contextlib.nullcontext()):
    import pymcel as pc
from astropy.time import Time
import spiceypy as spy
from spiceypy import SpiceyError
from pymcel.constantes import (mu_earth as MU_T, R_earth as R_T,
                               mu_sun as MU_SOL, mu_moon as MU_LUNA)
G_CONST = 6.674e-11              # solo de referencia
G0 = 9.80665
R_LUNA = 1_737_400.0  # radio físico de la Luna (m), solo para dibujar
OMEGA_T = 7.2921159e-5  # rotación sideral de la Tierra [rad/s]


# ============================================================
# TERCEROS CUERPOS (Sol, Luna, ...) — perturbaciones opcionales
# ============================================================
# Modelo: "N-cuerpos restringido". Sol/Luna/planetas van sobre sus
# propios rieles (no sienten a la nave ni entre sí); la nave es una
# partícula de prueba que siente la suma de todos. Como seguimos
# integrando en un marco centrado en la Tierra (no heliocéntrico),
# no alcanza con sumar la fuerza absoluta de cada cuerpo sobre la
# nave — eso reintroduce como "perturbación" la aceleración que el
# Sol/Luna le dan a la TIERRA MISMA, que es ~10,000 veces mayor que
# la perturbación real y no tiene sentido físico en este marco. Lo
# correcto es la aceleración de marea (diferencial):
#
#   a_pert = mu_i * [ (r_i - r_nave)/|r_i - r_nave|^3 - r_i/|r_i|^3 ]
#
# donde r_i es la posición del cuerpo i RELATIVA A LA TIERRA. El
# segundo término resta exactamente la aceleración que ese cuerpo le
# da a la Tierra, dejando solo la parte que perturba la órbita
# alrededor de la Tierra.
#
# Fuente de r_i(t), según `_TERCEROS_CUERPOS_FUENTE` (pymcel es
# dependencia obligatoria; el valor "sin_pymcel" lo fija la regresión
# para no depender de SPICE ni de la red): (1) "pymcel": SPICE, exacta
# y sin estado (si una consulta falla por datos o red, esa llamada cae
# a Kepler con aviso); (2) "sin_pymcel": una consulta real a JPL
# Horizons en t=0 + propagación Verlet (ver más abajo) si hay red;
# (3) elipse kepleriana simplificada propia si Horizons falla. Ver
# Simulacion._init_efemerides()/_posicion_cuerpo()/_estado_cuerpo().
#
# Limitación explícita: el simulador es 2D. Asumimos que el plano de
# la simulación coincide con el plano de la eclíptica, y que la
# dirección de lanzamiento (t=0) define el "ángulo cero" de
# referencia dentro de ese plano. Esto ignora la inclinación real del
# plano de la órbita respecto a la eclíptica y a la órbita lunar —
# es una simplificación consistente con el resto del modelo (que ya
# no tiene concepto de inclinación), no un intento de precisión real.
# (MU_SOL y MU_LUNA se importan arriba, junto con MU_T y R_T.)
_TERCEROS_CUERPOS_FUENTE = "pymcel"   # la regresión lo cambia a "sin_pymcel"

_UA = 1.495978707e11
_DIA = 86400.0


# ------------------------------------------------------------
# REGISTRO DE CUERPOS (árbol: estrella -> planetas -> lunas)
# ------------------------------------------------------------
# Un solo lugar con los datos de cada cuerpo (mu, radio, padre, elementos
# keplerianos de respaldo, ids de efemérides, semieje de referencia para
# la esfera de influencia). Reemplaza los diccionarios y los `if cuerpo ==
# "sol"` repartidos por el archivo. Desde v11.14.0 contiene Sol, los 8 planetas
# y la Luna (los planetas son solo datos, sin consumidor) y NO cambia ningún resultado; los helpers `hijos`, `radio_soi`,
# `radio_hill` (v11.13.0, solo HUD) y `perturbadores_por_soi` están
# disponibles, pero la simulación sigue usando `Config.cuerpos_perturbadores`:
# la regla por SOI NO se conecta mientras no exista marco heliocéntrico
# (ver NOTAS, «Regla de perturbación vs. física real»).
@dataclass(frozen=True)
class Cuerpo:
    nombre: str
    tipo: str                           # "estrella" | "planeta" | "luna"
    mu: float                           # m^3/s^2
    radio: Optional[float] = None       # radio físico [m]
    padre: Optional[str] = None         # nombre del cuerpo que orbita
    # (a [m], e, período [s]) de la órbita APARENTE respecto de la Tierra
    # (marco actual del simulador: solo Sol y Luna). Se generalizará a
    # elementos respecto del padre cuando exista un marco heliocéntrico.
    # Los planetas (salvo la Tierra, que es el origen) llevan None A
    # PROPÓSITO: sus elementos naturales son heliocéntricos y usarlos
    # aquí daría, sin avisar, posiciones geocéntricas falsas.
    kepler_geo: Optional[tuple] = None
    a_ref: Optional[float] = None       # semieje de referencia para la SOI [m]
    horizons_id: Optional[str] = None   # COMMAND de JPL Horizons
    pymcel_nombre: Optional[str] = None


CUERPOS = {
    "sol":    Cuerpo("sol", "estrella", MU_SOL, radio=6.957e8,
                     kepler_geo=(_UA, 0.0167, 365.256363 * _DIA),
                     horizons_id="10", pymcel_nombre="sun"),
    "mercurio": Cuerpo("mercurio", "planeta", 2.2032e13, radio=2.4397e6, padre="sol",
                     a_ref=0.38710 * _UA, horizons_id="199", pymcel_nombre="mercury"),
    "venus": Cuerpo("venus", "planeta", 3.24859e14, radio=6.0518e6, padre="sol",
                     a_ref=0.72333 * _UA, horizons_id="299", pymcel_nombre="venus"),
    "tierra": Cuerpo("tierra", "planeta", MU_T, radio=R_T, padre="sol",
                     a_ref=_UA, horizons_id="399", pymcel_nombre="earth"),
    "marte": Cuerpo("marte", "planeta", 4.2828e13, radio=3.3895e6, padre="sol",
                     a_ref=1.52368 * _UA, horizons_id="499", pymcel_nombre="mars"),
    "jupiter": Cuerpo("jupiter", "planeta", 1.26687e17, radio=6.9911e7, padre="sol",
                     a_ref=5.2026 * _UA, horizons_id="599", pymcel_nombre="jupiter"),
    "saturno": Cuerpo("saturno", "planeta", 3.7931e16, radio=5.8232e7, padre="sol",
                     a_ref=9.5549 * _UA, horizons_id="699", pymcel_nombre="saturn"),
    "urano": Cuerpo("urano", "planeta", 5.7940e15, radio=2.5362e7, padre="sol",
                     a_ref=19.2184 * _UA, horizons_id="799", pymcel_nombre="uranus"),
    "neptuno": Cuerpo("neptuno", "planeta", 6.8365e15, radio=2.4622e7, padre="sol",
                     a_ref=30.1104 * _UA, horizons_id="899", pymcel_nombre="neptune"),
    "luna":   Cuerpo("luna", "luna", MU_LUNA, radio=R_LUNA, padre="tierra",
                     kepler_geo=(3.844e8, 0.0549, 27.321661 * _DIA),
                     a_ref=384400e3, horizons_id="301", pymcel_nombre="moon"),
}


def hijos(nombre: str, cuerpos: Optional[dict] = None) -> tuple:
    """Nombres de los cuerpos que orbitan directamente a `nombre`, en el
    orden del registro."""
    cuerpos = CUERPOS if cuerpos is None else cuerpos
    return tuple(n for n, c in cuerpos.items() if c.padre == nombre)


def radio_soi(nombre: str, cuerpos: Optional[dict] = None) -> float:
    """Radio de la esfera de influencia de `nombre` respecto de su padre:
    a_ref * (mu / mu_padre) ** 0.4."""
    cuerpos = CUERPOS if cuerpos is None else cuerpos
    c = cuerpos[nombre]
    if c.padre is None or c.a_ref is None:
        raise ValueError(f"'{nombre}' no tiene padre o semieje de referencia")
    return c.a_ref * (c.mu / cuerpos[c.padre].mu) ** 0.4


def radio_hill(nombre: str, cuerpos: Optional[dict] = None) -> float:
    """Radio de Hill de `nombre` respecto de su padre:
    a_ref * (mu / (3 * mu_padre)) ** (1/3). Límite de la región donde un
    satélite sigue ligado al cuerpo a largo plazo (SOI es otra cosa: el
    criterio de cambio de marco en patched conics). Misma firma que
    `radio_soi`. Hoy solo lo consume el HUD (zona lunar); no interviene
    en la dinámica ni en el cambio de marco."""
    cuerpos = CUERPOS if cuerpos is None else cuerpos
    c = cuerpos[nombre]
    if c.padre is None or c.a_ref is None:
        raise ValueError(f"'{nombre}' no tiene padre o semieje de referencia")
    return c.a_ref * (c.mu / (3.0 * cuerpos[c.padre].mu)) ** (1.0 / 3.0)


def perturbadores_por_soi(primario: str,
                          cuerpos: Optional[dict] = None) -> tuple:
    """Cuerpos que perturban al cohete cuando su SOI más interna es la de
    `primario` (las perturbaciones son solo de planetas, lunas y Sol sobre
    el cohete: ni entre cuerpos ni del cohete sobre ellos):
      estrella -> sus planetas
      planeta  -> sus lunas
      luna     -> su planeta, las demás lunas de ese planeta y sus
                  propias lunas (si las tuviera)."""
    cuerpos = CUERPOS if cuerpos is None else cuerpos
    c = cuerpos[primario]
    if c.tipo in ("estrella", "planeta"):
        return hijos(primario, cuerpos)
    hermanas = tuple(n for n in hijos(c.padre, cuerpos) if n != primario)
    return (c.padre,) + hermanas + hijos(primario, cuerpos)


# Órbitas aparentes respecto de la Tierra (mismos valores que antes).
_ELEMENTOS_TERCEROS = {n: CUERPOS[n].kepler_geo for n in ("sol", "luna")}


def _epoca_a_segundos(fecha_iso: str) -> float:
    """Segundos desde una época de referencia fija (2000-01-01)."""
    epoca_ref = datetime(2000, 1, 1, tzinfo=timezone.utc)
    f = datetime.fromisoformat(fecha_iso)
    if f.tzinfo is None:
        f = f.replace(tzinfo=timezone.utc)
    return (f - epoca_ref).total_seconds()


def _kepler_E(M: float, e: float, tol: float = 1e-10,
              max_iter: int = 50) -> float:
    """Resuelve M = E - e*sin(E) por Newton-Raphson.

    (11.21.1) Ya no devuelve en silencio un valor sin converger: si tras
    `max_iter` iteraciones el último paso de Newton sigue por encima de
    `tol`, lanza RuntimeError con M, e y el residuo. También rechaza una
    excentricidad fuera de [0, 1) o una anomalía media no finita (antes
    devolvía NaN o un E arbitrario sin avisar). La aritmética de la
    iteración es la de siempre: cuando converge, el resultado es idéntico."""
    if not (0.0 <= e < 1.0):
        raise ValueError(f"_kepler_E: excentricidad e={e!r} fuera de [0, 1)")
    if not math.isfinite(M):
        raise ValueError(f"_kepler_E: anomalía media M={M!r} no finita")
    E = M if e < 0.8 else np.pi
    dE = float("inf")
    for _ in range(max_iter):
        dE = (E - e * np.sin(E) - M) / (1.0 - e * np.cos(E))
        E -= dE
        if abs(dE) < tol:
            return E
    raise RuntimeError(
        f"_kepler_E no convergió: M={M!r}, e={e!r}, último |dE|={abs(dE):.3e} "
        f"tras {max_iter} iteraciones (tol={tol:g})")


def _exigir_sol_o_luna(cuerpo: str) -> None:
    """Horizons+Verlet solo integra Sol y Luna: para cualquier otro
    cuerpo (p. ej. un planeta del registro) fallar en vez de devolver
    en silencio la posición de la Luna."""
    if cuerpo != "sol" and cuerpo != "luna":
        raise KeyError(
            f"'{cuerpo}': Horizons+Verlet solo cubre Sol y Luna hasta que "
            f"exista un marco heliocéntrico")


def _estado_kepler_simplificado(cuerpo: str, t_seg_epoca: float):
    """
    Respaldo propio (sin pymcel ni red): posición Y velocidad 2D del
    cuerpo relativa a la Tierra, en una elipse kepleriana con
    elementos medios, evaluada analíticamente (sin diferencia finita).
    Solo para validar el mecanismo de perturbación; no tiene
    precisión de efeméride real.
    """
    if cuerpo not in _ELEMENTOS_TERCEROS:
        raise KeyError(
            f"'{cuerpo}' no tiene órbita aparente respecto de la Tierra "
            f"(kepler_geo): el respaldo Kepler solo cubre Sol y Luna hasta "
            f"que exista un marco heliocéntrico")
    a, e, T = _ELEMENTOS_TERCEROS[cuerpo]
    n = 2.0 * np.pi / T
    M = (n * t_seg_epoca) % (2.0 * np.pi)
    E = _kepler_E(M, e)
    x = a * (np.cos(E) - e)
    y = a * np.sqrt(1.0 - e * e) * np.sin(E)
    Edot = n / (1.0 - e * np.cos(E))
    vx = -a * np.sin(E) * Edot
    vy = a * np.sqrt(1.0 - e * e) * np.cos(E) * Edot
    return (float(x), float(y)), (float(vx), float(vy))


_AVISOS_PYMCEL: set = set()    # para avisar una sola vez por tipo de fallo


# --- Caché de efemérides pymcel (v11.17.3) ---------------------------
# Cada `pc.consulta_spice` recarga TODOS los kernels de SPICE (descarga/
# comprobación + furnsh: ~30 líneas en consola y del orden de segundos por
# llamada), y la simulación pide la Luna/el Sol varias veces por fotograma
# (dibujo, HUD) y por paso (marea, marco móvil). Sin caché, la GUI queda
# congelada. Solución: se consulta SPICE en una rejilla de nodos cada
# `_EPH_PASO` s, en LOTES de `_EPH_NODOS_LOTE` épocas por llamada (una sola
# recarga de kernels por lote), y entre nodos se interpola con un
# polinomio cúbico de Hermite usando posición Y velocidad de SPICE. Error
# de interpolación de la Luna con paso de 600 s: ~1e-5 m (la órbita lunar
# es suave: ω⁴·a·h⁴/384). La caché es global al proceso y se comparte
# entre copias de la simulación (deepcopy); cada consulta es independiente
# de t_sim (no exige monotonía).
# Unidades y significado (11.21.1):
#   _EPH_PASO [s]        separación TEMPORAL entre dos nodos consecutivos de
#                        la rejilla (600 s = 10 min). Es el paso de la
#                        interpolación de Hermite: el error crece como h⁴.
#   _EPH_NODOS_LOTE [nº] cuántos nodos se piden en UNA llamada a SPICE (no es
#                        un tiempo). Un lote cubre _EPH_NODOS_LOTE·_EPH_PASO
#                        = 144·600 s = 86 400 s = 24 h. Debe ser != 6: la
#                        forma (N, 6) / (6, N) de `data` es ambigua con N=6.
#   Los lotes están ALINEADOS a múltiplos de _EPH_NODOS_LOTE: el nodo i
#   pertenece al lote i0 = (i // _EPH_NODOS_LOTE)·_EPH_NODOS_LOTE.
#   Si se cambia _EPH_PASO o _EPH_NODOS_LOTE, la clave de la caché en disco
#   (ver _eph_clave_disco) cambia sola y no se reutilizan datos viejos.
_EPH_PASO: float = 600.0           # s entre nodos
_EPH_NODOS_LOTE: int = 144         # nodos por consulta (24 h); distinto de 6
_EPH_CACHE: OrderedDict = OrderedDict()  # LRU: (cuerpo, fecha) -> {índice: (x, y, vx, vy)}
_EPH_CACHE_MAX_CLAVES = 16         # tope de combinaciones (cuerpo, fecha)
_EPH_CACHE_MAX_LOTES_POR_CLAVE = 2 # (12.13.5) k y k+1 pueden cruzar un límite de lote
_EPH_CACHE_MAX_NODOS_POR_CLAVE = _EPH_CACHE_MAX_LOTES_POR_CLAVE * _EPH_NODOS_LOTE
_EPH_FALLO: set = set()            # (cuerpo, fecha) que ya cayeron a Kepler


def _preparar_datos_pymcel() -> None:
    """(12.3.2) Copia los archivos de datos de pymcel (`kernels.txt`,
    `de430.tpc`, ...) a `pymcel/data/` (relativo al directorio de trabajo,
    que es donde `furnsh` los busca) sin usar `cp`. pymcel 0.9.25 hace
    `os.system("cp -rf ...")`, que no existe en Windows: los kernels grandes
    se descargan bien, pero `kernels.txt` nunca llega a `pymcel/data/` y
    SPICE falla con NOSUCHFILE. Solo copia lo que falta; no pisa nada."""
    origen = os.path.join(getattr(pc, "ROOTDIR", ""), "data")
    if not os.path.isdir(origen):
        return
    destino = os.path.join("pymcel", "data")
    os.makedirs(destino, exist_ok=True)
    for nombre in os.listdir(origen):
        src = os.path.join(origen, nombre)
        dst = os.path.join(destino, nombre)
        if os.path.isfile(src) and not os.path.exists(dst):
            shutil.copy2(src, dst)


_SPICE_LISTO = False     # (12.13.8) ¿ya se cargaron (furnsh) los kernels en ESTE proceso?


def _asegurar_kernels_spice() -> None:
    """(12.13.8) pymcel 0.9.25 solo descarga y carga (`furnsh`) los kernels dentro de
    `consulta_spice` si `pymcel/data/kernels.txt` NO existe, y `_preparar_datos_pymcel` lo crea
    antes: en un proceso nuevo nunca se cargaban y SPICE fallaba con MISSINGTIMEINFO (caía a
    Kepler). Aquí se descargan los kernels grandes si faltan y se cargan una sola vez.
    Los fallos de red/archivos suben como OSError/SpiceyError (los captura `_estado_pymcel`)."""
    global _SPICE_LISTO
    if _SPICE_LISTO:
        return
    _preparar_datos_pymcel()
    if not glob.glob(os.path.join("pymcel", "data", "*.bsp")):
        pc.descarga_kernels(verbose=False)
        _preparar_datos_pymcel()
    spy.furnsh(os.path.join("pymcel", "data", "kernels.txt"))
    _SPICE_LISTO = True


def _eph_cargar_lote(cuerpo: str, fecha_lanzamiento: str, i0: int) -> dict:
    """Consulta SPICE (una sola llamada) los nodos i0 .. i0+_EPH_NODOS_LOTE-1
    y devuelve {índice: (x, y, vx, vy)} en SI, marco eclíptico J2000,
    relativo a la Tierra. `data` con N épocas se acepta como (N, 6) o
    (6, N); cualquier otra forma es un error de firma y sube como
    ValueError (cae a Kepler con aviso, ver `_estado_pymcel`)."""
    n = _EPH_NODOS_LOTE
    t_epoca0 = _epoca_a_segundos(fecha_lanzamiento)
    base = datetime(2000, 1, 1, tzinfo=timezone.utc)
    fechas = [base + timedelta(seconds=t_epoca0 + (i0 + k) * _EPH_PASO)
              for k in range(n)]
    jd_tdb = [float(j) for j in np.atleast_1d(Time(fechas).tdb.jd)]
    _asegurar_kernels_spice()
    data, _ts, _df = pc.consulta_spice(
        id=CUERPOS[cuerpo].horizons_id,
        location=CUERPOS["tierra"].horizons_id,
        epochs=jd_tdb)
    arr = np.asarray(data, dtype=float)
    if arr.shape == (6, n):
        arr = arr.T
    if arr.shape != (n, 6):
        raise ValueError(f"consulta_spice devolvió forma {arr.shape} para "
                         f"{n} épocas; se esperaba ({n}, 6) o (6, {n})")
    return {i0 + k: (arr[k, 0], arr[k, 1], arr[k, 3], arr[k, 4])
            for k in range(n)}


# --- Caché de efemérides en disco (v11.20.0) -------------------------
# Los nodos que devuelve SPICE se guardan en `efemerides_cache.json` (junto
# a este archivo): en las corridas siguientes con la misma fecha NO se llama a
# `pc.consulta_spice` (ni se cargan kernels, ni hace falta red). Clave de
# cada lote: cuerpo | fecha | paso | nodos por lote | índice del primer nodo.
# Solo se guardan datos reales de SPICE (nunca el respaldo Kepler). `None`
# desactiva el disco (la regresión lo hace para no depender de él).
_EPH_DISCO = Path(__file__).with_name("efemerides_cache.json")
# "spk": "de430" = kernel de POSICIONES que carga pymcel (pymcel/data/
# kernels.desc: de430.bsp). Los mu de pymcel.constantes citan DE441; no entran
# en la caché (solo posición/velocidad de cada cuerpo).
_EPH_META = {"formato": 1, "marco": "ECLIPJ2000", "origen": "399",
             "aberracion": "none", "spk": "de430"}
_EPH_FUENTE: dict = {}             # (cuerpo, fecha) -> {"spice", "disco"}


def _eph_clave_disco(cuerpo: str, fecha: str, i0: int) -> str:
    return f"{cuerpo}|{fecha}|{_EPH_PASO:g}|{_EPH_NODOS_LOTE}|{i0}"


def _eph_leer_archivo() -> dict:
    """Entradas guardadas en disco ({} si no hay archivo, está corrupto o es de
    otro formato; en ese caso se vuelve a consultar SPICE y se reescribe)."""
    if _EPH_DISCO is None or not Path(_EPH_DISCO).is_file():
        return {}
    try:
        datos = json.loads(Path(_EPH_DISCO).read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        _eph_aviso_disco(f"caché de efemérides ilegible ({type(e).__name__}); se ignora")
        return {}
    if not isinstance(datos, dict) or datos.get("meta") != _EPH_META:
        return {}
    ent = datos.get("lotes")
    return ent if isinstance(ent, dict) else {}


def _eph_aviso_disco(texto: str) -> None:
    if texto not in _AVISOS_PYMCEL:
        _AVISOS_PYMCEL.add(texto)
        print(f"⚠ {texto}")


def _eph_leer_disco(cuerpo: str, fecha: str, i0: int):
    """Lote guardado como {índice: (x, y, vx, vy)}, o None si no está o no es
    válido (forma o valores no finitos)."""
    crudo = _eph_leer_archivo().get(_eph_clave_disco(cuerpo, fecha, i0))
    try:
        arr = np.asarray(crudo, dtype=float)
    except (TypeError, ValueError):
        return None
    if arr.shape != (_EPH_NODOS_LOTE, 4) or not np.all(np.isfinite(arr)):
        return None
    return {i0 + k: (arr[k, 0], arr[k, 1], arr[k, 2], arr[k, 3])
            for k in range(_EPH_NODOS_LOTE)}


def _eph_guardar_disco(cuerpo: str, fecha: str, i0: int, lote: dict) -> None:
    """Añade el lote al archivo (escritura atómica: temporal + os.replace)."""
    if _EPH_DISCO is None:
        return
    ent = _eph_leer_archivo()
    ent[_eph_clave_disco(cuerpo, fecha, i0)] = [
        [float(v) for v in lote[i0 + k]] for k in range(_EPH_NODOS_LOTE)]
    destino = Path(_EPH_DISCO)
    tmp = destino.with_name(destino.name + ".tmp")
    try:
        tmp.write_text(json.dumps({"meta": _EPH_META, "lotes": ent}),
                       encoding="utf-8")
        os.replace(tmp, destino)
    except OSError as e:
        _eph_aviso_disco(f"no se pudo guardar la caché de efemérides "
                         f"({type(e).__name__}: {e}); sigue en memoria")


def _eph_obtener_lote(cuerpo: str, fecha: str, i0: int) -> dict:
    """Disco primero; si no está, SPICE (y se guarda)."""
    lote = _eph_leer_disco(cuerpo, fecha, i0)
    fuente = "disco"
    if lote is None:
        lote = _eph_cargar_lote(cuerpo, fecha, i0)
        _eph_guardar_disco(cuerpo, fecha, i0, lote)
        fuente = "spice"
    usadas = _EPH_FUENTE.setdefault((cuerpo, fecha), set())
    if not usadas:
        print("✓ Efemérides de " + cuerpo + ": "
              + ("caché en disco" if fuente == "disco"
                 else "SPICE (consulta nueva" + (", guardada en disco)"
                                                  if _EPH_DISCO is not None else ")")))
    usadas.add(fuente)
    return lote


def _eph_descripcion(fecha: str, fallos: Optional[set] = None) -> str:
    """Fuente REAL de las efemérides pymcel para esta fecha (no la pedida)."""
    fallos = _EPH_FALLO if fallos is None else fallos
    claves = [(c, fecha) for c in ("luna", "sol")]
    if any(k in fallos for k in claves):
        return "Kepler (respaldo: SPICE falló)"
    usadas = set().union(*(_EPH_FUENTE.get(k, set()) for k in claves))
    if not usadas:
        return "pymcel (aún sin consultar)"
    if usadas == {"disco"}:
        return "pymcel: caché en disco"
    if usadas == {"spice"}:
        return "pymcel: SPICE"
    return "pymcel: caché en disco + SPICE"


def _biseccion_s(antes, n_iter: int, tol: Optional[float] = None) -> float:
    """(12.6.4) Bisección de la fracción s en [0, 1] de un paso donde ocurre un
    cruce: `antes(s)` es True mientras el cruce aún no ha ocurrido. Devuelve
    el extremo superior del intervalo final (el primer s ya cruzado). Para tras
    `n_iter` iteraciones o, si `tol` no es None, al quedar el intervalo <= tol."""
    lo, hi = 0.0, 1.0
    for _ in range(n_iter):
        mid = 0.5 * (lo + hi)
        if antes(mid):
            lo = mid
        else:
            hi = mid
        if tol is not None and hi - lo <= tol:
            break
    return hi


@_njit(cache=True)
def _hermite_coefs(s):
    """(12.6.4) Coeficientes del Hermite cúbico en la fracción s de un
    intervalo: (h00, h10, h01, h11) de la posición y (d00, d10, d01, d11) de
    la derivada respecto de s. Los d00 y d01 NO están divididos por la
    longitud del intervalo: quien interpole debe dividirlos por h (o dividir la
    suma entera, como `Simulacion._hermite_estado`)."""
    s2 = s * s
    s3 = s2 * s
    return (2.0 * s3 - 3.0 * s2 + 1.0, s3 - 2.0 * s2 + s,
            -2.0 * s3 + 3.0 * s2, s3 - s2,
            6.0 * s2 - 6.0 * s, 3.0 * s2 - 4.0 * s + 1.0,
            -6.0 * s2 + 6.0 * s, 3.0 * s2 - 2.0 * s)


def _estado_pymcel(cuerpo: str, fecha_lanzamiento: str, t_sim: float,
                   fallos: Optional[set] = None):
    """
    (posición, velocidad) 2D relativas a la Tierra vía pymcel/SPICE, en el
    marco eclíptico J2000 y SIN corrección de luz (posición geométrica, la
    que necesita la gravedad newtoniana). `pc.consulta_spice(id, location,
    epochs)` usa los ids NAIF (los mismos `horizons_id` del registro; el
    '@' de `location` es opcional: pymcel lo quita) y espera las épocas
    como lista de JD en TDB (un str lo leería como TT, ~69 s de error
    frente a UTC). Devuelve en SI (pymcel multiplica km y km/s por 1000):
    por época, un vector de 6 con (x, y, z, vx, vy, vz); aquí se toman
    x, y, vx, vy.
    No se consulta SPICE en cada llamada sino en una rejilla de nodos con
    interpolación cúbica de Hermite (ver «Caché de efemérides» arriba).
    Si pymcel/SPICE no puede responder por datos o red (KeyError,
    ValueError, OSError —p. ej. el kernel no se descarga— o SpiceyError,
    como una fecha fuera de la cobertura del kernel) avisa una vez, lo
    recuerda (no se reintenta en cada llamada) y cae al respaldo
    kepleriano. Cualquier otro error (firma, bug) sube: no se esconde.
    NOTA: escrito contra la firma de pymcel 0.9.25 y su código; el caso de
    varias épocas por llamada (forma de `data`) no está validado con un
    kernel SPICE real: si falla, el aviso lo dice y se usa Kepler.
    """
    clave = (cuerpo, fecha_lanzamiento)
    t_epoca = _epoca_a_segundos(fecha_lanzamiento) + t_sim
    fallos = _EPH_FALLO if fallos is None else fallos      # (12.8.4) por Simulacion
    if clave in fallos:
        return _estado_kepler_simplificado(cuerpo, t_epoca)
    if clave in _EPH_CACHE:
        _EPH_CACHE.move_to_end(clave)             # LRU: un hit renueva su antigüedad
    else:
        if len(_EPH_CACHE) >= _EPH_CACHE_MAX_CLAVES:
            _EPH_CACHE.popitem(last=False)        # descarta la menos recientemente usada
        _EPH_CACHE[clave] = {}
    tabla = _EPH_CACHE[clave]
    k = math.floor(t_sim / _EPH_PASO)
    try:
        cargado = False
        for i in (k, k + 1):
            if i not in tabla:
                i0 = (i // _EPH_NODOS_LOTE) * _EPH_NODOS_LOTE
                tabla.update(_eph_obtener_lote(cuerpo, fecha_lanzamiento, i0))
                cargado = True
        # (12.13.5) Solo hacen falta los lotes que contienen k y k+1.
        # Normalmente es uno; en la frontera entre lotes son dos. Los nodos
        # eliminados pueden volver a leerse de disco en _eph_obtener_lote.
        # (12.13.7) La tabla solo crece al cargar un lote, así que solo ahí hace
        # falta podar: antes se recorría en CADA llamada (~10 µs de más por consulta).
        if cargado:
            lotes_necesarios = {
                (i // _EPH_NODOS_LOTE) * _EPH_NODOS_LOTE for i in (k, k + 1)
            }
            for indice in tuple(tabla):
                i0 = (indice // _EPH_NODOS_LOTE) * _EPH_NODOS_LOTE
                if i0 not in lotes_necesarios:
                    del tabla[indice]
            # Guarda defensiva: la poda anterior debe mantener <= 2 lotes.
            if len(tabla) > _EPH_CACHE_MAX_NODOS_POR_CLAVE:
                raise RuntimeError(
                    f"caché de efemérides excedió {_EPH_CACHE_MAX_NODOS_POR_CLAVE} nodos "
                    f"para {clave!r}: {len(tabla)}")
    except (KeyError, ValueError, OSError, SpiceyError) as e:
        fallos.add(clave)
        nombre = type(e).__name__
        if nombre not in _AVISOS_PYMCEL:
            _AVISOS_PYMCEL.add(nombre)
            print(f"⚠ pymcel: consulta_spice falló ({nombre}: {e}); "
                  f"usando Kepler simplificado")
        return _estado_kepler_simplificado(cuerpo, t_epoca)
    dt = _EPH_PASO
    s = t_sim / dt - k
    x0, y0, vx0, vy0 = tabla[k]
    x1, y1, vx1, vy1 = tabla[k + 1]
    h00, h10, h01, h11, d00, d10, d01, d11 = _hermite_coefs(s)
    d00, d01 = d00 / dt, d01 / dt
    px = h00 * x0 + h10 * dt * vx0 + h01 * x1 + h11 * dt * vx1
    py = h00 * y0 + h10 * dt * vy0 + h01 * y1 + h11 * dt * vy1
    vx = d00 * x0 + d10 * vx0 + d01 * x1 + d11 * vx1
    vy = d00 * y0 + d10 * vy0 + d01 * y1 + d11 * vy1
    return (float(px), float(py)), (float(vx), float(vy))


def _posicion_pymcel(cuerpo: str, fecha_lanzamiento: str, t_sim: float):
    """Solo la posición 2D de `_estado_pymcel`."""
    return _estado_pymcel(cuerpo, fecha_lanzamiento, t_sim)[0]


# ------------------------------------------------------------
# Horizons (una consulta real por cuerpo, en t=0) + Verlet
# ------------------------------------------------------------
# Se usa como segunda opción cuando pymcel no está instalado: UNA
# consulta real a JPL Horizons por cuerpo da un estado vectorial
# preciso (posición+velocidad reales, no una elipse media); de ahí en
# adelante, Störmer-Verlet propaga Sol y Luna con su propia física
# (Sol: 2 cuerpos con la Tierra; Luna: gravedad terrestre + marea
# solar), sin necesitar más consultas de red.
#
# CORRECCIÓN: la URL se arma con urllib.parse.urlencode. Construirla
# a mano con f-strings (como en una versión anterior) deja espacios
# sin codificar en la fecha ("'2026-01-01 00:00'"), y
# urllib.request.urlopen rechaza esa URL de entrada con InvalidURL —
# ANTES de intentar conectarse. No es un problema de conectividad:
# con esa URL, la consulta fallaba siempre, con o sin red.
def _fetch_horizons_state(cuerpo: str, fecha_iso: str, timeout: float = 20.0):
    """Estado 2D (x, y, vx, vy) [SI] en el plano eclíptico J2000,
    reutilizando el mismo transporte y parser CSV que la ruta 3D."""
    fecha = _fecha_horizons_seg(fecha_iso)
    texto = _horizons_consulta_texto(cuerpo, fecha, fecha, "1 d", timeout)
    x, y, _z, vx, vy, _vz = _parsear_vectores_horizons(texto, cuerpo)
    return x, y, vx, vy


class _EphemerisVerlet:
    """
    Propaga Sol y Luna a partir de un estado inicial real (de
    Horizons), con Verlet de velocidades:

        x_{n+1} = x_n + v_n·dt + ½·a_n·dt²
        a_{n+1} = a(x_{n+1})
        v_{n+1} = v_n + ½·(a_n + a_{n+1})·dt

    Fuerzas: Sol en 2 cuerpos con la Tierra; Luna con gravedad
    terrestre + marea solar. advance_to(t) es monótono: solo avanza
    hacia t > t_actual (no puede retroceder).
    """
    def __init__(self, xS, yS, vxS, vyS, xL, yL, vxL, vyL):
        self.xS, self.yS = xS, yS
        self.vxS, self.vyS = vxS, vyS
        self.xL, self.yL = xL, yL
        self.vxL, self.vyL = vxL, vyL
        self.t_actual = 0.0

    def _accel_sun(self):
        r3 = (self.xS * self.xS + self.yS * self.yS) ** 1.5
        # Aceleración RELATIVA Tierra-Sol: (mu_sol + mu_T) (dos cuerpos).
        mu = MU_SOL + MU_T
        return -mu * self.xS / r3, -mu * self.yS / r3

    def _accel_moon(self):
        rL3 = (self.xL * self.xL + self.yL * self.yL) ** 1.5
        # Aceleración RELATIVA Tierra-Luna: (mu_T + mu_luna), no solo mu_T
        # (con mu_T solo la Luna sentía ~1.2 % menos atracción central:
        # un error sistemático de la órbita lunar en esta ruta de respaldo).
        mu = MU_T + MU_LUNA
        aTx, aTy = -mu * self.xL / rL3, -mu * self.yL / rL3
        dxS, dyS = self.xS - self.xL, self.yS - self.yL
        dS3 = (dxS * dxS + dyS * dyS) ** 1.5
        rS3 = (self.xS * self.xS + self.yS * self.yS) ** 1.5
        aSx = MU_SOL * (dxS / dS3 - self.xS / rS3)
        aSy = MU_SOL * (dyS / dS3 - self.yS / rS3)
        return aTx + aSx, aTy + aSy

    def _step(self, dt: float) -> None:
        axS, ayS = self._accel_sun()
        axL, ayL = self._accel_moon()
        self.xS += self.vxS * dt + 0.5 * axS * dt * dt
        self.yS += self.vyS * dt + 0.5 * ayS * dt * dt
        self.xL += self.vxL * dt + 0.5 * axL * dt * dt
        self.yL += self.vyL * dt + 0.5 * ayL * dt * dt
        axS_n, ayS_n = self._accel_sun()
        axL_n, ayL_n = self._accel_moon()
        self.vxS += 0.5 * (axS + axS_n) * dt
        self.vyS += 0.5 * (ayS + ayS_n) * dt
        self.vxL += 0.5 * (axL + axL_n) * dt
        self.vyL += 0.5 * (ayL + ayL_n) * dt
        self.t_actual += dt

    # Paso máximo de Verlet [s]: evita que un salto grande de t se
    # integre de un solo paso (el uso normal, dt=0.25 s, no cambia).
    _DT_MAX = 10.0
    # (12.13.4) Salto máximo de una llamada [s] (~1 año): un t absurdo (fecha mal parseada) no debe
    # integrar años de Verlet con la GUI colgada. Los usos reales (decenas de días) no se acercan.
    _AVANCE_MAX_S = 3.2e7

    def advance_to(self, t: float) -> None:
        if not math.isfinite(t):
            raise ValueError(f"advance_to(t={t!r}): la época no es finita; revise la fecha de lanzamiento")
        salto = t - self.t_actual
        if salto > self._AVANCE_MAX_S:
            raise ValueError(
                f"advance_to(t={t!r}): salto de {salto:.2e} s desde "
                f"t_actual={self.t_actual!r} supera {self._AVANCE_MAX_S:.1e} s "
                "(≈1 año); revise fecha_lanzamiento y el horizonte de efemérides")
        while self.t_actual < t - 1e-12:
            self._step(min(t - self.t_actual, self._DT_MAX))

    def pos_sun(self):
        return self.xS, self.yS

    def pos_moon(self):
        return self.xL, self.yL

    def vel_sun(self):
        return self.vxS, self.vyS

    def vel_moon(self):
        return self.vxL, self.vyL


# ============================================================
# EFEMÉRIDES DEL MODO LUNAR: HORIZONS + VERLET 3D (12.4.0)
# ============================================================
# Alternativa a SPICE para el modo lunar. Se consulta Horizons UNA vez por
# cuerpo (estado 3D geocéntrico, eclíptico J2000), se integra con Verlet de
# velocidades en 3D todo el horizonte de la misión y se guarda una tabla
# (cada `paso_tabla` s) que se consulta por Hermite en cualquier orden.
# Modelo: Luna con (mu_T + mu_L) más la marea directa del Sol; Sol en dos
# cuerpos con la Tierra. No incluye planetas, J2 ni relatividad: es un
# modelo simplificado y así se etiqueta. Su discrepancia se mide con
# `validar_horizons_verlet.py`; sin informe vigente no se considera
# validado para una época concreta.
ETIQUETA_HORIZONS_VERLET = "Horizons + Verlet 3D (modelo simplificado, no es SPICE)"


def _parsear_vectores_horizons(texto: str, cuerpo: str):
    """Primera línea del bloque $$SOE/$$EOE de una consulta VECTORS en CSV
    (KM-S) -> (x, y, z, vx, vy, vz) en SI. Lanza RuntimeError con el motivo."""
    for linea in texto.splitlines()[:40]:
        low = linea.lower()
        if ("cannot" in low or "no ephemeris" in low or "error" in low
                or "no such" in low):
            raise RuntimeError(f"Horizons: {linea.strip()[:200]}")
    m = re.search(r"\$\$SOE\s*(.*?)\s*\$\$EOE", texto, re.DOTALL)
    if not m or not m.group(1).strip():
        raise RuntimeError("Horizons: no se encontró un bloque $$SOE/$$EOE con datos")
    linea = m.group(1).strip().splitlines()[0]
    partes = [q.strip() for q in linea.split(",")]
    if len(partes) < 8:
        raise RuntimeError(f"Horizons: línea con {len(partes)} campos (esperaba >=8): {linea[:120]}")
    try:
        v = [float(partes[i]) * 1000.0 for i in (2, 3, 4, 5, 6, 7)]
    except ValueError as e:
        raise RuntimeError(f"Horizons: campo no numérico ({e})") from e
    if not all(math.isfinite(c) for c in v):
        raise RuntimeError("Horizons: estado no finito")
    d = math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)
    if cuerpo == "luna" and not (3.3e8 < d < 4.2e8):
        raise RuntimeError(f"Horizons: distancia Luna insospechada: {d/1e3:.0f} km")
    if cuerpo == "sol" and not (1.4e11 < d < 1.6e11):
        raise RuntimeError(f"Horizons: distancia Sol insospechada: {d/1e9:.3f} Gm")
    return tuple(v)


def _horizons_consulta_texto(cuerpo: str, inicio: str, fin: str, paso: str,
                             timeout: float = 30.0) -> str:
    """Texto crudo de Horizons (vectores geocéntricos, eclíptica J2000, km y
    km/s, CSV). `inicio`/`fin` en 'YYYY-MM-DD HH:MM:SS' (UT)."""
    import urllib.request
    import urllib.parse
    import urllib.error
    params = {
        "format": "text", "COMMAND": f"'{CUERPOS[cuerpo].horizons_id}'",
        "EPHEM_TYPE": "'VECTORS'", "CENTER": "'500@399'",
        "START_TIME": f"'{inicio}'", "STOP_TIME": f"'{fin}'",
        "STEP_SIZE": f"'{paso}'", "REF_PLANE": "'ECLIPTIC'",
        "VEC_TABLE": "'2'", "OUT_UNITS": "'KM-S'", "CSV_FORMAT": "'YES'",
        "VEC_LABELS": "'NO'",
    }
    url = "https://ssd.jpl.nasa.gov/api/horizons.api?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": "cohete-sim/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read().decode("utf-8", errors="replace")
    except (urllib.error.URLError, ValueError, OSError) as e:
        raise RuntimeError(f"Horizons: fallo de red/URL: {e}")


def _fecha_horizons_seg(fecha_iso: str, desplazamiento_s: float = 0.0) -> str:
    f = datetime.fromisoformat(fecha_iso)
    f = (f.astimezone(timezone.utc).replace(tzinfo=None)
         if f.tzinfo is not None else f)
    f = f + timedelta(seconds=desplazamiento_s)
    return f.strftime("%Y-%m-%d %H:%M:%S")


def _fetch_horizons_state_3d(cuerpo: str, fecha_iso: str, timeout: float = 30.0):
    """Estado 3D (x, y, z, vx, vy, vz) [SI] del Sol o la Luna respecto a la
    Tierra en `fecha_iso` (UT, con segundos)."""
    texto = _horizons_consulta_texto(
        cuerpo, _fecha_horizons_seg(fecha_iso), _fecha_horizons_seg(fecha_iso, 60.0),
        "1 m", timeout)
    return _parsear_vectores_horizons(texto, cuerpo)


@_njit(cache=True)
def _acc_sol_luna_3d(y, a, mu_st, mu_tl, mu_sol):
    """a[0:3] = aceleración del Sol (2 cuerpos con la Tierra); a[3:6] = la de
    la Luna (Tierra+Luna central + marea directa del Sol). y = (rS, vS, rL, vL)."""
    rsx, rsy, rsz = y[0], y[1], y[2]
    rlx, rly, rlz = y[6], y[7], y[8]
    rs3 = (rsx * rsx + rsy * rsy + rsz * rsz) ** 1.5
    rl3 = (rlx * rlx + rly * rly + rlz * rlz) ** 1.5
    dx, dy, dz = rsx - rlx, rsy - rly, rsz - rlz
    d3 = (dx * dx + dy * dy + dz * dz) ** 1.5
    a[0] = -mu_st * rsx / rs3
    a[1] = -mu_st * rsy / rs3
    a[2] = -mu_st * rsz / rs3
    a[3] = -mu_tl * rlx / rl3 + mu_sol * (dx / d3 - rsx / rs3)
    a[4] = -mu_tl * rly / rl3 + mu_sol * (dy / d3 - rsy / rs3)
    a[5] = -mu_tl * rlz / rl3 + mu_sol * (dz / d3 - rsz / rs3)


@_njit(cache=True)
def _k_verlet3d(y0, dt, n_sub, n_nodos, mu_st, mu_tl, mu_sol):
    """Verlet de velocidades. Devuelve (n_nodos, 12): una fila por nodo (cada
    n_sub pasos de dt). Columnas: Sol (x,y,z,vx,vy,vz), Luna (x,y,z,vx,vy,vz)."""
    out = np.empty((n_nodos, 12))
    y = y0.copy()
    a = np.empty(6)
    an = np.empty(6)
    _acc_sol_luna_3d(y, a, mu_st, mu_tl, mu_sol)
    for k in range(n_nodos):
        for j in range(12):
            out[k, j] = y[j]
        if k == n_nodos - 1:
            break
        for _ in range(n_sub):
            for c in range(2):
                o = 6 * c
                for i in range(3):
                    y[o + i] += y[o + 3 + i] * dt + 0.5 * a[3 * c + i] * dt * dt
            _acc_sol_luna_3d(y, an, mu_st, mu_tl, mu_sol)
            for c in range(2):
                o = 6 * c
                for i in range(3):
                    y[o + 3 + i] += 0.5 * (a[3 * c + i] + an[3 * c + i]) * dt
            for i in range(6):
                a[i] = an[i]
    return out


def _tabla_verlet3d_validada(y0, dt, n_sub, n_nodos, mu_st, mu_tl, mu_sol):
    """(12.9.1) `_k_verlet3d` con diagnóstico: una distancia nula o un estado no finito
    detienen la construcción de la tabla (no se sustituye nada por un épsilon)."""
    y0 = np.asarray(y0, dtype=float)
    if not np.all(np.isfinite(y0)):
        raise EfemeridesNoValidas("Verlet 3D: el estado inicial no es finito")
    try:
        tabla = _k_verlet3d(y0, dt, n_sub, n_nodos, mu_st, mu_tl, mu_sol)
    except ZeroDivisionError as e:
        raise EfemeridesNoValidas(
            "Verlet 3D: distancia nula entre cuerpos (singularidad gravitatoria)") from e
    if not np.all(np.isfinite(tabla)):
        k = int(np.argmax(~np.all(np.isfinite(tabla), axis=1)))
        raise EfemeridesNoValidas(
            f"Verlet 3D: estado no finito en el nodo {k} (t={k * n_sub * dt:.0f} s)")
    return tabla


class _ProveedorHorizonsVerlet:
    """Proveedor de efemérides `(cuerpo, t_sim) -> ((x, y), (vx, vy))` con el
    contrato del modo lunar (consulta en cualquier orden, determinista), a
    partir de UNA consulta a Horizons + Verlet 3D + Hermite. t_sim son los
    segundos desde `fecha_lanzamiento`; el resultado es eclíptico J2000
    geocéntrico proyectado al plano de la eclíptica (como la ruta SPICE)."""
    etiqueta = ETIQUETA_HORIZONS_VERLET

    def __init__(self, fecha_iso: str, horizonte_s: float, dt: float = 20.0,
                 paso_tabla: float = 600.0):
        n_sub = int(round(paso_tabla / dt))
        if n_sub < 1 or abs(n_sub * dt - paso_tabla) > 1e-9:
            raise ValueError("paso_tabla debe ser múltiplo entero de dt")
        self.t_fin = float(horizonte_s)
        self.paso = float(paso_tabla)
        n_nodos = int(math.ceil(self.t_fin / self.paso)) + 2
        sol = _fetch_horizons_state_3d("sol", fecha_iso)
        luna = _fetch_horizons_state_3d("luna", fecha_iso)
        y0 = np.array(list(sol) + list(luna), dtype=float)
        self.estado0 = y0.copy()
        t3 = _tabla_verlet3d_validada(y0, float(dt), n_sub, n_nodos,
                                      MU_SOL + MU_T, MU_T + MU_LUNA, MU_SOL)   # (12.13.4) sin atributo
        # Se conserva la tabla física 3D original (x,y,z,vx,vy,vz para cada
        # cuerpo). Así la interpolación de z usa sus columnas reales, sin una
        # tabla auxiliar con columnas artificiales.
        self._tab3d = np.ascontiguousarray(t3)

    def __call__(self, cuerpo: str, t: float):
        pos, vel = self.estado_3d(cuerpo, t)
        return (pos[0], pos[1]), (vel[0], vel[1])

    def estado_3d(self, cuerpo: str, t: float):
        """Devuelve ((x,y,z),(vx,vy,vz)) [SI] en geocéntrico eclíptico J2000."""
        if cuerpo not in ("sol", "luna"):
            raise EfemeridesNoValidas(f"{self.etiqueta}: cuerpo {cuerpo!r} no soportado")
        if not (-1.0 <= t <= self.t_fin + self.paso):
            raise EfemeridesNoValidas(f"{self.etiqueta}: t={t:.0f} s fuera de la tabla")
        offset = 0 if cuerpo == "sol" else 6
        pos = _herm_pos3d(self._tab3d, 0.0, self.paso, float(t), offset)
        vel = _herm_vel3d(self._tab3d, 0.0, self.paso, float(t), offset)
        return pos, vel

    def z(self, cuerpo: str, t: float) -> float:
        """Coordenada z [m]; conservada por compatibilidad con el HUD."""
        return self.estado_3d(cuerpo, t)[0][2]


ETIQUETA_REBOUND = "REBOUND IAS15 (masas puntuales; no es SPICE)"

# GM [m^3/s^2] de los baricentros planetarios (valores tipo DE440). Perturban a Sol, Tierra y Luna;
# sin ellos el Sol geocéntrico se aparta ~50 km a 10 d (sobre todo por Júpiter).
_GM_PLANETAS_BARICENTROS = {
    "1": 2.2031868551e13, "2": 3.2485859200e14, "4": 4.2828375816e13,
    "5": 1.2671276480e17, "6": 3.7940584842e16, "7": 5.7945486e15, "8": 6.8365271e15,
}


def _importar_rebound():
    try:
        import rebound
    except ImportError as e:
        raise EfemeridesNoValidas(
            "efemerides_lunar='rebound' exige REBOUND (`pip install rebound`); "
            "o elija 'spice' / 'horizons_verlet'") from e
    return rebound


def _estados_iniciales_rebound(fecha_iso: str):
    """Estados iniciales en un marco INERCIAL (eclíptico J2000): lista [(GM, estado6)] y los
    índices (Sol, Tierra, Luna). Con SPICE: Sol, Tierra, Luna y los 7 baricentros planetarios
    (marco baricéntrico). Sin SPICE (kernels/red): Sol y Luna geocéntricos de Horizons con la
    Tierra en reposo en el origen (invariancia galileana) y SIN planetas.
    Devuelve (cuerpos, i_sol, i_tierra, i_luna, origen, con_planetas)."""
    try:
        _asegurar_kernels_spice()
        f = datetime.fromisoformat(_fecha_utc_normalizada(fecha_iso))
        jd = float(Time(f).tdb.jd)

        def estado(id_):
            data, _ts, _df = pc.consulta_spice(id=id_, location="@0", epochs=[jd])
            arr = np.asarray(data, dtype=float).reshape(-1, 6)
            return arr[0]
        cuerpos = [(MU_SOL, estado("10")), (MU_T, estado("399")), (MU_LUNA, estado("301"))]
        cuerpos += [(gm, estado(k)) for k, gm in _GM_PLANETAS_BARICENTROS.items()]
        if not all(np.all(np.isfinite(e)) for _, e in cuerpos):
            raise ValueError("estado no finito de SPICE")
        return cuerpos, 0, 1, 2, "SPICE + planetas", True
    except (SpiceyError, OSError, ValueError, KeyError, RuntimeError):
        sol = np.array(_fetch_horizons_state_3d("sol", fecha_iso), dtype=float)
        luna = np.array(_fetch_horizons_state_3d("luna", fecha_iso), dtype=float)
        cuerpos = [(MU_SOL, sol), (MU_T, np.zeros(6)), (MU_LUNA, luna)]
        return cuerpos, 0, 1, 2, "Horizons, sin planetas", False


def _tabla_rebound(rebound, cuerpos, i_sol, i_tierra, i_luna, paso: float, n_nodos: int):
    """Tabla (n_nodos, 12) con Sol y Luna relativos a la Tierra (x,y,z,vx,vy,vz), integrando
    con IAS15 (G=1, masas = GM). Un nodo cada `paso` s."""
    sim = rebound.Simulation()
    sim.G = 1.0
    sim.integrator = "ias15"
    for mu, e in cuerpos:
        sim.add(m=float(mu), x=e[0], y=e[1], z=e[2], vx=e[3], vy=e[4], vz=e[5])
    sim.move_to_com()
    out = np.empty((n_nodos, 12))
    for k in range(n_nodos):
        sim.integrate(float(k * paso), exact_finish_time=1)
        p = sim.particles
        t = p[i_tierra]
        for col, idx in ((0, i_sol), (6, i_luna)):
            q = p[idx]
            out[k, col:col + 6] = (q.x - t.x, q.y - t.y, q.z - t.z,
                                   q.vx - t.vx, q.vy - t.vy, q.vz - t.vz)
    if not np.all(np.isfinite(out)):
        k = int(np.argmax(~np.all(np.isfinite(out), axis=1)))
        raise EfemeridesNoValidas(f"REBOUND: estado no finito en el nodo {k} (t={k * paso:.0f} s)")
    return out


class _ProveedorRebound(_ProveedorHorizonsVerlet):
    """(12.13.8) Mismo contrato que `_ProveedorHorizonsVerlet` (tabla + Hermite 3D, consulta en
    cualquier orden), pero la tabla sale de REBOUND/IAS15 con Sol, Tierra, Luna y los planetas.
    Medido contra SPICE (validar_rebound_v1.py, 2026-01-01, 10 d): Luna 0,33 km y Sol 0,07 km
    con planetas (Verlet: 0,44 km y 11 500 km). Los baricentros planetarios solo se usan si hay
    SPICE; sin él el Sol llega a ~50 km a 10 d y la Luna casi no cambia."""
    etiqueta = ETIQUETA_REBOUND

    def __init__(self, fecha_iso: str, horizonte_s: float, paso_tabla: float = 600.0,
                 estados=None):
        rebound = _importar_rebound()
        if not (paso_tabla > 0.0 and math.isfinite(paso_tabla)):
            raise ValueError("paso_tabla debe ser positivo y finito")
        self.t_fin = float(horizonte_s)
        self.paso = float(paso_tabla)
        n_nodos = int(math.ceil(self.t_fin / self.paso)) + 2
        cuerpos, i_s, i_t, i_l, self.origen, self.con_planetas = (
            estados if estados is not None else _estados_iniciales_rebound(fecha_iso))
        self.etiqueta = f"REBOUND IAS15 ({self.origen}; masas puntuales)"
        t3 = _tabla_rebound(rebound, cuerpos, i_s, i_t, i_l, self.paso, n_nodos)
        self.estado0 = t3[0].copy()
        self._tab3d = np.ascontiguousarray(t3)


VALIDACION_HORIZONS_ARCHIVO = Path(__file__).with_name("validacion_horizons_verlet.json")


def _fecha_utc_normalizada(fecha_iso: str) -> str:
    f = datetime.fromisoformat(fecha_iso)
    if f.tzinfo is None:
        f = f.replace(tzinfo=timezone.utc)
    else:
        f = f.astimezone(timezone.utc)
    return f.replace(microsecond=0).isoformat()


def _sha256_archivo(ruta: Path) -> str:
    sha = hashlib.sha256()
    with ruta.open("rb") as f:
        for bloque in iter(lambda: f.read(1024 * 1024), b""):
            sha.update(bloque)
    return sha.hexdigest()


def _leer_validacion_horizons_verlet(fecha_iso: str):
    """Solo acepta un informe de esta versión exacta y de la misma época T0."""
    try:
        datos = json.loads(VALIDACION_HORIZONS_ARCHIVO.read_text(encoding="utf-8"))
        esperado = _sha256_archivo(Path(__file__))
        if datos.get("schema") != 1 or datos.get("motor_sha256") != esperado:
            return None
        if datos.get("fecha_lanzamiento_utc") != _fecha_utc_normalizada(fecha_iso):
            return None
        modelo = datos.get("modelo", {})
        if (modelo.get("integrador") != "velocity-Verlet 3D"
                or modelo.get("dt_interno_s") != 20.0
                or modelo.get("paso_tabla_s") != 600.0
                or modelo.get("marco") != "geocéntrico eclíptico J2000"):
            return None
        if modelo.get("fuerzas") != (
                "Sol-Tierra dos cuerpos; Luna Tierra-Luna + marea solar; "
                "sin planetas/J2/relatividad"):
            return None
        if datos.get("muestras_dias") != [1, 2, 5, 10]:
            return None
        umbrales = datos.get("umbrales", {})
        if (umbrales.get("luna_posicion_km") != 5.0
                or umbrales.get("sol_posicion_km") != 50.0):
            return None
        errores = datos.get("errores")
        if not isinstance(errores, dict):
            return None
        for cuerpo in ("luna", "sol"):
            filas = errores.get(cuerpo)
            if not isinstance(filas, dict) or set(filas) != {"1", "2", "5", "10"}:
                return None
            for fila in filas.values():
                if not isinstance(fila, dict):
                    return None
                dr, dv = fila.get("dr_m"), fila.get("dv_m_s")
                if not (isinstance(dr, (int, float)) and math.isfinite(dr) and dr >= 0.0
                        and isinstance(dv, (int, float)) and math.isfinite(dv) and dv >= 0.0):
                    return None
        if not isinstance(datos.get("aceptada"), bool):
            return None
        # No confiar en el booleano almacenado: se recalcula desde los errores
        # y umbrales del informe para no aceptar un JSON inconsistente.
        max_luna = max(float(v["dr_m"]) for v in errores["luna"].values())
        max_sol = max(float(v["dr_m"]) for v in errores["sol"].values())
        aceptada_calculada = (max_luna <= 5_000.0 and max_sol <= 50_000.0)
        if datos["aceptada"] is not aceptada_calculada:
            return None
        return datos
    except (OSError, ValueError, TypeError, AttributeError):
        return None


# ============================================================
# MARCO MÓVIL: TIERRA -> CRUCERO (CRTBP, baricentro Tierra-Luna,
# marco ROTANTE) -> LUNA
# ============================================================
# Restricted: Tierra y Luna van sobre sus propios rieles (no sienten
# ni a la nave ni entre sí más que por su propia dinámica de 2
# cuerpos, que en el tramo de crucero se idealiza como circular —
# la aproximación estándar del CRTBP); la nave es una partícula de
# prueba. Tres formulaciones físicas, una por zona:
#
#   "tierra"   marco inercial centrado en la Tierra (el de siempre).
#              Tierra = primario. Sol/Luna = perturbación de marea
#              (si usar_terceros_cuerpos). Atmósfera/arrastre activos.
#   "crucero"  marco ROTANTE centrado en el baricentro Tierra-Luna,
#              girando a la velocidad angular Ω "congelada" en el
#              instante de entrada (se asume el movimiento relativo
#              Tierra-Luna circular durante todo el crucero: la
#              excentricidad lunar real, 0.055, queda fuera de esta
#              aproximación). Tierra y Luna quedan FIJAS en este
#              marco. Ecuaciones clásicas del CRTBP (Coriolis +
#              centrífuga + ambas gravedades completas). Sin
#              atmósfera ni perturbación solar (ver limitación abajo).
#   "luna"     marco inercial centrado en la Luna. Luna = primario.
#              Tierra/Sol = perturbación de marea. Sin atmósfera.
#
# Limitación explícita: durante "crucero" NO se modela la
# perturbación solar (eso requeriría el problema bicircular de 4
# cuerpos, porque el Sol no queda fijo en un marco que rota con el
# mes sideral) — queda pendiente como refinamiento futuro.
#
# Todo esto es puramente el mecanismo de propagación; no hay todavía
# fases de misión (inyección translunar, crucero, inserción lunar)
# que lo disparen desde el ascenso normal. Se activa solo si
# `usar_marco_movil=True` Y la nave efectivamente llega a esas
# distancias (por ahora, solo alcanzable a mano/con pruebas).

MU_SISTEMA_TL = MU_T + MU_LUNA                 # G(m_T+m_L) [m^3/s^2], parámetro gravitacional de 2 cuerpos
RAZON_MASA_TL = MU_LUNA / MU_SISTEMA_TL         # ADIMENSIONAL: m_L/(m_T+m_L) del CRTBP (~0.0122); no es un GM
_A_LUNA_REF = CUERPOS["luna"].a_ref           # semieje de referencia Tierra-Luna
R_SOI_LUNA  = radio_soi("luna")               # ~66,170 km
R_HILL_LUNA = radio_hill("luna")               # ~61,530 km (solo HUD)


def zona_lunar(r: float) -> str:
    """Región del entorno lunar según la distancia `r` [m] al centro de la
    Luna: "estable" (r < Hill), "transito" (Hill <= r < SOI) o "tierra"
    (r >= SOI). Informativa: el cambio de marco sigue usando SOI."""
    if r < R_HILL_LUNA:
        return "estable"
    return "transito" if r < R_SOI_LUNA else "tierra"


def _rot2d(vec, ang):
    c, s = np.cos(ang), np.sin(ang)
    x, y = vec
    return (c * x - s * y, s * x + c * y)


# --- Escala logarítmica de VISTA (solo dibujo) ---------------
# Distancias: r <= R_T queda igual (continuo y con derivada 1 en
# R_T); r > R_T se comprime como R_T·(1 + ln(r/R_T)). La Luna a
# ~60 R_T se dibuja a ~5.1 R_T. La dirección (ángulo) se conserva.
# Tamaños: s' = R_T·ln(1 + (e-1)·s/R_T), de modo que s = R_T queda
# igual (la Tierra no cambia) y los cuerpos pequeños se agrandan
# de forma logarítmica (el radio de la Luna: 0.27 R_T -> 0.38 R_T).
def vista_xy(x, y, activa: bool = True):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if not activa:
        return x, y
    r = np.hypot(x, y)
    r_safe = np.maximum(r, R_T)
    rv = np.where(r > R_T, R_T * (1.0 + np.log(r_safe / R_T)), r)
    f = np.divide(rv, r, out=np.ones_like(r, dtype=float), where=r > 0)
    return x * f, y * f


def vista_tam(s: float, activa: bool = True) -> float:
    if not activa:
        return float(s)
    return float(R_T * np.log1p((np.e - 1.0) * s / R_T))


def _cross_z(Omega, vec):
    x, y = vec
    return (-Omega * y, Omega * x)


def _inercial_a_rotante(r_in, v_in, Omega, theta):
    """r_in, v_in: posición/velocidad relativas al origen del marco
    rotante, expresadas en orientación INERCIAL (fija). Devuelve
    (r_rot, v_rot) en el marco rotante."""
    rx, ry = _rot2d(r_in, -theta)
    vx0, vy0 = _rot2d(v_in, -theta)
    cx, cy = _cross_z(Omega, (rx, ry))
    return (rx, ry), (vx0 - cx, vy0 - cy)


def _rotante_a_inercial(r_rot, v_rot, Omega, theta):
    """Inversa de _inercial_a_rotante."""
    cx, cy = _cross_z(Omega, r_rot)
    v_pre = (v_rot[0] + cx, v_rot[1] + cy)
    r_in = _rot2d(r_rot, theta)
    v_in = _rot2d(v_pre, theta)
    return r_in, v_in


# ------------------------------------------------------------
# Puntos de Lagrange L1-L5 (análisis / overlay visual, opcional)
# ------------------------------------------------------------
# Geometría de referencia únicamente (usa la distancia MEDIA
# Tierra-Luna, _A_LUNA_REF; la real varía ±13,000 km por la
# excentricidad lunar — irrelevante para un overlay de ubicación
# aproximada). Los puntos se calculan en coordenadas a lo largo de la
# línea Tierra-Luna media ("BCR": x medido desde la Tierra hacia la
# Luna, y perpendicular); para dibujarlos hay que rotarlos al ángulo
# REAL de la línea Tierra-Luna en el instante actual (no un ángulo
# idealizado que arranque en 0), para que queden alineados con la
# Luna real que se muestra en pantalla. Eso lo hace quien llama a
# puntos_lagrange_crtbp(), no esta función.
def _ecuacion_lagrange_colineal(x: float, mu: float) -> float:
    """
    Ecuación de equilibrio para los puntos colineales del CRTBP, en
    unidades normalizadas (distancia Tierra-Luna = 1): masa (1-mu)
    [Tierra] fija en x=-mu, masa mu [Luna] fija en x=1-mu.

    dΩ/dx = x - (1-mu)(x+mu)/|x+mu|³ - mu(x-1+mu)/|x-1+mu|³ = 0
    """
    xT, xL = -mu, 1.0 - mu
    return (x
            - (1.0 - mu) * (x - xT) / abs(x - xT) ** 3
            - mu * (x - xL) / abs(x - xL) ** 3)


def _lagrange_colineal(signo_L: int, mu: float) -> float:
    """
    Resuelve _ecuacion_lagrange_colineal con scipy.optimize.brentq
    (bracketed, a diferencia de un secante sin resguardo): signo_L=-1
    -> L1 (entre los dos cuerpos), +1 -> L2 (más allá de la Luna),
    +2 -> L3 (más allá de la Tierra, del otro lado).
    """
    from scipy.optimize import brentq
    xT, xL = -mu, 1.0 - mu
    eps = 1e-6
    if signo_L == -1:
        a, b = xT + eps, xL - eps
    elif signo_L == 1:
        a, b = xL + eps, xL + 1.5
    else:
        a, b = xT - 1.5, xT - eps
    return brentq(lambda x: _ecuacion_lagrange_colineal(x, mu), a, b)


def puntos_lagrange_crtbp():
    """
    Puntos de Lagrange L1-L5 del sistema Tierra-Luna, en coordenadas
    (x, y) medidas desde la Tierra a lo largo de la línea Tierra-Luna
    MEDIA (x) y perpendicular a ella (y). Solo para overlay visual —
    ver limitaciones arriba.
    """
    mu = RAZON_MASA_TL
    x1 = _lagrange_colineal(-1, mu) * _A_LUNA_REF
    x2 = _lagrange_colineal(1, mu) * _A_LUNA_REF
    x3 = _lagrange_colineal(2, mu) * _A_LUNA_REF
    xL4 = (0.5 - mu) * _A_LUNA_REF
    yL4 = np.sqrt(3.0) / 2.0 * _A_LUNA_REF
    return {
        "L1": (x1, 0.0), "L2": (x2, 0.0), "L3": (x3, 0.0),
        "L4": (xL4, yL4), "L5": (xL4, -yL4),
    }


# ============================================================
# CONFIGURACIÓN DE LA MISIÓN Y DE LAS ETAPAS
# ============================================================
@dataclass
class Config:
    modo_mision:      str   = "satelital"   # "satelital" | "tripulada" | "suborbital" | "gto" | "tli" | "escape" | "deorbit" | "lunar"
    perfil_tripulada: str   = "bcb"         # "bcb" | "directo" (solo aplica si modo_mision == "tripulada")

    # --- Parámetros operativos (los edita el usuario) ---
    umbral_encendido: float = 10.0
    VEL_MIN_SES:      float = 2500.0
    ALTURA_SEGURIDAD: float = 120000.0
    APOGEO_OBJETIVO:  float = 1300000.0

    THROTTLE_MARGEN:  float = 30000.0
    THROTTLE_MIN:     float = 0.08
    alt_inicio_giro:  float = 400.0
    angulo_kick:      float = 3.0
    sim_speed:        int   = 1

    # --- Parámetros exclusivos del modo suborbital ---
    # True : corta el motor cuando el apogeo proyectado alcanza
    #        APOGEO_OBJETIVO (vuelo de prueba a una altitud dada).
    # False: quema TODAS las etapas hasta agotarlas (literal), lo
    #        que con un Falcon 9 completo supera con mucho el
    #        objetivo.
    suborbital_cutoff_por_apogeo: bool = True
    KARMAN: float = 100_000.0   # límite del espacio (m)
    # Tolerancia de la meta de apogeo suborbital [m]. Desde 11.21.0 el corte
    # cae EXACTO en el apogeo de vacío proyectado (= objetivo); el arrastre
    # residual en la subida lo baja unos metros (3-4 m a 150 km), así que
    # exigir alt_max >= objetivo a rajatabla dependía de que el corte se
    # pasara de largo por el paso. 0 = criterio estricto.
    TOL_APOGEO_SUBORBITAL: float = 100.0

    # --- Parámetros exclusivos del modo gto ---
    GTO_RP_FINAL: float = 35_000_000.0   # periapsis mínimo tras la quema de apogeo (m)
    ECC_MAX_GTO:  float = 0.02           # excentricidad máxima para dar la GEO por lograda

    # --- Integrador del marco Tierra ---
    # "hibrido": RK4 de paso fijo SOLO en coast libre (sin empuje, por
    # encima de ALT_RK4_MIN, dinámica suave); RK45 (solve_ivp) con
    # empuje, en atmósfera y en cualquier otro caso. "rk45": siempre
    # solve_ivp (comportamiento de v11.8.x). Crucero y Luna siguen
    # siempre con RK45.
    integrador:   str   = "hibrido"
    # (12.0.0) "variable": masa y Δv como estados del ODE y fuerzas evaluadas
    # dentro del integrador. "congelada": dinámica de 11.21.x (masa y fuerzas
    # constantes en cada paso), conservada para comparar.
    dinamica:     str   = "variable"
    # (12.1.0) "integrador": corte y suelo como eventos de solve_ivp (solo con
    # dinamica="variable"); "paso": bisección sobre el paso + Hermite (12.0.1).
    eventos:      str   = "integrador"
    ALT_RK4_MIN:  float = 200_000.0   # por debajo: RK45 (atmósfera, perigeos bajos)
    RK4_PASO_MAX: float = 1.0         # s; un dt mayor se parte en subpasos <= esto

    # --- Parámetros exclusivos de tli / escape ---
    # En coast (sin empuje) el paso sube a DT_COAST_ALTA_ENERGIA para
    # que 5-7 días no sean millones de pasos; la integración sigue
    # siendo RK45 adaptativa dentro de cada paso. T_MAX corta la
    # simulación (en escape la nave no vuelve ni impacta).
    DT_COAST_ALTA_ENERGIA: float = 5.0
    T_MAX_ALTA_ENERGIA:    float = 7.0 * 86400.0
    # (12.10.1) suborbital sin cutoff por apogeo: si la trayectoria no tiene apogeo ni impacto
    # (energía de escape) la simulación terminaría nunca; este tope la corta.
    T_MAX_SUBORBITAL:      float = 3.0 * 3600.0

    # --- Aerodinámica y propulsión dependientes de la altitud (v11.17.0) ---
    # cd_vs_mach: Cd = Cd_etapa · f(Mach) (divergencia de arrastre transónica).
    # empuje_vs_altitud: T(h) = T_vacío − A_salida · p_ambiente(h), con
    #   Isp efectivo = T/(ṁ·g0). Apagados reproducen v11.16.0 bit a bit.
    cd_vs_mach:        bool = True
    empuje_vs_altitud: bool = True

    # --- Parámetros exclusivos del modo deorbit ---
    deorbit_activacion:   str   = "orbita"   # "sitio" | "orbita" | "t" (el preset de deorbit usa "sitio")
    deorbit_t_activacion: float = 300.0      # s (solo con activacion="t")
    # Modo "sitio": se prueban horas de quema en copias de la simulación
    # hasta que el punto de caída quede a menos de DEORBIT_SITIO_TOL_KM
    # de la base (o se agoten DEORBIT_SITIO_MAX_ITER refinamientos).
    DEORBIT_SITIO_TOL_KM:   float = 5.0
    DEORBIT_SITIO_MAX_ITER: int   = 12
    # Solo ritmo de la GUI (la física no cambia):
    DEORBIT_VEL_MAX_PLAN:   int   = 8     # pasos/fotograma máx. mientras el plan externo no está listo
    DEORBIT_PASO_QUEMA_MS:  int   = 60    # ms entre pasos durante la quema de desórbita
    DEORBIT_PASO_ESPERA_MS: int   = 50    # ms entre fotogramas si hay que esperar el plan
    LUNAR_PASOS_MIN_COAST: int = 10            # pasos/fotograma mínimos en espera_tli y translunar (12.5.6)
    DEORBIT_PLAN_PRESUPUESTO_S: float = 0.03   # s de cálculo del plan por fotograma de la GUI (12.1.2)
    DEORBIT_SITIO_HORIZONTE: float = 6000.0  # s simulados tras la quema de prueba
    DEORBIT_RP_OBJETIVO:  float = 40_000.0   # periapsis de reentrada (m)
    DEORBIT_ALT_REENTRADA: float = 120_000.0 # altitud que confirma la reentrada (m)

    # --- Parámetros exclusivos del modo tripulado ---
    G_MAX_TRIPULADA:           float = 4.0     # aceleración máxima tolerada (en g)
    ECC_MAX_TRIPULADA_BCB:     float = 0.01    # tolerancia de ecc. — perfil bcb (órbita casi circular)
    ECC_MAX_TRIPULADA_DIRECTO: float = 0.08    # tolerancia de ecc. — perfil directo (más elíptica)

    # --- Perturbaciones de terceros cuerpos (Sol, Luna, ...) ---
    # Apagado por defecto: no cambia ningún resultado existente a
    # menos que se active explícitamente. Ver sección "TERCEROS
    # CUERPOS" más abajo para el modelo y sus limitaciones.
    usar_terceros_cuerpos: bool = False
    cuerpos_perturbadores: tuple = ("sol", "luna")
    fecha_lanzamiento:     str   = "2026-01-01T00:00:00"  # ISO 8601, UTC

    # --- Marco móvil Tierra -> crucero (CRTBP) -> Luna ---
    # Apagado por defecto. Ver sección "MARCO MÓVIL" más arriba.
    usar_marco_movil: bool = False

    # --- Overlay visual (análisis CRTBP: puntos de Lagrange) ---
    mostrar_crtbp: bool = False

    # Escala logarítmica SOLO de visualización (animación y gráfico
    # final): comprime radialmente las distancias a la Tierra y los
    # tamaños para que Tierra, Luna y nave quepan en la misma vista.
    # No afecta a la física ni a los datos guardados.
    escala_log: bool = True

    # Marco de VISTA (solo dibujo; la física sigue en ECI):
    #   "tierra"     -> Tierra en el origen (ECI)
    #   "baricentro" -> baricentro Tierra-Luna en el origen (inercial);
    #                   la Tierra y la Luna giran despacio a su alrededor.
    marco_vista: str = "tierra"

    # Cuadrícula de coordenadas sobre la Tierra (meridianos y
    # paralelos vistos desde el polo norte). Solo dibujo: gira a la
    # velocidad angular efectiva del modelo, así que con
    # rotacion_terrestre=True se ve girar y con False queda quieta.
    mostrar_grilla_tierra: bool = True

    # --- Rotación terrestre ---
    # La simulación es inercial (ECI, 2D en el plano orbital), así que
    # Coriolis y centrífuga NO se suman: ya están contenidas en la
    # dinámica inercial (agregarlas sería contarlas dos veces). La
    # rotación entra por (1) la velocidad inicial de la plataforma y
    # (2) el arrastre, medido respecto al aire que co-rota con la Tierra.
    # Modelo plano: solo se usa la componente de v_rot en el plano de
    # vuelo, k = cos(lat)·sen(azimut). La componente fuera del plano
    # (v_rot·cos(az)) haría falta en 3D y aquí se descarta.
    rotacion_terrestre:  bool  = False
    latitud_lanzamiento: float = 0.0    # grados, [-90, 90]

    # --- Modo "lunar" (12.3.0): encuentro con la Luna ---
    # Longitud ECLÍPTICA (J2000) del sitio de lanzamiento en t=0, en grados
    # [0, 360], medida desde el eje x de las efemérides, en el sentido en que
    # gira la Luna. Con la fecha fija la fase inicial respecto de la Luna. Solo
    # el modo "lunar" la usa; en los demás debe quedar en 90 (valor histórico)
    # para que sus resultados no cambien ni un bit (se rechaza otro valor).
    # La simulación sigue corriendo en el «marco de lanzamiento» (sitio en
    # +y, vuelo hacia +x, como siempre) y lo que se transforma son las
    # efemérides de entrada; ver `Simulacion._a_marco_lanzamiento`.
    longitud_lanzamiento_ecliptica: float = 90.0
    # (12.4.0) fuente de efemérides del modo lunar: "spice" | "horizons_verlet"
    # (12.5.0: por defecto "horizons_verlet")
    efemerides_lunar: str = "rebound"   # (12.13.8; antes "horizons_verlet")
    # Órbita de aparcamiento: excentricidad máxima para darla por segura.
    ECC_MAX_APARCAMIENTO: float = 0.01
    # Paso de coast (sin empuje) del modo lunar [s]; se acorta para caer
    # exacto en el encendido TLI y cerca de la Luna.
    DT_COAST_LUNAR: float = 20.0
    # Tiempo máximo de misión [s desde el lanzamiento]; sin SOI -> NO CUMPLIDA.
    T_MAX_LUNAR: float = 10.0 * 86400.0
    # Radio geocéntrico del apogeo de la órbita de transferencia [m]: fija la
    # energía de la quema (ε = -μ/(rp + ra)).
    LUNAR_APOGEO_TLI: float = 384_400_000.0
    # Perilunio objetivo medido desde el centro de la Luna [m]. Por defecto,
    # 200 km dentro del radio lunar para probar el detector de impacto. Si se
    # desea un sobrevuelo, usar un valor > R_LUNA (p. ej. 15_000_000 m).
    LUNAR_DIST_OBJETIVO: float = R_LUNA - 200_000.0
    # Ventana de búsqueda del encendido: periodos de la órbita de
    # aparcamiento desde que se declara; paso de la rejilla gruesa [s].
    LUNAR_VENTANA_PERIODOS: float = 2.0
    LUNAR_PASO_VENTANA_S: float = 30.0
    # Horizonte de la propagación de planificación tras el encendido [s].
    LUNAR_HORIZONTE_TRANSFERENCIA: float = 6.0 * 86400.0
    # Δv disponible / Δv requerido mínimo para aceptar la quema TLI.
    LUNAR_MARGEN_DV: float = 1.02
    # (12.6.1) Captura lunar (LOI). Con LUNAR_LOI=True el periselene objetivo es
    # R_LUNA + LUNAR_ORBITA_ALT (órbita circular de esa altitud) y la misión solo
    # se cumple al verificar la órbita ligada durante LUNAR_PERIODOS_VERIF periodos.
    LUNAR_LOI: bool = False
    LUNAR_ORBITA_ALT: float = 100_000.0
    LUNAR_PERIODOS_VERIF: float = 3.0
    # (12.7.1) Tolerancias de la órbita objetivo con LOI: |alt(rp) − LUNAR_ORBITA_ALT| y
    # |alt(ra) − LUNAR_ORBITA_ALT| [m] y excentricidad máxima. Fuera de ellas la
    # captura es ligada y segura, pero la misión NO se cumple.
    LUNAR_ORBITA_TOL_ALT: float = 15_000.0
    LUNAR_ORBITA_ECC_MAX: float = 0.02
    # (12.5.11) Propagación extra del planificador si al llegar al horizonte la
    # nave ya está en la SOI y aún no pasó el periselene [s].
    LUNAR_EXTENSION_PERISELENE_S: float = 2.0 * 86400.0
    # |z| de la Luna (fuera del plano de la nave) por encima del cual un
    # contacto 2D no se considera físico en 3D [m].
    LUNAR_Z_MAX_FISICO: float = 100_000.0
    azimut_lanzamiento:  float = 90.0   # grados desde el Norte, [0, 180] (90 = Este)

    def factor_rotacion(self) -> float:
        """k = cos(lat)·sen(azimut): fracción de la velocidad de la
        plataforma que cae en el plano de vuelo (0 si está apagada)."""
        if not self.rotacion_terrestre:
            return 0.0
        return (math.cos(math.radians(self.latitud_lanzamiento))
                * math.sin(math.radians(self.azimut_lanzamiento)))

    def omega_efectiva(self) -> float:
        """Velocidad angular de la Tierra proyectada al plano de vuelo."""
        return OMEGA_T * self.factor_rotacion()

    def inclinacion_orbital(self) -> float:
        """Inclinación resultante [°]: cos i = cos(lat)·sen(azimut)
        (independiente de que la rotación esté activa)."""
        k = (np.cos(np.radians(self.latitud_lanzamiento))
             * np.sin(np.radians(self.azimut_lanzamiento)))
        return float(np.degrees(np.arccos(np.clip(k, -1.0, 1.0))))

    # Valores admitidos de los campos de texto (única fuente para la validación).
    _MODOS_VALIDOS = ("satelital", "tripulada", "suborbital", "gto", "tli",
                      "escape", "deorbit", "lunar")
    _PERFILES_VALIDOS = ("bcb", "directo")
    _INTEGRADORES_VALIDOS = ("hibrido", "rk45")
    _DINAMICAS_VALIDAS = ("variable", "congelada")
    _EVENTOS_VALIDOS = ("integrador", "paso")
    _EFEMERIDES_LUNAR_VALIDAS = ("spice", "horizons_verlet", "rebound")
    _DEORBIT_ACTIVACIONES = ("sitio", "orbita", "t")
    _MARCOS_VISTA = ("tierra", "baricentro")
    _CUERPOS_PERTURBADORES = ("sol", "luna")    # los únicos con efeméride

    def _validar_config(self) -> None:
        """(11.21.1) Lanza ValueError con TODOS los problemas encontrados, uno
        por línea, antes de crear la simulación. Solo rechaza lo que no tiene
        sentido físico o haría fallar la corrida más adelante con un error
        confuso (NaN, división por cero, un valor de texto inexistente);
        no impone límites «de buen gusto». Los valores por defecto y los
        presets pasan sin cambios."""
        errores: list[str] = []

        def exigir(cond: bool, texto: str) -> None:
            if not cond:
                errores.append(texto)

        def fin(x) -> bool:
            return isinstance(x, (int, float)) and math.isfinite(x)

        def num(nombre: str) -> float:
            return getattr(self, nombre)

        # Texto
        exigir(self.modo_mision in self._MODOS_VALIDOS,
               f"modo_mision={self.modo_mision!r}: debe ser uno de {self._MODOS_VALIDOS}")
        exigir(self.perfil_tripulada in self._PERFILES_VALIDOS,
               f"perfil_tripulada={self.perfil_tripulada!r}: debe ser uno de {self._PERFILES_VALIDOS}")
        exigir(self.integrador in self._INTEGRADORES_VALIDOS,
               f"integrador={self.integrador!r}: debe ser uno de {self._INTEGRADORES_VALIDOS}")
        exigir(self.dinamica in self._DINAMICAS_VALIDAS,
               f"dinamica={self.dinamica!r}: debe ser uno de {self._DINAMICAS_VALIDAS}")
        exigir(self.eventos in self._EVENTOS_VALIDOS,
               f"eventos={self.eventos!r}: debe ser uno de {self._EVENTOS_VALIDOS}")
        exigir(self.deorbit_activacion in self._DEORBIT_ACTIVACIONES,
               f"deorbit_activacion={self.deorbit_activacion!r}: debe ser uno de {self._DEORBIT_ACTIVACIONES}")
        exigir(self.marco_vista in self._MARCOS_VISTA,
               f"marco_vista={self.marco_vista!r}: debe ser uno de {self._MARCOS_VISTA}")

        # Ningún número puede ser NaN (APOGEO_OBJETIVO sí puede ser +inf: escape)
        for nombre in ("umbral_encendido", "VEL_MIN_SES", "ALTURA_SEGURIDAD",
                       "APOGEO_OBJETIVO", "THROTTLE_MARGEN", "THROTTLE_MIN",
                       "alt_inicio_giro", "angulo_kick", "KARMAN",
                       "TOL_APOGEO_SUBORBITAL", "GTO_RP_FINAL", "ECC_MAX_GTO",
                       "ALT_RK4_MIN", "RK4_PASO_MAX", "DT_COAST_ALTA_ENERGIA",
                       "T_MAX_ALTA_ENERGIA", "T_MAX_SUBORBITAL", "deorbit_t_activacion",
                       "DEORBIT_SITIO_TOL_KM", "DEORBIT_SITIO_HORIZONTE",
                       "DEORBIT_PLAN_PRESUPUESTO_S",
                       "DEORBIT_RP_OBJETIVO", "DEORBIT_ALT_REENTRADA",
                       "G_MAX_TRIPULADA", "ECC_MAX_TRIPULADA_BCB",
                       "ECC_MAX_TRIPULADA_DIRECTO", "latitud_lanzamiento",
                       "azimut_lanzamiento"):
            v = num(nombre)
            exigir(isinstance(v, (int, float)) and not math.isnan(v),
                   f"{nombre}={v!r}: debe ser un número (no NaN)")
        # Rangos
        exigir(fin(self.angulo_kick) and 0.0 <= self.angulo_kick < 90.0,
               f"angulo_kick={self.angulo_kick!r}: debe estar en [0, 90) grados")
        exigir(fin(self.latitud_lanzamiento) and -90.0 <= self.latitud_lanzamiento <= 90.0,
               f"latitud_lanzamiento={self.latitud_lanzamiento!r}: debe estar en [-90, 90]")
        exigir(fin(self.azimut_lanzamiento) and 0.0 <= self.azimut_lanzamiento <= 180.0,
               f"azimut_lanzamiento={self.azimut_lanzamiento!r}: debe estar en [0, 180] "
               f"(el modelo plano solo vuela hacia el Este)")
        exigir(isinstance(self.LUNAR_PASOS_MIN_COAST, int) and self.LUNAR_PASOS_MIN_COAST >= 1,
               f"LUNAR_PASOS_MIN_COAST={self.LUNAR_PASOS_MIN_COAST!r}: debe ser un entero >= 1")
        exigir(isinstance(self.sim_speed, int) and not isinstance(self.sim_speed, bool)
               and 1 <= self.sim_speed <= 10_000,
               f"sim_speed={self.sim_speed!r}: debe ser un entero en [1, 10000]")
        exigir(fin(self.THROTTLE_MIN) and 0.0 < self.THROTTLE_MIN <= 1.0,
               f"THROTTLE_MIN={self.THROTTLE_MIN!r}: debe estar en (0, 1]")
        for nombre in ("THROTTLE_MARGEN", "RK4_PASO_MAX", "DT_COAST_ALTA_ENERGIA",
                       "T_MAX_ALTA_ENERGIA", "T_MAX_SUBORBITAL", "DEORBIT_SITIO_TOL_KM",
                       "DEORBIT_SITIO_HORIZONTE", "G_MAX_TRIPULADA", "KARMAN",
                       "DEORBIT_PLAN_PRESUPUESTO_S"):
            v = num(nombre)
            exigir(fin(v) and v > 0.0, f"{nombre}={v!r}: debe ser > 0 y finito")
        for nombre in ("umbral_encendido", "VEL_MIN_SES", "ALTURA_SEGURIDAD",
                       "alt_inicio_giro", "TOL_APOGEO_SUBORBITAL",
                       "deorbit_t_activacion", "DEORBIT_RP_OBJETIVO",
                       "DEORBIT_ALT_REENTRADA", "ALT_RK4_MIN"):
            v = num(nombre)
            exigir(fin(v) and v >= 0.0, f"{nombre}={v!r}: debe ser >= 0 y finito")
        exigir(self.APOGEO_OBJETIVO > 0.0,
               f"APOGEO_OBJETIVO={self.APOGEO_OBJETIVO!r}: debe ser > 0 (o inf en escape)")
        if self.modo_mision != "escape":
            exigir(fin(self.APOGEO_OBJETIVO),
                   f"APOGEO_OBJETIVO={self.APOGEO_OBJETIVO!r}: debe ser finito; "
                   "solo el modo 'escape' admite inf")
        exigir(fin(self.GTO_RP_FINAL) and self.GTO_RP_FINAL > 0.0,
               f"GTO_RP_FINAL={self.GTO_RP_FINAL!r}: debe ser > 0 y finito")
        for nombre in ("ECC_MAX_GTO", "ECC_MAX_TRIPULADA_BCB",
                       "ECC_MAX_TRIPULADA_DIRECTO"):
            v = num(nombre)
            exigir(fin(v) and 0.0 < v < 1.0, f"{nombre}={v!r}: debe estar en (0, 1)")
        exigir(isinstance(self.DEORBIT_SITIO_MAX_ITER, int)
               and self.DEORBIT_SITIO_MAX_ITER >= 1,
               f"DEORBIT_SITIO_MAX_ITER={self.DEORBIT_SITIO_MAX_ITER!r}: debe ser un entero >= 1")
        # Modo lunar (12.3.0)
        exigir(self.efemerides_lunar in self._EFEMERIDES_LUNAR_VALIDAS,
               f"efemerides_lunar={self.efemerides_lunar!r}: debe ser uno de "
               f"{self._EFEMERIDES_LUNAR_VALIDAS}")
        lon = self.longitud_lanzamiento_ecliptica
        exigir(fin(lon) and 0.0 <= lon <= 360.0,
               f"longitud_lanzamiento_ecliptica={lon!r}: debe estar en [0, 360] grados")
        if fin(lon) and self.modo_mision != "lunar":
            exigir(lon == 90.0,
                   f"longitud_lanzamiento_ecliptica={lon!r}: solo el modo "
                   f"'lunar' la usa; en {self.modo_mision!r} debe ser 90 (valor "
                   f"histórico, resultados idénticos a las versiones anteriores)")
        for nombre in ("ECC_MAX_APARCAMIENTO", "DT_COAST_LUNAR", "T_MAX_LUNAR",
                       "LUNAR_APOGEO_TLI", "LUNAR_DIST_OBJETIVO",
                       "LUNAR_VENTANA_PERIODOS", "LUNAR_PASO_VENTANA_S",
                       "LUNAR_HORIZONTE_TRANSFERENCIA", "LUNAR_Z_MAX_FISICO",
                       "LUNAR_EXTENSION_PERISELENE_S", "LUNAR_ORBITA_ALT",
                       "LUNAR_PERIODOS_VERIF", "LUNAR_ORBITA_TOL_ALT"):
            v = num(nombre)
            exigir(fin(v) and v > 0.0, f"{nombre}={v!r}: debe ser > 0 y finito")
        exigir(fin(self.ECC_MAX_APARCAMIENTO) and self.ECC_MAX_APARCAMIENTO < 1.0,
               f"ECC_MAX_APARCAMIENTO={self.ECC_MAX_APARCAMIENTO!r}: debe estar en (0, 1)")
        exigir(fin(self.LUNAR_ORBITA_ECC_MAX) and 0.0 < self.LUNAR_ORBITA_ECC_MAX < 1.0,
               f"LUNAR_ORBITA_ECC_MAX={self.LUNAR_ORBITA_ECC_MAX!r}: debe estar en (0, 1)")
        exigir(fin(self.LUNAR_MARGEN_DV) and self.LUNAR_MARGEN_DV >= 1.0,
               f"LUNAR_MARGEN_DV={self.LUNAR_MARGEN_DV!r}: debe ser >= 1")
        exigir(isinstance(self.LUNAR_LOI, bool), f"LUNAR_LOI={self.LUNAR_LOI!r}: debe ser bool")
        if self.LUNAR_LOI:
            exigir(fin(self.LUNAR_ORBITA_ALT) and self.LUNAR_ORBITA_ALT >= 10_000.0,
                   f"LUNAR_ORBITA_ALT={self.LUNAR_ORBITA_ALT!r}: con LOI debe ser >= 10 km")
        if self.modo_mision == "lunar":
            exigir(self.usar_terceros_cuerpos and "luna" in self.cuerpos_perturbadores,
                   "modo 'lunar': exige usar_terceros_cuerpos=True y 'luna' en "
                   "cuerpos_perturbadores (la Luna es la que se persigue)")
            exigir(not self.usar_marco_movil,
                   "modo 'lunar': usar_marco_movil debe ser False (el encuentro "
                   "se propaga en inercial geocéntrico con efemérides, no en el "
                   "CRTBP de tli/escape)")
        # Fecha y cuerpos (solo importan si se usan efemérides)
        try:
            datetime.fromisoformat(self.fecha_lanzamiento)
        except (TypeError, ValueError):
            errores.append(f"fecha_lanzamiento={self.fecha_lanzamiento!r}: "
                           f"debe ser ISO 8601 (p. ej. 2026-01-01T00:00:00)")
        malos = [c for c in self.cuerpos_perturbadores
                 if c not in self._CUERPOS_PERTURBADORES]
        exigir(not malos,
               f"cuerpos_perturbadores={tuple(self.cuerpos_perturbadores)!r}: "
               f"{malos} no soportado(s); solo {self._CUERPOS_PERTURBADORES}")
        # (12.8.4) coherencia del objetivo con el perfil: solo configuraciones editadas a mano
        # (todos los presets las cumplen); escape (apogeo inf) no entra en ninguna rama
        if self.modo_mision == "suborbital":
            exigir(fin(self.APOGEO_OBJETIVO) and self.APOGEO_OBJETIVO >= self.KARMAN,
                   f"APOGEO_OBJETIVO={self.APOGEO_OBJETIVO!r}: en suborbital debe ser >= "
                   f"KARMAN={self.KARMAN!r}; si no, la misión es imposible")
        if self.modo_mision == "gto":
            exigir(fin(self.APOGEO_OBJETIVO) and self.GTO_RP_FINAL <= self.APOGEO_OBJETIVO,
                   f"GTO_RP_FINAL={self.GTO_RP_FINAL!r}: no puede superar APOGEO_OBJETIVO="
                   f"{self.APOGEO_OBJETIVO!r} (el periapsis final estaría por encima del apogeo)")
        if self.modo_mision in ("satelital", "tripulada", "gto", "deorbit"):
            exigir(fin(self.APOGEO_OBJETIVO) and self.APOGEO_OBJETIVO > self.ALTURA_SEGURIDAD,
                   f"APOGEO_OBJETIVO={self.APOGEO_OBJETIVO!r}: debe superar "
                   f"ALTURA_SEGURIDAD={self.ALTURA_SEGURIDAD!r}; el periapsis nunca alcanzaría "
                   f"la altura de seguridad y el motor quemaría hasta agotarse")
        if self.modo_mision == "lunar":
            horizonte_plan = (self.LUNAR_HORIZONTE_TRANSFERENCIA
                              + self.LUNAR_EXTENSION_PERISELENE_S)
            exigir(fin(self.T_MAX_LUNAR) and self.T_MAX_LUNAR > horizonte_plan,
                   f"T_MAX_LUNAR={self.T_MAX_LUNAR!r}: debe superar "
                   f"LUNAR_HORIZONTE_TRANSFERENCIA + LUNAR_EXTENSION_PERISELENE_S "
                   f"({horizonte_plan:.0f} s) para completar la búsqueda del periselene")
            if self.efemerides_lunar in ("horizons_verlet", "rebound"):
                exigir(fin(self.T_MAX_LUNAR) and self.T_MAX_LUNAR <= 30.0 * 86400.0,
                       f"T_MAX_LUNAR={self.T_MAX_LUNAR!r}: Horizons+Verlet / REBOUND preintegran todo "
                       "el horizonte antes del despegue; el máximo admitido es 30 días. "
                       "Para horizontes mayores use SPICE.")
            if self.LUNAR_LOI:
                exigir(fin(self.LUNAR_ORBITA_ALT)
                       and (R_LUNA + self.LUNAR_ORBITA_ALT) < R_SOI_LUNA,
                       f"LUNAR_ORBITA_ALT={self.LUNAR_ORBITA_ALT!r}: la órbita LOI debe caber "
                       f"dentro de la SOI lunar (altitud < {R_SOI_LUNA - R_LUNA:.0f} m)")
        if errores:
            raise ValueError("Configuración inválida:\n  - " + "\n  - ".join(errores))

    def aplicar_preset(self) -> None:
        """
        Ajusta los parámetros operativos a los valores típicos del
        modo elegido. El usuario puede sobreescribirlos después en la
        GUI; esto solo los pre-carga.
        """
        # El preset también fija usar_marco_movil: solo tli/escape lo
        # necesitan (la nave cruza la órbita lunar); el resto, apagado.
        self.usar_marco_movil = self.modo_mision in ("tli", "escape")
        if self.modo_mision == "deorbit":
            # La órbita previa a la desórbita: LEO 200 x 420 km. (La
            # especificación pedía ALTURA_SEGURIDAD = APOGEO = 0, pero
            # con APOGEO_OBJETIVO = 0 el corte por apogeo dispara en el
            # despegue y nunca se llega a órbita; el "sin periapsis
            # seguro" de la desórbita se logra desactivando el corrector
            # de altitud crítica durante la fase deorbit.)
            self.ALTURA_SEGURIDAD = 200_000.0
            self.APOGEO_OBJETIVO  = 420_000.0
            self.angulo_kick      = 3.0
            self.VEL_MIN_SES      = 2_500.0
            self.umbral_encendido = 10.0
            self.deorbit_activacion = "sitio"
        elif self.modo_mision in ("tli", "escape"):
            self.ALTURA_SEGURIDAD = 200_000.0
            # APOGEO_OBJETIVO es una ALTITUD en todo el código (se compara
            # con elementos["ra"] y entra en eps = -mu/(rp_radio + R_T +
            # APOGEO_OBJETIVO)). La órbita lunar está a 384 400 km del
            # CENTRO de la Tierra, así que la altitud es esa distancia
            # menos R_T (antes se usaba 384 400 km tal cual: ~6 371 km de
            # más, mezclando radio y altitud).
            self.APOGEO_OBJETIVO  = (384_400_000.0 - R_T
                                     if self.modo_mision == "tli"
                                     else float("inf"))
            self.angulo_kick      = 3.0
            self.VEL_MIN_SES      = 2_500.0
            self.umbral_encendido = 10.0
        elif self.modo_mision == "lunar":
            # Encuentro con la Luna (12.3.0): B-C-B a una órbita de
            # aparcamiento casi circular (rp >= 210 km: por encima de
            # ALT_RK4_MIN, así el coast usa siempre RK4), espera de ventana,
            # quema TLI y vuelo translunar hasta la SOI. Terceros cuerpos
            # (Sol y Luna) activos: son la dinámica del encuentro.
            self.ALTURA_SEGURIDAD = 210_000.0
            self.APOGEO_OBJETIVO  = 250_000.0
            self.angulo_kick      = 3.0
            self.VEL_MIN_SES      = 2_500.0
            self.umbral_encendido = 10.0
            self.usar_terceros_cuerpos = True
            self.cuerpos_perturbadores = ("sol", "luna")
        elif self.modo_mision == "gto":
            # Transferencia a GEO: apogeo a 35 786 km, circularización
            # en el apogeo (periapsis final 35 000 km, ecc < 0.02)
            self.ALTURA_SEGURIDAD = 200_000.0
            self.APOGEO_OBJETIVO  = 35_786_000.0
            self.GTO_RP_FINAL     = 35_000_000.0
            self.angulo_kick      = 3.0
            self.VEL_MIN_SES      = 2_500.0
            self.umbral_encendido = 10.0
        elif self.modo_mision == "suborbital":
            # Cohete sonda: sin periapsis seguro, ascenso casi vertical
            self.ALTURA_SEGURIDAD = 0.0
            self.APOGEO_OBJETIVO  = 150_000.0
            self.angulo_kick      = 1.5
            self.VEL_MIN_SES      = 2_500.0
            self.umbral_encendido = 10.0
        elif self.modo_mision == "tripulada":
            # LEO tripulada (referencia ISS): por debajo de Van Allen
            self.ALTURA_SEGURIDAD = 200_000.0
            self.APOGEO_OBJETIVO  = 420_000.0
            self.angulo_kick      = 3.0
            self.VEL_MIN_SES      = 2_700.0
            self.umbral_encendido = 0.5
        else:
            # LEO satelital alta (referencia clásica del simulador)
            self.ALTURA_SEGURIDAD = 120_000.0
            self.APOGEO_OBJETIVO  = 1_300_000.0
            self.angulo_kick      = 3.0
            self.VEL_MIN_SES      = 2_500.0
            self.umbral_encendido = 10.0


@dataclass(frozen=True)
class EtapaConfig:
    nombre: str
    m_estr: float
    m_comb: float
    u0:     float
    D:      float
    Cd:     float
    r:      float
    # Área de salida de las toberas [m²]: u0·D es el empuje de VACÍO y a
    # presión ambiente p el empuje es u0·D − A_sal·p. Primera: 9 Merlin 1D,
    # ΔT(nivel del mar) = 69 kN por motor -> A = 69e3/101325 = 0,68 m²
    # por motor (6,1 m² en total; ~7,5 % menos empuje a nivel del mar).
    # Segunda: Merlin Vacuum, tobera de ~3,3 m de diámetro (8,55 m²).
    A_sal:  float = 0.0


ETAPAS_CONFIG = (
    EtapaConfig("Primera", 22200.0, 411000.0, 3049.0, 2772.0, 0.3, 1.85, 6.1),
    # D de la 2ª etapa: empuje de vacío del Merlin Vacuum (981 kN, guía de
    # usuario de Falcon 9 de SpaceX) / u0 (3412 m/s) = 287,5 kg/s. Antes
    # 900 kg/s daba 3,07 MN (3 veces el real). La masa de propelente
    # (107,5 t) no se tocó: la duración a flujo máximo sale 374 s (publicado
    # ~397 s con el motor regulado).
    EtapaConfig("Segunda",  4000.0, 107500.0, 3412.0, 287.5, 0.3, 1.85, 8.55),
)


@_njit(cache=True, nogil=True)
def _rk4_kernel(px, py, vx, vy, dt, n, empuje_x, empuje_y, arrastre_x, arrastre_y,
                masa, perturbacion_x, perturbacion_y, mu):
    """n subpasos RK4 de tamaño h (mismas ecuaciones que `derivadas_marco_tierra`,
    con las fuerzas externas constantes). Compilado con numba si existe."""
    hm = 0.5 * dt
    ax0 = (empuje_x + arrastre_x) / masa + perturbacion_x
    ay0 = (empuje_y + arrastre_y) / masa + perturbacion_y
    for _ in range(n):
        r = math.sqrt((px) * (px) + (py) * (py)); g = mu / (r * r)
        k1x, k1y = vx, vy
        k1vx = ax0 - g * px / r; k1vy = ay0 - g * py / r

        qx = px + hm * k1x; qy = py + hm * k1y
        r = math.sqrt((qx) * (qx) + (qy) * (qy)); g = mu / (r * r)
        k2x = vx + hm * k1vx; k2y = vy + hm * k1vy
        k2vx = ax0 - g * qx / r; k2vy = ay0 - g * qy / r

        qx = px + hm * k2x; qy = py + hm * k2y
        r = math.sqrt((qx) * (qx) + (qy) * (qy)); g = mu / (r * r)
        k3x = vx + hm * k2vx; k3y = vy + hm * k2vy
        k3vx = ax0 - g * qx / r; k3vy = ay0 - g * qy / r

        qx = px + dt * k3x; qy = py + dt * k3y
        r = math.sqrt((qx) * (qx) + (qy) * (qy)); g = mu / (r * r)
        k4x = vx + dt * k3vx; k4y = vy + dt * k3vy
        k4vx = ax0 - g * qx / r; k4vy = ay0 - g * qy / r

        c = dt / 6.0
        px += c * (k1x + 2.0 * k2x + 2.0 * k3x + k4x)
        py += c * (k1y + 2.0 * k2y + 2.0 * k3y + k4y)
        vx += c * (k1vx + 2.0 * k2vx + 2.0 * k3vx + k4vx)
        vy += c * (k1vy + 2.0 * k2vy + 2.0 * k3vy + k4vy)
    return px, py, vx, vy


# ============================================================
# MECÁNICA ORBITAL — UNA SOLA IMPLEMENTACIÓN
# ============================================================
_ELEM_CACHE: list = [None, None]    # Caché global de 1 estado; función pura, no mejora concurrencia (puede thrash)


def elementos_orbitales(px: float, py: float, vx: float, vy: float) -> dict:
    """Con caché de un elemento: varias funciones piden los elementos
    del mismo estado en un paso. El dict devuelto es compartido: no mutar."""
    clave = (px, py, vx, vy)
    if _ELEM_CACHE[0] == clave:
        return _ELEM_CACHE[1]
    res = _elementos_orbitales(px, py, vx, vy)
    _ELEM_CACHE[0] = clave
    _ELEM_CACHE[1] = res
    return res


def _elementos_orbitales(px: float, py: float, vx: float, vy: float) -> dict:
    """
    Calcula los elementos orbitales keplerianos a partir del estado
    instantáneo. Devuelve:
        ecc, eps            : excentricidad y energía específica
        rp, ra              : periapsis/apoapsis como ALTITUD sobre la superficie (m)
        rp_radio, ra_radio  : mismas distancias desde el centro (m)
        a, b, c, omega      : parámetros de la elipse (NaN si no es cerrada)
    Notación: `omega` (ω) es la LONGITUD del periapsis en 2D (≡ argumento del periapsis porque Ω=0);
    `h` es el M.A.R.E. = |r×v| (en 2D h ≡ h_z). `_ELEM_PLANETAS` usa en cambio la convención de
    Standish (a, e, i, L, ϖ, Ω) con ϖ = Ω + ω: no son intercambiables directamente.
        es_elipse           : True si la trayectoria es una elipse cerrada
    """
    r  = math.hypot(px, py)
    if r < 1.0:
        return _elementos_vacios(r)
    v2 = vx * vx + vy * vy
    mu = MU_T
    eps = 0.5 * v2 - mu / r
    h  = px * vy - py * vx

    rv = px * vx + py * vy
    ex = ((v2 - mu / r) * px - rv * vx) / mu
    ey = ((v2 - mu / r) * py - rv * vy) / mu
    ecc = math.hypot(ex, ey)
    omega = math.atan2(ey, ex) if ecc > 1e-12 else 0.0

    es_elipse = eps < 0.0 and ecc < 1.0

    if es_elipse:
        a = -mu / (2.0 * eps)
        b = a * math.sqrt(max(0.0, 1.0 - ecc * ecc))
        c = a * ecc
        rp_radio = a * (1.0 - ecc)
        ra_radio = a * (1.0 + ecc)
    else:
        a = b = c = float("nan")
        if abs(h) > 1e-6 and ecc > 1e-12:
            p = (h * h) / mu
            rp_radio = p / (1.0 + ecc)
        else:
            rp_radio = r
        ra_radio = float("inf")

    return {
        "ecc": ecc,
        "eps": eps,
        "rp": rp_radio - R_T,
        "ra": (ra_radio - R_T) if math.isfinite(ra_radio) else float("inf"),
        "rp_radio": rp_radio,
        "ra_radio": ra_radio,
        "a": a, "b": b, "c": c, "omega": omega,
        "es_elipse": es_elipse,
    }


def _elementos_vacios(r: float) -> dict:
    return {
        "ecc": 0.0, "eps": 0.0,
        "rp": r - R_T, "ra": r - R_T,
        "rp_radio": r, "ra_radio": r,
        "a": float("nan"), "b": float("nan"),
        "c": float("nan"), "omega": 0.0,
        "es_elipse": False,
    }


def trayectoria_orbital(xs_orb: dict, n_puntos: int = 200):
    """
    Dada la salida de `elementos_orbitales`, devuelve los puntos
    (x, y) de la elipse completa en el plano inercial, más los
    puntos de periapsis y apoapsis. Devuelve None si no es elipse.
    """
    if not xs_orb["es_elipse"]:
        return None
    a, b, c, omega = xs_orb["a"], xs_orb["b"], xs_orb["c"], xs_orb["omega"]
    theta = np.linspace(0.0, 2.0 * np.pi, n_puntos)
    x_local = a * np.cos(theta) - c
    y_local = b * np.sin(theta)
    cos_o, sin_o = np.cos(omega), np.sin(omega)
    xs = x_local * cos_o - y_local * sin_o
    ys = x_local * sin_o + y_local * cos_o

    rp, ra = xs_orb["rp_radio"], xs_orb["ra_radio"]
    pe = (rp * np.cos(omega), rp * np.sin(omega))
    ap = (-ra * np.cos(omega), -ra * np.sin(omega))
    return xs, ys, pe, ap


_Z_CAPAS = np.array([
    0.0, 11000.0, 25000.0, 30000.0, 40000.0, 50000.0, 60000.0, 70000.0, 80000.0, 90000.0, 
    100000.0, 110000.0, 120000.0, 130000.0, 140000.0, 150000.0, 180000.0, 200000.0, 
    250000.0, 300000.0, 350000.0, 400000.0, 450000.0, 500000.0, 600000.0, 700000.0, 
    800000.0, 900000.0, 1000000.0
])

_RHO_BASE = np.array([
    1.225, 0.3639, 4.008e-2, 1.841e-2, 3.996e-3, 1.027e-3, 3.097e-4, 8.283e-5, 
    1.846e-5, 3.416e-6, 5.606e-7, 9.708e-8, 2.222e-8, 8.152e-9, 3.831e-9, 
    2.076e-9, 5.194e-10, 2.541e-10, 6.073e-11, 1.916e-11, 7.014e-12, 
    2.803e-12, 1.184e-12, 5.215e-13, 1.137e-13, 3.070e-14, 1.136e-14, 
    5.759e-15, 3.561e-15
])

# Alturas de escala reales del USSA76 (en metros) corregidas para consistencia matemática
# Escalas de altura derivadas de _Z_CAPAS/_RHO_BASE con
# H[i] = -(Z[i+1]-Z[i]) / ln(RHO[i+1]/RHO[i]), para que el modelo
# exponencial por tramos sea EXACTAMENTE continuo en cada frontera
# de capa (una tabla de escalas "a ojo" como la que había antes
# dejaba saltos de hasta ~60% en varias fronteras — ver densidad
# predicha vs. tabulada en z=110,120,180,900,1000 km, por ejemplo).
_H_ESCALA = np.diff(_Z_CAPAS) / np.log(_RHO_BASE[:-1] / _RHO_BASE[1:])

# Listas de Python espejo de las tablas: bisect/math.exp sobre floats
# puros evitan crear escalares numpy en cada llamada (~1/paso en ascenso).
_Z_CAPAS_L = _Z_CAPAS.tolist()
_RHO_BASE_L = _RHO_BASE.tolist()
_H_ESCALA_L = _H_ESCALA.tolist()
_N_TRAMOS = len(_H_ESCALA_L)


def densidad_atmosfera(z):
    if z < 0.0:
        return _RHO_BASE_L[0]
    if z >= 1000000.0:
        return 0.0
    # Índice de tramo: indexa _H_ESCALA (28 elementos, uno por tramo),
    # no _Z_CAPAS (29 fronteras); la cota evita un IndexError latente
    # si el umbral de arriba cambiara.
    i = bisect_right(_Z_CAPAS_L, z) - 1
    if i < 0:
        i = 0
    elif i > _N_TRAMOS - 1:
        i = _N_TRAMOS - 1
    return _RHO_BASE_L[i] * math.exp(-(z - _Z_CAPAS_L[i]) / _H_ESCALA_L[i])


# Temperatura (atmósfera estándar USSA76, capas hasta 86 km): base de cada
# capa en altitud GEOPOTENCIAL [m], temperatura base [K] y gradiente [K/m].
# Hace falta porque las capas de densidad (escala de altura media) no dan
# la velocidad del sonido ni la presión en la troposfera: a 10 km darían a
# +18 % y p +37 %. Por encima de 86 km se mantiene T constante (la
# densidad ya es despreciable). Aire seco: R = 287,053 J/(kg·K), γ = 1,4.
_R0_GEOPOT = 6356766.0
_H_GP_BASE = [0.0, 11000.0, 20000.0, 32000.0, 47000.0, 51000.0, 71000.0, 84852.0]
_T_BASE = [288.15, 216.65, 216.65, 228.65, 270.65, 270.65, 214.65, 186.946]
_L_GRAD = [-0.0065, 0.0, 0.001, 0.0028, 0.0, -0.0028, -0.002, 0.0]
_R_AIRE = 287.053
_GAMMA_AIRE = 1.4


def temperatura_atmosfera(z: float) -> float:
    """Temperatura [K] a la altitud geométrica z [m] (USSA76 hasta 86 km)."""
    if z < 0.0:
        z = 0.0
    hg = _R0_GEOPOT * z / (_R0_GEOPOT + z)
    i = bisect_right(_H_GP_BASE, hg) - 1
    return _T_BASE[i] + _L_GRAD[i] * (hg - _H_GP_BASE[i])


def velocidad_sonido(z: float) -> float:
    """a = sqrt(γ·R·T) [m/s]."""
    return math.sqrt(_GAMMA_AIRE * _R_AIRE * temperatura_atmosfera(z))


def presion_atmosfera(z: float) -> float:
    """p = ρ·R·T [Pa], con la ρ de este modelo (consistente con el arrastre)."""
    return densidad_atmosfera(z) * _R_AIRE * temperatura_atmosfera(z)


# Factor sobre el Cd subsónico de la etapa: divergencia de arrastre
# transónica (pico cerca de M ≈ 1,05) y decaimiento supersónico. Forma
# genérica de un cuerpo esbelto (Cd 0,3 -> ~0,57 en el pico -> ~0,22 a M 5),
# no la curva medida del Falcon 9. Interpolación lineal.
_MACH_PUNTOS = [0.0, 0.8, 0.95, 1.05, 1.2, 1.5, 2.0, 3.0, 5.0, 10.0]
_MACH_FACTOR = [1.0, 1.0, 1.5, 1.9, 1.8, 1.5, 1.25, 1.0, 0.75, 0.6]


def factor_cd_mach(mach: float) -> float:
    if mach <= _MACH_PUNTOS[0]:
        return _MACH_FACTOR[0]
    if mach >= _MACH_PUNTOS[-1]:
        return _MACH_FACTOR[-1]
    i = bisect_right(_MACH_PUNTOS, mach) - 1
    m0, m1 = _MACH_PUNTOS[i], _MACH_PUNTOS[i + 1]
    f0, f1 = _MACH_FACTOR[i], _MACH_FACTOR[i + 1]
    return f0 + (f1 - f0) * (mach - m0) / (m1 - m0)


def fuerza_arrastre(alt: float, vx: float, vy: float,
                    Cd: float, area: float, mach: bool = False):
    rho = densidad_atmosfera(alt)
    v2 = vx * vx + vy * vy
    v = math.sqrt(v2)
    if v == 0.0 or area == 0.0:
        return 0.0, 0.0, 0.0
    if mach and rho > 0.0:
        Cd = Cd * factor_cd_mach(v / velocidad_sonido(alt))
    arrastre_mag = 0.5 * rho * v2 * Cd * area
    return (-arrastre_mag * vx / v, -arrastre_mag * vy / v, arrastre_mag)



# ============================================================
# CAPA 2 — FÍSICA PURA (sin estado, sin config, sin efemérides)
# ============================================================
def velocidad_relativa_al_aire(px: float, py: float, vx: float, vy: float,
                               Omega: float) -> Vec2:
    """Velocidad respecto al aire, que co-rota con la Tierra:
    v_aire = Ω_ef × r = Ω_ef·(py, -px) (sentido horario, como el vuelo)."""
    return vx - Omega * py, vy + Omega * px


def aceleracion_de_marea(mu_perturbador: float, pos_nave: Vec2,
                         pos_perturbador: Vec2) -> Vec2:
    """Aceleración de marea (diferencial) de un tercer cuerpo:
        a = mu * [ (r_i - r)/|r_i - r|^3 - r_i/|r_i|^3 ]
    con r_i la posición del perturbador relativa al primario. Devuelve
    (0, 0) en el caso degenerado (división por cero)."""
    rix, riy = pos_perturbador
    ri = math.hypot(rix, riy)
    dx, dy = rix - pos_nave[0], riy - pos_nave[1]
    d = math.hypot(dx, dy)
    if d < 1.0 or ri < 1.0:
        return 0.0, 0.0
    return (mu_perturbador * (dx / d ** 3 - rix / ri ** 3),
            mu_perturbador * (dy / d ** 3 - riy / ri ** 3))


def derivadas_marco_tierra(t, y, empuje_x, empuje_y, arrastre_x, arrastre_y,
                           masa, perturbacion_x=0.0, perturbacion_y=0.0):
    """Marco inercial centrado en la Tierra: gravedad central + empuje y
    arrastre (constantes en el paso) + marea de terceros cuerpos."""
    px, py, vx, vy = y
    r = math.hypot(px, py)
    g = MU_T / (r * r)
    return [
        vx, vy,
        (empuje_x + arrastre_x) / masa - g * px / r + perturbacion_x,
        (empuje_y + arrastre_y) / masa - g * py / r + perturbacion_y,
    ]


class LeyV12:
    """(12.0.0) Parámetros de UN tramo de la dinámica 'variable': lo que
    permanece constante en el tramo (throttle, modo de guiado, marea) y lo
    que se evalúa con el estado dentro del ODE (todo lo demás)."""
    __slots__ = ("Omega", "u0_flujo", "flujo", "a_sal", "presion", "empuje_on",
                 "modo", "mantener", "cd", "area", "mach", "pert_x", "pert_y")

    def __init__(self, Omega, u0, flujo, a_sal, presion, empuje_on, modo,
                 mantener, cd, area, mach, pert_x, pert_y):
        self.Omega = Omega
        self.flujo = flujo                  # kg/s (D·throttle)
        self.u0_flujo = u0 * flujo          # empuje de vacío [N]
        self.a_sal = a_sal
        self.presion = presion              # True: restar A_sal·p_amb
        self.empuje_on = empuje_on
        self.modo = modo
        self.mantener = mantener            # guiado 'mantener altura' (BCB/deorbit)
        self.cd, self.area, self.mach = cd, area, mach
        self.pert_x, self.pert_y = pert_x, pert_y

    def apagada(self) -> "LeyV12":
        """Mismo tramo con el motor apagado (cola tras el corte)."""
        return LeyV12(self.Omega, 0.0, 0.0, self.a_sal, self.presion, False,
                      self.modo, self.mantener, self.cd, self.area, self.mach,
                      self.pert_x, self.pert_y)

    def fuerzas_congeladas(self, y):
        """Tupla de `derivadas_marco_tierra` para el coast libre (RK4): arrastre
        evaluado en `y`, sin empuje."""
        px, py, vx, vy, m = y[0], y[1], y[2], y[3], y[4]
        alt = math.hypot(px, py) - R_T
        vax, vay = velocidad_relativa_al_aire(px, py, vx, vy, self.Omega)
        ax_, ay_, _ = fuerza_arrastre(alt, vax, vay, self.cd, self.area, self.mach)
        return (0.0, 0.0, ax_, ay_, m, self.pert_x, self.pert_y)


def derivadas_v12(t, y, ley):
    """Marco Tierra con masa variable: y = (px, py, vx, vy, m, dv_prop).
    Gravedad, arrastre, empuje y su dirección se calculan con y."""
    px, py, vx, vy, m, _ = y
    if not math.isfinite(m) or m <= 0.0:
        raise ValueError(f"derivadas_v12: masa no válida m={m!r} kg; "
                         "el estado debe conservar masa positiva")
    r = math.hypot(px, py)
    if not math.isfinite(r) or r <= 0.0:
        raise ValueError(f"derivadas_v12: radio geocéntrico no válido r={r!r} m")
    g = MU_T / (r * r)
    ax = -g * px / r + ley.pert_x
    ay = -g * py / r + ley.pert_y
    alt = r - R_T
    vax, vay = velocidad_relativa_al_aire(px, py, vx, vy, ley.Omega)
    dx, dy, _ = fuerza_arrastre(alt, vax, vay, ley.cd, ley.area, ley.mach)
    ax += dx / m
    ay += dy / m
    dm = 0.0
    ddv = 0.0
    if ley.empuje_on:
        T = ley.u0_flujo
        if ley.presion:
            T -= ley.a_sal * presion_atmosfera(alt)
        if T > 0.0:
            # (12.0.1) solo se quema propelente si hay empuje neto: con la
            # tobera "bloqueada" por la presión (T <= 0) no se consume nada.
            dm = -ley.flujo
            if ley.modo == "gravity_turn":
                vgx, vgy, _, _ = velocidad_de_guiado(px, py, vx, vy, alt, ley.Omega)
                ux, uy = calcular_guiado(px, py, vgx, vgy, math.hypot(vgx, vgy),
                                         ley.modo)
            elif ley.modo == "circularizacion" and ley.mantener:
                ux, uy = direccion_empuje_mantener_altura(px, py, vx, vy, T, m)
            else:
                ux, uy = calcular_guiado(px, py, vx, vy, math.hypot(vx, vy),
                                         ley.modo)
            ax += T * ux / m
            ay += T * uy / m
            ddv = T / m
    return [vx, vy, ax, ay, dm, ddv]


def derivadas_crtbp_rotante(t, y, Omega, a, fuerza_x=0.0, fuerza_y=0.0, masa=1.0):
    """
    CRTBP dimensional, marco rotante centrado en el baricentro
    Tierra-Luna (Tierra y Luna fijas en (x1,0) y (x2,0)).
    Validado aparte por conservación de la constante de Jacobi
    antes de integrarlo aquí.
    """
    x, yv, vx, vy = y
    x1, y1 = -RAZON_MASA_TL * a, 0.0
    x2, y2 = (1.0 - RAZON_MASA_TL) * a, 0.0
    r1 = math.hypot(x - x1, yv - y1)
    r2 = math.hypot(x - x2, yv - y2)
    ax = (2 * Omega * vy + Omega ** 2 * x
          - MU_T * (x - x1) / r1 ** 3 - MU_LUNA * (x - x2) / r2 ** 3
          + fuerza_x / masa)
    ay = (-2 * Omega * vx + Omega ** 2 * yv
          - MU_T * (yv - y1) / r1 ** 3 - MU_LUNA * (yv - y2) / r2 ** 3
          + fuerza_y / masa)
    return [vx, vy, ax, ay]


def derivadas_marco_luna(t, y, perturbacion_x=0.0, perturbacion_y=0.0,
                         fuerza_x=0.0, fuerza_y=0.0, masa=1.0):
    """Inercial centrado en la Luna; Tierra/Sol entran como marea."""
    px, py, vx, vy = y
    r = math.hypot(px, py)
    g = MU_LUNA / (r * r)
    return [
        vx, vy,
        fuerza_x / masa - g * px / r + perturbacion_x,
        fuerza_y / masa - g * py / r + perturbacion_y,
    ]


# Distancia Tierra-nave a partir de la cual se pasa de marco Tierra a crucero.
UMBRAL_TIERRA_CRUCERO = _A_LUNA_REF - R_SOI_LUNA


# ============================================================
# GUIADO
# ============================================================
_SIN_5_GRADOS = math.sin(math.radians(5.0))


def _clip(x, a, b):
    """np.clip para escalares de Python (~10x más rápido). Propaga NaN
    igual que np.clip."""
    return a if x < a else (b if x > b else x)


def calcular_guiado(px: float, py: float, vx: float, vy: float,
                    speed: float, modo: str):
    r = math.hypot(px, py)
    if r < 1e-6:
        return 0.0, 1.0
    rx, ry = px / r, py / r

    if modo == "vertical":
        return rx, ry

    if modo == "gravity_turn":
        if speed < 5.0:
            return rx, ry
        tx, ty = vx / speed, vy / speed
        gamma = tx * rx + ty * ry
        if gamma <= 0.0:
            # La velocidad apunta hacia adentro u horizontal: se reconstruye
            # una dirección unitaria con componente radial EXACTA sen 5°
            # (sumar sen 5° a un vector que apunta hacia abajo no lo
            # invertía) y la misma parte tangencial que la velocidad. Sin
            # parte tangencial (caída vertical) se usa el sentido del este:
            # (ry, -rx), el mismo del kick y de la rotación terrestre
            # (para la base en +y, el este es +x; es giro horario).
            gamma_min = _SIN_5_GRADOS
            px_t, py_t = tx - gamma * rx, ty - gamma * ry
            n_t = math.hypot(px_t, py_t)
            if n_t < 1e-9:
                px_t, py_t, n_t = ry, -rx, 1.0
            c = math.sqrt(1.0 - gamma_min * gamma_min)
            tx = c * px_t / n_t + gamma_min * rx
            ty = c * py_t / n_t + gamma_min * ry
        return tx, ty

    if modo in ("prograde", "circularizacion"):
        if speed < 1.0:
            return rx, ry
        return vx / speed, vy / speed

    if modo == "retrograde":
        if speed < 1.0:
            return rx, ry
        return -vx / speed, -vy / speed

    if modo == "prograde_seguro":
        if speed < 1.0:
            return rx, ry
        tx = 0.5 * (vx / speed) + 0.5 * rx
        ty = 0.5 * (vy / speed) + 0.5 * ry
        n = math.hypot(tx, ty)
        if n < 1e-12:
            return rx, ry
        return tx / n, ty / n

    return rx, ry



def velocidad_de_guiado(px: float, py: float, vx: float, vy: float,
                        altitud: float, Omega: float
                        ) -> tuple[float, float, float, float]:
    """Velocidad para el giro por gravedad: respecto al aire en la
    atmósfera (así se guía un cohete real), inercial en el vacío.
    Relativa completa hasta 100 km y se desvanece linealmente hasta
    0 a 200 km, para que el rumbo no dé saltos. Devuelve (vgx, vgy,
    vaire_x, vaire_y): v_guia = v - v_aire·f(altitud)."""
    f = float(_clip(2.0 - altitud / 100_000.0, 0.0, 1.0))
    aire_x, aire_y = f * Omega * py, -f * Omega * px
    return vx - aire_x, vy - aire_y, aire_x, aire_y


def direccion_empuje_mantener_altura(px: float, py: float, vx: float, vy: float,
                                     empuje: float, masa: float) -> Vec2:
    """
    Guiado de la 2ª quema de tripulada/bcb. La quema está
    limitada por g, así que dura minutos y no es impulsiva: con
    empuje prógrado puro (alineado con la velocidad) la nave, ya
    en descenso desde el apogeo, apunta el empuje hacia la
    Tierra, cae a ~200 km y termina la quema con v ~ 10 km/s y
    apogeo de decenas de miles de km (ecc ~ 0.79).

    Aquí el empuje es horizontal local más una componente radial
    que (a) compensa el déficit de sustentación orbital
    g - v_t²/r y (b) amortigua la velocidad radial (τ = 20 s):
    la nave sube su v_t manteniendo la altura del apogeo. La
    inclinación se acota a ±45° del horizonte local.
    """
    r = math.hypot(px, py)
    rx, ry = px / r, py / r
    momento_angular = px * vy - py * vx
    sentido = 1.0 if momento_angular >= 0.0 else -1.0
    tx, ty = -sentido * ry, sentido * rx    # horizontal local, sentido del vuelo
    v_tangencial = vx * tx + vy * ty
    v_radial = vx * rx + vy * ry
    a_empuje = empuje / max(masa, 1.0)
    if a_empuje < 1e-6:
        return tx, ty
    g_local = MU_T / (r * r)
    a_radial_requerida = (g_local - v_tangencial * v_tangencial / r) - v_radial / 20.0
    seno = float(_clip(a_radial_requerida / a_empuje, -0.7071, 0.7071))
    coseno = math.sqrt(1.0 - seno * seno)
    return coseno * tx + seno * rx, coseno * ty + seno * ry


# ============================================================
# ETAPAS
# ============================================================
class Etapa:
    def __init__(self, cfg: EtapaConfig):
        self.nombre = cfg.nombre
        self.m_estr = cfg.m_estr
        self.m_comb = cfg.m_comb
        self.u0 = cfg.u0
        self.D  = cfg.D
        self.Cd = cfg.Cd
        self.r  = cfg.r
        self.A_sal = cfg.A_sal

        self.m_comb_restante = cfg.m_comb
        self.activa = False
        self.separada = False
        self.burnout_registrado = False
        self.datos_burnout: Optional[dict] = None
        self.v_inicio = 0.0
        # Δv propulsivo acumulado [m/s] = ∫ (T/m) dt mientras empuja (11.21.2).
        # NO es la variación de rapidez: esa incluye gravedad y arrastre.
        self.dv_prop = 0.0
        # Flag histórico: True desde la primera vez que esta etapa
        # produjo empuje, y ya no se vuelve a poner en False. Se usa
        # en _verificar_burnout en vez de `activa`, porque
        # _actualizar_mision corre ANTES que _verificar_burnout en
        # avanzar_paso: si el combustible llega a 0 en el mismo paso
        # en que _actualizar_mision decide apagar el motor (p. ej. al
        # cortar la fase "circularizacion" por falta de combustible),
        # `activa` ya es False cuando _verificar_burnout mira el
        # estado, y el burnout nunca quedaría registrado.
        self._fue_activada = False

    def masa_actual(self) -> float:
        return self.m_estr + self.m_comb_restante

    def tiene_combustible(self) -> bool:
        return self.m_comb_restante > 0.0

    def acumular_dv(self, empuje: float, masa: float, flujo: float,
                    dt: float) -> None:
        """Suma T/m·dt al Δv propulsivo, con la masa media del tramo
        (la del inicio menos la mitad de lo quemado: error O(dt²))."""
        if dt <= 0.0 or empuje <= 0.0:
            return
        m_media = masa - 0.5 * flujo * dt
        if m_media > 0.0:
            self.dv_prop += empuje / m_media * dt

    def consumir(self, flujo_efectivo: float, dt: float) -> None:
        self.m_comb_restante = max(0.0, self.m_comb_restante - flujo_efectivo * dt)
        if self.m_comb_restante < 1e-9:      # residuo de redondeo: agotada
            self.m_comb_restante = 0.0

    def perdida_por_presion(self, p_amb: float) -> float:
        """Empuje que se pierde por la presión ambiente [N]: A_sal·p."""
        return self.A_sal * p_amb

    def empuje_y_flujo(self, throttle: float, p_amb: float = 0.0):
        """Devuelve (empuje_N, flujo_masico_kg/s) en una sola llamada.
        T = ṁ·u_e + A_e(p_e − p_a) = T_vacío − A_e·p_a, con T_vacío = u0·D·throttle
        (el término de p_e ya está dentro de u0): independiente de cómo se
        reparta. Con p_amb = 0 es exactamente u0·D·throttle."""
        return max(0.0, self.u0 * self.D * throttle - self.A_sal * p_amb), self.D * throttle



# ============================================================
# MODO LUNAR (12.3.0): núcleos de planificación del encuentro
# ============================================================
class EfemeridesNoValidas(RuntimeError):
    """El modo lunar solo corre con efemérides válidas: SPICE real (pymcel) o
    un proveedor determinista inyectado. Un fallo de SPICE NO cae al respaldo
    Kepler aquí: detiene la misión con este error y un diagnóstico."""


@_njit(cache=True)
def _herm_pos(tab, t0, dt, t):
    """Posición (x, y) de una tabla (N, 4) = (x, y, vx, vy) en nodos cada `h`
    desde `t0`, por Hermite cúbico (la misma interpolación de la caché pymcel)."""
    k = int(math.floor((t - t0) / dt))
    if k < 0:
        k = 0
    if k > tab.shape[0] - 2:
        k = tab.shape[0] - 2
    s = (t - t0) / dt - k
    h00, h10, h01, h11, _d00, _d10, _d01, _d11 = _hermite_coefs(s)
    x = h00 * tab[k, 0] + h10 * dt * tab[k, 2] + h01 * tab[k + 1, 0] + h11 * dt * tab[k + 1, 2]
    y = h00 * tab[k, 1] + h10 * dt * tab[k, 3] + h01 * tab[k + 1, 1] + h11 * dt * tab[k + 1, 3]
    return x, y


@_njit(cache=True)
def _herm_vel(tab, t0, dt, t):
    """Velocidad (vx, vy): derivada de la misma interpolación de Hermite."""
    k = int(math.floor((t - t0) / dt))
    if k < 0:
        k = 0
    if k > tab.shape[0] - 2:
        k = tab.shape[0] - 2
    s = (t - t0) / dt - k
    _h00, _h10, _h01, _h11, d00, d10, d01, d11 = _hermite_coefs(s)
    d00 = d00 / dt
    d01 = d01 / dt
    vx = d00 * tab[k, 0] + d10 * tab[k, 2] + d01 * tab[k + 1, 0] + d11 * tab[k + 1, 2]
    vy = d00 * tab[k, 1] + d10 * tab[k, 3] + d01 * tab[k + 1, 1] + d11 * tab[k + 1, 3]
    return vx, vy


@_njit(cache=True)
def _herm_pos3d(tab, t0, dt, t, offset):
    """Posición XYZ de una tabla [x,y,z,vx,vy,vz] mediante Hermite cúbico."""
    k = int(math.floor((t - t0) / dt))
    if k < 0:
        k = 0
    if k > tab.shape[0] - 2:
        k = tab.shape[0] - 2
    s = (t - t0) / dt - k
    h00, h10, h01, h11, _d00, _d10, _d01, _d11 = _hermite_coefs(s)
    x = h00 * tab[k, offset] + h10 * dt * tab[k, offset + 3] + h01 * tab[k + 1, offset] + h11 * dt * tab[k + 1, offset + 3]
    y = h00 * tab[k, offset + 1] + h10 * dt * tab[k, offset + 4] + h01 * tab[k + 1, offset + 1] + h11 * dt * tab[k + 1, offset + 4]
    z = h00 * tab[k, offset + 2] + h10 * dt * tab[k, offset + 5] + h01 * tab[k + 1, offset + 2] + h11 * dt * tab[k + 1, offset + 5]
    return x, y, z


@_njit(cache=True)
def _herm_vel3d(tab, t0, dt, t, offset):
    """Velocidad XYZ: derivada temporal de la misma interpolación Hermite."""
    k = int(math.floor((t - t0) / dt))
    if k < 0:
        k = 0
    if k > tab.shape[0] - 2:
        k = tab.shape[0] - 2
    s = (t - t0) / dt - k
    _h00, _h10, _h01, _h11, d00, d10, d01, d11 = _hermite_coefs(s)
    d00 /= dt
    d01 /= dt
    vx = d00 * tab[k, offset] + d10 * tab[k, offset + 3] + d01 * tab[k + 1, offset] + d11 * tab[k + 1, offset + 3]
    vy = d00 * tab[k, offset + 1] + d10 * tab[k, offset + 4] + d01 * tab[k + 1, offset + 1] + d11 * tab[k + 1, offset + 4]
    vz = d00 * tab[k, offset + 2] + d10 * tab[k, offset + 5] + d01 * tab[k + 1, offset + 2] + d11 * tab[k + 1, offset + 5]
    return vx, vy, vz


@_njit(cache=True, nogil=True)
def _acel_geo(x, y, t, luna, sol, t0, ht, mu_t, mu_l, mu_s, usa_sol):
    """Gravedad terrestre + marea diferencial de la Luna (y del Sol): la
    misma dinámica del marco Tierra de la simulación."""
    r = math.sqrt((x) * (x) + (y) * (y))
    g = mu_t / (r * r * r)
    ax = -g * x
    ay = -g * y
    lx, ly = _herm_pos(luna, t0, ht, t)
    dx = lx - x
    dy = ly - y
    d = math.sqrt((dx) * (dx) + (dy) * (dy))
    rl = math.sqrt((lx) * (lx) + (ly) * (ly))
    ax += mu_l * (dx / (d * d * d) - lx / (rl * rl * rl))
    ay += mu_l * (dy / (d * d * d) - ly / (rl * rl * rl))
    if usa_sol:
        sx, sy = _herm_pos(sol, t0, ht, t)
        dx = sx - x
        dy = sy - y
        d = math.sqrt((dx) * (dx) + (dy) * (dy))
        rs = math.sqrt((sx) * (sx) + (sy) * (sy))
        ax += mu_s * (dx / (d * d * d) - sx / (rs * rs * rs))
        ay += mu_s * (dy / (d * d * d) - sy / (rs * rs * rs))
    return ax, ay


@_njit(cache=True, nogil=True)
def _k_coast(x, y, vx, vy, t, t_fin, dt, luna, sol, t0, ht, mu_t, mu_l, mu_s,
             usa_sol, r_soi, vigilar):
    """RK4 sin empuje de t a t_fin. Con `vigilar` sigue la distancia a la
    Luna y termina poco después del máximo acercamiento tras entrar a la SOI.
    Devuelve (x, y, vx, vy, t, d_min, t_dmin, t_entrada o -1, hz_ent, eps_ent):
    hz = momento angular específico (z) respecto de la Luna y eps la energía
    específica selenocéntrica (μ_L = mu_l) en el primer paso dentro de la SOI."""
    d_min = 1.0e30
    t_dmin = t
    t_ent = -1.0
    hz_ent = 0.0
    eps_ent = 0.0
    while t < t_fin - 1.0e-9:
        hh = dt if dt < t_fin - t else t_fin - t
        hm = 0.5 * hh
        a1x, a1y = _acel_geo(x, y, t, luna, sol, t0, ht, mu_t, mu_l, mu_s, usa_sol)
        qx = x + hm * vx
        qy = y + hm * vy
        k2vx = vx + hm * a1x
        k2vy = vy + hm * a1y
        a2x, a2y = _acel_geo(qx, qy, t + hm, luna, sol, t0, ht, mu_t, mu_l, mu_s, usa_sol)
        qx = x + hm * k2vx
        qy = y + hm * k2vy
        k3vx = vx + hm * a2x
        k3vy = vy + hm * a2y
        a3x, a3y = _acel_geo(qx, qy, t + hm, luna, sol, t0, ht, mu_t, mu_l, mu_s, usa_sol)
        qx = x + hh * k3vx
        qy = y + hh * k3vy
        k4vx = vx + hh * a3x
        k4vy = vy + hh * a3y
        a4x, a4y = _acel_geo(qx, qy, t + hh, luna, sol, t0, ht, mu_t, mu_l, mu_s, usa_sol)
        nx = x + hh / 6.0 * (vx + 2.0 * k2vx + 2.0 * k3vx + k4vx)
        ny = y + hh / 6.0 * (vy + 2.0 * k2vy + 2.0 * k3vy + k4vy)
        nvx = vx + hh / 6.0 * (a1x + 2.0 * a2x + 2.0 * a3x + a4x)
        nvy = vy + hh / 6.0 * (a1y + 2.0 * a2y + 2.0 * a3y + a4y)
        x, y, vx, vy = nx, ny, nvx, nvy
        t += hh
        if vigilar:
            lx, ly = _herm_pos(luna, t0, ht, t)
            d = math.sqrt((lx - x) * (lx - x) + (ly - y) * (ly - y))
            if d < d_min:
                d_min = d
                t_dmin = t
            if d < r_soi and t_ent < 0.0:
                t_ent = t
                lvx, lvy = _herm_vel(luna, t0, ht, t)
                rx = x - lx
                ry = y - ly
                wx = vx - lvx
                wy = vy - lvy
                hz_ent = rx * wy - ry * wx
                eps_ent = 0.5 * (wx * wx + wy * wy) - mu_l / d
            if t_ent >= 0.0 and d > d_min + 2.0e6:
                break
    return x, y, vx, vy, t, d_min, t_dmin, t_ent, hz_ent, eps_ent


LOI_TOL_EPS_REL = 1.0e-8     # tolerancia relativa del corte de la LOI sobre ε_obj
LOI_MAX_ITER_CORTE = 12      # iteraciones máximas del corte (secante + bisección)


def _conica_selenocentrica(rx, ry, wx, wy, mu_l):
    """(12.6.1) Cónica selenocéntrica osculadora de la posición (rx, ry) y la
    velocidad (wx, wy) relativas a la Luna: ε, h, ecc, rp, ra, a, periodo, r y
    velocidad radial (ra, a y T son inf si no está ligada)."""
    r = math.hypot(rx, ry)
    if not (all(math.isfinite(v) for v in (rx, ry, wx, wy, mu_l)) and r > 0.0 and mu_l > 0.0):
        raise ValueError(f"estado selenocéntrico no válido (r={r!r}, μ={mu_l!r}, "
                         f"w=({wx!r}, {wy!r}))")
    eps = 0.5 * (wx * wx + wy * wy) - mu_l / r
    h = abs(rx * wy - ry * wx)
    ecc = math.sqrt(max(0.0, 1.0 + 2.0 * eps * h * h / mu_l ** 2))
    rp = h * h / (mu_l * (1.0 + ecc))
    if eps < 0.0:
        a = -mu_l / (2.0 * eps)
        ra = a * (1.0 + ecc)
        T = 2.0 * math.pi * math.sqrt(a ** 3 / mu_l)
    else:
        a, ra, T = float("inf"), float("inf"), float("inf")
    return {"eps": eps, "h": h, "ecc": ecc, "rp": rp, "ra": ra, "a": a, "T": T,
            "r": r, "vr": (rx * wx + ry * wy) / r}


def _coast_hasta_periselene(x, y, vx, vy, t, horizonte, dt, prm, extension):
    """Propaga hasta el horizonte y, si entró a SOI sin pasar el periselene,
    extiende hasta `extension`. Devuelve los 10 valores de `_k_coast` más un
    booleano `periselene_alcanzado`, inferido de que el mínimo no quedó en el
    extremo final de la propagación (si quedó en el borde, no se da por probado)."""
    a = _k_coast(x, y, vx, vy, t, t + horizonte, dt, *prm, R_SOI_LUNA, True)
    xa, ya, vxa, vya, ta, d_min, t_dmin, t_ent, hz, eps = a
    x_fin, y_fin, vx_fin, vy_fin, t_fin_prop = xa, ya, vxa, vya, ta
    if t_ent > 0.0 and ta >= t + horizonte - 1.0e-6:
        b = _k_coast(xa, ya, vxa, vya, ta, ta + extension, dt, *prm, R_SOI_LUNA, True)
        xb, yb, vxb, vyb, tb, d_min_b, t_dmin_b, _, _, _ = b
        if d_min_b < d_min:
            d_min, t_dmin = d_min_b, t_dmin_b
        x_fin, y_fin, vx_fin, vy_fin, t_fin_prop = xb, yb, vxb, vyb, tb
    # Un mínimo tabulado anterior al último paso no basta: se comprueba que
    # la velocidad radial selenocéntrica al final sea positiva (ya se aleja).
    luna, _sol, t0, ht = prm[:4]
    lx, ly = _herm_pos(luna, t0, ht, t_fin_prop)
    lvx, lvy = _herm_vel(luna, t0, ht, t_fin_prop)
    dx, dy = x_fin - lx, y_fin - ly
    d_fin = math.hypot(dx, dy)
    vr_fin = ((dx * (vx_fin - lvx) + dy * (vy_fin - lvy)) / d_fin
              if d_fin > 0.0 else -float("inf"))
    periselene_alcanzado = (t_ent >= 0.0 and math.isfinite(vr_fin) and vr_fin > 0.0)
    return xa, ya, vxa, vya, ta, d_min, t_dmin, t_ent, hz, eps, periselene_alcanzado


@_njit(cache=True, nogil=True)
def _paso_quema_comun(x, y, vx, vy, t, m, empuje, flujo, hh, luna, sol, t0, ht,
                      mu_t, mu_l, mu_s, usa_sol, seleno):
    """(12.6.1) Un paso RK4 con empuje constante (fuerza) y masa que decrece a
    `flujo`. Con `seleno=False` el empuje va EN el sentido de la velocidad
    (prógrado geocéntrico, TLI); con `seleno=True`, en sentido contrario a la
    velocidad RELATIVA a la Luna en cada subpaso (retrógrado selenocéntrico,
    LOI). Misma dinámica geocéntrica con mareas (`_acel_geo`) en ambos."""
    sg = -1.0 if seleno else 1.0
    hm = 0.5 * hh
    lvx, lvy = 0.0, 0.0
    if seleno:
        lvx, lvy = _herm_vel(luna, t0, ht, t)
    wx, wy = vx - lvx, vy - lvy
    sp = max(math.sqrt((wx) * (wx) + (wy) * (wy)), 1.0e-9)
    a1x, a1y = _acel_geo(x, y, t, luna, sol, t0, ht, mu_t, mu_l, mu_s, usa_sol)
    b1 = empuje / m / sp
    a1x += sg * (b1 * wx)
    a1y += sg * (b1 * wy)
    qx = x + hm * vx
    qy = y + hm * vy
    v2x = vx + hm * a1x
    v2y = vy + hm * a1y
    if seleno:
        lvx, lvy = _herm_vel(luna, t0, ht, t + hm)
    wx, wy = v2x - lvx, v2y - lvy
    sp = max(math.sqrt((wx) * (wx) + (wy) * (wy)), 1.0e-9)
    a2x, a2y = _acel_geo(qx, qy, t + hm, luna, sol, t0, ht, mu_t, mu_l, mu_s, usa_sol)
    b2 = empuje / (m - flujo * hm) / sp
    a2x += sg * (b2 * wx)
    a2y += sg * (b2 * wy)
    qx = x + hm * v2x
    qy = y + hm * v2y
    v3x = vx + hm * a2x
    v3y = vy + hm * a2y
    wx, wy = v3x - lvx, v3y - lvy
    sp = max(math.sqrt((wx) * (wx) + (wy) * (wy)), 1.0e-9)
    a3x, a3y = _acel_geo(qx, qy, t + hm, luna, sol, t0, ht, mu_t, mu_l, mu_s, usa_sol)
    b3 = empuje / (m - flujo * hm) / sp
    a3x += sg * (b3 * wx)
    a3y += sg * (b3 * wy)
    qx = x + hh * v3x
    qy = y + hh * v3y
    v4x = vx + hh * a3x
    v4y = vy + hh * a3y
    if seleno:
        lvx, lvy = _herm_vel(luna, t0, ht, t + hh)
    wx, wy = v4x - lvx, v4y - lvy
    sp = max(math.sqrt((wx) * (wx) + (wy) * (wy)), 1.0e-9)
    a4x, a4y = _acel_geo(qx, qy, t + hh, luna, sol, t0, ht, mu_t, mu_l, mu_s, usa_sol)
    b4 = empuje / (m - flujo * hh) / sp
    a4x += sg * (b4 * wx)
    a4y += sg * (b4 * wy)
    nx = x + hh / 6.0 * (vx + 2.0 * v2x + 2.0 * v3x + v4x)
    ny = y + hh / 6.0 * (vy + 2.0 * v2y + 2.0 * v3y + v4y)
    nvx = vx + hh / 6.0 * (a1x + 2.0 * a2x + 2.0 * a3x + a4x)
    nvy = vy + hh / 6.0 * (a1y + 2.0 * a2y + 2.0 * a3y + a4y)
    return nx, ny, nvx, nvy


@_njit(cache=True)
def _paso_quema(x, y, vx, vy, t, m, empuje, flujo, hh, luna, sol, t0, ht,
                mu_t, mu_l, mu_s, usa_sol):
    """Quema prógrada geocéntrica (TLI): ver `_paso_quema_comun`."""
    return _paso_quema_comun(x, y, vx, vy, t, m, empuje, flujo, hh, luna, sol, t0,
                             ht, mu_t, mu_l, mu_s, usa_sol, False)


@_njit(cache=True)
def _paso_quema_loi(x, y, vx, vy, t, m, empuje, flujo, hh, luna, sol, t0, ht,
                    mu_t, mu_l, mu_s, usa_sol):
    """Quema retrógrada respecto de la Luna (LOI): ver `_paso_quema_comun`."""
    return _paso_quema_comun(x, y, vx, vy, t, m, empuje, flujo, hh, luna, sol, t0,
                             ht, mu_t, mu_l, mu_s, usa_sol, True)


@_njit(cache=True)
def _eps_selenocentrica(x, y, vx, vy, t, luna, t0, ht, mu_l):
    """Energía específica respecto de la Luna (interpolada en `t`)."""
    lx, ly = _herm_pos(luna, t0, ht, t)
    lvx, lvy = _herm_vel(luna, t0, ht, t)
    return 0.5 * ((vx - lvx) ** 2 + (vy - lvy) ** 2) - mu_l / math.hypot(x - lx, y - ly)


@_njit(cache=True, nogil=True)
def _k_quema(x, y, vx, vy, t, m, u0, D, a_sal_p, m_seca, eps_obj, hh, tau,
             thr_min, luna, sol, t0, ht, mu_t, mu_l, mu_s, usa_sol):
    """Quema prógrada hasta eps >= eps_obj con la misma regulación de empuje
    que la simulación (rampa por Δv faltante, constante en cada paso hh) y
    paso final fraccionado. Devuelve (x, y, vx, vy, t, m, estado):
    0 = correcto, 1 = no convergió/no finito, 2 = sin combustible,
    3 = sin empuje neto, 4 = energía estancada durante 100 pasos."""
    if not math.isfinite(eps_obj):
        return x, y, vx, vy, t, m, 1         # (12.13.7) objetivo no finito: no es "sin empuje neto"
    n_estancado = 0
    for _ in range(200000):
        r = math.sqrt((x) * (x) + (y) * (y))
        sp = math.sqrt((vx) * (vx) + (vy) * (vy))
        eps = 0.5 * sp * sp - mu_t / r
        if eps >= eps_obj:
            return x, y, vx, vy, t, m, 0
        dv_falta = (eps_obj - eps) / max(sp, 1.0)
        a_pleno = u0 * D / m
        thr = dv_falta / (a_pleno * tau)
        if thr > 1.0:
            thr = 1.0
        if thr < thr_min:
            thr = thr_min
        empuje = max(0.0, u0 * D * thr - a_sal_p)
        if not (empuje > 0.0):
            return x, y, vx, vy, t, m, 3     # sin empuje neto: no consume masa
        flujo = D * thr
        dt = hh
        ultimo = False
        if m - flujo * dt < m_seca:
            # (12.9.1) paso final limitado por el propelente: se intenta con lo que queda
            if not (flujo > 0.0):
                return x, y, vx, vy, t, m, 2
            dt = (m - m_seca) / flujo
            if not (dt > 1e-9):
                return x, y, vx, vy, t, m, 2
            ultimo = True
        nx, ny, nvx, nvy = _paso_quema(x, y, vx, vy, t, m, empuje, flujo, dt, luna,
                                       sol, t0, ht, mu_t, mu_l, mu_s, usa_sol)
        if not (math.isfinite(nx) and math.isfinite(ny)
                and math.isfinite(nvx) and math.isfinite(nvy)):
            return x, y, vx, vy, t, m, 1
        eps_n = 0.5 * (nvx * nvx + nvy * nvy) - mu_t / math.sqrt((nx) * (nx) + (ny) * (ny))
        if eps_n >= eps_obj:
            hh_base = dt                      # (12.9.1) el corte se refina dentro del paso (acortado o no)
            den = eps_n - eps
            if not (den > 0.0):
                return x, y, vx, vy, t, m, 1
            f = (eps_obj - eps) / den
            for _i in range(4):
                if not (f > 1e-12):          # (12.8.1) f en (0, 1]; NaN también cae aquí
                    return x, y, vx, vy, t, m, 1
                if f > 1.0:
                    f = 1.0
                dt = f * hh_base
                nx, ny, nvx, nvy = _paso_quema(x, y, vx, vy, t, m, empuje, flujo, dt,
                                               luna, sol, t0, ht, mu_t, mu_l, mu_s,
                                               usa_sol)
                eps_n = 0.5 * (nvx * nvx + nvy * nvy) - mu_t / math.sqrt((nx) * (nx) + (ny) * (ny))
                den = eps_n - eps
                if not (den > 0.0):
                    return x, y, vx, vy, t, m, 1
                f *= (eps_obj - eps) / den
            # (12.8.1) el corte debe haber convergido de verdad (nominal: ~1e-14)
            tol_eps = max(1e-9 * abs(eps_obj), 1e-6)  # J/kg; admite ε_obj = 0
            if not (abs(eps_n - eps_obj) <= tol_eps):
                return x, y, vx, vy, t, m, 1
            return nx, ny, nvx, nvy, t + dt, m - flujo * dt, 0
        if ultimo:
            return x, y, vx, vy, t, m, 2     # se agotó el propelente sin llegar a la energía
        # Progreso numérico: por debajo de este umbral el incremento no se
        # distingue de forma robusta de la resolución de energía del estado.
        umbral_progreso = max(1e-12, abs(eps) * 1e-14)
        if eps_n - eps <= umbral_progreso:
            n_estancado += 1
        else:
            n_estancado = 0
        x, y, vx, vy = nx, ny, nvx, nvy
        t += dt
        m -= flujo * dt
        if n_estancado >= 100:
            return x, y, vx, vy, t, m, 4
    return x, y, vx, vy, t, m, 1


# ============================================================
# ESTADO Y REGISTRO
# ============================================================
@dataclass
class EstadoCohete:
    px: float = 0.0
    py: float = R_T
    vx: float = 0.0
    vy: float = 0.0
    altitud: float = 0.0
    speed: float = 0.0
    empuje: float = 0.0            # empuje actual [N]
    arrastre_mag: float = 0.0      # módulo de la fuerza de arrastre [N]
    t: float = 0.0
    dt: float = 0.25
    pausado: bool = False
    simulacion_terminada: bool = False
    fase: str = "vertical"
    modo_guiado: str = "vertical"
    etapa_actual: int = 0
    advertencia_seguridad: bool = False
    cutoff_por_apogeo: bool = False
    esperando_apogeo_fisico: bool = False
    segunda_quema_realizada: bool = False
    v_radial_prev: float = 0.0

    # --- Marco móvil (ver sección MARCO MÓVIL) ---
    marco: str = "tierra"          # "tierra" | "crucero" | "luna"
    crucero_a: float = 0.0         # distancia Tierra-Luna "congelada" al entrar en crucero
    crucero_Omega: float = 0.0     # velocidad angular del marco rotante
    crucero_theta0: float = 0.0    # ángulo de la línea Tierra-Luna al entrar en crucero
    crucero_t0: float = 0.0        # instante (est.t) de entrada en crucero


@dataclass
class RegistroEventos:
    apogeo_registrado: bool = False
    max_q_registrado: bool = False
    orbita_detectada: bool = False
    impacto_registrado: bool = False
    datos_apogeo: dict = field(default_factory=dict)
    datos_max_q: dict = field(default_factory=lambda: {"q": -np.inf})
    datos_orbita: dict = field(default_factory=dict)
    datos_impacto: dict = field(default_factory=dict)
    eventos_etapas: list = field(default_factory=list)
    # --- Modo suborbital ---
    alt_max: float = 0.0
    mision_cumplida: Optional[bool] = None   # None = aún sin evaluar
    # --- tli / escape (medidos en ECI) ---
    r_max_eci: float = 0.0           # altitud geocéntrica máxima alcanzada (m)
    d_luna_min: float = float("inf") # distancia mínima a la Luna (efeméride) (m)
    entro_soi_luna: bool = False
    datos_soi_luna: dict = field(default_factory=dict)
    impacto_lunar_registrado: bool = False
    datos_impacto_lunar: dict = field(default_factory=dict)
    datos_mision: dict = field(default_factory=dict)
    # Texto del fallo si el integrador numérico no convergió (la
    # simulación se detiene en el último estado válido); None si no hubo.
    fallo_integrador: Optional[str] = None


# ============================================================
# CAPA 4 — SIMULACIÓN (control del bucle + máquina de estados)
# ============================================================
# Este bloque NO contiene ecuaciones físicas nuevas: orquesta las
# funciones puras de las capas anteriores (derivadas, mareas,
# guiado, elementos orbitales, ...) y decide QUÉ pasa en cada paso
# (fases, cortes, separaciones). La física vive en funciones puras
# a nivel de módulo; la `Simulacion` solo guarda estado y secuencia
# llamadas.
_MODO_POR_FASE = {
    "vertical":            "vertical",
    "gravity_turn":        "gravity_turn",
    "coasting":            "prograde",
    "coasting_interetapa": "prograde",
    "circularizacion":     "circularizacion",
    "deorbit":             "retrograde",
    "impacto":             "vertical",
    "espera_tli":          "prograde",     # 12.3.0 (modo lunar)
    "quema_tli":           "prograde",
    "translunar":          "prograde",
    "quema_loi":           "retrograde",   # 12.6.1 (retrógrado RESPECTO DE LA LUNA)
    "orbita_lunar":        "prograde",
}

# Series del historial en vivo que se recortan juntas (poda / snapshot).
_SERIES_HISTORIAL_ARRAY = ("hist_t", "hist_alt", "hist_v",
                           "tray_wx", "tray_wy", "tray_ex", "tray_ey")
_SERIES_HISTORIAL_FASE = ("hist_fase", "tray_wfase")

# Constante de tiempo de la regulación de empuje por Δv faltante [s]:
# empuje pleno hasta quedar ~TAU s de quema y luego proporcional.
TAU_REGULACION_THROTTLE = 5.0


def _limpiar_almacen(estado: dict) -> None:
    """Cierra y borra el archivo temporal de un AlmacenHistorial (también como finalizador)."""
    f, ruta = estado.get("f"), estado.get("ruta")
    estado["f"], estado["ruta"] = None, None
    if f is not None:
        with contextlib.suppress(OSError, ValueError):
            f.close()
    if ruta:
        try:
            os.remove(ruta)
        except OSError:
            gc.collect()                  # (12.13.4) Windows: soltar un memmap aún vivo y reintentar
            with contextlib.suppress(OSError):
                os.remove(ruta)


class AlmacenHistorial:
    """
    (12.12.0) Historial podado de la RAM, guardado en un archivo temporal: registros de 8 float64
    (t, alt, v, wx, wy, ex, ey, código de fase). Se escribe en bloques de BLOQUE registros para no
    frenar el bucle; se lee de vuelta solo cuando hace falta (`leer`, `leer_rango`).

    `segmentos` = [[pos, a, b], ...]: los registros [a, b) del almacén iban en la posición `pos`
    del historial en RAM cuando se borraron (así `historial_completo` los empalma en su sitio).
    Si el disco falla, `ok` pasa a False (se avisa una vez) y los datos nuevos se descartan.
    """
    COLUMNAS = ("t", "alt", "v", "wx", "wy", "ex", "ey", "fase")
    BLOQUE = 5000

    def __init__(self, directorio: Optional[str] = None):
        self._directorio = directorio
        self._estado = {"f": None, "ruta": None}
        self._pend: list = []
        self._n_pend = 0
        self.n = 0                              # registros en total (escritos + pendientes)
        self.segmentos: list = []
        self._fase_cod: dict = {}
        self._fase_nom: list = []
        self.ok = True
        self._finalizador = weakref.finalize(self, _limpiar_almacen, self._estado)

    def __deepcopy__(self, memo):
        """(12.13.1) Un almacén NO se copia: tiene un archivo abierto (no copiable) y dos almacenes no
        deben escribir en el mismo archivo. La copia (p. ej. la que usa el planificador de desórbita)
        recibe un almacén nuevo y vacío, con su propio archivo temporal creado al primer vaciado y borrado
        al descartarse la copia; el original no se toca."""
        nuevo = AlmacenHistorial(self._directorio)
        memo[id(self)] = nuevo
        return nuevo

    @property
    def ruta(self) -> Optional[str]:
        return self._estado["ruta"]

    def _codigo(self, fase: str) -> int:
        cod = self._fase_cod.get(fase)
        if cod is None:
            cod = self._fase_cod[fase] = len(self._fase_nom)
            self._fase_nom.append(fase)
        return cod

    def nombres_fase(self, codigos) -> list:
        return [self._fase_nom[int(c)] for c in codigos]

    def anexar(self, pos: int, t, alt, v, wx, wy, ex, ey, fase) -> None:
        """Guarda un tramo (series de igual longitud; ex/ey ausentes se guardan como NaN)."""
        k = len(t)
        if k == 0 or not self.ok:
            return
        bloque = np.empty((k, 8), dtype="<f8")
        for j, serie in enumerate((t, alt, v, wx, wy, ex, ey)):
            if len(serie) == k:
                bloque[:, j] = np.asarray(serie, dtype="<f8")
            else:
                bloque[:, j] = np.nan
        bloque[:, 7] = [self._codigo(f) for f in fase]
        if self.segmentos and self.segmentos[-1][0] == pos and self.segmentos[-1][2] == self.n:
            self.segmentos[-1][2] += k
        else:
            self.segmentos.append([pos, self.n, self.n + k])
        self.n += k
        self._pend.append(bloque)
        self._n_pend += k
        if self._n_pend >= self.BLOQUE:
            self._vaciar()

    def _vaciar(self) -> None:
        if not self._pend or not self.ok:
            return
        try:
            if self._estado["f"] is None:
                fd, ruta = tempfile.mkstemp(prefix="cohete_hist_", suffix=".bin",
                                            dir=self._directorio)
                os.close(fd)
                self._estado["ruta"] = ruta
                self._estado["f"] = open(ruta, "ab")
            np.concatenate(self._pend).tofile(self._estado["f"])
            self._estado["f"].flush()
        except OSError as exc:
            self.ok = False
            print(f"⚠ No se pudo escribir el historial a disco ({exc}); lo podado se descarta "
                  "(el historial completo quedará incompleto).")
        self._pend, self._n_pend = [], 0

    def leer_rango(self, a: int, b: int) -> np.ndarray:
        """Registros [a, b) como array (b-a, 8) cargado a la RAM (vacío si el disco falló)."""
        if not self.ok or self.n == 0:
            return np.empty((0, 8))
        self._vaciar()
        if not self.ok or self._estado["ruta"] is None:
            return np.empty((0, 8))
        a, b = max(0, int(a)), min(self.n, int(b))
        if b <= a:
            return np.empty((0, 8))
        mapa = np.memmap(self._estado["ruta"], dtype="<f8", mode="r").reshape(-1, 8)
        try:
            return np.array(mapa[a:b])
        finally:
            del mapa

    def leer(self) -> np.ndarray:
        return self.leer_rango(0, self.n)

    def cerrar(self) -> None:
        """Cierra y borra el archivo temporal (idempotente)."""
        _limpiar_almacen(self._estado)
        self._pend, self._n_pend = [], 0
        self.n = 0
        self.segmentos = []


@dataclass(frozen=True)
class ContextoMision:
    """Fotografía inmutable que la máquina de estados consume en un paso.

    - v_radial: componente radial [m/s]; positiva = ascenso.
    - etapa: etapa actual, o None si ya no hay etapa seleccionable.
    - tiene_combustible: la etapa puede encender y conserva propelente.
    - en_peligro: por debajo de ALTURA_SEGURIDAD, descendiendo con margen
      significativo y con una etapa capaz de corregir (no aplica a suborbital,
      impacto ni a la fase activa de desórbita).
    - elementos: elementos keplerianos del estado; dict compartido, no mutar.
    - apogeo_alcanzado: apoapsis osculador finito >= APOGEO_OBJETIVO.
    - subiendo: v_radial >= 0.
    - paso_por_apogeo: v_radial previo >= 0 y actual < 0.
    """
    v_radial: float
    etapa: Optional[Etapa]
    tiene_combustible: bool
    en_peligro: bool
    elementos: dict
    apogeo_alcanzado: bool
    subiendo: bool
    paso_por_apogeo: bool


# Localización del corte por bisección (11.21.1). El criterio de parada es la
# TOLERANCIA TEMPORAL; el tope de iteraciones es solo una guarda.
#   _CORTE_TOL_T [s]: anchura final del intervalo [lo, hi]. 1e-7 s da una
#       incertidumbre de velocidad < a·1e-7 ≈ 1e-5 m/s con a ≈ 100 m/s²,
#       tres órdenes por debajo del error del propio integrador (RK45 con
#       rtol=1e-6 sobre ~8 km/s ≈ 8e-3 m/s). Con dt = 0.25 s se alcanza en
#       22 iteraciones.
#   _CORTE_MAX_ITER: 25 (antes 40). Con dt ≤ 3.3 s la tolerancia se cumple
#       antes del tope; con un dt mayor manda el tope y la anchura residual
#       es dt·2⁻²⁵ (< 2e-6 s para dt ≤ 60 s).
# (12.1.0) Desfase del evento de corte, en las unidades mezcladas de
# `_margen_corte` (m, m/s, J/kg): ~1e-8 s de retraso, mucho mayor que el ruido
# de redondeo de los elementos orbitales (~1e-9).
_MARGEN_CORTE_EPS: float = 1e-6
# (12.1.2) Pasos de la copia de prueba entre cesiones del planificador de
# desórbita (~50 ms de CPU): granularidad del reparto en la GUI.
_PLAN_PASOS_POR_TRAMO: int = 200
# Reintentos del evento de corte con margen creciente; agotados, bisección.
_CORTE_INTENTOS = 3
_CORTE_TOL_T: float = 1e-7
_CORTE_MAX_ITER: int = 25


class Simulacion:
    """Encapsula TODO el estado mutable de una corrida y el control del
    bucle. Las ecuaciones viven en funciones puras de capas anteriores."""

    def __init__(self, config: Config,
                 proveedor_efemerides: Optional[Callable[[str, float], tuple]] = None,
                 log: Optional[Callable[[str], None]] = None):
        config._validar_config()        # ValueError descriptivo antes de arrancar (11.21.1)
        self.config = config
        self._log: Callable[[str], None] = log if log is not None else print
        self._validacion_horizons: Optional[dict] = None
        # (12.3.0) Modo lunar. `proveedor_efemerides(cuerpo, t_sim) ->
        # ((x, y), (vx, vy))` (inercial eclíptico, relativo a la Tierra, SI,
        # consultable en cualquier t) sustituye a pymcel: es para pruebas
        # deterministas y NO es una validación física de SPICE.
        self._proveedor_efemerides = proveedor_efemerides
        if (config.modo_mision == "lunar"
                and config.efemerides_lunar in ("horizons_verlet", "rebound")
                and proveedor_efemerides is None):
            clase = (_ProveedorRebound if config.efemerides_lunar == "rebound"
                     else _ProveedorHorizonsVerlet)
            try:
                self._proveedor_efemerides = clase(
                    config.fecha_lanzamiento, config.T_MAX_LUNAR + 2 * 86400.0)
            except EfemeridesNoValidas:
                raise
            except (RuntimeError, OSError, ValueError) as e:
                raise EfemeridesNoValidas(
                    f"modo lunar: no se pudo preparar {config.efemerides_lunar} "
                    f"({type(e).__name__}: {e}); no se acepta respaldo Kepler") from e
        self._lunar: bool = config.modo_mision == "lunar"
        lam = math.radians(config.longitud_lanzamiento_ecliptica)
        self._lon_90: bool = config.longitud_lanzamiento_ecliptica == 90.0
        self._cos_lon, self._sin_lon = math.cos(lam), math.sin(lam)
        self._coast_prm = None
        self._eph_fallo: set = set()            # (12.8.4) fallos de SPICE de ESTA simulación
        self._loi: Optional[dict] = None        # 12.6.1: estado de la captura lunar
        self._lunar_estado: str = "ascenso"   # ascenso|planificando|espera_tli|quema_tli|translunar|quema_loi|orbita_lunar
        self._lunar_plan: Optional[dict] = None
        self._lunar_plan_pendiente: bool = False
        self._lunar_d_luna: float = float("inf")
        self._lunar_dv0: float = 0.0
        self._t_log_lunar: float = 0.0
        self.estado = EstadoCohete()
        # Ω efectiva (constante en toda la corrida: los campos de rotación
        # se leen de la GUI antes de crear la Simulacion y nada los cambia
        # en vivo). Si algún día la rotación se hace conmutable durante el
        # vuelo, hay que invalidar/recalcular esta caché.
        self._Omega_tierra = config.omega_efectiva()
        # Velocidad inicial por rotación: v = Ω·R_T·cos(lat)·sen(az),
        # tangente al Este. La plataforma está en (0, R_T), cuyo
        # sentido de vuelo (kick) es +x.
        self.estado.vx = self._Omega_tierra * R_T
        self.estado.speed = abs(self.estado.vx)
        self.etapas = [Etapa(c) for c in ETAPAS_CONFIG]
        # El dv de la 1ª etapa no debe contar la velocidad de plataforma.
        self.etapas[0].v_inicio = self.estado.speed
        self.etapas[0].activa = True
        self.eventos = RegistroEventos()
        self.objetos_caidos: list = []
        # Mensaje del último fallo de solve_ivp (lo escribe el integrador y
        # lo lee `_fallo_integrador`). Declarado aquí para que exista en
        # todos los caminos (11.21.1; antes se leía con getattr).
        self._msg_fallo_integrador: str = "sin detalle"
        # (12.1.0) Cuántos cortes/impactos resolvió cada mecanismo (diagnóstico)
        self.stats_eventos = {"corte_evento": 0, "corte_biseccion": 0,
                              "suelo_evento": 0, "transicion_marco": 0}

        self._iniciar_historial()

        # Gestión del historial orbital
        self._idx_inicio_orbita: Optional[int] = None
        self._periodo_orbital: Optional[float] = None
        self._orbita_podada: bool = False
        self._snapshot_orbita: Optional[dict] = None
        # (12.11.1) resumen de lo podado, solo para la gráfica de vuelo
        self._almacen = AlmacenHistorial()           # (12.12.0) lo podado va a disco
        self._dt_base: float = self.estado.dt
        self._deorbit_activo: bool = False
        self._deorbit_v_ref: float = 0.0
        self._deorbit_dv_prop_ref: float = 0.0
        # Modo "sitio": hora de quema planificada (None hasta que se detecta
        # la órbita). `_deorbit_sitio` se lee una vez, como `_omega_rotacion`.
        self._deorbit_sitio: bool = (config.modo_mision == "deorbit"
                                     and config.deorbit_activacion == "sitio")
        self._deorbit_t_plan: Optional[float] = None
        # Planificador en proceso aparte (lo instala la GUI; None = planificar
        # aquí mismo, en el hilo que llama: es el camino de siempre y el de
        # los scripts sin GUI). `esperando_plan` lo lee el HUD y el bucle.
        self._planificador_externo: Optional[PlanificadorDeorbitExterno] = None
        self.esperando_plan: bool = False
        # (12.1.2) Planificación por tramos en este hilo: None = completa de
        # una vez (scripts/regresión/hijo); la GUI fija segundos por fotograma.
        self.presupuesto_plan_s: Optional[float] = None
        self._plan_gen = None
        self._plan_t0: float = 0.0
        self._t_chk_luna: float = -1e18
        self._efemerides_verlet: Optional[_EphemerisVerlet] = None
        self._efemerides_modo: str = "sin_uso"
        self._init_efemerides()

    def __deepcopy__(self, memo):
        """Copia defensiva para predicciones: no hereda procesos, pipes,
        generadores ni callbacks de GUI. El historial define su propia copia
        vacía; el resto del estado físico se copia normalmente."""
        cls = self.__class__
        nuevo = cls.__new__(cls)
        memo[id(self)] = nuevo
        no_copiables = {"_planificador_externo", "_plan_gen", "_log"}
        for clave, valor in self.__dict__.items():
            if clave in no_copiables:
                setattr(nuevo, clave, print if clave == "_log" else None)
            elif clave in ("esperando_plan", "_lunar_plan_pendiente"):
                setattr(nuevo, clave, False)
            else:
                setattr(nuevo, clave, copy.deepcopy(valor, memo))
        return nuevo

    def _iniciar_historial(self) -> None:
        """Historial de floats en array('d'): 8 B por valor en vez de
        32 B (puntero + objeto float) con list. Mismo API que usa el
        resto del código (append, slicing, del a[k:i], len, np.asarray).
        Las fases siguen en list[str] (strings internados: solo el
        puntero, 8 B)."""
        self.hist_t = array("d")
        self.hist_alt = array("d")
        self.hist_v = array("d")
        self.hist_fase: list[str] = []
        self.tray_wx = array("d")
        self.tray_wy = array("d")
        self.tray_wfase: list[str] = []
        # Posición del baricentro T-L respecto a la Tierra (ECI) en cada
        # punto de tray_*: solo se llena con marco_vista="baricentro".
        self.tray_ex = array("d")
        self.tray_ey = array("d")
        # Único punto de reinicio del historial, incluidos los resúmenes podados.
        self.hist_resumen_t: list[float] = []
        self.hist_resumen_alt: list[float] = []
        self.hist_resumen_v: list[float] = []
        self._idx_resumen: Optional[int] = None
        self._resumen_paso_t: Optional[float] = None
        self._resumen_proximo_t: float = float("-inf")

    # ========================================================
    # EFEMÉRIDES DE SOL/LUNA
    # (ver sección TERCEROS CUERPOS arriba:
    #  pymcel/SPICE > Horizons+Verlet > Kepler simplificado)
    # ========================================================
    # --- Marco de lanzamiento (12.3.0, solo modo lunar) ---
    # La simulación corre con el sitio en +y y el vuelo hacia +x (giro
    # horario). Las efemérides son eclípticas, con la Luna girando en sentido
    # antihorario y el sitio a la longitud λ. El cambio de marco es la
    # isometría (reflexión + giro) u = Rot(-λ)·p ; (x', y') = (u_y, u_x): el
    # sitio (λ) pasa a +y, el sentido antihorario pasa a horario y el
    # vuelo "al este" (antihorario inercial) queda hacia +x. La física
    # (gravedad central, mareas, arrastre, guiado) es invariante bajo
    # isometrías, así que el cohete no cambia de código. Con λ = 90° se usa
    # el caso explícito (x', y') = (-x, y), sin cos(π/2) ≈ 6e-17.
    def _a_marco_lanzamiento(self, x: float, y: float) -> Vec2:
        if self._lon_90:
            return -x, y
        c, sn = self._cos_lon, self._sin_lon
        return -x * sn + y * c, x * c + y * sn

    def _desde_marco_lanzamiento(self, xp: float, yp: float) -> Vec2:
        if self._lon_90:
            return -xp, yp
        c, sn = self._cos_lon, self._sin_lon
        return yp * c - xp * sn, yp * sn + xp * c

    def estado_inicial_inercial(self) -> tuple[Vec2, Vec2]:
        """Posición y velocidad iniciales del sitio en el marco ECI eclíptico
        (el de las efemérides): r0 = R_T·(cos λ, sin λ), v0 = Ω·R_T·(-sin λ,
        cos λ) (tangente antihoraria, hacia el este). λ = 90° usa los valores
        exactos (0, R_T) y (-Ω·R_T, 0)."""
        v = self._Omega_tierra * R_T
        if self._lon_90:
            return (0.0, R_T), (-v, 0.0)
        c, sn = self._cos_lon, self._sin_lon
        return (R_T * c, R_T * sn), (-v * sn, v * c)

    def _estado_lunar_bruto(self, cuerpo: str, t_sim: float):
        """Estado inercial eclíptico (sin transformar) con la política
        ESTRICTA del modo lunar: nunca un respaldo Kepler en silencio."""
        _exigir_sol_o_luna(cuerpo)
        if self._proveedor_efemerides is not None:
            pos, vel = self._proveedor_efemerides(cuerpo, t_sim)
            return (float(pos[0]), float(pos[1])), (float(vel[0]), float(vel[1]))
        est_ = _estado_pymcel(cuerpo, self.config.fecha_lanzamiento, t_sim,
                              fallos=self._eph_fallo)
        if (cuerpo, self.config.fecha_lanzamiento) in self._eph_fallo:
            raise EfemeridesNoValidas(
                f"modo lunar: SPICE falló para {cuerpo!r} y no se acepta el "
                f"respaldo Kepler; misión detenida")
        return est_

    def _dist_objetivo_lunar(self) -> float:
        """Distancia objetivo del periselene al centro de la Luna [m]: con LOI,
        el radio de la órbita de captura; si no, `LUNAR_DIST_OBJETIVO`."""
        cfg = self.config
        return R_LUNA + cfg.LUNAR_ORBITA_ALT if cfg.LUNAR_LOI else cfg.LUNAR_DIST_OBJETIVO

    def _tabla_efemerides_lunar(self, t0: float, n: int):
        """Tablas (x, y, vx, vy) de Luna y Sol cada 600 s desde `t0` (n nodos)
        para `_k_coast`; devuelve el `prm` que esa función espera."""
        usa_sol = "sol" in self.config.cuerpos_perturbadores
        ht = 600.0
        luna = np.empty((n, 4))
        sol = np.ones((n, 4))
        for i in range(n):
            ti = t0 + i * ht
            (x, y), (vx, vy) = self._estado_cuerpo("luna", ti)
            luna[i] = (x, y, vx, vy)
            if usa_sol:
                (x, y), (vx, vy) = self._estado_cuerpo("sol", ti)
                sol[i] = (x, y, vx, vy)
        return (luna, sol, t0, ht, MU_T, float(CUERPOS["luna"].mu),
                float(MU_SOL) if usa_sol else 0.0, usa_sol)

    def _tabla_coast_lunar(self):
        """Tablas (x, y, vx, vy) de Luna y Sol cada 600 s para `_k_coast`, las
        mismas que usa el planificador, sobre todo el horizonte de la misión."""
        if self._coast_prm is None:
            n = int((self.config.T_MAX_LUNAR + 86400.0) / 600.0) + 3
            self._coast_prm = self._tabla_efemerides_lunar(0.0, n)
        return self._coast_prm

    def _paso_coast_lunar(self) -> None:
        """(12.5.10) Un paso balístico translunar con RK4 y Luna/Sol evaluados
        en cada subpaso (la misma dinámica del planificador)."""
        est = self.estado
        if est.simulacion_terminada or est.pausado:
            return
        prm = self._tabla_coast_lunar()
        dt = est.dt
        x, y, vx, vy, t = _k_coast(est.px, est.py, est.vx, est.vy, est.t, est.t + dt,
                                   dt, *prm, R_SOI_LUNA, False)[:5]
        if not all(math.isfinite(v) for v in (x, y, vx, vy)):
            self._msg_fallo_integrador = "RK4 translunar no finito"
            self._fallo_integrador("coast lunar")
            return
        est.empuje = 0.0
        est.arrastre_mag = 0.0
        est.px, est.py, est.vx, est.vy = x, y, vx, vy
        est.t += dt
        self._actualizar_altitud_y_rapidez()
        if est.altitud <= 0.0:
            self._registrar_impacto()

    def _z_luna(self, t_sim: float):
        """z de la Luna [m] si el proveedor la conoce; None si no (SPICE 2D,
        proveedores de prueba)."""
        f = getattr(self._proveedor_efemerides, "z", None)
        if f is None:
            return None
        try:
            return float(f("luna", t_sim))
        except Exception:
            return None

    def _estado_lunar(self, cuerpo: str, t_sim: float):
        (x, y), (vx, vy) = self._estado_lunar_bruto(cuerpo, t_sim)
        return self._a_marco_lanzamiento(x, y), self._a_marco_lanzamiento(vx, vy)

    def _init_efemerides_lunar(self) -> None:
        """Política estricta: SPICE real (sin respaldo) o proveedor inyectado
        que cumpla el contrato; en ambos casos se validan los datos que se
        van a usar (todo el horizonte de la misión) antes de despegar."""
        cfg = self.config
        if self._proveedor_efemerides is not None:
            self._efemerides_modo = "proveedor"
        elif _TERCEROS_CUERPOS_FUENTE == "pymcel":
            self._efemerides_modo = "pymcel"
        else:
            raise EfemeridesNoValidas(
                "modo lunar: exige efemérides consultables en cualquier "
                "instante (pymcel/SPICE o un proveedor inyectado). La fuente "
                f"{_TERCEROS_CUERPOS_FUENTE!r} (Horizons+Verlet o Kepler) no "
                "las garantiza")
        cuerpos = ["luna"] + (["sol"] if "sol" in cfg.cuerpos_perturbadores else [])
        limites = {"luna": ((3.3e8, 4.2e8), (0.8e3, 1.3e3)),
                   "sol": ((1.4e11, 1.6e11), (2.8e4, 3.2e4))}
        horizonte = cfg.T_MAX_LUNAR + 86400.0
        tiempos = [k * 43200.0 for k in range(int(horizonte // 43200.0) + 1)]
        tiempos.append(horizonte)
        for cuerpo in cuerpos:
            (r_lo, r_hi), (v_lo, v_hi) = limites[cuerpo]
            try:
                for t in tiempos:
                    (x, y), (vx, vy) = self._estado_lunar_bruto(cuerpo, t)
                    vals = (x, y, vx, vy)
                    if not all(math.isfinite(v) for v in vals):
                        raise EfemeridesNoValidas(
                            f"modo lunar: efemérides de {cuerpo!r} no finitas en t={t:.0f} s")
                    r, v = math.hypot(x, y), math.hypot(vx, vy)
                    if not (r_lo <= r <= r_hi and v_lo <= v <= v_hi):
                        raise EfemeridesNoValidas(
                            f"modo lunar: efemérides de {cuerpo!r} fuera de rango en "
                            f"t={t:.0f} s (r={r:.4g} m, v={v:.4g} m/s)")
                # Contrato de consulta arbitraria: orden libre y repetible
                a = self._estado_lunar_bruto(cuerpo, tiempos[-1])
                self._estado_lunar_bruto(cuerpo, 0.0)
                if a != self._estado_lunar_bruto(cuerpo, tiempos[-1]):
                    raise EfemeridesNoValidas(
                        f"modo lunar: el proveedor de {cuerpo!r} no es determinista "
                        f"ante consultas en desorden")
            except EfemeridesNoValidas:
                raise
            except (RuntimeError, OSError, ValueError, KeyError) as e:
                raise EfemeridesNoValidas(
                    f"modo lunar: no hay efemérides válidas de {cuerpo!r} "
                    f"({type(e).__name__}: {e})") from e
        if isinstance(self._proveedor_efemerides, _ProveedorRebound):
            if self._proveedor_efemerides.con_planetas:
                self._log("✓ REBOUND con planetas: validado contra SPICE (2026-01-01, 10 d): "
                          "Luna ≤ 0,4 km, Sol ≤ 0,1 km (validar_rebound_v1.py).")
            else:
                self._log("⚠ REBOUND sin planetas (no hay SPICE): la Luna casi no cambia, "
                          "pero el Sol puede apartarse ~50 km a 10 d.")
        elif type(self._proveedor_efemerides) is _ProveedorHorizonsVerlet:
            self._validacion_horizons = _leer_validacion_horizons_verlet(cfg.fecha_lanzamiento)
            if self._validacion_horizons is None:
                self._log("⚠ Horizons+Verlet: discrepancia empírica NO medida para "
                          "esta versión y esta fecha. Ejecute "
                          "validar_horizons_verlet.py antes de confiar "
                          "en resultados de precisión.")
            elif not self._validacion_horizons.get("aceptada", False):
                self._log("⚠ Horizons+Verlet: la validación guardada no supera los "
                          "umbrales configurados; revise el informe JSON.")
            else:
                err = self._validacion_horizons["errores"]["luna"]
                max_luna = max(float(v["dr_m"]) for v in err.values())
                self._log(f"✓ Horizons+Verlet validado contra Horizons para esta fecha; "
                          f"error máximo lunar={max_luna / 1000:.3f} km.")
        self._log(f"✓ Efemérides aceptadas para la propagación ({self.efemerides_efectiva()}); "
                  "la precisión externa se informa por separado.")

    def _init_efemerides(self) -> None:
        cfg = self.config
        if self._lunar:
            self._init_efemerides_lunar()
            return
        if not (cfg.usar_terceros_cuerpos or cfg.usar_marco_movil):
            self._efemerides_modo = "sin_uso"
            return
        if _TERCEROS_CUERPOS_FUENTE == "pymcel":
            self._efemerides_modo = "pymcel"
            return
        try:
            xS, yS, vxS, vyS = _fetch_horizons_state("sol", cfg.fecha_lanzamiento)
            xL, yL, vxL, vyL = _fetch_horizons_state("luna", cfg.fecha_lanzamiento)
            self._efemerides_verlet = _EphemerisVerlet(
                xS, yS, vxS, vyS, xL, yL, vxL, vyL
            )
            self._efemerides_modo = "horizons_verlet"
            print(f"✓ Efemérides: Horizons + Verlet  |  "
                  f"Sol a {np.hypot(xS, yS) / 1e9:.3f} Gm  |  "
                  f"Luna a {np.hypot(xL, yL) / 1e6:.1f} Mm")
        except (RuntimeError, OSError, ValueError) as e:
            self._log(f"⚠ Horizons no disponible ({type(e).__name__}: {e})")
            self._log("  Usando Kepler analítico simplificado para Sol/Luna.")
            self._efemerides_modo = "kepler_simplificado"

    def efemerides_efectiva(self) -> str:
        """Fuente real de las efemérides (para HUD y consola). `_efemerides_modo`
        es la que se PIDIÓ; en modo pymcel puede haber caído a Kepler."""
        if self._efemerides_modo == "proveedor":
            return (getattr(self._proveedor_efemerides, "etiqueta", None)
                    or "proveedor inyectado (determinista, no es SPICE)")
        if self._efemerides_modo == "pymcel":
            return _eph_descripcion(self.config.fecha_lanzamiento, fallos=self._eph_fallo)
        return self._efemerides_modo

    def _posicion_cuerpo(self, cuerpo: str, t_sim: float) -> Vec2:
        """Posición 2D (x, y) de `cuerpo` ("sol"/"luna") relativa a la
        Tierra, en el instante fecha_lanzamiento + t_sim (12.6.4: la parte de
        posición de `_estado_cuerpo`, que concentra el despacho)."""
        return self._estado_cuerpo(cuerpo, t_sim)[0]

    def _estado_cuerpo(self, cuerpo: str, t_sim: float) -> tuple[Vec2, Vec2]:
        """(posición, velocidad) de `cuerpo` relativa a la Tierra.
        Horizons+Verlet y Kepler dan velocidad exacta (analítica /
        del propio integrador); pymcel usa la velocidad que devuelve
        SPICE en la misma consulta (una sola llamada, sin diferencia
        finita; cada consulta es independiente, sin restricción de
        monotonía, a diferencia del Verlet)."""
        if self._lunar:
            return self._estado_lunar(cuerpo, t_sim)
        if self._efemerides_modo == "horizons_verlet":
            _exigir_sol_o_luna(cuerpo)
            self._efemerides_verlet.advance_to(t_sim)
            if cuerpo == "sol":
                return self._efemerides_verlet.pos_sun(), self._efemerides_verlet.vel_sun()
            return self._efemerides_verlet.pos_moon(), self._efemerides_verlet.vel_moon()
        if self._efemerides_modo != "pymcel":
            t_epoca = _epoca_a_segundos(self.config.fecha_lanzamiento) + t_sim
            return _estado_kepler_simplificado(cuerpo, t_epoca)
        return _estado_pymcel(cuerpo, self.config.fecha_lanzamiento, t_sim,
                              fallos=self._eph_fallo)

    # ========================================================
    # CONSULTAS DE MASA / ÁREA / ETAPA
    # ========================================================
    def masa_total(self) -> float:
        return sum(e.masa_actual() for e in self.etapas if not e.separada)

    def _ultima_etapa_no_separada(self) -> Optional[Etapa]:
        for etapa in reversed(self.etapas):
            if not etapa.separada:
                return etapa
        return None

    def area_frontal(self) -> float:
        etapa = self._ultima_etapa_no_separada()
        return np.pi * etapa.r * etapa.r if etapa is not None else 0.0

    def Cd_actual(self) -> float:
        etapa = self._ultima_etapa_no_separada()
        return etapa.Cd if etapa is not None else 0.0

    def etapa_actual(self) -> Optional[Etapa]:
        if self.estado.etapa_actual < len(self.etapas):
            return self.etapas[self.estado.etapa_actual]
        return None

    def _hay_combustible_en_alguna_etapa(self) -> bool:
        return any((not e.separada) and e.tiene_combustible()
                   for e in self.etapas)

    @staticmethod
    def _etapa_puede_encender(etapa: Optional[Etapa]) -> bool:
        """Hay etapa viva con combustible (aunque no esté encendida)."""
        return (etapa is not None and not etapa.separada
                and etapa.tiene_combustible())

    @staticmethod
    def _etapa_esta_encendida(etapa: Optional[Etapa]) -> bool:
        """Empuje: solo si la etapa está ACTIVA (no basta con tener
        combustible)."""
        return (etapa is not None and etapa.activa
                and not etapa.separada and etapa.tiene_combustible())

    def _aceleracion_a_empuje_pleno(self) -> Optional[float]:
        """a = u0·D / m con la etapa actual a throttle 1; None si no hay
        etapa o masa utilizable."""
        etapa = self.etapa_actual()
        masa = self.masa_total()
        if etapa is None or masa <= 0.0:
            return None
        return etapa.u0 * etapa.D / masa

    # ========================================================
    # PERFIL DE MISIÓN (predicados sobre la configuración)
    #
    # usa_bcb() devuelve True SOLO cuando el perfil activo debe ejecutar
    # Burn-Coast-Burn (cutoff intermedio por apogeo + segunda quema en
    # el apogeo físico):
    #
    #   satelital              → True
    #   tripulada + perfil bcb → True
    #   tripulada + perfil dir → False  ← ascenso continuo
    #
    # Nota sobre _throttle_ventana_apogeo (por qué solo aplica a
    # tripulada/directo):
    #   · directo: la quema de circularización empieza mientras el
    #     apogeo objetivo recién se alcanzó, TODAVÍA ASCENDIENDO
    #     (el ascenso sigue por inercia aunque el empuje se corte),
    #     así que cortar lejos del apogeo actual es seguro y evita
    #     que una quema larga (limitada por g) infle el apogeo hasta
    #     una trayectoria hiperbólica.
    #   · bcb: su segunda quema arranca exactamente EN el apogeo
    #     físico y empieza a descender de inmediato — cortar el
    #     empuje ahí sería caída libre real, no un "coast" seguro
    #     hasta el siguiente paso por el apogeo (eso solo existe una
    #     vez que el periapsis ya no es negativo, algo que con
    #     empuje limitado por g puede tardar más de lo que el
    #     descenso tarda en volverse peligroso).
    #   · satelital: el empuje no está limitado por g, la quema es
    #     casi impulsiva (segundos) y termina antes de alejarse del
    #     apogeo, así que la ventana no hace falta.
    # ========================================================
    def usa_bcb(self) -> bool:
        cfg = self.config
        # gto comparte la secuencia B–C–B con satelital: si se toca
        # esta función, gto se rompe en silencio (ver preset gto).
        if cfg.modo_mision in ("satelital", "gto", "deorbit", "lunar"):
            return True
        if cfg.modo_mision == "tripulada":
            return cfg.perfil_tripulada == "bcb"
        return False

    def es_tripulada_bcb(self) -> bool:
        cfg = self.config
        return (cfg.modo_mision == "tripulada"
                and cfg.perfil_tripulada == "bcb")

    def es_tripulada_directa(self) -> bool:
        cfg = self.config
        return (cfg.modo_mision == "tripulada"
                and cfg.perfil_tripulada == "directo")

    def es_deorbit(self) -> bool:
        return self.config.modo_mision == "deorbit"

    def es_alta_energia(self) -> bool:
        return self.config.modo_mision in ("tli", "escape")

    def es_lunar(self) -> bool:
        return self._lunar

    def rp_objetivo_circ(self) -> float:
        """Periapsis (altitud, m) que da por terminada la circularización."""
        cfg = self.config
        return cfg.GTO_RP_FINAL if cfg.modo_mision == "gto" else cfg.ALTURA_SEGURIDAD

    def exige_ecc_en_corte(self) -> bool:
        """Perfiles cuyo corte de circularización exige también ecc."""
        return (self.es_tripulada_bcb() or self.es_deorbit()
                or self.config.modo_mision in ("gto", "lunar"))

    def ecc_max_efectiva(self) -> float:
        cfg = self.config
        if cfg.modo_mision == "gto":
            return cfg.ECC_MAX_GTO
        if cfg.modo_mision == "lunar":
            return cfg.ECC_MAX_APARCAMIENTO
        if cfg.modo_mision == "deorbit":
            # La órbita previa a la desórbita es la LEO casi circular
            # de tripulada/bcb (misma guía de 2ª quema y mismo corte).
            return cfg.ECC_MAX_TRIPULADA_BCB
        if cfg.modo_mision != "tripulada":
            return 1.0
        if cfg.perfil_tripulada == "directo":
            return cfg.ECC_MAX_TRIPULADA_DIRECTO
        return cfg.ECC_MAX_TRIPULADA_BCB

    # ========================================================
    # DATOS PARA LAS GRÁFICAS FINALES
    # ========================================================
    def datos_para_graficas(self) -> dict:
        if self._snapshot_orbita is not None:
            return self._snapshot_orbita
        return {
            "hist_t":       self.hist_t,
            "hist_alt":     self.hist_alt,
            "hist_v":       self.hist_v,
            "hist_fase":    self.hist_fase,
            "tray_wx":      self.tray_wx,
            "tray_wy":      self.tray_wy,
            "tray_wfase":   self.tray_wfase,
            "tray_ex":      self.tray_ex,
            "tray_ey":      self.tray_ey,
            "t_orbita":     None,
            "periodo":      None,
            "datos_orbita": None,
        }

    # ========================================================
    # BUCLE PRINCIPAL
    # ========================================================
    def avanzar_paso(self) -> None:
        est = self.estado
        if est.simulacion_terminada or est.pausado:
            return

        if self._deorbit_sitio and self._debe_planificar_deorbit():
            if not self._resolver_plan_deorbit():
                self.esperando_plan = True      # el plan externo aún no llegó
                return
            self.esperando_plan = False
            if est.simulacion_terminada:        # (12.13.5) timeout/fallo terminal del plan
                return

        if self._lunar_plan_pendiente:
            if not self._resolver_plan_lunar():
                self.esperando_plan = True
                return
            self.esperando_plan = False
            if est.simulacion_terminada:
                return

        if self.es_alta_energia():
            self._ajustar_dt_alta_energia()
        elif self._lunar:
            self._ajustar_dt_lunar()

        if self._lunar:
            y_prev, t_prev = (est.px, est.py, est.vx, est.vy), est.t

        if self._lunar and est.fase in ("translunar", "orbita_lunar"):
            self._paso_coast_lunar()
        elif self._lunar and est.fase == "quema_loi":
            self._paso_quema_loi_sim()
        elif (self.config.usar_marco_movil and self.config.eventos == "integrador"
                and self.config.dinamica == "variable"):
            self._fisica_con_transiciones()       # 12.2.0: cruce exacto
        else:
            if self.config.usar_marco_movil:
                self._verificar_transicion_marco()
            self._paso_fisica()
        if self._lunar and not est.simulacion_terminada and est.t > t_prev:
            # La superficie lunar es un evento físico distinto de la SOI.
            # Se comprueba con el estado completo del paso, antes de que el
            # detector de SOI pueda localizar el cruce y recortar el estado.
            self._detectar_impacto_lunar(y_prev, t_prev)
            if not est.simulacion_terminada:
                self._detectar_soi_lunar(y_prev, t_prev)  # entrada SOI exacta
        if not est.simulacion_terminada:
            self._actualizar_logica_de_mision()

        px_d, py_d = self.posicion_display_eci()

        if self.es_alta_energia():
            self._seguimiento_alta_energia(px_d, py_d)
        elif self._lunar:
            self._seguimiento_lunar()

        self._registrar_historial(px_d, py_d)
        self._gestionar_historial_orbital()
        if est.simulacion_terminada:
            print("✓ Paso terminal registrado; avanzar_paso terminó.")

    def _ajustar_dt_alta_energia(self) -> None:
        """Paso grande solo en coast sin empuje y fuera de la atmósfera
        densa; con empuje (o si hay que reencender) vuelve al base."""
        est = self.estado
        en_coast = (est.marco != "tierra"
                    or (est.fase == "coasting" and est.empuje == 0.0
                        and est.altitud > 200_000.0))
        est.dt = (self.config.DT_COAST_ALTA_ENERGIA if en_coast
                  else self._dt_base)

    def _actualizar_logica_de_mision(self) -> None:
        """Fases de misión: solo existen en el marco Tierra. En
        "crucero"/"luna" todavía no hay fases propias (inyección
        translunar, inserción lunar, etc.): la nave se propaga
        balísticamente en el marco activo."""
        cfg, est = self.config, self.estado
        if (cfg.modo_mision == "suborbital" and not est.simulacion_terminada
                and est.t >= cfg.T_MAX_SUBORBITAL):
            self._fin_suborbital_por_t_max()
            return
        if self.estado.marco != "tierra":
            return
        self._actualizar_mision()
        self._verificar_burnout()
        self._verificar_orbita()
        if self.eventos.apogeo_registrado:
            self._evaluar_mision_suborbital()

    def _fin_suborbital_por_t_max(self) -> None:
        """(12.10.1) Sin apogeo ni impacto en `T_MAX_SUBORBITAL` la trayectoria no es suborbital
        (p. ej. cutoff por apogeo apagado y energía de escape): NO CUMPLIDA con motivo explícito.
        Si ya había veredicto, solo se termina la simulación."""
        est, ev, cfg = self.estado, self.eventos, self.config
        if ev.mision_cumplida is None:
            ev.mision_cumplida = False
            ev.datos_mision.update({
                "alt_max": ev.alt_max, "objetivo": cfg.APOGEO_OBJETIVO,
                "cruzo_karman": ev.alt_max > cfg.KARMAN, "fin_por_t_max": True,
                "motivo": (f"sin apogeo ni impacto en T_MAX_SUBORBITAL "
                           f"({cfg.T_MAX_SUBORBITAL / 3600:.1f} h): trayectoria no suborbital")})
        est.simulacion_terminada = True
        print(f"⏱ Fin por T_MAX_SUBORBITAL ({est.t / 3600:.2f} h)")

    def _registrar_historial(self, px_display: float, py_display: float) -> None:
        est = self.estado
        self.hist_t.append(est.t)
        self.hist_alt.append(est.altitud)
        self.hist_v.append(est.speed)
        self.hist_fase.append(est.fase)
        self.tray_wx.append(px_display)
        self.tray_wy.append(py_display)
        self.tray_wfase.append(est.fase)
        bary_x, bary_y = self.baricentro_eci()
        self.tray_ex.append(bary_x)
        self.tray_ey.append(bary_y)

    def _seguimiento_alta_energia(self, px_d: float, py_d: float) -> None:
        """r_max y distancia mínima a la Luna, SIEMPRE desde la posición
        ECI de dibujo (est.px/py están en coordenadas del marco activo:
        crucero/luna no sirven para esto). Fin por entrada a la esfera
        de influencia lunar o por T_MAX_ALTA_ENERGIA."""
        est = self.estado
        ev = self.eventos
        r_alt = float(math.hypot(px_d, py_d) - R_T)
        if r_alt > ev.r_max_eci:
            ev.r_max_eci = r_alt
        if est.t - self._t_chk_luna >= 30.0:
            self._t_chk_luna = est.t
            lx, ly = self._posicion_cuerpo("luna", est.t)
            d = math.hypot(lx - px_d, ly - py_d)
            if d < ev.d_luna_min:
                ev.d_luna_min = d
        if est.marco == "luna" and not ev.entro_soi_luna:
            ev.entro_soi_luna = True
            ev.datos_soi_luna = {"t": est.t, "r_alt": r_alt}
            if (self.config.modo_mision == "tli" and ev.mision_cumplida is None
                    and ev.datos_mision.get("resultado") == "llego_luna"):
                ev.mision_cumplida = True
                ev.datos_mision["t_soi"] = est.t
            est.simulacion_terminada = True
            print(f"🌕 ENTRADA A LA ESFERA DE INFLUENCIA LUNAR en t={est.t / 3600:.1f} h")
        elif est.t >= self.config.T_MAX_ALTA_ENERGIA:
            if self.config.modo_mision == "tli" and ev.mision_cumplida is None:
                # Estado terminal inequívoco (11.21.1): «NO CUMPLIDA
                # (T_MAX sin SOI)», distinto de un fallo del integrador o de
                # un impacto (que fijan su propio motivo).
                ev.mision_cumplida = False
                if ev.datos_mision.get("resultado") == "llego_luna":
                    ev.datos_mision["motivo"] = "T_MAX sin SOI"
                else:       # el tiempo se agotó antes de completar el corte TLI
                    ev.datos_mision["motivo"] = "T_MAX sin completar el corte TLI"
                ev.datos_mision["fin_por_t_max"] = True
            est.simulacion_terminada = True
            print(f"⏱ Fin por T_MAX_ALTA_ENERGIA ({est.t / 86400:.1f} d)")

    # ========================================================
    # THROTTLE
    # Cada regulador devuelve un factor en [THROTTLE_MIN, 1] (o 1.0 si
    # no aplica); el throttle final es el MÍNIMO de todos.
    # ========================================================
    def _throttle(self) -> float:
        return min(
            self._throttle_circularizacion(),
            self._throttle_g_limit(),
            self._throttle_ventana_apogeo(),
            self._throttle_alta_energia(),
            self._throttle_deorbit(),
            self._throttle_tli(),
        )

    def _throttle_proporcional_por_dv(self, dv_falta: float,
                                      aceleracion_plena: float) -> float:
        """Empuje pleno hasta quedar ~TAU s de quema y luego proporcional
        (aproximación exponencial), con piso THROTTLE_MIN."""
        throttle_min = self.config.THROTTLE_MIN
        if dv_falta <= 0.0:
            return throttle_min
        return float(_clip(dv_falta / (aceleracion_plena * TAU_REGULACION_THROTTLE),
                           throttle_min, 1.0))

    def _throttle_circularizacion(self) -> float:
        """Reduce el empuje en circularización al acercarse al objetivo."""
        est = self.estado
        cfg = self.config
        if est.fase != "circularizacion":
            return 1.0
        orb = elementos_orbitales(est.px, est.py, est.vx, est.vy)
        if cfg.modo_mision == "gto":
            if est.segunda_quema_realizada:
                return self._throttle_gto_segunda_quema()
            return self._throttle_gto_primera_quema(orb)
        return self._throttle_por_periapsis(orb)

    def _throttle_gto_primera_quema(self, orb: dict) -> float:
        """gto, quema 1: a energías de transferencia el apogeo sube
        ~50 km por cada m/s, así que un paso a empuje pleno se
        pasa del objetivo por cientos de km. Se reduce el empuje
        en los últimos 10 % de apogeo antes del corte."""
        cfg = self.config
        falta_ra = cfg.APOGEO_OBJETIVO - orb["ra"]
        margen_ra = 0.10 * cfg.APOGEO_OBJETIVO
        if falta_ra >= margen_ra:
            return 1.0
        if falta_ra <= 0.0:
            return cfg.THROTTLE_MIN
        return (cfg.THROTTLE_MIN
                + (1.0 - cfg.THROTTLE_MIN) * falta_ra / margen_ra)

    def _throttle_gto_segunda_quema(self) -> float:
        """gto, quema 2 (en el apogeo): el periapsis sube casi de golpe
        en los últimos ~100 m/s, así que una rampa por periapsis
        llega tarde y un paso a empuje pleno (decenas de m/s) salta
        la ventana rp >= GTO_RP_FINAL y ecc < ECC_MAX_GTO. Se
        regula por el Δv que falta hasta la velocidad circular
        local."""
        est = self.estado
        a_pleno = self._aceleracion_a_empuje_pleno()
        if a_pleno is None:
            return 1.0
        v_circ = math.sqrt(MU_T / math.hypot(est.px, est.py))
        dv_falta = v_circ - est.speed
        return self._throttle_proporcional_por_dv(dv_falta, a_pleno)

    def _throttle_por_periapsis(self, orb: dict) -> float:
        """Rampa lineal al acercarse al periapsis objetivo."""
        cfg = self.config
        rp_objetivo = self.rp_objetivo_circ()
        falta = rp_objetivo - orb["rp"]
        # El margen se escala con el objetivo (5 %): en LEO sigue
        # siendo THROTTLE_MARGEN (30 km); en GEO ~1750 km.
        margen = max(cfg.THROTTLE_MARGEN, 0.05 * rp_objetivo)
        if falta <= 0.0:
            return cfg.THROTTLE_MIN
        if falta >= margen:
            return 1.0
        fraccion = falta / margen
        return cfg.THROTTLE_MIN + (1.0 - cfg.THROTTLE_MIN) * fraccion

    def _throttle_g_limit(self) -> float:
        """
        En modo tripulado, limita el empuje para no superar la
        aceleración máxima tolerada. Lo que siente la tripulación es la
        aceleración propia (fuerza no gravitatoria / masa), que se aproxima
        como T/m: la gravedad NO se suma (desde 11.19.0; antes se restaba
        g_local al límite y el efectivo quedaba ~1 g por debajo). El
        arrastre se ignora (frena, así que reduce la carga axial).
        """
        if self.config.modo_mision != "tripulada":
            return 1.0

        est = self.estado
        etapa = self.etapa_actual()
        if (etapa is None or not etapa.activa
                or etapa.separada or not etapa.tiene_combustible()):
            return 1.0

        masa = self.masa_total()
        if masa <= 0.0:
            return 1.0

        a_max = self.config.G_MAX_TRIPULADA * G0
        empuje_max = a_max * masa

        empuje_nominal = etapa.u0 * etapa.D
        if empuje_nominal <= 0.0:
            return 1.0
        # El empuje real es T_vacío·throttle − A·p: para que sea empuje_max
        # hay que compensar la pérdida por presión.
        empuje_max += etapa.perdida_por_presion(self._presion_ambiente())
        return float(_clip(empuje_max / empuje_nominal, 0.0, 1.0))

    def _throttle_ventana_apogeo(self) -> float:
        """
        Solo para tripulada/directo, durante la quema real de
        circularización (modo_guiado == "circularizacion", ya con
        apogeo objetivo alcanzado): el empuje prógrado solo es
        eficiente cerca del apogeo actual, así que se corta lejos de
        él para no inflarlo con una quema larga (limitada por g).
        Nunca bloquea el empuje bajo ALTURA_SEGURIDAD. Por qué no
        aplica a bcb ni a satelital: ver comentario sobre usa_bcb().
        """
        est = self.estado
        cfg = self.config
        if not self.es_tripulada_directa():
            return 1.0
        if est.fase != "circularizacion" or est.modo_guiado != "circularizacion":
            return 1.0
        if est.altitud < cfg.ALTURA_SEGURIDAD:
            return 1.0
        orb = elementos_orbitales(est.px, est.py, est.vx, est.vy)
        if not orb["es_elipse"]:
            return 1.0
        falta_apogeo = orb["ra"] - est.altitud
        ventana = max(orb["ra"] * 0.05, 10000.0)
        return 1.0 if falta_apogeo <= ventana else 0.0

    def _throttle_alta_energia(self) -> float:
        """
        tli/escape, quema única: a energías de transferencia el
        apogeo sube cientos de km por cada m/s (sin rampa, el corte
        a 384 400 km se pasó hasta 1 225 000 km). Se regula por el Δv
        que falta hasta la energía objetivo (ε_obj - ε)/v. Escape:
        ε_obj = 0. Tli: ε de una elipse con el periapsis actual y
        apogeo = objetivo.
        """
        cfg = self.config
        est = self.estado
        if not self.es_alta_energia() or est.cutoff_por_apogeo:
            return 1.0
        a_pleno = self._aceleracion_a_empuje_pleno()
        if a_pleno is None:
            return 1.0
        orb = elementos_orbitales(est.px, est.py, est.vx, est.vy)
        if cfg.modo_mision == "escape":
            eps_objetivo = 0.0
        else:
            # OJO: se mira eps, no es_elipse: en el despegue la órbita
            # instantánea es rectilínea (ecc = 1, eps < 0) y es_elipse
            # da False; con eso el motor quedaba al piso y no despegaba.
            if orb["eps"] >= 0.0:
                return cfg.THROTTLE_MIN
            eps_objetivo = -MU_T / (orb["rp_radio"] + cfg.APOGEO_OBJETIVO + R_T)
        dv_falta = (eps_objetivo - orb["eps"]) / max(est.speed, 1.0)
        return self._throttle_proporcional_por_dv(dv_falta, a_pleno)

    def _throttle_tli(self) -> float:
        """Modo lunar, quema TLI: regulación por el Δv que falta hasta la
        energía objetivo del plan (la misma rampa que tli/escape)."""
        est = self.estado
        if not self._lunar or est.fase != "quema_tli" or self._lunar_plan is None:
            return 1.0
        a_pleno = self._aceleracion_a_empuje_pleno()
        if a_pleno is None:
            return 1.0
        orb = elementos_orbitales(est.px, est.py, est.vx, est.vy)
        dv_falta = (self._lunar_plan["eps_obj"] - orb["eps"]) / max(est.speed, 1.0)
        return self._throttle_proporcional_por_dv(dv_falta, a_pleno)

    def _throttle_deorbit(self) -> float:
        """
        Fase deorbit: la quema retrógrada es corta (~100-200 m/s) y a
        empuje pleno cada paso avanza decenas de m/s, así que el
        periapsis se pasa de 40 km. Se regula por el Δv que falta
        hasta la energía de la elipse con el MISMO radio actual y
        periapsis = DEORBIT_RP_OBJETIVO: (ε - ε_obj)/v.
        """
        cfg = self.config
        est = self.estado
        if not self.es_deorbit() or est.fase != "deorbit":
            return 1.0
        a_pleno = self._aceleracion_a_empuje_pleno()
        if a_pleno is None:
            return 1.0
        orb = elementos_orbitales(est.px, est.py, est.vx, est.vy)
        r = math.hypot(est.px, est.py)
        rp_objetivo_radio = R_T + cfg.DEORBIT_RP_OBJETIVO
        eps_objetivo = -MU_T / (r + rp_objetivo_radio)
        dv_falta = (orb["eps"] - eps_objetivo) / max(est.speed, 1.0)
        return self._throttle_proporcional_por_dv(dv_falta, a_pleno)

    # ========================================================
    # PERTURBACIONES DE TERCEROS CUERPOS (consulta efemérides + física pura)
    # ========================================================
    def _aceleracion_terceros_cuerpos(self) -> Vec2:
        """
        Suma la aceleración de marea (perturbación diferencial, ver
        nota en la sección TERCEROS CUERPOS) de cada cuerpo activo en
        `cuerpos_perturbadores`, evaluada en el instante actual
        `est.t`. Se llama una vez por paso físico externo (no por
        sub-paso de integración) porque la posición de estos cuerpos
        cambia de forma despreciable en la escala de un solo paso
        frente al tamaño de la perturbación misma.
        """
        cfg = self.config
        if not cfg.usar_terceros_cuerpos:
            return 0.0, 0.0
        est = self.estado
        perturbacion_x = perturbacion_y = 0.0
        for cuerpo in cfg.cuerpos_perturbadores:
            pos_cuerpo = self._posicion_cuerpo(cuerpo, est.t)
            ax, ay = aceleracion_de_marea(
                CUERPOS[cuerpo].mu, (est.px, est.py), pos_cuerpo)
            perturbacion_x += ax
            perturbacion_y += ay
        return float(perturbacion_x), float(perturbacion_y)

    def _aceleracion_terceros_cuerpos_luna(self) -> Vec2:
        """
        Análogo a _aceleracion_terceros_cuerpos pero para la zona
        "luna": el primario es la Luna, y la Tierra (siempre) más el
        Sol (si está en cuerpos_perturbadores) actúan como
        perturbación de marea relativa a la Luna.
        """
        cfg = self.config
        est = self.estado
        # La Tierra perturba siempre que haya efemérides en uso (con
        # marco móvil también, para ser coherente con el CRTBP de la
        # zona "crucero" y no crear un salto al cruzar el SOI).
        if not (cfg.usar_terceros_cuerpos or cfg.usar_marco_movil):
            return 0.0, 0.0
        perturbacion_x = perturbacion_y = 0.0
        pos_nave = (est.px, est.py)

        luna_x, luna_y = self._posicion_cuerpo("luna", est.t)
        # Tierra relativa a la Luna = -(Luna relativa a la Tierra)
        ax, ay = aceleracion_de_marea(MU_T, pos_nave, (-luna_x, -luna_y))
        perturbacion_x += ax
        perturbacion_y += ay

        if cfg.usar_terceros_cuerpos and "sol" in cfg.cuerpos_perturbadores:
            sol_x, sol_y = self._posicion_cuerpo("sol", est.t)
            # Aproximamos la posición del Sol relativa a la Luna como
            # igual a la relativa a la Tierra (la distancia Tierra-Luna
            # es ~0.26% de la distancia Tierra-Sol, error despreciable).
            ax, ay = aceleracion_de_marea(
                MU_SOL, pos_nave, (sol_x - luna_x, sol_y - luna_y))
            perturbacion_x += ax
            perturbacion_y += ay
        return float(perturbacion_x), float(perturbacion_y)

    # ========================================================
    # MARCO MÓVIL: consultas de posición y transiciones
    # ========================================================
    def baricentro_eci(self) -> Vec2:
        """Baricentro Tierra-Luna respecto a la Tierra (ECI), o (0, 0)
        si la vista no es baricéntrica (así no se consulta la
        efeméride en cada paso sin necesidad)."""
        if self.config.marco_vista != "baricentro":
            return 0.0, 0.0
        luna_x, luna_y = self._posicion_cuerpo("luna", self.estado.t)
        return RAZON_MASA_TL * luna_x, RAZON_MASA_TL * luna_y

    def offset_vista(self, escala_log: bool) -> Vec2:
        """Vector que se SUMA a las coordenadas de vista (ECI, ya con
        escala log si aplica) para pasar al marco de vista elegido:
        la Tierra se dibuja en -baricentro. (0, 0) con vista Tierra."""
        bary_x, bary_y = self.baricentro_eci()
        ox, oy = vista_xy(bary_x, bary_y, escala_log)
        return -float(ox), -float(oy)

    def _angulo_marco_rotante(self) -> float:
        """Ángulo de la línea Tierra-Luna (marco crucero) en est.t."""
        est = self.estado
        return est.crucero_theta0 + est.crucero_Omega * (est.t - est.crucero_t0)

    def posicion_display_eci(self) -> Vec2:
        """
        Posición de la nave SOLO para dibujar, convertida siempre a
        inercial centrado en la Tierra (ECI), sin importar en qué
        marco dinámico esté activo ahora mismo (tierra/crucero/luna).
        No toca el estado real de la simulación — si se usara
        est.px/est.py directamente para graficar, la nave "saltaría"
        en pantalla en el instante exacto de cada cambio de marco,
        aunque la trayectoria física sea continua.
        """
        est = self.estado
        if est.marco == "tierra":
            return est.px, est.py
        luna_x, luna_y = self._posicion_cuerpo("luna", est.t)
        if est.marco == "luna":
            return est.px + luna_x, est.py + luna_y
        # "crucero": rotar de vuelta a orientación inercial y sumar
        # la posición del baricentro relativa a la Tierra.
        theta = self._angulo_marco_rotante()
        bary_x, bary_y = RAZON_MASA_TL * luna_x, RAZON_MASA_TL * luna_y
        r_inercial = _rot2d((est.px, est.py), theta)
        return r_inercial[0] + bary_x, r_inercial[1] + bary_y

    def _verificar_transicion_marco(self) -> None:
        """
        Chequea si toca cambiar de marco (tierra <-> crucero <-> luna).
        El único umbral físicamente "real" es la esfera de influencia
        lunar (R_SOI_LUNA); el límite tierra/crucero es una elección
        práctica (su complemento). Se usa un 5% de histéresis en los
        reingresos para no oscilar justo en el borde.
        """
        marco = self.estado.marco
        if marco == "tierra":
            self._transicion_desde_tierra()
        elif marco == "crucero":
            self._transicion_desde_crucero()
        elif marco == "luna":
            self._transicion_desde_luna()

    def _transicion_desde_tierra(self, forzar: bool = False) -> None:
        """tierra -> crucero al alejarse más allá de (a_Luna - R_SOI).
        `forzar` (12.2.0): el cruce ya se localizó; no se vuelve a comprobar."""
        est = self.estado
        r_tierra = math.hypot(est.px, est.py)
        if r_tierra < UMBRAL_TIERRA_CRUCERO and not forzar:
            return
        (luna_x, luna_y), (luna_vx, luna_vy) = self._estado_cuerpo("luna", est.t)
        a = math.hypot(luna_x, luna_y)
        Omega = math.sqrt(MU_SISTEMA_TL / a ** 3)
        theta0 = float(np.arctan2(luna_y, luna_x))
        bary_x, bary_y = RAZON_MASA_TL * luna_x, RAZON_MASA_TL * luna_y
        bary_vx, bary_vy = RAZON_MASA_TL * luna_vx, RAZON_MASA_TL * luna_vy
        r_rot, v_rot = _inercial_a_rotante(
            (est.px - bary_x, est.py - bary_y),
            (est.vx - bary_vx, est.vy - bary_vy),
            Omega, theta0,
        )
        est.px, est.py = r_rot
        est.vx, est.vy = v_rot
        est.marco = "crucero"
        est.crucero_a, est.crucero_Omega = a, Omega
        est.crucero_theta0, est.crucero_t0 = theta0, est.t
        print(f"🌗 Tierra → crucero (CRTBP) en t={est.t:.1f}s "
              f"(r_tierra={r_tierra/1000:.0f} km)")

    def _transicion_desde_crucero(self, destino: Optional[str] = None) -> None:
        """crucero -> luna (entra a la SOI lunar) o crucero -> tierra.
        `destino` (12.2.0): "luna" o "tierra" si el cruce ya se localizó."""
        est = self.estado
        theta = self._angulo_marco_rotante()
        x_luna_rot = (1.0 - RAZON_MASA_TL) * est.crucero_a
        x_tierra_rot = -RAZON_MASA_TL * est.crucero_a
        d_luna = math.hypot(est.px - x_luna_rot, est.py)
        d_tierra = math.hypot(est.px - x_tierra_rot, est.py)

        if (destino == "luna") or (destino is None and d_luna <= R_SOI_LUNA):
            r_in, v_in = _rotante_a_inercial(
                (est.px - x_luna_rot, est.py), (est.vx, est.vy),
                est.crucero_Omega, theta,
            )
            est.px, est.py = r_in
            est.vx, est.vy = v_in
            est.marco = "luna"
            print(f"🌕 crucero → Luna en t={est.t:.1f}s (d_luna={d_luna/1000:.0f} km)")
            return

        if ((destino == "tierra")
                or (destino is None and d_tierra <= UMBRAL_TIERRA_CRUCERO * 0.95)):
            # (est.px - x_tierra_rot, est.py) YA es la posición de
            # la nave relativa a la TIERRA (en coordenadas del
            # marco rotante); _rotante_a_inercial solo rota ese
            # vector a orientación inercial, no cambia de origen.
            # r_in/v_in son entonces directamente el nuevo estado
            # tierra-céntrico — sumarles encima la posición del
            # baricentro (como se hacía antes) es un error: da
            # (nave-Tierra)+(baricentro-Tierra), que no es ninguna
            # cantidad física (salto de ~μ·|r_Luna| ≈ 4700 km en
            # posición y ~μ·|v_Luna| ≈ 12 m/s en velocidad). La
            # transición gemela crucero→luna ya hace esto bien (sin
            # sumar nada extra); este caso debe ser simétrico.
            r_in, v_in = _rotante_a_inercial(
                (est.px - x_tierra_rot, est.py), (est.vx, est.vy),
                est.crucero_Omega, theta,
            )
            est.px, est.py = r_in
            est.vx, est.vy = v_in
            est.marco = "tierra"
            print(f"🌍 crucero → Tierra en t={est.t:.1f}s (d_tierra={d_tierra/1000:.0f} km)")

    def _transicion_desde_luna(self, forzar: bool = False) -> None:
        """luna -> crucero al salir de la SOI lunar (con 5 % de histéresis)."""
        est = self.estado
        d_luna = math.hypot(est.px, est.py)
        if d_luna <= R_SOI_LUNA * 1.05 and not forzar:
            return
        theta = self._angulo_marco_rotante()
        x_luna_rot = (1.0 - RAZON_MASA_TL) * est.crucero_a
        dr_rot, dv_rot = _inercial_a_rotante(
            (est.px, est.py), (est.vx, est.vy),
            est.crucero_Omega, theta,
        )
        est.px, est.py = dr_rot[0] + x_luna_rot, dr_rot[1]
        est.vx, est.vy = dv_rot
        est.marco = "crucero"
        print(f"🌗 Luna → crucero en t={est.t:.1f}s (d_luna={d_luna/1000:.0f} km)")

    # --- (12.2.0) Transiciones de marco en el instante del cruce ----------
    def _fronteras_marco(self):
        """Fronteras del marco actual: [(destino, g(px, py))] con g >= 0 una
        vez cruzada (los mismos umbrales e histéresis de las transiciones por
        paso)."""
        marco = self.estado.marco
        if marco == "tierra":
            return [("crucero",
                     lambda x, y: math.hypot(x, y) - UMBRAL_TIERRA_CRUCERO)]
        if marco == "crucero":
            a = self.estado.crucero_a
            x_l, x_t = (1.0 - RAZON_MASA_TL) * a, -RAZON_MASA_TL * a
            return [("luna", lambda x, y: R_SOI_LUNA - math.hypot(x - x_l, y)),
                    ("tierra", lambda x, y: UMBRAL_TIERRA_CRUCERO * 0.95
                     - math.hypot(x - x_t, y))]
        if marco == "luna":
            return [("crucero",
                     lambda x, y: math.hypot(x, y) - R_SOI_LUNA * 1.05)]
        return []

    @staticmethod
    def _hermite_estado(y0, y1, dt: float, s: float) -> Estado4:
        """Estado (posición y velocidad) del polinomio cúbico de Hermite en la
        fracción s del paso."""
        p0x, p0y, v0x, v0y = y0[:4]
        p1x, p1y, v1x, v1y = y1[:4]
        if not dt > 0.0:                 # (12.10.1) paso nulo: el estado final (evita dividir por 0)
            return (p1x, p1y, v1x, v1y)
        h00, h10, h01, h11, d00, d10, d01, d11 = _hermite_coefs(s)
        return (h00 * p0x + h10 * dt * v0x + h01 * p1x + h11 * dt * v1x,
                h00 * p0y + h10 * dt * v0y + h01 * p1y + h11 * dt * v1y,
                (d00 * p0x + d10 * dt * v0x + d01 * p1x + d11 * dt * v1x) / dt,
                (d00 * p0y + d10 * dt * v0y + d01 * p1y + d11 * dt * v1y) / dt)

    def _cruce_de_marco(self, y0, y1, dt: float):
        """Primera frontera del marco actual que el paso y0 -> y1 cruzó:
        (fracción s del paso, estado en el cruce, destino) o None. La
        bisección de g sobre el Hermite da s con precisión de máquina."""
        mejor = None
        for destino, g in self._fronteras_marco():
            if not (g(y0[0], y0[1]) < 0.0 <= g(y1[0], y1[1])):
                continue
            def antes(s, g=g):
                e = self._hermite_estado(y0, y1, dt, s)
                return g(e[0], e[1]) < 0.0
            hi = _biseccion_s(antes, 80)
            if mejor is None or hi < mejor[0]:
                mejor = (hi, self._hermite_estado(y0, y1, dt, hi), destino)
        return mejor

    def _forzar_transicion(self, destino: str) -> None:
        marco = self.estado.marco
        if marco == "tierra":
            self._transicion_desde_tierra(forzar=True)
        elif marco == "crucero":
            self._transicion_desde_crucero(destino)
        elif marco == "luna":
            self._transicion_desde_luna(forzar=True)
        self.stats_eventos["transicion_marco"] += 1

    def _fisica_con_transiciones(self) -> None:
        """Un paso de física con las transiciones de marco en el instante del
        cruce: se integra el paso en el marco actual, si cruzó una frontera
        se localiza el instante, se transforma el estado ahí y el resto del
        paso se integra en el marco nuevo (a lo sumo 3 transiciones por paso)."""
        est = self.estado
        self._verificar_transicion_marco()     # estado que ya empieza cruzado
        dt_total, t_fin = est.dt, est.t + est.dt
        t_ini = est.t
        try:
            for _ in range(4):
                marco0 = est.marco
                y0 = (est.px, est.py, est.vx, est.vy)
                t0 = est.t
                est.dt = t_fin - t0
                if est.dt <= 1e-9:
                    break
                self._paso_fisica()
                if (est.simulacion_terminada or est.marco != marco0
                        or est.empuje > 0.0):
                    break
                y1 = (est.px, est.py, est.vx, est.vy)
                cruce = self._cruce_de_marco(y0, y1, est.dt)
                if cruce is None:
                    break
                s, ys, destino = cruce
                est.px, est.py, est.vx, est.vy = ys
                est.t = t0 + s * (t_fin - t0)
                self._forzar_transicion(destino)
                if est.marco == "tierra":
                    self._actualizar_altitud_y_rapidez()
                else:                          # altitud sobre la Tierra (ECI)
                    px_d, py_d = self.posicion_display_eci()
                    est.altitud = float(math.hypot(px_d, py_d) - R_T)
                    est.speed = math.hypot(est.vx, est.vy)
        finally:
            est.dt = dt_total
        est.t = t_ini + dt_total

    # ========================================================
    # PASO FÍSICO
    # ========================================================
    def _paso_fisica(self) -> None:
        est = self.estado
        if est.simulacion_terminada or est.pausado:
            return

        if est.marco != "tierra":
            self._paso_fisica_espacio_profundo()
            return

        self._actualizar_altitud_y_rapidez()

        if est.altitud <= 0.0 and est.t > 1.0:
            self._registrar_impacto()
            return

        masa = self.masa_total()
        etapa = self.etapa_actual()
        etapa_encendida = self._etapa_esta_encendida(etapa)

        empuje, flujo_masico = self._calcular_empuje(etapa, etapa_encendida)
        dir_x, dir_y = self._direccion_empuje(empuje, masa)
        empuje_x, empuje_y = empuje * dir_x, empuje * dir_y

        v_aire_rel_x, v_aire_rel_y = velocidad_relativa_al_aire(
            est.px, est.py, est.vx, est.vy, self._Omega_tierra)
        arrastre_x, arrastre_y = self._calcular_arrastre(v_aire_rel_x, v_aire_rel_y)
        self._registrar_max_q(v_aire_rel_x, v_aire_rel_y)

        perturbacion_x, perturbacion_y = self._aceleracion_terceros_cuerpos()
        variable = self.config.dinamica == "variable"
        if variable:
            fuerzas = LeyV12(
                self._Omega_tierra,
                etapa.u0 if etapa_encendida else 0.0,
                flujo_masico, etapa.A_sal if etapa_encendida else 0.0,
                self.config.empuje_vs_altitud, etapa_encendida,
                est.modo_guiado,
                bool(self.es_tripulada_bcb() or self.es_deorbit() or self._lunar),
                self.Cd_actual(), self.area_frontal(), self.config.cd_vs_mach,
                perturbacion_x, perturbacion_y)
            y0 = (est.px, est.py, est.vx, est.vy, masa, 0.0)
        else:
            fuerzas = (empuje_x, empuje_y, arrastre_x, arrastre_y, masa,
                       perturbacion_x, perturbacion_y)
            y0 = (est.px, est.py, est.vx, est.vy)
        dt = est.dt
        # Duración efectiva del empuje dentro del paso (<= dt): el motor
        # no puede quemar más combustible del que queda (11.21.0) ni
        # seguir empujando después del instante del corte (11.21.0).
        t_emp = dt
        if etapa_encendida and empuje > 0.0 and flujo_masico > 0.0:
            t_comb = etapa.m_comb_restante / flujo_masico
            if t_comb < dt * (1.0 - 1e-12):
                t_emp = t_comb
        if variable and self.config.eventos == "integrador":
            self._cierre_paso_eventos(y0, masa, etapa, etapa_encendida,
                                      empuje, t_emp, fuerzas)
            return
        y_fin = self._avanzar_con_empuje(y0, est.t, dt, t_emp, fuerzas,
                                         flujo_masico, empuje)
        if y_fin is None:
            self._fallo_integrador("marco tierra")
            return
        if (etapa_encendida and empuje > 0.0
                and self._corte_por_estado(y_fin)
                and not self._corte_por_estado(y0)):
            t_emp, y_fin = self._localizar_corte(y0, est.t, dt, t_emp,
                                                 fuerzas, flujo_masico,
                                                 empuje, y_fin)

        # Suelo: si el paso lo cruzó, el impacto se registra en el instante
        # del cruce (interpolación), no al final del paso.
        if (est.altitud > 0.0 and est.t > 1.0
                and math.hypot(y_fin[0], y_fin[1]) - R_T <= 0.0):
            y_imp, frac = self._cruce_suelo(y0, y_fin, dt)
            if etapa_encendida:
                t_q = min(t_emp, frac * dt)
                if variable:
                    # masa: quemado analítico hasta el impacto; Δv: parte
                    # proporcional del tramo con empuje
                    etapa.consumir(flujo_masico, t_q)
                    etapa.dv_prop += y_fin[5] * (t_q / t_emp if t_emp > 0 else 0.0)
                else:
                    etapa.acumular_dv(empuje, masa, flujo_masico, t_q)
                    etapa.consumir(flujo_masico, t_q)
            est.px, est.py, est.vx, est.vy = y_imp
            est.t += frac * dt
            self._actualizar_altitud_y_rapidez()
            self._registrar_impacto()
            return

        est.px, est.py, est.vx, est.vy = y_fin[:4]

        # Consumo: acoplado al encendido real (solo mientras empuja)
        if etapa_encendida:
            if variable:
                # la masa y el Δv salen del estado integrado
                etapa.consumir(1.0, masa - y_fin[4])
                etapa.dv_prop += y_fin[5]
            else:
                etapa.acumular_dv(empuje, masa, flujo_masico, t_emp)
                etapa.consumir(flujo_masico, t_emp)

        self._actualizar_altitud_y_rapidez()
        est.t += dt

    def _cierre_paso_eventos(self, y0, masa: float, etapa, etapa_encendida: bool,
                             empuje: float, t_emp: float, ley: "LeyV12") -> None:
        """(12.1.0) Un paso del marco Tierra con corte y suelo como eventos de
        solve_ivp. Deja el estado, la masa, el Δv, el combustible y el tiempo
        como `_paso_fisica` en los demás caminos."""
        est = self.estado
        dt = est.dt
        buscar_corte = (etapa_encendida and empuje > 0.0
                        and not self._corte_por_estado(y0))
        suelo = est.altitud > 0.0 and est.t > 1.0
        eps = _MARGEN_CORTE_EPS
        res = None   # solo con _CORTE_INTENTOS = 0 (el bucle no corre)
        resuelto = False
        for _ in range(_CORTE_INTENTOS):
            res = self._avanzar_v12_eventos(y0, est.t, dt, t_emp, ley, empuje,
                                            buscar_corte, suelo, eps)
            if res is None:
                self._fallo_integrador("marco tierra")
                return
            kind = res[3]
            # La lógica de misión mira el FIN del paso: tras el coast del
            # resto, la condición de corte debe seguir cumplida.
            if kind != "corte" or self._corte_por_estado(res[0]):
                resuelto = True
                if kind == "corte":
                    self.stats_eventos["corte_evento"] += 1
                break
            # El arrastre del resto del paso erosionó la condición: se
            # repite pidiendo un margen que cubra lo perdido.
            eps = 2.0 * (eps - self._margen_corte(res[0]))
            if not (eps < 1.0e4):
                break
        if res is None and not buscar_corte:     # _CORTE_INTENTOS = 0
            res = self._avanzar_v12_eventos(y0, est.t, dt, t_emp, ley, empuje,
                                            False, suelo)
            if res is None:
                self._fallo_integrador("marco tierra")
                return
        elif not resuelto and buscar_corte:
            # Reintentos agotados (o desactivados): bisección sobre el paso
            # (12.0.1), que garantiza la condición al final del paso.
            y_max = self._avanzar_con_empuje(y0, est.t, dt, t_emp, ley,
                                             ley.flujo, empuje)
            if y_max is None:
                self._fallo_integrador("marco tierra")
                return
            if self._corte_por_estado(y_max):
                t_emp_b, _ = self._localizar_corte(y0, est.t, dt, t_emp, ley,
                                                   ley.flujo, empuje, y_max)
                usa_biseccion = True
            else:                                # no hay corte en este paso
                t_emp_b, usa_biseccion = t_emp, False
            res = self._avanzar_v12_eventos(y0, est.t, dt, t_emp_b, ley,
                                            empuje, False, suelo)
            if res is None:
                self._fallo_integrador("marco tierra")
                return
            res = (res[0], t_emp_b, res[2], res[3])
            if usa_biseccion:
                self.stats_eventos["corte_biseccion"] += 1
        y_fin, t_emp_ef, t_imp, kind = res
        est.px, est.py, est.vx, est.vy = y_fin[:4]
        if etapa_encendida:
            # masa y Δv salen del estado integrado (también hasta el impacto)
            etapa.consumir(1.0, masa - y_fin[4])
            etapa.dv_prop += y_fin[5]
        if kind == "suelo":
            self.stats_eventos["suelo_evento"] += 1
            est.t = t_imp
            self._actualizar_altitud_y_rapidez()
            self._registrar_impacto()
            return
        self._actualizar_altitud_y_rapidez()
        est.t += dt
        # Red de seguridad: un paso sin evento (RK4 en coast) que terminó
        # bajo el suelo se registra como impacto al final del paso.
        if suelo and est.altitud <= 0.0:
            self._registrar_impacto()

    def _actualizar_altitud_y_rapidez(self) -> None:
        est = self.estado
        est.altitud = float(math.hypot(est.px, est.py) - R_T)
        est.speed = math.hypot(est.vx, est.vy)

    def _registrar_impacto(self) -> None:
        """Fin de la simulación por contacto con el suelo, y cierre de
        los criterios de misión que dependen del impacto."""
        est = self.estado
        ev = self.eventos
        est.simulacion_terminada = True
        ev.impacto_registrado = True
        ev.datos_impacto = {"t": est.t, "v": est.speed}
        if self.es_deorbit():
            d = self._error_angular_base()
            ev.datos_impacto.update(ang_base_deg=math.degrees(d),
                                    dist_base_km=abs(d) * R_T / 1000.0)
        self._evaluar_mision_suborbital()
        if self.es_deorbit() and ev.mision_cumplida is None:
            ev.mision_cumplida = False
            ev.datos_mision.setdefault("deorbit", True)
            ev.datos_mision.update({
                "motivo": ("impacto sin reentrada confirmada"
                           if self._deorbit_activo
                           else "impacto antes de activar la desórbita")})
        if self.config.modo_mision == "gto" and ev.mision_cumplida is None:
            ev.mision_cumplida = False
            ev.datos_mision.update({"t": est.t, "motivo": "impacto"})
        if self.es_alta_energia() and ev.mision_cumplida is None:
            ev.mision_cumplida = False
            if "resultado" in ev.datos_mision:      # ya hubo corte
                ev.datos_mision["motivo"] = "impacto antes de la SOI lunar"
        if self._lunar and ev.mision_cumplida is None:
            ev.mision_cumplida = False
            ev.datos_mision["motivo"] = ("impacto antes de la SOI lunar"
                                         if self._lunar_estado in ("quema_tli", "translunar")
                                         else "impacto antes de completar el aparcamiento")

    def _presion_ambiente(self) -> float:
        """Presión [Pa] a la altitud actual, o 0 con empuje_vs_altitud apagado."""
        if not self.config.empuje_vs_altitud:
            return 0.0
        return presion_atmosfera(self.estado.altitud)

    def _calcular_empuje(self, etapa: Optional[Etapa],
                         etapa_encendida: bool) -> tuple[float, float]:
        """(empuje [N], flujo másico [kg/s]) de este paso; fija est.empuje."""
        throttle = self._throttle()
        if etapa_encendida:
            etapa._fue_activada = True
            empuje, flujo_masico = etapa.empuje_y_flujo(
                throttle, self._presion_ambiente())
        else:
            empuje, flujo_masico = 0.0, 0.0
        self.estado.empuje = empuje
        return empuje, flujo_masico

    def _direccion_empuje(self, empuje: float, masa: float) -> Vec2:
        """Vector unitario de empuje según el modo de guiado."""
        est = self.estado
        if est.modo_guiado == "gravity_turn":
            vgx, vgy, _, _ = self._vel_guia()
            return calcular_guiado(est.px, est.py, vgx, vgy,
                                   math.hypot(vgx, vgy), est.modo_guiado)
        if (est.modo_guiado == "circularizacion"
                and (self.es_tripulada_bcb() or self.es_deorbit() or self._lunar)
                and empuje > 0.0):
            return self._empuje_circ_mantener_altura(empuje, masa)
        return calcular_guiado(est.px, est.py, est.vx, est.vy,
                               est.speed, est.modo_guiado)

    def _calcular_arrastre(self, v_aire_rel_x: float,
                           v_aire_rel_y: float) -> Vec2:
        """Fuerza de arrastre (x, y) respecto al aire que co-rota con la
        Tierra; fija est.arrastre_mag."""
        est = self.estado
        arrastre_x, arrastre_y, est.arrastre_mag = fuerza_arrastre(
            est.altitud, v_aire_rel_x, v_aire_rel_y,
            self.Cd_actual(), self.area_frontal(),
            mach=self.config.cd_vs_mach,
        )
        return arrastre_x, arrastre_y

    def _registrar_max_q(self, v_aire_rel_x: float, v_aire_rel_y: float) -> None:
        """Max-Q (con velocidad relativa al aire)."""
        est = self.estado
        v_rel = math.hypot(v_aire_rel_x, v_aire_rel_y)
        q = 0.5 * densidad_atmosfera(est.altitud) * v_rel ** 2
        if q > self.eventos.datos_max_q["q"]:
            self.eventos.datos_max_q = {
                "q": q, "altitud": est.altitud,
                "t": est.t, "speed": est.speed,
            }

    def _integrar_tramo(self, y0, t0: float, dur: float, fuerzas: tuple,
                        empuje: float) -> Optional[Estado4]:
        """Avanza px,py,vx,vy `dur` s: RK4 en coast libre, RK45 en el resto.
        Devuelve el estado final, o None si RK45 no convergió (11.21.0:
        antes se tomaba `sol.y[:, -1]` aunque `sol.success` fuera False,
        un estado de un tiempo anterior al pedido)."""
        if dur <= 0.0:
            return tuple(y0)
        if isinstance(fuerzas, LeyV12):
            return self._integrar_tramo_v12(y0, t0, dur, fuerzas, empuje)
        if self._en_coast_libre(empuje):
            return self._integrar_rk4(list(y0), t0, dur, fuerzas)
        sol = solve_ivp(
            derivadas_marco_tierra,
            (t0, t0 + dur),
            list(y0),
            args=fuerzas,
            method="RK45", rtol=1e-6, atol=1e-8,
        )
        if not sol.success:
            self._msg_fallo_integrador = str(sol.message)
            return None
        return tuple(float(sol.y[i, -1]) for i in range(4))

    def _integrar_tramo_v12(self, y0, t0: float, dur: float, ley: "LeyV12",
                            empuje: float):
        """(12.0.0) Tramo con estado de 6 componentes. Coast libre: RK4 como
        antes (arrastre evaluado al inicio; despreciable por encima de
        ALT_RK4_MIN); resto: RK45 sobre `derivadas_v12`."""
        if self._en_coast_libre(empuje):
            y4 = self._integrar_rk4(list(y0[:4]), t0, dur,
                                    ley.fuerzas_congeladas(y0))
            # empuje == 0 aquí (condición de coast libre): no se quema nada,
            # ni siquiera con la etapa encendida y T <= 0 (ver derivadas_v12)
            return (*y4, y0[4], y0[5])
        sol = solve_ivp(derivadas_v12, (t0, t0 + dur), list(y0), args=(ley,),
                        method="RK45", rtol=1e-6,
                        atol=[1e-8, 1e-8, 1e-8, 1e-8, 1e-6, 1e-8])
        if not sol.success:
            self._msg_fallo_integrador = str(sol.message)
            return None
        return tuple(float(sol.y[i, -1]) for i in range(6))

    def _avanzar_con_empuje(self, y0, t0: float, dt: float, t_emp: float,
                            fuerzas: tuple, flujo: float,
                            empuje: float) -> Optional[Estado4]:
        """Un paso de `dt` con empuje solo durante los primeros `t_emp` s
        y coast el resto (el arrastre y la marea se mantienen como en el
        resto del paso; la masa baja lo quemado). Con t_emp == dt es la
        integración de siempre, un solo tramo."""
        if t_emp >= dt - 1e-12:
            return self._integrar_tramo(y0, t0, dt, fuerzas, empuje)
        y1 = self._integrar_tramo(y0, t0, t_emp, fuerzas, empuje)
        if y1 is None:
            return None
        if isinstance(fuerzas, LeyV12):     # cola sin empuje; la masa ya bajó
            return self._integrar_tramo(y1, t0 + t_emp, dt - t_emp,
                                        fuerzas.apagada(), 0.0)
        masa_cola = fuerzas[4] - flujo * t_emp
        cola = (0.0, 0.0, fuerzas[2], fuerzas[3], masa_cola,
                fuerzas[5], fuerzas[6])
        return self._integrar_tramo(y1, t0 + t_emp, dt - t_emp, cola, 0.0)

    def _fallo_integrador(self, ruta: str) -> None:
        """Termina la simulación de forma controlada si el integrador no
        convergió: el estado queda en el último paso válido (ni t ni el
        combustible avanzan) y el motivo se guarda en el registro."""
        est = self.estado
        msg = self._msg_fallo_integrador
        self.eventos.fallo_integrador = f"{ruta}: {msg} (t={est.t:.1f} s)"
        est.simulacion_terminada = True
        print(f"⚠ FALLO DEL INTEGRADOR ({ruta}) en t={est.t:.1f} s: {msg}. "
              f"Simulación detenida en el último estado válido.")

    def _corte_orbita_ok(self, orb: dict) -> bool:
        """Periapsis >= objetivo y, en perfiles que prometen órbita casi
        circular (tripulada/bcb, deorbit, gto), también ecc < ecc_max."""
        ok = orb["rp"] >= self.rp_objetivo_circ()
        if self.exige_ecc_en_corte():
            ok = ok and orb["ecc"] < self.ecc_max_efectiva()
        return ok

    def _corte_por_estado(self, y) -> bool:
        """¿La lógica de misión cortaría el motor con la nave en `y`?
        Espejo SIN efectos de las condiciones de corte de
        `_actualizar_mision` (mismo orden: deorbit → apogeo B-C-B →
        suborbital → alta energía → periapsis de circularización). Sirve
        para localizar el instante del corte dentro del paso.

        Asimetría deliberada: escape corta por ε >= 0 sin exigir v_r >= 0,
        porque la condición de escape es energética e independiente de la
        dirección radial. TLI sí exige ascenso en su corte nominal: debe
        seleccionar la rama de transferencia hacia la Luna, no una rama
        descendente. `_margen_corte` conserva exactamente estos criterios."""
        est = self.estado
        cfg = self.config
        px, py, vx, vy = y[:4]
        r = math.hypot(px, py)
        if r < 1.0:
            return False
        orb = elementos_orbitales(px, py, vx, vy)
        if self._lunar and est.fase == "quema_tli":
            return orb["eps"] >= self._lunar_plan["eps_obj"]
        if self.es_deorbit() and self._deorbit_activo:
            return est.fase == "deorbit" and orb["rp"] <= cfg.DEORBIT_RP_OBJETIVO
        subiendo = (vx * px + vy * py) / r >= 0.0
        ra = orb["ra"]
        apogeo = math.isfinite(ra) and ra >= cfg.APOGEO_OBJETIVO
        if not est.cutoff_por_apogeo:
            if self.usa_bcb() and not est.segunda_quema_realizada:
                if apogeo and subiendo:
                    return True
            if (cfg.modo_mision == "suborbital"
                    and cfg.suborbital_cutoff_por_apogeo
                    and apogeo and subiendo):
                return True
            if self.es_alta_energia():
                escape = bool(orb["eps"] >= 0.0)
                if cfg.modo_mision == "escape":
                    # En escape el criterio es ε >= 0, independientemente del signo de v_r.
                    if escape:
                        return True
                elif (escape or apogeo) and subiendo:
                    # TLI: se conserva el requisito ascendente del corte nominal.
                    return True
        if est.fase == "circularizacion":
            return self._corte_orbita_ok(orb)
        return False

    def _margen_corte(self, y, desfase: float = 0.0) -> float:
        """(12.1.0) Función escalar continua del estado con
        margen >= 0 <=> `_corte_por_estado(y)`: las condiciones en Y son un
        min, las alternativas un max. Su cruce de negativo a positivo es el
        evento de corte de solve_ivp. Unidades mezcladas (m, m/s, J/kg): solo
        importa el signo y la continuidad. `desfase` (>= 0) se resta de los
        términos de los elementos orbitales (no de vr >= 0, que es una
        condición de dirección): exige que se cumplan con ese margen.
        Frontera de ecc: `_corte_orbita_ok` pide ecc < ecc_max (estricto) y
        aquí margen = (ecc_max - ecc)·R_T es 0 en ecc == ecc_max; como el
        evento exige un cruce de negativo a positivo (y el desfase mínimo
        1e-6 desplaza la frontera hacia dentro), la diferencia es de medida
        cero y no cambia ningún comportamiento."""
        est = self.estado
        cfg = self.config
        px, py, vx, vy = y[0], y[1], y[2], y[3]
        r = math.hypot(px, py)
        if r < 1.0:
            return -1.0
        orb = elementos_orbitales(px, py, vx, vy)
        if self._lunar and est.fase == "quema_tli":
            return self._sano(orb["eps"] - self._lunar_plan["eps_obj"] - desfase)
        if self.es_deorbit() and self._deorbit_activo:
            if est.fase != "deorbit":
                return -1.0
            return self._sano(cfg.DEORBIT_RP_OBJETIVO - orb["rp"] - desfase)
        vr = (vx * px + vy * py) / r              # >= 0 <=> subiendo
        ra = orb["ra"]
        if math.isfinite(ra) and math.isfinite(cfg.APOGEO_OBJETIVO):
            m_ap = ra - cfg.APOGEO_OBJETIVO - desfase
        else:
            m_ap = -1.0e9
        cand = []
        if not est.cutoff_por_apogeo:
            if self.usa_bcb() and not est.segunda_quema_realizada:
                cand.append(min(m_ap, vr))
            if (cfg.modo_mision == "suborbital"
                    and cfg.suborbital_cutoff_por_apogeo):
                cand.append(min(m_ap, vr))
            if self.es_alta_energia():
                m_esc = orb["eps"] - desfase
                if cfg.modo_mision == "escape":
                    # Escape se corta por energía; no debe retrasarse hasta v_r >= 0.
                    cand.append(m_esc)
                else:
                    # TLI conserva (energía o apogeo) AND ascenso.
                    cand.append(min(max(m_esc, m_ap), vr))
        if est.fase == "circularizacion":
            m = orb["rp"] - self.rp_objetivo_circ() - desfase
            if self.exige_ecc_en_corte():
                m = min(m, (self.ecc_max_efectiva() - orb["ecc"]) * R_T - desfase)
            cand.append(m)
        if not cand:
            return -1.0
        return self._sano(max(cand))

    @staticmethod
    def _sano(x: float) -> float:
        return -1.0 if x != x else float(x)

    def _tramo_v12_eventos(self, y0, t0: float, dur: float, ley: "LeyV12",
                           empuje: float, corte: bool, suelo: bool,
                           eps_corte: float = 1e-6):
        """(12.1.0) Como `_integrar_tramo_v12` pero con eventos terminales:
        corte (margen: - -> +) y suelo (altitud: + -> -). Devuelve
        (estado, t, tipo) con tipo None | "corte" | "suelo", o None si el
        integrador falla. Sin eventos aplicables (coast libre con RK4 o sin
        corte ni suelo) integra como siempre."""
        if dur <= 0.0:
            return tuple(y0), t0, None
        if self._en_coast_libre(empuje) or not (corte or suelo):
            y = self._integrar_tramo_v12(y0, t0, dur, ley, empuje)
            return (None if y is None else (y, t0 + dur, None))
        eventos, tipos = [], []
        if corte:
            def ev_corte(t, y, ley_):
                # Desfase eps_corte: la raíz cae del lado VERDADERO de la
                # condición, con margen para que el coast del resto del paso
                # (arrastre, que erosiona p. ej. el periapsis) no la devuelva
                # al lado falso antes de que la lógica de misión la mire.
                return self._margen_corte(y, eps_corte)
            ev_corte.terminal, ev_corte.direction = True, 1.0
            eventos.append(ev_corte); tipos.append("corte")
        if suelo:
            def ev_suelo(t, y, ley_):
                return math.hypot(y[0], y[1]) - R_T
            ev_suelo.terminal, ev_suelo.direction = True, -1.0
            eventos.append(ev_suelo); tipos.append("suelo")
        sol = solve_ivp(derivadas_v12, (t0, t0 + dur), list(y0), args=(ley,),
                        method="RK45", rtol=1e-6,
                        atol=[1e-8, 1e-8, 1e-8, 1e-8, 1e-6, 1e-8],
                        events=eventos)
        if not sol.success:
            self._msg_fallo_integrador = str(sol.message)
            return None
        for i, tipo in enumerate(tipos):
            if len(sol.t_events[i]) > 0:
                y = tuple(float(v) for v in sol.y_events[i][0])
                return y, float(sol.t_events[i][0]), tipo
        return tuple(float(sol.y[i, -1]) for i in range(6)), t0 + dur, None

    def _avanzar_v12_eventos(self, y0, t0: float, dt: float, t_emp: float,
                             ley: "LeyV12", empuje: float,
                             buscar_corte: bool, suelo: bool,
                             eps_corte: float = 1e-6):
        """(12.1.0) Un paso con empuje los primeros `t_emp` s y coast el
        resto, con eventos. Devuelve (estado final, t_emp efectivo, t del
        impacto o None, tipo) o None si el integrador falló. Si hay corte,
        t_emp efectivo es el instante del corte y el resto del paso es coast
        desde el estado exacto del corte."""
        lleno = t_emp >= dt - 1e-12
        dur1 = dt if lleno else t_emp
        r1 = self._tramo_v12_eventos(y0, t0, dur1, ley, empuje,
                                     buscar_corte, suelo, eps_corte)
        if r1 is None:
            return None
        y1, t1, k1 = r1
        if k1 == "suelo":
            return y1, t1 - t0, t1, "suelo"
        if k1 is None and lleno:
            return y1, dt, None, None
        t_emp_ef = (t1 - t0) if k1 == "corte" else t_emp
        resto = dt - t_emp_ef
        if resto <= 1e-12:
            return y1, t_emp_ef, None, k1
        r2 = self._tramo_v12_eventos(y1, t0 + t_emp_ef, resto, ley.apagada(),
                                     0.0, False, suelo)
        if r2 is None:
            return None
        y2, t2, k2 = r2
        if k2 == "suelo":
            return y2, t_emp_ef, t2, "suelo"
        return y2, t_emp_ef, None, k1

    def _localizar_corte(self, y0, t0: float, dt: float, t_max: float,
                         fuerzas: tuple, flujo: float, empuje: float,
                         y_max: Estado4) -> tuple[float, Estado4]:
        """Bisección del instante (0, t_max] en que se cumple la condición
        de corte con empuje hasta ese instante y coast el resto del paso
        (lo que verá la lógica de misión al cerrar el paso). Devuelve
        (t_emp, estado final del paso). Si el integrador falla en una
        evaluación se queda con el último valor válido."""
        lo, hi, y_hi = 0.0, t_max, y_max
        for _ in range(_CORTE_MAX_ITER):
            if hi - lo < _CORTE_TOL_T:
                break
            mid = 0.5 * (lo + hi)
            ym = self._avanzar_con_empuje(y0, t0, dt, mid, fuerzas, flujo,
                                          empuje)
            if ym is None:
                break
            if self._corte_por_estado(ym):
                hi, y_hi = mid, ym
            else:
                lo = mid
        return hi, y_hi

    @staticmethod
    def _cruce_suelo(y0, y1, dt: float) -> tuple[Estado4, float]:
        """Cruce del suelo dentro del paso: interpolación cúbica de Hermite
        de la posición (con las velocidades de ambos extremos) y bisección
        de |p(s)| = R_T. Devuelve (estado en el cruce, fracción s del paso)."""
        def antes(s):
            e = Simulacion._hermite_estado(y0, y1, dt, s)
            return math.hypot(e[0], e[1]) > R_T
        s = _biseccion_s(antes, 60)
        return Simulacion._hermite_estado(y0, y1, dt, s), s

    def _paso_fisica_espacio_profundo(self) -> None:
        """
        Propagación en las zonas "crucero" (CRTBP rotante) y "luna"
        (inercial centrado en la Luna). Sin atmósfera (no aplica en
        espacio profundo) y, por ahora, balística pura: todavía no
        hay fases de misión propias para crucero/inserción lunar.
        """
        est = self.estado
        y0 = [est.px, est.py, est.vx, est.vy]
        est.empuje = 0.0
        est.arrastre_mag = 0.0

        if est.marco == "crucero":
            ruta = "crucero (CRTBP)"
            sol = solve_ivp(
                derivadas_crtbp_rotante,
                (est.t, est.t + est.dt), y0,
                args=(est.crucero_Omega, est.crucero_a),
                method="RK45", rtol=1e-9, atol=1e-6,   # (12.10.1) con |y| ~ 1e8 m manda rtol·|y|, no atol
            )
        else:  # "luna"
            ruta = "zona lunar"
            perturbacion_x, perturbacion_y = self._aceleracion_terceros_cuerpos_luna()
            sol = solve_ivp(
                derivadas_marco_luna,
                (est.t, est.t + est.dt), y0,
                args=(perturbacion_x, perturbacion_y),
                method="RK45", rtol=1e-8, atol=1e-6,
            )
        if not sol.success:
            self._msg_fallo_integrador = str(sol.message)
            self._fallo_integrador(ruta)
            return
        est.px, est.py, est.vx, est.vy = (float(sol.y[i, -1]) for i in range(4))
        est.speed = math.hypot(est.vx, est.vy)
        # altitud se usa en el HUD, el drawer y el panel síncrono como
        # "altura sobre la Tierra" — sin esto quedaba congelada en su
        # último valor de la zona "tierra" al entrar en crucero/luna,
        # incoherente con la posición (que sí se actualiza vía
        # posicion_display_eci).
        est.t += est.dt  # avanzar t ANTES: la Luna se evalúa en t+dt
        px_d, py_d = self.posicion_display_eci()
        est.altitud = float(math.hypot(px_d, py_d) - R_T)

    def _en_coast_libre(self, empuje: float) -> bool:
        """Zona de dinámica suave: marco Tierra, motor apagado en este
        paso y por encima de la atmósfera densa. Ahí RK4 de paso fijo
        iguala a RK45 (rtol 1e-6) con una fracción del costo. No mira
        la fase: cubre coasting, coasting_interetapa y cualquier coast
        con el motor apagado."""
        cfg = self.config
        est = self.estado
        return (cfg.integrador == "hibrido"
                and est.marco == "tierra"
                and empuje == 0.0
                and est.altitud > cfg.ALT_RK4_MIN)

    def _integrar_rk4(self, y0: list, t: float, dt: float,
                      fuerzas: tuple) -> Estado4:
        """RK4 clásico sobre `derivadas_marco_tierra` con las fuerzas
        externas constantes en el paso (igual que con solve_ivp). Si dt
        supera RK4_PASO_MAX se parte en subpasos iguales (dt = 5 s en
        coast de alta energía: 5 x 1 s, ~1 mm/vuelta GTO, contra 3 cm
        con un solo paso de 5 s)."""
        n_subpasos = max(1, int(math.ceil(dt / self.config.RK4_PASO_MAX - 1e-12)))
        px, py, vx, vy = y0
        return _rk4_kernel(px, py, vx, vy, dt / n_subpasos, n_subpasos,
                           *fuerzas, MU_T)

    # ========================================================
    # GUIADO
    # ========================================================
    def _empuje_circ_mantener_altura(self, empuje: float, masa: float) -> Vec2:
        est = self.estado
        return direccion_empuje_mantener_altura(
            est.px, est.py, est.vx, est.vy, empuje, masa)

    def _vel_guia(self) -> tuple[float, float, float, float]:
        """Velocidad para el giro por gravedad: respecto al aire en la
        atmósfera, inercial en el vacío (ver `velocidad_de_guiado`)."""
        est = self.estado
        return velocidad_de_guiado(est.px, est.py, est.vx, est.vy,
                                   est.altitud, self._Omega_tierra)

    def _aplicar_kick(self) -> None:
        est = self.estado
        r = math.hypot(est.px, est.py)
        kick = math.radians(self.config.angulo_kick)
        perp_x = est.py / r
        perp_y = -est.px / r
        # El kick se aplica a la velocidad de guiado (relativa al aire
        # con rotación activa; con rotación apagada es la inercial).
        vgx, vgy, aire_x, aire_y = self._vel_guia()
        rapidez_guia = math.hypot(vgx, vgy)
        dv = rapidez_guia * math.sin(kick)
        vgx += perp_x * dv
        vgy += perp_y * dv
        rapidez_nueva = math.hypot(vgx, vgy)
        if rapidez_nueva > 0.0:
            vgx = vgx / rapidez_nueva * rapidez_guia
            vgy = vgy / rapidez_nueva * rapidez_guia
        est.vx = vgx + aire_x
        est.vy = vgy + aire_y

    # ========================================================
    # MÁQUINA DE ESTADOS DE LA MISIÓN
    # ========================================================
    def _fijar_fase(self, fase: str, modo: Optional[str] = None) -> None:
        self.estado.fase = fase
        self.estado.modo_guiado = (
            modo if modo is not None else _MODO_POR_FASE.get(fase, "vertical")
        )

    def _encender_etapa(self, etapa: Etapa) -> None:
        """Activa el motor de `etapa` y recuerda la rapidez de inicio (Δv)."""
        etapa.activa = True
        etapa.v_inicio = self.estado.speed

    def _actualizar_mision(self) -> None:
        """Un paso de la lógica de misión. Orden (importa):
        corrección de altitud crítica → deorbit → B-C-B → suborbital →
        alta energía → fase actual. Los pasos que cambian de fase y
        "consumen" el paso terminan aquí mismo (return anticipado)."""
        ctx = self._construir_contexto_mision()
        self._aplicar_correccion_altitud_critica(ctx)

        if self.es_deorbit() and self._actualizar_deorbit(ctx):
            self._cerrar_paso_mision(ctx)
            return

        if self.usa_bcb() and not self.estado.segunda_quema_realizada:
            self._gestionar_burn_coast_burn(ctx)

        if self._gestionar_corte_suborbital(ctx):
            self._cerrar_paso_mision(ctx)
            return

        if self.es_alta_energia():
            if self._gestionar_corte_alta_energia(ctx):
                self._cerrar_paso_mision(ctx)
                return
            self._registrar_combustible_insuficiente(ctx)

        self._ejecutar_fase_actual(ctx)
        self._cerrar_paso_mision(ctx)

    def _cerrar_paso_mision(self, ctx: ContextoMision) -> None:
        self.estado.v_radial_prev = ctx.v_radial

    def _construir_contexto_mision(self) -> ContextoMision:
        est = self.estado
        cfg = self.config
        r = math.hypot(est.px, est.py)
        v_radial = (est.vx * est.px + est.vy * est.py) / r

        etapa = self.etapa_actual()
        tiene_combustible = self._etapa_puede_encender(etapa)

        # En suborbital no existe periapsis/altura "segura": la caída
        # es el final esperado del vuelo, no un peligro a corregir.
        en_peligro = (
            cfg.modo_mision != "suborbital"
            and not self._deorbit_activo
            and est.altitud < cfg.ALTURA_SEGURIDAD
            and v_radial < -30.0
            and tiene_combustible
            and est.fase not in ("impacto", "vertical")
        )

        elementos = elementos_orbitales(est.px, est.py, est.vx, est.vy)
        apoapsis = elementos["ra"]
        apogeo_alcanzado = (math.isfinite(apoapsis)
                            and apoapsis >= cfg.APOGEO_OBJETIVO)
        return ContextoMision(
            v_radial=v_radial,
            etapa=etapa,
            tiene_combustible=tiene_combustible,
            en_peligro=en_peligro,
            elementos=elementos,
            apogeo_alcanzado=apogeo_alcanzado,
            subiendo=v_radial >= 0.0,
            paso_por_apogeo=est.v_radial_prev >= 0.0 and v_radial < 0.0,
        )

    def _aplicar_correccion_altitud_critica(self, ctx: ContextoMision) -> None:
        est = self.estado
        cfg = self.config
        if ctx.en_peligro:
            est.modo_guiado = "prograde_seguro"
            if not est.advertencia_seguridad:
                print(f" Altitud crítica ({est.altitud:.0f} m). Corrigiendo rumbo.")
                est.advertencia_seguridad = True
        elif est.advertencia_seguridad and est.altitud > cfg.ALTURA_SEGURIDAD * 1.1:
            est.advertencia_seguridad = False

    # --- Burn-Coast-Burn (satelital, gto, tripulada/bcb, deorbit) ---
    def _gestionar_burn_coast_burn(self, ctx: ContextoMision) -> None:
        """Secuencia de UNA sola vez antes de la quema final de
        circularización (corte por apogeo + espera + segunda quema).
        Una vez que la segunda quema ya arrancó
        (segunda_quema_realizada) no debe volver a intervenir: si lo
        hiciera, un descenso pasajero del apogeo calculado (p. ej. por
        arrastre atmosférico en un tramo sin empuje) podría resetear
        "cutoff_por_apogeo" y disparar OTRO corte, interrumpiendo la
        circularización real antes de que el periapsis llegue a un
        valor seguro. De ahí en adelante, el único corte válido es el
        de periapsis real, en la fase "circularizacion"."""
        est = self.estado
        self._cortar_por_apogeo_bcb(ctx)
        self._iniciar_segunda_quema(ctx)
        if not ctx.apogeo_alcanzado and est.cutoff_por_apogeo:
            est.cutoff_por_apogeo = False

    def _cortar_por_apogeo_bcb(self, ctx: ContextoMision) -> None:
        """Corta en cuanto el apogeo proyectado alcanza el objetivo y
        todavía se está ascendiendo. NO se exige periapsis
        no-suborbital aquí: el periapsis solo deja de ser negativo
        cuando ya casi se está en órbita, mucho después de superar el
        apogeo objetivo, así que exigirlo en este primer corte impide
        que se corte jamás (el motor quemaría hasta agotar combustible
        hasta una trayectoria hiperbólica). La órbita segura
        (periapsis >= ALTURA_SEGURIDAD) se exige en el corte real."""
        est = self.estado
        etapa = ctx.etapa
        puede_cortar = ctx.apogeo_alcanzado and ctx.subiendo
        if not (puede_cortar and ctx.tiene_combustible
                and not est.cutoff_por_apogeo
                and etapa is not None and etapa.activa):
            return
        orb = ctx.elementos
        est.cutoff_por_apogeo = True
        est.esperando_apogeo_fisico = True
        etapa.activa = False
        print(f"✂ CUTOFF POR APOGEO: ra = {orb['ra'] / 1000.0:.1f} km")
        if self.config.modo_mision == "gto":
            self.eventos.datos_mision.update({
                "t_cutoff1": est.t, "ra_cutoff1": orb["ra"],
                "rp_cutoff1": orb["rp"], "ecc_cutoff1": orb["ecc"],
            })
        print("⏳ Esperando apogeo físico para segunda quema...")
        print(f"💾 Combustible preservado: {etapa.m_comb_restante:.0f} kg")

    def _iniciar_segunda_quema(self, ctx: ContextoMision) -> None:
        est = self.estado
        if not (est.esperando_apogeo_fisico and ctx.paso_por_apogeo
                and ctx.tiene_combustible and not est.segunda_quema_realizada):
            return
        etapa = ctx.etapa
        est.segunda_quema_realizada = True
        est.esperando_apogeo_fisico = False
        self._encender_etapa(etapa)
        self._fijar_fase("circularizacion")
        if self.config.modo_mision == "gto":
            self.eventos.datos_mision.update({
                "t_burn2": est.t, "alt_burn2": est.altitud,
                "comb_burn2": etapa.m_comb_restante,
            })
        print("🔥 SEGUNDA QUEMA ACTIVADA en apogeo físico")
        print(f"📍 Altitud apogeo: {est.altitud / 1000.0:.1f} km")
        print(f"🎯 Objetivo: circularizar a periapsis >= "
              f"{self.rp_objetivo_circ() / 1000.0:.0f} km")

    # --- Suborbital ---
    def _gestionar_corte_suborbital(self, ctx: ContextoMision) -> bool:
        """Seguimiento de h_max y corte opcional por apogeo. Devuelve
        True si cambió de fase (la rama "coasting" correrá desde el
        siguiente paso, no en este mismo frame)."""
        cfg = self.config
        est = self.estado
        etapa = ctx.etapa
        if cfg.modo_mision != "suborbital":
            return False
        self.eventos.alt_max = max(self.eventos.alt_max, est.altitud)
        if (cfg.suborbital_cutoff_por_apogeo
                and ctx.apogeo_alcanzado and ctx.subiendo
                and ctx.tiene_combustible
                and not est.cutoff_por_apogeo
                and etapa is not None and etapa.activa):
            est.cutoff_por_apogeo = True
            etapa.activa = False
            self._fijar_fase("coasting")
            print(f"✂ CUTOFF SUBORBITAL: ra = {ctx.elementos['ra'] / 1000.0:.1f} km")
            return True
        return False

    # --- tli / escape ---
    def _gestionar_corte_alta_energia(self, ctx: ContextoMision) -> bool:
        """Corte único por apogeo / energía. True si cortó (cambia de fase)."""
        cfg = self.config
        est = self.estado
        etapa = ctx.etapa
        orb = ctx.elementos
        escape = bool(orb["eps"] >= 0.0)
        if cfg.modo_mision == "escape":
            alcanzado = escape
        else:
            alcanzado = escape or bool(ctx.apogeo_alcanzado)
        direccion_ok = (True if cfg.modo_mision == "escape" else ctx.subiendo)
        if not (alcanzado and direccion_ok and ctx.tiene_combustible
                and not est.cutoff_por_apogeo
                and etapa is not None and etapa.activa):
            return False

        est.cutoff_por_apogeo = True
        etapa.activa = False
        self._fijar_fase("coasting")
        if cfg.modo_mision == "escape":
            resultado = "escape"
        else:
            resultado = "escape" if escape else "llego_luna"
        esperado = "escape" if cfg.modo_mision == "escape" else "llego_luna"
        if cfg.modo_mision == "tli" and resultado == "llego_luna":
            # «llego_luna» es solo el resultado del CORTE (apogeo
            # proyectado >= órbita lunar). La misión se cumple al ENTRAR a
            # la SOI lunar; mientras tanto queda pendiente (None) y falla
            # por T_MAX_ALTA_ENERGIA o por impacto (11.21.0).
            self.eventos.mision_cumplida = None
        else:
            self.eventos.mision_cumplida = bool(resultado == esperado)
        self.eventos.datos_mision.update({
            # `t_cutoff` = est.t al cierre del paso en que se detectó el corte: hasta un paso
            # (DT_COAST_ALTA_ENERGIA / dt base) después del apagado exacto (12.10.1: aclarado)
            "resultado": resultado, "t_cutoff": est.t,
            "ra_cutoff": orb["ra"], "rp_cutoff": orb["rp"],
            "ecc_cutoff": orb["ecc"],
            "comb_restante": etapa.m_comb_restante,
        })
        print(f"✂ CUTOFF ALTA ENERGÍA ({cfg.modo_mision}): "
              f"resultado={resultado}, ra={orb['ra'] / 1000.0:.1f} km, "
              f"ecc={orb['ecc']:.4f}")
        return True

    def _registrar_combustible_insuficiente(self, ctx: ContextoMision) -> None:
        """tli/escape: sin combustible en ninguna etapa y sin haber
        alcanzado el objetivo → "insuficiente"."""
        if (self.eventos.mision_cumplida is None
                and "resultado" not in self.eventos.datos_mision
                and not self._hay_combustible_en_alguna_etapa()):
            est = self.estado
            orb = ctx.elementos
            self.eventos.mision_cumplida = False
            self.eventos.datos_mision.update({
                "resultado": "insuficiente", "t_cutoff": est.t,
                "ra_cutoff": orb["ra"], "rp_cutoff": orb["rp"],
                "ecc_cutoff": orb["ecc"], "comb_restante": 0.0,
            })

    # --- Fases ---
    def _ejecutar_fase_actual(self, ctx: ContextoMision) -> None:
        manejador = {
            "vertical":            self._fase_vertical,
            "gravity_turn":        self._fase_gravity_turn,
            "coasting_interetapa": self._fase_coasting_interetapa,
            "circularizacion":     self._fase_circularizacion,
            "coasting":            self._fase_coasting,
            "espera_tli":          self._fase_espera_tli,
            "quema_tli":           self._fase_quema_tli,
            "translunar":          self._fase_translunar,
            "quema_loi":           self._fase_quema_loi,
            "orbita_lunar":        self._fase_orbita_lunar,
        }.get(self.estado.fase)
        if manejador is not None:
            manejador(ctx)

    def _fase_vertical(self, ctx: ContextoMision) -> None:
        est = self.estado
        if not ctx.en_peligro:
            est.modo_guiado = "vertical"
        if est.altitud >= self.config.alt_inicio_giro:
            self._fijar_fase("gravity_turn")
            self._aplicar_kick()

    def _fase_gravity_turn(self, ctx: ContextoMision) -> None:
        if not ctx.en_peligro:
            self.estado.modo_guiado = "gravity_turn"
        if not self._hay_combustible_en_alguna_etapa() and not ctx.en_peligro:
            self._fijar_fase("coasting")
            self.eventos.max_q_registrado = True

    def _fase_coasting_interetapa(self, ctx: ContextoMision) -> None:
        if not ctx.en_peligro:
            self.estado.modo_guiado = "prograde"
        self._logica_coasting_interetapa(ctx)

    def _logica_coasting_interetapa(self, ctx: ContextoMision) -> None:
        """Decide cuándo reencender la etapa tras una separación en
        la que no se encendió de inmediato."""
        est = self.estado
        cfg = self.config
        etapa = ctx.etapa
        orb = ctx.elementos
        periapsis_bajo_tierra = orb["rp"] < 0.0
        ra_finita = orb["es_elipse"]
        altura_al_apogeo = (orb["ra"] - est.altitud) if ra_finita else float("inf")
        umbral_m = (max(orb["ra"] * (cfg.umbral_encendido / 100.0), 5000.0)
                    if ra_finita else 0.0)
        ventana_hohmann = (
            ra_finita and altura_al_apogeo <= umbral_m and ctx.v_radial >= 0.0
        )
        no_esperando_segunda = not est.esperando_apogeo_fisico

        if (ctx.tiene_combustible and no_esperando_segunda
                and (periapsis_bajo_tierra or ventana_hohmann
                     or ctx.paso_por_apogeo or ctx.en_peligro)):
            self._encender_etapa(etapa)
            if ctx.en_peligro:
                self._fijar_fase("gravity_turn", "prograde_seguro")
            else:
                self._fijar_fase("circularizacion")

        if not ctx.tiene_combustible and ctx.paso_por_apogeo:
            self._fijar_fase("coasting")

    def _fase_circularizacion(self, ctx: ContextoMision) -> None:
        """Válido para satelital, tripulada/bcb (segunda quema) y
        tripulada/directo (quema única continua). El corte real se
        decide SOLO por periapsis (más ecc donde corresponde): no se
        corta por velocidad/energía como atajo, porque con velocidad
        radial residual "rapidez ~ v_circ local" no garantiza que el
        periapsis ya sea seguro."""
        est = self.estado
        etapa = ctx.etapa
        if not ctx.en_peligro:
            est.modo_guiado = self._modo_guiado_circularizacion(ctx)

        sin_combustible = (
            (etapa is None) or (not etapa.activa)
            or (etapa.m_comb_restante <= 0.0)
        )
        periapsis_ok = self._corte_circularizacion_logrado(ctx)
        self._registrar_fallo_gto_sin_combustible(ctx, periapsis_ok)
        if sin_combustible or periapsis_ok:
            self._fijar_fase("coasting")
            if etapa is not None:
                etapa.activa = False

    def _modo_guiado_circularizacion(self, ctx: ContextoMision) -> str:
        """tripulada-directo llega aquí desde el separador, todavía con
        ángulo de vuelo alto. Empujar prógrado puro desde ahí dispara
        el apogeo (la energía se va casi toda a apogeo, poco a
        periapsis) y termina en una trayectoria hiperbólica sin
        combustible. Por eso se mantiene guiado tipo gravity-turn
        mientras el apogeo proyectado no alcance el objetivo, y solo
        entonces se pasa a empuje prógrado puro. El resto (satelital,
        bcb) ya llega con ángulo de vuelo casi cero: prógrado puro."""
        usa_ascenso_directo = (self.es_tripulada_directa()
                               and not ctx.apogeo_alcanzado)
        return "gravity_turn" if usa_ascenso_directo else "circularizacion"

    def _corte_circularizacion_logrado(self, ctx: ContextoMision) -> bool:
        """Periapsis >= objetivo y, en perfiles que prometen órbita casi
        circular (tripulada/bcb, deorbit, gto), también ecc < ecc_max
        (con rp = ALTURA_SEGURIDAD y ra = APOGEO_OBJETIVO la ecc. sigue
        en ~0.016, así que no basta el periapsis). La condición vive en
        `_corte_orbita_ok` (la comparte la localización del corte)."""
        return self._corte_orbita_ok(ctx.elementos)

    def _registrar_fallo_gto_sin_combustible(self, ctx: ContextoMision,
                                             periapsis_ok: bool) -> None:
        """gto: sin combustible en plena circularización y sin haber
        logrado periapsis/ecc → misión fallida (no basta con "etapa
        inactiva": eso también ocurre en el cutoff 1)."""
        est = self.estado
        etapa = ctx.etapa
        if (self.config.modo_mision == "gto" and not periapsis_ok
                and etapa is not None and etapa.m_comb_restante <= 0.0
                and est.segunda_quema_realizada
                and self.eventos.mision_cumplida is None):
            orb = ctx.elementos
            self.eventos.mision_cumplida = False
            self.eventos.datos_mision.update({
                "t": est.t, "rp": orb["rp"], "ra": orb["ra"],
                "ecc": orb["ecc"], "motivo": "sin combustible",
            })

    def _fase_coasting(self, ctx: ContextoMision) -> None:
        est = self.estado
        if not ctx.en_peligro:
            est.modo_guiado = "prograde"
        elif ctx.tiene_combustible:
            # _fijar_fase solo toca fase/modo_guiado: si no
            # reactivamos la etapa acá, queda en "gravity_turn"
            # con combustible pero el motor apagado, sin empuje,
            # hasta el final de la simulación (puede pasar en
            # bcb si tras el cutoff por apogeo la nave entra en
            # peligro antes de la segunda quema).
            self._encender_etapa(ctx.etapa)
            self._fijar_fase("gravity_turn", "prograde_seguro")
        if ctx.paso_por_apogeo and not self.eventos.apogeo_registrado:
            self.eventos.apogeo_registrado = True
            self.eventos.datos_apogeo = {
                "t": est.t, "alt": est.altitud, "v": est.speed,
            }
        if (self._lunar and self._lunar_estado == "ascenso"
                and self.eventos.orbita_detectada and not ctx.en_peligro
                and not (ctx.etapa is not None and ctx.etapa.activa)):
            self._lunar_iniciar_aparcamiento(ctx)

    # ========================================================
    # BURNOUT Y SEPARACIÓN DE ETAPAS
    # ========================================================
    def _verificar_burnout(self) -> None:
        est = self.estado
        etapa = self.etapa_actual()
        if etapa is None or etapa.separada or etapa.burnout_registrado:
            return
        if not etapa._fue_activada:
            return
        if etapa.m_comb_restante > 0.0:
            return

        self._registrar_datos_burnout(etapa)

        if est.etapa_actual + 1 < len(self.etapas):
            self._separar_etapa(etapa)
            self._encender_siguiente_etapa_segun_mision()
            self.eventos.eventos_etapas.append(
                {"tipo": "separacion", "etapa": est.etapa_actual - 1, "t": est.t}
            )
        else:
            etapa.activa = False

    def _registrar_datos_burnout(self, etapa: Etapa) -> None:
        est = self.estado
        px, py, vx, vy = est.px, est.py, est.vx, est.vy
        r_burnout = math.hypot(px, py)
        rx, ry = px / r_burnout, py / r_burnout
        v_radial = vx * rx + vy * ry
        v_tangencial = math.sqrt(max(0.0, est.speed ** 2 - v_radial ** 2))
        angulo_vuelo = math.degrees(math.atan2(v_radial, v_tangencial))
        delta_rapidez = est.speed - etapa.v_inicio   # NO es el Δv del motor
        orb = elementos_orbitales(px, py, vx, vy)

        etapa.burnout_registrado = True
        etapa.datos_burnout = {
            "t": est.t, "alt": est.altitud, "v": est.speed,
            "ang": round(angulo_vuelo, 1),
            "dv_prop": etapa.dv_prop,           # ∫T/m dt (Δv del motor)
            "delta_rapidez": delta_rapidez,     # rapidez final − inicial
            "ecc": round(orb["ecc"], 6),
            "rp": orb["rp"], "ra": orb["ra"],
        }
        self.eventos.eventos_etapas.append(
            {"tipo": "burnout", "etapa": est.etapa_actual, "t": est.t}
        )

    def _separar_etapa(self, etapa: Etapa) -> None:
        """Suelta la etapa agotada (queda como objeto en caída libre)
        y pasa a la siguiente."""
        est = self.estado
        self.objetos_caidos.append({
            "px": est.px, "py": est.py, "vx": est.vx, "vy": est.vy,
            "m": etapa.m_estr, "Cd": etapa.Cd, "r": etapa.r,
        })
        etapa.activa = False
        etapa.separada = True
        est.etapa_actual += 1

    def _fase_tras_separacion(self) -> Optional[str]:
        """Fase a la que pasa la nave al separar, o None si la nueva
        etapa NO enciende de inmediato (→ coasting_interetapa)."""
        cfg = self.config
        modo = cfg.modo_mision
        if modo in ("suborbital", "tli", "escape"):
            # Ascenso propulsado continuo: la siguiente etapa
            # enciende al separar y se mantiene el gravity turn
            # (casi vertical). Sin circularización.
            return "gravity_turn"
        if modo in ("tripulada", "gto", "lunar"):
            # En AMBOS perfiles tripulados (y en gto) la 2ª etapa enciende
            # al separar. La diferencia aparece después, en
            # _actualizar_mision:
            #   · bcb    → usa_bcb() hará cutoff por apogeo, coast, y
            #              segunda quema en el apogeo físico.
            #   · directo→ usa_bcb() es False: quema continua que solo
            #              corta por periapsis real ("circularizacion").
            return "circularizacion"
        if modo == "satelital" and self.estado.speed > cfg.VEL_MIN_SES:
            return "circularizacion"
        return None

    def _encender_siguiente_etapa_segun_mision(self) -> None:
        fase = self._fase_tras_separacion()
        if fase is None:
            self._fijar_fase("coasting_interetapa")
            return
        self._encender_etapa(self.etapas[self.estado.etapa_actual])
        self._fijar_fase(fase)

    # ========================================================
    # DETECCIÓN DE ÓRBITA ESTABLE
    # ========================================================
    def _verificar_orbita(self) -> None:
        if self.eventos.orbita_detectada:
            return
        if self.config.modo_mision in ("suborbital", "tli", "escape"):
            return   # vuelo balístico / alta energía: nunca se declara órbita
        est = self.estado
        if est.fase not in ("coasting", "circularizacion"):
            return
        orb = elementos_orbitales(est.px, est.py, est.vx, est.vy)

        if not (0.0 <= orb["ecc"] < self.ecc_max_efectiva()
                and est.altitud > 80000.0
                and orb["rp"] >= self._periapsis_minimo_orbita()):
            return
        if self._lunar and not orb["es_elipse"]:
            return                 # aparcamiento: elipse cerrada, no una trayectoria abierta
        self._declarar_orbita(orb)

    def _periapsis_minimo_orbita(self) -> float:
        """En tripulada, deorbit y satelital exigimos periapsis por encima de
        la altura de seguridad, no solo > 0: una excentricidad baja con un
        periapsis apenas sobre el suelo NO es una órbita estable, aunque
        orbitalmente sea válida (no reingresaría de inmediato, pero decaería
        en pocas vueltas). Desde 11.19.0 también en satelital (antes devolvía
        0 y se declaraba órbita con periapsis de decenas de km)."""
        cfg = self.config
        if cfg.modo_mision == "gto":
            return cfg.GTO_RP_FINAL
        if cfg.modo_mision in ("tripulada", "deorbit", "satelital", "lunar"):
            return cfg.ALTURA_SEGURIDAD
        return 0.0

    def _declarar_orbita(self, orb: dict) -> None:
        est = self.estado
        self.eventos.orbita_detectada = True
        self.eventos.datos_orbita = {
            "t": est.t, "alt": est.altitud, "v": est.speed,
            "ecc": orb["ecc"], "rp": orb["rp"], "ra": orb["ra"],
        }
        if self.config.modo_mision == "gto":
            self.eventos.mision_cumplida = True
            self.eventos.datos_mision.update({
                "t": est.t, "rp": orb["rp"], "ra": orb["ra"],
                "ecc": orb["ecc"],
            })
        self._idx_inicio_orbita = len(self.hist_t)
        if orb["es_elipse"]:
            semieje = 0.5 * (orb["rp_radio"] + orb["ra_radio"])
            self._periodo_orbital = 2.0 * np.pi * np.sqrt(semieje ** 3 / MU_T)

    # ========================================================
    # MODO LUNAR (12.3.0): aparcamiento -> ventana TLI -> quema ->
    # vuelo translunar -> entrada a la SOI (evento exacto)
    # "Éxito" = encuentro lunar: entrada a la SOI confirmada. NO es captura
    # (LOI, MCC y órbita lunar objetivo quedan para versiones posteriores).
    # ========================================================
    def _lunar_fallar(self, motivo: str) -> None:
        est, ev = self.estado, self.eventos
        if ev.mision_cumplida is None:
            ev.mision_cumplida = False
            ev.datos_mision["motivo"] = motivo
        est.simulacion_terminada = True
        getattr(self, "_log", print)(f"✗ MODO LUNAR: {motivo}")

    def _ajustar_dt_lunar(self) -> None:
        """Paso grande solo en coast: espera del encendido (acortado para
        caer EXACTO en t_encendido) y vuelo translunar (acortado cerca de la
        Luna). El resto de las fases, el paso base."""
        est, cfg = self.estado, self.config
        dt = self._dt_base
        if est.fase == "espera_tli" and self._lunar_plan is not None:
            dt = cfg.DT_COAST_LUNAR
            resta = self._lunar_plan["t_encendido"] - est.t
            if resta < dt:
                dt = max(resta, 1.0e-6)
        elif est.fase == "quema_loi":
            dt = self._dt_base
        elif est.fase in ("translunar", "orbita_lunar"):
            dt = cfg.DT_COAST_LUNAR
            # El umbral anterior era 3*R_SOI (~198 000 km), por lo que el
            # paso de 5 s se aplicaba durante días y daba la impresión de
            # congelación. Refinamos solo cerca del perilunio/superficie.
            radio_fino = max(5.0 * R_LUNA,
                             min(1.5 * self._dist_objetivo_lunar(), 20_000_000.0))
            if self._lunar_d_luna < radio_fino:
                dt = min(dt, 5.0)
            if (est.fase == "translunar" and cfg.LUNAR_LOI and self._loi is None
                    and self.eventos.entro_soi_luna):
                # (12.6.1) caer EXACTO en el encendido de la LOI
                try:
                    info = self._loi_estimar()
                except ValueError:       # la fase translunar lo reporta como fallo
                    info = None
                if (info is not None and info["vr"] < 0.0
                        and math.isfinite(info["t_burn"])):
                    resta = info["tau"] - 0.5 * info["t_burn"]
                    if resta < dt:
                        dt = max(resta, 1.0e-6)
        est.dt = dt

    def _lunar_iniciar_aparcamiento(self, ctx: ContextoMision) -> None:
        """Órbita declarada y motor apagado: se verifica el aparcamiento
        (no basta con haber alcanzado una altitud) y se pide el plan."""
        est, cfg, ev = self.estado, self.config, self.eventos
        orb = ctx.elementos
        # 1 km de tolerancia: el arrastre del paso entre la declaración y esta
        # verificación puede erosionar el periapsis por debajo del umbral exacto.
        seguro = (orb["es_elipse"] and orb["rp"] >= cfg.ALTURA_SEGURIDAD - 1000.0
                  and orb["ecc"] < cfg.ECC_MAX_APARCAMIENTO
                  and orb["rp"] > cfg.KARMAN)
        ev.datos_mision["aparcamiento"] = {
            "t": est.t, "rp": orb["rp"], "ra": orb["ra"], "ecc": orb["ecc"],
            "periodo": self._periodo_orbital, "seguro": bool(seguro)}
        if not seguro:
            self._lunar_fallar("órbita de aparcamiento no segura "
                               f"(rp={orb['rp'] / 1000:.0f} km, ecc={orb['ecc']:.4f})")
            return
        print(f"🛰 Aparcamiento verificado: rp={orb['rp'] / 1000:.1f} km, "
              f"ra={orb['ra'] / 1000:.1f} km, ecc={orb['ecc']:.5f}, "
              f"periodo={self._periodo_orbital / 60:.1f} min")
        self._lunar_estado = "planificando"
        self._lunar_plan_pendiente = True

    def _resolver_plan_lunar(self) -> bool:
        """True al terminar el plan (por tramos con `presupuesto_plan_s`, de
        una vez sin él). Mientras no termina, la física espera."""
        if self._plan_gen is None:
            self._plan_gen = self._planificar_tli_gen()
            self._plan_t0 = _monotonic()
            print("⏳ Planificando la ventana TLI (búsqueda numérica del encuentro)")
        if self.presupuesto_plan_s is None:
            self._correr_generador(self._plan_gen)
            self._plan_gen = None
            terminado = True
            print(f"⏱ Planificación TLI: {_monotonic() - self._plan_t0:.1f} s")
        else:
            terminado = self._avanzar_plan_por_tramos()
        if terminado:
            self._lunar_plan_pendiente = False
        return terminado

    def _planificar_tli_gen(self):
        """Generador: busca (t_encendido) tal que, con la quema TLI finita
        hasta la energía ε_obj (apogeo LUNAR_APOGEO_TLI), la trayectoria pase
        a LUNAR_DIST_OBJETIVO del centro de la Luna. Variables:
          fijas      ε_obj = -μ/(rp + LUNAR_APOGEO_TLI), guiado prógrado,
                     empuje pleno con la rampa de la simulación;
          ajustada   t_encendido (hora del encendido, s desde el lanzamiento).
        Algoritmo determinista: (1) barrido grueso de t_encendido cada
        LUNAR_PASO_VENTANA_S hasta el primer tramo contiguo con entrada a la
        SOI (se toma su centro); (2) secante sobre el residuo suave de
        momento angular selenocéntrico a la entrada, en la variable más
        sensible (t_encendido o apogeo TLI), hasta d_min = LUNAR_DIST_OBJETIVO
        (si no, el plan apuntaría al centro de la Luna). Propaga
        con la misma dinámica (gravedad + mareas de Luna y Sol, efemérides en
        el marco de lanzamiento) en numba, con tablas de efemérides."""
        cfg, est, ev = self.config, self.estado, self.eventos
        etapa = self.etapa_actual()
        masa = self.masa_total()
        orb = elementos_orbitales(est.px, est.py, est.vx, est.vy)
        if etapa is None or not etapa.tiene_combustible():
            self._lunar_fallar("sin combustible para la quema TLI")
            return
        periodo = float(self._periodo_orbital)
        t_p = est.t
        usa_sol = "sol" in cfg.cuerpos_perturbadores
        mu_l = float(CUERPOS["luna"].mu)
        mu_s = float(MU_SOL) if usa_sol else 0.0
        ventana = cfg.LUNAR_VENTANA_PERIODOS * periodo
        horiz = cfg.LUNAR_HORIZONTE_TRANSFERENCIA
        ht = 600.0
        t0tab = math.floor(t_p / ht) * ht - ht
        n = int((t_p + ventana + horiz + cfg.LUNAR_EXTENSION_PERISELENE_S + 3600.0 - t0tab) / ht) + 3
        prm = self._tabla_efemerides_lunar(t0tab, n)
        yield
        rp_radio = orb["rp_radio"]
        apo_nom = cfg.LUNAR_APOGEO_TLI
        m_seca = masa - etapa.m_comb_restante
        a_sal_p = float(etapa.A_sal * self._presion_ambiente())
        evals = [0]
        fallo = [None]          # (12.8.4) motivo real si `evaluar` devuelve None
        fallo_fatal = [False]   # (12.10.1) True si el fallo no depende del instante (combustible)

        def evaluar(t_enc: float, apo: float, h_coast: float):
            """Quema finita y vuelo hasta la SOI; None si el plan no puede
            seguir (el motivo queda en `fallo[0]`: sin combustible, corte sin
            converger o propagación no finita)."""
            fallo[0] = None
            fallo_fatal[0] = False
            eps_o = -MU_T / (rp_radio + apo)
            x, y, vx, vy, t = _k_coast(est.px, est.py, est.vx, est.vy, t_p, t_enc,
                                       10.0, *prm, R_SOI_LUNA, False)[:5]
            x, y, vx, vy, t, m, estado_q = _k_quema(
                x, y, vx, vy, t, masa, etapa.u0, etapa.D, a_sal_p, m_seca,
                eps_o, 0.25, TAU_REGULACION_THROTTLE, cfg.THROTTLE_MIN, *prm)
            evals[0] += 1
            if estado_q == 2:
                fallo_fatal[0] = True
                fallo[0] = ("combustible insuficiente para alcanzar la energía TLI "
                            f"(apogeo {apo / 1e6:.0f} Mm)")
                return None
            if estado_q == 3:
                fallo[0] = ("sin empuje neto durante la quema TLI (la presión ambiente "
                            f"iguala/supera el empuje; apogeo {apo / 1e6:.1f} Mm)")
                return None
            if estado_q == 4:
                fallo[0] = ("energía de la quema TLI estancada durante 100 pasos "
                            f"(apogeo {apo / 1e6:.1f} Mm, t_enc={t_enc:.0f} s)")
                return None
            if estado_q != 0:
                fallo[0] = ("la quema TLI no convergió al corte de energía "
                            f"(apogeo {apo / 1e6:.1f} Mm, t_enc={t_enc:.0f} s)")
                return None
            (_, _, _, _, _, d_min, t_dmin, t_ent, hz, eps_l,
             periselene_alcanzado) = _coast_hasta_periselene(
                x, y, vx, vy, t, horiz, h_coast, prm, cfg.LUNAR_EXTENSION_PERISELENE_S)
            if not periselene_alcanzado:
                fallo[0] = ("el periselene no se alcanzó dentro del horizonte + extensión; "
                            f"mínimo provisional a t={t_dmin:.0f} s (t_enc={t_enc:.0f} s)")
                return None
            if not all(math.isfinite(v) for v in (d_min, t_dmin, hz, eps_l)):
                fallo[0] = ("propagación del apuntado TLI no finita "
                            f"(d_min={d_min!r}, hz={hz!r}, eps_l={eps_l!r}, t_enc={t_enc:.0f} s)")
                return None
            return {"d_min": d_min, "t_dmin": t_dmin, "t_ent": t_ent, "hz": hz,
                    "eps_l": eps_l, "post": (x, y, vx, vy, t, m), "t_fin": t,
                    "eps_obj": eps_o, "apo": apo, "t_enc": t_enc}

        def fallo_plan():
            self._lunar_fallar(fallo[0] or "combustible insuficiente para alcanzar la "
                               f"energía TLI (apogeo {apo_nom / 1e6:.0f} Mm)")

        # 1) Fase de la ventana: barrido grueso de t_encendido (ε nominal) hasta
        #    el primer tramo contiguo con entrada a la SOI; se toma su centro.
        paso = cfg.LUNAR_PASO_VENTANA_S
        t_k = t_p + paso
        run = []
        k = 0
        n_barrido = n_fallos = seguidos = 0
        while t_k <= t_p + ventana + paso:
            r = evaluar(t_k, apo_nom, 60.0)
            n_barrido += 1
            if r is None:
                if fallo_fatal[0]:           # sin combustible: igual en todos los instantes
                    fallo_plan()
                    return
                # (12.10.1) un fallo numérico puntual no invalida el resto de la ventana;
                # (12.10.2) ni trunca el tramo con encuentro: solo 3 fallos seguidos lo cortan
                n_fallos += 1
                seguidos += 1
                if n_fallos == 1:
                    print(f"⚠ Barrido TLI: primer fallo en t_enc={t_k:.0f} s ({fallo[0]}); "
                          "se sigue el barrido")
                if run and seguidos >= 3:
                    break
            else:
                seguidos = 0
                if r["t_ent"] > 0.0:
                    run.append(t_k)
                elif run:
                    break
            k += 1
            if k % 6 == 0:
                yield
            t_k += paso
        if n_fallos:
            print(f"⚠ Barrido TLI: {n_fallos}/{n_barrido} instantes omitidos por fallo numérico")
        if not run and n_barrido > 0 and n_fallos == n_barrido:
            fallo_plan()                     # (12.10.1) fallaron TODOS los instantes
            return
        if not run:
            self._lunar_fallar("sin ventana TLI: ninguna hora de encendido "
                               f"en {cfg.LUNAR_VENTANA_PERIODOS:g} periodos de "
                               "aparcamiento da encuentro con la Luna")
            return
        x_t = 0.5 * (run[0] + run[-1])
        yield

        # 2) Apuntado: residuo suave F = hz - signo·h_obj(eps_l), con hz el
        #    momento angular selenocéntrico a la entrada y h_obj el que da un
        #    periselene de LUNAR_DIST_OBJETIVO (cónica con la energía de
        #    entrada). Se resuelve por secante en UNA variable: la de mayor
        #    sensibilidad entre t_encendido y el apogeo TLI (según la geometría
        #    del encuentro, una u otra controla la distancia de paso).
        d_obj = self._dist_objetivo_lunar()

        def residuo(r, signo):
            v_p = math.sqrt(max(0.0, 2.0 * (r["eps_l"] + mu_l / d_obj)))
            return r["hz"] - signo * d_obj * v_p

        def f_de(var, valor, signo, h_coast=20.0):
            r = evaluar(valor if var == "t" else x_t, apo_nom if var == "t" else valor,
                        h_coast)
            if r is None or not (r["t_ent"] > 0.0):
                return None, r
            return residuo(r, signo), r

        base = evaluar(x_t, apo_nom, 20.0)
        if base is None:
            fallo_plan()
            return
        if not (base["t_ent"] > 0.0):
            x_t = run[len(run) // 2]
            base = evaluar(x_t, apo_nom, 20.0)
        signo = 1.0 if base["hz"] >= 0.0 else -1.0
        F0 = residuo(base, signo)
        cand = {}
        # "alcance" = |pendiente| x recorrido útil de la variable (300 s de
        # encendido, 30 000 km de apogeo): elige la que puede llegar al objetivo
        for var, d0, rango in (("t", 30.0, 300.0), ("apo", 1.0e6, 3.0e7)):
            v0 = x_t if var == "t" else apo_nom
            f1, _ = f_de(var, v0 + d0, signo)
            if f1 is not None:
                cand[var] = (abs(f1 - F0) / d0 * rango, d0, f1)
        yield
        if not cand:
            self._lunar_fallar("el apuntado no encontró sensibilidad (sin entrada a la SOI)")
            return
        var = max(cand, key=lambda k_: cand[k_][0])
        paso_max = 150.0 if var == "t" else 4.0e6
        xa = x_t if var == "t" else apo_nom
        xb = xa + cand[var][1]
        fa, fb = F0, cand[var][2]
        for i in range(24):
            if abs(fb) < 1.0e-4 * d_obj * 1000.0 or fb == fa:
                break
            xn = xb - fb * (xb - xa) / (fb - fa)
            dx = max(-paso_max, min(paso_max, xn - xb))
            xn = xb + dx
            fn, _ = f_de(var, xn, signo)
            while fn is None and abs(dx) > 1e-3:
                dx *= 0.5
                xn = xb + dx
                fn, _ = f_de(var, xn, signo)
            if fn is None:
                break
            xa, fa, xb, fb = xb, fb, xn, fn
            if i % 3 == 2:
                yield
        t_star = xb if var == "t" else x_t
        apo_star = apo_nom if var == "t" else xb
        fin = evaluar(t_star, apo_star, 20.0)
        if fin is None:
            fallo_plan()
            return
        if not (fin["t_ent"] > 0.0):
            self._lunar_fallar("el apuntado no logró entrar a la SOI lunar en la "
                               "predicción")
            return
        if t_star < t_p + 1.0:
            # El encendido no puede quedar antes de la hora de planificación
            # (la secante de la etapa 2 no tenía esa cota): se fija en el
            # primer instante posible y la etapa 3 corrige con el apogeo TLI.
            t_star = t_p + 1.0
            fin = evaluar(t_star, apo_star, 20.0)
            if fin is None:
                fallo_plan()
                return
            if not (fin["t_ent"] > 0.0):
                self._lunar_fallar("el apuntado no logró entrar a la SOI lunar con el "
                                   "encendido en el primer instante posible")
                return
        # 3) (12.5.7) Refinamiento sobre la distancia mínima REAL de la
        #    propagación (la cónica de 2 cuerpos ignora la marea terrestre).
        tol_d = max(1000.0, 2.0e-3 * d_obj)
        if abs(fin["d_min"] - d_obj) > tol_d:
            limites = {"t": (t_p + 1.0, t_p + ventana + paso),
                       "apo": (1.0e7, 6.0e8)}
            pasos_ref = {"t": (1.0, 20.0), "apo": (1.0e5, 5.0e5)}
            mejor = (abs(fin["d_min"] - d_obj), t_star, apo_star, fin)

            def g_de(var_, valor, t_b, apo_b):
                r_ = evaluar(valor if var_ == "t" else t_b,
                             apo_b if var_ == "t" else valor, 20.0)
                if r_ is None or not (r_["t_ent"] > 0.0):
                    return None, r_
                return r_["d_min"] - d_obj, r_
            for var_ in (var, "apo" if var == "t" else "t"):
                _, t_b, apo_b, f_b = mejor[0], mejor[1], mejor[2], mejor[3]
                lo, hi = limites[var_]
                d0, paso_m = pasos_ref[var_]
                xa = t_b if var_ == "t" else apo_b
                ga = f_b["d_min"] - d_obj
                xb = min(hi, max(lo, xa + d0))
                if xb == xa:
                    xb = min(hi, max(lo, xa - d0))
                gb, rb = g_de(var_, xb, t_b, apo_b)
                if gb is None:
                    continue
                if abs(gb) < mejor[0]:
                    mejor = (abs(gb), xb if var_ == "t" else t_b,
                             apo_b if var_ == "t" else xb, rb)
                for i in range(10):
                    if mejor[0] <= tol_d or gb == ga:
                        break
                    xn = xb - gb * (xb - xa) / (gb - ga)
                    xn = min(hi, max(lo, xb + max(-paso_m, min(paso_m, xn - xb))))
                    if xn == xb:
                        break
                    gn, rn = g_de(var_, xn, t_b, apo_b)
                    if gn is None:
                        break
                    xa, ga, xb, gb = xb, gb, xn, gn
                    if abs(gn) < mejor[0]:
                        mejor = (abs(gn), xn if var_ == "t" else t_b,
                                 apo_b if var_ == "t" else xn, rn)
                    if i % 3 == 2:
                        yield
                yield
                if mejor[0] <= tol_d:
                    break
            _, t_star, apo_star, fin = mejor
            if apo_star != apo_nom and var == "t":
                var = "apo"
        eps_obj = fin["eps_obj"]
        # Estado predicho a la entrada (velocidad relativa)
        x, y, vx, vy, t, m = fin["post"]
        e = _k_coast(x, y, vx, vy, t, fin["t_ent"], 5.0, *prm, R_SOI_LUNA, False)
        (_, _), (lvx, lvy) = self._estado_cuerpo("luna", fin["t_ent"])
        v_rel_pred = math.hypot(e[2] - lvx, e[3] - lvy)
        dv_req = etapa.u0 * math.log(masa / m)
        dv_disp = etapa.u0 * math.log(masa / m_seca)
        for clave in ("d_min", "t_ent", "t_dmin", "t_fin", "eps_obj", "eps_l"):
            if not math.isfinite(float(fin[clave])):      # (12.8.4) NaN no pasa los `if err_d > ...`
                self._lunar_fallar(f"plan TLI corrupto: {clave}={fin[clave]!r}")
                return
        dv_loi_req = 0.0
        if cfg.LUNAR_LOI:
            # Δv de la LOI: velocidad en el periselene (cónica con la energía de
            # entrada) menos la circular a ese radio; usa lo que queda tras la TLI.
            eps_l = float(fin["eps_l"])
            r_peri = max(float(fin["d_min"]), R_LUNA + 1.0)
            v_peri = math.sqrt(max(0.0, 2.0 * (eps_l + mu_l / r_peri)))
            dv_loi_req = max(0.0, v_peri - math.sqrt(mu_l / r_peri))
            dv_resto = etapa.u0 * math.log(m / m_seca)
            if dv_resto < cfg.LUNAR_MARGEN_DV * dv_loi_req:
                self._lunar_fallar(f"Δv insuficiente para la LOI: quedan "
                                   f"{dv_resto:.0f} m/s tras la TLI, requeridos "
                                   f"≈{dv_loi_req:.0f} m/s")
                return
        if dv_disp < cfg.LUNAR_MARGEN_DV * dv_req:
            self._lunar_fallar(f"Δv insuficiente: disponible {dv_disp:.0f} m/s, "
                               f"requerido {dv_req:.0f} m/s (margen "
                               f"{cfg.LUNAR_MARGEN_DV:g})")
            return
        err_d = abs(float(fin["d_min"]) - d_obj)
        tol_ok = err_d <= tol_d
        if err_d > 0.10 * d_obj:
            self._lunar_fallar(f"apuntado fallido: d_min predicho "
                               f"{fin['d_min'] / 1000:.0f} km vs objetivo "
                               f"{d_obj / 1000:.0f} km (error {err_d / 1000:.0f} km "
                               f"> 10 %)")
            return
        if not tol_ok:
            print(f"⚠ Apuntado lunar fuera de tolerancia: error {err_d / 1000:.1f} km "
                  f"(tol. {tol_d / 1000:.1f} km); se continúa.")
        self._lunar_plan = {
            "error_d_min_m": float(err_d), "tolerancia_cumplida": bool(tol_ok),
            "t_plan": t_p, "t_encendido": float(t_star), "eps_obj": float(eps_obj),
            "d_min_pred": float(fin["d_min"]), "t_soi_pred": float(fin["t_ent"]),
            "v_rel_pred": float(v_rel_pred), "dv_requerido": float(dv_req),
            "dv_disponible": float(dv_disp), "evaluaciones": evals[0],
            "dv_loi_requerido": float(dv_loi_req), "t_peri_pred": float(fin["t_dmin"]),
            "t_fin_quema_pred": float(fin["t_fin"]),
            "variables_ajustadas": (("t_encendido",) if apo_star == apo_nom
                                    else ("t_encendido", "eps_obj")),
            "variable_de_apuntado": "t_encendido" if var == "t" else "eps_obj",
            "apogeo_tli_m": float(apo_star),
            "variables_fijas": {"apogeo_tli_nominal_m": apo_nom,
                                "dist_objetivo_m": d_obj, "loi": bool(cfg.LUNAR_LOI),
                                "guiado": "progrado, empuje pleno con rampa"}}
        ev.datos_mision["plan_tli"] = dict(self._lunar_plan)
        self._lunar_estado = "espera_tli"
        self._fijar_fase("espera_tli")
        print(f"🎯 Plan TLI: encendido en t={t_star:.1f} s "
              f"(en {t_star - t_p:.0f} s), Δv≈{dv_req:.0f} m/s "
              f"(disp. {dv_disp:.0f}); predicción: d_min={fin['d_min'] / 1e3:.0f} km, "
              f"SOI en t={fin['t_ent'] / 3600:.1f} h, v_rel={v_rel_pred:.0f} m/s "
              f"[{evals[0]} evaluaciones]")

    def _fase_espera_tli(self, ctx: ContextoMision) -> None:
        est = self.estado
        est.modo_guiado = "prograde"
        if est.t < self._lunar_plan["t_encendido"] - 1.0e-6:
            return
        etapa = ctx.etapa
        if not ctx.tiene_combustible:
            self._lunar_fallar("sin combustible al llegar al encendido TLI")
            return
        self._encender_etapa(etapa)
        self._fijar_fase("quema_tli")
        self._lunar_estado = "quema_tli"
        # Desde aquí el historial ya no es «una órbita»: sin poda ni snapshot
        self._idx_inicio_orbita = None
        self._snapshot_orbita = None
        self.eventos.datos_mision["t_encendido_real"] = est.t
        self._lunar_dv0 = float(etapa.dv_prop)    # Δv del motor ya gastado (circularización)
        print(f"🔥 ENCENDIDO TLI en t={est.t:.1f} s (plan {self._lunar_plan['t_encendido']:.1f} s)")

    def _fase_quema_tli(self, ctx: ContextoMision) -> None:
        est, ev = self.estado, self.eventos
        etapa = ctx.etapa
        eps = ctx.elementos["eps"]
        if eps >= self._lunar_plan["eps_obj"]:
            if etapa is not None:
                etapa.activa = False
            self._fijar_fase("translunar")
            self._lunar_estado = "translunar"
            ev.datos_mision.update({
                "resultado": "llego_luna", "t_cutoff": est.t,
                "ra_cutoff": ctx.elementos["ra"], "rp_cutoff": ctx.elementos["rp"],
                "ecc_cutoff": ctx.elementos["ecc"],
                "comb_restante": etapa.m_comb_restante if etapa else 0.0,
                "dv_prop": (etapa.dv_prop - self._lunar_dv0) if etapa else 0.0})
            print(f"✂ CORTE TLI: ra={ctx.elementos['ra'] / 1000:.0f} km, "
                  f"Δv motor de la quema≈{etapa.dv_prop - self._lunar_dv0:.0f} m/s")
        elif not ctx.tiene_combustible:
            ev.datos_mision["resultado"] = "insuficiente"
            self._lunar_fallar("sin combustible antes de alcanzar la energía TLI")

    def _fase_translunar(self, ctx: ContextoMision) -> None:
        est, ev = self.estado, self.eventos
        est.modo_guiado = "prograde"
        if not (self.config.LUNAR_LOI and ev.entro_soi_luna) or self._loi is not None:
            return
        try:
            info = self._loi_estimar()
        except ValueError as e:
            self._lunar_fallar(f"LOI: {e}")
            return
        if info is None:
            self._lunar_fallar("LOI: sin etapa para la quema de captura")
            return
        if info["vr"] >= 0.0:
            self._lunar_fallar("LOI perdida: se pasó el periselene sin encender")
            return
        if info["eps"] <= 0.0:
            self._lunar_fallar("LOI: la nave ya está ligada a la Luna antes de la quema")
            return
        if info["tau"] <= 0.5 * info["t_burn"] + 0.05:
            self._loi_encender(ctx, info)

    # --- LOI (12.6.1) ---
    def _estado_selenocentrico(self):
        """(rx, ry, wx, wy) de la nave respecto de la Luna en est.t."""
        est = self.estado
        (lx, ly), (lvx, lvy) = self._estado_cuerpo("luna", est.t)
        return est.px - lx, est.py - ly, est.vx - lvx, est.vy - lvy

    def _elementos_selenocentricos(self) -> dict:
        """ε, h, ecc, rp, ra, a y periodo de la cónica selenocéntrica osculadora."""
        return _conica_selenocentrica(*self._estado_selenocentrico(),
                                      float(CUERPOS["luna"].mu))

    def _duracion_quema_loi(self, dv: float, masa: float, etapa) -> float:
        """Duración estimada de la quema LOI; infinito si el propelente no alcanza."""
        try:
            combustible = float(etapa.m_comb_restante)
            m = float(masa)
            dv_rest = float(dv)
            u0, D = float(etapa.u0), float(etapa.D)
            thr_min = float(self.config.THROTTLE_MIN)
        except (AttributeError, TypeError, ValueError):
            return float("inf")
        if not all(math.isfinite(v) for v in (combustible, m, dv_rest, u0, D, thr_min)):
            return float("inf")
        if dv_rest <= 1.0e-6:
            return 0.0
        if combustible <= 0.0 or m <= combustible or u0 <= 0.0 or D <= 0.0:
            return float("inf")

        # Masa no consumible = masa total inicial menos el propelente restante
        # de esta etapa (incluye carga útil/estructuras que la etapa soporta).
        m_seca = m - combustible
        t = 0.0
        while dv_rest > 1.0e-6 and t < 600.0:
            if m <= m_seca + 1.0e-12:
                return float("inf")
            a_pleno = u0 * D / m
            thr = min(1.0, max(thr_min, dv_rest / (a_pleno * TAU_REGULACION_THROTTLE)))
            a = a_pleno * thr
            flujo = D * thr
            if not (a > 0.0 and flujo > 0.0 and math.isfinite(a) and math.isfinite(flujo)):
                return float("inf")

            # No simular un paso de 0.25 s si el combustible se agota antes.
            dt_disponible = (m - m_seca) / flujo
            dt = min(0.25, dt_disponible)
            t_necesario = dv_rest / a
            if t_necesario <= dt:
                return t + t_necesario
            if dt < 0.25 - 1.0e-12:
                return float("inf")
            dv_rest -= a * 0.25
            m -= flujo * 0.25
            t += 0.25
        return t if dv_rest <= 1.0e-6 else float("inf")

    def _loi_estimar(self):
        """Tiempo al periselene (cónica selenocéntrica), Δv y duración de la
        quema de captura circular en ese radio; None si no hay etapa."""
        etapa = self.etapa_actual()
        if etapa is None:
            return None
        mu_l = float(CUERPOS["luna"].mu)
        el = self._elementos_selenocentricos()
        eps, ecc, r, rp = el["eps"], el["ecc"], el["r"], el["rp"]
        if not (rp > 0.0 and math.isfinite(rp)):
            raise ValueError(f"momento angular nulo (rp={rp!r}): trayectoria radial")
        tau = float("inf")
        if 0.0 < eps < 1.0e-6 * mu_l / r:        # (12.10.1) casi parabólica: a, n y tau no son fiables
            raise ValueError(f"trayectoria casi parabólica respecto de la Luna (eps={eps!r} J/kg)")
        if eps > 0.0:                    # (12.6.2) eps > 0 basta; ecc >= 1 por redondeo
            a = mu_l / (2.0 * eps)
            H = math.acosh(max((r / a + 1.0) / ecc, 1.0))
            M = ecc * math.sinh(H) - H
            n = math.sqrt(mu_l / a ** 3)
            tau = M / n if el["vr"] < 0.0 else -M / n
        v_p = el["h"] / rp
        v_c = math.sqrt(mu_l / rp)
        dv = max(0.0, v_p - v_c)
        masa = self.masa_total()
        if tau < 300.0:
            t_burn = self._duracion_quema_loi(dv, masa, etapa)
        else:
            t_burn = masa * (1.0 - math.exp(-dv / etapa.u0)) / etapa.D
        return {"tau": tau, "t_burn": t_burn, "dv": dv, "rp": rp, "eps": eps,
                "vr": el["vr"], "eps_obj": -mu_l / (2.0 * rp), "ecc": ecc}

    def _loi_encender(self, ctx: ContextoMision, info: dict) -> None:
        est, ev = self.estado, self.eventos
        etapa = ctx.etapa
        masa = self.masa_total()
        m_seca = masa - etapa.m_comb_restante
        dv_resto = etapa.u0 * math.log(masa / m_seca)
        if not ctx.tiene_combustible or dv_resto < self.config.LUNAR_MARGEN_DV * info["dv"]:
            self._lunar_fallar(f"Δv insuficiente para la LOI: quedan {dv_resto:.0f} m/s, "
                               f"requeridos ≈{info['dv']:.0f} m/s")
            return
        self._encender_etapa(etapa)
        self._fijar_fase("quema_loi")
        self._lunar_estado = "quema_loi"
        self._loi = {"t_ign": est.t, "eps_obj": info["eps_obj"], "rp_pred": info["rp"],
                     "dv_req": info["dv"], "t_burn_est": info["t_burn"],
                     "tau_ign": info["tau"], "dv0": float(etapa.dv_prop),
                     "comb0": float(etapa.m_comb_restante), "cortado": False,
                     "t_cut": None, "t_verif0": None, "T": None}
        ev.datos_mision["t_encendido_loi"] = est.t
        print(f"🔥 ENCENDIDO LOI en t={est.t:.1f} s: periselene previsto en {info['tau']:.1f} s, "
              f"r_p={info['rp'] / 1000:.1f} km (alt {(info['rp'] - R_LUNA) / 1000:.0f} km), "
              f"Δv≈{info['dv']:.0f} m/s, duración≈{info['t_burn']:.1f} s")

    def _paso_quema_loi_sim(self) -> None:
        """(12.6.1) Un paso de la quema LOI: empuje retrógrado respecto de la
        Luna con la regulación por Δv faltante de la TLI, RK4 con Luna y Sol
        en cada subpaso y corte localizado dentro del paso (como `_k_quema`)."""
        est, loi = self.estado, self._loi
        if est.simulacion_terminada or est.pausado or loi is None:
            return
        etapa = self.etapa_actual()
        if etapa is None or not etapa.tiene_combustible():
            est.empuje = 0.0
            self._lunar_fallar("sin combustible en la quema LOI")
            return
        if not (math.isfinite(est.dt) and est.dt > 0.0):
            est.empuje = 0.0
            self._lunar_fallar(f"paso temporal inválido en la quema LOI (dt={est.dt!r})")
            return
        prm = self._tabla_coast_lunar()
        luna, _sol, t0, ht, _mu_t, mu_l = prm[0], prm[1], prm[2], prm[3], prm[4], prm[5]
        x, y, vx, vy, t = est.px, est.py, est.vx, est.vy, est.t
        m = self.masa_total()
        dt = est.dt
        eps_obj = loi["eps_obj"]
        eps = _eps_selenocentrica(x, y, vx, vy, t, luna, t0, ht, mu_l)
        lvx, lvy = _herm_vel(luna, t0, ht, t)
        w = math.hypot(vx - lvx, vy - lvy)
        thr = self._throttle_proporcional_por_dv(
            (eps - eps_obj) / max(w, 1.0), etapa.u0 * etapa.D / m)
        empuje, flujo = etapa.empuje_y_flujo(thr, self._presion_ambiente())
        if not (math.isfinite(flujo) and flujo > 0.0 and empuje > 0.0):
            est.empuje = 0.0
            self._lunar_fallar("sin empuje neto en la quema LOI")
            return
        dt0 = dt
        if flujo * dt0 >= etapa.m_comb_restante:
            dt0 = etapa.m_comb_restante / flujo
        if not (math.isfinite(dt0) and dt0 > 0.0):
            est.empuje = 0.0
            self._lunar_fallar("sin combustible para avanzar la quema LOI")
            return
        nx, ny, nvx, nvy = _paso_quema_loi(x, y, vx, vy, t, m, empuje, flujo, dt0, *prm)
        eps_n = _eps_selenocentrica(nx, ny, nvx, nvy, t + dt0, luna, t0, ht, mu_l)
        if not math.isfinite(eps_n):               # (12.13.4) antes de usarla para decidir el corte
            est.empuje = 0.0
            self._msg_fallo_integrador = "energía selenocéntrica no finita en la quema LOI"
            self._fallo_integrador("quema LOI")
            return
        dt_sub = dt0
        cortado = eps_n <= eps_obj
        if cortado:
            # (12.6.2) corte: secante sobre h con intervalo [lo, hi] (en lo aún no
            # se alcanza ε_obj, en hi sí); si la corrección no es finita o sale del
            # intervalo, bisección. Para al cumplir la tolerancia o tras
            # LOI_MAX_ITER_CORTE; el residuo queda registrado.
            tol = LOI_TOL_EPS_REL * abs(eps_obj)
            lo_dt, hi_dt = 0.0, dt0
            den = eps_n - eps
            f = (eps_obj - eps) / den if den != 0.0 and math.isfinite(den) else 0.5
            err = eps_n - eps_obj
            n_it = 0
            for _ in range(LOI_MAX_ITER_CORTE):
                n_it += 1
                dt_sub = f * dt0
                if not (lo_dt < dt_sub < hi_dt):
                    dt_sub = 0.5 * (lo_dt + hi_dt)
                nx, ny, nvx, nvy = _paso_quema_loi(x, y, vx, vy, t, m, empuje, flujo, dt_sub, *prm)
                eps_n = _eps_selenocentrica(nx, ny, nvx, nvy, t + dt_sub, luna, t0, ht, mu_l)
                err = eps_n - eps_obj
                if abs(err) <= tol:
                    break
                if err > 0.0:
                    lo_dt = dt_sub
                else:
                    hi_dt = dt_sub
                den = eps_n - eps
                f = ((dt_sub / dt0) * (eps_obj - eps) / den
                     if den != 0.0 and math.isfinite(den) else 0.5 * (lo_dt + hi_dt) / dt0)
            loi["eps_error_corte"] = float(err)
            loi["corte_convergido"] = bool(abs(err) <= tol)
            loi["iter_corte"] = n_it
            if not loi["corte_convergido"]:
                est.empuje = 0.0
                self._lunar_fallar(f"LOI: corte sin converger tras {n_it} iteraciones "
                                   f"(residuo {err:.3g} J/kg, tolerancia {tol:.3g})")
                return
        m_media = m - 0.5 * flujo * dt_sub
        etapa.consumir(flujo, dt_sub)
        etapa.dv_prop += empuje / m_media * dt_sub
        est.empuje = 0.0 if cortado else empuje
        est.arrastre_mag = 0.0
        if cortado:
            loi["cortado"], loi["t_cut"] = True, t + dt_sub
        # corte o sin combustible a mitad de paso: coast del resto del paso
        if cortado or dt_sub < dt - 1.0e-12:
            resto = dt - dt_sub
            if resto > 1.0e-9:
                nx, ny, nvx, nvy = _k_coast(nx, ny, nvx, nvy, t + dt_sub, t + dt, resto,
                                            *prm, R_SOI_LUNA, False)[:4]
            dt_sub = dt
        if not all(math.isfinite(v) for v in (nx, ny, nvx, nvy)):
            self._msg_fallo_integrador = "RK4 de la quema LOI no finito"
            self._fallo_integrador("quema LOI")
            return
        est.px, est.py, est.vx, est.vy = nx, ny, nvx, nvy
        est.t = t + dt_sub
        self._actualizar_altitud_y_rapidez()

    def _fase_quema_loi(self, ctx: ContextoMision) -> None:
        est, ev, loi = self.estado, self.eventos, self._loi
        etapa = ctx.etapa
        est.modo_guiado = "retrograde"
        if loi is not None and loi["cortado"]:
            if etapa is not None:
                etapa.activa = False
            self._fijar_fase("orbita_lunar")
            self._lunar_estado = "orbita_lunar"
            try:
                el = self._elementos_selenocentricos()
            except ValueError as e:
                self._lunar_fallar(f"LOI: {e}")
                return
            ev.datos_mision["loi"] = {
                "t_ign": loi["t_ign"], "t_cut": loi["t_cut"],
                "dv_prop": float(etapa.dv_prop - loi["dv0"]) if etapa else 0.0,
                "comb_gastado": float(loi["comb0"] - etapa.m_comb_restante) if etapa else 0.0,
                "comb_restante": float(etapa.m_comb_restante) if etapa else 0.0,
                "rp": el["rp"], "ra": el["ra"], "ecc": el["ecc"], "eps": el["eps"],
                "periodo": el["T"], "dv_requerido": loi["dv_req"],
                "eps_error_corte": loi.get("eps_error_corte"),
                "corte_convergido": loi.get("corte_convergido"),
                "iter_corte": loi.get("iter_corte")}
            print(f"✂ CORTE LOI en t={loi['t_cut']:.1f} s: Δv motor≈"
                  f"{ev.datos_mision['loi']['dv_prop']:.0f} m/s, órbita selenocéntrica "
                  f"rp={(el['rp'] - R_LUNA) / 1000:.0f} km × ra={(el['ra'] - R_LUNA) / 1000:.0f} km "
                  f"de altitud, ecc={el['ecc']:.4f}")
        elif not ctx.tiene_combustible:
            self._lunar_fallar("sin combustible antes de alcanzar la energía de captura LOI")

    def _fase_orbita_lunar(self, ctx: ContextoMision) -> None:
        """Verificación de la captura: órbita selenocéntrica ligada con
        periselene sobre la superficie durante LUNAR_PERIODOS_VERIF periodos."""
        est, ev, loi, cfg = self.estado, self.eventos, self._loi, self.config
        est.modo_guiado = "prograde"
        try:
            el = self._elementos_selenocentricos()
        except ValueError as e:
            self._lunar_fallar(f"LOI: {e}")
            return
        if el["eps"] >= 0.0:
            self._lunar_fallar(f"LOI: la nave no quedó ligada a la Luna (ε={el['eps']:.1f} J/kg)")
            return
        if el["rp"] < R_LUNA:
            self._lunar_fallar(f"LOI: el periselene queda bajo la superficie "
                               f"({(el['rp'] - R_LUNA) / 1000:.0f} km)")
            return
        if loi["t_verif0"] is None:
            loi["t_verif0"], loi["T"] = est.t, el["T"]
        d = ev.datos_mision["loi"]
        for clave, campo, f in (("rp_min_verif", "rp", min), ("ra_max_verif", "ra", max),
                                ("eps_min_verif", "eps", min), ("eps_max_verif", "eps", max),
                                ("T_min_verif", "T", min), ("T_max_verif", "T", max),
                                ("ecc_max_verif", "ecc", max)):
            d[clave] = f(d.get(clave, el[campo]), el[campo])
        if est.t - loi["t_verif0"] >= cfg.LUNAR_PERIODOS_VERIF * loi["T"]:
            d["periodos_verificados"] = (est.t - loi["t_verif0"]) / loi["T"]
            # (12.7.1) tres niveles: ligada (ε<0) y segura (rp > R_LUNA) ya se
            # comprobaron en cada paso; falta la órbita OBJETIVO con tolerancias
            e_rp = d["rp_min_verif"] - R_LUNA - cfg.LUNAR_ORBITA_ALT
            e_ra = d["ra_max_verif"] - R_LUNA - cfg.LUNAR_ORBITA_ALT
            ok_obj = (abs(e_rp) <= cfg.LUNAR_ORBITA_TOL_ALT and abs(e_ra) <= cfg.LUNAR_ORBITA_TOL_ALT
                      and d["ecc_max_verif"] <= cfg.LUNAR_ORBITA_ECC_MAX)
            d.update({"captura_ligada": True, "captura_segura": True, "orbita_objetivo": bool(ok_obj),
                      "error_rp_objetivo": float(e_rp), "error_ra_objetivo": float(e_ra)})
            ev.datos_mision.update({"t_captura": est.t, "captura_validada": False, "modelo_captura": "2D"})
            if not ok_obj:
                ev.datos_mision["resultado"] = "captura_fuera_de_objetivo"
                self._lunar_fallar(
                    f"captura ligada y segura, pero fuera de la órbita objetivo: "
                    f"rp {(d['rp_min_verif'] - R_LUNA) / 1000:.1f} km, ra {(d['ra_max_verif'] - R_LUNA) / 1000:.1f} km, "
                    f"ecc {d['ecc_max_verif']:.4f} (objetivo {cfg.LUNAR_ORBITA_ALT / 1000:.0f} km "
                    f"± {cfg.LUNAR_ORBITA_TOL_ALT / 1000:.0f} km, ecc <= {cfg.LUNAR_ORBITA_ECC_MAX})")
                return
            ev.mision_cumplida = True
            ev.datos_mision["resultado"] = "captura_lunar"
            est.simulacion_terminada = True
            print(f"🌕 CAPTURA LUNAR (2D, provisional): órbita rp={(d['rp_min_verif'] - R_LUNA) / 1000:.0f} km × "
                  f"ra={(d['ra_max_verif'] - R_LUNA) / 1000:.0f} km de altitud, T={loi['T'] / 60:.0f} min, "
                  f"verificada {d['periodos_verificados']:.1f} periodos")

    def _detectar_soi_lunar(self, y_prev, t_prev: float) -> None:
        """Entrada a la SOI como EVENTO: cruce de g = R_SOI - |r - r_Luna(t)|
        de negativo a positivo dentro del paso, localizado por bisección sobre
        la interpolación cúbica de Hermite de la nave y la Luna de la
        efeméride en cada instante. El estado queda en el cruce exacto; la
        propagación continúa, salvo que haya otro evento terminal."""
        est, ev = self.estado, self.eventos
        if ev.entro_soi_luna or est.fase not in ("quema_tli", "translunar"):
            return
        dt = est.t - t_prev
        y1 = (est.px, est.py, est.vx, est.vy)

        def g(y, t):
            lx, ly = self._posicion_cuerpo("luna", t)
            return R_SOI_LUNA - math.hypot(y[0] - lx, y[1] - ly)

        if g(y1, est.t) < 0.0:
            return
        if g(y_prev, t_prev) >= 0.0:
            s_c = 0.0
        else:
            s_c = _biseccion_s(
                lambda s: g(self._hermite_estado(y_prev, y1, dt, s), t_prev + s * dt) < 0.0,
                80, 1.0e-15)
        y_c = self._hermite_estado(y_prev, y1, dt, s_c)
        est.px, est.py, est.vx, est.vy = y_c
        est.t = t_prev + s_c * dt
        self._actualizar_altitud_y_rapidez()
        (lx, ly), (lvx, lvy) = self._estado_cuerpo("luna", est.t)
        rx, ry = est.px - lx, est.py - ly
        wx, wy = est.vx - lvx, est.vy - lvy
        d = math.hypot(rx, ry)
        v_rel = math.hypot(wx, wy)
        mu_l = float(CUERPOS["luna"].mu)
        rp_pred = _conica_selenocentrica(rx, ry, wx, wy, mu_l)["rp"]
        ev.entro_soi_luna = True
        ev.d_luna_min = min(ev.d_luna_min, d)
        ev.datos_soi_luna = {
            "t": est.t, "r_alt": float(math.hypot(est.px, est.py) - R_T),
            "d": d, "v_rel": v_rel, "v_rad_rel": (rx * wx + ry * wy) / d,
            "d_min_dos_cuerpos": rp_pred, "paso_dt": dt, "fraccion_paso": s_c,
            "z_luna": self._z_luna(est.t)}
        if not self.config.LUNAR_LOI:       # con LOI solo la captura cumple (12.6.1)
            ev.mision_cumplida = True
        ev.datos_mision.update({
            "resultado": "encuentro_lunar", "t_soi": est.t,
            "continuar_tras_soi": True,
        })
        # IMPORTANTE: la SOI define el éxito del encuentro, no el fin de la
        # propagación. Seguimos integrando la dinámica actual (geocéntrica con
        # mareas del Sol y la Luna) hasta T_MAX_LUNAR, impacto u otro fallo.
        # No se atribuye captura orbital: aún no hay maniobra LOI.
        print(f"🌕 ENTRADA A LA SOI LUNAR (encuentro, no captura) en "
              f"t={est.t / 3600:.3f} h: v_rel={v_rel:.0f} m/s, "
              f"acercamiento previsto (2 cuerpos)={rp_pred / 1000:.0f} km; "
              f"la propagación continúa hasta T_MAX_LUNAR o evento físico")

    def _detectar_impacto_lunar(self, y_prev, t_prev: float) -> None:
        """Localiza el contacto con la superficie lunar como evento terminal.

        La nave y la Luna se interpolan en el mismo instante. El paso fino en
        las proximidades lunares garantiza que un cruce no pueda atravesar la
        superficie entre dos extremos sin cambiar el signo del margen.
        """
        est, ev = self.estado, self.eventos
        if not self._lunar or ev.impacto_lunar_registrado or est.simulacion_terminada:
            return
        dt = est.t - t_prev
        if dt <= 0.0:
            return
        y_fin = (est.px, est.py, est.vx, est.vy)

        def margen(y, t):
            lx, ly = self._posicion_cuerpo("luna", t)
            return math.hypot(y[0] - lx, y[1] - ly) - R_LUNA

        g0 = margen(y_prev, t_prev)
        g1 = margen(y_fin, est.t)
        if g0 > 0.0 and g1 > 0.0:
            return
        if g0 <= 0.0:
            s_hit = 0.0
        else:
            s_hit = _biseccion_s(
                lambda s: margen(self._hermite_estado(y_prev, y_fin, dt, s), t_prev + s * dt) > 0.0,
                64, 1.0e-13)

        y_hit = self._hermite_estado(y_prev, y_fin, dt, s_hit) if s_hit > 0.0 else tuple(y_prev)
        t_hit = t_prev + s_hit * dt
        (lx, ly), (lvx, lvy) = self._estado_cuerpo("luna", t_hit)
        v_rel = math.hypot(y_hit[2] - lvx, y_hit[3] - lvy)
        est.px, est.py, est.vx, est.vy = y_hit
        est.t = t_hit
        self._actualizar_altitud_y_rapidez()
        ev.impacto_lunar_registrado = True
        ev.datos_impacto_lunar = {
            "t": float(t_hit),
            "v_rel": float(v_rel),
            "d_centro": float(math.hypot(y_hit[0] - lx, y_hit[1] - ly)),
        }
        zl = self._z_luna(t_hit)
        ev.datos_impacto_lunar["z_luna"] = zl
        ev.datos_impacto_lunar["fisico_3d"] = (
            None if zl is None else bool(abs(zl) <= self.config.LUNAR_Z_MAX_FISICO))
        ev.d_luna_min = min(ev.d_luna_min, R_LUNA)
        ev.datos_mision["resultado"] = "impacto_lunar"
        ev.datos_mision["t_impacto_lunar"] = float(t_hit)
        if self.config.LUNAR_LOI and ev.mision_cumplida is None:
            ev.mision_cumplida = False
            ev.datos_mision["motivo"] = ("impacto lunar sin captura" if self._loi is None
                                         else "impacto lunar durante/después de la LOI")
        # El encuentro SOI puede haber establecido CUMPLIDA; el impacto es
        # el final físico de la propagación, no una captura orbital.
        est.simulacion_terminada = True
        print(f"💥 IMPACTO LUNAR en t={t_hit / 3600:.3f} h: "
              f"v_rel={v_rel:.0f} m/s, r={ev.datos_impacto_lunar['d_centro'] / 1000:.1f} km "
              f"(radio lunar={R_LUNA / 1000:.1f} km). Simulación terminada por contacto."
              f"{_nota_3d(ev.datos_impacto_lunar)}")

    def _seguimiento_lunar(self) -> None:
        """r máx, distancia mínima a la Luna y fin por T_MAX_LUNAR."""
        est, ev = self.estado, self.eventos
        if est.fase in ("quema_tli", "translunar", "quema_loi", "orbita_lunar"):
            lx, ly = self._posicion_cuerpo("luna", est.t)
            d = math.hypot(est.px - lx, est.py - ly)
            self._lunar_d_luna = d
            if d < ev.d_luna_min:
                ev.d_luna_min = d
            r_alt = math.hypot(est.px, est.py) - R_T
            if r_alt > ev.r_max_eci:
                ev.r_max_eci = r_alt
        if (ev.entro_soi_luna and not est.simulacion_terminada
                and est.t - self._t_log_lunar >= 6.0 * 3600.0):
            self._t_log_lunar = est.t
            print(f"… Seguimiento post-SOI: t={est.t / 3600:.1f} h, "
                  f"distancia a la Luna={self._lunar_d_luna / 1000:.0f} km, "
                  f"paso={est.dt:.0f} s")
        if (est.t >= self.config.T_MAX_LUNAR and not est.simulacion_terminada):
            if self.eventos.mision_cumplida is None:
                self.eventos.mision_cumplida = False
                self.eventos.datos_mision["motivo"] = (
                    "T_MAX sin completar la captura lunar"
                    if self.config.LUNAR_LOI and ev.entro_soi_luna
                    else "T_MAX sin SOI" if self._lunar_estado == "translunar"
                    else "T_MAX sin completar el corte TLI")
                self.eventos.datos_mision["fin_por_t_max"] = True
            est.simulacion_terminada = True
            print(f"⏱ Fin por T_MAX_LUNAR ({est.t / 86400:.1f} d)")

    # ========================================================
    # DESÓRBITA
    # ========================================================
    def _actualizar_deorbit(self, ctx: ContextoMision) -> bool:
        """Devuelve True cuando deorbit ya controla la fase (activo)."""
        est = self.estado
        if not self._deorbit_activo:
            if not self._debe_activar_deorbit():
                return False
            self._activar_deorbit(ctx)

        if est.fase == "deorbit":
            self._gestionar_quema_deorbit(ctx)
        else:
            self._confirmar_reentrada(ctx)
        return True

    def _debe_activar_deorbit(self) -> bool:
        cfg = self.config
        est = self.estado
        if cfg.deorbit_activacion == "t":
            return est.t >= cfg.deorbit_t_activacion
        if cfg.deorbit_activacion == "sitio":
            return (self._deorbit_t_plan is not None
                    and est.t >= self._deorbit_t_plan)
        return self.eventos.orbita_detectada and est.fase == "coasting"

    # --- Desórbita al sitio: la hora de quema se busca con copias ---
    def _debe_planificar_deorbit(self) -> bool:
        return (self._deorbit_t_plan is None
                and self.eventos.orbita_detectada
                and self.estado.fase == "coasting")

    def _resolver_plan_deorbit(self) -> bool:
        """Obtiene la hora de quema. True: ya está en `_deorbit_t_plan`.
        False: hay un planificador externo que aún no terminó; el llamador
        no debe avanzar la física este paso.

        El plan externo se acepta solo si el hijo planificó desde EXACTAMENTE
        el mismo estado (t, px, py, vx, vy; igualdad de floats, sin
        tolerancia): el ascenso es determinista, así que lo normal es que
        coincida. Si no coincide (otra versión de numba/numpy, un cambio de
        física en vuelo...) o el hijo falló, se descarta y se planifica aquí."""
        if self._plan_gen is not None:           # planificación por tramos en curso
            return self._avanzar_plan_por_tramos()
        pl = self._planificador_externo
        if pl is None:
            return self._planificar_en_este_hilo("sin proceso aparte")
        res = pl.consultar()
        if res is None:
            return False
        est = self.estado
        self._planificador_externo = None
        espera = pl.transcurrido()
        if res[0] == "ok" and res[2] == (est.t, est.px, est.py, est.vx, est.vy):
            self._deorbit_t_plan = res[1]
            for linea in res[3]:
                print(f"{linea}  [planificada en un proceso aparte]")
            t_asc, t_tot = res[4]
            print(f"⏱ Plan externo recibido a los {espera:.1f} s del lanzamiento "
                  f"(hijo: ascenso {t_asc:.1f} s, ascenso+plan {t_tot:.1f} s)")
            return True
        if res[0] == "error":
            motivo = res[1]
        else:
            motivo = "el estado de planificación no coincide con el de la simulación"
        print(f"⚠ Plan en proceso aparte descartado ({motivo}); se planifica aquí. "
              f"Espera previa: {espera:.1f} s")
        if res[0] == "ok":
            print(f"    padre: {(est.t, est.px, est.py, est.vx, est.vy)!r}")
            print(f"    hijo:  {res[2]!r}")
        return self._planificar_en_este_hilo("proceso aparte descartado")

    def _planificar_en_este_hilo(self, motivo: str) -> bool:
        """Planifica en el hilo que llama. Con `presupuesto_plan_s` (GUI) lo
        hace por tramos y devuelve False hasta terminar (la GUI sigue
        refrescándose); sin él, completo de una vez (True)."""
        if self.presupuesto_plan_s is not None:
            print(f"⏳ Planificando la desórbita en este hilo, por tramos de "
                  f"{self.presupuesto_plan_s * 1000:.0f} ms ({motivo}); la "
                  f"física espera, la GUI sigue activa")
            self._plan_gen = self._planificar_deorbit_sitio_gen()
            self._plan_t0 = _monotonic()
            return self._avanzar_plan_por_tramos()
        print(f"⏳ Planificando la desórbita de una vez en este hilo ({motivo}); "
              f"no hay GUI que atender")
        t_loc = _monotonic()
        self._planificar_deorbit_sitio()
        print(f"⏱ Planificación completa: {_monotonic() - t_loc:.1f} s")
        return True

    # (12.13.4) Tope del plan por tramos [s]; el nominal (TLI o desórbita al sitio) tarda 30-120 s.
    PLAN_TIMEOUT_S = 900.0

    def _fallar_planificacion(self, motivo: str) -> None:
        """Cierra la misión de forma terminal si vence un planificador."""
        self._plan_gen = None
        self._planificador_externo = None
        self._lunar_plan_pendiente = False
        self.esperando_plan = False
        self.eventos.datos_mision["fallo_planificacion"] = True
        if self.config.modo_mision == "lunar":
            self._lunar_fallar(motivo)
            return
        if self.eventos.mision_cumplida is None:
            self.eventos.mision_cumplida = False
        self.eventos.datos_mision["motivo"] = motivo
        self.estado.simulacion_terminada = True
        getattr(self, "_log", print)(f"✗ PLANIFICACIÓN: {motivo}. Misión terminada como NO CUMPLIDA.")

    def _avanzar_plan_por_tramos(self) -> bool:
        """Trabaja `presupuesto_plan_s` segundos. True al concluir, incluso si falla.
        El timeout registra NO CUMPLIDA y deja la simulación en estado terminal."""
        transcurrido = _monotonic() - self._plan_t0
        if transcurrido >= self.PLAN_TIMEOUT_S:
            motivo = (f"la planificación superó {self.PLAN_TIMEOUT_S:.0f} s "
                      "sin terminar")
            self._fallar_planificacion(motivo)
            return True
        limite = _monotonic() + max(0.0, self.presupuesto_plan_s or 0.0)
        try:
            while True:
                next(self._plan_gen)
                if _monotonic() >= limite:
                    return False
        except StopIteration:
            self._plan_gen = None
            print(f"⏱ Planificación por tramos: {_monotonic() - self._plan_t0:.1f} s")
            return True

    def _error_angular_base(self) -> float:
        """Ángulo (rad, con signo, en (-π, π]) entre la posición actual y
        la base de lanzamiento. La base está en π/2 en t=0 y gira con la
        rotación efectiva (misma convención que el dibujo de la grilla)."""
        est = self.estado
        d = (math.atan2(est.py, est.px)
             - (math.pi / 2.0 - self._Omega_tierra * est.t))
        return (d + math.pi) % (2.0 * math.pi) - math.pi

    def _error_base_gen(self, t_quema: float):
        """Generador (12.1.2): error angular respecto de la base al IMPACTO si
        la quema de desórbita empieza en `t_quema`; None si no hay impacto
        dentro del horizonte. Simula en una copia profunda (la simulación real
        no cambia) con la misma física, por lo que predice sin modelo aparte
        la quema finita, el arrastre y el cambio de masa. Cede cada
        `_PLAN_PASOS_POR_TRAMO` pasos; el valor final es el `return`."""
        gen_, pl_ = self._plan_gen, self._planificador_externo
        self._plan_gen = self._planificador_externo = None   # no copiables
        try:
            clon = copy.deepcopy(self)
        finally:
            self._plan_gen, self._planificador_externo = gen_, pl_
        clon._deorbit_sitio = False
        clon.config.deorbit_activacion = "t"
        clon.config.deorbit_t_activacion = t_quema
        horizonte = t_quema + self.config.DEORBIT_SITIO_HORIZONTE
        while (not clon.estado.simulacion_terminada
               and clon.estado.t < horizonte):
            with contextlib.redirect_stdout(io.StringIO()):
                for _ in range(_PLAN_PASOS_POR_TRAMO):
                    if (clon.estado.simulacion_terminada
                            or clon.estado.t >= horizonte):
                        break
                    clon.avanzar_paso()
            yield
        if not clon.eventos.impacto_registrado:
            return None
        return clon._error_angular_base()

    @staticmethod
    def _correr_generador(gen):
        """Ejecuta un generador hasta el final y devuelve su `return`."""
        try:
            while True:
                next(gen)
        except StopIteration as e:
            return e.value

    def _error_base_si_quema_en(self, t_quema: float) -> Optional[float]:
        return self._correr_generador(self._error_base_gen(t_quema))

    def _planificar_deorbit_sitio(self) -> None:
        """Planificación completa de una vez (scripts, regresión, proceso hijo)."""
        self._correr_generador(self._planificar_deorbit_sitio_gen())

    def _planificar_deorbit_sitio_gen(self):
        """Busca la hora de quema que deja la caída sobre la base. El error
        angular baja de forma continua con la hora de quema y cruza cero
        una vez por órbita: se barre un período en 8 pasos hasta encontrar
        el cruce y se refina con regula falsi (Illinois)."""
        cfg = self.config
        est = self.estado
        orb = elementos_orbitales(est.px, est.py, est.vx, est.vy)
        periodo = 2.0 * math.pi * math.sqrt(orb["a"] ** 3 / MU_T)
        tol = cfg.DEORBIT_SITIO_TOL_KM * 1000.0 / R_T
        evaluaciones = 0

        t0 = est.t + 5.0
        paso = periodo / 8.0
        a = fa = b = fb = None
        t_prev = d_prev = None
        for k in range(9):
            t = t0 + k * paso
            evaluaciones += 1
            d = yield from self._error_base_gen(t)
            if (d is not None and d_prev is not None
                    and d_prev > 0.0 > d and abs(d_prev - d) < math.pi):
                a, fa, b, fb = t_prev, d_prev, t, d
                break
            t_prev, d_prev = t, d
        mejor = None                         # (|error|, t)
        if a is not None:
            mejor = min((abs(fa), a), (abs(fb), b))
            for _ in range(cfg.DEORBIT_SITIO_MAX_ITER):
                if mejor[0] < tol:
                    break
                c = b - fb * (b - a) / (fb - fa)
                evaluaciones += 1
                fc = yield from self._error_base_gen(c)
                if fc is None:
                    break
                mejor = min(mejor, (abs(fc), c))
                if fc * fb < 0.0:
                    a, fa = b, fb
                else:
                    fa *= 0.5
                b, fb = c, fc
        if mejor is None:
            self._deorbit_t_plan = est.t
            print("⚠ Desórbita al sitio: no se encontró una ventana de quema; "
                  "se desorbita de inmediato (no caerá en la base)")
            return
        self._deorbit_t_plan = mejor[1]
        print(f"🎯 Desórbita a la base: quema en t={mejor[1]:.0f} s "
              f"(en {mejor[1] - est.t:.0f} s); error previsto "
              f"{mejor[0] * R_T / 1000.0:.1f} km ({evaluaciones} simulaciones)")

    def _activar_deorbit(self, ctx: ContextoMision) -> None:
        est = self.estado
        etapa = ctx.etapa
        orb = ctx.elementos
        self._deorbit_activo = True
        self._deorbit_v_ref = est.speed
        self._deorbit_dv_prop_ref = etapa.dv_prop if etapa is not None else 0.0
        self.eventos.datos_mision.update({
            "deorbit": True, "t_activacion": est.t,
            "alt_activacion": est.altitud, "rp_activacion": orb["rp"],
            "ra_activacion": orb["ra"], "v_activacion": est.speed,
            "comb_activacion": (etapa.m_comb_restante
                                if etapa is not None else 0.0),
        })
        if etapa is not None and ctx.tiene_combustible:
            self._encender_etapa(etapa)
        self._fijar_fase("deorbit")
        print(f"🛬 DESÓRBITA ACTIVADA en t={est.t:.0f} s "
              f"(alt={est.altitud / 1000:.0f} km, rp={orb['rp'] / 1000:.0f} km)")

    def _gestionar_quema_deorbit(self, ctx: ContextoMision) -> None:
        """Quema retrógrada: corta al llegar al periapsis objetivo o al
        quedarse sin combustible."""
        cfg = self.config
        est = self.estado
        ev = self.eventos
        etapa = ctx.etapa
        orb = ctx.elementos
        periapsis_logrado = orb["rp"] <= cfg.DEORBIT_RP_OBJETIVO
        sin_combustible = (etapa is None or not etapa.activa
                           or etapa.m_comb_restante <= 0.0)
        if not (periapsis_logrado or sin_combustible):
            return
        if etapa is not None:
            etapa.activa = False
        self._fijar_fase("coasting")
        ev.datos_mision.update({
            "t_cutoff": est.t, "rp_cutoff": orb["rp"],
            "ra_cutoff": orb["ra"], "ecc_cutoff": orb["ecc"],
            "dv_prop": ((etapa.dv_prop - self._deorbit_dv_prop_ref)
                        if etapa is not None else 0.0),
            "delta_rapidez": self._deorbit_v_ref - est.speed,
            "rp_logrado": bool(periapsis_logrado),
        })
        print(f"✂ CUTOFF DESÓRBITA: rp={orb['rp'] / 1000:.1f} km, "
              f"Δv motor≈{ev.datos_mision['dv_prop']:.0f} m/s "
              f"(Δrapidez {self._deorbit_v_ref - est.speed:.0f} m/s)")
        if not periapsis_logrado and ev.mision_cumplida is None:
            ev.mision_cumplida = False
            ev.datos_mision["motivo"] = "sin combustible"

    def _confirmar_reentrada(self, ctx: ContextoMision) -> None:
        """Coast tras el corte: confirma la reentrada al cruzar la altitud."""
        cfg = self.config
        est = self.estado
        ev = self.eventos
        datos = ev.datos_mision
        if ("reentrada_t" in datos or not datos.get("rp_logrado")
                or est.altitud > cfg.DEORBIT_ALT_REENTRADA
                or not ctx.v_radial < 0.0):
            return
        datos.update({"reentrada_t": est.t, "reentrada_v": est.speed,
                      "reentrada_alt": est.altitud})
        if ev.mision_cumplida is None:
            ev.mision_cumplida = True
        print(f"🔥 REENTRADA CONFIRMADA en t={est.t:.0f} s "
              f"(v={est.speed:.0f} m/s)")

    # ========================================================
    # CRITERIO DE ÉXITO SUBORBITAL
    # ========================================================
    def _evaluar_mision_suborbital(self) -> None:
        """Éxito si alt_max >= APOGEO_OBJETIVO - TOL_APOGEO_SUBORBITAL y
        alt_max > Kármán.
        Se evalúa una sola vez: al registrarse el apogeo físico (o,
        como respaldo, en el impacto)."""
        cfg = self.config
        ev = self.eventos
        if cfg.modo_mision != "suborbital" or ev.mision_cumplida is not None:
            return
        ev.mision_cumplida = bool(
            ev.alt_max >= cfg.APOGEO_OBJETIVO - cfg.TOL_APOGEO_SUBORBITAL
            and ev.alt_max > cfg.KARMAN)
        ev.datos_mision = {
            "alt_max": ev.alt_max,
            "objetivo": cfg.APOGEO_OBJETIVO,
            "cruzo_karman": ev.alt_max > cfg.KARMAN,
        }

    # ========================================================
    # GESTIÓN DEL HISTORIAL ORBITAL
    # ========================================================
    def _gestionar_historial_orbital(self) -> None:
        """
        1. Snapshot de la primera vuelta completa (para gráficas finales).
        2. Poda del inicio de la sección orbital (para acotar memoria).
        """
        if not self.eventos.orbita_detectada:
            return
        if self._idx_inicio_orbita is None or self._periodo_orbital is None:
            return

        if self._snapshot_orbita is None:
            self._tomar_snapshot_si_completo_una_vuelta()
        else:
            self._podar_historial_orbital()

    def _tomar_snapshot_si_completo_una_vuelta(self) -> None:
        est = self.estado
        t_orbita = self.eventos.datos_orbita["t"]
        if est.t - t_orbita < self._periodo_orbital:
            return
        snapshot = {nombre: getattr(self, nombre)[:]
                    for nombre in _SERIES_HISTORIAL_ARRAY}
        snapshot.update({nombre: list(getattr(self, nombre))
                         for nombre in _SERIES_HISTORIAL_FASE})
        snapshot.update({
            "t_orbita":     t_orbita,
            "periodo":      self._periodo_orbital,
            "datos_orbita": dict(self.eventos.datos_orbita),
        })
        self._snapshot_orbita = snapshot
        print("📸 Snapshot de primera vuelta orbital guardado "
              f"(t = {est.t:.1f} s)")

    def _podar_historial_orbital(self) -> None:
        """Conserva solo la última vuelta orbital del historial en vivo."""
        est = self.estado
        inicio = self._idx_inicio_orbita
        n_orbital = len(self.hist_t) - inicio
        n_por_periodo = self._periodo_orbital / max(est.dt, 1e-6)
        if n_orbital < 1.5 * n_por_periodo:
            return

        t_min = est.t - self._periodo_orbital
        fin = inicio
        while fin < len(self.hist_t) and self.hist_t[fin] < t_min:
            fin += 1
        if fin <= inicio:
            return

        self._guardar_resumen_podado(inicio, fin)
        self._almacen.anexar(                            # (12.12.0) nada se pierde: se escribe antes de borrar
            inicio, self.hist_t[inicio:fin], self.hist_alt[inicio:fin], self.hist_v[inicio:fin],
            self.tray_wx[inicio:fin], self.tray_wy[inicio:fin],
            self.tray_ex[inicio:fin], self.tray_ey[inicio:fin], self.hist_fase[inicio:fin])
        for nombre in _SERIES_HISTORIAL_ARRAY + _SERIES_HISTORIAL_FASE:
            del getattr(self, nombre)[inicio:fin]
        self._orbita_podada = True

    def historial_completo(self) -> dict:
        """(12.12.0) Todo el historial en orden: ascenso + lo podado (leído del disco) + lo que sigue
        en RAM. Claves: hist_t, hist_alt, hist_v, tray_wx, tray_wy, tray_ex, tray_ey (arrays),
        hist_fase (lista de str) e `incompleto` (True si el disco falló y faltan tramos)."""
        alm = self._almacen
        nombres = (("hist_t", 0), ("hist_alt", 1), ("hist_v", 2), ("tray_wx", 3),
                   ("tray_wy", 4), ("tray_ex", 5), ("tray_ey", 6))
        ram = {n: np.asarray(getattr(self, n), dtype=float) for n, _ in nombres}
        disco = alm.leer() if (alm.ok and alm.n) else np.empty((0, 8))
        trozos = {n: [] for n, _ in nombres}
        fases: list = []
        cursor = 0
        for pos, a, b in (alm.segmentos if len(disco) else []):
            for n, _ in nombres:
                trozos[n].append(ram[n][cursor:pos])
            fases.extend(self.hist_fase[cursor:pos])
            for n, j in nombres:
                trozos[n].append(disco[a:b, j])
            fases.extend(alm.nombres_fase(disco[a:b, 7]))
            cursor = pos
        for n, _ in nombres:
            trozos[n].append(ram[n][cursor:])
        fases.extend(self.hist_fase[cursor:])
        salida = {n: np.concatenate(trozos[n]) for n, _ in nombres}
        salida["hist_fase"] = fases
        salida["incompleto"] = not alm.ok
        return salida

    def cerrar_historial(self) -> None:
        """Borra el archivo temporal del historial podado (lo podado deja de estar disponible)."""
        self._almacen.cerrar()

    RESUMEN_MAX_PUNTOS = 20000
    RESUMEN_PUNTOS_POR_PERIODO = 60

    def _guardar_resumen_podado(self, inicio: int, fin: int) -> None:
        """(12.11.1) Antes de borrar hist_*[inicio:fin], conserva un punto cada
        periodo/60 s de misión para que la gráfica de vuelo no tenga hueco. Si el resumen
        supera RESUMEN_MAX_PUNTOS se diezma a la mitad y se duplica el espaciado."""
        if self._idx_resumen is None:
            self._idx_resumen = inicio              # el resumen se empalma aquí (índice estable)
        if self._resumen_paso_t is None:
            self._resumen_paso_t = max(self._periodo_orbital or 0.0, 0.0) / self.RESUMEN_PUNTOS_POR_PERIODO
        for i in range(inicio, fin):
            t = self.hist_t[i]
            if t >= self._resumen_proximo_t:
                self.hist_resumen_t.append(t)
                self.hist_resumen_alt.append(self.hist_alt[i])
                self.hist_resumen_v.append(self.hist_v[i])
                self._resumen_proximo_t = t + self._resumen_paso_t
        if len(self.hist_resumen_t) > self.RESUMEN_MAX_PUNTOS:
            for lista in (self.hist_resumen_t, self.hist_resumen_alt, self.hist_resumen_v):
                del lista[::2]
            self._resumen_paso_t *= 2.0


# ============================================================
# CAPA 5 — INTERFAZ GRÁFICA (tkinter + matplotlib)
# Solo dibuja y lee la configuración: nunca contiene física. Pide
# todo a `Simulacion` y a las funciones puras de las capas anteriores.
# ============================================================

# --- Paleta y tipografía (un solo lugar para el "look") ---
COLOR_FONDO_VENTANA = "#050a12"
COLOR_FONDO_PANEL   = "#0a0e14"
COLOR_FONDO_ENTRADA = "#001428"
COLOR_TEXTO_ETIQUETA = "#8090a0"
COLOR_TEXTO_ACENTO  = "#00c8ff"
COLOR_TEXTO_ACTIVO  = "#40ff80"
COLOR_TEXTO_ERROR   = "#ff6060"
FUENTE_NORMAL = ("Courier New", 9)
FUENTE_NEGRITA = ("Courier New", 9, "bold")

# --- (12.8.0) Look de control de misión: barra superior y tarjetas ---
COLOR_TARJETA = "#0d141d"
COLOR_BORDE_TARJETA = "#1c2a3a"
COLOR_TITULO_TARJETA = "#e6edf5"
COLOR_VALOR = "#19d3ff"
COLOR_OK = "#2dff8a"
COLOR_AVISO = "#ffb347"
FUENTE_TITULO_TARJETA = ("Courier New", 10, "bold")
FUENTE_VALOR = ("Courier New", 13, "bold")
# (12.9.0) tipografía de los números: la primera disponible en el sistema
FUENTES_MONO_PREFERIDAS = ("Consolas", "DejaVu Sans Mono", "Menlo", "Courier New")
COLOR_EVENTO = {"titulo": "#ffe066", "exito": COLOR_OK, "fallo": COLOR_TEXTO_ERROR}
FUENTE_MET = ("Courier New", 22, "bold")
# color del valor según la etiqueta (tag) de cada fila del HUD
COLOR_POR_TAG = {"normal": COLOR_VALOR, "fase": "#ffe066", "empuje_on": "#ff9020",
                 "empuje_off": COLOR_OK, "alerta": COLOR_TEXTO_ERROR}
# tarjetas de telemetría: título -> filas (por nombre) que van en ella; lo que no esté
# en ninguna lista cae en la última (MISIÓN). Tiempo y fase ya están en la barra superior.
TARJETAS_TELEMETRIA = (
    ("TRAYECTORIA", ("Altitud:", "Velocidad:", "Apoapsis:", "Periapsis:", "Excentric.:",
                     "Alt. máx:", "r máx ECI:")),
    ("PROPULSIÓN", ("Empuje:", "Combustible:", "Carga g:")),
    ("MISIÓN", ()),
)
FILAS_EN_BARRA = ("Tiempo:", "Fase:")
# filas que muestran las tarjetas cuando no hay misión en curso
FILAS_REPOSO = [("Altitud:", "—", "normal"), ("Velocidad:", "—", "normal"),
                ("Apoapsis:", "—", "normal"), ("Periapsis:", "—", "normal"),
                ("Empuje:", "—", "normal"), ("Combustible:", "—", "normal"),
                ("Modo:", "—", "normal")]


def _entero_estricto(texto) -> int:
    """(12.9.1) «5», «5.0» o «1e3» valen; «1.9», «nan» e «inf» se rechazan (no se trunca)."""
    v = float(texto)
    if not math.isfinite(v) or v != int(v):
        raise ValueError(f"{texto!r} no es un entero")
    return int(v)


def _elegir_fuente_mono(disponibles) -> str:
    """(12.9.0) Primera familia de `FUENTES_MONO_PREFERIDAS` que exista; Courier New si ninguna."""
    try:
        existentes = {str(f) for f in disponibles}
    except TypeError:
        existentes = set()
    for familia in FUENTES_MONO_PREFERIDAS:
        if familia in existentes:
            return familia
    return "Courier New"


def _familias_tk(root) -> list:
    try:
        return list(root.tk.splitlist(root.tk.call("font", "families")))
    except Exception:
        return []


_RE_NUMERO_GRANDE = re.compile(r"^(-?)([1-9]\d{3,})(\.\d+)?(\s.*)?$")


def _con_miles(texto) -> str:
    """(12.9.0) «12345.6 m» -> «12,345.6 m». Solo si el valor EMPIEZA por un número con 4+ dígitos
    enteros seguido de espacio/fin: fechas, tiempos y textos quedan intactos."""
    texto = str(texto)
    m = _RE_NUMERO_GRANDE.match(texto)
    if not m:
        return texto
    signo, entero, decimal, resto = m.groups()
    return f"{signo}{int(entero):,}{decimal or ''}{resto or ''}"


def _formatear_met(t: float) -> str:
    """Tiempo de misión «+HH:MM:SS» (las horas crecen si pasan de 99)."""
    if not math.isfinite(t) or t < 0.0:
        t = 0.0
    horas, resto = divmod(int(t), 3600)
    minutos, segundos = divmod(resto, 60)
    return f"+{horas:02d}:{minutos:02d}:{segundos:02d}"


def _estado_global(sim) -> tuple:
    """(texto, color, detalle) del estado de la misión para la barra superior."""
    ev, est = sim.eventos, sim.estado
    if ev.mision_cumplida is True:
        if est.simulacion_terminada:
            return "MISIÓN CUMPLIDA", COLOR_OK, ""
        return "MISIÓN CUMPLIDA · SEGUIMIENTO", COLOR_OK, ""
    if ev.mision_cumplida is False:
        motivo = str(ev.datos_mision.get("motivo", "") or "")
        return "NO CUMPLIDA", COLOR_TEXTO_ERROR, (motivo if len(motivo) <= 60 else motivo[:59] + "…")
    if est.simulacion_terminada:
        if getattr(ev, "fallo_integrador", None):         # (12.10.1) terminó sin veredicto: se distingue por qué
            return "ABORTADA", COLOR_TEXTO_ERROR, "fallo del integrador"
        if getattr(ev, "impacto_registrado", False):
            return "IMPACTO", COLOR_TEXTO_ERROR, ""
        return "FINALIZADA", COLOR_TEXTO_ETIQUETA, ""
    if getattr(sim, "esperando_plan", False):
        return "PLANIFICANDO", COLOR_AVISO, ""
    return "EN CURSO", COLOR_OK, ""


def _agrupar_filas_en_tarjetas(filas: list) -> list:
    """[(título, [filas])] en el orden de TARJETAS_TELEMETRIA (siempre todas las
    tarjetas, aunque estén vacías). Las filas ya mostradas en la barra se omiten."""
    grupos = {titulo: [] for titulo, _ in TARJETAS_TELEMETRIA}
    por_nombre = {n: t for t, nombres in TARJETAS_TELEMETRIA for n in nombres}
    ultimo = TARJETAS_TELEMETRIA[-1][0]
    vistos = set()
    for fila in filas:
        if fila[0] in FILAS_EN_BARRA or fila[0] in vistos:
            continue          # (12.8.3) una fila repetida se muestra una sola vez (gana la primera)
        vistos.add(fila[0])
        grupos[por_nombre.get(fila[0], ultimo)].append(fila)
    return [(titulo, grupos[titulo]) for titulo, _ in TARJETAS_TELEMETRIA]


# ============================================================
# SUBMUESTREO PARA DIBUJO (la física guarda TODOS los pasos; la GUI no
# necesita dibujarlos todos). Coste por fotograma acotado por
# GUI_MAX_PUNTOS en vez de crecer con la longitud de la trayectoria.
# ============================================================
GUI_MAX_PUNTOS = 3000


def _paso_vista(n: int, max_pts: int = GUI_MAX_PUNTOS) -> int:
    """Salto de submuestreo para que n puntos queden en <= ~max_pts."""
    return max(1, -(-n // max_pts))


def _submuestrear(serie, paso: int):
    """serie[::paso] incluyendo siempre el último elemento."""
    salida = serie[::paso]
    if (len(serie) - 1) % paso:
        salida.append(serie[-1])
    return salida


def _vista_al_final(yview) -> bool:
    """(12.11.0) True si una vista (primero, ultimo) de un Canvas llega al final (o no hay nada
    que desplazar). Tolera que el widget no responda con una tupla."""
    try:
        return float(yview[1]) >= 0.999
    except (TypeError, ValueError, IndexError):
        return True


def _tierra_en_vista(xlim, ylim, radio: float, margen: float) -> bool:
    """(12.11.0) True si el disco de la Tierra (centrado en el origen, radio*margen) cabe
    entero en los límites del mapa."""
    r = radio * margen
    return (min(xlim) <= -r and max(xlim) >= r
            and min(ylim) <= -r and max(ylim) >= r)


EXPORT_COLUMNAS_CSV = ("t_s", "altitud_m", "velocidad_ms", "x_m", "y_m", "baricentro_x_m",
                       "baricentro_y_m", "fase")


def _carpeta_proyecto() -> Path:
    """(12.12.1) Carpeta donde vive este archivo (destino de las exportaciones automáticas)."""
    return Path(__file__).resolve().parent


def _ruta_exportacion(modo: str, extension: str = ".npz", carpeta=None, ahora=None) -> Path:
    """(12.12.1) `historial_<modo>_<AAAAMMDD_HHMMSS><ext>` en `carpeta` (por defecto, la del proyecto);
    si existe se añade _2, _3… para no sobrescribir."""
    carpeta = Path(carpeta) if carpeta is not None else _carpeta_proyecto()
    ahora = ahora or datetime.now()
    seguro = re.sub(r"[^A-Za-z0-9_-]+", "_", str(modo)) or "mision"
    base = f"historial_{seguro}_{ahora:%Y%m%d_%H%M%S}"
    ruta = carpeta / f"{base}{extension}"
    n = 2
    while ruta.exists():
        ruta = carpeta / f"{base}_{n}{extension}"
        n += 1
    return ruta


def exportar_historial(sim, ruta) -> str:
    """(12.12.0) Escribe el historial completo (también lo podado) en `ruta`: .csv (una fila por
    paso) o .npz (series + metadatos en JSON). Devuelve la ruta. Lanza OSError si no puede escribir."""
    datos = sim.historial_completo()
    ruta = str(ruta)
    ext = Path(ruta).suffix.lower()
    if ext == ".csv":
        import csv
        with open(ruta, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(EXPORT_COLUMNAS_CSV)
            cols = [datos[k] for k in ("hist_t", "hist_alt", "hist_v", "tray_wx", "tray_wy",
                                       "tray_ex", "tray_ey")]
            for i, fase in enumerate(datos["hist_fase"]):
                w.writerow([repr(float(c[i])) for c in cols] + [fase])
        return ruta
    meta = {"modo_mision": sim.config.modo_mision, "incompleto": datos["incompleto"],
            "datos_orbita": dict(sim.eventos.datos_orbita or {}),
            "periodo_orbital": sim._periodo_orbital,
            "config": asdict(sim.config)}
    if not ruta.lower().endswith(".npz"):
        ruta += ".npz"
    np.savez(ruta, hist_t=datos["hist_t"], hist_alt=datos["hist_alt"], hist_v=datos["hist_v"],
             tray_wx=datos["tray_wx"], tray_wy=datos["tray_wy"], tray_ex=datos["tray_ex"],
             tray_ey=datos["tray_ey"], hist_fase=np.array(datos["hist_fase"], dtype=str),
             meta=np.array(json.dumps(meta, default=str)))
    return ruta


def agrupar_en_polilineas(segmentos, colores):
    """(12.13.0) Une segmentos consecutivos (el final de uno es el inicio del siguiente) del mismo
    color en polilíneas. `segmentos` (n, 2, 2) y `colores` (n, 4) → (lista de arrays (k, 2),
    colores (m, 4)). Equivale a dibujar los n segmentos, con muchos menos objetos."""
    segmentos = np.asarray(segmentos, dtype=float)
    colores = np.asarray(colores, dtype=float)
    n = len(segmentos)
    if n == 0:
        return [], colores[:0]
    corte = np.zeros(n - 1, dtype=bool)
    if n > 1:
        corte |= ~np.all(segmentos[:-1, 1] == segmentos[1:, 0], axis=1)     # discontinuidad
        corte |= np.any(colores[:-1] != colores[1:], axis=1)                # cambio de color
    inicios = np.concatenate(([0], np.nonzero(corte)[0] + 1))
    fines = np.concatenate((inicios[1:], [n]))
    inicio_pts = segmentos[:, 0, :]
    polilineas = [np.vstack((inicio_pts[a:b], segmentos[b - 1, 1])) for a, b in zip(inicios, fines)]
    return polilineas, colores[inicios]


def _serie_con_resumen(hist, resumen, idx: int, paso: int, paso_resumen: int = 1) -> list:
    """(12.11.1) hist[:idx] + resumen + hist[idx:] como lista; el historial se submuestrea con
    `paso` y el resumen (ya ralo: ~60 puntos por periodo) con `paso_resumen`. Cada tramo conserva
    su último elemento."""
    partes = []
    for tramo, paso in ((hist[:idx], paso), (resumen, paso_resumen), (hist[idx:], paso)):
        if len(tramo):
            partes.extend(tramo[::paso])                 # array o lista: sin copiar el tramo entero
            if (len(tramo) - 1) % paso:
                partes.append(tramo[-1])
    return partes


def _serie_con_corte(serie, j: int) -> list:
    """(12.10.3) `serie` como lista con un NaN en la posición `j` (rompe la línea al dibujar)."""
    lista = list(serie)
    return lista[:j] + [float("nan")] + lista[j:]


def _indices_vista(n: int, paso: int) -> np.ndarray:
    seleccion = np.arange(0, n, paso)
    if seleccion[-1] != n - 1:
        seleccion = np.append(seleccion, n - 1)
    return seleccion


def construir_segmentos_trayectoria(wx, wy, wfase, ex, ey, escala_log: bool,
                                    en_baricentro: bool,
                                    idx_corte, podada: bool,
                                    rgba_por_fase: dict,
                                    max_pts: int = GUI_MAX_PUNTOS):
    """
    Segmentos y colores de la trayectoria para dibujar, submuestreados.
    Con n <= max_pts es idéntico al dibujo completo anterior. El color
    de cada segmento es el de la fase de su punto final. Si la sección
    orbital fue podada, se omite el segmento que une el ascenso con la
    órbita actual (la "cuerda" que cruzaría la elipse).
    """
    n = len(wx)
    paso = _paso_vista(n, max_pts)
    seleccion = _indices_vista(n, paso)
    tx, ty = vista_xy(np.asarray(_submuestrear(wx, paso)),
                      np.asarray(_submuestrear(wy, paso)), escala_log)
    if en_baricentro:
        bx, by = vista_xy(np.asarray(_submuestrear(ex, paso)),
                          np.asarray(_submuestrear(ey, paso)), escala_log)
        tx, ty = tx - bx, ty - by
    puntos = np.column_stack((tx, ty))
    segmentos = np.stack([puntos[:-1], puntos[1:]], axis=1)
    fases = _submuestrear(wfase, paso)[1:]
    colores = np.array([rgba_por_fase.get(f, rgba_por_fase["_otro"])
                        for f in fases])
    if podada and idx_corte is not None and 0 < idx_corte < n:
        j = int(np.searchsorted(seleccion, idx_corte, side="left")) - 1
        if 0 <= j < len(segmentos):
            conservar = np.ones(len(segmentos), dtype=bool)
            conservar[j] = False
            segmentos = segmentos[conservar]
            colores = colores[conservar]
    return segmentos, colores


def _etiqueta_modo(config: Config) -> str:
    """Texto del modo de misión para HUD y gráficas ("tripulada/bcb", ...)."""
    if config.modo_mision == "tripulada":
        return f"tripulada/{config.perfil_tripulada}"
    return config.modo_mision


def _formatear_altitud(alt_m: float) -> str:
    return (f"{alt_m / 1000.0:.1f} km"
            if abs(alt_m) >= 1000.0 else f"{alt_m:.0f} m")


# ============================================================
# HELPERS DE WIDGETS
# ============================================================
def _estilo_seleccion() -> dict:
    """Estilo común de Checkbutton/Radiobutton del panel."""
    return dict(bg=COLOR_FONDO_PANEL, fg=COLOR_TEXTO_ACENTO,
                selectcolor=COLOR_FONDO_ENTRADA,
                activebackground=COLOR_FONDO_PANEL,
                activeforeground=COLOR_TEXTO_ACTIVO,
                font=FUENTE_NORMAL)


def _crear_checkbutton(parent, texto: str, variable, **opciones):
    return tk.Checkbutton(parent, text=texto, variable=variable,
                          **_estilo_seleccion(), **opciones)


def _crear_radiobuttons(parent, opciones: list, variable, **extra_pack):
    """`opciones`: lista de (valor, etiqueta). Un Radiobutton por opción."""
    for valor, etiqueta in opciones:
        tk.Radiobutton(parent, text=etiqueta, variable=variable, value=valor,
                       **_estilo_seleccion()).pack(anchor="w", **extra_pack)


# ============================================================
# DESÓRBITA AL SITIO: PLANIFICACIÓN EN UN PROCESO APARTE
# ============================================================
# Por qué un proceso y no un hilo: la planificación son varias
# simulaciones completas en Python (solve_ivp) y un hilo compartiría el
# GIL con la GUI (tirones), `contextlib.redirect_stdout` (global al
# proceso: se tragaría los print del hilo principal y dos solapados dejan
# sys.stdout roto) y la copia en vivo de la simulación (puede salir
# incoherente). El proceso hijo construye su propia Simulacion desde la
# Config, no comparte memoria y se cancela con terminate().
def _plan_deorbit_en_proceso(config: Config, conn) -> None:
    """Cuerpo del proceso hijo. Va a nivel de módulo para poder importarse
    con "spawn" (Windows/macOS). Recorre el mismo ascenso que la simulación
    principal y planifica con el camino de siempre (`avanzar_paso`). Envía:
    ("ok", t_plan, (t, px, py, vx, vy) al planificar, [mensajes]) o
    ("error", texto)."""
    try:
        t_ini = _monotonic()
        sim = Simulacion(config)
        buf = io.StringIO()
        estado_plan = None
        t_ascenso = None
        with contextlib.redirect_stdout(buf):       # solo afecta a ESTE proceso
            while not sim.estado.simulacion_terminada:
                if estado_plan is None and sim._debe_planificar_deorbit():
                    e = sim.estado
                    estado_plan = (e.t, e.px, e.py, e.vx, e.vy)
                    t_ascenso = _monotonic() - t_ini
                sim.avanzar_paso()
                if sim._deorbit_t_plan is not None:
                    break
        if sim._deorbit_t_plan is None or estado_plan is None:
            conn.send(("error", "la simulación terminó sin llegar a planificar"))
        else:
            mensajes = [m for m in buf.getvalue().splitlines() if "Desórbita" in m]
            tiempos = (t_ascenso, _monotonic() - t_ini)     # (ascenso, ascenso+plan) en el hijo
            conn.send(("ok", sim._deorbit_t_plan, estado_plan, mensajes, tiempos))
    except BaseException as e:                      # el padre decide qué hacer
        try:
            conn.send(("error", f"{type(e).__name__}: {e}"))
        except Exception:
            pass
    finally:
        conn.close()


class PlanificadorDeorbitExterno:
    """Lanza `_plan_deorbit_en_proceso` y deja consultar el resultado sin
    bloquear. Dueño: la GUI (no vive dentro de Simulacion, así que
    Simulacion sigue siendo copiable con deepcopy)."""
    TIMEOUT_S: float = 300.0

    @staticmethod
    def aplicable(config: Config) -> bool:
        """Solo si el ascenso es reproducible en otro proceso: sin terceros
        cuerpos ni marco móvil (sin SPICE ni red, que podrían dar otra cosa)."""
        return (config.modo_mision == "deorbit"
                and config.deorbit_activacion == "sitio"
                and not config.usar_terceros_cuerpos
                and not config.usar_marco_movil)

    def __init__(self, config: Config, objetivo=None,
                 timeout_s: Optional[float] = None):
        ctx = multiprocessing.get_context("spawn")
        self._rx, tx = ctx.Pipe(duplex=False)
        self._proc = ctx.Process(
            target=objetivo or _plan_deorbit_en_proceso,
            args=(copy.deepcopy(config), tx), daemon=True, name="deorbit-sitio")
        self._timeout_s = self.TIMEOUT_S if timeout_s is None else timeout_s
        self._resultado = None
        self._cerrado = False
        self._proc.start()
        tx.close()      # el padre solo lee: si el hijo muere, recv() da EOF en vez de colgar
        self._t0 = _monotonic()

    def listo(self) -> bool:
        """True si el hijo ya respondió (o falló). No bloquea; el resultado
        queda guardado y `consultar()` lo devuelve después."""
        return self.consultar() is not None

    def transcurrido(self) -> float:
        """Segundos de reloj desde que se lanzó el hijo."""
        return _monotonic() - self._t0

    def consultar(self):
        """No bloquea. None = pendiente; si no, el resultado final (las
        llamadas siguientes devuelven lo mismo)."""
        if self._resultado is not None or self._cerrado:
            return self._resultado
        try:
            if self._rx.poll():
                self._resultado = self._rx.recv()
            elif not self._proc.is_alive():
                self._resultado = ("error", f"el proceso terminó sin resultado "
                                            f"(código {self._proc.exitcode})")
            elif _monotonic() - self._t0 > self._timeout_s:
                self._resultado = ("error", f"sin respuesta tras {self._timeout_s:.0f} s")
        except (EOFError, OSError) as e:
            self._resultado = ("error", f"canal roto ({type(e).__name__})")
        if self._resultado is not None:
            self.cancelar()
        return self._resultado

    def cancelar(self) -> None:
        """Detiene al hijo (idempotente). El resultado ya recibido se conserva."""
        if self._cerrado:
            return
        self._cerrado = True
        if self._proc.is_alive():
            self._proc.terminate()
        self._proc.join(timeout=2.0)
        try:
            self._rx.close()
        except OSError:
            pass


# ============================================================
# VENTANAS DE LANZAMIENTO INTERPLANETARIAS (12.7.0)
# ============================================================
# Planificador independiente del integrador de la nave (no usa `Simulacion`):
# elementos keplerianos de Standish (JPL, 1800-2050) + Lambert (variables
# universales, una revolución, sentido prógrado, sin multi-revolución) + malla
# salida x llegada. Entrega fechas, v∞, C3 y Δv (patched conics, desde una
# órbita circular de estacionamiento del origen hasta una circular del
# destino); NO propaga la trayectoria ni simula el vuelo. Destinos: planetas
# (las lunas y asteroides quedan fuera). Ver NOTAS, «Ventanas de lanzamiento».
_JD_J2000 = 2451545.0
_EPOCA_J2000 = datetime(2000, 1, 1, 12, tzinfo=timezone.utc)   # = _JD_J2000

# GM [m³/s²], radio [m]; órbita de referencia (estacionamiento / captura) [m de altitud]
_ALT_PARQUEO = {"mercurio": 200e3, "venus": 300e3, "tierra": 200e3, "marte": 400e3,
                "jupiter": 1.0e6, "saturno": 1.0e6, "urano": 1.0e6, "neptuno": 1.0e6}
_CAT_VENTANAS = {n: dict(mu=CUERPOS[n].mu, radio=CUERPOS[n].radio, alt=alt)
                 for n, alt in _ALT_PARQUEO.items()}

# Standish (JPL "Keplerian Elements for Approximate Positions of the Major Planets",
# tabla 1, 1800–2050): (a, e, I, L, ϖ, Ω) y sus tasas por siglo juliano.
# a en _UA; ángulos en grados.
_ELEM_PLANETAS = {
    "mercurio": ((0.38709927, 0.20563593, 7.00497902, 252.25032350, 77.45779628, 48.33076593),
                 (0.00000037, 0.00001906, -0.00594749, 149472.67411175, 0.16047689, -0.12534081)),
    "venus": ((0.72333566, 0.00677672, 3.39467605, 181.97909950, 131.60246718, 76.67984255),
              (0.00000390, -0.00004107, -0.00078890, 58517.81538729, 0.00268329, -0.27769418)),
    "tierra": ((1.00000261, 0.01671123, -0.00001531, 100.46457166, 102.93768193, 0.0),
               (0.00000562, -0.00004392, -0.01294668, 35999.37244981, 0.32327364, 0.0)),
    "marte": ((1.52371034, 0.09339410, 1.84969142, -4.55343205, -23.94362959, 49.55953891),
              (0.00001847, 0.00007882, -0.00813131, 19140.30268499, 0.44441088, -0.29257343)),
    "jupiter": ((5.20288700, 0.04838624, 1.30439695, 34.39644051, 14.72847983, 100.47390909),
                (-0.00011607, -0.00013253, -0.00183714, 3034.74612775, 0.21252668, 0.20469106)),
    "saturno": ((9.53667594, 0.05386179, 2.48599187, 49.95424423, 92.59887831, 113.66242448),
                (-0.00125060, -0.00050991, 0.00193609, 1222.49362201, -0.41897216, -0.28867794)),
    "urano": ((19.18916464, 0.04725744, 0.77263783, 313.23810451, 170.95427630, 74.01692503),
              (-0.00196176, -0.00004397, -0.00242939, 428.48202785, 0.40805281, 0.04240589)),
    "neptuno": ((30.06992276, 0.00859048, 1.77004347, -55.12002969, 44.96476227, 131.78422574),
                (0.00026291, 0.00005105, 0.00035372, 218.45945325, -0.32241464, -0.00508664)),
}


class EfemeridesPlanetas:
    """Elementos keplerianos aproximados de Standish (1800–2050)."""
    jd_min, jd_max = 2378496.5, 2469807.5     # 1800-01-01 .. 2050-01-01

    def estado(self, cuerpo: str, jd: float):
        if not (self.jd_min <= jd <= self.jd_max):
            raise ValueError(f"JD {jd:.1f} fuera de 1800–2050: efemérides de Standish no válidas")
        try:
            e0, rate = _ELEM_PLANETAS[cuerpo]
        except KeyError:
            raise ValueError(f"cuerpo sin efemérides: {cuerpo!r}") from None
        T = (jd - _JD_J2000) / 36525.0
        a, e, inc, L, varpi, node = (e0[i] + rate[i] * T for i in range(6))
        a *= _UA
        inc, node = math.radians(inc), math.radians(node)
        w = math.radians(varpi) - node
        M = math.remainder(math.radians(L - varpi), 2 * math.pi)
        E = _kepler_E(M, e, tol=1e-14, max_iter=30)
        xp, yp = a * (math.cos(E) - e), a * math.sqrt(1 - e * e) * math.sin(E)
        fac = math.sqrt(MU_SOL * a) / (a * (1 - e * math.cos(E)))
        vxp, vyp = -fac * math.sin(E), fac * math.sqrt(1 - e * e) * math.cos(E)
        cw, sw, ci, si, cn, sn = math.cos(w), math.sin(w), math.cos(inc), math.sin(inc), \
            math.cos(node), math.sin(node)
        R = np.array([[cn * cw - sn * sw * ci, -cn * sw - sn * cw * ci],
                      [sn * cw + cn * sw * ci, -sn * sw + cn * cw * ci],
                      [sw * si, cw * si]])
        return R @ np.array([xp, yp]), R @ np.array([vxp, vyp])


# ----------------------------------------------------------------------------
# Lambert (Curtis, variables universales), UNA revolución, sentido prógrado (+z):
# el ángulo de transferencia se mide en sentido antihorario desde r1 hasta r2
# (0..2π), así que el arco es corto o largo según la geometría de las fechas;
# no hay solución retrógrada ni multi-revolución.
# ----------------------------------------------------------------------------
@_njit(cache=True)
def _stumpff(z):
    if z > 1e-8:
        s = math.sqrt(z)
        return (1.0 - math.cos(s)) / z, (s - math.sin(s)) / (s * s * s)
    if z < -1e-8:
        s = math.sqrt(-z)
        return (math.cosh(s) - 1.0) / (-z), (math.sinh(s) - s) / (s * s * s)
    return 0.5 - z / 24.0, 1.0 / 6.0 - z / 120.0


@_njit(cache=True)
def _lambert(r1, r2, dt, mu):
    """(v1, v2, ok). `r1`, `r2`: vectores 3D [m]; `dt` [s]; sentido prógrado
    respecto de +z. ok=False si dt <= 0, transferencia a 180° o no converge."""
    v1 = np.zeros(3)
    v2 = np.zeros(3)
    n1 = math.sqrt(r1[0] ** 2 + r1[1] ** 2 + r1[2] ** 2)
    n2 = math.sqrt(r2[0] ** 2 + r2[1] ** 2 + r2[2] ** 2)
    if dt <= 0.0:
        return v1, v2, False
    cr = (r1[0] * r2[0] + r1[1] * r2[1] + r1[2] * r2[2]) / (n1 * n2)
    cr = min(1.0, max(-1.0, cr))
    dth = math.acos(cr)
    if r1[0] * r2[1] - r1[1] * r2[0] < 0.0:
        dth = 2.0 * math.pi - dth
    s = math.sin(dth)
    if abs(s) < 1e-9:
        return v1, v2, False
    A = s * math.sqrt(n1 * n2 / (1.0 - math.cos(dth)))
    sm = math.sqrt(mu)
    lo, hi = -4.0 * math.pi ** 2 * 4.0, 4.0 * math.pi ** 2 - 1e-9
    for _ in range(200):
        z = 0.5 * (lo + hi)
        C, S = _stumpff(z)
        y = n1 + n2 + A * (z * S - 1.0) / math.sqrt(C)
        if y < 0.0:
            lo = z                       # F no definida: la raíz está por encima
            continue
        F = (y / C) ** 1.5 * S + A * math.sqrt(y) - sm * dt
        if F < 0.0:
            lo = z
        else:
            hi = z
        if hi - lo < 1e-13 * (1.0 + abs(z)):
            break
    z = 0.5 * (lo + hi)
    C, S = _stumpff(z)
    y = n1 + n2 + A * (z * S - 1.0) / math.sqrt(C)
    if y <= 0.0:
        return v1, v2, False
    F = (y / C) ** 1.5 * S + A * math.sqrt(y) - sm * dt
    if abs(F) > 1e-6 * sm * dt:
        return v1, v2, False
    f = 1.0 - y / n1
    g = A * math.sqrt(y / mu)
    gd = 1.0 - y / n2
    for k in range(3):
        v1[k] = (r2[k] - f * r1[k]) / g
        v2[k] = (gd * r2[k] - r1[k]) / g
    return v1, v2, True


@_njit(cache=True)
def _malla_lambert(r_o, v_o, r_d, v_d, dep_s, arr_s, mu):
    """Para cada par (salida i, llegada j): v∞ de salida y de llegada [m/s]
    (NaN si no hay solución). r_o/v_o: estados del origen en cada fecha de
    salida (N×3); r_d/v_d: del destino en cada fecha de llegada (M×3)."""
    N, M = r_o.shape[0], r_d.shape[0]
    vinf_s = np.full((N, M), np.nan)
    vinf_l = np.full((N, M), np.nan)
    for i in range(N):
        for j in range(M):
            dt = arr_s[j] - dep_s[i]
            if dt <= 0.0:
                continue
            v1, v2, ok = _lambert(r_o[i], r_d[j], dt, mu)
            if not ok:
                continue
            a = 0.0
            b = 0.0
            for k in range(3):
                a += (v1[k] - v_o[i, k]) ** 2
                b += (v2[k] - v_d[j, k]) ** 2
            vinf_s[i, j] = math.sqrt(a)
            vinf_l[i, j] = math.sqrt(b)
    return vinf_s, vinf_l


# ----------------------------------------------------------------------------
# Costes y búsqueda
# ----------------------------------------------------------------------------
def _dv_hiperbolico(vinf: np.ndarray, mu: float, r_park: float) -> np.ndarray:
    """Δv desde una órbita circular de radio r_park hasta la hipérbola de v∞."""
    return np.sqrt(vinf ** 2 + 2.0 * mu / r_park) - math.sqrt(mu / r_park)


def _dv_captura(vinf: np.ndarray, mu: float, r_p: float) -> np.ndarray:
    """Δv para frenar en el periápside de la hipérbola de llegada a una órbita
    circular de radio r_p (misma fórmula que la salida; es la inserción)."""
    return _dv_hiperbolico(vinf, mu, r_p)


@dataclass
class RestriccionesVentana:
    c3_max: float = math.inf             # m²/s²
    vinf_llegada_max: float = math.inf   # m/s
    tof_min: Optional[float] = None      # s; None = 0.3 × TOF de Hohmann del par
    tof_max: Optional[float] = None      # s; None = 1.5 × TOF de Hohmann del par
    dv_total_max: float = math.inf       # m/s (salida + captura)


@dataclass
class VentanaLanzamiento:
    salida_jd: float
    llegada_jd: float
    tof_dias: float
    c3_km2s2: float
    vinf_salida: float
    vinf_llegada: float
    dv_salida: float
    dv_captura: float
    dv_total: float
    ciclo: int
    refinada: bool = True     # False: mínimo de la malla gruesa (el refinado no tuvo celdas válidas)

    @staticmethod
    def fecha(jd: float) -> str:
        d = _EPOCA_J2000 + timedelta(days=jd - _JD_J2000)
        return d.strftime("%Y-%m-%d")


def _jd_de_fecha(texto: str) -> float:
    d = datetime.fromisoformat(texto)
    # (12.7.1) con zona horaria se CONVIERTE a UTC; sin ella se supone UTC
    d = d.replace(tzinfo=timezone.utc) if d.tzinfo is None else d.astimezone(timezone.utc)
    return _JD_J2000 + (d - _EPOCA_J2000).total_seconds() / _DIA


def _periodo_orbital(a: float) -> float:
    """Periodo kepleriano alrededor del Sol [s] para el semieje `a` [m]."""
    return 2.0 * math.pi * math.sqrt(a ** 3 / MU_SOL)


class PlanificadorVentanas:
    def __init__(self, origen: str, destino: str, efemerides: Optional[object] = None,
                 restricciones: Optional[RestriccionesVentana] = None, capturar: bool = True,
                 alt_origen: Optional[float] = None, alt_destino: Optional[float] = None,
                 dv_salida_max: float = math.inf):
        for c in (origen, destino):
            if c not in _CAT_VENTANAS:
                raise ValueError(f"cuerpo desconocido: {c!r} (soportados: {', '.join(_CAT_VENTANAS)})")
        if origen == destino:
            raise ValueError("origen y destino deben ser distintos")
        self.o, self.d = origen, destino
        self.eph = efemerides or EfemeridesPlanetas()
        self.res = restricciones or RestriccionesVentana()
        self.capturar = capturar
        self.dv_salida_max = dv_salida_max       # m/s desde la órbita de estacionamiento
        co, cd = _CAT_VENTANAS[origen], _CAT_VENTANAS[destino]
        self.r_park = co["radio"] + (co["alt"] if alt_origen is None else alt_origen)
        self.r_cap = cd["radio"] + (cd["alt"] if alt_destino is None else alt_destino)
        self._a1, self._a2 = _ELEM_PLANETAS[origen][0][0] * _UA, _ELEM_PLANETAS[destino][0][0] * _UA
        self.malla = None

    def _estados(self, cuerpo, jds):
        rv = [self.eph.estado(cuerpo, jd) for jd in jds]
        return np.array([x[0] for x in rv]), np.array([x[1] for x in rv])

    def tof_hohmann(self) -> float:
        """Tiempo de vuelo de la transferencia de Hohmann entre las órbitas medias."""
        return 0.5 * _periodo_orbital(0.5 * (self._a1 + self._a2))

    def periodo_sinodico(self) -> float:
        T1, T2 = _periodo_orbital(self._a1), _periodo_orbital(self._a2)
        return abs(1.0 / (1.0 / T1 - 1.0 / T2))

    def _evaluar(self, dep, arr, tmin: float, tmax: float) -> dict:
        """Malla dep × arr: v∞, Δv, TOF y la máscara de celdas válidas."""
        r = self.res
        ro, vo = self._estados(self.o, dep)
        rd, vd = self._estados(self.d, arr)
        vs, vl = _malla_lambert(ro, vo, rd, vd, (dep - _JD_J2000) * _DIA, (arr - _JD_J2000) * _DIA, MU_SOL)
        tof = (arr[None, :] - dep[:, None]) * _DIA
        mu_o, mu_d = _CAT_VENTANAS[self.o]["mu"], _CAT_VENTANAS[self.d]["mu"]
        dvs = _dv_hiperbolico(vs, mu_o, self.r_park)
        dvc = _dv_captura(vl, mu_d, self.r_cap) if self.capturar else np.zeros_like(vl)
        tot = dvs + dvc
        ok = (np.isfinite(tot) & (tof >= tmin) & (tof <= tmax)
              & (dvs <= self.dv_salida_max) & (vs ** 2 <= r.c3_max) & (vl <= r.vinf_llegada_max)
              & (tot <= r.dv_total_max))
        return dict(dep=dep, arr=arr, vinf_s=vs, vinf_l=vl, dv_s=dvs, dv_c=dvc, dv_tot=tot,
                    tof=tof, valido=ok)

    def calcular(self, jd_ini: float, jd_fin: float, paso_dias: float = 2.0,
                 tof_max_dias: Optional[float] = None):
        """Malla salida × llegada. Guarda y devuelve el dict con las matrices."""
        r = self.res
        for nom, x in (("jd_ini", jd_ini), ("jd_fin", jd_fin)):
            if not math.isfinite(x):
                raise ValueError(f"{nom} debe ser finito")
        if not (math.isfinite(paso_dias) and paso_dias > 0):
            raise ValueError("el paso de la malla debe ser finito y mayor que cero")
        for nom, x in (("TOF mínimo", r.tof_min), ("TOF máximo", r.tof_max),
                       ("TOF máximo (días)", None if tof_max_dias is None else tof_max_dias * _DIA)):
            if x is not None and not (math.isfinite(x) and x > 0):
                raise ValueError(f"el {nom} debe ser finito y positivo")
        if not jd_ini < jd_fin:
            raise ValueError("la fecha final debe ser posterior a la inicial")
        tmin = r.tof_min if r.tof_min is not None else 0.3 * self.tof_hohmann()
        tmax = r.tof_max if r.tof_max is not None else 1.5 * self.tof_hohmann()
        if tof_max_dias is not None:
            tmax = min(tmax, tof_max_dias * _DIA)
        if not tmin < tmax:
            raise ValueError("el TOF mínimo debe ser menor que el máximo")
        jd_lo, jd_hi = getattr(self.eph, "jd_min", -math.inf), getattr(self.eph, "jd_max", math.inf)
        if jd_ini < jd_lo or jd_fin + tmax / _DIA > jd_hi:
            raise ValueError(
                f"el cálculo necesita efemérides de {VentanaLanzamiento.fecha(jd_ini)} a "
                f"{VentanaLanzamiento.fecha(jd_fin + tmax / _DIA)} (la llegada llega hasta el TOF máximo) "
                f"y solo hay de {VentanaLanzamiento.fecha(jd_lo)} a {VentanaLanzamiento.fecha(jd_hi)}")
        self._tmin, self._tmax, self._paso = tmin, tmax, paso_dias
        self._lim_dep, self._lim_arr = (jd_ini, jd_fin), (jd_lo, jd_hi)
        dep = np.arange(jd_ini, jd_fin + 1e-9, paso_dias)
        arr = np.arange(jd_ini + tmin / _DIA, jd_fin + tmax / _DIA + 1e-9, paso_dias)
        self.malla = self._evaluar(dep, arr, tmin, tmax)
        return self.malla

    @staticmethod
    def _validar_criterio(criterio: str) -> None:
        if criterio not in ("dv_total", "c3"):
            raise ValueError(f"criterio desconocido: {criterio!r} (usa 'dv_total' o 'c3')")

    @staticmethod
    def _coste(m: dict, criterio: str) -> np.ndarray:
        PlanificadorVentanas._validar_criterio(criterio)
        return np.where(m["valido"], m["dv_tot"] if criterio == "dv_total" else m["vinf_s"] ** 2, np.inf)

    @staticmethod
    def _ventana_de(m: dict, i: int, j: int, ciclo: int, refinada: bool = True) -> VentanaLanzamiento:
        return VentanaLanzamiento(
            salida_jd=float(m["dep"][i]), llegada_jd=float(m["arr"][j]),
            tof_dias=float(m["tof"][i, j] / _DIA), c3_km2s2=float(m["vinf_s"][i, j] ** 2 / 1e6),
            vinf_salida=float(m["vinf_s"][i, j]), vinf_llegada=float(m["vinf_l"][i, j]),
            dv_salida=float(m["dv_s"][i, j]), dv_captura=float(m["dv_c"][i, j]),
            dv_total=float(m["dv_tot"][i, j]), ciclo=ciclo, refinada=refinada)

    def _refinar(self, i: int, j: int, criterio: str, ciclo: int) -> VentanaLanzamiento:
        """(12.7.1) Mínimo de la malla gruesa -> malla local fina (±1 paso grueso en
        salida y llegada, paso/16). La ventana deja de ser un múltiplo del paso."""
        m = self.malla
        paso = self._paso
        # (12.7.2) dentro de lo pedido: salida en [inicio, fin], llegada en las efemérides
        dep = np.clip(m["dep"][i] + np.linspace(-paso, paso, 33), *self._lim_dep)
        arr = np.clip(m["arr"][j] + np.linspace(-paso, paso, 33), *self._lim_arr)
        f = self._evaluar(dep, arr, self._tmin, self._tmax)
        c = self._coste(f, criterio)
        if not np.isfinite(c).any():
            return self._ventana_de(m, i, j, ciclo, refinada=False)
        fi, fj = np.unravel_index(int(np.argmin(c)), c.shape)
        return self._ventana_de(f, int(fi), int(fj), ciclo)

    def ventanas(self, criterio: str = "dv_total", refinar: bool = True) -> list[VentanaLanzamiento]:
        """Mejor oportunidad de cada ciclo sinódico (mínimo del criterio entre las
        celdas válidas): 'dv_total' o 'c3'. Con `refinar` cada una se afina en una
        malla local; sin él es el mínimo de la malla gruesa."""
        m = self.malla
        if m is None:
            raise RuntimeError("llama primero a calcular()")
        coste = self._coste(m, criterio)
        self.criterio = criterio
        mejor_j = np.argmin(coste, axis=1)
        mejor = coste[np.arange(len(mejor_j)), mejor_j]
        T_sin = self.periodo_sinodico() / _DIA
        dep = m["dep"]
        # una ventana por ciclo sinódico: cajones de ancho T_sinódico centrados en
        # el mejor mínimo global; en cada cajón, el mínimo del criterio
        if not np.isfinite(mejor).any():
            return []
        i0 = int(np.argmin(mejor))
        k = np.floor((dep - dep[i0]) / T_sin + 0.5).astype(int)
        elegidos: list[int] = []
        for kk in np.unique(k):
            idx = np.where(k == kk)[0]
            i = int(idx[np.argmin(mejor[idx])])
            if not math.isfinite(mejor[i]):
                continue
            # un mínimo pegado al borde de la malla no es una ventana completa
            if i in (0, len(dep) - 1) and len(idx) > 1:
                continue
            elegidos.append(i)
        out = []
        for i in sorted(elegidos):
            j = int(mejor_j[i])
            n = len(out) + 1
            out.append(self._refinar(i, j, criterio, n) if refinar else self._ventana_de(m, i, j, n, refinada=False))
        return out


def texto_tabla_ventanas(vs: list, capturar: bool = True) -> str:
    """Tabla de texto de las ventanas (una por ciclo sinódico)."""
    filas = [f"{'#':>2} {'salida':>10} {'llegada':>10} {'TOF d':>6} {'C3 km²/s²':>10} "
             f"{'v∞ lleg':>8} {'Δv sal':>7} {'Δv cap':>7} {'Δv tot':>7}   (km/s)"]
    for v in vs:
        filas.append(
            f"{v.ciclo:>2} {VentanaLanzamiento.fecha(v.salida_jd):>10} "
            f"{VentanaLanzamiento.fecha(v.llegada_jd):>10} {v.tof_dias:6.0f} {v.c3_km2s2:10.2f} "
            f"{v.vinf_llegada / 1e3:8.2f} {v.dv_salida / 1e3:7.2f} "
            f"{(v.dv_captura if capturar else 0.0) / 1e3:7.2f} {v.dv_total / 1e3:7.2f}"
            + ("" if v.refinada else " *"))
    if not all(v.refinada for v in vs):
        filas.append("* malla gruesa (el refinado local no encontró celdas válidas)")
    return "\n".join(filas)


def calcular_ventanas(origen: str, destino: str, inicio: str, fin: str,
                      c3_max_km2s2: float = math.inf, dv_salida_max_kms: float = math.inf,
                      vinf_llegada_max_kms: float = math.inf, criterio: str = "dv_total",
                      capturar: bool = True, paso_dias: float = 2.0, efemerides=None,
                      tof_min_dias: Optional[float] = None, tof_max_dias: Optional[float] = None):
    """Punto de entrada: (plan, ventanas). `inicio`/`fin`: fechas ISO. Las
    restricciones son límites que pone el usuario (no se deducen del cohete)."""
    for nom, x in (("c3_max_km2s2", c3_max_km2s2), ("dv_salida_max_kms", dv_salida_max_kms),
                   ("vinf_llegada_max_kms", vinf_llegada_max_kms)):
        if not (x > 0.0):                       # rechaza NaN y <= 0; inf = sin límite
            raise ValueError(f"{nom} debe ser positivo (inf = sin límite), no {x!r}")
    res = RestriccionesVentana(
        c3_max=c3_max_km2s2 * 1e6, vinf_llegada_max=vinf_llegada_max_kms * 1e3,
        tof_min=None if tof_min_dias is None else tof_min_dias * _DIA,
        tof_max=None if tof_max_dias is None else tof_max_dias * _DIA)
    plan = PlanificadorVentanas(origen, destino, efemerides=efemerides, restricciones=res,
                                capturar=capturar, dv_salida_max=dv_salida_max_kms * 1e3)
    PlanificadorVentanas._validar_criterio(criterio)
    jd0, jd1 = _jd_de_fecha(inicio), _jd_de_fecha(fin)
    if jd1 <= jd0:
        raise ValueError("la fecha final debe ser posterior a la inicial")
    plan.calcular(jd0, jd1, paso_dias)
    return plan, plan.ventanas(criterio)


def figura_porkchop(plan: "PlanificadorVentanas", v: "VentanaLanzamiento", medio_ancho: float = 120.0):
    """Figura (matplotlib.figure.Figure, sin pyplot) del porkchop de UNA ventana:
    C3 de salida (color) y v∞ de llegada (líneas) en ±medio_ancho días."""
    from matplotlib.figure import Figure
    m = plan.malla
    i = np.abs(m["dep"] - v.salida_jd) <= medio_ancho
    j = np.abs(m["arr"] - v.llegada_jd) <= medio_ancho
    sub = np.ix_(i, j)
    ok = m["valido"][sub]
    c3 = np.where(ok, m["vinf_s"][sub] ** 2 / 1e6, np.nan)
    vl = np.where(ok, m["vinf_l"][sub] / 1e3, np.nan)
    if not ok.any() or not np.nanmin(c3) > 0.0:
        raise ValueError("el porkchop necesita celdas válidas con C3 > 0 alrededor de la ventana")
    fig = Figure(figsize=(7.0, 5.5))
    ax = fig.add_subplot(111)
    niv = np.nanmin(c3) * np.array([1.0, 1.1, 1.25, 1.5, 2.0, 3.0, 4.0])
    cs = ax.contourf(m["dep"][i] - v.salida_jd, m["arr"][j] - v.llegada_jd, c3.T,
                     levels=niv, cmap="viridis_r", extend="max")
    fig.colorbar(cs, ax=ax, label="C3 de salida (km²/s²)")
    cl = ax.contour(m["dep"][i] - v.salida_jd, m["arr"][j] - v.llegada_jd, vl.T,
                    levels=8, colors="white", linewidths=0.7)
    ax.clabel(cl, fmt="%.1f", fontsize=7)
    etiqueta = "mín. C3" if getattr(plan, "criterio", "dv_total") == "c3" else "mín. Δv"
    ax.plot([0], [0], "r*", ms=12, label=f"{etiqueta}: {VentanaLanzamiento.fecha(v.salida_jd)} → "
                                         f"{VentanaLanzamiento.fecha(v.llegada_jd)}")
    ax.set_xlabel(f"Días desde la salida óptima ({VentanaLanzamiento.fecha(v.salida_jd)})")
    ax.set_ylabel(f"Días desde la llegada óptima ({VentanaLanzamiento.fecha(v.llegada_jd)})")
    ax.set_title(f"{plan.o} → {plan.d}: C3 (color), v∞ de llegada km/s (líneas)", fontsize=10)
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    return fig

def parsear_campos_ventanas(campos: dict) -> dict:
    """Convierte los textos del diálogo de ventanas en argumentos de
    `calcular_ventanas`. Vacío = sin límite en los opcionales. Lanza
    ValueError con un mensaje legible."""
    def num(clave, etiqueta, opcional=True):
        t = str(campos.get(clave, "")).strip().replace(",", ".")
        if not t:
            if opcional:
                return math.inf
            raise ValueError(f"{etiqueta}: obligatorio")
        try:
            v = float(t)
        except ValueError:
            raise ValueError(f"{etiqueta}: «{t}» no es un número") from None
        if not (math.isfinite(v) and v > 0.0):
            raise ValueError(f"{etiqueta}: debe ser > 0")
        return v

    for c in ("inicio", "fin"):
        try:
            datetime.fromisoformat(str(campos.get(c, "")).strip())
        except ValueError:
            raise ValueError(f"fecha {c}: formato AAAA-MM-DD") from None
    return dict(origen=campos["origen"], destino=campos["destino"],
                inicio=str(campos["inicio"]).strip(), fin=str(campos["fin"]).strip(),
                c3_max_km2s2=num("c3_max", "C3 máx"), dv_salida_max_kms=num("dv_salida_max", "Δv salida máx"),
                vinf_llegada_max_kms=num("vinf_llegada_max", "v∞ llegada máx"),
                tof_min_dias=(lambda v: None if math.isinf(v) else v)(num("tof_min", "TOF mín")),
                tof_max_dias=(lambda v: None if math.isinf(v) else v)(num("tof_max", "TOF máx")),
                criterio=campos.get("criterio", "dv_total"), capturar=bool(campos.get("capturar", True)))


# ============================================================
# INTERFAZ GRÁFICA
# ============================================================
class InterfazSimulacion:
    ANCHO_VENTANA      = 1400
    ALTO_VENTANA       = 820
    ANCHO_COLUMNA_IZQ  = 330
    ANCHO_COLUMNA_DER  = 290       # (12.8.0) tarjetas de telemetría
    ALTO_BARRA         = 72        # (12.8.0) barra superior (MET, fase, estado)
    ANCHO_TAB          = 32
    ANCHO_DRAWER       = 420

    COLORES_FASE = {
        "vertical":            "#ffc800",
        "gravity_turn":        "#00dcff",
        "coasting":            "#8ca0b4",
        "coasting_interetapa": "#5599cc",
        "circularizacion":     "#ff3cff",
        "deorbit":             "#ff8c00",
        "impacto":             "#ff3232",
        "espera_tli":          "#6fbf73",
        "quema_tli":           "#ff6ec7",
        "translunar":          "#9ad0ff",
        "quema_loi":           "#ffb347",
        "orbita_lunar":        "#7dffb3",
    }

    # Cámara híbrida (escala log): distancia a la Tierra [en R_T] donde
    # empieza y termina la transición "seguir a la nave" -> "vista log".
    CAM_R_MIN = 2.0
    CAM_R_MAX = 8.0

    MODOS_MISION = [
        ("satelital", "Satelital (B–C–B)"),
        ("tripulada", "Tripulada"),
        ("suborbital", "Suborbital (sonda)"),
        ("gto", "GTO (geoestacionaria)"),
        ("tli", "TLI (translunar)"),
        ("escape", "Escape"),
        ("deorbit", "Desórbita / reentrada"),
        ("lunar", "Lunar (encuentro/impacto)"),
    ]
    PERFILES_TRIPULADA = [
        ("bcb",     "Burn-Coast-Burn  (2 encendidos)"),
        ("directo", "Ascenso directo   (1 encendido)"),
    ]
    # (clave de Config, etiqueta) en el orden en que se muestran.
    CAMPOS_NUMERICOS = [
        ("angulo_kick",                 "Ángulo de kick (°)"),
        ("alt_inicio_giro",             "Altitud inicio giro (m)"),
        ("APOGEO_OBJETIVO",             "Apogeo objetivo (m)"),
        ("ALTURA_SEGURIDAD",            "Altura de seguridad (m)"),
        ("VEL_MIN_SES",                 "Vel. mínima 2da etapa (m/s)"),
        ("G_MAX_TRIPULADA",             "g máxima (tripulada)"),
        ("ECC_MAX_TRIPULADA_BCB",       "ecc máx (bcb)"),
        ("ECC_MAX_TRIPULADA_DIRECTO",   "ecc máx (directo)"),
        ("latitud_lanzamiento",         "Latitud de lanzamiento (°)"),
        ("azimut_lanzamiento",          "Azimut (° desde N, 0–180)"),
        ("longitud_lanzamiento_ecliptica", "Long. ecl. (°, solo lunar)"),
        ("LUNAR_DIST_OBJETIVO",        "Perilunio objetivo lunar (km)"),
        ("LUNAR_ORBITA_ALT",           "Altitud órbita LOI (km)"),
        ("fecha_lanzamiento",          "Fecha UTC (ISO)"),
        ("sim_speed",                   "Velocidad de simulación (x)"),
    ]
    # (12.9.0) pestañas de la columna izquierda y pestaña de cada campo numérico
    PESTANAS_IZQ = (("mision", "MISIÓN"), ("lanz", "LANZ."),
                    ("opc", "OPCIONES"), ("eventos", "EVENTOS"))
    PESTANA_DE_CAMPO = {
        "angulo_kick": "mision", "alt_inicio_giro": "mision", "APOGEO_OBJETIVO": "mision",
        "ALTURA_SEGURIDAD": "mision", "VEL_MIN_SES": "mision", "G_MAX_TRIPULADA": "mision",
        "ECC_MAX_TRIPULADA_BCB": "mision", "ECC_MAX_TRIPULADA_DIRECTO": "mision",
        "latitud_lanzamiento": "lanz", "azimut_lanzamiento": "lanz",
        "longitud_lanzamiento_ecliptica": "lanz", "LUNAR_DIST_OBJETIVO": "lanz",
        "LUNAR_ORBITA_ALT": "lanz", "fecha_lanzamiento": "lanz", "sim_speed": "opc",
    }
    # (clave de Config, conversor) en el orden en que se leen al iniciar.
    CONVERSORES_CAMPOS = [
        ("angulo_kick", float), ("alt_inicio_giro", float),
        ("APOGEO_OBJETIVO", float), ("ALTURA_SEGURIDAD", float),
        ("VEL_MIN_SES", float), ("G_MAX_TRIPULADA", float),
        ("ECC_MAX_TRIPULADA_BCB", float), ("ECC_MAX_TRIPULADA_DIRECTO", float),
        ("sim_speed", _entero_estricto),
        ("latitud_lanzamiento", float), ("azimut_lanzamiento", float),
        ("longitud_lanzamiento_ecliptica", float),
        ("LUNAR_DIST_OBJETIVO", lambda texto: float(texto) * 1000.0),
        ("LUNAR_ORBITA_ALT", lambda texto: float(texto) * 1000.0),
        ("fecha_lanzamiento", lambda texto: texto.strip()),
    ]

    def __init__(self, root: tk.Tk):
        self.root = root
        global FUENTE_VALOR, FUENTE_MET
        familia = _elegir_fuente_mono(_familias_tk(root))      # (12.9.0)
        FUENTE_VALOR, FUENTE_MET = (familia, 13, "bold"), (familia, 22, "bold")
        self.config = Config()
        self.sim: Optional[Simulacion] = None
        self._after_id: Optional[str] = None
        self._planificador: Optional[PlanificadorDeorbitExterno] = None
        self._t_drawer: float = float("-inf")

        self.camara_libre = {"activa": False}
        self.arrastre = {"activo": False, "x0": None, "y0": None}
        self.drawer_estado = {"abierto": False}
        self.eventos_ya_mostrados: set = set()
        self.entradas: dict[str, tk.StringVar] = {}

        self._configurar_ventana()
        self._construir_ui()

    # ========================================================
    # CONSTRUCCIÓN DE LA INTERFAZ
    # ========================================================
    def _configurar_ventana(self):
        self.root.title("Simulación de Cohete 2D")
        self.root.geometry(f"{self.ANCHO_VENTANA}x{self.ALTO_VENTANA}")
        self.root.resizable(False, False)
        self.root.configure(bg=COLOR_FONDO_VENTANA)

    def _crear_columna_izquierda(self, alto_ventana: int):
        """(12.9.0) Columna izquierda con pestañas (`PESTANAS_IZQ`): una barra de
        títulos, un cuerpo con un Frame por pestaña y un pie fijo (error + botones).
        Rellena `self._pestanas` (clave -> Frame) y `self._pie_izq`."""
        marco = tk.Frame(self.root, bg=COLOR_FONDO_VENTANA)
        marco.place(x=0, y=self.ALTO_BARRA, width=self.ANCHO_COLUMNA_IZQ, height=alto_ventana)
        barra = tk.Frame(marco, bg=COLOR_FONDO_VENTANA)
        barra.pack(fill="x", padx=6, pady=(6, 0))
        pie = tk.Frame(marco, bg=COLOR_FONDO_VENTANA)
        pie.pack(side="bottom", fill="x", padx=6, pady=6)
        cuerpo = tk.Frame(marco, bg=COLOR_FONDO_PANEL,
                          highlightbackground=COLOR_BORDE_TARJETA, highlightthickness=1)
        cuerpo.pack(fill="both", expand=True, padx=6, pady=(0, 0))
        cuerpo.grid_rowconfigure(0, weight=1)
        cuerpo.grid_columnconfigure(0, weight=1)
        self._pestanas = {}
        self._botones_pestana = {}
        self._pestana_activa = None
        for i, (clave, titulo) in enumerate(self.PESTANAS_IZQ):
            boton = tk.Label(barra, text=titulo, font=("Courier New", 8, "bold"),
                             cursor="hand2", pady=5)
            boton.grid(row=0, column=i, sticky="ew")
            barra.grid_columnconfigure(i, weight=1)
            boton.bind("<Button-1>", lambda _e, c=clave: self._elegir_pestana(c))
            self._botones_pestana[clave] = boton
            marco_p = tk.Frame(cuerpo, bg=COLOR_FONDO_PANEL)
            marco_p.grid(row=0, column=0, sticky="nsew", padx=8, pady=8)   # todas apiladas
            self._pestanas[clave] = marco_p
        self._pie_izq = pie
        self._elegir_pestana(self.PESTANAS_IZQ[0][0])

    def _elegir_pestana(self, clave: str) -> None:
        marcos = getattr(self, "_pestanas", None)
        if not marcos or clave not in marcos:
            return
        for k, marco in marcos.items():
            activa = (k == clave)
            if activa:
                marco.tkraise()
            self._botones_pestana[k].config(
                bg=COLOR_TARJETA if activa else COLOR_FONDO_VENTANA,
                fg=COLOR_VALOR if activa else COLOR_TEXTO_ETIQUETA)
        self._pestana_activa = clave
        self.root.update_idletasks()             # (12.13.0) repinta ya la pestaña recién elegida

    def _construir_ui(self):
        ancho_ventana = self.ANCHO_VENTANA
        alto_util = self.ALTO_VENTANA - self.ALTO_BARRA
        self._construir_barra_superior()
        self._crear_columna_izquierda(alto_util)

        ancho_anim = (ancho_ventana - self.ANCHO_COLUMNA_IZQ
                      - self.ANCHO_COLUMNA_DER - self.ANCHO_TAB)
        panel_anim = tk.Frame(self.root, bg=COLOR_FONDO_VENTANA)
        panel_anim.place(x=self.ANCHO_COLUMNA_IZQ, y=self.ALTO_BARRA,
                         width=ancho_anim, height=alto_util)

        self._construir_canvas_animacion(panel_anim, ancho_anim, alto_util)
        self._construir_columna_derecha(alto_util)
        self._construir_drawer()
        self._construir_panel_config()
        self._construir_panel_eventos()

    # --- Canvas de la animación ---
    MARCAS_MAPA = 4

    def _construir_canvas_animacion(self, parent, ancho, alto):
        self.fig, self.ax = plt.subplots(figsize=(ancho / 100.0, alto / 100.0))
        self.fig.patch.set_facecolor(COLOR_FONDO_VENTANA)
        self.ax.set_facecolor(COLOR_FONDO_VENTANA)
        self.ax.set_aspect("equal")
        self.fig.subplots_adjust(left=0.07, right=0.98, bottom=0.06, top=0.98)   # (12.8.2)

        radio_vista = R_T * 1.4
        self.ax.set_xlim(-radio_vista, radio_vista)
        self.ax.set_ylim(-radio_vista, radio_vista)
        # (12.13.0) las marcas y sus etiquetas eran ~20 ms del dibujo (con la cámara siguiendo a la
        # nave se recalculan cada cuadro): con ~5 marcas por eje bajan un tercio y se leen igual
        from matplotlib.ticker import MaxNLocator
        for eje in (self.ax.xaxis, self.ax.yaxis):
            eje.set_major_locator(MaxNLocator(nbins=self.MARCAS_MAPA))

        self._crear_artistas_tierra()
        self._crear_artistas_trayectoria()
        self._crear_artistas_luna_y_hud()
        self._crear_artistas_inset()

        self.canvas = FigureCanvasTkAgg(self.fig, master=parent)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self._conectar_eventos_canvas()

    def _crear_artistas_tierra(self):
        # Se guardan para poder moverlos en la vista baricéntrica.
        self.patches_tierra = [
            Circle((0, 0), R_T, facecolor="#0f2d5a",
                   edgecolor="#1e3c78", zorder=1),
            # (12.10.0) halo de la atmósfera: capas translúcidas que se suman hacia la superficie
            Circle((0, 0), R_T + 100000, facecolor="#4f9bff", alpha=0.06,
                   edgecolor="none", zorder=0),
            Circle((0, 0), R_T + 60000, facecolor="#4f9bff", alpha=0.10,
                   edgecolor="none", zorder=0),
            Circle((0, 0), R_T + 30000, facecolor="#4f9bff", alpha=0.16,
                   edgecolor="none", zorder=0),
            Circle((0, 0), R_T + 100000,
                   facecolor="none", edgecolor="#ff4040",
                   linewidth=0.8, linestyle="--",
                   alpha=0.5, zorder=1),
        ]
        for patch in self.patches_tierra:
            self.ax.add_patch(patch)

        # Cuadrícula de coordenadas de la Tierra (vista desde el polo
        # norte): paralelos fijos, meridianos que giran con la Tierra,
        # el meridiano de la base en amarillo con un punto en la base.
        self.lc_grilla = LineCollection([], colors="#4a86d8", linewidths=0.6,
                                        alpha=0.55, zorder=1.5)
        self.lc_meridiano0 = LineCollection([], colors="#ffe040",
                                            linewidths=1.2, alpha=0.9,
                                            zorder=1.6)
        self.ax.add_collection(self.lc_grilla)
        self.ax.add_collection(self.lc_meridiano0)
        (self.punto_base,) = self.ax.plot([], [], "o", color="#ffe040",
                                          markersize=4, zorder=3)
        angulos = np.linspace(0.0, 2.0 * np.pi, 120)
        self._paralelos = [
            np.column_stack((R_T * np.cos(np.radians(lat)) * np.cos(angulos),
                             R_T * np.cos(np.radians(lat)) * np.sin(angulos)))
            for lat in (30.0, 60.0)]
        self._dibujar_grilla_tierra(np.pi / 2.0, 0.0, 0.0, True)

    GLOW_TRAYECTORIA = True
    ANCHO_GLOW = 5.0
    ALPHA_GLOW = 0.2
    GLOW_MAX_SEGMENTOS = 600

    def _crear_artistas_trayectoria(self):
        # (12.10.0) el glow va debajo y comparte segmentos y colores con la trayectoria
        self.lc_tray_glow = LineCollection([], linewidths=self.ANCHO_GLOW, zorder=1.9,
                                           alpha=self.ALPHA_GLOW, capstyle="round",
                                           visible=self.GLOW_TRAYECTORIA)
        self.ax.add_collection(self.lc_tray_glow)
        self.lc_tray = LineCollection([], linewidths=1.3, zorder=2)
        self.ax.add_collection(self.lc_tray)

        (self.linea_orbita,) = self.ax.plot([], [], "--", color="#c8dcff",
                                            linewidth=0.9, alpha=0.7, zorder=1)
        (self.punto_peri,) = self.ax.plot([], [], "o", color="#ff6060",
                                          markersize=4, zorder=2)
        (self.punto_apo,)  = self.ax.plot([], [], "o", color="#60ff90",
                                          markersize=4, zorder=2)
        (self.punto_cohete,) = self.ax.plot([], [], "o", color="white",
                                            markersize=7, zorder=3)

    def _crear_artistas_luna_y_hud(self):
        # Luna: se posiciona cada frame según la efeméride activa
        # (ver _dibujar_luna). Radio real para que la escala
        # tenga sentido cuando la cámara se acerca.
        self.patch_luna = Circle((1e12, 0), R_LUNA, facecolor="#9a9a9a",
                                 edgecolor="#c8c8c8", linewidth=0.6, zorder=1)
        self.ax.add_patch(self.patch_luna)

        # Overlay opcional de puntos de Lagrange (ver Config.mostrar_crtbp)
        self.lagrange_artists = {}

        self.texto_hud = self.ax.text(
            0.02, 0.98, "", transform=self.ax.transAxes,
            va="top", ha="left", fontsize=9, color="#7fa8c9",
            family="monospace",
        )
        self.ax.text(0.98, 0.02, "scroll: zoom   arrastrar: mover   F: seguir   E: exportar",
                     transform=self.ax.transAxes, va="bottom", ha="right",
                     fontsize=7, color="#4a6a85")

    INSET_RECT = (0.02, 0.02, 0.24, 0.24)     # (12.11.0) en fracción del mapa
    INSET_MAX_PUNTOS = 400
    INSET_MARGEN_OCULTAR = 1.15               # la Tierra entera con este margen: se oculta
    INSET_FACTOR_LUNA = 3.0                   # la Luna cuenta para la escala si está a < 3x lo recorrido

    def _crear_artistas_inset(self):
        """(12.11.0) Recuadro con la Tierra, la Luna, la nave y su trayectoria centradas en la
        Tierra, para no perder la referencia cuando la cámara deja la Tierra fuera del mapa."""
        ax = self.ax.inset_axes(list(self.INSET_RECT), zorder=5)
        ax.set_facecolor(COLOR_FONDO_PANEL)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for borde in ax.spines.values():
            borde.set_edgecolor("#1e3c78")
            borde.set_linewidth(0.8)
        ax.add_patch(Circle((0, 0), R_T, facecolor="#0f2d5a", edgecolor="#1e3c78"))
        (self.inset_tray,) = ax.plot([], [], "-", color="#7fa8c9", linewidth=0.9)
        (self.inset_luna,) = ax.plot([], [], "o", color="#c8c8c8", markersize=4)
        (self.inset_nave,) = ax.plot([], [], "o", color="white", markersize=4)
        ax.set_visible(False)
        self.ax_inset = ax
        self._inset_visible = False

    def _ocultar_inset(self):
        ax = getattr(self, "ax_inset", None)
        if ax is None:
            return
        ax.set_visible(False)
        self._inset_visible = False
        for linea in (self.inset_tray, self.inset_luna, self.inset_nave):
            linea.set_data([], [])

    def _actualizar_inset(self, sim: Simulacion, pos_eci: Vec2, luna: dict, offset: Vec2):
        """Muestra u oculta el recuadro según se vea la Tierra entera en el mapa, y lo dibuja."""
        ax = getattr(self, "ax_inset", None)
        if ax is None:
            return
        ox, oy = offset
        xlim = tuple(v - ox for v in self.ax.get_xlim())
        ylim = tuple(v - oy for v in self.ax.get_ylim())
        if _tierra_en_vista(xlim, ylim, R_T, self.INSET_MARGEN_OCULTAR):
            mostrar = False
        elif not _tierra_en_vista(xlim, ylim, R_T, 1.0):
            mostrar = True
        else:
            mostrar = self._inset_visible              # histéresis: entre ambos umbrales no cambia
        if not mostrar:
            if self._inset_visible:
                self._ocultar_inset()
            return
        self._inset_visible = True
        ax.set_visible(True)
        escala_log = sim.config.escala_log
        nave = tuple(float(v) for v in vista_xy(pos_eci[0], pos_eci[1], escala_log))
        moon = tuple(float(v) for v in vista_xy(luna["x"], luna["y"], escala_log))
        n = len(sim.tray_wx)
        extension = max(R_T * 1.2, float(np.hypot(*nave)))
        if n >= 2:
            paso = _paso_vista(n, self.INSET_MAX_PUNTOS)
            tx, ty = vista_xy(np.asarray(_submuestrear(sim.tray_wx, paso)),
                              np.asarray(_submuestrear(sim.tray_wy, paso)), escala_log)
            extension = max(extension, float(np.max(np.hypot(tx, ty))))
            tx, ty = list(tx), list(ty)
            corte = getattr(sim, "_idx_inicio_orbita", None)
            if getattr(sim, "_orbita_podada", False) and corte is not None and 0 < corte < n:
                j = int(np.searchsorted(_indices_vista(n, paso), corte, side="left"))
                tx, ty = _serie_con_corte(tx, j), _serie_con_corte(ty, j)
            self.inset_tray.set_data(tx, ty)
        else:
            self.inset_tray.set_data([], [])
        d_luna = float(np.hypot(*moon))
        if d_luna < self.INSET_FACTOR_LUNA * extension:
            extension = max(extension, d_luna * 1.1)
        extension *= 1.15
        self.inset_nave.set_data([nave[0]], [nave[1]])
        self.inset_luna.set_data([moon[0]], [moon[1]])
        ax.set_xlim(-extension, extension)
        ax.set_ylim(-extension, extension)

    def _conectar_eventos_canvas(self):
        self.canvas.mpl_connect("scroll_event", self._on_scroll)
        self.canvas.mpl_connect("button_press_event", self._on_press)
        self.canvas.mpl_connect("motion_notify_event", self._on_motion)
        self.canvas.mpl_connect("button_release_event", self._on_release)
        self.canvas.mpl_connect("key_press_event", self._on_key)
        self.canvas.get_tk_widget().bind(
            "<Enter>", lambda e: self.canvas.get_tk_widget().focus_set()
        )

    # --- Drawer con la gráfica de vuelo ---
    def _construir_drawer(self):
        ancho_ventana = self.ANCHO_VENTANA
        alto_ventana = self.ALTO_VENTANA - self.ALTO_BARRA
        self.drawer_frame = tk.Frame(self.root, bg="#04070e",
                                     highlightbackground="#1e3c78",
                                     highlightthickness=1)
        self.x_drawer = ancho_ventana - self.ANCHO_TAB - self.ANCHO_DRAWER

        self.fig2, self.ax2 = plt.subplots(
            figsize=(self.ANCHO_DRAWER / 100.0, alto_ventana / 100.0)
        )
        self.fig2.patch.set_facecolor("#04070e")
        self.ax2.set_facecolor("#04070e")
        self.ax2.set_title("Altitud y velocidad vs. tiempo",
                           color="#00dcff", fontsize=9)
        self.ax2.tick_params(colors="#5a8aa8", labelsize=8)
        (self.linea_alt,) = self.ax2.plot([], [], color="#00dcff",
                                          linewidth=1.2, label="Altitud (km)")
        self.ax2b = self.ax2.twinx()
        self.ax2b.tick_params(colors="#5a8aa8", labelsize=8)
        (self.linea_vel,) = self.ax2b.plot([], [], color="#ff9020",
                                           linewidth=1.0, label="Velocidad (km/s)")

        self.canvas2 = FigureCanvasTkAgg(self.fig2, master=self.drawer_frame)
        self.canvas2.get_tk_widget().pack(fill="both", expand=True, padx=8, pady=8)

        self._construir_pestana_drawer()

    def _construir_pestana_drawer(self):
        """Pestaña lateral con texto vertical que abre/cierra el drawer."""
        ancho_ventana = self.ANCHO_VENTANA
        alto_ventana = self.ALTO_VENTANA - self.ALTO_BARRA
        pestana = tk.Frame(self.root, bg=COLOR_FONDO_PANEL, cursor="hand2")
        pestana.place(x=ancho_ventana - self.ANCHO_TAB, y=self.ALTO_BARRA,
                      width=self.ANCHO_TAB, height=alto_ventana)

        self.tab_canvas = tk.Canvas(pestana, bg=COLOR_FONDO_PANEL,
                                    highlightthickness=0, bd=0,
                                    width=self.ANCHO_TAB, height=alto_ventana)
        self.tab_canvas.pack(fill="both", expand=True)

        self.tab_texto = self.tab_canvas.create_text(
            self.ANCHO_TAB // 2, alto_ventana // 2,
            text="◀ GRÁFICA DE VUELO",
            fill=COLOR_TEXTO_ACENTO,
            font=("Courier New", 9, "bold"),
            angle=270,
        )

        pestana.bind("<Button-1>", self._alternar_drawer)
        self.tab_canvas.bind("<Button-1>", self._alternar_drawer)

    # --- Panel de configuración ---
    def _construir_panel_config(self):
        self._construir_selector_modo(self._pestanas["mision"])
        self._construir_selector_perfil(self._pestanas["mision"])
        self._sep_config = tk.Frame(self._pestanas["mision"], bg="#1e3246", height=1)
        self._sep_config.pack(fill="x", pady=6)
        self._construir_campos_numericos()
        self._construir_casillas_opciones(self._pestanas["opc"])
        self._construir_etiqueta_error_y_botones()

    def _construir_selector_modo(self, parent):
        tk.Label(parent, text="Modo de misión",
                 bg=COLOR_FONDO_PANEL, fg=COLOR_TEXTO_ETIQUETA,
                 font=FUENTE_NEGRITA).pack(anchor="w", pady=(0, 2))

        self.var_modo = tk.StringVar(value=self.config.modo_mision)
        for valor, etiqueta in self.MODOS_MISION:
            tk.Radiobutton(
                parent, text=etiqueta,
                variable=self.var_modo, value=valor,
                command=self._cambiar_modo,
                **_estilo_seleccion(),
            ).pack(anchor="w")

    def _construir_selector_perfil(self, parent):
        """Sub-selector de perfil tripulado (visible solo en tripulada)."""
        self.frame_perfil_tripulada = tk.Frame(parent, bg=COLOR_FONDO_PANEL)
        tk.Label(self.frame_perfil_tripulada, text="Perfil tripulado",
                 bg=COLOR_FONDO_PANEL, fg=COLOR_TEXTO_ETIQUETA,
                 font=("Courier New", 8, "bold")).pack(anchor="w", pady=(2, 0))

        self.var_perfil = tk.StringVar(value=self.config.perfil_tripulada)
        _crear_radiobuttons(self.frame_perfil_tripulada,
                            self.PERFILES_TRIPULADA, self.var_perfil,
                            padx=(12, 0))
        self._actualizar_visibilidad_perfil()

    def _construir_campos_numericos(self):
        for clave, etiqueta in self.CAMPOS_NUMERICOS:
            fila = tk.Frame(self._pestanas[self.PESTANA_DE_CAMPO.get(clave, "mision")],
                            bg=COLOR_FONDO_PANEL)
            fila.pack(fill="x", pady=2)
            tk.Label(fila, text=etiqueta, bg=COLOR_FONDO_PANEL,
                     fg=COLOR_TEXTO_ETIQUETA,
                     font=FUENTE_NORMAL, anchor="w").pack(side="left")
            valor_inicial = getattr(self.config, clave)
            if clave in ("LUNAR_DIST_OBJETIVO", "LUNAR_ORBITA_ALT"):
                valor_inicial = valor_inicial / 1000.0  # GUI en km; Config en m
            variable = tk.StringVar(value=f"{valor_inicial:g}"
                                    if isinstance(valor_inicial, (int, float))
                                    else str(valor_inicial))
            tk.Entry(fila, textvariable=variable,
                     width=19 if clave == "fecha_lanzamiento" else 9,
                     bg=COLOR_FONDO_ENTRADA, fg=COLOR_TEXTO_ACENTO,
                     insertbackground=COLOR_TEXTO_ACENTO,
                     relief="solid", bd=1,
                     font=FUENTE_NORMAL).pack(side="right")
            self.entradas[clave] = variable

    def _construir_casillas_opciones(self, panel):
        cfg = self.config

        self.var_cutoff_subor = tk.BooleanVar(
            value=cfg.suborbital_cutoff_por_apogeo)
        self.chk_cutoff_subor = _crear_checkbutton(
            panel, "Suborbital: cutoff por apogeo", self.var_cutoff_subor)

        self.var_rotacion = tk.BooleanVar(value=cfg.rotacion_terrestre)
        self.chk_rotacion = _crear_checkbutton(
            panel, "Rotación terrestre", self.var_rotacion)
        self.chk_rotacion.pack(anchor="w", pady=(6, 0))
        self._actualizar_visibilidad_cutoff_subor()

        self.var_crtbp = tk.BooleanVar(value=cfg.mostrar_crtbp)
        _crear_checkbutton(panel, "Mostrar CRTBP (L1–L5)",
                           self.var_crtbp).pack(anchor="w", pady=(6, 0))

        self.var_grilla = tk.BooleanVar(value=cfg.mostrar_grilla_tierra)
        _crear_checkbutton(panel, "Cuadrícula en la Tierra",
                           self.var_grilla).pack(anchor="w")

        self.var_vista_bari = tk.BooleanVar(value=cfg.marco_vista == "baricentro")
        _crear_checkbutton(panel, "Vista baricentro Tierra–Luna",
                           self.var_vista_bari).pack(anchor="w")

        self.var_efem_spice = tk.BooleanVar(
            value=cfg.efemerides_lunar == "spice")
        _crear_checkbutton(panel, "Lunar: usar SPICE (kernels)",
                           self.var_efem_spice).pack(anchor="w")

        self.var_loi = tk.BooleanVar(value=cfg.LUNAR_LOI)
        _crear_checkbutton(panel, "Lunar: captura LOI (2D, provisional)",
                           self.var_loi).pack(anchor="w")

        self.var_escala_log = tk.BooleanVar(value=cfg.escala_log)
        _crear_checkbutton(panel, "Escala logarítmica (vista)",
                           self.var_escala_log,
                           command=self._al_cambiar_escala_log).pack(anchor="w")

    def _construir_etiqueta_error_y_botones(self):
        pie = self._pie_izq
        self.etiqueta_error = tk.Label(pie, text="",
                                       bg=COLOR_FONDO_VENTANA, fg=COLOR_TEXTO_ERROR,
                                       font=("Courier New", 8), wraplength=260)
        self.etiqueta_error.pack(fill="x", pady=(4, 0))

        fila_botones = tk.Frame(pie, bg=COLOR_FONDO_VENTANA)
        fila_botones.pack(fill="x", pady=(8, 0))

        self.boton_iniciar = tk.Button(
            fila_botones, text="▶  INICIAR",
            bg="#003318", fg=COLOR_TEXTO_ACTIVO,
            font=FUENTE_NEGRITA,
            relief="solid", bd=1, activebackground="#004d24",
            command=self._iniciar,
        )
        self.boton_iniciar.pack(side="left", fill="x", expand=True, padx=(0, 4))

        self.boton_reiniciar = tk.Button(
            fila_botones, text="⟲  REINICIAR",
            bg="#331400", fg="#ffa040",
            font=FUENTE_NEGRITA,
            relief="solid", bd=1, activebackground="#4d2000",
            command=self._reiniciar,
        )
        self.boton_reiniciar.pack(side="right", fill="x", expand=True, padx=(4, 0))

        self.boton_ventanas = tk.Button(
            self._pestanas["lanz"], text="☄  VENTANAS DE LANZAMIENTO…",
            bg="#001a33", fg=COLOR_TEXTO_ACENTO,
            font=FUENTE_NEGRITA,
            relief="solid", bd=1, activebackground="#002b52",
            command=self._abrir_ventanas,
        )
        self.boton_ventanas.pack(fill="x", pady=(6, 0))

    # --- Ventanas de lanzamiento (12.7.0) ---
    def _abrir_ventanas(self):
        """Diálogo del planificador de ventanas: origen/destino, fechas y
        límites; calcula en un hilo (la GUI no se congela) y lista una ventana
        por ciclo sinódico; el botón de porkchop dibuja la seleccionada."""
        top = getattr(self, "_top_ventanas", None)
        if top is not None and top.winfo_exists():
            top.lift()
            return
        top = tk.Toplevel(self.root)
        self._top_ventanas = top
        top.title("Ventanas de lanzamiento")
        top.configure(bg=COLOR_FONDO_VENTANA)
        planetas = list(_CAT_VENTANAS)
        try:
            ini = datetime.fromisoformat(self.entradas["fecha_lanzamiento"].get().strip())
        except (ValueError, KeyError):
            ini = datetime.now()
        v = {"origen": tk.StringVar(value="tierra"), "destino": tk.StringVar(value="marte"),
             "inicio": tk.StringVar(value=ini.strftime("%Y-%m-%d")),
             "fin": tk.StringVar(value=(ini + timedelta(days=3 * 365)).strftime("%Y-%m-%d")),
             "c3_max": tk.StringVar(), "dv_salida_max": tk.StringVar(),
             "vinf_llegada_max": tk.StringVar(), "tof_min": tk.StringVar(), "tof_max": tk.StringVar(),
             "criterio": tk.StringVar(value="dv_total"),
             "capturar": tk.BooleanVar(value=True)}
        marco = tk.Frame(top, bg=COLOR_FONDO_PANEL)
        marco.pack(fill="x", padx=8, pady=8)

        def fila(texto, widget_fn):
            f = tk.Frame(marco, bg=COLOR_FONDO_PANEL)
            f.pack(fill="x", pady=1)
            tk.Label(f, text=texto, bg=COLOR_FONDO_PANEL, fg=COLOR_TEXTO_ETIQUETA,
                     font=FUENTE_NORMAL, anchor="w", width=26).pack(side="left")
            widget_fn(f).pack(side="right")

        def entrada(clave, ancho=12):
            return lambda f: tk.Entry(f, textvariable=v[clave], width=ancho, bg=COLOR_FONDO_ENTRADA,
                                      fg=COLOR_TEXTO_ACENTO, insertbackground=COLOR_TEXTO_ACENTO,
                                      relief="solid", bd=1, font=FUENTE_NORMAL)

        def menu(clave, opciones):
            return lambda f: tk.OptionMenu(f, v[clave], *opciones)
        fila("Origen (órbita de estac.)", menu("origen", planetas))
        fila("Destino (órbita de captura)", menu("destino", planetas))
        fila("Desde (AAAA-MM-DD)", entrada("inicio"))
        fila("Hasta (AAAA-MM-DD)", entrada("fin"))
        fila("C3 máx (km²/s², vacío=∞)", entrada("c3_max"))
        fila("Δv salida máx (km/s)", entrada("dv_salida_max"))
        fila("v∞ llegada máx (km/s)", entrada("vinf_llegada_max"))
        fila("TOF mín (días, vacío=auto)", entrada("tof_min"))
        fila("TOF máx (días, vacío=auto)", entrada("tof_max"))
        fila("Optimizar", menu("criterio", ["dv_total", "c3"]))
        _crear_checkbutton(marco, "Incluir captura en el destino", v["capturar"]).pack(anchor="w")
        dv2 = ETAPAS_CONFIG[1]
        ref = dv2.u0 * math.log((dv2.m_estr + dv2.m_comb) / dv2.m_estr) / 1000.0
        tk.Label(marco, text=(f"Efemérides válidas {VentanaLanzamiento.fecha(EfemeridesPlanetas.jd_min)} a "
                              f"{VentanaLanzamiento.fecha(EfemeridesPlanetas.jd_max)} (la llegada también debe caer ahí). "
                              f"Referencia: Δv de la 2ª etapa llena ≈ {ref:.1f} km/s, SIN descontar el "
                              f"ascenso a LEO ni la carga; los límites son tuyos, no se deducen del cohete."),
                 bg=COLOR_FONDO_PANEL, fg=COLOR_TEXTO_ETIQUETA, font=("Courier New", 8),
                 wraplength=420, justify="left").pack(fill="x", pady=(4, 0))
        estado = tk.Label(top, text="", bg=COLOR_FONDO_VENTANA, fg=COLOR_TEXTO_ERROR,
                          font=("Courier New", 9), wraplength=520, justify="left")
        lista = tk.Listbox(top, width=84, height=10, bg=COLOR_FONDO_ENTRADA, fg="#c8d8e8",
                           font=("Courier New", 9), selectbackground="#003a6b", exportselection=False)
        resultado = {"plan": None, "ventanas": [], "hilo": None}

        def calcular():
            if resultado["hilo"] is not None and resultado["hilo"].is_alive():
                return
            try:
                kw = parsear_campos_ventanas({k: x.get() for k, x in v.items()})
            except ValueError as e:
                estado.config(text=f"⚠ {e}", fg=COLOR_TEXTO_ERROR)
                return
            estado.config(text="⏳ Calculando…", fg=COLOR_TEXTO_ACENTO)
            lista.delete(0, "end")
            salida = {}

            def trabajo():
                try:
                    salida["r"] = calcular_ventanas(**kw)
                except Exception as e:         # se informa en el diálogo, no se pierde
                    salida["e"] = e

            def sondear():
                if not top.winfo_exists():
                    return
                if resultado["hilo"].is_alive():
                    top.after(200, sondear)
                    return
                if "e" in salida:
                    estado.config(text=f"⚠ {salida['e']}", fg=COLOR_TEXTO_ERROR)
                    return
                plan, vs = salida["r"]
                resultado["plan"], resultado["ventanas"] = plan, vs
                if not vs:
                    estado.config(text="Sin ventanas que cumplan las restricciones en ese intervalo.",
                                  fg=COLOR_TEXTO_ERROR)
                    return
                for linea in texto_tabla_ventanas(vs, kw["capturar"]).split("\n"):
                    lista.insert("end", linea)
                estado.config(text=f"{len(vs)} ventana(s). Selecciona una fila y pulsa «Porkchop».",
                              fg="#00ffb4")
            resultado["hilo"] = threading.Thread(target=trabajo, daemon=True)
            resultado["hilo"].start()
            top.after(200, sondear)

        def porkchop():
            sel = lista.curselection()
            if resultado["plan"] is None or not sel or sel[0] == 0:
                estado.config(text="Selecciona primero una ventana de la lista.", fg=COLOR_TEXTO_ERROR)
                return
            vent = resultado["ventanas"][sel[0] - 1]
            t2 = tk.Toplevel(top)
            t2.title(f"Porkchop {resultado['plan'].o} → {resultado['plan'].d}")
            try:
                fig = figura_porkchop(resultado["plan"], vent)
            except ValueError as e:
                t2.destroy()
                estado.config(text=f"⚠ {e}", fg=COLOR_TEXTO_ERROR)
                return
            cv = FigureCanvasTkAgg(fig, master=t2)
            cv.draw()
            cv.get_tk_widget().pack(fill="both", expand=True)

        fb = tk.Frame(top, bg=COLOR_FONDO_VENTANA)
        fb.pack(fill="x", padx=8)
        tk.Button(fb, text="CALCULAR", command=calcular, bg="#003318", fg=COLOR_TEXTO_ACTIVO,
                  font=FUENTE_NEGRITA, relief="solid", bd=1).pack(side="left", padx=(0, 4))
        tk.Button(fb, text="PORKCHOP", command=porkchop, bg="#001a33", fg=COLOR_TEXTO_ACENTO,
                  font=FUENTE_NEGRITA, relief="solid", bd=1).pack(side="left", padx=4)
        tk.Button(fb, text="CERRAR", command=top.destroy, bg="#331400", fg="#ffa040",
                  font=FUENTE_NEGRITA, relief="solid", bd=1).pack(side="right")
        estado.pack(fill="x", padx=8, pady=(4, 0))
        lista.pack(fill="both", expand=True, padx=8, pady=8)

    # --- Paneles de texto ---
    # --- (12.8.0) Barra superior y tarjetas de telemetría ---
    def _construir_barra_superior(self):
        barra = tk.Frame(self.root, bg=COLOR_FONDO_PANEL,
                         highlightbackground=COLOR_BORDE_TARJETA, highlightthickness=1)
        barra.place(x=0, y=0, width=self.ANCHO_VENTANA, height=self.ALTO_BARRA)
        tk.Label(barra, text="◉ FLIGHT CONTROL", bg=COLOR_FONDO_PANEL,
                 fg=COLOR_TITULO_TARJETA, font=FUENTE_TITULO_TARJETA).place(
                     x=14, rely=0.5, anchor="w")

        centro = tk.Frame(barra, bg=COLOR_FONDO_PANEL)
        centro.place(relx=0.5, rely=0.5, anchor="center")
        fila_met = tk.Frame(centro, bg=COLOR_FONDO_PANEL)
        fila_met.pack()
        tk.Label(fila_met, text="MET", bg=COLOR_FONDO_PANEL, fg=COLOR_TEXTO_ETIQUETA,
                 font=FUENTE_NEGRITA).pack(side="left", padx=(0, 8))
        self.lbl_met = tk.Label(fila_met, text=_formatear_met(0.0), bg=COLOR_FONDO_PANEL,
                                fg=COLOR_VALOR, font=FUENTE_MET)
        self.lbl_met.pack(side="left")
        self.lbl_fase = tk.Label(centro, text="—", bg=COLOR_TARJETA,
                                 fg=COLOR_TEXTO_ETIQUETA, font=FUENTE_NEGRITA,
                                 padx=14, pady=1)
        self.lbl_fase.pack(pady=(0, 2))

        derecha = tk.Frame(barra, bg=COLOR_FONDO_PANEL)
        derecha.place(relx=1.0, x=-14, rely=0.5, anchor="e")
        self.lbl_estado = tk.Label(derecha, text="LISTO", bg=COLOR_FONDO_PANEL,
                                   fg=COLOR_TEXTO_ETIQUETA, font=("Courier New", 11, "bold"))
        self.lbl_estado.pack(anchor="e")
        self.lbl_modo = tk.Label(derecha, text="", bg=COLOR_FONDO_PANEL,
                                 fg=COLOR_TEXTO_ETIQUETA, font=FUENTE_NORMAL)
        self.lbl_modo.pack(anchor="e")
        self.lbl_detalle = tk.Label(derecha, text="", bg=COLOR_FONDO_PANEL,
                                    fg=COLOR_TEXTO_ERROR, font=("Courier New", 8))
        self.lbl_detalle.pack(anchor="e")

    def _actualizar_barra_superior(self, sim: Optional[Simulacion]):
        if sim is None:
            self.lbl_met.config(text=_formatear_met(0.0))
            self.lbl_fase.config(text="—", fg=COLOR_TEXTO_ETIQUETA)
            self.lbl_estado.config(text="LISTO", fg=COLOR_TEXTO_ETIQUETA)
            cfg = getattr(self, "config", None)
            self.lbl_modo.config(text=_etiqueta_modo(cfg) if cfg is not None else "")
            self.lbl_detalle.config(text="")
            return
        fase = sim.estado.fase
        self.lbl_met.config(text=_formatear_met(sim.estado.t))
        self.lbl_fase.config(text=str(fase).replace("_", " ").upper(),
                             fg=self.COLORES_FASE.get(fase, COLOR_TEXTO_ACENTO))
        texto, color, detalle = _estado_global(sim)
        self.lbl_estado.config(text=texto, fg=color)
        self.lbl_modo.config(text=_etiqueta_modo(sim.config))
        self.lbl_detalle.config(text=detalle)

    def _construir_columna_derecha(self, alto: int):
        marco = tk.Frame(self.root, bg=COLOR_FONDO_VENTANA)
        marco.place(x=self.ANCHO_VENTANA - self.ANCHO_TAB - self.ANCHO_COLUMNA_DER,
                    y=self.ALTO_BARRA, width=self.ANCHO_COLUMNA_DER, height=alto)
        self._marco_tarjetas = marco
        self._tarjetas: dict = {}
        for titulo, _ in TARJETAS_TELEMETRIA:
            self._crear_tarjeta(titulo)
        self._actualizar_tarjetas(FILAS_REPOSO)

    def _crear_tarjeta(self, titulo: str) -> dict:
        contenedor = tk.Frame(self._marco_tarjetas, bg=COLOR_TARJETA,
                              highlightbackground=COLOR_BORDE_TARJETA, highlightthickness=1)
        contenedor.pack(fill="x", padx=8, pady=(8, 0))
        tk.Label(contenedor, text=titulo, bg=COLOR_TARJETA, fg=COLOR_TITULO_TARJETA,
                 font=FUENTE_TITULO_TARJETA, anchor="w").pack(fill="x", padx=10, pady=(8, 2))
        tk.Frame(contenedor, bg=COLOR_BORDE_TARJETA, height=1).pack(fill="x", padx=10)
        cuerpo = tk.Frame(contenedor, bg=COLOR_TARJETA)
        cuerpo.pack(fill="x", padx=10, pady=(4, 8))
        tarjeta = {"cuerpo": cuerpo, "filas": {}, "orden": [], "ultimo": {}}
        self._tarjetas[titulo] = tarjeta
        return tarjeta

    def _crear_fila_tarjeta(self, cuerpo, nombre: str) -> tuple:
        fila = tk.Frame(cuerpo, bg=COLOR_TARJETA)
        lbl_nombre = tk.Label(fila, text=nombre.rstrip(":"), bg=COLOR_TARJETA,
                              fg=COLOR_TEXTO_ETIQUETA, font=FUENTE_NORMAL, anchor="w")
        lbl_nombre.pack(side="left")
        lbl_valor = tk.Label(fila, text="", bg=COLOR_TARJETA, fg=COLOR_VALOR,
                             font=FUENTE_VALOR, anchor="e", justify="right", wraplength=190)
        lbl_valor.pack(side="right")
        return fila, lbl_nombre, lbl_valor

    def _actualizar_tarjetas(self, filas: list):
        """Actualiza las tarjetas con las filas del HUD. Los widgets de cada fila se
        crean una vez y solo se reordenan cuando cambia el conjunto de filas."""
        for titulo, filas_t in _agrupar_filas_en_tarjetas(filas):
            tarjeta = self._tarjetas[titulo]
            nombres = [f[0] for f in filas_t]
            if nombres != tarjeta["orden"]:
                for fila, _, _ in tarjeta["filas"].values():
                    fila.pack_forget()
                for nombre in nombres:
                    if nombre not in tarjeta["filas"]:
                        tarjeta["filas"][nombre] = self._crear_fila_tarjeta(
                            tarjeta["cuerpo"], nombre)
                    tarjeta["filas"][nombre][0].pack(fill="x", pady=1)
                tarjeta["orden"] = nombres
                tarjeta["ultimo"] = {}
            for nombre, valor, tag in filas_t:
                color = COLOR_POR_TAG.get(tag, COLOR_VALOR)
                if tarjeta["ultimo"].get(nombre) != (valor, color):
                    tarjeta["filas"][nombre][2].config(text=_con_miles(valor), fg=color)
                    tarjeta["ultimo"][nombre] = (valor, color)

    def _construir_panel_eventos(self):
        """(12.9.0) Línea de tiempo vertical: un Canvas donde cada evento es un punto de
        color unido al anterior por una línea, con su título y sus detalles debajo."""
        marco = self._pestanas["eventos"]
        lienzo = tk.Canvas(marco, bg=COLOR_FONDO_PANEL, highlightthickness=0)
        barra = tk.Scrollbar(marco, orient="vertical", command=lienzo.yview, width=10,
                             bd=0, relief="flat", highlightthickness=0,
                             troughcolor=COLOR_FONDO_PANEL, bg=COLOR_BORDE_TARJETA,
                             activebackground=COLOR_TEXTO_ACENTO)
        visible = {"barra": False}

        def al_desplazar(primero, ultimo):
            # (12.11.0) la barra solo aparece cuando el contenido no cabe
            barra.set(primero, ultimo)
            hace_falta = float(primero) > 0.0 or float(ultimo) < 1.0
            if hace_falta and not visible["barra"]:
                barra.pack(side="right", fill="y", before=lienzo)
                visible["barra"] = True
            elif not hace_falta and visible["barra"]:
                barra.pack_forget()
                visible["barra"] = False

        lienzo.configure(yscrollcommand=al_desplazar)
        lienzo.pack(side="left", fill="both", expand=True)

        def rueda(evento):
            if getattr(evento, "num", None) == 4:
                pasos = -1
            elif getattr(evento, "num", None) == 5:
                pasos = 1
            else:
                delta = getattr(evento, "delta", 0)
                pasos = -int(delta / 120) if abs(delta) >= 120 else (-1 if delta > 0 else 1)
            lienzo.yview_scroll(pasos, "units")

        for secuencia in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
            lienzo.bind(secuencia, rueda)
        self._linea_tiempo = {"lienzo": lienzo, "y": 8, "y_punto": None, "rueda": rueda,
                              "barra": barra}

    # ========================================================
    # CALLBACKS DE CÁMARA Y TECLADO
    # ========================================================
    def _on_scroll(self, event):
        if event.inaxes != self.ax or event.xdata is None:
            return
        factor = 0.85 if event.button == "up" else 1.15
        xlim, ylim = self.ax.get_xlim(), self.ax.get_ylim()
        x_cursor, y_cursor = event.xdata, event.ydata
        self.ax.set_xlim(x_cursor - (x_cursor - xlim[0]) * factor,
                         x_cursor + (xlim[1] - x_cursor) * factor)
        self.ax.set_ylim(y_cursor - (y_cursor - ylim[0]) * factor,
                         y_cursor + (ylim[1] - y_cursor) * factor)
        self.camara_libre["activa"] = True
        self.canvas.draw_idle()

    def _on_press(self, event):
        if event.inaxes != self.ax or event.xdata is None:
            return
        self.arrastre["activo"] = True
        self.arrastre["x0"] = event.xdata
        self.arrastre["y0"] = event.ydata

    def _on_motion(self, event):
        if not self.arrastre["activo"] or event.inaxes != self.ax:
            return
        if event.xdata is None:
            return
        dx = self.arrastre["x0"] - event.xdata
        dy = self.arrastre["y0"] - event.ydata
        xlim, ylim = self.ax.get_xlim(), self.ax.get_ylim()
        self.ax.set_xlim(xlim[0] + dx, xlim[1] + dx)
        self.ax.set_ylim(ylim[0] + dy, ylim[1] + dy)
        self.camara_libre["activa"] = True
        self.canvas.draw_idle()

    def _on_release(self, _event):
        self.arrastre["activo"] = False

    def _on_key(self, event):
        if event.key == "f":
            self.camara_libre["activa"] = False
        elif event.key == "e":
            self._exportar_historial(".npz")
        elif event.key == "E":
            self._exportar_historial(".csv")
        elif event.key == "m":
            # (12.10.3) la medición bloquea la ventana ~10 s: se avisa antes
            self._mostrar_mensaje_temporal(
                "Midiendo render… la ventana no responde ~10 s", COLOR_AVISO, 30000)
            self.root.update_idletasks()
            try:
                self._medir_render()
                self._medir_animacion()
            finally:
                self.etiqueta_error.config(text="", fg=COLOR_TEXTO_ERROR)

    # ========================================================
    # MEDICIÓN DEL COSTE DEL FOTOGRAMA (tecla M; imprime en consola)
    # ========================================================
    def _medir_render(self, n=30):
        """Coste de canvas.draw() con cada grupo de artistas oculto.

        Δ = baseline - tiempo_sin_grupo: lo que cuesta DIBUJAR ese grupo.
        No mide set_data ni la recreación de artistas (ver
        _medir_animacion). Conviene correrla con la simulación avanzada.
        BLOQUEA LA GUI ~10 s (≈10 grupos x n draws + 2 baselines, a
        ~30 ms por fotograma): diagnóstico de una sola vez.
        """
        import time
        if self.sim is None:
            print("medir_render: no hay simulación")
            return
        artistas = [
            ("tierra",      self.patches_tierra),
            ("grilla",      [self.lc_grilla, self.lc_meridiano0]),
            ("punto_base",  [self.punto_base]),
            ("trayectoria", [self.lc_tray, self.lc_tray_glow]),
            ("orbita",      [self.linea_orbita]),
            ("peri/apo",    [self.punto_peri, self.punto_apo]),
            ("cohete",      [self.punto_cohete]),
            ("luna",        [self.patch_luna]),
            ("hud",         [self.texto_hud]),
            ("inset",       [self.ax_inset]),
            ("lagrange",    list(self.lagrange_artists.values())),
        ]

        def _tiempo():
            t0 = time.perf_counter()
            for _ in range(n):
                self.canvas.draw()
            return (time.perf_counter() - t0) / n

        self.canvas.draw()                      # calentamiento
        base = _tiempo()
        print(f"baseline: {base*1000:.1f} ms "
              f"(trayectoria: {len(self.lc_tray.get_segments())} polilíneas, "
              f"{sum(len(p) - 1 for p in self.lc_tray.get_segments())} segmentos)")
        for nombre, grupo in artistas:
            if not grupo:
                print(f"  {nombre:<12}: sin artistas (omitido)")
                continue
            previo = [a.get_visible() for a in grupo]
            for a in grupo:
                a.set_visible(False)
            dt = _tiempo()
            print(f"  sin {nombre:<12}: {dt*1000:.1f} ms  "
                  f"(Δ {(base-dt)*1000:+.1f})")
            for a, v in zip(grupo, previo):
                a.set_visible(v)
        print(f"baseline final: {_tiempo()*1000:.1f} ms (deriva del ruido)")

    def _medir_animacion(self, n=20):
        """Coste de _actualizar_animacion (no avanza la física).

        Tres pasadas, la 1ª y la 2ª con la cámara automática CONGELADA
        (camara_libre activa) para que el encuadre adaptativo no se
        mezcle con Lagrange:
          1. cámara libre, mostrar_crtbp como está
          2. cámara libre, mostrar_crtbp=False   -> Δ lagrange = 1 - 2
          3. cámara automática, mostrar_crtbp=False -> Δ cámara = 3 - 2
        La escala log queda como esté (es parte del coste real y es
        igual en todas las pasadas). Δ cámara solo cubre el código de
        encuadre de _actualizar_animacion, no el re-trazado de ejes que
        ocurre dentro de canvas.draw(). BLOQUEA LA GUI ~1-2 s.
        """
        import time
        sim = self.sim
        if sim is None:
            return
        cfg = sim.config
        crtbp_orig = cfg.mostrar_crtbp
        cam_orig = self.camara_libre["activa"]

        def _tiempo():
            t0 = time.perf_counter()
            for _ in range(n):
                self._actualizar_animacion()
            return (time.perf_counter() - t0) / n

        try:
            self.camara_libre["activa"] = True
            self._actualizar_animacion()        # calentamiento
            t_actual = _tiempo()
            cfg.mostrar_crtbp = False
            t_sin = _tiempo()
            self.camara_libre["activa"] = False
            t_auto = _tiempo()
        finally:
            cfg.mostrar_crtbp = crtbp_orig
            self.camara_libre["activa"] = cam_orig
            self._actualizar_animacion()
        print(f"_actualizar_animacion (cámara libre): {t_actual*1000:.1f} ms "
              f"(mostrar_crtbp={crtbp_orig}); con mostrar_crtbp=False: "
              f"{t_sin*1000:.1f} ms  (Δ lagrange "
              f"{(t_actual-t_sin)*1000:+.1f})")
        print(f"  cámara automática, mostrar_crtbp=False: {t_auto*1000:.1f} ms  "
              f"(Δ cámara {(t_auto-t_sin)*1000:+.1f})")

    # ========================================================
    # ACCIONES DE LA INTERFAZ (drawer, modo, visibilidad)
    # ========================================================
    def _alternar_drawer(self, _event=None):
        if self.drawer_estado["abierto"]:
            self.drawer_frame.place_forget()
            self.tab_canvas.itemconfig(self.tab_texto,
                                       text="◀ GRÁFICA DE VUELO")
        else:
            self.drawer_frame.place(x=self.x_drawer, y=self.ALTO_BARRA,
                                    width=self.ANCHO_DRAWER,
                                    height=self.ALTO_VENTANA - self.ALTO_BARRA)
            self.drawer_frame.lift()
            self.tab_canvas.itemconfig(self.tab_texto,
                                       text="▶ GRÁFICA DE VUELO")
        self.drawer_estado["abierto"] = not self.drawer_estado["abierto"]
        self._t_drawer = float("-inf")           # (12.13.0) al abrir se dibuja en el siguiente cuadro

    def _mision_activa(self) -> bool:
        return self.sim is not None and not self.sim.estado.simulacion_terminada

    def _cambiar_modo(self):
        if self._mision_activa():
            # (12.7.1) el modo no cambia en pleno vuelo: se revierte el selector
            self.var_modo.set(self.sim.config.modo_mision)
            self._mostrar_mensaje_temporal(
                "No se puede cambiar el modo durante una misión: pulsa REINICIAR",
                COLOR_TEXTO_ERROR, 3000)
            return
        self.config.modo_mision = self.var_modo.get()
        self.config.aplicar_preset()
        if hasattr(self, "lbl_modo"):
            self._actualizar_barra_superior(None)       # (12.8.2) la barra muestra el modo elegido
        self._refrescar_campos_del_preset()
        self._actualizar_visibilidad_perfil()
        self._actualizar_visibilidad_cutoff_subor()
        self._mostrar_mensaje_temporal(
            f"Preset aplicado: modo {self.config.modo_mision}",
            COLOR_TEXTO_ACTIVO)

    def _refrescar_campos_del_preset(self):
        """Refresca los campos que `aplicar_preset` pudo modificar."""
        cfg = self.config
        self.entradas["APOGEO_OBJETIVO"].set(
            str(int(cfg.APOGEO_OBJETIVO))
            if np.isfinite(cfg.APOGEO_OBJETIVO) else "inf"
        )
        self.entradas["ALTURA_SEGURIDAD"].set(str(int(cfg.ALTURA_SEGURIDAD)))
        self.entradas["VEL_MIN_SES"].set(str(int(cfg.VEL_MIN_SES)))
        self.entradas["angulo_kick"].set(str(cfg.angulo_kick))

    def _exportar_historial(self, extension: str = ".npz"):
        """(12.12.0/12.12.1) Tecla E (.npz) / Shift+E (.csv): guarda el historial completo (incluye
        lo podado) en la carpeta del proyecto, sin diálogo."""
        sim = self.sim
        if sim is None or len(sim.hist_t) == 0:
            self._mostrar_mensaje_temporal("No hay historial que exportar", COLOR_AVISO, 4000)
            return
        try:
            destino = exportar_historial(
                sim, _ruta_exportacion(sim.config.modo_mision, extension))
        except (OSError, ValueError) as exc:
            self._mostrar_mensaje_temporal(f"No se pudo exportar: {exc}", COLOR_TEXTO_ERROR, 8000)
            return
        print(f"Historial exportado: {destino}")
        self._mostrar_mensaje_temporal(f"Historial exportado: {Path(destino).name}", COLOR_OK, 6000)

    def _mostrar_mensaje_temporal(self, texto: str, color: str,
                                  duracion_ms: int = 1500):
        self.etiqueta_error.config(text=texto, fg=color)
        self.root.after(
            duracion_ms,
            lambda: self.etiqueta_error.config(text="", fg=COLOR_TEXTO_ERROR)
        )

    def _mostrar_error(self, texto: str):
        self.etiqueta_error.config(text=texto, fg=COLOR_TEXTO_ERROR)

    def _al_cambiar_escala_log(self):
        """Aplica el checkbox de escala log en vivo (solo vista) y
        reactiva la cámara automática para reencuadrar."""
        valor = bool(self.var_escala_log.get())
        self.config.escala_log = valor
        if self.sim is not None:
            self.sim.config.escala_log = valor
            self._actualizar_animacion()
        self.camara_libre["activa"] = False
        self.canvas.draw_idle()

    def _actualizar_visibilidad_cutoff_subor(self):
        if self.var_modo.get() == "suborbital":
            self.chk_cutoff_subor.pack(anchor="w", pady=(6, 0),
                                       before=self.chk_rotacion)
        else:
            self.chk_cutoff_subor.pack_forget()

    def _actualizar_visibilidad_perfil(self):
        if self.var_modo.get() == "tripulada":
            kw = {"before": self._sep_config} if getattr(self, "_sep_config", None) else {}
            self.frame_perfil_tripulada.pack(fill="x", pady=(4, 0), **kw)
        else:
            self.frame_perfil_tripulada.pack_forget()

    @staticmethod
    def _formatear_alt(alt_m: float) -> str:
        return _formatear_altitud(alt_m)

    # ========================================================
    # CICLO DE VIDA DE LA SIMULACIÓN (iniciar / reiniciar / bucle)
    # ========================================================
    def _leer_config_desde_widgets(self) -> bool:
        """Vuelca los widgets en `self.config`. Si algo es inválido
        muestra el error y devuelve False."""
        cfg = copy.deepcopy(self.config)      # (12.9.1) transaccional: self.config solo cambia si todo es válido
        cfg.modo_mision = self.var_modo.get()
        cfg.perfil_tripulada = self.var_perfil.get()
        cfg.mostrar_crtbp = bool(self.var_crtbp.get())
        cfg.suborbital_cutoff_por_apogeo = bool(self.var_cutoff_subor.get())
        cfg.escala_log = bool(self.var_escala_log.get())
        cfg.mostrar_grilla_tierra = bool(self.var_grilla.get())
        cfg.marco_vista = ("baricentro" if self.var_vista_bari.get()
                           else "tierra")
        etiquetas = dict(self.CAMPOS_NUMERICOS)
        valores = {}
        for clave, convertir in self.CONVERSORES_CAMPOS:
            texto = self.entradas[clave].get()
            try:
                valores[clave] = convertir(texto)
            except (ValueError, OverflowError) as e:      # (12.8.4) OverflowError: int(float("1e400"))
                self._mostrar_error(
                    f"⚠ Campo «{etiquetas.get(clave, clave)}»: {texto!r} no es válido "
                    f"({type(e).__name__})")
                return False
        for clave, valor in valores.items():              # solo si TODOS convirtieron
            setattr(cfg, clave, valor)
        cfg.rotacion_terrestre = bool(self.var_rotacion.get())
        var_loi = getattr(self, "var_loi", None)
        if var_loi is not None:
            cfg.LUNAR_LOI = bool(var_loi.get())
        var_efem = getattr(self, "var_efem_spice", None)
        cfg.efemerides_lunar = ("spice" if (var_efem is not None and var_efem.get())
                                else "rebound")
        if not (-90.0 <= cfg.latitud_lanzamiento <= 90.0
                and 0.0 <= cfg.azimut_lanzamiento <= 180.0):
            self._mostrar_error(
                "⚠ Latitud en [-90, 90] y azimut en [0, 180] "
                "(el modelo plano solo vuela hacia el Este)")
            return False
        try:
            cfg._validar_config()
        except ValueError as e:
            problemas = str(e).split("\n  - ")[1:]
            self._mostrar_error("⚠ " + "; ".join(problemas[:3])
                                + (f" (+{len(problemas) - 3} más)"
                                   if len(problemas) > 3 else ""))
            return False
        self.config = cfg
        return True

    def _iniciar(self):
        # Tk procesa callbacks secuencialmente. Este guard evita que una
        # segunda llamada de botón reinicialice una misión que ya avanza.
        if (self._after_id is not None and self.sim is not None
                and not self.sim.estado.simulacion_terminada):
            print("⚠ Inicio duplicado ignorado: ya hay una misión activa.")
            return
        if not self._leer_config_desde_widgets():
            return
        cfg = self.config

        self._cancelar_loop()
        self.etiqueta_error.config(text="", fg=COLOR_TEXTO_ERROR)
        self._limpiar_paneles_y_animacion()

        self._cancelar_planificador()
        try:
            # (12.7.1) la simulación recibe una COPIA: los widgets siguen editando
            # `self.config` sin tocar la misión en curso
            self.sim = Simulacion(
                copy.deepcopy(cfg),
                log=lambda mensaje: (print(mensaje), self._agregar_evento(
                    mensaje, "fallo" if str(mensaje).startswith(("⚠", "✗")) else None)))
        except (EfemeridesNoValidas, ValueError) as e:
            # (12.3.2) modo lunar: sin efemérides válidas la misión no arranca;
            # se dice por qué en vez de dejar caer el bucle de Tk.
            self.sim = None
            print(f"✗ No se pudo iniciar: {e}")
            self._mostrar_error("⚠ " + str(e).replace("\n", " ")[:200])
            self.boton_iniciar.config(state="normal", text="▶  INICIAR")
            return
        self.sim.presupuesto_plan_s = cfg.DEORBIT_PLAN_PRESUPUESTO_S
        self._instalar_planificador(self.sim.config)
        self._imprimir_resumen_inicio()

        self.boton_iniciar.config(state="disabled", text="◉  SIMULANDO...")
        self._elegir_pestana("eventos")
        self._pasos_restantes = None
        self._after_id = self.root.after(10, self._paso_simulacion)

    def _imprimir_resumen_inicio(self):
        sim = self.sim
        cfg = sim.config
        print(f"▶ INICIO: modo={_etiqueta_modo(cfg)}, usa_bcb={sim.usa_bcb()}, "
              f"terceros={cfg.usar_terceros_cuerpos}, marco_movil={cfg.usar_marco_movil}, "
              f"crtbp_overlay={cfg.mostrar_crtbp}, efemerides={sim.efemerides_efectiva()}, "
              f"rotacion={cfg.rotacion_terrestre} "
              f"(v0={sim.estado.vx:.1f} m/s, i={cfg.inclinacion_orbital():.1f}°)")

    def _instalar_planificador(self, cfg: Config) -> None:
        """Desórbita al sitio: lanza el cálculo de la hora de quema en un
        proceso aparte, en paralelo al ascenso. Si no se puede, queda el
        camino de siempre (planificar al detectar la órbita)."""
        if (self.sim is None or not self.sim._deorbit_sitio
                or not PlanificadorDeorbitExterno.aplicable(cfg)):
            return
        try:
            self._planificador = PlanificadorDeorbitExterno(cfg)
        except Exception as e:
            print(f"⚠ No se pudo lanzar el proceso de planificación "
                  f"({type(e).__name__}: {e}); se planificará al detectar la órbita")
            return
        self.sim._planificador_externo = self._planificador
        print(f"⏳ Planificador de desórbita lanzado en un proceso aparte "
              f"(pid {self._planificador._proc.pid}); la quema se calcula en paralelo al ascenso")

    def _cancelar_planificador(self) -> None:
        if self._planificador is not None:
            self._planificador.cancelar()
            self._planificador = None

    def _reiniciar(self):
        self._cancelar_loop()
        self._error_refresco_visto = False
        self._cancelar_planificador()
        if self.sim is not None:
            self.sim.cerrar_historial()               # (12.12.0) borra el archivo temporal
        self.sim = None
        self._limpiar_paneles_y_animacion()
        self.etiqueta_error.config(text="", fg=COLOR_TEXTO_ERROR)
        self.boton_iniciar.config(state="normal", text="▶  INICIAR")
        self._elegir_pestana("mision")

    def _limpiar_paneles_y_animacion(self):
        self.eventos_ya_mostrados.clear()
        self._limpiar_linea_tiempo()
        self._limpiar_texto_sync()
        self._limpiar_animacion()

    def _cancelar_loop(self):
        self._pasos_restantes = None
        if self._after_id is not None:
            self.root.after_cancel(self._after_id)
            self._after_id = None

    def _finalizar_bucle_simulacion(self) -> None:
        """Detiene solo la animación; la ventana principal sigue activa."""
        self._after_id = None
        self.boton_iniciar.config(state="normal", text="↻  NUEVA SIMULACIÓN")

    def _cancelar_redibujados_pendientes(self) -> None:
        """Cancela dibujados idle ya encolados por TkAgg al terminar.

        La API de Matplotlib no ofrece un método público para cancelar un
        draw_idle. Se comprueba el atributo interno de TkAgg de forma defensiva;
        si cambia en otra versión, se ignora y la misión ya sigue terminada.
        """
        for canvas in (getattr(self, "canvas", None),
                       getattr(self, "canvas2", None)):
            if canvas is None:
                continue
            draw_id = getattr(canvas, "_idle_draw_id", None)
            if draw_id:
                try:
                    canvas.get_tk_widget().after_cancel(draw_id)
                except (tk.TclError, AttributeError):
                    pass
                try:
                    canvas._idle_draw_id = None
                except Exception:
                    pass

    def _refrescar_terminal_sin_graficas(self) -> None:
        """Presenta el resultado terminal sin reconstruir gráficos pesados.

        Se llama al terminar la propagación, por ejemplo en T_MAX_LUNAR
        después de un encuentro SOI (encuentro, no captura). No se cierra Tk:
        la ventana permanece utilizable y permite iniciar otra simulación.
        Se cancelan repintados pendientes para evitar
        que una trayectoria grande se dibuje después del final.
        """
        if self.sim is None:
            return
        self._cancelar_redibujados_pendientes()
        try:
            self._actualizar_panel_sincrono()
        except Exception as e:
            print(f"⚠ No se pudo refrescar el panel final: {type(e).__name__}: {e}")
        try:
            self._actualizar_eventos_incrementales()
        except Exception as e:
            print(f"⚠ No se pudo refrescar el panel de eventos final: {type(e).__name__}: {e}")

        ev = self.sim.eventos
        if self.sim.config.modo_mision == "lunar" and ev.impacto_lunar_registrado:
            d = ev.datos_impacto_lunar
            mensaje = (f"💥 IMPACTO LUNAR a t={d['t'] / 3600:.2f} h; "
                       f"v relativa={d['v_rel']:.0f} m/s. Simulación terminada por contacto."
                       f"{_nota_3d(d)}")
            color = COLOR_TEXTO_ERROR
        elif self.sim.config.modo_mision == "lunar" and ev.datos_mision.get("resultado") == "captura_lunar":
            lo = ev.datos_mision["loi"]
            mensaje = (f"🌕 CAPTURA LUNAR (2D, provisional): {(lo['rp'] - R_LUNA) / 1000:.0f}x"
                       f"{(lo['ra'] - R_LUNA) / 1000:.0f} km de altitud, "
                       f"{lo.get('periodos_verificados', 0):.1f} periodos verificados.")
            color = "#00ffb4"
        elif (self.sim.config.modo_mision == "lunar" and ev.entro_soi_luna
                and ev.mision_cumplida is False):
            mensaje = "MISIÓN NO CUMPLIDA: " + str(ev.datos_mision.get("motivo", "sin captura"))
            color = COLOR_TEXTO_ERROR
        elif self.sim.config.modo_mision == "lunar" and ev.entro_soi_luna:
            mensaje = "✓ ENCUENTRO SOI lunar (sobrevuelo, sin captura ni impacto)."
            color = "#00ffb4"
        elif ev.mision_cumplida is True:
            mensaje = "✓ MISIÓN CUMPLIDA. Ventana abierta; puedes iniciar otra simulación."
            color = "#00ffb4"
        elif ev.mision_cumplida is False:
            mensaje = "MISIÓN NO CUMPLIDA: " + str(ev.datos_mision.get("motivo", "fin terminal"))
            color = COLOR_TEXTO_ERROR
        else:
            mensaje = "Simulación detenida. Puedes iniciar otra misión."
            color = COLOR_TEXTO_ETIQUETA
        try:
            self.etiqueta_error.config(text=mensaje, fg=color)
        except tk.TclError:
            pass

        self._finalizar_bucle_simulacion()
        print("✓ GUI: cierre del bucle terminal completado; Tk permanece abierto.")

    PRESUPUESTO_FISICA_S = 0.040

    def _paso_simulacion(self):
        sim = self.sim
        if sim is None:
            return
        est = sim.estado
        if est.simulacion_terminada:
            self._refrescar_terminal_sin_graficas()
            return

        cfg = sim.config
        pasos = max(1, cfg.sim_speed)
        if sim._lunar and est.fase in ("espera_tli", "translunar", "orbita_lunar"):
            pasos = max(pasos, cfg.LUNAR_PASOS_MIN_COAST)     # 12.5.6
        retardo = 10
        pl = sim._planificador_externo
        if pl is not None and not pl.listo():
            # El plan de desórbita se calcula en otro proceso (~15 s). Con
            # velocidades altas la nave llegaría a la órbita antes y la
            # simulación se quedaría parada esperando; se limita la
            # velocidad hasta que el plan esté listo para que el ascenso
            # siga fluido y llegue a la órbita cuando el plan ya existe.
            pasos = min(pasos, max(1, cfg.DEORBIT_VEL_MAX_PLAN))
        if est.fase == "deorbit":
            # La quema de desórbita dura ~7 s simulados (~30 pasos): a
            # 100 pasos/s pasa en una fracción de segundo y apsides y
            # periapsis parecen saltar. Un paso por fotograma, más lento.
            pasos = 1
            retardo = max(10, cfg.DEORBIT_PASO_QUEMA_MS)

        restantes = getattr(self, "_pasos_restantes", None)
        if restantes is None:
            restantes = pasos                # lote nuevo
        limite = _monotonic() + self.PRESUPUESTO_FISICA_S
        try:
            while restantes > 0:
                if est.simulacion_terminada:
                    restantes = 0
                    break
                sim.avanzar_paso()
                restantes -= 1
                if sim.esperando_plan:      # sin plan no hay física: no reintentar en este fotograma
                    restantes = 0
                    break
                if est.fase == "deorbit" and pasos > 1:
                    retardo = max(10, cfg.DEORBIT_PASO_QUEMA_MS)
                    restantes = 0
                    break                   # empieza la quema: el resto va un paso por fotograma
                if restantes > 0 and _monotonic() >= limite:
                    break                   # (12.13.0) rebanada agotada: se cede el control a Tk
        except Exception as e:              # 12.5.6: que no muera el bucle de Tk sin avisar
            import traceback
            traceback.print_exc()
            self._mostrar_error(f"⚠ Error de simulación: {type(e).__name__}: {e}"[:200])
            self._pasos_restantes = None
            self._finalizar_bucle_simulacion()
            return
        if restantes > 0 and not est.simulacion_terminada:
            # (12.13.0) lote a medias: Tk atiende clics y repintados. El límite es best-effort:
            # un paso individual puede exceder el presupuesto, pero se completa al menos un paso.
            self._pasos_restantes = restantes
            self._after_id = self.root.after(1, self._paso_simulacion)
            return
        self._pasos_restantes = None
        if sim.esperando_plan:
            # Planificando por tramos: fotogramas seguidos (el tiempo de
            # cálculo ya está acotado por el presupuesto). Esperando al
            # proceso aparte: ritmo lento de espera.
            retardo = (10 if sim._plan_gen is not None
                       else max(retardo, cfg.DEORBIT_PASO_ESPERA_MS))

        if est.simulacion_terminada:
            self._refrescar_terminal_sin_graficas()
            return
        self._refrescar_todo_seguro()
        self._after_id = self.root.after(retardo, self._paso_simulacion)

    def _refrescar_todo_seguro(self):
        """(12.5.6) `_refrescar_todo` sin dejar que una excepción de dibujo
        detenga el bucle: traceback una sola vez y aviso en el panel."""
        try:
            self._refrescar_todo()
        except Exception as e:
            if not getattr(self, "_error_refresco_visto", False):
                import traceback
                traceback.print_exc()
                self._mostrar_error(
                    f"⚠ Error al dibujar ({type(e).__name__}); la simulación sigue. "
                    f"Mira la consola.")
            self._error_refresco_visto = True

    def _refrescar_todo(self):
        self._actualizar_panel_sincrono()
        self._actualizar_eventos_incrementales()
        self._actualizar_animacion()

    # ========================================================
    # DIBUJO: LIMPIEZA Y GRILLA
    # ========================================================
    def _dibujar_grilla_tierra(self, angulo_base, offset_x, offset_y, visible):
        """Meridianos cada 30° (el 0 es el de la base de lanzamiento) y
        paralelos a 30° y 60°, centrados en (offset_x, offset_y).
        `angulo_base` es el ángulo actual del meridiano de la base."""
        if not visible:
            self.lc_grilla.set_segments([])
            self.lc_meridiano0.set_segments([])
            self.punto_base.set_data([], [])
            return
        centro = np.array([offset_x, offset_y])
        angulos = angulo_base - np.radians(30.0) * np.arange(12)  # sentido horario
        puntas = centro + R_T * np.column_stack((np.cos(angulos), np.sin(angulos)))
        meridianos = [np.array([centro, punta]) for punta in puntas]
        self.lc_meridiano0.set_segments([meridianos[0]])
        self.lc_grilla.set_segments(meridianos[1:]
                                    + [paralelo + centro for paralelo in self._paralelos])
        self.punto_base.set_data([puntas[0, 0]], [puntas[0, 1]])

    def _limpiar_animacion(self):
        self._dibujar_grilla_tierra(np.pi / 2.0, 0.0, 0.0,
                                    self.config.mostrar_grilla_tierra)
        self.lc_tray.set_segments([])
        self.lc_tray.set_color([])
        self.lc_tray_glow.set_segments([])
        self.lc_tray_glow.set_color([])
        self.linea_orbita.set_data([], [])
        self.punto_peri.set_data([], [])
        self.punto_apo.set_data([], [])
        self.punto_cohete.set_data([], [])
        self.texto_hud.set_text("")
        self._ocultar_inset()
        self._borrar_puntos_lagrange()
        self.patch_luna.center = (1e12, 0)
        for patch in self.patches_tierra:
            patch.center = (0, 0)
        radio_vista = R_T * 1.4
        self.ax.set_xlim(-radio_vista, radio_vista)
        self.ax.set_ylim(-radio_vista, radio_vista)
        self.camara_libre["activa"] = False
        self.canvas.draw_idle()

        self.linea_alt.set_data([], [])
        self.linea_vel.set_data([], [])
        self.canvas2.draw_idle()

    # ========================================================
    # PANEL DE DATOS SÍNCRONOS
    # ========================================================
    def _limpiar_texto_sync(self):
        """(12.8.0) Vacía la barra superior y las tarjetas de telemetría."""
        self._actualizar_barra_superior(None)
        self._actualizar_tarjetas(FILAS_REPOSO)

    def _actualizar_panel_sincrono(self):
        sim = self.sim
        if sim is None:
            return
        filas = (self._filas_estado_general(sim)
                 + self._filas_distancias_efemerides(sim)
                 + self._filas_carga_g_y_orbita(sim))

        self._actualizar_barra_superior(sim)
        self._actualizar_tarjetas(filas)

    def _filas_estado_general(self, sim: Simulacion) -> list:
        """Filas (nombre, valor, tag): tiempo, fase, modo, marco, empuje..."""
        est = sim.estado
        etapa = sim.etapa_actual()
        combustible = etapa.m_comb_restante if etapa else 0.0

        filas = [
            ("Tiempo:",     f"{est.t:.1f} s", "normal"),
            ("Altitud:",    _formatear_altitud(est.altitud), "normal"),
            ("Velocidad:",  f"{est.speed:.1f} m/s", "normal"),
            ("Fase:",       est.fase, "fase"),
            ("Modo:",       _etiqueta_modo(sim.config), "normal"),
            ("B-C-B:",      "sí" if sim.usa_bcb() else "no",
             "fase" if sim.usa_bcb() else "normal"),
        ]
        if sim.esperando_plan:
            pl = sim._planificador_externo
            if pl is not None:
                seg = f" ({pl.transcurrido():.0f} s)"
            elif sim._plan_gen is not None:
                seg = f" ({_monotonic() - sim._plan_t0:.0f} s)"
            else:
                seg = ""
            filas.append(("Estado:", ("planificando TLI..." if sim.es_lunar()
                                      else "planificando desórbita...") + seg, "fase"))
        if sim._efemerides_modo != "sin_uso":
            filas.append(("Efemérides:", sim.efemerides_efectiva(), "fase"))
        if sim._validacion_horizons is not None:
            val = sim._validacion_horizons
            err_l = max(float(v["dr_m"]) for v in val["errores"]["luna"].values()) / 1000.0
            filas.append(("Validación efemérides:",
                          f"Horizons · Δr Luna máx. {err_l:.3f} km",
                          "normal" if val.get("aceptada", False) else "alerta"))
        elif isinstance(sim._proveedor_efemerides, _ProveedorRebound):
            ok = sim._proveedor_efemerides.con_planetas
            filas.append(("Validación efemérides:",
                          "REBOUND vs SPICE · Luna 0,3 km, Sol 0,1 km (10 d)" if ok
                          else "REBOUND sin planetas · Sol ~50 km a 10 d",
                          "normal" if ok else "alerta"))
        elif isinstance(sim._proveedor_efemerides, _ProveedorHorizonsVerlet):
            filas.append(("Validación efemérides:", "NO MEDIDA contra Horizons", "alerta"))
        if est.marco != "tierra":
            filas.append(("Marco:", est.marco, "fase"))
        if est.marco == "luna":
            r_luna = math.hypot(est.px, est.py)
            zona = zona_lunar(r_luna)
            filas.append(("Zona lunar:",
                          f"{zona} (r/Hill={r_luna / R_HILL_LUNA:.2f})",
                          "fase" if zona == "estable" else "normal"))
        if sim.config.modo_mision == "suborbital":
            filas.append(("Alt. máx:", _formatear_altitud(sim.eventos.alt_max),
                          "normal"))
        if sim.es_alta_energia():
            filas.append(("r máx ECI:",
                          _formatear_altitud(sim.eventos.r_max_eci), "normal"))
            if np.isfinite(sim.eventos.d_luna_min):
                filas.append(("Dmin Luna:",
                              _formatear_altitud(sim.eventos.d_luna_min),
                              "normal"))
        if sim.es_lunar():
            filas.extend(self._filas_lunar(sim))
        filas.append(("Combustible:", f"{combustible:.0f} kg", "normal"))
        filas.append(("Empuje:", f"{est.empuje / 1000:.1f} kN",
                      "empuje_on" if est.empuje > 0 else "empuje_off"))
        return filas

    def _filas_lunar(self, sim: Simulacion) -> list:
        """(12.3.2) Filas del HUD del modo lunar: aparcamiento, plan TLI,
        cuenta atrás al encendido, distancia a la Luna y SOI."""
        est, ev = sim.estado, sim.eventos
        filas = []
        ap = ev.datos_mision.get("aparcamiento")
        if ap:
            filas.append(("Aparcam.:", f"{ap['rp'] / 1000:.0f}x{ap['ra'] / 1000:.0f} km"
                          + (" (seguro)" if ap["seguro"] else " (NO seguro)"), "normal"))
        pl = sim._lunar_plan
        if pl is not None:
            if est.fase == "espera_tli":
                filas.append(("Encendido:", f"en {pl['t_encendido'] - est.t:.0f} s", "fase"))
            filas.append(("SOI prevista:", f"{pl['t_soi_pred'] / 3600:.1f} h  "
                          f"(paso {pl['d_min_pred'] / 1000:.0f} km)", "normal"))
        if est.fase in ("quema_tli", "translunar", "quema_loi", "orbita_lunar") and np.isfinite(sim._lunar_d_luna):
            filas.append(("Dist. Luna:", _formatear_altitud(sim._lunar_d_luna), "normal"))
        lo = ev.datos_mision.get("loi")
        if sim._loi is not None and est.fase == "quema_loi":
            filas.append(("LOI:", "quemando (retrógrado a la Luna)", "empuje_on"))
        elif lo:
            filas.append(("Órbita lunar:",
                          f"{(lo['rp'] - R_LUNA) / 1000:.0f}x{(lo['ra'] - R_LUNA) / 1000:.0f} km alt"
                          + (" (CAPTURA 2D)" if ev.datos_mision.get("resultado") == "captura_lunar"
                             else " (verificando)"), "empuje_on"))
        if ev.entro_soi_luna:
            so = ev.datos_soi_luna
            filas.append(("SOI lunar:", f"t={so['t'] / 3600:.2f} h, v_rel={so['v_rel']:.0f} m/s",
                          "empuje_on"))
        if ev.impacto_lunar_registrado:
            imp = ev.datos_impacto_lunar
            filas.append(("Impacto lunar:", f"t={imp['t'] / 3600:.2f} h / "
                          f"{imp['v_rel']:.0f} m/s"
                          + (" (2D, no físico 3D)" if imp.get("fisico_3d") is False else ""),
                          "alerta"))
        elif ev.mision_cumplida is False:
            filas.append(("Misión:", ev.datos_mision.get("motivo", "NO CUMPLIDA"),
                          "empuje_off"))
        return filas

    def _filas_distancias_efemerides(self, sim: Simulacion) -> list:
        """Distancias a Luna y Sol (solo si hay efemérides en uso)."""
        if not (sim.config.usar_terceros_cuerpos or sim.config.usar_marco_movil):
            return []
        est = sim.estado
        px_d, py_d = sim.posicion_display_eci()
        luna_x, luna_y = sim._posicion_cuerpo("luna", est.t)
        d_luna = float(np.hypot(luna_x - px_d, luna_y - py_d))
        filas = [("Dist. Luna:", _formatear_altitud(d_luna), "normal")]
        if sim.config.usar_terceros_cuerpos:
            sol_x, sol_y = sim._posicion_cuerpo("sol", est.t)
            d_sol = float(np.hypot(sol_x - px_d, sol_y - py_d))
            filas.append(("Dist. Sol:", f"{d_sol / 1e6:.2f} Mm", "normal"))
        return filas

    def _filas_carga_g_y_orbita(self, sim: Simulacion) -> list:
        """Carga g y elementos orbitales: solo tienen sentido en la zona
        "tierra" (es la única con guiado/empuje y noción de órbita por
        ahora; en crucero/luna px,py no se miden desde la Tierra)."""
        est = sim.estado
        if est.marco != "tierra":
            return []
        orb = elementos_orbitales(est.px, est.py, est.vx, est.vy)

        # Carga g = aceleración propia ≈ T/m (la gravedad no se suma: en
        # órbita, sin empuje, la tripulación siente 0 g, no 1 g).
        a_total = est.empuje / max(1.0, sim.masa_total())
        g_sentida = a_total / G0
        tag_g = ("alerta"
                 if (sim.config.modo_mision == "tripulada"
                     and g_sentida > sim.config.G_MAX_TRIPULADA)
                 else "normal")

        ecc_max = sim.ecc_max_efectiva()
        tag_ecc = "alerta" if orb["ecc"] >= ecc_max else "normal"
        rp_txt = (f"{orb['rp'] / 1000:.1f} km"
                  + (" ⚠" if orb["rp"] < 0 else ""))
        ra_txt = ("∞" if not np.isfinite(orb["ra"])
                  else f"{orb['ra'] / 1000:.1f} km")
        return [
            ("Carga g:",    f"{g_sentida:.2f} g", tag_g),
            ("Excentric.:", (f"{orb['ecc']:.4f}" if ecc_max >= 1.0
                             else f"{orb['ecc']:.4f} (≤{ecc_max:.3f})"), tag_ecc),
            ("Periapsis:",  rp_txt, "alerta" if orb["rp"] < 0 else "normal"),
            ("Apoapsis:",   ra_txt, "normal"),
        ]

    # ========================================================
    # PANEL DE EVENTOS
    # ========================================================
    X_PUNTO_EVENTO = 14
    X_TEXTO_EVENTO = 30
    ANCHO_TEXTO_EVENTO = 265

    def _limpiar_linea_tiempo(self):
        lt = self._linea_tiempo
        lt["lienzo"].delete("all")
        lt["y"], lt["y_punto"] = 8, None
        lt["lienzo"].configure(scrollregion=(0, 0, 1, 1))

    def _agregar_evento(self, texto, tag=None):
        """Con `tag` (titulo/exito/fallo) abre un evento nuevo; sin tag el texto es un
        detalle del último evento."""
        lt = self._linea_tiempo
        lienzo = lt["lienzo"]
        texto = str(texto).strip()
        if not texto:
            return
        estaba_abajo = _vista_al_final(lienzo.yview())    # (12.11.0) antes de crecer el contenido
        if tag in COLOR_EVENTO:
            y = lt["y"] + (6 if lt["y_punto"] is not None else 0)
            xp = self.X_PUNTO_EVENTO
            if lt["y_punto"] is not None:
                lienzo.create_line(xp, lt["y_punto"], xp, y + 7,
                                   fill=COLOR_BORDE_TARJETA, width=2, tags="linea")
                lienzo.tag_lower("linea")
            lienzo.create_oval(xp - 5, y + 2, xp + 5, y + 12,
                               fill=COLOR_EVENTO[tag], outline="")
            item = lienzo.create_text(self.X_TEXTO_EVENTO, y, text=texto, anchor="nw",
                                      fill=COLOR_TITULO_TARJETA, width=self.ANCHO_TEXTO_EVENTO,
                                      font=("Courier New", 9, "bold"))
            lt["y_punto"] = y + 7
            lt["y"] = lienzo.bbox(item)[3] + 2
        else:
            item = lienzo.create_text(self.X_TEXTO_EVENTO, lt["y"], text=texto, anchor="nw",
                                      fill=COLOR_TEXTO_ETIQUETA, width=self.ANCHO_TEXTO_EVENTO,
                                      font=("Courier New", 8))
            lt["y"] = lienzo.bbox(item)[3] + 3
        lienzo.configure(scrollregion=(0, 0, self.ANCHO_COLUMNA_IZQ, lt["y"] + 8))
        if estaba_abajo:
            lienzo.yview_moveto(1.0)

    def _es_evento_nuevo(self, clave: str) -> bool:
        """True (y lo recuerda) la primera vez que se pregunta por `clave`."""
        if clave in self.eventos_ya_mostrados:
            return False
        self.eventos_ya_mostrados.add(clave)
        return True

    def _actualizar_eventos_incrementales(self):
        sim = self.sim
        if sim is None:
            return
        self._mostrar_eventos_burnout(sim)
        self._mostrar_evento_apogeo(sim.eventos)
        self._mostrar_evento_orbita(sim.eventos)
        self._mostrar_evento_mision(sim)
        self._mostrar_evento_soi_luna(sim.eventos)
        self._mostrar_evento_impacto_lunar(sim.eventos)
        self._mostrar_evento_impacto(sim.eventos)

    def _mostrar_eventos_burnout(self, sim: Simulacion):
        for i, etapa in enumerate(sim.etapas):
            if etapa.burnout_registrado and self._es_evento_nuevo(f"burnout_{i}"):
                datos = etapa.datos_burnout
                self._agregar_evento(f"BURNOUT ETAPA {i}", "titulo")
                self._agregar_evento(
                    f"  t={datos['t']:.1f}s  alt={datos['alt'] / 1000:.1f}km  "
                    f"Δv motor={datos['dv_prop']:.0f}m/s  "
                    f"Δrapidez={datos['delta_rapidez']:.0f}m/s\n"
                )

    def _mostrar_evento_apogeo(self, ev: RegistroEventos):
        if ev.apogeo_registrado and self._es_evento_nuevo("apogeo"):
            self._agregar_evento("APOGEO", "titulo")
            self._agregar_evento(
                f"  t={ev.datos_apogeo['t']:.1f}s  "
                f"alt={ev.datos_apogeo['alt'] / 1000:.1f}km\n"
            )

    def _mostrar_evento_orbita(self, ev: RegistroEventos):
        if ev.orbita_detectada and self._es_evento_nuevo("orbita"):
            datos = ev.datos_orbita
            self._agregar_evento("◈ ÓRBITA DETECTADA", "exito")
            self._agregar_evento(
                f"  e={datos['ecc']:.4f}  rp={datos['rp'] / 1000:.1f}km  "
                f"ra={datos['ra'] / 1000:.1f}km\n"
            )

    def _mostrar_evento_mision(self, sim: Simulacion):
        ev = sim.eventos
        if ev.mision_cumplida is None or not self._es_evento_nuevo("mision"):
            return
        modo_mayusculas = sim.config.modo_mision.upper()
        self._agregar_evento(
            f"◈ MISIÓN {modo_mayusculas} " + ("CUMPLIDA" if ev.mision_cumplida
                                             else "NO CUMPLIDA"),
            "exito" if ev.mision_cumplida else "fallo")
        self._agregar_evento(self._detalle_mision(sim))

    @staticmethod
    def _detalle_mision(sim: Simulacion) -> str:
        """Línea de detalle bajo el veredicto de la misión."""
        datos = sim.eventos.datos_mision
        if sim.config.modo_mision == "suborbital":
            return (f"  h_max={datos['alt_max'] / 1000:.1f}km  "
                    f"objetivo={datos['objetivo'] / 1000:.0f}km\n")
        if datos.get("deorbit"):
            if "reentrada_t" in datos:
                return (f"  rp={datos['rp_cutoff'] / 1000:.1f}km  Δv motor={datos['dv_prop']:.0f}m/s  "
                        f"reentrada v={datos['reentrada_v']:.0f}m/s\n")
            return f"  {datos.get('motivo', '')}\n"
        if "resultado" in datos and "ra_cutoff" in datos:
            motivo = f"  ({datos['motivo']})" if "motivo" in datos else ""
            return (f"  {datos['resultado']}  ra={datos['ra_cutoff'] / 1000:.0f}km  "
                    f"e={datos['ecc_cutoff']:.4f}{motivo}\n")
        if "rp" in datos:
            return (f"  rp={datos['rp'] / 1000:.0f}km  ra={datos['ra'] / 1000:.0f}km  "
                    f"e={datos['ecc']:.4f}\n")
        return f"  {datos.get('motivo', '')}\n"

    def _mostrar_evento_soi_luna(self, ev: RegistroEventos):
        if ev.entro_soi_luna and self._es_evento_nuevo("soi"):
            self._agregar_evento("🌕 ENTRADA A LA ESFERA DE INFLUENCIA LUNAR", "exito")
            self._agregar_evento(f"  t={ev.datos_soi_luna['t'] / 3600:.1f} h\n")

    def _mostrar_evento_impacto_lunar(self, ev: RegistroEventos):
        if ev.impacto_lunar_registrado and self._es_evento_nuevo("impacto_lunar"):
            d = ev.datos_impacto_lunar
            self._agregar_evento("💥 IMPACTO CON LA LUNA", "fallo")
            self._agregar_evento(f"  t={d['t'] / 3600:.3f} h  "
                                 f"v_rel={d['v_rel']:.0f} m/s{_nota_3d(d)}\n")

    def _mostrar_evento_impacto(self, ev: RegistroEventos):
        if ev.impacto_registrado and self._es_evento_nuevo("impacto"):
            self._agregar_evento("💥 IMPACTO", "fallo")
            self._agregar_evento(f"  t={ev.datos_impacto['t']:.1f}s\n")
            if "dist_base_km" in ev.datos_impacto:
                self._agregar_evento(
                    f"  a {ev.datos_impacto['dist_base_km']:.0f} km de la base")

    # ========================================================
    # ANIMACIÓN (un fotograma = una pasada por estos pasos)
    # ========================================================
    def _actualizar_animacion(self):
        sim = self.sim
        if sim is None:
            return
        est = sim.estado
        escala_log = sim.config.escala_log

        self._dibujar_trayectoria(sim)

        px_d, py_d = sim.posicion_display_eci()
        offset_x, offset_y = sim.offset_vista(escala_log)   # (0, 0) con vista Tierra
        for patch in self.patches_tierra:
            patch.center = (offset_x, offset_y)
        self._dibujar_grilla_tierra(
            np.pi / 2.0 - sim._Omega_tierra * est.t, offset_x, offset_y,
            sim.config.mostrar_grilla_tierra)
        pxv, pyv = (float(v) for v in vista_xy(px_d, py_d, escala_log))
        pxv, pyv = pxv + offset_x, pyv + offset_y
        self.punto_cohete.set_data([pxv], [pyv])
        self.punto_cohete.set_color("#ff8800" if est.empuje > 0 else "white")

        luna = self._dibujar_luna(sim, escala_log, offset_x, offset_y)
        self._dibujar_orbita_analitica(sim, escala_log, offset_x, offset_y)
        self._dibujar_puntos_lagrange(sim, luna, escala_log, offset_x, offset_y)

        if not self.camara_libre["activa"]:
            self._encuadrar_camara(sim, (px_d, py_d), (pxv, pyv), luna,
                                   (offset_x, offset_y))

        self._actualizar_inset(sim, (px_d, py_d), luna, (offset_x, offset_y))
        self.texto_hud.set_text(
            f"t={est.t:.1f}s\n"
            f"alt={_formatear_altitud(est.altitud)}\n"
            f"v={est.speed:.0f}m/s\n"
            f"fase={est.fase}"
        )
        self.canvas.draw_idle()
        if self._drawer_toca(sim):
            self._actualizar_drawer(sim)

    def _dibujar_trayectoria(self, sim: Simulacion):
        """Trayectoria (submuestreada: ver construir_segmentos_trayectoria)."""
        if len(sim.tray_wx) < 2:
            return
        if not hasattr(self, "_rgba_fase"):
            from matplotlib.colors import to_rgba
            self._rgba_fase = {fase: to_rgba(color)
                               for fase, color in self.COLORES_FASE.items()}
            self._rgba_fase["_otro"] = to_rgba("#b0b0b0")
        segmentos, colores = construir_segmentos_trayectoria(
            sim.tray_wx, sim.tray_wy, sim.tray_wfase,
            sim.tray_ex, sim.tray_ey,
            sim.config.escala_log,
            sim.config.marco_vista == "baricentro",
            sim._idx_inicio_orbita if sim._orbita_podada else None,
            sim._orbita_podada, self._rgba_fase)
        polis, col_polis = agrupar_en_polilineas(segmentos, colores)     # (12.13.0) pocas polilíneas
        self.lc_tray.set_segments(polis)
        self.lc_tray.set_color(col_polis)
        if self.GLOW_TRAYECTORIA:
            n = self.GLOW_MAX_SEGMENTOS      # solo la cola reciente: acota el coste por fotograma
            polis_g, col_g = agrupar_en_polilineas(segmentos[-n:], colores[-n:])
            self.lc_tray_glow.set_segments(polis_g)
            self.lc_tray_glow.set_color(col_g)

    def _dibujar_luna(self, sim: Simulacion, escala_log: bool,
                      offset_x: float, offset_y: float) -> dict:
        """Luna: posición real según la efeméride activa (pymcel,
        Horizons+Verlet o Kepler simplificado), siempre relativa a
        la Tierra — independiente de en qué marco esté la nave. Se
        dibuja siempre (con las flags apagadas la posición sale del
        Kepler simplificado). Devuelve posición real y de vista más el
        radio de vista."""
        radio_vista = vista_tam(R_LUNA, escala_log)
        x, y = sim._posicion_cuerpo("luna", sim.estado.t)
        xv, yv = (float(v) for v in vista_xy(x, y, escala_log))
        xv, yv = xv + offset_x, yv + offset_y
        self.patch_luna.set_radius(radio_vista)
        self.patch_luna.center = (xv, yv)
        return {"x": x, "y": y, "xv": xv, "yv": yv, "radio_vista": radio_vista}

    def _dibujar_orbita_analitica(self, sim: Simulacion, escala_log: bool,
                                  offset_x: float, offset_y: float):
        """Órbita analítica: solo tiene sentido en la zona "tierra"
        (elementos_orbitales asume foco en la Tierra)."""
        est = sim.estado
        elipse = None
        if est.marco == "tierra":
            orb = elementos_orbitales(est.px, est.py, est.vx, est.vy)
            elipse = trayectoria_orbital(orb)
        if elipse is None:
            self.linea_orbita.set_data([], [])
            self.punto_peri.set_data([], [])
            self.punto_apo.set_data([], [])
            return
        xs, ys, periapsis, apoapsis = elipse
        xs, ys = vista_xy(xs, ys, escala_log)
        periapsis = tuple(float(v) for v in vista_xy(periapsis[0], periapsis[1], escala_log))
        apoapsis = tuple(float(v) for v in vista_xy(apoapsis[0], apoapsis[1], escala_log))
        xs, ys = xs + offset_x, ys + offset_y
        periapsis = (periapsis[0] + offset_x, periapsis[1] + offset_y)
        apoapsis = (apoapsis[0] + offset_x, apoapsis[1] + offset_y)
        self.linea_orbita.set_data(xs, ys)
        self.punto_peri.set_data([periapsis[0]], [periapsis[1]])
        self.punto_apo.set_data([apoapsis[0]], [apoapsis[1]])

    def _borrar_puntos_lagrange(self):
        for artista in self.lagrange_artists.values():
            artista.remove()
        self.lagrange_artists.clear()

    def _dibujar_puntos_lagrange(self, sim: Simulacion, luna: dict,
                                 escala_log: bool, offset_x: float,
                                 offset_y: float):
        """Overlay de puntos de Lagrange (opcional): se rotan al
        ÁNGULO REAL de la línea Tierra-Luna en este instante (no un
        ángulo idealizado que arranque en 0 en t=0), para que
        queden alineados con la Luna real que se dibuja."""
        self._borrar_puntos_lagrange()
        if not sim.config.mostrar_crtbp:
            return
        luna_x, luna_y = luna["x"], luna["y"]
        theta = float(np.arctan2(luna_y, luna_x))
        # (x_bari, y_bari) están medidos desde el baricentro: pasar a ECI
        # sumando la posición del baricentro respecto a la Tierra.
        bary_x, bary_y = RAZON_MASA_TL * luna_x, RAZON_MASA_TL * luna_y
        for nombre, (x_bari, y_bari) in puntos_lagrange_crtbp().items():
            x_eci, y_eci = _rot2d((x_bari, y_bari), theta)
            x_eci, y_eci = (float(v) for v in
                            vista_xy(x_eci + bary_x, y_eci + bary_y, escala_log))
            x_eci, y_eci = x_eci + offset_x, y_eci + offset_y
            (marcador,) = self.ax.plot([x_eci], [y_eci], "o", color="#ffe040",
                                       markersize=5, zorder=4)
            texto = self.ax.text(x_eci, y_eci, f" {nombre}", color="#ffe040",
                                 fontsize=7, family="monospace", zorder=4)
            self.lagrange_artists[nombre] = marcador
            self.lagrange_artists[nombre + "_txt"] = texto

    def _encuadrar_camara(self, sim: Simulacion, pos_eci: Vec2, pos_vista: Vec2,
                          luna: dict, offset: Vec2):
        """Cámara automática (F la reactiva; rueda/arrastre la liberan).
         - Escala log: HÍBRIDA. Cerca de la Tierra sigue a la nave;
           al alejarse se abre, con mezcla suave, a la vista log
           Tierra+Luna+nave centrada en la Tierra.
         - Escala lineal: igual que antes (siguiendo a la nave si no
           hay terceros cuerpos/marco móvil; encuadre adaptativo si sí)."""
        px_d, py_d = pos_eci
        pxv, pyv = pos_vista
        offset_x, offset_y = offset
        luna_x, luna_y = luna["x"], luna["y"]
        distancia = float(np.hypot(px_d, py_d))
        usa_efemerides = (sim.config.usar_terceros_cuerpos
                          or sim.config.usar_marco_movil)
        if sim.config.escala_log:
            centro_x, centro_y, zoom = self._encuadre_escala_log(
                distancia, pos_vista, luna, offset)
        elif not usa_efemerides:
            zoom = max(distancia * 1.4, R_T * 1.3)
            centro_x, centro_y = px_d + offset_x, py_d + offset_y
        else:
            d_luna = float(np.hypot(luna_x - px_d, luna_y - py_d))
            if distancia < 5e6 and d_luna < 1e6:
                zoom = max(distancia * 1.4, R_T * 1.3)
                centro_x, centro_y = px_d + offset_x, py_d + offset_y
            elif distancia < 5e6:
                zoom = max(distancia * 2.0, R_T * 4)
                centro_x, centro_y = offset_x, offset_y
            else:
                zoom = max(d_luna * 0.7, 5e8)
                centro_x = 0.5 * luna_x + offset_x
                centro_y = 0.5 * luna_y + offset_y
        self.ax.set_xlim(centro_x - zoom, centro_x + zoom)
        self.ax.set_ylim(centro_y - zoom, centro_y + zoom)

    def _encuadre_escala_log(self, distancia: float, pos_vista: Vec2,
                             luna: dict, offset: Vec2):
        pxv, pyv = pos_vista
        offset_x, offset_y = offset
        dist_vista = float(np.hypot(pxv - offset_x, pyv - offset_y))  # desde la Tierra
        zoom_nave = max(dist_vista * 1.4, R_T * 1.3)
        extension = max(float(np.hypot(pxv, pyv)),
                        float(np.hypot(offset_x, offset_y)) + R_T)
        extension = max(extension,
                        float(np.hypot(luna["xv"], luna["yv"]))
                        + luna["radio_vista"])
        zoom_log = extension * 1.25
        mezcla = float(np.clip((distancia - self.CAM_R_MIN * R_T)
                               / ((self.CAM_R_MAX - self.CAM_R_MIN) * R_T),
                               0.0, 1.0))
        mezcla = mezcla * mezcla * (3.0 - 2.0 * mezcla)  # smoothstep: sin saltos
        centro_x = (1.0 - mezcla) * pxv
        centro_y = (1.0 - mezcla) * pyv
        zoom = (1.0 - mezcla) * zoom_nave + mezcla * zoom_log
        return centro_x, centro_y, zoom

    DRAWER_INTERVALO_S = 0.4

    def _drawer_toca(self, sim) -> bool:
        """(12.13.0) La gráfica de vuelo cambia despacio y cuesta ~28 ms por dibujo: se refresca como
        mucho cada DRAWER_INTERVALO_S; siempre al abrirla (`_t_drawer` a -inf) y al terminar."""
        if not self.drawer_estado["abierto"]:
            return False
        ahora = _monotonic()
        terminada = bool(getattr(getattr(sim, "estado", None), "simulacion_terminada", False))
        if not terminada and ahora - self._t_drawer < self.DRAWER_INTERVALO_S:
            return False
        self._t_drawer = ahora
        return True

    def _actualizar_drawer(self, sim: Simulacion):
        if not (self.drawer_estado["abierto"] and len(sim.hist_t) > 1):
            return
        n = len(sim.hist_t)
        res_t = getattr(sim, "hist_resumen_t", None)
        idx_res = getattr(sim, "_idx_resumen", None)
        if res_t and idx_res is not None and 0 <= idx_res <= n:
            # (12.11.1) ascenso + resumen de lo podado + última vuelta: línea continua
            paso = _paso_vista(n)
            paso_res = _paso_vista(len(res_t), GUI_MAX_PUNTOS // 2)   # el resumen casi nunca se recorta
            tiempos = _serie_con_resumen(sim.hist_t, res_t, idx_res, paso, paso_res)
            altitud_km = [a / 1000.0 for a in _serie_con_resumen(sim.hist_alt, sim.hist_resumen_alt, idx_res, paso, paso_res)]
            velocidad_kms = [v / 1000.0 for v in _serie_con_resumen(sim.hist_v, sim.hist_resumen_v, idx_res, paso, paso_res)]
            self.linea_alt.set_data(tiempos, altitud_km)
            self.linea_vel.set_data(tiempos, velocidad_kms)
            self.ax2.set_xlim(0, max(tiempos))
            self.ax2.set_ylim(0, max(altitud_km) * 1.1)
            self.ax2b.set_ylim(0, max(velocidad_kms) * 1.1)
            self.canvas2.draw_idle()
            return
        paso = _paso_vista(n)
        tiempos = _submuestrear(sim.hist_t, paso)
        altitud_km = [a / 1000.0 for a in _submuestrear(sim.hist_alt, paso)]
        velocidad_kms = [v / 1000.0 for v in _submuestrear(sim.hist_v, paso)]
        t_ds, a_ds, v_ds = tiempos, altitud_km, velocidad_kms     # para los límites (sin NaN)
        corte = getattr(sim, "_idx_inicio_orbita", None)
        if getattr(sim, "_orbita_podada", False) and corte is not None and 0 < corte < n:
            # (12.10.3) la poda borró el tramo orbital antiguo: no unir ascenso y última vuelta
            j = int(np.searchsorted(_indices_vista(n, paso), corte, side="left"))
            tiempos, altitud_km, velocidad_kms = (_serie_con_corte(x, j)
                                                  for x in (tiempos, altitud_km, velocidad_kms))
        self.linea_alt.set_data(tiempos, altitud_km)
        self.linea_vel.set_data(tiempos, velocidad_kms)
        tiempos, altitud_km, velocidad_kms = t_ds, a_ds, v_ds
        self.ax2.set_xlim(0, max(tiempos))
        self.ax2.set_ylim(0, max(altitud_km) * 1.1)
        self.ax2b.set_ylim(0, max(velocidad_kms) * 1.1)
        self.canvas2.draw_idle()


# ============================================================
# CAPA 6 — GRÁFICAS FINALES (PLOTLY) Y RESUMEN EN CONSOLA
# ============================================================
_SEPARADOR = "=" * 60


def _imprimir_encabezado(titulo: str) -> None:
    print(_SEPARADOR)
    print(titulo)
    print(_SEPARADOR)


def _nota_3d(imp: dict) -> str:
    """Aviso si el impacto 2D no es físico en 3D (Luna fuera del plano)."""
    if imp.get("fisico_3d") is False:
        return (f" ⚠ solo proyección 2D: la Luna está a z={imp['z_luna'] / 1000:.0f} km "
                f"del plano de la nave (no es físico en 3D)")
    return ""


def _veredicto(sim: Simulacion) -> str:
    return "CUMPLIDA" if sim.eventos.mision_cumplida else "NO CUMPLIDA"


def _anotacion_motivo(datos: dict) -> str:
    return f" ({datos['motivo']})" if "motivo" in datos else ""


# --- Figura 1: trayectoria ---
def _preparar_vista_trayectoria(sim: Simulacion, datos: dict) -> dict:
    """Trayectoria en km (con escala log/baricentro si aplica), fases,
    altitud real, desplazamiento de la Tierra y límite de ejes."""
    escala_log = sim.config.escala_log
    x_real = np.asarray(datos["tray_wx"], dtype=float)
    y_real = np.asarray(datos["tray_wy"], dtype=float)
    altitud_real_km = (np.hypot(x_real, y_real) - R_T) / 1000.0
    x_vista, y_vista = vista_xy(x_real, y_real, escala_log)
    if sim.config.marco_vista == "baricentro" and datos.get("tray_ex"):
        bary_x, bary_y = vista_xy(np.asarray(datos["tray_ex"], dtype=float),
                                  np.asarray(datos["tray_ey"], dtype=float),
                                  escala_log)
        x_vista, y_vista = x_vista - bary_x, y_vista - bary_y
    # Desplazamiento de la Tierra (y de lo que gira con ella) al
    # instante final, en km. (0, 0) con vista Tierra.
    offset_x_km, offset_y_km = (v / 1000.0 for v in sim.offset_vista(escala_log))
    x_km = x_vista / 1000.0
    y_km = y_vista / 1000.0

    radio_atmosfera_km = R_T / 1000.0 + 100.0
    distancia_max_km = max(
        float(np.max(np.abs(x_km))),
        float(np.max(np.abs(y_km))),
        radio_atmosfera_km + float(np.hypot(offset_x_km, offset_y_km)),
    )
    return {
        "x_km": x_km, "y_km": y_km,
        "fases": np.asarray(datos["tray_wfase"]),
        "altitud_real_km": altitud_real_km,
        "offset_x_km": offset_x_km, "offset_y_km": offset_y_km,
        "limite": distancia_max_km * 1.08,
    }


def _agregar_tierra_y_atmosfera(fig, offset_x_km: float, offset_y_km: float):
    import plotly.graph_objects as go
    radio_tierra_km = R_T / 1000.0
    radio_atmosfera_km = radio_tierra_km + 100.0
    theta = np.linspace(0.0, 2.0 * np.pi, 500)
    fig.add_trace(go.Scatter(
        x=offset_x_km + radio_tierra_km * np.cos(theta),
        y=offset_y_km + radio_tierra_km * np.sin(theta),
        mode="lines", name="Tierra",
        line=dict(width=3), fill="toself",
        hovertemplate="Tierra<br>X: %{x:.1f} km<br>Y: %{y:.1f} km<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=offset_x_km + radio_atmosfera_km * np.cos(theta),
        y=offset_y_km + radio_atmosfera_km * np.sin(theta),
        mode="lines", name="Atmósfera (100 km)",
        line=dict(width=1, dash="dash"),
        hovertemplate="Límite visual de atmósfera<br>Radio: %{r:.1f} km<extra></extra>",
    ))


def _agregar_luna_final(fig, sim: Simulacion, offset_x_km: float,
                        offset_y_km: float, limite: float) -> float:
    """Dibuja la Luna en su posición final y devuelve el límite de ejes
    ampliado si hace falta."""
    import plotly.graph_objects as go
    escala_log = sim.config.escala_log
    luna_x, luna_y = sim._posicion_cuerpo("luna", sim.estado.t)
    distancia_luna_km = float(np.hypot(luna_x, luna_y)) / 1000.0
    luna_xv, luna_yv = (float(v) for v in vista_xy(luna_x, luna_y, escala_log))
    luna_xv, luna_yv = luna_xv + offset_x_km * 1000.0, luna_yv + offset_y_km * 1000.0
    radio_luna_km = vista_tam(R_LUNA, escala_log) / 1000.0
    theta = np.linspace(0.0, 2.0 * np.pi, 100)
    fig.add_trace(go.Scatter(
        x=luna_xv / 1000.0 + radio_luna_km * np.cos(theta),
        y=luna_yv / 1000.0 + radio_luna_km * np.sin(theta),
        mode="lines", name="Luna (posición final)",
        line=dict(width=1), fill="toself", fillcolor="#4a4a4a",
        hovertemplate=(f"Luna<br>Distancia a la Tierra: "
                       f"{distancia_luna_km:.0f} km<extra></extra>"),
    ))
    limite = max(limite,
                 (float(np.hypot(luna_xv, luna_yv)) / 1000.0 + radio_luna_km) * 1.08)
    return limite


def _agregar_trayectoria_por_fase(fig, vista: dict) -> None:
    """Un trazo por tramo contiguo de fase."""
    import plotly.graph_objects as go
    fases = vista["fases"]
    inicio = 0
    for i in range(1, len(fases) + 1):
        hay_cambio = i == len(fases) or fases[i] != fases[i - 1]
        if not hay_cambio:
            continue
        xs = vista["x_km"][inicio:i]
        ys = vista["y_km"][inicio:i]
        fase = fases[inicio]
        if len(xs) >= 2:
            fig.add_trace(go.Scatter(
                x=xs, y=ys, mode="lines", name=str(fase),
                line=dict(width=2),
                customdata=vista["altitud_real_km"][inicio:i],
                hovertemplate=(f"Fase: {fase}<br>Altitud real: "
                               "%{customdata:.1f} km<extra></extra>"),
            ))
        inicio = i


def _agregar_orbita_analitica_final(fig, sim: Simulacion, offset_x_km: float,
                                    offset_y_km: float) -> None:
    """Órbita analítica final (con el estado al cierre)."""
    import plotly.graph_objects as go
    escala_log = sim.config.escala_log
    est = sim.estado
    orb = elementos_orbitales(est.px, est.py, est.vx, est.vy)
    elipse = trayectoria_orbital(orb, n_puntos=1000)
    if elipse is None:
        return
    offset_x, offset_y = offset_x_km * 1000.0, offset_y_km * 1000.0
    xs, ys, periapsis, apoapsis = elipse
    altitud_real_km = np.hypot(xs, ys) / 1000.0 - R_T / 1000.0
    xs, ys = vista_xy(xs, ys, escala_log)
    periapsis = tuple(float(v) for v in vista_xy(periapsis[0], periapsis[1], escala_log))
    apoapsis = tuple(float(v) for v in vista_xy(apoapsis[0], apoapsis[1], escala_log))
    xs, ys = xs + offset_x, ys + offset_y
    periapsis = (periapsis[0] + offset_x, periapsis[1] + offset_y)
    apoapsis = (apoapsis[0] + offset_x, apoapsis[1] + offset_y)
    fig.add_trace(go.Scatter(
        x=xs / 1000.0, y=ys / 1000.0, mode="lines",
        name="Órbita analítica",
        line=dict(width=2, dash="dot"),
        customdata=altitud_real_km,
        hovertemplate=("Órbita analítica<br>Altitud real: "
                       "%{customdata:.1f} km<extra></extra>"),
    ))
    if np.isfinite(orb["rp_radio"]):
        fig.add_trace(go.Scatter(
            x=[periapsis[0] / 1000.0], y=[periapsis[1] / 1000.0],
            mode="markers", name="Periapsis",
            marker=dict(size=9),
            hovertemplate=(f"Periapsis<br>Altitud: "
                           f"{orb['rp'] / 1000:.2f} km<extra></extra>"),
        ))
    if np.isfinite(orb["ra_radio"]):
        fig.add_trace(go.Scatter(
            x=[apoapsis[0] / 1000.0], y=[apoapsis[1] / 1000.0],
            mode="markers", name="Apoapsis",
            marker=dict(size=9),
            hovertemplate=(f"Apoapsis<br>Altitud: "
                           f"{orb['ra'] / 1000:.2f} km<extra></extra>"),
        ))


def _construir_figura_trayectoria(sim: Simulacion, datos: dict):
    import plotly.graph_objects as go
    escala_log = sim.config.escala_log
    vista = _preparar_vista_trayectoria(sim, datos)
    offset_x_km, offset_y_km = vista["offset_x_km"], vista["offset_y_km"]

    fig = go.Figure()
    _agregar_tierra_y_atmosfera(fig, offset_x_km, offset_y_km)
    limite = _agregar_luna_final(fig, sim, offset_x_km, offset_y_km,
                                 vista["limite"])
    _agregar_trayectoria_por_fase(fig, vista)
    _agregar_orbita_analitica_final(fig, sim, offset_x_km, offset_y_km)

    fig.update_layout(
        title=f"Trayectoria del cohete — {_etiqueta_modo(sim.config)}",
        xaxis=dict(title=("X (km, vista log radial)" if escala_log else "X (km)"),
                   range=[-limite, limite],
                   scaleanchor="y", scaleratio=1,
                   tickformat=".0f", gridcolor="gray"),
        yaxis=dict(title=("Y (km, vista log radial)" if escala_log else "Y (km)"),
                   range=[-limite, limite],
                   tickformat=".0f", gridcolor="gray"),
        width=850, height=850,
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="center", x=0.5),
        template="plotly_dark",
    )
    return fig


# --- Figura 2: altitud y velocidad ---
def _periodo_de_orbita_detectada(datos_orbita: dict) -> float:
    rp_radio = datos_orbita["rp"] + R_T
    ra_radio = datos_orbita["ra"] + R_T
    semieje = 0.5 * (rp_radio + ra_radio)
    return 2.0 * np.pi * np.sqrt(semieje ** 3 / MU_T)


def _recortar_series_a_una_vuelta(sim: Simulacion, datos: dict, t_total,
                                  alt_total, vel_total):
    """Con órbita detectada: ascenso + una vuelta completa. El snapshot ya
    viene cortado; el historial en vivo se corta aquí."""
    if not sim.eventos.orbita_detectada:
        return t_total, alt_total, vel_total
    if datos["t_orbita"] is not None:
        return t_total, alt_total, vel_total
    periodo = _periodo_de_orbita_detectada(sim.eventos.datos_orbita)
    t_final = sim.eventos.datos_orbita["t"] + periodo
    mascara = t_total <= t_final
    return t_total[mascara], alt_total[mascara], vel_total[mascara]


def _construir_figura_altitud_velocidad(sim: Simulacion, datos: dict,
                                        t_orbita: Optional[float]):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    t_total = np.asarray(datos["hist_t"], dtype=float)
    alt_total = np.asarray(datos["hist_alt"], dtype=float) / 1000.0
    vel_total = np.asarray(datos["hist_v"], dtype=float) / 1000.0
    t_g, alt_g, vel_g = _recortar_series_a_una_vuelta(
        sim, datos, t_total, alt_total, vel_total)

    fig = make_subplots(rows=1, cols=1, specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Scatter(x=t_g, y=alt_g, mode="lines", name="Altitud",
                   line=dict(width=2),
                   hovertemplate=("Tiempo: %{x:.1f} s"
                                  "<br>Altitud: %{y:.2f} km<extra></extra>")),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(x=t_g, y=vel_g, mode="lines", name="Velocidad",
                   line=dict(width=2),
                   hovertemplate=("Tiempo: %{x:.1f} s"
                                  "<br>Velocidad: %{y:.3f} km/s<extra></extra>")),
        secondary_y=True,
    )
    if t_orbita is not None:
        fig.add_vline(x=t_orbita, line_dash="dash",
                      annotation_text="Órbita detectada",
                      annotation_position="top")
    fig.update_layout(
        title=f"Evolución de la altitud y velocidad — {_etiqueta_modo(sim.config)}",
        xaxis_title="Tiempo (s)",
        width=1000, height=550,
        template="plotly_dark",
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="center", x=0.5),
    )
    fig.update_yaxes(title_text="Altitud (km)", secondary_y=False)
    fig.update_yaxes(title_text="Velocidad (km/s)", secondary_y=True)
    return fig


# --- Resúmenes por consola ---
def _imprimir_resumen_orbita(sim: Simulacion, modo_txt: str) -> None:
    datos = sim.eventos.datos_orbita
    semieje = 0.5 * ((datos["rp"] + R_T) + (datos["ra"] + R_T))
    periodo = _periodo_de_orbita_detectada(datos)
    _imprimir_encabezado(f"INFORMACIÓN DE LA ÓRBITA — {modo_txt}")
    print(f"Tiempo de inserción: {datos['t']:.2f} s")
    print(f"Periapsis: {datos['rp'] / 1000:.3f} km")
    print(f"Apoapsis: {datos['ra'] / 1000:.3f} km")
    print(f"Excentricidad: {datos['ecc']:.8f}")
    print(f"Semieje mayor: {semieje / 1000:.3f} km")
    print(f"Período orbital: {periodo:.2f} s")
    print(f"Período orbital: {periodo / 60:.2f} min")
    if sim._snapshot_orbita is not None:
        print("Nota: se usó el snapshot de la primera vuelta completa.")
    print(_SEPARADOR)


def _imprimir_resumen_deorbit(sim: Simulacion, modo_txt: str) -> None:
    d = sim.eventos.datos_mision
    _imprimir_encabezado(f"RESULTADO DESÓRBITA — {modo_txt}")
    print(f"Activación: t={d['t_activacion']:.0f} s, alt={d['alt_activacion'] / 1000:.0f} km, "
          f"órbita previa rp={d['rp_activacion'] / 1000:.0f} km / ra={d['ra_activacion'] / 1000:.0f} km")
    if "t_cutoff" in d:
        print(f"Corte (t={d['t_cutoff']:.0f} s): rp={d['rp_cutoff'] / 1000:.1f} km, "
              f"Δv motor≈{d['dv_prop']:.0f} m/s, Δrapidez {d['delta_rapidez']:.0f} m/s (objetivo rp <= {sim.config.DEORBIT_RP_OBJETIVO / 1000:.0f} km)")
    if "reentrada_t" in d:
        print(f"Reentrada confirmada: t={d['reentrada_t']:.0f} s, "
              f"v={d['reentrada_v']:.0f} m/s a {d['reentrada_alt'] / 1000:.0f} km")
    if sim.eventos.datos_impacto:
        print(f"Impacto: t={sim.eventos.datos_impacto['t']:.0f} s, "
              f"v={sim.eventos.datos_impacto['v']:.0f} m/s")
        if "dist_base_km" in sim.eventos.datos_impacto:
            print(f"Distancia a la base: "
                  f"{sim.eventos.datos_impacto['dist_base_km']:.1f} km")
    print(f"Misión: {_veredicto(sim)}{_anotacion_motivo(d)}")
    print(_SEPARADOR)


def _imprimir_resumen_alta_energia(sim: Simulacion) -> None:
    d = sim.eventos.datos_mision
    ev = sim.eventos
    efemerides = (sim.efemerides_efectiva() if sim._efemerides_modo != "sin_uso"
                  else "kepler_simplificado")
    _imprimir_encabezado(f"RESULTADO {sim.config.modo_mision.upper()} "
                         "(alta energía, sin phasing ni encuentro)")
    veredicto = (_veredicto(sim) if ev.mision_cumplida is not None
                 else "PENDIENTE (sin entrar aún a la SOI lunar)")
    if "t_cutoff" not in d:             # T_MAX (u otro fin) antes de completar el corte
        print(f"Misión: {veredicto}{_anotacion_motivo(d)}")
        print(_SEPARADOR)
        return
    print(f"Resultado: {d['resultado']}   |   Misión: {veredicto}"
          f"{_anotacion_motivo(d)}")
    ra_txt = ("inf" if not np.isfinite(d["ra_cutoff"])
              else f"{d['ra_cutoff'] / 1000:.0f} km")
    print(f"Corte (t={d['t_cutoff']:.0f} s): ra={ra_txt}, "
          f"ecc={d['ecc_cutoff']:.4f}, combustible restante={d['comb_restante']:.0f} kg")
    print(f"r máx (ECI): {ev.r_max_eci / 1000:.0f} km")
    if np.isfinite(ev.d_luna_min):
        print(f"Distancia mínima a la Luna: {ev.d_luna_min / 1000:.0f} km "
              f"(informativa; efemérides={efemerides})")
    if ev.entro_soi_luna:
        print(f"Entró a la esfera de influencia lunar en t={ev.datos_soi_luna['t'] / 3600:.1f} h")
    print(_SEPARADOR)


def _imprimir_resumen_lunar(sim: Simulacion) -> None:
    d = sim.eventos.datos_mision
    ev = sim.eventos
    if d.get("resultado") == "captura_lunar":
        _tit = "RESULTADO LUNAR (CAPTURA en órbita selenocéntrica; 2D, provisional)"
    elif "loi" in d or "t_encendido_loi" in d:
        _tit = "RESULTADO LUNAR (LOI intentada; sin captura verificada)"
    elif ev.impacto_lunar_registrado:
        _tit = "RESULTADO LUNAR (impacto tras el encuentro; sin LOI/captura)"
    elif ev.entro_soi_luna:
        _tit = "RESULTADO LUNAR (encuentro SOI: sobrevuelo, sin LOI/captura)"
    else:
        _tit = "RESULTADO LUNAR (sin encuentro)"
    _imprimir_encabezado(_tit)
    print(f"Efemérides: {sim.efemerides_efectiva()}  |  longitud "
          f"{sim.config.longitud_lanzamiento_ecliptica:g}°")
    ap = d.get("aparcamiento")
    if ap:
        print(f"Aparcamiento: rp={ap['rp'] / 1000:.1f} km, ra={ap['ra'] / 1000:.1f} km, "
              f"ecc={ap['ecc']:.5f}, seguro={'sí' if ap['seguro'] else 'no'}")
    pl = d.get("plan_tli")
    if pl:
        print(f"Plan TLI: encendido t={pl['t_encendido']:.1f} s, Δv≈{pl['dv_requerido']:.0f} m/s "
              f"(disp. {pl['dv_disponible']:.0f}); ajustado: {', '.join(pl['variables_ajustadas'])}; "
              f"predicho d_min={pl['d_min_pred'] / 1000:.0f} km, SOI t={pl['t_soi_pred'] / 3600:.2f} h")
    if "t_cutoff" in d:
        print(f"Corte TLI (t={d['t_cutoff']:.0f} s): ra={d['ra_cutoff'] / 1000:.0f} km, "
              f"combustible restante={d['comb_restante']:.0f} kg")
    if np.isfinite(ev.d_luna_min):
        print(f"Distancia mínima a la Luna: {ev.d_luna_min / 1000:.0f} km")
    if pl and ev.entro_soi_luna and not ev.impacto_lunar_registrado and np.isfinite(ev.d_luna_min):
        print(f"Paso real vs plan: real {ev.d_luna_min / 1000:.1f} km, previsto "
              f"{pl['d_min_pred'] / 1000:.1f} km (Δ={(ev.d_luna_min - pl['d_min_pred']) / 1000:+.1f} km)")
    if ev.entro_soi_luna:
        so = ev.datos_soi_luna
        print(f"SOI lunar: t={so['t'] / 3600:.3f} h, v_rel={so['v_rel']:.0f} m/s, "
              f"acercamiento previsto={so['d_min_dos_cuerpos'] / 1000:.0f} km"
              + (f", z_luna={so['z_luna'] / 1000:.0f} km" if so.get("z_luna") is not None else ""))
    lo = d.get("loi")
    if lo:
        print(f"LOI: encendido t={lo['t_ign']:.1f} s, corte t={lo['t_cut']:.1f} s, "
              f"Δv motor≈{lo['dv_prop']:.0f} m/s (requerido {lo['dv_requerido']:.0f}), "
              f"propelente {lo['comb_gastado']:.0f} kg (restan {lo['comb_restante']:.0f})")
        print(f"Órbita lunar: altitud {(lo['rp'] - R_LUNA) / 1000:.1f} x {(lo['ra'] - R_LUNA) / 1000:.1f} km, "
              f"ecc={lo['ecc']:.4f}, T={lo['periodo'] / 60:.1f} min"
              + (f"; verificada {lo['periodos_verificados']:.1f} periodos "
                 f"(rp mín {(lo['rp_min_verif'] - R_LUNA) / 1000:.1f} km)" if "periodos_verificados" in lo else ""))
    if ev.impacto_lunar_registrado:
        imp = ev.datos_impacto_lunar
        print(f"IMPACTO LUNAR: t={imp['t'] / 3600:.3f} h, "
              f"v_rel={imp['v_rel']:.1f} m/s, "
              f"distancia al centro={imp['d_centro'] / 1000:.1f} km{_nota_3d(imp)}")
    _det = ("captura verificada (modelo 2D)" if d.get("resultado") == "captura_lunar"
            else "encuentro SOI cumplido; termina en impacto" if (ev.impacto_lunar_registrado and not sim.config.LUNAR_LOI)
            else "" if sim.config.LUNAR_LOI
            else "encuentro SOI cumplido (sobrevuelo)" if ev.entro_soi_luna else "")
    print(f"Misión: {_veredicto(sim) if ev.mision_cumplida is not None else 'PENDIENTE'}"
          f"{' — ' + _det if _det else ''}{_anotacion_motivo(d)}")
    print(_SEPARADOR)


def _imprimir_resumen_gto(sim: Simulacion, modo_txt: str) -> None:
    d = sim.eventos.datos_mision
    _imprimir_encabezado(f"RESULTADO GTO — {modo_txt}")
    if "t_cutoff1" in d:
        print(f"Corte 1 (t={d['t_cutoff1']:.0f} s): ra={d['ra_cutoff1'] / 1000:.0f} km, "
              f"rp={d['rp_cutoff1'] / 1000:.0f} km, ecc={d['ecc_cutoff1']:.3f}")
    if "t_burn2" in d:
        print(f"Quema 2 (t={d['t_burn2']:.0f} s, alt={d['alt_burn2'] / 1000:.0f} km); "
              f"espera en coast: {(d['t_burn2'] - d['t_cutoff1']) / 3600:.2f} h")
    if "rp" in d:
        print(f"Órbita final: rp={d['rp'] / 1000:.0f} km, ra={d['ra'] / 1000:.0f} km, "
              f"ecc={d['ecc']:.4f}")
    print(f"Misión: {_veredicto(sim)}{_anotacion_motivo(d)}")
    print(_SEPARADOR)


def _imprimir_resumen_suborbital(sim: Simulacion, modo_txt: str) -> None:
    d = sim.eventos.datos_mision
    _imprimir_encabezado(f"RESULTADO SUBORBITAL — {modo_txt}")
    print(f"Altitud máxima: {d['alt_max'] / 1000:.3f} km")
    print(f"Objetivo: {d['objetivo'] / 1000:.1f} km")
    print(f"Cruzó la línea de Kármán: {'sí' if d['cruzo_karman'] else 'no'}")
    print(f"Misión: {_veredicto(sim)}")
    if sim.eventos.datos_apogeo:
        print(f"Tiempo al apogeo: {sim.eventos.datos_apogeo['t']:.1f} s")
    if sim.eventos.datos_impacto:
        print(f"Impacto: t = {sim.eventos.datos_impacto['t']:.1f} s, "
              f"v = {sim.eventos.datos_impacto['v']:.1f} m/s")
    print(_SEPARADOR)


def _imprimir_resumenes_de_mision(sim: Simulacion, modo_txt: str) -> None:
    ev = sim.eventos
    if ev.fallo_integrador:
        print(f"⚠ El integrador no convergió: {ev.fallo_integrador}")
    st = getattr(sim, "stats_eventos", None)
    if st and (st["corte_evento"] or st["corte_biseccion"] or st["suelo_evento"]):
        print(f"  Eventos del integrador: {st['corte_evento']} cortes por "
              f"evento, {st['corte_biseccion']} por bisección, "
              f"{st['suelo_evento']} impacto(s) por evento")
        if st["corte_biseccion"]:
            print("  (corte por bisección = respaldo: el evento no cumplió la "
                  "condición al final del paso)")
    if ev.orbita_detectada:
        _imprimir_resumen_orbita(sim, modo_txt)
    if sim.es_deorbit() and ev.datos_mision.get("deorbit"):
        _imprimir_resumen_deorbit(sim, modo_txt)
    if sim.es_alta_energia() and ev.datos_mision:
        _imprimir_resumen_alta_energia(sim)
    if sim.es_lunar() and ev.datos_mision:
        _imprimir_resumen_lunar(sim)
    if sim.config.modo_mision == "gto" and ev.mision_cumplida is not None:
        _imprimir_resumen_gto(sim, modo_txt)
    if sim.config.modo_mision == "suborbital" and ev.datos_mision:
        _imprimir_resumen_suborbital(sim, modo_txt)


def mostrar_graficas_finales(sim: Simulacion) -> None:
    """
    Genera las figuras finales de trayectoria y de vuelo.
    En modo suborbital, además, imprime al final el resultado de
    la misión (h_max, objetivo, Kármán, impacto).

    Usa el snapshot de la primera vuelta orbital si existe (para
    que la gráfica muestre ascenso + una órbita completa), o los
    historiales en vivo si la simulación terminó antes.
    """
    datos = sim.datos_para_graficas()
    if not datos["tray_wx"]:
        print("No hay datos suficientes para generar la trayectoria.")
        return
    try:
        import plotly  # noqa: F401
    except ImportError:
        print("⚠ plotly no está instalado; no se generan las gráficas finales.")
        return

    _construir_figura_trayectoria(sim, datos).show()

    if not datos["hist_t"]:
        return
    t_orbita = (sim.eventos.datos_orbita["t"]
                if sim.eventos.orbita_detectada else None)
    _construir_figura_altitud_velocidad(sim, datos, t_orbita).show()

    _imprimir_resumenes_de_mision(sim, _etiqueta_modo(sim.config))


# ============================================================
# PUNTO DE ENTRADA
# ============================================================
def _main_ventanas(argv: list) -> int:
    """Consola: `--ventanas ORIGEN DESTINO DESDE HASTA [límites]` (sin Tk)."""
    ap = argparse.ArgumentParser(prog="cohete --ventanas",
                                 description="Ventanas de lanzamiento interplanetarias")
    ap.add_argument("origen", choices=list(_CAT_VENTANAS))
    ap.add_argument("destino", choices=list(_CAT_VENTANAS))
    ap.add_argument("inicio", help="AAAA-MM-DD")
    ap.add_argument("fin", help="AAAA-MM-DD")
    ap.add_argument("--c3-max", type=float, default=math.inf, help="km²/s²")
    ap.add_argument("--dv-salida-max", type=float, default=math.inf, help="km/s")
    ap.add_argument("--vinf-llegada-max", type=float, default=math.inf, help="km/s")
    ap.add_argument("--tof-min", type=float, default=None, help="días (defecto 0.3×Hohmann)")
    ap.add_argument("--tof-max", type=float, default=None, help="días (defecto 1.5×Hohmann)")
    ap.add_argument("--criterio", choices=("dv_total", "c3"), default="dv_total")
    ap.add_argument("--sin-captura", action="store_true", help="sobrevuelo: sin Δv de captura")
    ap.add_argument("--paso", type=float, default=2.0, help="paso de la malla en días")
    ap.add_argument("--png", default=None, help="guarda el porkchop de la ventana --ventana")
    ap.add_argument("--ventana", type=int, default=1)
    a = ap.parse_args(argv)
    try:
        plan, vs = calcular_ventanas(a.origen, a.destino, a.inicio, a.fin, a.c3_max, a.dv_salida_max,
                                     a.vinf_llegada_max, a.criterio, not a.sin_captura, a.paso,
                                     tof_min_dias=a.tof_min, tof_max_dias=a.tof_max)
    except ValueError as e:
        print(f"⚠ {e}")
        return 2
    if not vs:
        print("Sin ventanas que cumplan las restricciones en ese intervalo.")
        return 1
    print(texto_tabla_ventanas(vs, not a.sin_captura))
    if a.png:
        if not 1 <= a.ventana <= len(vs):
            print(f"⚠ --ventana debe estar entre 1 y {len(vs)}")
            return 2
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        fig = figura_porkchop(plan, vs[a.ventana - 1])
        FigureCanvasAgg(fig)
        fig.savefig(a.png, dpi=130)
        print(f"porkchop de la ventana {a.ventana}: {a.png}")
    return 0


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--ventanas":
        raise SystemExit(_main_ventanas(sys.argv[2:]))
    if tk is None:
        print("⚠ tkinter no está disponible: la interfaz gráfica no puede abrirse "
              "(en consola solo funciona --ventanas).")
        raise SystemExit(1)
    root = tk.Tk()
    app = InterfazSimulacion(root)
    root.mainloop()
    app._cancelar_planificador()

    if app.sim is not None:
        mostrar_graficas_finales(app.sim)
        app.sim.cerrar_historial()          # (12.13.4) borra el temporal sin esperar al finalizador


if __name__ == "__main__":
    multiprocessing.freeze_support()    # Windows con ejecutables congelados
    main()