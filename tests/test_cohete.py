"""Suite estable de pruebas del simulador.

Ejecución (desde la raíz del repositorio):
    py -m pytest -q

Por defecto carga el motor en src/simulador/cohete.py. Para probar otro archivo,
define COHETE_ARCHIVO antes de iniciar pytest. Ejemplo PowerShell:
    $env:COHETE_ARCHIVO = "ruta/a/otro_cohete.py"
    py -m pytest -q

Las pruebas son focalizadas; no sustituyen una regresión de vuelos completos
(tests/regresion_cohete.py). Si faltan pymcel, astropy o spiceypy, usa sustitutos
limitados para que las pruebas aisladas no dependan de SPICE ni de la red. NumPy,
SciPy y Numba sí son necesarias para probar el núcleo numérico compilado; las
pruebas del proveedor REBOUND se omiten si `rebound` no está instalado.
"""
from __future__ import annotations

import importlib.util
import math
import os
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


def _buscar_motor() -> Path:
    especificado = os.environ.get("COHETE_ARCHIVO")
    if especificado:
        ruta = Path(especificado)
        return ruta if ruta.is_absolute() else Path.cwd() / ruta
    return Path(__file__).resolve().parents[1] / "src" / "simulador" / "cohete.py"


ARCHIVO = _buscar_motor()


def _instalar_sustitutos_si_faltan():
    """Permite correr estos tests aislados sin descargar efemérides ni SPICE."""
    try:
        import pymcel  # noqa: F401
        from pymcel.constantes import mu_earth, R_earth, mu_sun, mu_moon  # noqa: F401
    except ModuleNotFoundError:
        import types
        pymcel = types.ModuleType("pymcel")
        pymcel.consulta_spice = lambda *a, **k: (_ for _ in ()).throw(
            RuntimeError("SPICE no está disponible en la suite focalizada"))
        constantes = types.ModuleType("pymcel.constantes")
        constantes.mu_earth = 3.986004418e14
        constantes.R_earth = 6_378_137.0
        constantes.mu_sun = 1.32712440018e20
        constantes.mu_moon = 4.902800066e12
        pymcel.constantes = constantes
        sys.modules["pymcel"] = pymcel
        sys.modules["pymcel.constantes"] = constantes
    try:
        from astropy.time import Time  # noqa: F401
    except ModuleNotFoundError:
        import types
        astropy = types.ModuleType("astropy")
        atime = types.ModuleType("astropy.time")
        atime.Time = object
        astropy.time = atime
        sys.modules["astropy"] = astropy
        sys.modules["astropy.time"] = atime
    try:
        from spiceypy import SpiceyError  # noqa: F401
    except ModuleNotFoundError:
        import types
        spiceypy = types.ModuleType("spiceypy")
        spiceypy.SpiceyError = type("SpiceyError", (Exception,), {})
        sys.modules["spiceypy"] = spiceypy


@pytest.fixture(scope="module")
def mod():
    if not ARCHIVO.is_file():
        pytest.fail(f"No se encontró el motor: {ARCHIVO.name}")
    # Mostrar siempre qué archivo de código está bajo prueba.
    print(f"\n[test_cohete] Motor bajo prueba: {ARCHIVO.name}")
    _instalar_sustitutos_si_faltan()
    import matplotlib
    from unittest.mock import MagicMock

    # El motor usa TkAgg y Tkinter en la GUI; estas pruebas no crean ventana.
    sys.modules.setdefault("matplotlib.backends.backend_tkagg", MagicMock())
    sys.modules.setdefault("tkinter", MagicMock())
    use_original = matplotlib.use
    matplotlib.use = lambda *args, **kwargs: None
    try:
        nombre_modulo = "cohete_bajo_prueba"
        spec = importlib.util.spec_from_file_location(nombre_modulo, ARCHIVO)
        assert spec is not None and spec.loader is not None
        modulo = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = modulo
        spec.loader.exec_module(modulo)
    finally:
        matplotlib.use = use_original
    # (12.13.8) Que ninguna prueba intente descargar/cargar kernels reales; las que
    # prueban esa carga reponen la bandera con monkeypatch.
    modulo._SPICE_LISTO = True
    return modulo


def _sim_escape_descendiendo(mod):
    sim = mod.Simulacion.__new__(mod.Simulacion)
    sim.config = SimpleNamespace(
        modo_mision="escape",
        APOGEO_OBJETIVO=math.inf,
        suborbital_cutoff_por_apogeo=False,
    )
    # Trayectoria hiperbólica, con velocidad radial negativa (descenso).
    sim.estado = SimpleNamespace(
        px=1.0e7, py=0.0, vx=-12_000.0, vy=0.0,
        cutoff_por_apogeo=False, segunda_quema_realizada=False,
        fase="gravity_turn", t=50.0,
    )
    sim._lunar = False
    sim._deorbit_activo = False
    return sim


def test_escape_corta_por_energia_tambien_en_descenso(mod):
    sim = _sim_escape_descendiendo(mod)
    y = (sim.estado.px, sim.estado.py, sim.estado.vx, sim.estado.vy)
    orb = mod.elementos_orbitales(*y)
    assert orb["eps"] > 0.0
    assert sim._corte_por_estado(y) is True
    assert sim._margen_corte(y) > 0.0

    # La lógica real de misión debe aceptar el mismo criterio que el evento.
    etapa = SimpleNamespace(activa=True, m_comb_restante=123.0)
    sim.eventos = SimpleNamespace(mision_cumplida=None, datos_mision={})
    sim._fijar_fase = lambda fase: setattr(sim.estado, "fase", fase)
    ctx = SimpleNamespace(
        etapa=etapa, elementos=orb, subiendo=False,
        tiene_combustible=True, apogeo_alcanzado=False,
    )
    assert sim._gestionar_corte_alta_energia(ctx) is True
    assert sim.estado.cutoff_por_apogeo is True
    assert etapa.activa is False
    assert sim.eventos.datos_mision["resultado"] == "escape"


def test_k_quema_acepta_objetivo_de_energia_cero_con_residuo_pequeno(mod, monkeypatch):
    # Se aísla el refinamiento del corte: el integrador de paso devuelve una
    # energía final a 5e-7 J/kg de ε_obj=0. La tolerancia absoluta debe aceptarla.
    energia_residual = 5.0e-7
    mu = 1.0e6
    radio = 10_000.0
    velocidad = math.sqrt(2.0 * (mu / radio + energia_residual))

    def paso_falso(x, y, vx, vy, t, m, empuje, flujo, h, luna, sol,
                   t0, ht, mu_t, mu_l, mu_s, usa_sol):
        return radio, 0.0, 0.0, velocidad

    monkeypatch.setattr(mod, "_paso_quema", paso_falso)
    # Usamos py_func para probar la lógica Python con el paso instrumentado.
    funcion = getattr(mod._k_quema, "py_func", None)
    assert funcion is not None, "Se esperaba la función Python de la rutina Numba"
    resultado = funcion(
        radio, 0.0, 0.0, 0.0, 0.0, 1000.0,
        1000.0, 1.0, 0.0, 1.0, 0.0, 0.25, 5.0, 0.1,
        None, None, 0.0, 600.0, mu, 0.0, 0.0, False,
    )
    assert resultado[-1] == 0, f"El refinamiento no aceptó el corte: estado={resultado[-1]}"


def test_k_quema_compilado_acepta_objetivo_cero_con_residuo_pequeno(mod):
    """Ejecuta la ruta Numba real con el paso de quema real (sin monkeypatch)."""
    import numpy as np

    # Dos nodos lejos del cohete: el término de marea se anula con mu_l=0,
    # pero las posiciones siguen siendo válidas para la interpolación Hermite.
    tabla_lejana = np.array(
        [[1.0e12, 0.0, 0.0, 0.0], [1.0e12, 0.0, 0.0, 0.0]],
        dtype=np.float64,
    )
    mu = 1.0e6
    r0, vy0 = 10_000.0, 10.0
    eps0 = 0.5 * vy0 * vy0 - mu / r0
    assert eps0 < 0.0

    # Este escenario real cruza epsilon=0; el refinamiento compilado debe
    # terminar con residuo menor que el suelo absoluto de 1e-6 J/kg.
    resultado = mod._k_quema(
        r0, 0.0, 0.0, vy0, 0.0, 1.0e6,
        100.0, 100.0, 0.0, 1.0e5, 0.0, 10.0, 1.0, 0.05,
        tabla_lejana, tabla_lejana, 0.0, 100.0, mu, 0.0, 0.0, False,
    )
    assert resultado[-1] == 0, f"El núcleo Numba no aceptó el corte: estado={resultado[-1]}"
    x, y, vx, vy = resultado[:4]
    eps_final = 0.5 * (vx * vx + vy * vy) - mu / math.hypot(x, y)
    assert abs(eps_final) <= 1e-6, f"Residuo compilado excesivo: {eps_final:.3e} J/kg"
    assert math.isfinite(resultado[4]) and resultado[4] > 0.0


def test_k_quema_compilado_objetivo_no_finito_es_estado_1(mod):
    """(12.13.7) Un eps_obj NaN/inf no es "sin empuje neto" (3): es no convergencia (1)."""
    import numpy as np
    tabla = np.array([[1.0e12, 0.0, 0.0, 0.0], [1.0e12, 0.0, 0.0, 0.0]], dtype=np.float64)
    for malo in (float("nan"), float("inf")):
        res = mod._k_quema(
            10_000.0, 0.0, 0.0, 10.0, 0.0, 1.0e6,
            100.0, 100.0, 0.0, 1.0e5, malo, 10.0, 1.0, 0.05,
            tabla, tabla, 0.0, 100.0, 1.0e6, 0.0, 0.0, False,
        )
        assert res[-1] == 1, f"eps_obj={malo}: estado {res[-1]}"
        assert res[5] == 1.0e6                       # no consumió masa


def test_frontera_circularizacion_respeta_desfase_real(mod, monkeypatch):
    """En la frontera ecc == ecc_max, el margen con desfase real es negativo."""
    rp_objetivo = 200_000.0
    ecc_max = 0.01
    sim = mod.Simulacion.__new__(mod.Simulacion)
    sim.config = SimpleNamespace(
        modo_mision="tripulada", perfil_tripulada="bcb",
        APOGEO_OBJETIVO=1_300_000.0, suborbital_cutoff_por_apogeo=True,
    )
    sim._lunar = False
    sim._lunar_plan = None
    sim._deorbit_activo = False
    sim.estado = SimpleNamespace(
        px=mod.R_T + 300_000.0, py=0.0, vx=0.0, vy=7_000.0,
        fase="circularizacion", cutoff_por_apogeo=True,
        segunda_quema_realizada=True,
    )
    sim.rp_objetivo_circ = lambda: rp_objetivo
    sim.exige_ecc_en_corte = lambda: True
    sim.ecc_max_efectiva = lambda: ecc_max
    orbita_frontera = {
        "eps": -1.0, "ra": mod.R_T + 400_000.0,
        "rp": rp_objetivo, "ecc": ecc_max,
    }
    monkeypatch.setattr(mod, "elementos_orbitales", lambda *args: dict(orbita_frontera))
    y = (sim.estado.px, sim.estado.py, sim.estado.vx, sim.estado.vy)

    # Con desfase cero, el margen toca exactamente cero, pero la lógica de
    # misión exige ecc estrictamente menor: el caso ilustra la frontera.
    assert sim._corte_por_estado(y) is False
    assert sim._margen_corte(y, 0.0) >= 0.0
    # El solve_ivp real usa eps_corte=1e-6: ese margen ya queda del lado falso.
    assert sim._margen_corte(y, 1e-6) < 0.0
    assert (sim._margen_corte(y, 1e-6) >= 0.0) == sim._corte_por_estado(y)


def test_estimacion_loi_inalcanzable_devuelve_infinito(mod):
    objeto = SimpleNamespace(config=SimpleNamespace(THROTTLE_MIN=0.05))
    etapa = SimpleNamespace(u0=3000.0, D=1.0, m_comb_restante=1.0e-4)
    estimado = mod.Simulacion._duracion_quema_loi(objeto, 1000.0, 1000.0, etapa)
    assert math.isinf(estimado)


def test_estimacion_loi_con_combustible_suficiente_es_finita(mod):
    objeto = SimpleNamespace(config=SimpleNamespace(THROTTLE_MIN=0.05))
    etapa = SimpleNamespace(u0=3000.0, D=1.0, m_comb_restante=100.0)
    estimado = mod.Simulacion._duracion_quema_loi(objeto, 10.0, 1000.0, etapa)
    assert math.isfinite(estimado)
    assert estimado > 0.0


def test_loi_sin_combustible_falla_en_vez_de_perder_un_paso(mod):
    sim = mod.Simulacion.__new__(mod.Simulacion)
    sim.estado = SimpleNamespace(
        simulacion_terminada=False, pausado=False, empuje=10.0,
        dt=0.25, px=1.0, py=0.0, vx=0.0, vy=0.0, t=0.0,
    )
    sim._loi = {"eps_obj": -1.0}
    etapa = SimpleNamespace(tiene_combustible=lambda: False)
    sim.etapa_actual = lambda: etapa
    fallos = []

    def fallar(motivo):
        fallos.append(motivo)
        sim.estado.simulacion_terminada = True
        sim.estado.empuje = 0.0

    sim._lunar_fallar = fallar
    sim._paso_quema_loi_sim()
    assert sim.estado.simulacion_terminada is True
    assert sim.estado.empuje == 0.0
    assert fallos and "combustible" in fallos[0]


@pytest.mark.parametrize(
    "modo,perfil",
    [
        ("satelital", "bcb"),
        ("tripulada", "bcb"),
        ("tripulada", "directo"),
        ("suborbital", "bcb"),
        ("gto", "bcb"),
        ("tli", "bcb"),
        ("escape", "bcb"),
        ("deorbit", "bcb"),
        ("lunar", "bcb"),
    ],
)
def test_equivalencia_margen_y_logica_de_corte_en_10000_estados(mod, modo, perfil):
    """Prueba de propiedad determinista: el signo del margen refleja la lógica.

    Son estados sintéticos finitos para detectar incoherencias entre las dos
    rutas de corte; no constituyen una validación de vuelo ni de dinámica orbital.
    """
    import numpy as np

    # Semilla fija e independiente por combinación, reproducible entre ejecuciones.
    semillas = {
        ("satelital", "bcb"): 20261029,
        ("tripulada", "bcb"): 20261030,
        ("tripulada", "directo"): 20261031,
        ("suborbital", "bcb"): 20261032,
        ("gto", "bcb"): 20261033,
        ("tli", "bcb"): 20261034,
        ("escape", "bcb"): 20261035,
        ("deorbit", "bcb"): 20261036,
        ("lunar", "bcb"): 20261037,
    }
    semilla = semillas[(modo, perfil)]
    rng = np.random.default_rng(semilla)
    fallos = []
    fases = ("gravity_turn", "circularizacion", "quema_tli", "deorbit", "coasting")

    cfg = SimpleNamespace(
        modo_mision=modo,
        perfil_tripulada=perfil,
        APOGEO_OBJETIVO=float(1_300_000.0 if modo == "satelital" else rng.uniform(1e5, 4e8)),
        suborbital_cutoff_por_apogeo=True,
        DEORBIT_RP_OBJETIVO=40_000.0,
        ALTURA_SEGURIDAD=(0.0 if modo == "suborbital" else 200_000.0),
        GTO_RP_FINAL=35_000_000.0,
        ECC_MAX_GTO=0.02,
        ECC_MAX_APARCAMIENTO=0.05,
        ECC_MAX_TRIPULADA_BCB=0.01,
        ECC_MAX_TRIPULADA_DIRECTO=0.08,
    )
    sim = mod.Simulacion.__new__(mod.Simulacion)
    sim.config = cfg
    sim._lunar = (modo == "lunar")
    sim._lunar_plan = {"eps_obj": -1.0e6}
    sim._deorbit_activo = (modo == "deorbit")

    for i in range(10_000):
        # Muestrea una amplia gama de órbitas elípticas e hiperbólicas.
        r = float(rng.uniform(mod.R_T + 1.0, 2.0e8))
        theta_r = float(rng.uniform(-math.pi, math.pi))
        px, py = r * math.cos(theta_r), r * math.sin(theta_r)
        rapidez = float(rng.uniform(0.0, 50_000.0))
        theta_v = float(rng.uniform(-math.pi, math.pi))
        vx, vy = rapidez * math.cos(theta_v), rapidez * math.sin(theta_v)
        fase = fases[int(rng.integers(0, len(fases)))]
        sim.estado = SimpleNamespace(
            px=px, py=py, vx=vx, vy=vy,
            fase=fase,
            cutoff_por_apogeo=bool(rng.integers(0, 2)),
            segunda_quema_realizada=bool(rng.integers(0, 2)),
        )
        y = (px, py, vx, vy)
        margen_corta = sim._margen_corte(y) >= 0.0
        logica_corta = sim._corte_por_estado(y)
        if margen_corta != logica_corta:
            fallos.append((i, fase, y, sim._margen_corte(y), logica_corta))
            if len(fallos) >= 5:
                break

    assert not fallos, (
        f"Equivalencia rota en modo={modo}/{perfil}, semilla={semilla}; "
        f"primeros casos (índice, fase, estado, margen, lógica)={fallos}"
    )


def test_ajustar_dt_lunar_ignora_duracion_loi_infinita(mod):
    """Una LOI inalcanzable no debe fabricar un paso de 1 microsegundo."""
    sim = mod.Simulacion.__new__(mod.Simulacion)
    sim.config = SimpleNamespace(DT_COAST_LUNAR=30.0, LUNAR_LOI=True)
    sim._dt_base = 0.25
    sim._lunar_plan = None
    sim._lunar_d_luna = 1.0e8
    sim._loi = None
    sim.eventos = SimpleNamespace(entro_soi_luna=True)
    sim.estado = SimpleNamespace(fase="translunar", t=0.0, dt=0.25)
    sim._dist_objetivo_lunar = lambda: 1.0e7
    sim._loi_estimar = lambda: {
        "vr": -10.0,
        "tau": 100.0,
        "t_burn": float("inf"),
    }

    sim._ajustar_dt_lunar()

    assert sim.estado.dt == 30.0


# ---------------------------------------------------------------------------
# 12.13.4: guardas defensivas (la física no cambia)
# ---------------------------------------------------------------------------
def test_advance_to_rechaza_saltos_absurdos(mod):
    eph = mod._EphemerisVerlet(1.5e11, 0.0, 0.0, 29_780.0, 3.8e8, 0.0, 0.0, 1_022.0)
    for t_malo in (1.0e9, float("inf"), float("nan")):
        with pytest.raises(ValueError):
            eph.advance_to(t_malo)
    assert eph.t_actual == 0.0                      # el rechazo no integra nada
    eph.advance_to(100.0)                           # el uso normal sigue igual
    assert eph.t_actual == pytest.approx(100.0)


def test_validar_config_infinito_en_satelital_tiene_mensaje_propio(mod):
    cfg = mod.Config()
    cfg.modo_mision = "satelital"
    cfg.APOGEO_OBJETIVO = math.inf
    with pytest.raises(ValueError) as exc:
        cfg._validar_config()
    assert "finito" in str(exc.value)
    assert "APOGEO_OBJETIVO" in str(exc.value)


def test_plan_por_tramos_vence_por_timeout(mod):
    # (12.13.5) El timeout ya no lanza: cierra la misión como NO CUMPLIDA y libera la espera.
    import time
    sim = mod.Simulacion(mod.Config())
    sim.presupuesto_plan_s = 0.03
    sim.esperando_plan = True
    sim._plan_gen = iter([None, None])
    sim._plan_t0 = time.monotonic() - 2.0 * mod.Simulacion.PLAN_TIMEOUT_S
    assert sim._avanzar_plan_por_tramos() is True
    assert sim._plan_gen is None
    assert sim.esperando_plan is False
    assert sim.estado.simulacion_terminada is True
    assert sim.eventos.mision_cumplida is False
    assert sim.eventos.datos_mision["fallo_planificacion"] is True
    # un plan recién empezado sigue avanzando con normalidad
    sim2 = mod.Simulacion(mod.Config())
    sim2.presupuesto_plan_s = 0.03
    sim2._plan_gen = iter([None, None])
    sim2._plan_t0 = time.monotonic()
    assert sim2._avanzar_plan_por_tramos() in (True, False)
    assert sim2.estado.simulacion_terminada is False


def test_eph_cache_queda_acotada(mod, monkeypatch):
    lote = mod._EPH_NODOS_LOTE

    def lote_falso(cuerpo, fecha, i0):
        return {i: (1.0, 2.0, 3.0, 4.0) for i in range(i0, i0 + lote)}

    monkeypatch.setattr(mod, "_eph_obtener_lote", lote_falso)
    guardado = dict(mod._EPH_CACHE)
    mod._EPH_CACHE.clear()
    try:
        for dia in range(1, 29):
            for mes in (1, 2):
                mod._estado_pymcel("luna", f"2026-{mes:02d}-{dia:02d}T00:00:00", 100.0,
                                   fallos=set())
        assert len(mod._EPH_CACHE) <= mod._EPH_CACHE_MAX_CLAVES
    finally:
        mod._EPH_CACHE.clear()
        mod._EPH_CACHE.update(guardado)


def test_eph_cache_poda_solo_al_cargar_y_no_se_recarga_de_mas(mod, monkeypatch):
    # (12.13.7) Barrido por varios lotes: la tabla nunca pasa de 2 lotes y cada lote
    # se carga una sola vez (sin recargas en cada consulta).
    lote = mod._EPH_NODOS_LOTE
    cargas = []

    def lote_falso(cuerpo, fecha, i0):
        cargas.append(i0)
        return {i: (1.0, 2.0, 3.0, 4.0) for i in range(i0, i0 + lote)}

    monkeypatch.setattr(mod, "_eph_obtener_lote", lote_falso)
    guardado = dict(mod._EPH_CACHE)
    mod._EPH_CACHE.clear()
    fecha = "2026-03-03T00:00:00"
    try:
        n_lotes = 5
        for t in range(0, int(n_lotes * lote * mod._EPH_PASO), 3000):
            mod._estado_pymcel("luna", fecha, float(t), fallos=set())
            assert len(mod._EPH_CACHE[("luna", fecha)]) <= 2 * lote
        assert len(cargas) == len(set(cargas)) <= n_lotes + 1
    finally:
        mod._EPH_CACHE.clear()
        mod._EPH_CACHE.update(guardado)


@pytest.mark.skipif(not os.environ.get("COHETE_LENTOS"),
                    reason="lento (~1 min); activar con COHETE_LENTOS=1")
def test_escape_completo_hasta_t_max(mod, monkeypatch):
    # El escape cruza a crucero y termina por T_MAX_ALTA_ENERGIA con la misión cumplida.
    cfg = mod.Config()
    cfg.modo_mision = "escape"
    cfg.fecha_lanzamiento = "2026-01-01T00:00:00"
    cfg.aplicar_preset()
    def _sin_red(*a, **k):
        raise RuntimeError("sin red en la prueba")
    monkeypatch.setattr(mod, "_fetch_horizons_state", _sin_red)
    monkeypatch.setattr(mod, "_TERCEROS_CUERPOS_FUENTE", "sin_pymcel")
    monkeypatch.setattr(mod, "_EPH_DISCO", None, raising=False)
    sim = mod.Simulacion(cfg)
    n = 0
    while not sim.estado.simulacion_terminada and n < 250_000:
        sim.avanzar_paso()
        n += 1
    assert sim.estado.simulacion_terminada
    assert sim.estado.marco == "crucero"
    assert sim.eventos.mision_cumplida is True


def test_proveedor_horizons_verlet_no_guarda_tabla3d(mod):
    import inspect
    assert "self.tabla3d" not in inspect.getsource(mod._ProveedorHorizonsVerlet.__init__)


def test_limpiar_almacen_reintenta_el_borrado(mod, monkeypatch):
    llamadas = []

    def remove_falso(ruta):
        llamadas.append(ruta)
        if len(llamadas) == 1:
            raise PermissionError("archivo en uso (Windows)")

    monkeypatch.setattr(os, "remove", remove_falso)
    mod._limpiar_almacen({"f": None, "ruta": "temporal_falso.bin"})
    assert len(llamadas) == 2


def test_main_cierra_el_historial_al_salir(mod, monkeypatch):
    cerrado = []
    sim = SimpleNamespace(cerrar_historial=lambda: cerrado.append(True))
    app = SimpleNamespace(sim=sim, _cancelar_planificador=lambda: None)
    monkeypatch.setattr(mod, "tk", SimpleNamespace(Tk=lambda: SimpleNamespace(mainloop=lambda: None)))
    monkeypatch.setattr(mod, "InterfazSimulacion", lambda root: app)
    monkeypatch.setattr(mod, "mostrar_graficas_finales", lambda s: None)
    monkeypatch.setattr(sys, "argv", ["cohete"])
    mod.main()
    assert cerrado == [True]


def test_graficas_finales_sin_plotly_avisan_y_salen(mod, monkeypatch, capsys):
    sim = SimpleNamespace(datos_para_graficas=lambda: {"tray_wx": [1.0], "hist_t": [0.0]})

    def no_debe_llamarse(*a, **k):
        raise AssertionError("se intentó dibujar sin plotly")

    monkeypatch.setitem(sys.modules, "plotly", None)          # fuerza ImportError
    monkeypatch.setattr(mod, "_construir_figura_trayectoria", no_debe_llamarse)
    mod.mostrar_graficas_finales(sim)
    assert "plotly" in capsys.readouterr().out


def test_el_modulo_carga_sin_tkinter(tmp_path):
    """Sin Tk el motor se importa y la GUI avisa y termina (los demás tests simulan Tk)."""
    import subprocess
    codigo = (
        "import sys, importlib.util, os\n"
        "sys.modules['tkinter'] = None\n"
        "sys.argv = ['cohete']\n"
        f"spec = importlib.util.spec_from_file_location('cohete_sin_tk', r'{ARCHIVO}')\n"
        "m = importlib.util.module_from_spec(spec); sys.modules['cohete_sin_tk'] = m\n"
        "spec.loader.exec_module(m)\n"
        "assert m.tk is None and m.FigureCanvasTkAgg is None\n"
        "try:\n"
        "    m.main()\n"
        "except SystemExit as e:\n"
        "    assert e.code == 1\n"
        "else:\n"
        "    raise AssertionError('main debía terminar sin Tk')\n"
    )
    entorno = dict(os.environ, NUMBA_CACHE_DIR=str(tmp_path / "numba"))
    r = subprocess.run([sys.executable, "-c", codigo], capture_output=True, text=True, env=entorno)
    assert r.returncode == 0, r.stderr[-2000:]


# ---------------------------------------------------------------------------
# 12.13.8: carga de kernels SPICE y proveedor REBOUND
# ---------------------------------------------------------------------------
def _estados_kepler_3d(mod, fecha="2026-01-01T00:00:00", mu_luna=None):
    """Sol y Luna geocéntricos (respaldo Kepler, solo condiciones iniciales) como cuerpos de REBOUND."""
    import numpy as np
    t_ep = mod._epoca_a_segundos(fecha)
    (xs, ys), (vxs, vys) = mod._estado_kepler_simplificado("sol", t_ep)
    (xl, yl), (vxl, vyl) = mod._estado_kepler_simplificado("luna", t_ep)
    inc = math.radians(5.145)
    sol = np.array([xs, ys, 0.0, vxs, vys, 0.0])
    luna = np.array([xl, yl * math.cos(inc), yl * math.sin(inc), vxl, vyl * math.cos(inc),
                     vyl * math.sin(inc)])
    mu_l = mod.MU_LUNA if mu_luna is None else mu_luna
    return [(mod.MU_SOL, sol), (mod.MU_T, np.zeros(6)), (mu_l, luna)], sol, luna


def test_asegurar_kernels_spice_descarga_si_faltan_y_carga_una_vez(mod, monkeypatch, tmp_path):
    llamadas = []
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(mod, "_SPICE_LISTO", False)
    monkeypatch.setattr(mod, "_preparar_datos_pymcel", lambda: llamadas.append("preparar"))

    def descarga(verbose=False):
        llamadas.append("descarga")
        (tmp_path / "pymcel" / "data").mkdir(parents=True)
        (tmp_path / "pymcel" / "data" / "de430.bsp").write_text("x")
    monkeypatch.setattr(mod.pc, "descarga_kernels", descarga, raising=False)
    monkeypatch.setattr(mod.spy, "furnsh", lambda ruta: llamadas.append(("furnsh", str(ruta))))
    mod._asegurar_kernels_spice()
    mod._asegurar_kernels_spice()                      # idempotente
    assert llamadas.count("descarga") == 1
    assert [x for x in llamadas if isinstance(x, tuple)] == [
        ("furnsh", str(Path("pymcel") / "data" / "kernels.txt"))]
    assert mod._SPICE_LISTO is True


def test_asegurar_kernels_spice_no_descarga_si_ya_estan(mod, monkeypatch, tmp_path):
    (tmp_path / "pymcel" / "data").mkdir(parents=True)
    (tmp_path / "pymcel" / "data" / "de430.bsp").write_text("x")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(mod, "_SPICE_LISTO", False)
    monkeypatch.setattr(mod, "_preparar_datos_pymcel", lambda: None)
    monkeypatch.setattr(mod.pc, "descarga_kernels",
                        lambda **k: pytest.fail("no debía descargar"), raising=False)
    cargados = []
    monkeypatch.setattr(mod.spy, "furnsh", lambda ruta: cargados.append(ruta))
    mod._asegurar_kernels_spice()
    assert len(cargados) == 1


def test_eph_cargar_lote_carga_kernels_antes_de_consultar(mod, monkeypatch):
    import numpy as np
    orden = []
    monkeypatch.setattr(mod, "_asegurar_kernels_spice", lambda: orden.append("kernels"))

    def consulta(**k):
        orden.append("consulta")
        return np.zeros((mod._EPH_NODOS_LOTE, 6)), None, None
    monkeypatch.setattr(mod.pc, "consulta_spice", consulta)
    mod._eph_cargar_lote("luna", "2026-01-01T00:00:00", 0)
    assert orden == ["kernels", "consulta"]


def test_proveedor_rebound_es_la_fuente_lunar_por_defecto_y_se_valida(mod):
    cfg = mod.Config()
    assert cfg.efemerides_lunar == "rebound"
    assert "rebound" in cfg._EFEMERIDES_LUNAR_VALIDAS
    cfg.efemerides_lunar = "otra"
    cfg.modo_mision = "lunar"
    with pytest.raises(ValueError):
        cfg._validar_config()


def test_proveedor_rebound_luna_coincide_con_el_verlet_del_motor(mod):
    pytest.importorskip("rebound")
    import numpy as np
    cuerpos, sol, luna = _estados_kepler_3d(mod)
    prov = mod._ProveedorRebound("2026-01-01T00:00:00", 10 * 86400.0,
                                 estados=(cuerpos, 0, 1, 2, "prueba", False))
    n_sub, dt = 30, 20.0
    n_nodos = int(math.ceil(10 * 86400.0 / 600.0)) + 2
    tab = mod._tabla_verlet3d_validada(np.concatenate([sol, luna]), dt, n_sub, n_nodos,
                                       mod.MU_SOL + mod.MU_T, mod.MU_T + mod.MU_LUNA, mod.MU_SOL)
    peor = max(np.linalg.norm(prov.estado_3d("luna", k * 600.0)[0] - tab[k, 6:9])
               for k in range(0, n_nodos, 72))
    assert peor < 1.0e3, f"Luna: REBOUND vs Verlet {peor / 1e3:.3f} km"      # medido: 0,2-0,4 km a 10 d


def test_proveedor_rebound_sol_exacto_cuando_la_luna_no_mueve_a_la_tierra(mod):
    """Con la Luna sin masa el modelo del Verlet (Sol-Tierra a 2 cuerpos) es exacto: el resto de la
    diferencia con REBOUND (~1e4 km a 10 d) es el vaivén de la Tierra por la Luna."""
    pytest.importorskip("rebound")
    import numpy as np
    cuerpos, sol, luna = _estados_kepler_3d(mod, mu_luna=0.0)
    prov = mod._ProveedorRebound("2026-01-01T00:00:00", 10 * 86400.0,
                                 estados=(cuerpos, 0, 1, 2, "prueba", False))
    n_nodos = int(math.ceil(10 * 86400.0 / 600.0)) + 2
    tab = mod._tabla_verlet3d_validada(np.concatenate([sol, luna]), 20.0, 30, n_nodos,
                                       mod.MU_SOL + mod.MU_T, mod.MU_T, mod.MU_SOL)
    peor = max(np.linalg.norm(prov.estado_3d("sol", k * 600.0)[0] - tab[k, 0:3])
               for k in range(0, n_nodos, 72))
    assert peor < 50.0, f"Sol: {peor:.3f} m"


def test_proveedor_rebound_contrato_orden_libre_y_cuerpos(mod):
    pytest.importorskip("rebound")
    cuerpos, _, _ = _estados_kepler_3d(mod)
    prov = mod._ProveedorRebound("2026-01-01T00:00:00", 3 * 86400.0,
                                 estados=(cuerpos, 0, 1, 2, "prueba", False))
    a = prov("luna", 2.0e5)
    prov("sol", 10.0)
    assert prov("luna", 2.0e5) == a                    # determinista y en desorden
    with pytest.raises(mod.EfemeridesNoValidas):
        prov("marte", 0.0)
    with pytest.raises(mod.EfemeridesNoValidas):
        prov("luna", 4 * 86400.0)                      # fuera de la tabla
    assert "prueba" in prov.etiqueta and prov.con_planetas is False


def test_estados_iniciales_rebound_con_spice_incluye_sol_tierra_luna_y_planetas(mod, monkeypatch):
    import numpy as np
    monkeypatch.setattr(mod, "_asegurar_kernels_spice", lambda: None)
    pedidos = []

    def consulta(id, location, epochs):
        pedidos.append((id, location))
        v = np.arange(6, dtype=float) + float(len(pedidos))
        return v, None, None
    monkeypatch.setattr(mod.pc, "consulta_spice", consulta)
    cuerpos, i_s, i_t, i_l, origen, con_pl = mod._estados_iniciales_rebound("2026-01-01T00:00:00")
    assert [p[0] for p in pedidos[:3]] == ["10", "399", "301"]
    assert {p[0] for p in pedidos[3:]} == set(mod._GM_PLANETAS_BARICENTROS)
    assert all(p[1] == "@0" for p in pedidos)
    assert len(cuerpos) == 3 + len(mod._GM_PLANETAS_BARICENTROS)
    assert (i_s, i_t, i_l) == (0, 1, 2) and con_pl is True and "SPICE" in origen


def test_estados_iniciales_rebound_cae_a_horizons_sin_spice(mod, monkeypatch):
    def falla():
        raise OSError("sin red")
    monkeypatch.setattr(mod, "_asegurar_kernels_spice", falla)
    monkeypatch.setattr(mod, "_fetch_horizons_state_3d",
                        lambda cuerpo, fecha: (1.0e11, 0.0, 0.0, 0.0, 3.0e4, 0.0) if cuerpo == "sol"
                        else (3.8e8, 0.0, 0.0, 0.0, 1.0e3, 0.0))
    cuerpos, _, _, _, origen, con_pl = mod._estados_iniciales_rebound("2026-01-01T00:00:00")
    assert len(cuerpos) == 3 and con_pl is False and "sin planetas" in origen


def test_sin_rebound_el_proveedor_avisa_como_instalarlo(mod, monkeypatch):
    import builtins
    real = builtins.__import__

    def sin_rebound(nombre, *a, **k):
        if nombre == "rebound":
            raise ImportError("simulado")
        return real(nombre, *a, **k)
    monkeypatch.setattr(builtins, "__import__", sin_rebound)
    with pytest.raises(mod.EfemeridesNoValidas, match="pip install rebound"):
        mod._importar_rebound()


# --- Estados no finitos (12.13.9) -------------------------------------------------

def test_elementos_orbitales_rechaza_estado_no_finito(mod):
    nan, inf = float("nan"), float("inf")
    for estado in ((nan, 7e6, 0.0, 7500.0), (7e6, 0.0, inf, 7500.0), (7e6, 0.0, 0.0, -inf)):
        with pytest.raises(ValueError, match="no finito"):
            mod.elementos_orbitales(*estado)
    assert mod.elementos_orbitales(7.0e6, 0.0, 0.0, 7500.0)["ecc"] >= 0.0   # estado válido


def test_funciones_de_evento_toleran_estado_no_finito(mod):
    # solve_ivp puede probar subpasos no finitos: el evento debe decir "no hay corte", no lanzar.
    sim = mod.Simulacion(mod.Config())
    malo = [float("nan"), 7.0e6, 0.0, 7500.0]
    assert sim._corte_por_estado(malo) is False
    assert sim._margen_corte(malo) == -1.0


def test_estado_integrado_no_finito_detiene_la_simulacion(mod):
    sim = mod.Simulacion(mod.Config())
    est = sim.estado
    assert sim._estado_integrado_valido((1.0, 2.0, 3.0, 4.0), "marco tierra") is True
    assert est.simulacion_terminada is False
    assert sim._estado_integrado_valido((float("nan"), 2.0, 3.0, 4.0), "marco tierra") is False
    assert est.simulacion_terminada is True
    assert "no finito" in sim.eventos.fallo_integrador


def test_paso_con_integrador_que_devuelve_nan_no_propaga_el_nan(mod, monkeypatch, capsys):
    sim = mod.Simulacion(mod.Config())
    for _ in range(3):
        sim.avanzar_paso()
    antes = (sim.estado.px, sim.estado.py, sim.estado.vx, sim.estado.vy)
    monkeypatch.setattr(sim, "_avanzar_con_empuje",
                        lambda *a, **k: (float("nan"), 0.0, 0.0, 0.0), raising=False)
    monkeypatch.setattr(sim.config, "eventos", "paso", raising=False)
    t = sim.estado.t
    sim.avanzar_paso()
    est = sim.estado
    assert est.simulacion_terminada is True
    assert (est.px, est.py, est.vx, est.vy) == antes          # último estado válido
    assert est.t == t
    assert "no finito" in sim.eventos.fallo_integrador


def test_z_luna_solo_silencia_errores_de_datos(mod):
    sim = mod.Simulacion.__new__(mod.Simulacion)

    def proveedor(exc):
        def z(cuerpo, t):
            raise exc
        return SimpleNamespace(z=z)

    sim._proveedor_efemerides = proveedor(ValueError("fuera de tabla"))
    assert sim._z_luna(0.0) is None
    sim._proveedor_efemerides = proveedor(TypeError("bug de programación"))
    with pytest.raises(TypeError):
        sim._z_luna(0.0)


# --- Guardas completas y planificación que lanza (12.13.10) ------------------------

def test_funciones_de_evento_toleran_posicion_infinita_con_velocidad_finita(mod):
    sim = mod.Simulacion(mod.Config())
    for malo in ([float("inf"), 7.0e6, 0.0, 7500.0], [7.0e6, -float("inf"), 1.0, 1.0]):
        assert sim._corte_por_estado(malo) is False
        assert sim._margen_corte(malo) == -1.0


def test_integrar_tramo_rechaza_resultado_no_finito(mod, monkeypatch):
    sim = mod.Simulacion(mod.Config())
    nan = float("nan")
    monkeypatch.setattr(sim, "_en_coast_libre", lambda empuje: True)
    monkeypatch.setattr(sim, "_integrar_rk4", lambda *a, **k: (nan, 0.0, 0.0, 0.0))
    assert sim._integrar_tramo((7.0e6, 0.0, 0.0, 7500.0), 0.0, 1.0, (), 0.0) is None
    assert "no finito" in sim._msg_fallo_integrador
    ley = SimpleNamespace(fuerzas_congeladas=lambda y: ())
    monkeypatch.setattr(sim, "_integrar_rk4", lambda *a, **k: (nan, 0.0, 0.0, 0.0))
    assert sim._integrar_tramo_v12((7.0e6, 0.0, 0.0, 7500.0, 1.0, 0.0), 0.0, 1.0, ley, 0.0) is None


def test_plan_que_lanza_cierra_la_mision_en_vez_de_quedar_a_medias(mod):
    import time

    def plan():
        yield
        raise TypeError("bug en el planificador")

    sim = mod.Simulacion(mod.Config())
    sim.presupuesto_plan_s = 0.03
    sim.esperando_plan = True
    sim._plan_gen = plan()
    sim._plan_t0 = time.monotonic()
    assert sim._avanzar_plan_por_tramos() is True
    assert sim._plan_gen is None and sim.esperando_plan is False
    assert sim.estado.simulacion_terminada is True
    assert sim.eventos.mision_cumplida is False
    assert "TypeError" in sim.eventos.datos_mision["motivo"]


# --- 12.13.11: booleanos, planificador huérfano, almacén con disco caído ------------

@pytest.mark.parametrize("campo", ["suborbital_cutoff_por_apogeo", "empuje_vs_altitud",
                                   "usar_terceros_cuerpos", "usar_marco_movil", "mostrar_crtbp",
                                   "escala_log", "mostrar_grilla_tierra", "LUNAR_LOI",
                                   "cd_vs_mach", "rotacion_terrestre"])
def test_validar_config_exige_booleanos_de_verdad(mod, campo):
    for malo in ("false", 1, 0, None):
        cfg = mod.Config()
        setattr(cfg, campo, malo)
        with pytest.raises(ValueError, match=campo):
            cfg._validar_config()
    cfg = mod.Config()
    cfg._validar_config()                     # los valores por defecto siguen pasando


def test_error_de_fisica_cancela_el_planificador_externo(mod):
    sim = SimpleNamespace(
        estado=SimpleNamespace(simulacion_terminada=False, fase="gravity_turn"),
        config=SimpleNamespace(sim_speed=1, DEORBIT_VEL_MAX_PLAN=1, DEORBIT_PASO_QUEMA_MS=10,
                               LUNAR_PASOS_MIN_COAST=1),
        _lunar=False, _planificador_externo=None, esperando_plan=False,
    )

    def avanzar_paso():
        raise RuntimeError("fallo de física")

    sim.avanzar_paso = avanzar_paso
    llamadas = []
    ui = mod.InterfazSimulacion.__new__(mod.InterfazSimulacion)
    ui.sim = sim
    ui._cancelar_planificador = lambda: llamadas.append("cancelar")
    ui._finalizar_bucle_simulacion = lambda: llamadas.append("finalizar")
    ui._mostrar_error = lambda texto: llamadas.append("error")
    ui._pasos_restantes = None
    ui._paso_simulacion()
    assert llamadas == ["error", "cancelar", "finalizar"]


def test_almacen_con_disco_caido_no_promete_filas(mod, monkeypatch, capsys):
    import numpy as np
    alm = mod.AlmacenHistorial()
    serie = np.arange(5.0)
    alm.anexar(0, serie, serie, serie, serie, serie, serie, serie, ["gravity_turn"] * 5)
    assert alm.n == 5

    def falla(*a, **k):
        raise OSError("disco lleno")

    monkeypatch.setattr(mod.tempfile, "mkstemp", falla)
    alm._vaciar()
    assert alm.ok is False and alm.n == 0 and alm.segmentos == []
    assert alm.leer().shape[0] == 0
    alm.anexar(5, serie, serie, serie, serie, serie, serie, serie, ["gravity_turn"] * 5)
    assert alm.n == 0                          # descartado, sin crecer
    alm.cerrar()


def test_tablas_de_efemerides_del_plan_y_del_vuelo_comparten_nodos(mod):
    # (A6) Planificador (t0 alineado a 600 s) y vuelo (t0 = 0) evalúan la Luna en los mismos
    # instantes absolutos: los nodos coinciden bit a bit, no hay remuestreo distinto.
    sim = mod.Simulacion(mod.Config())
    vuelo = sim._tabla_efemerides_lunar(0.0, 40)
    plan = sim._tabla_efemerides_lunar(600.0 * 7, 20)
    assert (vuelo[0][7:27] == plan[0]).all()


def test_validar_config_cubre_todos_los_campos_bool(mod):
    import dataclasses
    campos = [f.name for f in dataclasses.fields(mod.Config) if str(f.type) == "bool"]
    assert len(campos) >= 10
    for nombre in campos:
        cfg = mod.Config()
        setattr(cfg, nombre, "false")
        with pytest.raises(ValueError, match=nombre):
            cfg._validar_config()


# --- Capa de cuerpos del mapa (12.14.0) ----------------------------------------------

def _jd_2026(mod, fecha="2026-01-01T00:00:00"):
    return mod._JD_2000_ENE_1 + mod._epoca_a_segundos(fecha) / 86400.0


def test_vista_lineal_por_defecto_y_capa_de_cuerpos_activa(mod):
    cfg = mod.Config()
    assert cfg.escala_log is False and cfg.mostrar_cuerpos is True


def test_posiciones_cuerpos_geo_son_las_reales(mod):
    pos = mod.posiciones_cuerpos_geo(_jd_2026(mod))
    assert set(pos) == {"sol", *[c for c, *_ in mod.PLANETAS_MAPA]}
    ua = mod._UA
    sx, sy = pos["sol"]
    assert 0.98 * ua < math.hypot(sx, sy) < 1.02 * ua                 # el Sol está a ~1 UA de la Tierra
    for clave, (a_min, a_max) in {"jupiter": (4.9, 5.5), "saturno": (9.0, 10.1), "neptuno": (29.5, 30.5),
                                  "venus": (0.70, 0.74)}.items():
        x, y = pos[clave]
        r_helio = math.hypot(x - sx, y - sy) / ua                     # geo(planeta) - geo(Sol) = heliocéntrico (2D)
        assert a_min < r_helio < a_max, (clave, r_helio)


def test_posiciones_cuerpos_geo_fuera_de_standish_devuelve_vacio(mod):
    assert mod.posiciones_cuerpos_geo(2300000.0) == {}               # antes de 1800
    assert mod.posiciones_cuerpos_geo(2500000.0) == {}               # después de 2050


def test_en_vista_y_seleccion_de_etiquetas(mod):
    assert mod.en_vista(0.0, 0.0, (-1, 1), (-1, 1)) is True
    assert mod.en_vista(2.0, 0.0, (-1, 1), (-1, 1)) is False
    assert mod.en_vista(1.05, 0.0, (-1, 1), (-1, 1), margen=0.05) is True
    items = [("a", 1, 0.0, 0.0), ("b", 0, 10.0, 0.0), ("c", 2, 100.0, 0.0)]
    # b (prioridad 0) gana a a (10 px, se tapan); c queda lejos
    assert mod.seleccionar_etiquetas(items, 40.0) == {"b", "c"}
    assert mod.seleccionar_etiquetas(items, 5.0) == {"a", "b", "c"}


def _gui_con_ejes_reales(mod):
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from unittest.mock import MagicMock
    h = object.__new__(mod.InterfazSimulacion)
    fig = Figure(figsize=(6, 6)); FigureCanvasAgg(fig)
    h.fig, h.ax = fig, fig.add_subplot(111)
    h.ax.set_aspect("equal")
    h.canvas, h.camara_libre = MagicMock(), {"activa": False}
    h._crear_artistas_tierra(); h._crear_artistas_trayectoria(); h._crear_artistas_luna_y_hud()
    return h


def _luna_vista(mod, sim, log):
    x, y = sim._posicion_cuerpo("luna", sim.estado.t)
    xv, yv = (float(v) for v in mod.vista_xy(x, y, log))
    return {"x": x, "y": y, "xv": xv, "yv": yv, "radio_vista": mod.vista_tam(mod.R_LUNA, log)}


def test_capa_de_cuerpos_culling_y_nivel_de_detalle_segun_el_zoom(mod):
    h = _gui_con_ejes_reales(mod)
    sim = mod.Simulacion(mod.Config())
    h._calcular_cuerpos(sim, False, (0.0, 0.0), _luna_vista(mod, sim, False))
    assert {c["clave"] for c in h._cuerpos_vista} == {"tierra", "luna", "sol",
                                                      *[c for c, *_ in mod.PLANETAS_MAPA]}

    def visibles(tipo):
        i = 0 if tipo == "marcador" else 1
        return {k for k, par in h.artistas_cuerpos.items() if par[i].get_visible()}

    # Cerca de la Tierra (±20 000 km): disco real grande, sin marcadores, nada más a la vista.
    h.ax.set_xlim(-2e7, 2e7); h.ax.set_ylim(-2e7, 2e7); h._pintar_cuerpos()
    assert visibles("marcador") == set() and visibles("etiqueta") == {"tierra"}
    # Sistema solar entero: Tierra y Luna son puntos (marcador) y los planetas existen.
    h._encuadrar_sistema_solar()
    assert "tierra" in visibles("marcador") and "neptuno" in visibles("marcador")
    assert "sol" in visibles("etiqueta") and "neptuno" in visibles("etiqueta")
    assert h.camara_libre["activa"] is True
    # Culling: a ±1e9 m solo quedan Tierra y Luna.
    h.ax.set_xlim(-1e9, 1e9); h.ax.set_ylim(-1e9, 1e9); h._pintar_cuerpos()
    assert visibles("etiqueta") <= {"tierra", "luna"} and "neptuno" not in visibles("marcador")


def test_capa_de_cuerpos_apagada_o_sin_simulacion_no_dibuja(mod):
    h = _gui_con_ejes_reales(mod)
    sim = mod.Simulacion(mod.Config())
    h._calcular_cuerpos(sim, False, (0.0, 0.0), _luna_vista(mod, sim, False))
    h._encuadrar_sistema_solar()
    sim.config.mostrar_cuerpos = False
    h._calcular_cuerpos(sim, False, (0.0, 0.0), _luna_vista(mod, sim, False))
    h._pintar_cuerpos()
    assert not any(a.get_visible() for par in h.artistas_cuerpos.values() for a in par)
    h._ocultar_cuerpos()          # idempotente


def test_capa_de_cuerpos_en_escala_log_los_planetas_caben_cerca(mod):
    h = _gui_con_ejes_reales(mod)
    sim = mod.Simulacion(mod.Config())
    h._calcular_cuerpos(sim, True, (0.0, 0.0), _luna_vista(mod, sim, True))
    r_max = max(math.hypot(c["x"], c["y"]) for c in h._cuerpos_vista)
    assert r_max < 16 * mod.R_T          # Neptuno comprimido a ~14 R_T


def test_longitud_ecliptica_del_sol_en_equinoccios_y_solsticios(mod):
    # Sanidad de marco y época: la longitud del Sol medida desde el equinoxio J2000 va ~0,36° por
    # debajo de la «de la fecha» por la precesión desde 2000 (de ahí 359,6° y no 0° en el equinoccio).
    for fecha, esperado in (("2026-03-20T15:00:00", 0.0), ("2026-06-21T08:00:00", 90.0),
                            ("2026-09-23T00:00:00", 180.0), ("2026-12-21T20:00:00", 270.0)):
        x, y = mod.posiciones_cuerpos_geo(_jd_2026(mod, fecha))["sol"]
        lon = math.degrees(math.atan2(y, x)) % 360.0
        dif = (lon - esperado + 180.0) % 360.0 - 180.0
        assert abs(dif) < 0.6, (fecha, lon)


def test_calcular_cuerpos_aplica_el_marco_de_lanzamiento_en_modo_lunar(mod):
    h = _gui_con_ejes_reales(mod)
    sim = SimpleNamespace(
        _lunar=True, _a_marco_lanzamiento=lambda x, y: (-x, y),
        config=SimpleNamespace(mostrar_cuerpos=True, fecha_lanzamiento="2026-01-01T00:00:00"),
        estado=SimpleNamespace(t=0.0))
    luna = {"x": 0.0, "y": 0.0, "xv": 0.0, "yv": 0.0, "radio_vista": 1.0}
    h._calcular_cuerpos(sim, False, (0.0, 0.0), luna)
    sol_vista = next(c for c in h._cuerpos_vista if c["clave"] == "sol")
    sx, sy = mod.posiciones_cuerpos_geo(_jd_2026(mod))["sol"]
    assert (sol_vista["x"], sol_vista["y"]) == pytest.approx((-sx, sy))
    sim._lunar = False
    h._cuerpos_cache = (None, {})
    h._calcular_cuerpos(sim, False, (0.0, 0.0), luna)
    sol_vista = next(c for c in h._cuerpos_vista if c["clave"] == "sol")
    assert (sol_vista["x"], sol_vista["y"]) == pytest.approx((sx, sy))
