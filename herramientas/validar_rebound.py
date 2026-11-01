"""Valida el Verlet 3D de cohete.py (Sol y Luna geocéntricos) con REBOUND,
usando SPICE (pymcel, sin red) como verdad.

    python herramientas/validar_rebound.py [ruta_motor] [fecha_iso] [dias]

Sin argumentos usa src/simulador/cohete.py. Necesita `pip install rebound`.

Compara cada 12 h, en posición 3D (km), contra SPICE:
  Verlet   : la tabla Verlet 3D del motor (Sol-Tierra 2 cuerpos, Luna con marea solar).
  RB 3c    : REBOUND IAS15 con Sol, Tierra y Luna (mismas masas, N-cuerpos completo).
  RB +pl   : REBOUND IAS15 con Sol, Tierra, Luna y los 7 baricentros planetarios.
Sin SPICE ni red (sin kernels) usa el respaldo Kepler del motor solo como condiciones
iniciales y compara Verlet contra REBOUND; con REB_LUNA_SIN_MASA=1 la Luna no mueve a la Tierra
(diagnóstico). El camino con SPICE no se pudo probar en el entorno donde se escribió.
Lectura: Verlet vs RB 3c aísla el error del integrador/modelo simplificado; RB +pl vs RB 3c
mide lo que aportan los planetas; el resto frente a SPICE es física que ninguno modela
(oblatitud, relatividad, figura de la Luna).
Los umbrales son los del motor: Luna 5 km, Sol 50 km."""
import importlib.util
import math
import sys
from datetime import datetime, timezone
from unittest.mock import MagicMock

import numpy as np

import re
from pathlib import Path

AQUI = Path(__file__).resolve().parents[1] / "src" / "simulador"


def _buscar_motor(arg):
    """Sin argumento: src/simulador/cohete.py. Con argumento: esa ruta (desde la carpeta actual)."""
    c = Path(arg) if arg else AQUI / "cohete.py"
    if not c.exists():
        sys.exit(f"No encuentro el motor: {c}")
    return str(c)


ruta = _buscar_motor(sys.argv[1] if len(sys.argv) > 1 else None)
fecha = sys.argv[2] if len(sys.argv) > 2 else "2026-01-01T00:00:00"
dias = float(sys.argv[3]) if len(sys.argv) > 3 else 10.0
print(f"Motor: {ruta}")

for nombre in ("tkinter", "matplotlib.backends.backend_tkagg"):
    sys.modules.setdefault(nombre, MagicMock())
import matplotlib  # noqa: E402
matplotlib.use = lambda *a, **k: None
spec = importlib.util.spec_from_file_location("cohete_val", ruta)
mod = importlib.util.module_from_spec(spec)
sys.modules["cohete_val"] = mod
spec.loader.exec_module(mod)

import pymcel as pc  # noqa: E402
import rebound  # noqa: E402
from astropy.time import Time  # noqa: E402

PASO_S = 43200.0
n_ep = int(round(dias * 86400.0 / PASO_S)) + 1
t_s = np.arange(n_ep) * PASO_S
t0 = datetime.fromisoformat(fecha).replace(tzinfo=timezone.utc)
jd_tdb = (Time(t0).tdb.jd + t_s / 86400.0)


def spice(id_, loc):
    d, _, _ = pc.consulta_spice(id=id_, location=loc, epochs=[float(j) for j in jd_tdb])
    a = np.asarray(d, dtype=float)
    return a.T if a.shape == (6, n_ep) else a


def _asegurar_kernels():
    """pymcel 0.9.25: `consulta_spice` solo descarga y carga (`furnsh`) los kernels si
    `pymcel/data/kernels.txt` NO existe; el motor lo crea antes (`_preparar_datos_pymcel`), así
    que en un proceso nuevo nunca se cargan y SPICE falla con MISSINGTIMEINFO. Aquí se hace a mano."""
    import glob
    import os
    import spiceypy as spy
    os.chdir(AQUI)                      # pymcel/data se busca relativo al directorio de trabajo
    mod._preparar_datos_pymcel()
    if not glob.glob("pymcel/data/*.bsp"):
        print("Descargando kernels SPICE (una sola vez, ~100 MB)...")
        pc.descarga_kernels(verbose=True)
    mod._preparar_datos_pymcel()
    spy.furnsh("pymcel/data/kernels.txt")


try:
    _asegurar_kernels()
    sol_g, luna_g = spice("10", "399"), spice("301", "399")
    bary = {k: spice(k, "@0") for k in ("10", "399", "301")}
    HAY_VERDAD = True
    print("Verdad: SPICE (pymcel).")
except Exception as e:                       # sin kernels ni red: solo comparación entre integradores
    HAY_VERDAD = False
    print(f"SPICE no disponible ({type(e).__name__}); se usa el respaldo Kepler del motor SOLO para "
          "las condiciones iniciales.\nSin verdad externa: se compara Verlet contra REBOUND.")
    t_ep = mod._epoca_a_segundos(fecha)
    inc = math.radians(5.145)                # inclinación de la órbita lunar respecto de la eclíptica
    (xs, ys), (vxs, vys) = mod._estado_kepler_simplificado("sol", t_ep)
    (xl, yl), (vxl, vyl) = mod._estado_kepler_simplificado("luna", t_ep)
    ci, si = math.cos(inc), math.sin(inc)
    sol_g = np.array([[xs, ys, 0.0, vxs, vys, 0.0]])
    luna_g = np.array([[xl, yl * 0 + yl * ci, yl * si, vxl, vyl * ci, vyl * si]])
    bary = {"10": sol_g, "399": np.zeros((1, 6)), "301": luna_g}

# --- Verlet 3D del motor ---
y0 = np.concatenate([sol_g[0], luna_g[0]])
n_sub, dt = 30, 20.0
n_nodos = int(math.ceil((dias + 1.0) * 86400.0 / 600.0)) + 2
tab = mod._tabla_verlet3d_validada(y0, dt, n_sub, n_nodos, mod.MU_SOL + mod.MU_T,
                                   mod.MU_T + mod.MU_LUNA, mod.MU_SOL)
k_nodo = (t_s / 600.0).astype(int)
verlet_sol, verlet_luna = tab[k_nodo, 0:6], tab[k_nodo, 6:12]

# --- REBOUND ---
if not HAY_VERDAD:   # solo hay una época: se propaga el Verlet y REBOUND desde ella
    pass
GM_PLANETAS = {  # baricentros de sistema (GM, m^3/s^2)
    "1": 2.2031868551e13, "2": 3.2485859200e14, "4": 4.2828375816e13,
    "5": 1.2671276480e17, "6": 3.7940584842e16, "7": 5.7945486e15, "8": 6.8365271e15,
}


def correr_rebound(con_planetas: bool):
    sim = rebound.Simulation()
    sim.G = 1.0                       # masas = GM [m^3/s^2]
    sim.integrator = "ias15"           # epsilon por defecto (1e-9)
    cuerpos = [("10", mod.MU_SOL), ("399", mod.MU_T), ("301", mod.MU_LUNA)]
    if con_planetas and HAY_VERDAD:
        cuerpos += [(k, v) for k, v in GM_PLANETAS.items()]
    import os
    for id_, mu in cuerpos:
        if id_ == "301" and os.environ.get("REB_LUNA_SIN_MASA"):
            mu = 0.0          # diagnóstico: la Luna no mueve a la Tierra
        b = bary[id_][0] if id_ in bary else spice(id_, "@0")[0]
        sim.add(m=mu, x=b[0], y=b[1], z=b[2], vx=b[3], vy=b[4], vz=b[5])
    sim.move_to_com()                 # solo traslada el origen; se comparan vectores relativos a la Tierra
    out = []
    for t in t_s:
        sim.integrate(float(t), exact_finish_time=1)
        p = sim.particles
        s = np.array([p[0].x, p[0].y, p[0].z, p[0].vx, p[0].vy, p[0].vz])
        e = np.array([p[1].x, p[1].y, p[1].z, p[1].vx, p[1].vy, p[1].vz])
        l = np.array([p[2].x, p[2].y, p[2].z, p[2].vx, p[2].vy, p[2].vz])
        out.append((s - e, l - e))
    return (np.array([o[0] for o in out]), np.array([o[1] for o in out]))


rb3_sol, rb3_luna = correr_rebound(False)
rbp_sol, rbp_luna = correr_rebound(True) if HAY_VERDAD else (rb3_sol, rb3_luna)


def err_km(a, ref):
    return np.linalg.norm(a[:, :3] - ref[:, :3], axis=1) / 1e3


def tabla(nombre, ref, v, r3, rp, umbral_km):
    if not HAY_VERDAD:
        d = err_km(v, r3)
        print(f"\n{nombre.upper()}: Verlet 3D vs REBOUND IAS15 (Sol-Tierra-Luna, mismas ICs y masas) [km]")
        print(f"{'t [d]':>6} {'|dr| km':>10} {'|dv| m/s':>10}")
        for k in range(n_ep):
            dv = np.linalg.norm(v[k, 3:] - r3[k, 3:])
            print(f"{t_s[k]/86400:6.1f} {d[k]:10.4f} {dv:10.5f}")
        for dd in (1, 2, 5, 10):
            k = int(round(dd * 86400.0 / PASO_S))
            if k < n_ep:
                print(f"  a {dd:>2} d: {d[k]:.4f} km")
        print(f"  máximo: {d.max():.4f} km   (umbral del motor para el error total: {umbral_km:g} km)")
        return d.max(), d.max(), d.max()
    print(f"\n{nombre.upper()}: error de posición 3D vs SPICE [km]  (umbral del motor: {umbral_km:g} km)")
    print(f"{'t [d]':>6} {'Verlet':>10} {'RB 3c':>10} {'RB +pl':>10}")
    ev, e3, ep = err_km(v, ref), err_km(r3, ref), err_km(rp, ref)
    for k in range(n_ep):
        print(f"{t_s[k]/86400:6.1f} {ev[k]:10.3f} {e3[k]:10.3f} {ep[k]:10.3f}")
    for d in (1, 2, 5, 10):
        k = int(round(d * 86400.0 / PASO_S))
        if k < n_ep:
            print(f"  a {d:>2} d: Verlet {ev[k]:8.3f}  RB3c {e3[k]:8.3f}  RB+pl {ep[k]:8.3f} km")
    print(f"  máximo: Verlet {ev.max():.3f}  RB 3c {e3.max():.3f}  RB +pl {ep.max():.3f} km")
    # Verlet frente al propio REBOUND 3c (integrador/modelo simplificado, sin SPICE)
    d = err_km(v, r3)
    print(f"  Verlet vs RB 3c (máx): {d.max():.3f} km")
    return ev.max(), e3.max(), ep.max()


mv_l, m3_l, mp_l = tabla("luna", luna_g, verlet_luna, rb3_luna, rbp_luna, 5.0)
mv_s, m3_s, mp_s = tabla("sol", sol_g, verlet_sol, rb3_sol, rbp_sol, 50.0)
print("\nResumen (máx, km):")
print(f"  Luna: Verlet {mv_l:.3f} | RB 3c {m3_l:.3f} | RB +planetas {mp_l:.3f}   umbral 5")
print(f"  Sol : Verlet {mv_s:.3f} | RB 3c {m3_s:.3f} | RB +planetas {mp_s:.3f}   umbral 50")
