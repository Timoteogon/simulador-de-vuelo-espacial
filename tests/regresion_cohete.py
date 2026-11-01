#!/usr/bin/env python3
"""
Prueba de regresión para cohete*.py (corre la física SIN ventana).

Uso:
    python tests/regresion_cohete.py src/simulador/cohete.py --guardar   # crea/actualiza la referencia
    python tests/regresion_cohete.py src/simulador/cohete.py             # compara contra la referencia

Versionado: el repositorio guarda una sola versión (la final) del motor, de este script y de
su referencia (regresion_ref.json). Las referencias regresion_ref_vX.Y.Z.json son de motores
históricos (11.21.2, 12.0.1, 12.2.0) y fijan que la física antigua se sigue reproduciendo.
Para comparar contra otra referencia se usa --ref.

La referencia queda en regresion_ref.json (junto a este script). Se compara
con tolerancia (rtol), no con hash: al refactorizar (p. ej. a vectores) el
orden de las sumas en coma flotante cambia en ~1e-16 aunque la física sea
la misma. Cuando un cambio es INTENCIONADO (p. ej. paso adaptativo), se
regenera la referencia a propósito con --guardar.

Tolerancia: --rtol (1e-9 por defecto). Medido con una perturbación de nivel
redondeo en las derivadas (4e-16 relativo), la desviación máxima en los
cuatro escenarios es 5e-12 .. 8e-11, y crece ~linealmente con la
perturbación (4e-14 -> 7e-10 .. 4e-9). Para refactors que reordenan
sumas, 1e-9 suele alcanzar; si hay falsos positivos, subir a 1e-7.

Determinismo: fecha de lanzamiento fija, Horizons desactivado y pymcel
ignorado (se fuerza el Kepler simplificado), para que el resultado no
dependa de la red ni de librerías opcionales.
"""
from __future__ import annotations

import argparse
import math
import os
import re
import importlib.util
import json
import sys
import time
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock

import numpy as np

REF = Path(__file__).with_name("regresion_ref.json")
MUESTREO_PASOS = 200          # una fila cada 200 pasos (50 s con dt=0.25)
MAX_PASOS = 40_000            # tope de seguridad (10 000 s simulados)
RTOL = 1e-9
ATOL = 1e-5     # (12.13.8) m; 2 µm de ruido en una coordenada ~0 (px≈405 m) entre hypot y sqrt

# Escenario que ejercita crucero (CRTBP) y la zona luna: la nave arranca ya
# fuera del umbral Tierra->crucero, sobre la línea Tierra-Luna y yendo hacia
# la Luna, sin empuje, con dt grande. No depende de la máquina de estados de
# la misión (en marcos != "tierra" no se ejecuta).
TRANSITO = dict(inicial=dict(radio=3.2e8, vel_hacia_luna=800.0, dt=30.0),
                pasos=6000, muestreo=100)

ESCENARIOS = {
    "satelital":          dict(modo_mision="satelital"),
    # Flags de 11.17.0 apagados (Cd constante, empuje sin pérdida por presión).
    # Desde 12.2.0 ya NO reproduce 11.16.0 bit a bit: cambió D de la 2ª etapa.
    "satelital_clasico":  dict(modo_mision="satelital", cd_vs_mach=False,
                               empuje_vs_altitud=False, _solo_v11=True),
    "tripulada_bcb":      dict(modo_mision="tripulada", perfil_tripulada="bcb"),
    "tripulada_directo":  dict(modo_mision="tripulada", perfil_tripulada="directo"),
    # Cubre integrador="rk45" (solve_ivp en todo el vuelo, sin RK4 de coast).
    "satelital_rk45":     dict(modo_mision="satelital", integrador="rk45"),
    "satelital_terceros": dict(modo_mision="satelital",
                               usar_terceros_cuerpos=True,
                               usar_marco_movil=True),
    "transito_lunar":     dict(modo_mision="satelital",
                               usar_terceros_cuerpos=True,
                               usar_marco_movil=True,
                               _transito=True),
    # Modos que solo existen desde cohete_v11.x (se omiten si el archivo
    # bajo prueba no los tiene). suborbital y deorbit terminan en impacto
    # (cubren `impacto_registrado`); tli/escape usan marco móvil.
    "suborbital":         dict(modo_mision="suborbital", _solo_v11=True),
    "gto":                dict(modo_mision="gto", _solo_v11=True),
    "tli":                dict(modo_mision="tli", _solo_v11=True),
    "escape":             dict(modo_mision="escape", _solo_v11=True),
    # deorbit usa el preset (deorbit_activacion="sitio" desde v12.2.0: la
    # quema se programa para caer sobre la base); deorbit_orbita es el
    # comportamiento anterior (quema apenas se detecta la órbita) y
    # deorbit_rot añade rotación terrestre (la base se mueve).
    # _max_pasos: con la 2ª etapa de 981 kN (12.2.0) el planificador espera
    # ~5400 s a que la base quede bajo la órbita y la reentrada cae después
    # de los 40 000 pasos por defecto.
    "deorbit":            dict(modo_mision="deorbit", _solo_v11=True,
                               _max_pasos=80000),
    "deorbit_orbita":     dict(modo_mision="deorbit", deorbit_activacion="orbita",
                               _solo_v11=True),
    "deorbit_rot":        dict(modo_mision="deorbit", rotacion_terrestre=True,
                               _solo_v11=True),
    # 12.2.0: dinámica "congelada" (= 11.21.x). Cada fila debe coincidir con
    # el escenario homónimo de regresion_ref_v11.21.2.json (verificar_legacy).
    "satelital_congelada":     dict(modo_mision="satelital", dinamica="congelada"),
    "tripulada_bcb_congelada": dict(modo_mision="tripulada", perfil_tripulada="bcb",
                                    dinamica="congelada"),
    "suborbital_congelada":    dict(modo_mision="suborbital", dinamica="congelada",
                                    _solo_v11=True),
    "gto_congelada":           dict(modo_mision="gto", dinamica="congelada",
                                    _solo_v11=True),
    # 12.2.0: eventos="paso" (bisección + Hermite) = 12.0.1 (verificar_legacy).
    "satelital_paso":     dict(modo_mision="satelital", eventos="paso"),
    "tripulada_bcb_paso": dict(modo_mision="tripulada", perfil_tripulada="bcb",
                               eventos="paso"),
    "suborbital_paso":    dict(modo_mision="suborbital", eventos="paso",
                               _solo_v11=True),
    "deorbit_paso":       dict(modo_mision="deorbit", eventos="paso",
                               _solo_v11=True, _max_pasos=80000),
    # 12.2.0: con marco móvil, eventos="paso" = transiciones al inicio del paso
    # (12.1.2).
    "satelital_terceros_paso": dict(modo_mision="satelital",
                                    usar_terceros_cuerpos=True,
                                    usar_marco_movil=True, eventos="paso"),
    "transito_lunar_paso": dict(modo_mision="satelital",
                                usar_terceros_cuerpos=True,
                                usar_marco_movil=True, eventos="paso",
                                _transito=True),
    "tli_paso":           dict(modo_mision="tli", eventos="paso", _solo_v11=True),
    "escape_paso":        dict(modo_mision="escape", eventos="paso", _solo_v11=True),
}
# nombre -> (archivo de referencia, escenario homónimo en él)
LEGACY_DE = {
    "satelital_congelada": ("regresion_ref_v11.21.2.json", "satelital"),
    "tripulada_bcb_congelada": ("regresion_ref_v11.21.2.json", "tripulada_bcb"),
    "suborbital_congelada": ("regresion_ref_v11.21.2.json", "suborbital"),
    "gto_congelada": ("regresion_ref_v11.21.2.json", "gto"),
    "satelital_paso": ("regresion_ref_v12.0.1.json", "satelital"),
    "tripulada_bcb_paso": ("regresion_ref_v12.0.1.json", "tripulada_bcb"),
    "suborbital_paso": ("regresion_ref_v12.0.1.json", "suborbital"),
    "deorbit_paso": ("regresion_ref_v12.0.1.json", "deorbit"),
    "satelital_terceros_paso": ("regresion_ref_v12.0.1.json", "satelital_terceros"),
    "transito_lunar_paso": ("regresion_ref_v12.0.1.json", "transito_lunar"),
    "tli_paso": ("regresion_ref_v12.0.1.json", "tli"),
    "escape_paso": ("regresion_ref_v12.0.1.json", "escape"),
}
# Escenarios en los que la caída debe quedar sobre la base (modo "sitio") y
# distancia máxima aceptada [km] (el planificador apunta a 5 km).
SITIO = {"deorbit": 10.0, "deorbit_rot": 10.0}


def cargar_modulo(ruta: str) -> ModuleType:
    """Importa el archivo de la simulación sin tkinter ni backend gráfico."""
    for nombre in ("tkinter", "matplotlib.backends.backend_tkagg"):
        if nombre not in sys.modules:
            sys.modules[nombre] = MagicMock()
    import matplotlib
    matplotlib.use = lambda *a, **k: None   # el archivo pide TkAgg
    spec = importlib.util.spec_from_file_location("cohete_bajo_prueba", ruta)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["cohete_bajo_prueba"] = mod
    spec.loader.exec_module(mod)
    mod._SPICE_LISTO = True      # (12.13.8) nada de descargar/cargar kernels reales en las pruebas
    return mod


def _elementos(mod, est) -> dict:
    """Llama a elementos_orbitales con la firma que tenga la versión
    bajo prueba: (px, py, vx, vy) o (r, v)."""
    import inspect
    if len(inspect.signature(mod.elementos_orbitales).parameters) == 2:
        return mod.elementos_orbitales(est.r, est.v)
    return mod.elementos_orbitales(est.px, est.py, est.vx, est.vy)


def _fila(mod, sim) -> dict:
    est = sim.estado
    dx, dy = (float(v) for v in sim.posicion_display_eci())
    fila = dict(t=est.t, alt=est.altitud, v=est.speed,
                px=est.px, py=est.py, fase=est.fase,
                marco=est.marco, dx=dx, dy=dy)
    if est.marco == "tierra":
        orb = _elementos(mod, est)
        for k in ("ecc", "rp", "ra"):
            val = orb.get(k)
            fila[k] = float(val) if val is not None and np.isfinite(val) else None
    return fila


def correr(mod, nombre: str, params: dict) -> dict:
    # Determinismo: sin red, sin pymcel, fecha fija.
    def _sin_red(*a, **k):
        raise RuntimeError("Horizons desactivado en la prueba de regresión")
    mod._fetch_horizons_state = _sin_red
    mod._TERCEROS_CUERPOS_FUENTE = "sin_pymcel"
    if hasattr(mod, "_EPH_DISCO"):
        mod._EPH_DISCO = None       # sin caché en disco (determinismo)

    params = dict(params)
    params.pop("_solo_v11", None)
    transito = params.pop("_transito", False)
    max_pasos_esc = params.pop("_max_pasos", None)
    cfg = mod.Config()
    # Primero modo/perfil y preset; después el resto de parámetros: en
    # versiones nuevas aplicar_preset() también fija usar_marco_movil
    # según el modo, y no debe pisar lo que pide el escenario.
    for k in ("modo_mision", "perfil_tripulada"):
        if k in params:
            setattr(cfg, k, params[k])
    cfg.fecha_lanzamiento = "2026-01-01T00:00:00"
    cfg.aplicar_preset()
    for k, val in params.items():
        if k not in ("modo_mision", "perfil_tripulada"):
            setattr(cfg, k, val)
    sim = mod.Simulacion(cfg)

    max_pasos, muestreo = max_pasos_esc or MAX_PASOS, MUESTREO_PASOS
    if transito:
        ini = TRANSITO["inicial"]
        est = sim.estado
        m = np.asarray(sim._posicion_cuerpo("luna", 0.0), dtype=float)
        u = m / np.hypot(m[0], m[1])
        est.px, est.py = (float(c) for c in ini["radio"] * u)
        est.vx, est.vy = (float(c) for c in ini["vel_hacia_luna"] * u)
        est.dt = ini["dt"]
        est.fase = "coasting"
        for et in sim.etapas:       # sin empuje
            et.activa = False
        max_pasos, muestreo = TRANSITO["pasos"], TRANSITO["muestreo"]

    filas = []
    paso = 0
    while not sim.estado.simulacion_terminada and paso < max_pasos:
        sim.avanzar_paso()
        paso += 1
        if paso % muestreo == 0:
            filas.append(_fila(mod, sim))
    filas.append(_fila(mod, sim))   # estado final

    ev = sim.eventos
    resumen = dict(
        pasos=paso,
        terminada=bool(sim.estado.simulacion_terminada),
        orbita_detectada=bool(ev.orbita_detectada),
        apogeo_registrado=bool(ev.apogeo_registrado),
        impacto_registrado=bool(ev.impacto_registrado),
        efemerides=sim._efemerides_modo,
        datos_orbita={k: float(v) for k, v in ev.datos_orbita.items()
                      if isinstance(v, (int, float, np.floating))},
    )
    return dict(resumen=resumen, filas=filas)


def verificar_arbol(mod) -> bool:
    """Comprobaciones estructurales del registro de cuerpos (no es física:
    no entra en la referencia). Se omite si el archivo no lo tiene."""
    if not hasattr(mod, "CUERPOS"):
        print("[arbol_cuerpos] omitido (el archivo no tiene registro de cuerpos)")
        return True
    errores = []

    def chk(cond, msg):
        if not cond:
            errores.append(msg)

    C = mod.CUERPOS
    # estructura: todo padre existe y solo la estrella no tiene padre
    for n, c in C.items():
        chk(c.nombre == n, f"{n}: nombre distinto de la clave")
        chk(c.padre is None or c.padre in C, f"{n}: padre inexistente")
        chk((c.padre is None) == (c.tipo == "estrella"),
            f"{n}: solo la estrella puede no tener padre")
    # valores heredados del código anterior
    chk(mod.R_SOI_LUNA == mod._A_LUNA_REF * (mod.MU_LUNA / mod.MU_T) ** 0.4,
        "R_SOI_LUNA distinto de la fórmula original")
    chk(mod.radio_soi("luna") == mod.R_SOI_LUNA, "radio_soi('luna') != R_SOI_LUNA")
    chk(C["sol"].mu == mod.MU_SOL and C["luna"].mu == mod.MU_LUNA
        and C["tierra"].mu == mod.MU_T, "mu del registro != constantes")
    dia = 86400.0
    chk(mod._ELEMENTOS_TERCEROS == {
        "sol":  (1.495978707e11, 0.0167, 365.256363 * dia),
        "luna": (3.844e8,        0.0549, 27.321661 * dia)},
        "elementos keplerianos distintos de los originales")
    chk((C["sol"].horizons_id, C["luna"].horizons_id) == ("10", "301"),
        "ids de Horizons distintos")
    chk((C["sol"].pymcel_nombre, C["luna"].pymcel_nombre) == ("sun", "moon"),
        "nombres de pymcel distintos")
    # radio de Hill y zonas lunares (v11.14.1+; se omite si no existen)
    if hasattr(mod, "radio_hill"):
        hill, soi = mod.radio_hill("luna"), mod.radio_soi("luna")
        chk(hill == mod.R_HILL_LUNA, "R_HILL_LUNA != radio_hill('luna')")
        chk(6.15e7 < hill < 6.16e7, f"Hill lunar fuera de rango: {hill}")
        chk(hill < soi, "Hill debe ser menor que SOI para Tierra-Luna")
        chk(abs(soi / hill - 3 ** (1 / 3) * (mod.MU_LUNA / mod.MU_T) ** (1 / 15))
            < 1e-12, "SOI/Hill != 3^(1/3)*(m/M)^(1/15)")
        chk(mod.zona_lunar(0.5 * hill) == "estable", "zona estable")
        chk(mod.zona_lunar(0.5 * (hill + soi)) == "transito", "zona transito")
        chk(mod.zona_lunar(1.01 * soi) == "tierra", "zona tierra")
    # regla de perturbadores por SOI (tabla acordada)
    planetas = ("mercurio", "venus", "tierra", "marte", "jupiter",
                "saturno", "urano", "neptuno")
    if "jupiter" in C:      # v11.14.1+: planetas en el registro
        chk(mod.hijos("sol") == planetas, "hijos('sol') != los 8 planetas")
        chk(mod.perturbadores_por_soi("sol") == planetas, "sol -> 8 planetas")
        for n in planetas:
            chk(C[n].kepler_geo is None,
                f"{n}: kepler_geo debe ser None")
            chk(C[n].horizons_id is not None and C[n].pymcel_nombre is not None,
                f"{n}: faltan ids de efemérides")
        # SOI/Hill reales [Gm] (JPL GM, semiejes medios)
        for n, soi_gm, hill_gm in (("jupiter", 48.2, 53.1),
                                   ("saturno", 54.5, 65.2),
                                   ("tierra", 0.925, 1.50)):
            chk(abs(mod.radio_soi(n) / 1e9 - soi_gm) < 0.02 * soi_gm,
                f"radio_soi({n}) fuera de rango")
            chk(abs(mod.radio_hill(n) / 1e9 - hill_gm) < 0.02 * hill_gm,
                f"radio_hill({n}) fuera de rango")
        # el fallback Kepler no debe inventar posiciones de planetas
        try:
            mod._estado_kepler_simplificado("marte", 0.0)
            errores.append("kepler de 'marte' no falló")
        except KeyError:
            pass
        try:
            mod._estado_kepler_simplificado("tierra", 0.0)
            errores.append("kepler de 'tierra' no falló")
        except KeyError:
            pass
    else:
        chk(mod.perturbadores_por_soi("sol") == ("tierra",), "sol")
    chk(mod.perturbadores_por_soi("tierra") == ("luna",), "tierra")
    chk(mod.perturbadores_por_soi("luna") == ("tierra",), "luna")
    # árbol sintético con un planeta de varias lunas
    Cu = mod.Cuerpo
    T = {"sol": Cu("sol", "estrella", 1e20),
         "p": Cu("p", "planeta", 1e15, padre="sol", a_ref=1e11),
         "a": Cu("a", "luna", 1e10, padre="p", a_ref=1e8),
         "b": Cu("b", "luna", 1e10, padre="p", a_ref=2e8),
         "c": Cu("c", "luna", 1e10, padre="p", a_ref=3e8),
         "a1": Cu("a1", "luna", 1e5, padre="a", a_ref=1e6)}
    chk(mod.perturbadores_por_soi("p", T) == ("a", "b", "c"), "planeta -> sus lunas")
    chk(mod.perturbadores_por_soi("b", T) == ("p", "a", "c"),
        "luna -> planeta + hermanas")
    chk(mod.perturbadores_por_soi("a", T) == ("p", "b", "c", "a1"),
        "luna -> planeta + hermanas + sus lunas")
    chk(mod.perturbadores_por_soi("sol", T) == ("p",), "estrella -> planetas")
    chk(abs(mod.radio_soi("b", T) - 2e8 * (1e10 / 1e15) ** 0.4) == 0.0,
        "radio_soi sintético")
    # Horizons+Verlet no debe devolver la Luna para un cuerpo que no sea
    # Sol/Luna (v11.14.1+; se omite si no existe el helper)
    if hasattr(mod, "_exigir_sol_o_luna"):
        class _V:
            def advance_to(self, t): pass
            def pos_sun(self): return (1.0, 2.0)
            def pos_moon(self): return (3.0, 4.0)
            def vel_sun(self): return (0.1, 0.2)
            def vel_moon(self): return (0.3, 0.4)
        cfg = mod.Config()
        cfg.fecha_lanzamiento = "2026-01-01T00:00:00"
        sim = mod.Simulacion(cfg)
        sim._efemerides_modo, sim._efemerides_verlet = "horizons_verlet", _V()
        chk(sim._posicion_cuerpo("sol", 0.0) == (1.0, 2.0), "verlet sol")
        chk(sim._posicion_cuerpo("luna", 0.0) == (3.0, 4.0), "verlet luna")
        chk(sim._estado_cuerpo("luna", 0.0)[1] == (0.3, 0.4), "verlet vel luna")
        for fn in (sim._posicion_cuerpo, sim._estado_cuerpo):
            try:
                fn("marte", 0.0)
                errores.append(f"{fn.__name__}('marte') devolvió algo en verlet")
            except KeyError:
                pass
    if errores:
        print(f"[arbol_cuerpos] FALLA ({len(errores)}):")
        for e in errores:
            print("   ", e)
        return False
    print("[arbol_cuerpos] OK")
    return True


def verificar_pymcel_spice(mod) -> bool:
    """Comprueba la LLAMADA a pc.consulta_spice sin kernel SPICE (desde
    11.17.1 se consulta por LOTES de _EPH_NODOS_LOTE épocas y se interpola
    con Hermite). Un falso valida los argumentos contra la firma real
    (inspect.signature) y devuelve un movimiento rectilíneo conocido:
    la interpolación debe reproducirlo exacto. Verifica firma, forma de
    `data` y uso de la salida, NO valores físicos."""
    if not hasattr(mod, "_eph_cargar_lote"):
        print("[pymcel_spice] omitido (el archivo no tiene _eph_cargar_lote)")
        return True
    import inspect
    errores = []
    sig = inspect.signature(mod.pc.consulta_spice)
    llamadas = []
    n = mod._EPH_NODOS_LOTE
    paso = mod._EPH_PASO
    X0, Y0, VX, VY = 1.5e8, -2.5e7, 1.0, 2.0

    def datos(traspuesta=False):
        t = np.arange(n) * paso
        arr = np.column_stack([X0 + VX * t, Y0 + VY * t, np.full(n, 3.0e6),
                               np.full(n, VX), np.full(n, VY), np.full(n, 3.0)])
        return arr.T if traspuesta else arr

    def falso_con(traspuesta=False):
        def falso(*a, **k):
            sig.bind(*a, **k)        # TypeError si la firma real no lo acepta
            llamadas.append(k)
            return datos(traspuesta), np.zeros(n), None
        return falso

    if hasattr(mod, "_EPH_DISCO"):
        mod._EPH_DISCO = None       # esta prueba no toca el disco

    def limpiar():
        mod._EPH_CACHE.clear()
        mod._EPH_FALLO.clear()
        getattr(mod, "_EPH_FUENTE", {}).clear()
        getattr(mod, "_AVISOS_PYMCEL", set()).clear()

    real = mod.pc.consulta_spice
    try:
        for traspuesta in (False, True):
            for cuerpo, nid in (("luna", "301"), ("sol", "10")):
                limpiar(); llamadas.clear()
                mod.pc.consulta_spice = falso_con(traspuesta)
                et = f"{cuerpo}{' (6,N)' if traspuesta else ''}"
                try:
                    pos, vel = mod._estado_pymcel(cuerpo, "2026-01-01T00:00:00", 10.0)
                except Exception as e:
                    errores.append(f"{et}: la llamada falló ({type(e).__name__}: {e})")
                    continue
                esp_pos = (X0 + VX * 10.0, Y0 + VY * 10.0)
                if (abs(pos[0] - esp_pos[0]) > 1e-3 or abs(pos[1] - esp_pos[1]) > 1e-3
                        or abs(vel[0] - VX) > 1e-9 or abs(vel[1] - VY) > 1e-9):
                    errores.append(f"{et}: esperaba {esp_pos} y ({VX}, {VY}); dio {pos} {vel}")
                k = llamadas[0] if llamadas else {}
                if (k.get("id"), k.get("location")) != (nid, "399"):
                    errores.append(f"{et}: id/location = {k.get('id')}/{k.get('location')}")
                ep = k.get("epochs")
                if not (isinstance(ep, list) and len(ep) == n
                        and abs(ep[0] - 2461041.5008) < 1e-3
                        and abs((ep[1] - ep[0]) * 86400.0 - paso) < 1e-3):
                    errores.append(f"{et}: epochs debía ser {n} JD TDB cada {paso} s desde "
                                   f"2026-01-01; dio {None if ep is None else (len(ep), ep[:2])}")
                if len(llamadas) != 1:
                    errores.append(f"{et}: {len(llamadas)} consultas (debía ser 1 por lote)")
        # cache: otra época del mismo lote no vuelve a consultar
        limpiar(); llamadas.clear()
        mod.pc.consulta_spice = falso_con()
        mod._estado_pymcel("luna", "2026-01-01T00:00:00", 10.0)
        mod._estado_pymcel("luna", "2026-01-01T00:00:00", 5000.0)
        if len(llamadas) != 1:
            errores.append(f"caché: {len(llamadas)} consultas para dos épocas del mismo lote")
        # fallos esperables -> respaldo Kepler (con aviso); otros errores suben
        def lanza(exc):
            def f(*a, **k):
                raise exc
            return f
        limpiar()
        mod.pc.consulta_spice = lanza(OSError("sin red"))
        try:
            pos = mod._posicion_pymcel("luna", "2026-01-01T00:00:00", 0.0)
            if not (2e8 < float(np.hypot(*pos)) < 6e8):
                errores.append("OSError: el respaldo Kepler no dio una distancia lunar")
        except Exception as e:
            errores.append(f"OSError debía caer a Kepler; subió {type(e).__name__}")
        limpiar()
        mod.pc.consulta_spice = lambda *a, **k: (np.zeros(6), None, None)
        try:   # forma incorrecta -> ValueError interno -> Kepler con aviso
            mod._posicion_pymcel("luna", "2026-01-01T00:00:00", 0.0)
        except Exception as e:
            errores.append(f"forma incorrecta debía caer a Kepler; subió {type(e).__name__}")
        limpiar()
        mod.pc.consulta_spice = lanza(TypeError("firma rota"))
        try:
            mod._posicion_pymcel("luna", "2026-01-01T00:00:00", 0.0)
            errores.append("TypeError no debía esconderse")
        except TypeError:
            pass
    finally:
        mod.pc.consulta_spice = real
        limpiar()
    if errores:
        print(f"[pymcel_spice] FALLA ({len(errores)}):")
        for e in errores:
            print("   ", e)
        return False
    print("[pymcel_spice] OK (firma, forma de data, caché por lotes; valores sin validar)")
    return True


def verificar_cache_disco(mod) -> bool:
    """Caché de efemérides en disco (12.2.0): segunda corrida sin llamar a
    SPICE, valores idénticos, archivo corrupto/ajeno ignorado, Kepler nunca
    guardado, escritura atómica y fuente real descrita."""
    if not hasattr(mod, "_eph_obtener_lote"):
        print("[cache_disco] omitido (el archivo no tiene caché en disco)")
        return True
    import tempfile
    errores = []
    n, paso = mod._EPH_NODOS_LOTE, mod._EPH_PASO
    t = np.arange(n) * paso
    datos = np.column_stack([1.5e8 + t, -2.5e7 + 2 * t, np.full(n, 3.0e6),
                             np.full(n, 1.0), np.full(n, 2.0), np.full(n, 3.0)])
    llamadas = []

    def falso(*a, **k):
        llamadas.append(k)
        return datos, np.zeros(n), None

    def no_debe_llamarse(*a, **k):
        raise AssertionError("se llamó a consulta_spice con la caché en disco")

    def limpiar_memoria():
        mod._EPH_CACHE.clear(); mod._EPH_FALLO.clear(); mod._EPH_FUENTE.clear()
        getattr(mod, "_AVISOS_PYMCEL", set()).clear()

    real = mod.pc.consulta_spice
    disco_orig = mod._EPH_DISCO
    F = "2026-01-01T00:00:00"
    try:
        with tempfile.TemporaryDirectory() as d:
            ruta = Path(d, "efemerides_cache.json")
            mod._EPH_DISCO = ruta
            # 1) primera vez: consulta SPICE y guarda
            limpiar_memoria(); llamadas.clear(); mod.pc.consulta_spice = falso
            a = mod._estado_pymcel("luna", F, 10.0)
            if len(llamadas) != 1 or not ruta.is_file():
                errores.append(f"1ª vez: {len(llamadas)} consultas, archivo={ruta.is_file()}")
            if "SPICE" not in mod._eph_descripcion(F):
                errores.append(f"descripción tras SPICE: {mod._eph_descripcion(F)}")
            if list(Path(d).glob("*.tmp")):
                errores.append("quedó un archivo temporal")
            # 2) memoria vacía (otra corrida): sale de disco, SIN llamar a SPICE
            limpiar_memoria(); mod.pc.consulta_spice = no_debe_llamarse
            try:
                b = mod._estado_pymcel("luna", F, 10.0)
            except AssertionError as e:
                errores.append(str(e)); b = None
            if b != a:
                errores.append(f"disco != SPICE: {b} vs {a}")
            if mod._eph_descripcion(F) != "pymcel: caché en disco":
                errores.append(f"descripción tras disco: {mod._eph_descripcion(F)}")
            # 3) otra fecha: sí consulta (la clave incluye la fecha)
            limpiar_memoria(); llamadas.clear(); mod.pc.consulta_spice = falso
            mod._estado_pymcel("luna", "2026-02-01T00:00:00", 10.0)
            if len(llamadas) != 1:
                errores.append(f"otra fecha: {len(llamadas)} consultas (debía ser 1)")
            # 4) archivo corrupto: se ignora, consulta y reescribe válido
            ruta.write_text("{ esto no es json", encoding="utf-8")
            limpiar_memoria(); llamadas.clear()
            mod._estado_pymcel("luna", F, 10.0)
            if len(llamadas) != 1:
                errores.append(f"corrupto: {len(llamadas)} consultas (debía ser 1)")
            try:
                json.loads(ruta.read_text(encoding="utf-8"))
            except ValueError:
                errores.append("corrupto: no se reescribió un JSON válido")
            # 5) metadatos ajenos (otro marco/kernel): se ignora
            crudo = json.loads(ruta.read_text(encoding="utf-8"))
            crudo["meta"]["marco"] = "OTRO"
            ruta.write_text(json.dumps(crudo), encoding="utf-8")
            limpiar_memoria(); llamadas.clear()
            mod._estado_pymcel("luna", F, 10.0)
            if len(llamadas) != 1:
                errores.append(f"meta ajena: {len(llamadas)} consultas (debía ser 1)")
            # 6) Kepler (SPICE falla) NO se guarda y se describe como respaldo
            ruta.unlink()
            limpiar_memoria()
            def sin_red(*a, **k):
                raise OSError("sin red")
            mod.pc.consulta_spice = sin_red
            mod._estado_pymcel("luna", F, 10.0)
            if ruta.exists():
                errores.append("el respaldo Kepler dejó un archivo en disco")
            if not mod._eph_descripcion(F).startswith("Kepler"):
                errores.append(f"descripción tras fallo: {mod._eph_descripcion(F)}")
            # 7) lote con valores no finitos en disco: se descarta
            limpiar_memoria(); llamadas.clear(); mod.pc.consulta_spice = falso
            mod._estado_pymcel("luna", F, 10.0)
            crudo = json.loads(ruta.read_text(encoding="utf-8"))
            clave = next(iter(crudo["lotes"]))
            crudo["lotes"][clave][3][0] = None
            ruta.write_text(json.dumps(crudo), encoding="utf-8")
            limpiar_memoria(); llamadas.clear()
            mod._estado_pymcel("luna", F, 10.0)
            if len(llamadas) != 1:
                errores.append(f"lote inválido: {len(llamadas)} consultas (debía ser 1)")
    finally:
        mod.pc.consulta_spice = real
        mod._EPH_DISCO = disco_orig
        limpiar_memoria()
    if errores:
        print(f"[cache_disco] FALLA ({len(errores)}):")
        for e in errores:
            print("   ", e)
        return False
    print("[cache_disco] OK (2ª corrida sin SPICE, corrupto/ajeno/inválido ignorado, "
          "Kepler no se guarda)")
    return True


def verificar_deorbit_sitio(mod, resultados: dict) -> bool:
    """La caída de deorbit en modo "sitio" debe quedar a menos de SITIO[...]
    km de la base. Se calcula con la última fila de cada escenario (t, px,
    py) y la rotación del escenario; no corre nada extra. Se omite si el
    archivo no tiene el modo "sitio"."""
    if not hasattr(mod.Simulacion, "_planificar_deorbit_sitio"):
        print("[deorbit_sitio] omitido (el archivo no tiene el modo 'sitio')")
        return True
    errores = []
    for nombre, tope in SITIO.items():
        if nombre not in resultados:
            continue
        res = resultados[nombre]
        ultima = res["filas"][-1]
        if not res["resumen"]["terminada"]:
            errores.append(f"{nombre}: no terminó (sin impacto)")
            continue
        omega = (mod.OMEGA_T if ESCENARIOS[nombre].get("rotacion_terrestre")
                 else 0.0)
        d = (np.arctan2(ultima["py"], ultima["px"])
             - (np.pi / 2.0 - omega * ultima["t"]) + np.pi) % (2 * np.pi) - np.pi
        km = abs(float(d)) * mod.R_T / 1000.0
        print(f"[deorbit_sitio] {nombre}: cae a {km:.2f} km de la base")
        if km > tope:
            errores.append(f"{nombre}: cae a {km:.1f} km de la base (tope {tope} km)")
    if errores:
        print(f"[deorbit_sitio] FALLA ({len(errores)}):")
        for e in errores:
            print("   ", e)
        return False
    print("[deorbit_sitio] OK")
    return True


def verificar_fisica_17(mod, resultados: dict) -> bool:
    """Atmósfera (a, p), Cd(Mach), empuje vs altitud y equivalencia con 11.16.0."""
    if not hasattr(mod, "factor_cd_mach"):
        return True
    ok = True
    def chk(nombre, val, esp, tol):
        nonlocal ok
        bien = abs(val - esp) <= tol * abs(esp)
        ok = ok and bien
        print(f"[fisica_17] {nombre}: {val:.5g} (esperado {esp:.5g}) "
              f"{'OK' if bien else 'FALLA'}")
    chk("a(0) m/s", mod.velocidad_sonido(0.0), 340.29, 2e-3)
    chk("a(11 km) m/s", mod.velocidad_sonido(11000.0), 295.07, 5e-3)
    chk("p(0) Pa", mod.presion_atmosfera(0.0), 101325.0, 1e-2)
    chk("p(10 km) Pa", mod.presion_atmosfera(10000.0), 26436.0, 0.03)
    chk("p(20 km) Pa", mod.presion_atmosfera(20000.0), 5475.0, 0.05)
    f = mod.factor_cd_mach
    for nombre, val, esp in (("f(0.3)", f(0.3), 1.0), ("f(1.05)", f(1.05), 1.9),
                             ("f(10)", f(10.0), 0.6), ("f(50)", f(50.0), 0.6)):
        chk(nombre, val, esp, 1e-9)
    e1 = mod.ETAPAS_CONFIG[0]
    cfg = mod.Config(); cfg.aplicar_preset()
    sim = mod.Simulacion(cfg)
    et = sim.etapas[0]
    T0, _ = et.empuje_y_flujo(1.0, 101325.0)
    Tv, _ = et.empuje_y_flujo(1.0, 0.0)
    chk("T_nivel_mar/T_vacio (etapa 1)", T0 / Tv, 0.925, 0.02)
    return ok


def verificar_propulsion_18(mod) -> bool:
    """Empuje de las dos etapas contra la guía de usuario de Falcon 9
    (SpaceX): 2ª etapa 981 kN en vacío; 1ª etapa 7,686 MN a nivel del mar."""
    if not hasattr(mod, "factor_cd_mach"):
        return True
    ok = True
    cfg = mod.Config(); cfg.aplicar_preset()
    et1, et2 = mod.Simulacion(cfg).etapas[:2]
    def chk(nombre, val, esp, tol):
        nonlocal ok
        bien = abs(val - esp) <= tol * abs(esp)
        ok = ok and bien
        print(f"[propulsion] {nombre}: {val:.5g} (esperado {esp:.5g}, tol {tol:.1%}) "
              f"{'OK' if bien else 'FALLA'}")
    chk("2ª etapa, empuje de vacío [N]", et2.empuje_y_flujo(1.0, 0.0)[0], 981e3, 1e-3)
    chk("1ª etapa, empuje a nivel del mar [N]", et1.empuje_y_flujo(1.0, 101325.0)[0], 7.686e6, 0.03)
    chk("2ª etapa, duración a flujo máximo [s]", et2.m_comb_restante / et2.D, 397.0, 0.08)
    return ok


def verificar_protecciones_19(mod) -> bool:
    """Pruebas aisladas (sin vuelo completo) de 12.2.0: guardarraíl del
    gravity turn, límite de g y periapsis mínimo de la órbita."""
    if not hasattr(mod.Simulacion, "_periapsis_minimo_orbita"):
        return True
    import math
    errores = []
    # --- 1. gravity_turn: componente radial >= sen 5° cuando gamma <= 0 ---
    gmin = math.sin(math.radians(5.0))
    R = 6.5e6
    casos = {                       # (vx, vy) con la nave en +y
        "cayendo vertical": (0.0, -100.0),
        "cayendo oblicuo":  (80.0, -60.0),
        "horizontal":       (100.0, 0.0),
        "horizontal oeste": (-100.0, 0.0),
        "subiendo (sin cambio)": (30.0, 70.0),
    }
    for nombre, (vx, vy) in casos.items():
        sp = math.hypot(vx, vy)
        tx, ty = mod.calcular_guiado(0.0, R, vx, vy, sp, "gravity_turn")
        rad = ty                    # r̂ = (0, 1)
        norma = math.hypot(tx, ty)
        if abs(norma - 1.0) > 1e-12:
            errores.append(f"gravity_turn '{nombre}': norma {norma}")
        if nombre.startswith("subiendo"):
            if abs(tx - vx / sp) > 1e-15 or abs(ty - vy / sp) > 1e-15:
                errores.append("gravity_turn: con gamma > 0 no debe cambiar la dirección")
        elif rad < gmin - 1e-12:
            errores.append(f"gravity_turn '{nombre}': componente radial {rad:.4f} < sen 5° = {gmin:.4f}")
    print(f"[proteccion] gravity_turn: {len(casos)} casos "
          f"{'OK' if not errores else 'FALLA'}")

    # --- 2. límite de g: T·throttle/m = G_MAX·g0 cuando limita ---
    n0 = len(errores)
    cfg = mod.Config(); cfg.modo_mision = "tripulada"; cfg.perfil_tripulada = "bcb"
    cfg.fecha_lanzamiento = "2026-01-01T00:00:00"; cfg.aplicar_preset()
    sim = mod.Simulacion(cfg)
    sim.etapas[0].separada = True
    sim.etapas[0].activa = False
    sim.estado.etapa_actual = 1
    et = sim.etapas[1]
    et.activa = True
    et.m_comb_restante = 5000.0               # masa chica -> el límite aprieta
    est = sim.estado
    est.px, est.py = 0.0, mod.R_T + 300e3     # alto: sin pérdida por presión
    est.altitud = 300e3
    thr = sim._throttle_g_limit()
    T = et.empuje_y_flujo(thr, 0.0)[0]
    g_sentida = T / sim.masa_total() / mod.G0
    if not (0.0 < thr < 1.0):
        errores.append(f"límite de g: el throttle debía limitar (dio {thr})")
    if abs(g_sentida - cfg.G_MAX_TRIPULADA) > 1e-6:
        errores.append(f"límite de g: carga {g_sentida:.4f} g, esperada {cfg.G_MAX_TRIPULADA} g")
    print(f"[proteccion] límite de g: throttle {thr:.3f}, carga {g_sentida:.3f} g "
          f"(límite {cfg.G_MAX_TRIPULADA} g) {'OK' if len(errores) == n0 else 'FALLA'}")

    # --- 3. periapsis mínimo: 50, 119, 120, 121 km (satelital/tripulada) ---
    n0 = len(errores)
    for modo in ("satelital", "tripulada"):
        for rp_km, esperado in ((50, False), (119, False), (120, True), (121, True)):
            c = mod.Config(); c.modo_mision = modo
            c.fecha_lanzamiento = "2026-01-01T00:00:00"; c.aplicar_preset()
            c.ALTURA_SEGURIDAD = 120_000.0
            s2 = mod.Simulacion(c)
            e = s2.estado
            ra = mod.R_T + 400e3                        # en el apogeo, casi circular
            rp = mod.R_T + rp_km * 1e3
            v_ap = math.sqrt(2 * mod.MU_T * rp / (ra * (ra + rp)))
            e.px, e.py, e.vx, e.vy = 0.0, ra, v_ap, 0.0   # apogeo: v tangencial
            e.altitud = ra - mod.R_T
            e.fase = "circularizacion"
            s2.ecc_max_efectiva = lambda: 1.0              # solo se prueba el periapsis
            s2._verificar_orbita()
            if s2.eventos.orbita_detectada != esperado:
                errores.append(f"periapsis {rp_km} km en {modo}: detectada="
                               f"{s2.eventos.orbita_detectada}, esperado {esperado}")
    print(f"[proteccion] periapsis mínimo (50/119/120/121 km): "
          f"{'OK' if len(errores) == n0 else 'FALLA'}")
    for e in errores:
        print("   ", e)
    return not errores


def verificar_import_hijo(archivo: str) -> bool:
    """pymcel imprime un banner al importarse; el hijo "spawn" del
    planificador de desórbita reimporta el archivo y NO debe repetirlo."""
    import subprocess, tempfile, textwrap
    with tempfile.TemporaryDirectory() as d:
        Path(d, "auxiliar_hijo.py").write_text(textwrap.dedent(f"""
            import importlib.util, sys
            from unittest.mock import MagicMock
            for n in ("tkinter", "matplotlib.backends.backend_tkagg"):
                sys.modules.setdefault(n, MagicMock())
            import matplotlib; matplotlib.use = lambda *a, **k: None
            def cargar():
                s = importlib.util.spec_from_file_location("cohete_hijo", {os.path.abspath(archivo)!r})
                m = importlib.util.module_from_spec(s); sys.modules["cohete_hijo"] = m
                s.loader.exec_module(m)
            def hijo():
                cargar()
        """), encoding="utf-8")
        Path(d, "principal.py").write_text(textwrap.dedent("""
            import multiprocessing, auxiliar_hijo
            if __name__ == "__main__":
                ctx = multiprocessing.get_context("spawn")
                p = ctx.Process(target=auxiliar_hijo.hijo); p.start(); p.join(); print('EXIT', p.exitcode)
        """), encoding="utf-8")
        env = dict(os.environ, PYTHONPATH=d)
        r = subprocess.run([sys.executable, "principal.py"], cwd=d, env=env,
                           capture_output=True, text=True, timeout=300)
    n = r.stdout.count("Bienvenido a PyMCel")
    ok = (r.returncode == 0 and "EXIT 0" in r.stdout and n == 0)
    print(f"[import_hijo] banner de pymcel en el hijo: {n} (debe ser 0) "
          f"{'OK' if ok else 'FALLA ' + r.stderr[-300:]}")
    return ok


def verificar_dv_prop(mod) -> bool:
    """Δv propulsivo (12.2.0): ∫T/m dt contra la ecuación del cohete, y
    registro de burnout/desórbita con `dv_prop` y `delta_rapidez` (sin `dv`)."""
    if not hasattr(mod.Etapa, "acumular_dv"):
        print("[dv_prop] omitido (el archivo no tiene Δv propulsivo)")
        return True
    import math
    errores = []
    # 1) unidad: empuje constante en vacío -> u0·ln(m0/m1) (error O(dt²))
    cfg = mod.Config(); cfg.aplicar_preset()
    et = mod.Simulacion(cfg).etapas[1]
    m_ini = et.m_estr + et.m_comb_restante
    masa, dt = m_ini, 0.25
    for _ in range(int(100.0 / dt)):
        T, flujo = et.empuje_y_flujo(1.0, 0.0)
        et.acumular_dv(T, masa, flujo, dt)
        et.consumir(flujo, dt)
        masa -= flujo * dt
    ideal = et.u0 * math.log(m_ini / masa)
    rel = abs(et.dv_prop - ideal) / ideal
    print(f"[dv_prop] unidad: {et.dv_prop:.3f} m/s vs u0·ln(m0/m1) = {ideal:.3f} (err {rel:.1e})")
    if rel > 1e-5:
        errores.append(f"Δv propulsivo no coincide con la ecuación del cohete (err {rel:.2e})")
    # 2) empuje cero o paso cero no suman
    antes = et.dv_prop
    et.acumular_dv(0.0, 1e4, 0.0, 1.0); et.acumular_dv(1e6, 1e4, 1.0, 0.0)
    if et.dv_prop != antes:
        errores.append("acumular_dv sumó con empuje o paso nulo")
    # 3) vuelo real: registro de burnout y desórbita
    c = mod.Config(); c.modo_mision = "deorbit"; c.fecha_lanzamiento = "2026-01-01T00:00:00"
    c.aplicar_preset(); c.deorbit_activacion = "orbita"
    sim = mod.Simulacion(c)
    import contextlib, io
    with contextlib.redirect_stdout(io.StringIO()):
        for _ in range(80000):
            if sim.estado.simulacion_terminada:
                break
            sim.avanzar_paso()
    for i, e in enumerate(sim.etapas):
        d = e.datos_burnout
        if d is None:
            continue
        if "dv" in d or "dv_prop" not in d or "delta_rapidez" not in d:
            errores.append(f"burnout etapa {i}: claves {sorted(d)}")
            continue
        print(f"[dv_prop] burnout etapa {i}: Δv motor {d['dv_prop']:.0f} m/s, "
              f"Δrapidez {d['delta_rapidez']:.0f} m/s")
        if d["dv_prop"] <= d["delta_rapidez"] and i == 0:
            errores.append("etapa 1: el Δv del motor debía superar al Δrapidez (pérdidas por gravedad)")
    dm = sim.eventos.datos_mision
    if "dv" in dm or "dv_prop" not in dm or "delta_rapidez" not in dm:
        errores.append(f"desórbita: claves {sorted(k for k in dm if 'dv' in k or 'rapidez' in k)}")
    else:
        rel = abs(dm["dv_prop"] - dm["delta_rapidez"]) / dm["delta_rapidez"]
        print(f"[dv_prop] desórbita: Δv motor {dm['dv_prop']:.1f} m/s, Δrapidez "
              f"{dm['delta_rapidez']:.1f} m/s (dif {rel:.1%})")
        if rel > 0.10:
            errores.append(f"desórbita: Δv motor y Δrapidez difieren {rel:.1%} (>10 %)")
    for e in errores:
        print("   ", e)
    print(f"[dv_prop] {'OK' if not errores else 'FALLA'}")
    return not errores


def verificar_legacy(resultados: dict) -> bool:
    """dinamica="congelada" reproduce 11.21.2 y eventos="paso" reproduce
    12.0.1 (rtol 1e-10 desde 12.13.8), si los JSON de referencia están junto a este script."""
    global RTOL
    # (12.13.8) 1e-10: hypot -> sqrt (12.13.6) mueve el redondeo a ~1e-12 relativo tras 40 000
    # pasos; antes se exigía 1e-12 (ruido numérico, no física).
    viejo, RTOL = RTOL, 1e-10
    ok = True
    refs = {}
    try:
        for nombre, (archivo, base) in LEGACY_DE.items():
            if nombre not in resultados:
                continue
            ruta = Path(__file__).with_name(archivo)
            if not ruta.exists():
                print(f"[legacy] {nombre}: omitido (falta {archivo})")
                continue
            ref = refs.setdefault(archivo, json.loads(ruta.read_text(encoding="utf-8")))
            if base not in ref:
                continue
            dif = comparar(ref[base], json.loads(json.dumps(resultados[nombre])))
            print(f"[legacy] {nombre} vs {archivo[14:-5]}/{base}: "
                  f"{'idéntico' if not dif else f'{len(dif)} diferencias'}")
            for d in dif[:5]:
                print("   ", d)
            ok = ok and not dif
    finally:
        RTOL = viejo
    print(f"[legacy] {'OK' if ok else 'FALLA'}")
    return ok


def _sim_ascenso(mod, modo, perfil, dt, dinamica):
    """Ascenso hasta detectar órbita; devuelve (burnout etapa 1, sim)."""
    import contextlib, io
    c = mod.Config(); c.modo_mision = modo
    if perfil:
        c.perfil_tripulada = perfil
    c.fecha_lanzamiento = "2026-01-01T00:00:00"; c.aplicar_preset()
    c.dinamica = dinamica
    sim = mod.Simulacion(c); sim.estado.dt = dt; sim._dt_base = dt
    with contextlib.redirect_stdout(io.StringIO()):
        while (not sim.estado.simulacion_terminada and sim.estado.t < 700
               and not sim.eventos.orbita_detectada):
            sim.avanzar_paso()
    return sim.etapas[0].datos_burnout, sim


def verificar_dinamica_v12(mod) -> bool:
    """Dinámica variable (12.2.0): balance de masa, Δv, órbita analítica,
    convergencia, validación de config y fallo del integrador."""
    if not hasattr(mod, "LeyV12"):
        print("[dinamica_v12] omitido (el archivo no tiene dinámica variable)")
        return True
    import math
    errores = []
    cfg = mod.Config(); cfg.aplicar_preset(); cfg.integrador = "rk45"
    sim = mod.Simulacion(cfg)
    R, MU = mod.R_T, mod.MU_T
    r0 = R + 400e3
    vc = math.sqrt(MU / r0)

    def ley(on, flujo=300.0, u0=3000.0, presion=False):
        return mod.LeyV12(0.0, u0, flujo, 6.0, presion, on, "vertical", False,
                          0.3, 10.0, False, 0.0, 0.0)

    # 1) balance de masa y Δv con empuje constante en vacío (sin gravedad no
    #    se puede apagar: se compara con la solución analítica de m y de Δv)
    m0, dur = 5.0e5, 100.0
    y = sim._integrar_tramo(( r0, 0.0, 0.0, vc, m0, 0.0), 0.0, dur, ley(True), 1.0)
    m_esp = m0 - 300.0 * dur
    dv_esp = 3000.0 * math.log(m0 / m_esp)
    e_m, e_dv = abs(y[4] - m_esp), abs(y[5] - dv_esp) / dv_esp
    print(f"[dinamica_v12] balance de masa: |m - (m0 - ṁt)| = {e_m:.2e} kg; "
          f"Δv = {y[5]:.3f} vs u0·ln(m0/m1) = {dv_esp:.3f} (err {e_dv:.1e})")
    if e_m > 1e-6 or e_dv > 1e-8:
        errores.append("masa o Δv integrados no coinciden con la solución analítica")
    # 2) motor apagado: ni masa ni Δv cambian
    y = sim._integrar_tramo((r0, 0.0, 0.0, vc, m0, 5.0), 0.0, 50.0, ley(True).apagada(), 0.0)
    if y[4] != m0 or y[5] != 5.0:
        errores.append("sin empuje la masa o el Δv cambiaron")
    # 2b) T <= 0 por presión (tobera bloqueada): ni masa ni Δv cambian
    bloq = mod.LeyV12(0.0, 100.0, 10.0, 6.0, True, True, "vertical", False,
                      0.0, 0.0, False, 0.0, 0.0)     # T_vac = 1 kN < A_sal·p(0)
    y = sim._integrar_tramo((R, 0.0, 0.0, 0.0, m0, 0.0), 0.0, 1.0, bloq, 0.0)
    if y[4] != m0 or y[5] != 0.0:
        errores.append("con T <= 0 la masa o el Δv cambiaron")
    # 3) órbita circular sin empuje ni arrastre: radio y energía conservados
    T = 2 * math.pi * r0 / vc
    sin_arr = mod.LeyV12(0.0, 0.0, 0.0, 0.0, False, False, "vertical", False,
                         0.0, 0.0, False, 0.0, 0.0)
    n_p = int(round(T / 5.0))                # pasos de 5 s encadenados
    y, t_p = (r0, 0.0, 0.0, vc, m0, 0.0), 0.0
    for _ in range(n_p):
        y = sim._integrar_tramo(y, t_p, T / n_p, sin_arr, 0.0)
        t_p += T / n_p
    cierre = math.hypot(y[0] - r0, y[1])
    dr = abs(math.hypot(y[0], y[1]) - r0)
    de = abs((0.5 * (y[2]**2 + y[3]**2) - MU / math.hypot(y[0], y[1]))
             / (-0.5 * MU / r0) - 1.0)
    print(f"[dinamica_v12] órbita circular 1 periodo (pasos de 5 s): cierre {cierre:.2f} m, "
          f"|Δr| {dr:.2f} m, ΔE/E {de:.1e}")
    if cierre > 100.0 or dr > 50.0 or de > 1e-7:
        errores.append("órbita circular: cierre/energía fuera de tolerancia")
    # 4) convergencia: con dt = 0.25 la dinámica variable queda más cerca del
    #    límite (dt = 1/32) que la congelada, en altitud y rapidez del corte
    b_ref, _ = _sim_ascenso(mod, "tripulada", "directo", 0.03125, "variable")
    b_var, _ = _sim_ascenso(mod, "tripulada", "directo", 0.25, "variable")
    b_con, _ = _sim_ascenso(mod, "tripulada", "directo", 0.25, "congelada")
    for k in ("alt", "v"):
        ev, ec = abs(b_var[k] - b_ref[k]), abs(b_con[k] - b_ref[k])
        print(f"[dinamica_v12] convergencia {k} corte etapa 1: error variable "
              f"{ev:.1f} vs congelada {ec:.1f}")
        if not ev < ec:
            errores.append(f"convergencia en {k}: la dinámica variable no mejora")
    # 5) Config inválida
    c = mod.Config(); c.dinamica = "x"
    try:
        c._validar_config(); errores.append("dinamica inválida no lanzó ValueError")
    except ValueError:
        pass
    # 6) fallo del integrador (RK45 no converge) -> fin controlado
    import contextlib, io
    c = mod.Config(); c.modo_mision = "satelital"; c.fecha_lanzamiento = "2026-01-01T00:00:00"
    c.aplicar_preset()
    s = mod.Simulacion(c)
    class _Mal: success = False; message = "forzado por la prueba"
    orig = mod.solve_ivp
    mod.solve_ivp = lambda *a, **k: _Mal()
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            for _ in range(2000):
                if s.estado.simulacion_terminada:
                    break
                s.avanzar_paso()
    finally:
        mod.solve_ivp = orig
    if not (s.estado.simulacion_terminada and s.eventos.fallo_integrador):
        errores.append("fallo del integrador no terminó la simulación")
    # 7) la masa total tras el ascenso coincide con estructura + combustible
    _, s2 = _sim_ascenso(mod, "satelital", None, 0.25, "variable")
    esperado = sum(e.m_estr + e.m_comb_restante for e in s2.etapas if not e.separada)
    if abs(s2.masa_total() - esperado) > 1e-9:
        errores.append("masa_total inconsistente")
    for e in errores:
        print("   ", e)
    print(f"[dinamica_v12] {'OK' if not errores else 'FALLA'}")
    return not errores


def _sim_hasta_primer_corte(mod, modo, perfil, **cfg_extra):
    """Corre hasta el primer corte resuelto por evento; en ese punto compara el
    instante con la solución de referencia (DOP853, rtol 1e-13) y con la
    bisección. Devuelve (t_evento, t_exacto, t_biseccion) medidos desde el
    inicio del paso."""
    import contextlib, io
    import numpy as _np
    from scipy.integrate import solve_ivp as _ivp
    from scipy.optimize import brentq
    mod._fetch_horizons_state = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("x"))
    mod._TERCEROS_CUERPOS_FUENTE = "sin_pymcel"
    c = mod.Config(); c.modo_mision = modo
    if perfil:
        c.perfil_tripulada = perfil
    c.fecha_lanzamiento = "2026-01-01T00:00:00"; c.aplicar_preset()
    for k, v in cfg_extra.items():
        setattr(c, k, v)
    cap = []
    orig = mod.Simulacion._avanzar_v12_eventos

    def spy(self, y0, t0, dt, t_emp, ley, empuje, bc, suelo, eps=1e-6):
        r = orig(self, y0, t0, dt, t_emp, ley, empuje, bc, suelo, eps)
        if r[3] == "corte" and not cap:
            sol = _ivp(mod.derivadas_v12, (t0, t0 + t_emp), list(y0), args=(ley,),
                       method="DOP853", rtol=1e-13, atol=1e-10, dense_output=True)
            f = lambda t: self._margen_corte(sol.sol(t))
            ts = _np.linspace(t0, t0 + t_emp, 2000)
            g = [f(t) for t in ts]
            i = next(k for k in range(len(g) - 1) if g[k] < 0 <= g[k + 1])
            tc = brentq(f, ts[i], ts[i + 1], xtol=1e-14)
            y_max = self._avanzar_con_empuje(y0, t0, dt, t_emp, ley, ley.flujo, empuje)
            tb, _ = self._localizar_corte(y0, t0, dt, t_emp, ley, ley.flujo,
                                          empuje, y_max)
            cap.append((r[1], tc - t0, tb))
        return r
    mod.Simulacion._avanzar_v12_eventos = spy
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            s = mod.Simulacion(c)
            while not cap and not s.estado.simulacion_terminada:
                s.avanzar_paso()
    finally:
        mod.Simulacion._avanzar_v12_eventos = orig
    return cap[0] if cap else None


def verificar_eventos_v12(archivo: str) -> bool:
    """Eventos de solve_ivp (12.2.0): margen de corte equivalente a
    `_corte_por_estado`, instante del corte frente a la solución de
    referencia, impacto con masa/Δv del integrador, reintentos y bisección
    de respaldo, y validación de Config."""
    mod = cargar_modulo(archivo)
    if not hasattr(mod.Simulacion, "_margen_corte"):
        print("[eventos_v12] omitido (el archivo no tiene eventos de integrador)")
        return True
    import contextlib, io, random
    errores = []
    # 1) margen >= 0  <=>  _corte_por_estado, en estados aleatorios de cada modo
    mod._fetch_horizons_state = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("x"))
    mod._TERCEROS_CUERPOS_FUENTE = "sin_pymcel"
    rng = random.Random(12)
    discrepancias = total = 0
    for modo, perfil in (("satelital", None), ("tripulada", "bcb"), ("tripulada", "directo"),
                         ("suborbital", None), ("gto", None), ("tli", None),
                         ("escape", None), ("deorbit", None)):
        c = mod.Config(); c.modo_mision = modo
        if perfil:
            c.perfil_tripulada = perfil
        c.fecha_lanzamiento = "2026-01-01T00:00:00"; c.aplicar_preset()
        with contextlib.redirect_stdout(io.StringIO()):
            sim = mod.Simulacion(c)
        for fase in ("gravity_turn", "circularizacion", "deorbit", "coasting"):
            for cutoff in (False, True):
                sim.estado.fase = fase
                sim.estado.cutoff_por_apogeo = cutoff
                sim._deorbit_activo = (fase == "deorbit")
                for _ in range(12):
                    r = mod.R_T + rng.uniform(100e3, 4e7)
                    th = rng.uniform(0, 6.28)
                    v = rng.uniform(2e3, 12e3)
                    ph = rng.uniform(-1.6, 1.6)
                    y = (r * math.cos(th), r * math.sin(th),
                         v * math.cos(th + 1.5708 + ph * 0.3) , v * math.sin(th + 1.5708 + ph * 0.3))
                    a = sim._margen_corte(y) >= 0.0
                    b = sim._corte_por_estado(y)
                    total += 1
                    discrepancias += (a != b)
    print(f"[eventos_v12] margen vs _corte_por_estado: {discrepancias} discrepancias en {total} estados")
    if discrepancias:
        errores.append("margen de corte no equivale a _corte_por_estado")
    # 2) instante del corte: evento vs referencia (DOP853 rtol 1e-13) vs bisección
    for modo, perfil in (("suborbital", None), ("tripulada", "bcb"), ("gto", None)):
        r = _sim_hasta_primer_corte(cargar_modulo(archivo), modo, perfil)
        if r is None:
            errores.append(f"{modo}: no hubo corte por evento"); continue
        t_ev, t_ex, t_bi = r
        print(f"[eventos_v12] corte {modo}: evento {t_ev - 0:.9f} s, exacto {t_ex:.9f}, "
              f"error evento {t_ev - t_ex:+.1e} s, bisección {t_bi - t_ex:+.1e} s")
        if abs(t_ev - t_ex) > 1e-7:
            errores.append(f"{modo}: el corte por evento se aparta {t_ev - t_ex:.2e} s de la referencia")
    # 3) un vuelo completo: sin respaldo en los modos nominales, impacto bien resuelto
    def vuelo(modo, **extra):
        m2 = cargar_modulo(archivo)
        m2._fetch_horizons_state = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("x"))
        m2._TERCEROS_CUERPOS_FUENTE = "sin_pymcel"
        for k, v in extra.pop("_mod", {}).items():
            setattr(m2, k, v)
        c = m2.Config(); c.modo_mision = modo
        c.fecha_lanzamiento = "2026-01-01T00:00:00"; c.aplicar_preset()
        for k, v in extra.items():
            setattr(c, k, v)
        with contextlib.redirect_stdout(io.StringIO()):
            s = m2.Simulacion(c)
            for _ in range(60000):
                if s.estado.simulacion_terminada:
                    break
                s.avanzar_paso()
        return m2, s
    m2, s = vuelo("suborbital")
    st = s.stats_eventos
    h = math.hypot(s.estado.px, s.estado.py) - m2.R_T
    print(f"[eventos_v12] suborbital: stats {st}, altitud al impacto {h:.2e} m")
    if st["suelo_evento"] != 1 or abs(h) > 1e-3 or st["corte_biseccion"] != 0:
        errores.append("suborbital: impacto o corte no resueltos por evento")
    # masa tras el vuelo = estructura + combustible restante (sin pérdidas)
    esperado = sum(e.m_estr + e.m_comb_restante for e in s.etapas if not e.separada)
    if abs(s.masa_total() - esperado) > 1e-9:
        errores.append("suborbital: masa total inconsistente")
    # 4) respaldo: sin reintentos el corte se resuelve por bisección y da el mismo vuelo
    _, s_ev = vuelo("tripulada", perfil_tripulada="bcb")
    _, s_bi = vuelo("tripulada", perfil_tripulada="bcb", _mod={"_CORTE_INTENTOS": 0})
    dt_ = abs(s_ev.estado.t - s_bi.estado.t)
    dv_ = math.hypot(s_ev.estado.px - s_bi.estado.px, s_ev.estado.py - s_bi.estado.py)
    print(f"[eventos_v12] respaldo: bisección {s_bi.stats_eventos['corte_biseccion']} cortes "
          f"(evento {s_ev.stats_eventos['corte_evento']}), Δposición final {dv_:.2e} m")
    if s_bi.stats_eventos["corte_biseccion"] < 1 or s_ev.stats_eventos["corte_biseccion"] != 0:
        errores.append("el respaldo por bisección no se activó / se activó sin necesidad")
    # 5) Config
    c = mod.Config(); c.eventos = "x"
    try:
        c._validar_config(); errores.append("eventos inválido no lanzó ValueError")
    except ValueError:
        pass
    for e in errores:
        print("   ", e)
    print(f"[eventos_v12] {'OK' if not errores else 'FALLA'}")
    return not errores


def verificar_plan_por_tramos(archivo: str) -> bool:
    """Planificación de la desórbita por tramos (12.2.0): mismo plan que la
    completa (bit a bit), la física espera sin avanzar, ninguna llamada es
    larga, y el descarte del proceso aparte no bloquea."""
    mod = cargar_modulo(archivo)
    if not hasattr(mod.Simulacion, "_planificar_deorbit_sitio_gen"):
        print("[plan_tramos] omitido (el archivo no planifica por tramos)")
        return True
    import contextlib, copy, io
    mod._fetch_horizons_state = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("x"))
    mod._TERCEROS_CUERPOS_FUENTE = "sin_pymcel"
    errores = []
    c = mod.Config(); c.modo_mision = "deorbit"
    c.fecha_lanzamiento = "2026-01-01T00:00:00"; c.aplicar_preset()
    with contextlib.redirect_stdout(io.StringIO()):
        sim = mod.Simulacion(c)
        for _ in range(60000):
            if sim._debe_planificar_deorbit():
                break
            sim.avanzar_paso()
    if not sim._debe_planificar_deorbit():
        print("[plan_tramos] FALLA: no se llegó al punto de planificación")
        return False
    with contextlib.redirect_stdout(io.StringIO()):
        a = copy.deepcopy(sim)
        a._planificar_deorbit_sitio()
    t_ref = a._deorbit_t_plan

    def por_tramos(s, motivo):
        s.presupuesto_plan_s = 0.0          # un tramo por llamada
        t0 = s.estado.t
        llamadas, peor = 0, 0.0
        import time as _t
        with contextlib.redirect_stdout(io.StringIO()):
            while s._deorbit_t_plan is None and llamadas < 20000:
                u = _t.perf_counter()
                s.avanzar_paso()
                peor = max(peor, _t.perf_counter() - u)
                llamadas += 1
                if s._deorbit_t_plan is None and (s.estado.t != t0 or not s.esperando_plan):
                    errores.append(f"{motivo}: la física avanzó o no marcó espera")
                    break
        return llamadas, peor

    b = copy.deepcopy(sim)
    n, peor = por_tramos(b, "por tramos")
    print(f"[plan_tramos] por tramos: {n} llamadas, la más larga {peor * 1000:.0f} ms; "
          f"plan {b._deorbit_t_plan!r} vs completo {t_ref!r}")
    if b._deorbit_t_plan != t_ref:
        errores.append("el plan por tramos difiere del completo")
    if n < 20:
        errores.append("la planificación no se repartió en tramos")
    if peor > 2.0:
        errores.append(f"una llamada tardó {peor:.2f} s (> 2 s)")
    # proceso aparte descartado (error): sigue por tramos, no de una vez
    class _Falso:
        def consultar(self): return ("error", "falla simulada")
        def transcurrido(self): return 1.0
        def cancelar(self): pass
        def listo(self): return True
    d = copy.deepcopy(sim)
    d._planificador_externo = _Falso()
    n2, peor2 = por_tramos(d, "descarte")
    print(f"[plan_tramos] tras descartar el proceso aparte: {n2} llamadas, la más "
          f"larga {peor2 * 1000:.0f} ms; plan igual: {d._deorbit_t_plan == t_ref}")
    if d._deorbit_t_plan != t_ref or n2 < 20:
        errores.append("el descarte del proceso aparte no planificó por tramos")
    # sin presupuesto (scripts): completo y mismo plan
    e = copy.deepcopy(sim)
    with contextlib.redirect_stdout(io.StringIO()) as buf:
        e.avanzar_paso()
    if e._deorbit_t_plan != t_ref or "Planificación completa" not in buf.getvalue():
        errores.append("sin presupuesto no planificó completo con mensaje de tiempo")
    for x in errores:
        print("   ", x)
    print(f"[plan_tramos] {'OK' if not errores else 'FALLA'}")
    return not errores


def _sim_marco(mod, modo="satelital", terceros=False, **extra):
    """Simulación con marco móvil y efemérides Kepler, sin red."""
    import contextlib, io
    mod._fetch_horizons_state = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("x"))
    mod._TERCEROS_CUERPOS_FUENTE = "sin_pymcel"
    if hasattr(mod, "_EPH_DISCO"):
        mod._EPH_DISCO = None
    c = mod.Config(); c.modo_mision = modo
    c.fecha_lanzamiento = "2026-01-01T00:00:00"; c.aplicar_preset()
    c.usar_marco_movil = True
    c.usar_terceros_cuerpos = terceros
    for k, v in extra.items():
        setattr(c, k, v)
    with contextlib.redirect_stdout(io.StringIO()):
        return mod.Simulacion(c)


def _estado_eci(mod, sim):
    """(posición ECI, velocidad ECI) de la nave en el marco actual."""
    est = sim.estado
    (lx, ly), (lvx, lvy) = sim._estado_cuerpo("luna", est.t)
    if est.marco == "tierra":
        return (est.px, est.py), (est.vx, est.vy)
    if est.marco == "luna":
        return (est.px + lx, est.py + ly), (est.vx + lvx, est.vy + lvy)
    theta = sim._angulo_marco_rotante()
    r_in, v_in = mod._rotante_a_inercial((est.px, est.py), (est.vx, est.vy),
                                         est.crucero_omega, theta)
    mu = mod.RAZON_MASA_TL
    return ((r_in[0] + mu * lx, r_in[1] + mu * ly),
            (v_in[0] + mu * lvx, v_in[1] + mu * lvy))


def verificar_transiciones_v12(archivo: str) -> bool:
    """Transiciones de marco en el instante del cruce (12.2.0): continuidad
    de posición/velocidad inercial al cambiar de marco (ida y vuelta), instante
    del cruce frente a la referencia (DOP853, rtol 1e-13) y frente a la
    detección por paso, y vuelo completo de tli."""
    mod = cargar_modulo(archivo)
    if not hasattr(mod.Simulacion, "_fisica_con_transiciones"):
        print("[transiciones_v12] omitido (el archivo no tiene transiciones exactas)")
        return True
    import contextlib, io, math
    import numpy as _np
    from scipy.integrate import solve_ivp as _ivp
    errores = []
    # 1) continuidad. Tierra<->crucero: posición/velocidad ECI (inercial)
    #    continuas. crucero<->Luna: se conserva el estado RELATIVO a la Luna del
    #    modelo (la geometría del encuentro). Con la Luna circular congelada del
    #    crucero y la Luna de las efemérides del marco lunar, el estado ECI NO
    #    es continuo en general (límite conocido del modelo, medido en 4):
    #    aquí el cambio es en el instante de congelación, donde la posición
    #    coincide y la velocidad difiere en ω×r - v_Luna_efemérides.
    sim = _sim_marco(mod)
    est = sim.estado
    est.t = 150_000.0
    ang = 0.7
    est.px, est.py = 3.0e8 * math.cos(ang), 3.0e8 * math.sin(ang)
    est.vx, est.vy = 400.0, 900.0
    inicio = (est.px, est.py, est.vx, est.vy)

    def rel_luna(s):
        """Estado relativo a la Luna del modelo (orientación inercial)."""
        e = s.estado
        if e.marco == "luna":
            return (e.px, e.py), (e.vx, e.vy)
        a_ = e.crucero_a
        x_l = (1.0 - mod.RAZON_MASA_TL) * a_
        return mod._rotante_a_inercial((e.px - x_l, e.py), (e.vx, e.vy),
                                       e.crucero_omega, s._angulo_marco_rotante())
    peor_eci = [0.0, 0.0]
    peor_rel = [0.0, 0.0]
    for destino, nombre in (("crucero", "tierra->crucero"), ("luna", "crucero->luna"),
                            ("crucero", "luna->crucero"), ("tierra", "crucero->tierra")):
        (r0, v0) = _estado_eci(mod, sim)
        rl0 = rel_luna(sim) if sim.estado.marco in ("crucero", "luna") else None
        with contextlib.redirect_stdout(io.StringIO()):
            sim._forzar_transicion(destino)
        (r1, v1) = _estado_eci(mod, sim)
        if est.marco != destino:
            errores.append(f"{nombre}: el marco quedó en {est.marco}")
        if nombre in ("tierra->crucero",):
            peor_eci[0] = max(peor_eci[0], math.hypot(r1[0] - r0[0], r1[1] - r0[1]))
            peor_eci[1] = max(peor_eci[1], math.hypot(v1[0] - v0[0], v1[1] - v0[1]))
        elif nombre in ("crucero->luna", "luna->crucero"):
            rl1 = rel_luna(sim)
            peor_rel[0] = max(peor_rel[0], math.hypot(rl1[0][0] - rl0[0][0], rl1[0][1] - rl0[0][1]))
            peor_rel[1] = max(peor_rel[1], math.hypot(rl1[1][0] - rl0[1][0], rl1[1][1] - rl0[1][1]))
    fin = (est.px, est.py, est.vx, est.vy)
    ida_vuelta = (math.hypot(fin[0] - inicio[0], fin[1] - inicio[1]),
                  math.hypot(fin[2] - inicio[2], fin[3] - inicio[3]))
    print(f"[transiciones_v12] Tierra→crucero ECI: posición {peor_eci[0]:.1e} m, velocidad "
          f"{peor_eci[1]:.1e} m/s; crucero↔Luna relativo a la Luna: {peor_rel[0]:.1e} m, "
          f"{peor_rel[1]:.1e} m/s; ida y vuelta Tierra→…→Tierra: {ida_vuelta[0]:.1e} m, "
          f"{ida_vuelta[1]:.2f} m/s (crucero→Tierra usa la Tierra del modelo: μ·Δv_Luna)")
    if peor_eci[0] > 1e-3 or peor_eci[1] > 1e-6:
        errores.append("tierra->crucero no conserva posición/velocidad inercial")
    if peor_rel[0] > 1e-3 or peor_rel[1] > 1e-6:
        errores.append("crucero<->luna no conserva el estado relativo a la Luna")
    if ida_vuelta[0] > 1e-3 or ida_vuelta[1] > 2.0:
        errores.append("la ida y vuelta se aparta más de lo esperado (>2 m/s)")
    # 2) instante del cruce: evento vs referencia vs detección por paso
    def correr_hasta(eventos, marco_destino, estado_inicial=None, dt=30.0, maxp=40000):
        s = _sim_marco(mod, eventos=eventos)
        e = s.estado
        m_ = _np.asarray(s._posicion_cuerpo("luna", 0.0), dtype=float)
        u = m_ / _np.hypot(m_[0], m_[1])
        e.px, e.py = (float(c) for c in (mod.UMBRAL_TIERRA_CRUCERO - 4.0e5) * u)
        e.vx, e.vy = (float(c) for c in 1000.0 * u)
        e.dt, e.fase = dt, "coasting"
        for et in s.etapas:
            et.activa = False
        registro = []
        orig = s._forzar_transicion
        def espia(destino, _o=orig):
            registro.append((destino, e.t, (e.px, e.py, e.vx, e.vy), e.marco))
            return _o(destino)
        s._forzar_transicion = espia
        orig_v = s._verificar_transicion_marco
        def espia_v(_o=orig_v):
            antes = e.marco
            _o()
            if e.marco != antes:
                registro.append((e.marco, e.t, None, antes))
        s._verificar_transicion_marco = espia_v
        y_ini = (e.px, e.py, e.vx, e.vy); t_ini = e.t
        with contextlib.redirect_stdout(io.StringIO()):
            for _ in range(maxp):
                s.avanzar_paso()
                if e.marco == marco_destino or e.simulacion_terminada:
                    break
        return s, registro, y_ini, t_ini
    # 2a) Tierra -> crucero
    s_ev, reg_ev, y_ini, t_ini = correr_hasta("integrador", "crucero")
    s_pa, reg_pa, _, _ = correr_hasta("paso", "crucero")
    ref = _ivp(mod.derivadas_marco_tierra, (0.0, 4e5), list(y_ini),
               args=(0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0), method="DOP853",
               rtol=1e-13, atol=1e-10,
               events=lambda t, y, *a: math.hypot(y[0], y[1]) - mod.UMBRAL_TIERRA_CRUCERO)
    t_ref = t_ini + float(ref.t_events[0][0])
    if not reg_ev or not reg_pa:
        errores.append("tierra->crucero: no hubo transición")
    else:
        e_ev, e_pa = reg_ev[0][1] - t_ref, reg_pa[0][1] - t_ref
        print(f"[transiciones_v12] Tierra→crucero: error del instante, evento {e_ev:+.2e} s, "
              f"por paso {e_pa:+.1f} s")
        if abs(e_ev) > 1e-4:
            errores.append(f"tierra->crucero: el evento se aparta {e_ev:.2e} s de la referencia")
    # 2b) crucero -> Luna (referencia CRTBP desde el estado del cambio anterior)
    s_ev2, reg2, _, _ = correr_hasta("integrador", "luna")
    if len(reg2) < 2:
        errores.append("crucero->luna: no hubo transición")
    else:
        d_c, t_c, y_c, _m = reg2[0]
        e = s_ev2.estado
        # estado justo tras entrar a crucero: se reconstruye con el espía
        # (y_c es el estado Hermite en el marco inercial antes de transformar);
        # la referencia parte del estado en el marco rotante tras la transición.
        s_ref = _sim_marco(mod, eventos="integrador")
        s_ref.estado.t = t_c
        s_ref.estado.px, s_ref.estado.py, s_ref.estado.vx, s_ref.estado.vy = y_c[:4]
        with contextlib.redirect_stdout(io.StringIO()):
            s_ref._transicion_desde_tierra(forzar=True)
        er = s_ref.estado
        a_ = er.crucero_a
        x_l = (1.0 - mod.RAZON_MASA_TL) * a_
        g = lambda t, y, *ar: mod.R_SOI_LUNA - math.hypot(y[0] - x_l, y[1])
        g.terminal, g.direction = True, 1.0
        rf = _ivp(mod.derivadas_crtbp_rotante, (0.0, 3e5),
                  [er.px, er.py, er.vx, er.vy], args=(er.crucero_omega, a_),
                  method="DOP853", rtol=1e-13, atol=1e-9, events=g)
        t_ref2 = t_c + float(rf.t_events[0][0])
        t_ev2 = reg2[1][1]
        s_pa2, reg_pa2, _, _ = correr_hasta("paso", "luna")
        t_pa2 = reg_pa2[1][1] if len(reg_pa2) > 1 else float("nan")
        print(f"[transiciones_v12] crucero→Luna (SOI): error del instante, evento "
              f"{t_ev2 - t_ref2:+.2e} s, por paso {t_pa2 - t_ref2:+.1f} s")
        if abs(t_ev2 - t_ref2) > 5e-3:
            errores.append(f"crucero->luna: el evento se aparta {t_ev2 - t_ref2:.2e} s de la referencia")
    # 3) tli completo: dos transiciones por evento, misión cumplida; se mide
    #    el salto ECI al entrar a la SOI (informativo: límite del modelo)
    s_t = _sim_marco(mod, modo="tli")
    salto = []
    orig_c = mod.Simulacion._transicion_desde_crucero
    def esp_c(self, destino=None, _o=orig_c):
        r0, v0 = _estado_eci(mod, self); m0 = self.estado.marco
        _o(self, destino) if destino else _o(self)
        if self.estado.marco != m0:
            r1, v1 = _estado_eci(mod, self)
            salto.append((math.hypot(r1[0] - r0[0], r1[1] - r0[1]) / 1e3,
                          math.hypot(v1[0] - v0[0], v1[1] - v0[1])))
    mod.Simulacion._transicion_desde_crucero = esp_c
    with contextlib.redirect_stdout(io.StringIO()):
        for _ in range(120000):
            if s_t.estado.simulacion_terminada:
                break
            s_t.avanzar_paso()
    mod.Simulacion._transicion_desde_crucero = orig_c
    if salto:
        print(f"[transiciones_v12] (informativo) salto ECI al entrar a la SOI en tli: "
              f"{salto[0][0]:.0f} km, {salto[0][1]:.1f} m/s — Luna circular congelada del "
              f"crucero frente a la Luna de las efemérides; igual en 12.1.2")
    print(f"[transiciones_v12] tli: transiciones por evento {s_t.stats_eventos['transicion_marco']}, "
          f"misión cumplida {s_t.eventos.mision_cumplida}, marco {s_t.estado.marco}")
    if (s_t.stats_eventos["transicion_marco"] != 2 or s_t.eventos.mision_cumplida is not True
            or s_t.estado.marco != "luna"):
        errores.append("tli: transiciones o resultado inesperados")
    for x in errores:
        print("   ", x)
    print(f"[transiciones_v12] {'OK' if not errores else 'FALLA'}")
    return not errores



# ---------------------------------------------------------------------------
# 12.3.0 — compatibilidad histórica y modo lunar
# ---------------------------------------------------------------------------
def verificar_compat_v122(resultados: dict) -> bool:
    """Los escenarios de siempre (los siete modos) deben dar EXACTAMENTE lo que
    daba 12.2.0 (rtol 1e-12) si su referencia está junto a este script."""
    global RTOL
    ruta = Path(__file__).with_name("regresion_ref_v12.2.0.json")
    if not ruta.exists():
        print("[compat_v12.2.0] omitido (falta regresion_ref_v12.2.0.json)")
        return True
    ref = json.loads(ruta.read_text(encoding="utf-8"))
    ruta3 = Path(__file__).with_name("regresion_ref_v12.3.0.json")
    ref3 = json.loads(ruta3.read_text(encoding="utf-8")) if ruta3.exists() else {}
    viejo, RTOL = RTOL, 1e-10                # (12.13.8) ver verificar_legacy
    ok, n = True, 0
    try:
        for nombre, res in resultados.items():
            if nombre not in ref:
                continue
            n += 1
            dif = comparar(ref[nombre], json.loads(json.dumps(res)))
            if nombre in ref3:       # 12.6.1 no cambia la física de 12.3.0
                dif += comparar(ref3[nombre], json.loads(json.dumps(res)))
            if dif:
                ok = False
                print(f"[compat_v12.2.0] {nombre}: {len(dif)} diferencias")
                for d in dif[:5]:
                    print("   ", d)
    finally:
        RTOL = viejo
    print(f"[compat_v12.2.0] {'OK' if ok else 'FALLA'} ({n} escenarios idénticos a 12.2.0 y 12.3.0)")
    return ok


def _proveedor_circular(ang0: float = 0.5):
    """Efemérides artificiales deterministas (Luna circular a 384 400 km, Sol
    circular a 1 UA, antihorarias, consultables en cualquier t). NO validan
    SPICE: solo ejercitan la dinámica y la política estricta."""
    def prov(cuerpo, t):
        if cuerpo == "luna":
            R, w, a = 384.4e6, 2 * math.pi / (27.321661 * 86400), ang0
        else:
            R, w, a = 1.496e11, 2 * math.pi / (365.25 * 86400), 1.0
        a = a + w * t
        return ((R * math.cos(a), R * math.sin(a)),
                (-R * w * math.sin(a), R * w * math.cos(a)))
    return prov


def _cfg_lunar(mod, lon=90.0, **extra):
    c = mod.Config()
    c.modo_mision = "lunar"
    c.fecha_lanzamiento = "2026-01-01T00:00:00"
    c.aplicar_preset()
    c.longitud_lanzamiento_ecliptica = lon
    c.efemerides_lunar = "spice"       # (12.5.0) el valor por defecto es horizons_verlet
    c.LUNAR_DIST_OBJETIVO = 15_000_000.0   # (12.5.5) por defecto apunta a un impacto; aquí, sobrevuelo
    for k, v in extra.items():
        setattr(c, k, v)
    return c


def _volar_lunar_tmax_corto(mod, *a, **kw):
    """(12.8.5) Estas pruebas necesitan T_MAX_LUNAR < LUNAR_HORIZONTE_TRANSFERENCIA a
    propósito (ejercitar el fin por T_MAX); la validación nueva lo rechaza, así que
    se desactiva SOLO aquí. La física no cambia."""
    orig = mod.Config._validar_config
    mod.Config._validar_config = lambda self: None
    try:
        return _volar_lunar(mod, *a, **kw)
    finally:
        mod.Config._validar_config = orig


def _volar_lunar(mod, lon=90.0, ang0=0.5, proveedor=None, presupuesto=None,
                 max_pasos=400_000, **extra):
    import contextlib, io
    cfg = _cfg_lunar(mod, lon, **extra)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        sim = mod.Simulacion(cfg, proveedor_efemerides=proveedor or _proveedor_circular(ang0))
        sim.presupuesto_plan_s = presupuesto
        n = 0
        while not sim.estado.simulacion_terminada and n < max_pasos:
            sim.avanzar_paso()
            n += 1
    return sim, n, buf.getvalue()


def verificar_lunar_v13(archivo: str) -> bool:
    """Criterios de aceptación del modo lunar (12.3.0)."""
    mod = cargar_modulo(archivo)
    if not hasattr(mod, "EfemeridesNoValidas"):
        print("[lunar] omitido (el archivo no tiene el modo lunar)")
        return True
    mod._TERCEROS_CUERPOS_FUENTE = "sin_pymcel"
    mod._fetch_horizons_state = lambda *a, **k: (_ for _ in ()).throw(
        RuntimeError("Horizons desactivado en la prueba"))
    if hasattr(mod, "_EPH_DISCO"):
        mod._EPH_DISCO = None
    errores = []

    def chk(cond, msg):
        if not cond:
            errores.append(msg)

    R_T, OM = mod.R_T, mod.OMEGA_T
    fecha = "2026-01-01T00:00:00"

    # 1) Compatibilidad histórica del estado inicial (los 7 modos, con y sin
    #    rotación): (0, R_T) y vx = Ω_ef·R_T, exactos con la longitud por defecto.
    for modo, perfil in (("satelital", "bcb"), ("tripulada", "bcb"), ("suborbital", "bcb"),
                         ("gto", "bcb"), ("tli", "bcb"), ("escape", "bcb"), ("deorbit", "bcb")):
        for rot in (False, True):
            c = mod.Config(); c.modo_mision = modo; c.fecha_lanzamiento = fecha
            c.aplicar_preset(); c.rotacion_terrestre = rot
            if rot:
                c.latitud_lanzamiento = 28.5
            s_ = mod.Simulacion(c)
            e = s_.estado
            vx_esp = c.omega_efectiva() * R_T
            chk(e.px == 0.0 and e.py == R_T and e.vx == vx_esp and e.vy == 0.0,
                f"estado inicial de {modo} (rot={rot}) cambió: {(e.px, e.py, e.vx, e.vy)}")
    # Config: longitud != 90 solo en lunar
    c = mod.Config(); c.modo_mision = "satelital"; c.longitud_lanzamiento_ecliptica = 45.0
    try:
        mod.Simulacion(c); errores.append("longitud 45° en satelital debía rechazarse")
    except ValueError:
        pass
    for mala in (-1.0, 361.0, float("nan")):
        c = _cfg_lunar(mod, mala)
        try:
            mod.Simulacion(c, proveedor_efemerides=_proveedor_circular()); errores.append(f"longitud {mala} debía rechazarse")
        except ValueError:
            pass

    # 2) Geometría: la longitud y la fecha fijan la fase respecto de la Luna
    prov = _proveedor_circular(0.5)
    fases = []
    for lon in (0.0, 45.0, 90.0, 200.0, 333.0):
        c = _cfg_lunar(mod, lon)
        s_ = mod.Simulacion(c, proveedor_efemerides=prov)
        (x0, y0), (vx0, vy0) = s_.estado_inicial_inercial()
        lam = math.radians(lon)
        if lon == 90.0:
            chk((x0, y0) == (0.0, R_T) and vy0 == 0.0, "λ=90°: caso explícito inexacto")
        else:
            chk(abs(x0 - R_T * math.cos(lam)) < 1e-6 and abs(y0 - R_T * math.sin(lam)) < 1e-6,
                f"λ={lon}: posición inercial mal")
        # el estado del marco de lanzamiento (0, R_T, Ω R_T) sale de ese por la isometría
        px, py = s_._a_marco_lanzamiento(x0, y0)
        chk(abs(px) < 1e-6 and abs(py - R_T) < 1e-6, f"λ={lon}: el sitio no queda en +y ({px}, {py})")
        wx, wy = s_._a_marco_lanzamiento(vx0, vy0)
        chk(abs(wx - s_.estado.vx) < 1e-9 and abs(wy) < 1e-9,
            f"λ={lon}: la velocidad de plataforma no es +x ({wx}, {wy})")
        # ida y vuelta
        bx, by = s_._desde_marco_lanzamiento(*s_._a_marco_lanzamiento(1.234e8, -5.6e7))
        chk(abs(bx - 1.234e8) < 1e-4 and abs(by + 5.6e7) < 1e-4, f"λ={lon}: ida y vuelta")
        # ángulo sitio-Luna conservado en módulo; la fase cambia con λ
        (lx, ly), _v = prov("luna", 0.0)
        ang_in = math.atan2(ly, lx) - lam
        mx, my = s_._posicion_cuerpo("luna", 0.0)
        ang_lan = math.atan2(my, mx) - math.pi / 2
        d = (ang_in + ang_lan + math.pi) % (2 * math.pi) - math.pi     # reflexión: opuestos
        chk(abs(d) < 1e-9, f"λ={lon}: la fase sitio-Luna no se conserva ({d})")
        fases.append(round(math.atan2(my, mx), 6))
    chk(len(set(fases)) == len(fases), f"las longitudes deben dar fases distintas: {fases}")

    # 3) Efemérides estrictas: sin respaldo Kepler en silencio
    def debe_fallar(nombre, **kw):
        try:
            mod.Simulacion(_cfg_lunar(mod), **kw)
            errores.append(f"{nombre}: debía lanzar EfemeridesNoValidas")
        except mod.EfemeridesNoValidas:
            pass
        except Exception as e:
            errores.append(f"{nombre}: lanzó {type(e).__name__} en vez de EfemeridesNoValidas")

    debe_fallar("fuente sin consulta arbitraria")                      # sin proveedor ni pymcel
    debe_fallar("proveedor con NaN", proveedor_efemerides=lambda c, t: ((float("nan"), 0.0), (0.0, 0.0)))
    debe_fallar("Luna fuera de rango", proveedor_efemerides=lambda c, t: ((1e7, 0.0), (0.0, 1e3)))
    import random
    rnd = random.Random(1)

    def ruidoso(c, t):
        (x, y), v = _proveedor_circular()(c, t)
        return (x + rnd.random(), y), v
    debe_fallar("proveedor no determinista", proveedor_efemerides=ruidoso)
    mod._TERCEROS_CUERPOS_FUENTE = "pymcel"
    real = mod.pc.consulta_spice
    try:
        mod._EPH_CACHE.clear(); mod._EPH_FALLO.clear()

        def sin_red(*a, **k):
            raise OSError("sin red")
        mod.pc.consulta_spice = sin_red
        try:
            mod.Simulacion(_cfg_lunar(mod))
            errores.append("SPICE caído: el modo lunar debía detenerse (no usar Kepler)")
        except mod.EfemeridesNoValidas as e:
            chk("Kepler" in str(e) or "SPICE" in str(e), "mensaje de SPICE caído poco claro")
        # (12.8.4) el fallo vive en la Simulacion que lo sufrió, no en el proceso
        chk(not mod._EPH_FALLO, f"el fallo de SPICE quedó en el global del proceso: {mod._EPH_FALLO}")
        # SPICE falso pero válido (Luna y Sol circulares, forma (n, 6)) -> arranca,
        # SIN limpiar nada a mano: una misión nueva no hereda el fallo anterior
        mod._EPH_CACHE.clear()
        n, paso = mod._EPH_NODOS_LOTE, mod._EPH_PASO
        base = mod._epoca_a_segundos(fecha)

        def falso(*a, **k):
            cid = k["id"]
            ts = [(j - 2451545.0) * 86400.0 for j in k["epochs"]]
            filas = []
            for t in ts:
                (x, y), (vx, vy) = _proveedor_circular()("luna" if cid == "301" else "sol", t - base)
                filas.append([x, y, 0.0, vx, vy, 0.0])
            return np.array(filas), np.zeros(n), None
        mod.pc.consulta_spice = falso
        try:
            s_ = mod.Simulacion(_cfg_lunar(mod))
            chk(s_._efemerides_modo == "pymcel", "SPICE falso válido: debía usar pymcel")
        except Exception as e:
            errores.append(f"SPICE falso válido: no arrancó ({type(e).__name__}: {e})")
    finally:
        mod.pc.consulta_spice = real
        mod._TERCEROS_CUERPOS_FUENTE = "sin_pymcel"
        mod._EPH_CACHE.clear(); mod._EPH_FALLO.clear()

    # 4) Misiones completas con efemérides artificiales: ascenso -> aparcamiento
    #    verificado -> ventana -> TLI -> SOI como evento exacto
    t_plan_sync = None
    for lon, ang0 in ((90.0, 0.5), (45.0, 2.0), (200.0, 4.0), (90.0, 3.3)):
        sim, n, log = _volar_lunar(mod, lon, ang0)
        ev, d = sim.eventos, sim.eventos.datos_mision
        et = f"λ={lon:g}° fase={ang0}"
        chk(ev.mision_cumplida is True and ev.entro_soi_luna, f"{et}: no entró a la SOI ({d.get('motivo')})")
        if not ev.entro_soi_luna:
            continue
        ap, pl, so = d.get("aparcamiento", {}), d.get("plan_tli", {}), ev.datos_soi_luna
        chk(ap.get("seguro") is True and ap.get("ecc", 1) < 0.01 and ap.get("rp", 0) >= 209e3,
            f"{et}: aparcamiento mal verificado {ap}")
        chk(d.get("t_encendido_real", -1) >= pl.get("t_encendido", 0) - 1e-5 and
            abs(d.get("t_encendido_real", 0) - pl.get("t_encendido", 0)) < 1e-4,
            f"{et}: el encendido no cayó en la hora planificada")
        chk(abs(so["d"] - mod.R_SOI_LUNA) < 1e-3, f"{et}: el cruce no es exacto (d-R_SOI={so['d'] - mod.R_SOI_LUNA})")
        chk(0.0 < so["fraccion_paso"] < 1.0, f"{et}: cruce en el borde del paso")
        chk(abs(so["t"] - pl["t_soi_pred"]) < 120.0,
            f"{et}: SOI real {so['t']:.0f} s vs plan {pl['t_soi_pred']:.0f} s")
        # (12.6.1) el apuntado se ajusta a la distancia mínima REAL de la propagación
        # (d_min_pred); la cónica de 2 cuerpos al entrar a la SOI ya no coincide con 15 000 km
        # exactamente (la marea terrestre la desplaza ~8 %).
        chk(abs(pl["d_min_pred"] - 1.5e7) < 3e4,
            f"{et}: d_min predicha {pl['d_min_pred'] / 1e3:.0f} km (objetivo 15 000 km)")
        chk(abs(so["d_min_dos_cuerpos"] - 1.5e7) < 1.5e6,
            f"{et}: acercamiento 2 cuerpos {so['d_min_dos_cuerpos'] / 1e3:.0f} km (objetivo 15 000 km)")
        chk(pl["variables_ajustadas"] and "t_encendido" in pl["variables_ajustadas"],
            f"{et}: no documenta variables ajustadas")
        chk(pl["dv_disponible"] >= pl["dv_requerido"], f"{et}: Δv")
        chk(mod.R_SOI_LUNA > 0 and math.isfinite(so["v_rel"]), f"{et}: v_rel")
        if (lon, ang0) == (90.0, 0.5):
            t_plan_sync = pl["t_encendido"]
            # reproducible: otra corrida igual da el mismo plan
            sim2, _, _ = _volar_lunar(mod, lon, ang0)
            chk(sim2.eventos.datos_mision["plan_tli"]["t_encendido"] == pl["t_encendido"]
                and sim2.estado.t == sim.estado.t, f"{et}: la misión no es reproducible")
            # SOI contra una propagación independiente (DOP853 estricta) desde el corte
            chk(_soi_independiente(mod, sim, lon, ang0), f"{et}: SOI vs DOP853 independiente")
    # plan por tramos (GUI) == plan de una vez
    sim3, _, _ = _volar_lunar(mod, 90.0, 0.5, presupuesto=0.002)
    chk(t_plan_sync is not None
        and sim3.eventos.datos_mision.get("plan_tli", {}).get("t_encendido") == t_plan_sync,
        "plan por tramos distinto del plan de una vez")
    # camino de eventos="paso" / dinamica congelada también llega
    sim4, _, _ = _volar_lunar(mod, 45.0, 2.0, eventos="paso", dinamica="congelada")
    chk(sim4.eventos.mision_cumplida is True, "eventos='paso'+congelada: no llegó a la SOI")

    # 5) Resultados terminales explícitos
    sim5, _, _ = _volar_lunar_tmax_corto(mod, 90.0, 0.5, T_MAX_LUNAR=1.5 * 86400.0)
    d5 = sim5.eventos.datos_mision
    chk(sim5.eventos.mision_cumplida is False and d5.get("fin_por_t_max")
        and "T_MAX" in d5.get("motivo", "") and sim5.estado.simulacion_terminada,
        f"T_MAX sin SOI debía ser NO CUMPLIDA explícita: {d5.get('motivo')}")
    sim6, _, _ = _volar_lunar(mod, 90.0, 0.5, LUNAR_MARGEN_DV=3.0)
    chk(sim6.eventos.mision_cumplida is False and "Δv insuficiente" in sim6.eventos.datos_mision.get("motivo", ""),
        f"Δv insuficiente debía fallar explícito: {sim6.eventos.datos_mision.get('motivo')}")
    sim7, _, _ = _volar_lunar(mod, 90.0, 3.3, LUNAR_VENTANA_PERIODOS=0.05)
    chk(sim7.eventos.mision_cumplida is False and ("ventana" in sim7.eventos.datos_mision.get("motivo", "")
                                              or "periselene no se alcanz" in sim7.eventos.datos_mision.get("motivo", "")),
        f"sin ventana debía fallar explícito: {sim7.eventos.datos_mision.get('motivo')}")

    if errores:
        print(f"[lunar] FALLA ({len(errores)}):")
        for e in errores:
            print("   ", e)
        return False
    print("[lunar] OK (compatibilidad, geometría, efemérides estrictas, TLI, SOI exacta, "
          "terminales explícitos)")
    return True


def _soi_independiente(mod, sim, lon, ang0) -> bool:
    """Repite el vuelo translunar desde el estado de corte con DOP853
    (rtol 1e-12) y la misma fuerza (gravedad + mareas de Luna y Sol en el
    marco de lanzamiento): el instante de entrada a la SOI debe coincidir."""
    from scipy.integrate import solve_ivp
    ev = sim.eventos
    t_cut = ev.datos_mision["t_cutoff"]
    # (el estado de corte no se guarda: se reconstruye parando la simulación en él)
    import contextlib, io
    cfg = _cfg_lunar(mod, lon)
    with contextlib.redirect_stdout(io.StringIO()):
        s2 = mod.Simulacion(cfg, proveedor_efemerides=_proveedor_circular(ang0))
        while not s2.estado.simulacion_terminada and s2.estado.fase != "translunar":
            s2.avanzar_paso()
    e = s2.estado
    MU_T, MU_L, MU_S = mod.MU_T, float(mod.CUERPOS["luna"].mu), float(mod.MU_SOL)

    def rhs(t, y):
        x, yy, vx, vy = y
        r = math.hypot(x, yy)
        ax, ay = -MU_T * x / r ** 3, -MU_T * yy / r ** 3
        for cuerpo, mu in (("luna", MU_L), ("sol", MU_S)):
            bx, by = s2._posicion_cuerpo(cuerpo, t)
            (tx, ty) = mod.aceleracion_de_marea(mu, (x, yy), (bx, by))
            ax += tx; ay += ty
        return [vx, vy, ax, ay]

    def g(t, y):
        lx, ly = s2._posicion_cuerpo("luna", t)
        return mod.R_SOI_LUNA - math.hypot(y[0] - lx, y[1] - ly)
    g.terminal, g.direction = True, 1.0
    r = solve_ivp(rhs, (e.t, e.t + 8 * 86400.0), [e.px, e.py, e.vx, e.vy],
                  method="DOP853", rtol=1e-12, atol=1e-6, events=g)
    if len(r.t_events[0]) == 0:
        print("   DOP853 no encontró la SOI")
        return False
    dif = abs(float(r.t_events[0][0]) - ev.datos_soi_luna["t"])
    print(f"   [lunar] SOI vs DOP853 independiente: Δt = {dif:.3f} s")
    return dif < 15.0



def verificar_gui_lunar(archivo: str) -> bool:
    """(12.3.1) GUI del modo lunar sin abrir ventana (tkinter simulado): campo
    de longitud, filas del HUD y camino de error si no hay efemérides."""
    mod = cargar_modulo(archivo)
    if not hasattr(mod, "InterfazSimulacion") or not hasattr(mod, "EfemeridesNoValidas"):
        print("[gui_lunar] omitido")
        return True
    mod._TERCEROS_CUERPOS_FUENTE = "sin_pymcel"
    mod._fetch_horizons_state = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("off"))
    if hasattr(mod, "_EPH_DISCO"):
        mod._EPH_DISCO = None
    errores = []

    def chk(c, m):
        if not c:
            errores.append(m)

    I = mod.InterfazSimulacion
    chk(any(k == "longitud_lanzamiento_ecliptica" for k, _ in I.CAMPOS_NUMERICOS),
        "falta el campo de longitud en el panel")
    chk(any(k == "longitud_lanzamiento_ecliptica" for k, _ in I.CONVERSORES_CAMPOS),
        "falta el conversor de la longitud")
    chk(any(k == "lunar" for k, _ in I.MODOS) if hasattr(I, "MODOS") else True, "modo lunar fuera del selector")

    # filas del HUD a lo largo de una misión
    import contextlib, io
    gui = object.__new__(I)
    cfg = _cfg_lunar(mod, 45.0)
    vistos = {}
    with contextlib.redirect_stdout(io.StringIO()):
        sim = mod.Simulacion(cfg, proveedor_efemerides=_proveedor_circular(2.0))
        n = 0
        while not sim.estado.simulacion_terminada and n < 400000:
            sim.avanzar_paso(); n += 1
            if sim.estado.fase not in vistos and sim.estado.fase in ("espera_tli", "quema_tli", "translunar"):
                vistos[sim.estado.fase] = [f[0] for f in gui._filas_estado_general(sim)]
        vistos["fin"] = [f[0] for f in gui._filas_estado_general(sim)]
    chk("Encendido:" in vistos.get("espera_tli", []), f"HUD sin cuenta atrás en espera_tli: {vistos.get('espera_tli')}")
    chk("Aparcam.:" in vistos.get("espera_tli", []) and "SOI prevista:" in vistos.get("espera_tli", []),
        "HUD sin aparcamiento/SOI prevista")
    chk("Dist. Luna:" in vistos.get("translunar", []), "HUD sin distancia a la Luna en vuelo translunar")
    chk("SOI lunar:" in vistos["fin"], f"HUD final sin SOI: {vistos['fin']}")

    # error de efemérides al iniciar: se muestra, no se cae
    from unittest.mock import MagicMock
    gui2 = object.__new__(I)
    gui2.config = _cfg_lunar(mod, 90.0)
    gui2._after_id = None
    gui2.sim = None
    msgs = []
    gui2._leer_config_desde_widgets = lambda: True
    gui2._cancelar_loop = lambda: None
    gui2.etiqueta_error = MagicMock()
    gui2._limpiar_paneles_y_animacion = lambda: None
    gui2._cancelar_planificador = lambda: None
    gui2._mostrar_error = lambda t: msgs.append(t)
    gui2.boton_iniciar = MagicMock()
    gui2.root = MagicMock()
    with contextlib.redirect_stdout(io.StringIO()):
        gui2._iniciar()           # sin proveedor y con fuente sin_pymcel: EfemeridesNoValidas
    chk(msgs and "efemérides" in msgs[0].lower() and gui2.sim is None,
        f"la GUI no mostró el error de efemérides: {msgs}")
    chk(not gui2.root.after.called, "la GUI arrancó el bucle pese al error")
    if errores:
        print(f"[gui_lunar] FALLA ({len(errores)}):")
        for e in errores:
            print("   ", e)
        return False
    print("[gui_lunar] OK (campo, HUD por fases, error de efemérides)")
    return True


def verificar_gui_bucle(archivo: str) -> bool:
    """(12.6.1) Bucle de la GUI sin ventana: un error de dibujo no lo mata, un
    error de física se muestra, el fin del modo lunar se anuncia y el vuelo
    translunar avanza varios pasos por fotograma."""
    mod = cargar_modulo(archivo)
    if not hasattr(mod, "InterfazSimulacion") or not hasattr(mod.Config, "LUNAR_PASOS_MIN_COAST"):
        print("[gui_bucle] omitido")
        return True
    import contextlib, io
    from unittest.mock import MagicMock
    errores = []

    def chk(c, m):
        if not c:
            errores.append(m)
    I = mod.InterfazSimulacion

    def nueva_gui(sim):
        g = object.__new__(I)
        g.sim = sim
        g.config = sim.config
        g.root = MagicMock()
        g.etiqueta_error = MagicMock()
        g.boton_iniciar = MagicMock()
        g.msgs = []
        g._mostrar_error = lambda t: g.msgs.append(t)
        g._after_id = None
        # (12.13.8) El refresco de paneles necesita decenas de widgets que este mock no
        # crea; fallaba con AttributeError (capturado y mostrado como ⚠) sin que la prueba
        # lo ejerciera. Se sustituyen a propósito: lo que se verifica es el estado terminal.
        g._actualizar_panel_sincrono = lambda: None
        g._actualizar_eventos_incrementales = lambda: None
        return g
    with contextlib.redirect_stdout(io.StringIO()):
        sim = mod.Simulacion(_cfg_lunar(mod, 90.0), proveedor_efemerides=_proveedor_circular(0.5))
        while sim.estado.fase != "translunar" and not sim.estado.simulacion_terminada:
            sim.avanzar_paso()
    chk(sim.estado.fase == "translunar", f"no llegó a translunar ({sim.estado.fase})")

    # a) velocidad 1: en translunar avanza LUNAR_PASOS_MIN_COAST pasos por fotograma
    g = nueva_gui(sim)
    g._refrescar_todo = lambda: None
    n = [0]
    orig = sim.avanzar_paso
    sim.avanzar_paso = lambda: (n.__setitem__(0, n[0] + 1), orig())[1]
    sim.config.sim_speed = 1
    g.PRESUPUESTO_FISICA_S = 1e9                     # (12.13.0) sin rebanadas: un lote por fotograma
    g._paso_simulacion()
    chk(n[0] == sim.config.LUNAR_PASOS_MIN_COAST, f"pasos por fotograma en translunar: {n[0]}")
    chk(g.root.after.called, "el bucle no se reprogramó")
    sim.avanzar_paso = orig

    # b) error al dibujar: el bucle sigue y se avisa una sola vez
    g = nueva_gui(sim)
    def malo():
        raise RuntimeError("boom de dibujo")
    g._refrescar_todo = malo
    with contextlib.redirect_stderr(io.StringIO()):
        g._paso_simulacion(); g._paso_simulacion()
    chk(g.root.after.call_count == 2, "un error de dibujo detuvo el bucle")
    chk(len(g.msgs) == 1 and "dibujar" in g.msgs[0], f"aviso de dibujo: {g.msgs}")

    # c) error de física: se detiene y se muestra
    g = nueva_gui(sim)
    g._refrescar_todo = lambda: None
    def peor():
        raise ValueError("boom de física")
    sim.avanzar_paso = peor
    with contextlib.redirect_stderr(io.StringIO()):
        g._paso_simulacion()
    sim.avanzar_paso = orig
    chk(g.msgs and "boom de física" in g.msgs[0], f"error de física no mostrado: {g.msgs}")
    chk(not g.root.after.called, "tras un error de física no debía reprogramarse")

    # d) fin de misión lunar: se anuncia (éxito) y el botón cambia
    with contextlib.redirect_stdout(io.StringIO()):
        while not sim.estado.simulacion_terminada:
            sim.avanzar_paso()
    g = nueva_gui(sim)
    g._refrescar_todo = lambda: None
    g._paso_simulacion()
    textos = [c.kwargs.get("text", "") for c in g.etiqueta_error.config.call_args_list]
    chk(any("SOI" in t for t in textos), f"fin sin anuncio de SOI: {textos}")
    chk("NUEVA SIMULACIÓN" in str(g.boton_iniciar.config.call_args), "el botón no pasó a NUEVA SIMULACIÓN")
    chk(not g.root.after.called, "tras terminar no debía reprogramarse")

    # e) fin por fallo: el motivo se muestra como error
    sim.eventos.entro_soi_luna = False
    sim.eventos.mision_cumplida = False
    sim.eventos.datos_mision["motivo"] = "T_MAX sin SOI"
    g = nueva_gui(sim)
    g._refrescar_todo = lambda: None
    g._paso_simulacion()
    textos = [c.kwargs.get("text", "") for c in g.etiqueta_error.config.call_args_list]
    chk(any("T_MAX sin SOI" in t for t in textos), f"motivo de fallo no mostrado: {textos}")
    if errores:
        print(f"[gui_bucle] FALLA ({len(errores)}):")
        for e in errores:
            print("   ", e)
        return False
    print("[gui_bucle] OK (errores de dibujo y de física, aviso de fin lunar, pasos por fotograma)")
    return True


def verificar_impacto_lunar(archivo: str) -> bool:
    """(12.5.5/12.6.1) La SOI ya no termina la simulación: con el objetivo por
    defecto (200 km bajo la superficie) la nave impacta; con un objetivo de
    sobrevuelo sigue hasta T_MAX_LUNAR sin impacto."""
    mod = cargar_modulo(archivo)
    if not hasattr(mod, "RegistroEventos") \
            or "impacto_lunar_registrado" not in mod.RegistroEventos.__dataclass_fields__:
        print("[lunar_impacto] omitido")
        return True
    mod._TERCEROS_CUERPOS_FUENTE = "sin_pymcel"
    if hasattr(mod, "_EPH_DISCO"):
        mod._EPH_DISCO = None
    errores = []

    def chk(c, m):
        if not c:
            errores.append(m)
    chk(abs(mod.Config().LUNAR_DIST_OBJETIVO - (mod.R_LUNA - 200_000.0)) < 1e-6,
        "el objetivo por defecto debe ser R_LUNA - 200 km")
    # (12.6.1) el apuntado corrige sobre la distancia mínima real: impactan todas.
    for lon, ang0 in ((90.0, 3.3), (90.0, 0.5), (45.0, 2.0), (200.0, 4.0), (90.0, 1.5), (120.0, 5.0)):
        sim, n, log = _volar_lunar(mod, lon, ang0, LUNAR_DIST_OBJETIVO=mod.R_LUNA - 200_000.0)
        ev = sim.eventos
        et = f"impacto λ={lon:g}° fase={ang0}"
        chk(ev.entro_soi_luna and ev.mision_cumplida is True, f"{et}: debía entrar a la SOI (cumplida)")
        chk(ev.impacto_lunar_registrado and sim.estado.simulacion_terminada, f"{et}: sin impacto/terminal")
        if not ev.impacto_lunar_registrado:
            continue
        d = ev.datos_impacto_lunar
        chk(abs(d["d_centro"] - mod.R_LUNA) < 1e-2, f"{et}: contacto no exacto ({d['d_centro'] - mod.R_LUNA:.4f} m)")
        chk(d["t"] > ev.datos_soi_luna["t"], f"{et}: el impacto debe ser posterior a la SOI")
        chk(300.0 < d["v_rel"] < 3000.0, f"{et}: v_rel de impacto inverosímil {d['v_rel']:.0f} m/s")
        chk(ev.datos_mision.get("resultado") == "impacto_lunar", f"{et}: resultado {ev.datos_mision.get('resultado')}")
        chk(sim.estado.t == d["t"], f"{et}: el estado final no es el del contacto")
        pl = ev.datos_mision.get("plan_tli", {})
        chk(pl.get("t_encendido", 0) >= pl.get("t_plan", 1e18) + 1.0 - 1e-9, f"{et}: encendido antes de la planificación")
        chk(abs(ev.datos_mision.get("t_encendido_real", -1) - pl.get("t_encendido", 0)) < 1e-4, f"{et}: encendido real != planificado")
    sim, n, log = _volar_lunar(mod, 90.0, 0.5, LUNAR_DIST_OBJETIVO=15_000_000.0)
    ev = sim.eventos
    chk(ev.entro_soi_luna and not ev.impacto_lunar_registrado, "sobrevuelo: no debía impactar")
    chk(sim.estado.t >= ev.datos_soi_luna["t"] + 3600.0, "sobrevuelo: la propagación no continuó tras la SOI")
    chk(ev.mision_cumplida is True, "sobrevuelo: la misión sigue cumplida (SOI) al llegar a T_MAX")
    chk(0.9 * 15e6 < ev.d_luna_min < 1.1 * 15e6, f"sobrevuelo: d_min {ev.d_luna_min / 1e3:.0f} km")
    if errores:
        print(f"[lunar_impacto] FALLA ({len(errores)}):")
        for e in errores:
            print("   ", e)
        return False
    print("[lunar_impacto] OK (impacto exacto tras la SOI, sobrevuelo sin impacto, propagación continua)")
    return True


class _ProvConZ:
    """Proveedor circular con z de la Luna constante (prueba del aviso 3D)."""
    def __init__(self, z):
        self._p = _proveedor_circular(3.3)
        self._z = z

    def __call__(self, cuerpo, t):
        return self._p(cuerpo, t)

    def z(self, cuerpo, t):
        return self._z


def verificar_auditoria_1259(archivo: str) -> bool:
    """(12.6.1) tolerancia del apuntado registrada/rechazada, z de la Luna y
    marca 2D del impacto, resultados separados."""
    import contextlib, io
    mod = cargar_modulo(archivo)
    if not hasattr(mod, "_nota_3d"):
        print("[auditoria_1259] omitido")
        return True
    mod._TERCEROS_CUERPOS_FUENTE = "sin_pymcel"
    if hasattr(mod, "_EPH_DISCO"):
        mod._EPH_DISCO = None
    errores = []

    def chk(c, m):
        if not c:
            errores.append(m)
    # 1) rechazo por apuntado muy fuera de tolerancia (objetivo inalcanzable)
    sim, n, log = _volar_lunar(mod, 90.0, 0.5, LUNAR_DIST_OBJETIVO=1.0e6)
    ev = sim.eventos
    chk(ev.mision_cumplida is False and "apuntado fallido" in ev.datos_mision.get("motivo", ""),
        f"debía rechazar el apuntado: {ev.datos_mision.get('motivo')}")
    chk(not ev.entro_soi_luna and sim.estado.t < 3 * 86400.0, "rechazo: no debía quemar/volar")
    # 2) plan con tolerancia registrada (sobrevuelo) y resultado separado
    sim, n, log = _volar_lunar(mod, 90.0, 0.5, LUNAR_DIST_OBJETIVO=15_000_000.0)
    ev = sim.eventos
    pl = ev.datos_mision.get("plan_tli", {})
    chk("error_d_min_m" in pl and "tolerancia_cumplida" in pl, "plan sin error_d_min_m/tolerancia_cumplida")
    chk(pl.get("tolerancia_cumplida") is True and pl.get("error_d_min_m", 1e18) <= 30_000.0,
        f"tolerancia: {pl.get('error_d_min_m')}")
    chk(ev.datos_soi_luna.get("z_luna", 0) is None, "sin z() el z_luna debe ser None")
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        mod._imprimir_resumen_lunar(sim)
    t = buf.getvalue()
    chk("sobrevuelo" in t and "impacto tras" not in t, "resumen de sobrevuelo mal etiquetado")
    # 3) impacto con Luna en el plano / fuera del plano
    for z, esperado in ((0.0, True), (500_000.0, False)):
        sim, n, log = _volar_lunar(mod, 90.0, 3.3, proveedor=_ProvConZ(z),
                                   LUNAR_DIST_OBJETIVO=mod.R_LUNA - 200_000.0)
        ev = sim.eventos
        et = f"z={z / 1e3:.0f} km"
        chk(ev.impacto_lunar_registrado, f"{et}: sin impacto")
        if not ev.impacto_lunar_registrado:
            continue
        d = ev.datos_impacto_lunar
        chk(d.get("z_luna") == z and d.get("fisico_3d") is esperado, f"{et}: fisico_3d={d.get('fisico_3d')}")
        chk(ev.datos_soi_luna.get("z_luna") == z, f"{et}: z_luna en la SOI")
        chk(("proyección 2D" in mod._nota_3d(d)) is (not esperado), f"{et}: nota 3D")
        chk(("proyección 2D" in log) is (not esperado), f"{et}: aviso 3D en el log")
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            mod._imprimir_resumen_lunar(sim)
        t = buf.getvalue()
        chk("impacto tras el encuentro" in t, f"{et}: encabezado de impacto")
        chk(("proyección 2D" in t) is (not esperado), f"{et}: resumen 3D")
    # 4) proveedor Horizons+Verlet expone z() y es coherente con la tabla
    if hasattr(mod, "_ProveedorHorizonsVerlet"):
        P = mod._ProveedorHorizonsVerlet
        chk(callable(getattr(P, "z", None)), "_ProveedorHorizonsVerlet sin z()")
    if errores:
        print(f"[auditoria_1259] FALLA ({len(errores)}):")
        for e in errores:
            print("   ", e)
        return False
    print("[auditoria_1259] OK (rechazo/tolerancia del apuntado, z_luna y marca 2D, resultados separados)")
    return True


def verificar_convergencia_lunar(archivo: str) -> bool:
    """(12.6.1) El vuelo translunar converge al refinar el paso (RK4 con Luna y
    Sol evaluados en cada subpaso): instante y velocidad de impacto, d_min y
    entrada a la SOI; el sobrevuelo rasante no registra impacto con ningún paso."""
    mod = cargar_modulo(archivo)
    if not hasattr(mod.Simulacion, "_paso_coast_lunar"):
        print("[convergencia_lunar] omitido")
        return True
    mod._TERCEROS_CUERPOS_FUENTE = "sin_pymcel"
    if hasattr(mod, "_EPH_DISCO"):
        mod._EPH_DISCO = None
    errores = []

    def chk(c, m):
        if not c:
            errores.append(m)
    RL = mod.R_LUNA
    casos = (("frontal", RL - 200e3), ("sobrevuelo_rasante", RL + 20e3), ("sobrevuelo_15000", 15e6))
    for nom, obj in casos:
        res = {}
        for dt in (20.0, 5.0, 1.25):
            sim, n, log = _volar_lunar(mod, 90.0, 3.3, LUNAR_DIST_OBJETIVO=obj, DT_COAST_LUNAR=dt)
            res[dt] = (sim.eventos, sim.estado)
        e0 = {k: v[0] for k, v in res.items()}
        chk(all(e.entro_soi_luna for e in e0.values()), f"{nom}: sin SOI")
        ts = [e.datos_soi_luna["t"] for e in e0.values()]
        chk(max(ts) - min(ts) < 0.1, f"{nom}: t_SOI depende del paso ({max(ts) - min(ts):.3f} s)")
        dm = [e.d_luna_min for e in e0.values()]
        if nom == "frontal":
            chk(all(e.impacto_lunar_registrado for e in e0.values()), f"{nom}: debía impactar con todos los pasos")
            ti = [e.datos_impacto_lunar["t"] for e in e0.values()]
            vi = [e.datos_impacto_lunar["v_rel"] for e in e0.values()]
            chk(max(ti) - min(ti) < 0.1, f"{nom}: t_impacto depende del paso ({max(ti) - min(ti):.3f} s)")
            chk(max(vi) - min(vi) < 0.01, f"{nom}: v_impacto depende del paso ({max(vi) - min(vi):.4f} m/s)")
        else:
            chk(not any(e.impacto_lunar_registrado for e in e0.values()), f"{nom}: impacto falso")
            chk(max(dm) - min(dm) < 100.0, f"{nom}: d_min depende del paso ({max(dm) - min(dm):.1f} m)")
            chk(min(dm) > RL, f"{nom}: d_min por debajo de la superficie sin impacto")
        print(f"   [convergencia_lunar] {nom}: Δt_SOI={max(ts) - min(ts):.3g} s, Δd_min={max(dm) - min(dm):.3g} m")
    if errores:
        print(f"[convergencia_lunar] FALLA ({len(errores)}):")
        for e in errores:
            print("   ", e)
        return False
    print("[convergencia_lunar] OK (SOI, impacto y d_min independientes del paso a <0.1 s / <0.01 m/s / <100 m)")
    return True


def verificar_apuntado_vuelo(archivo: str) -> bool:
    """(12.6.1) El paso REAL por la Luna coincide con el predicho por el plan
    (antes: 1013 km de error cuando el periselene ocurría después del horizonte
    de 6 d del planificador)."""
    import contextlib, io
    mod = cargar_modulo(archivo)
    if "LUNAR_EXTENSION_PERISELENE_S" not in mod.Config.__dataclass_fields__:
        print("[apuntado_vuelo] omitido")
        return True
    mod._TERCEROS_CUERPOS_FUENTE = "sin_pymcel"
    if hasattr(mod, "_EPH_DISCO"):
        mod._EPH_DISCO = None
    errores = []

    def chk(c, m):
        if not c:
            errores.append(m)
    for lon, ang0, obj in ((90.0, 0.5, 15e6), (90.0, 3.3, 5e6), (90.0, 3.3, 15e6), (45.0, 2.0, 8e6)):
        sim, n, log = _volar_lunar(mod, lon, ang0, LUNAR_DIST_OBJETIVO=obj)
        ev = sim.eventos
        pl = ev.datos_mision.get("plan_tli", {})
        et = f"λ={lon:g} fase={ang0} obj={obj / 1e3:.0f} km"
        if not pl:
            chk(False, f"{et}: sin plan ({ev.datos_mision.get('motivo')})")
            continue
        real = ev.d_luna_min
        chk(abs(real - pl["d_min_pred"]) < 20_000.0,
            f"{et}: paso real {real / 1e3:.1f} km vs plan {pl['d_min_pred'] / 1e3:.1f} km")
        chk(abs(real - obj) < 0.01 * obj, f"{et}: paso real {real / 1e3:.1f} km lejos del objetivo")
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            mod._imprimir_resumen_lunar(sim)
        chk("Paso real vs plan" in buf.getvalue(), f"{et}: falta la línea «Paso real vs plan»")
    if errores:
        print(f"[apuntado_vuelo] FALLA ({len(errores)}):")
        for e in errores:
            print("   ", e)
        return False
    print("[apuntado_vuelo] OK (paso real = plan en 4 geometrías de sobrevuelo)")
    return True


def verificar_loi(archivo: str) -> bool:
    """(12.6.1) Captura lunar: con LUNAR_LOI=True la nave queda en una órbita
    selenocéntrica ligada de ~100 km (plano 2D, provisional) y la misión solo se
    cumple al verificarla; sin captura (T_MAX antes del periselene) no cumple;
    la quema converge al refinar el paso; la validación rechaza datos malos."""
    import contextlib, io
    mod = cargar_modulo(archivo)
    if "LUNAR_LOI" not in mod.Config.__dataclass_fields__:
        print("[loi] omitido")
        return True
    mod._TERCEROS_CUERPOS_FUENTE = "sin_pymcel"
    if hasattr(mod, "_EPH_DISCO"):
        mod._EPH_DISCO = None
    errores = []

    def chk(c, m):
        if not c:
            errores.append(m)
    R = mod.R_LUNA
    # 1) varias geometrías: captura circular ~100 km
    sim_cap = None
    for lon, ang0 in ((90.0, 3.3), (90.0, 0.5), (45.0, 2.0), (200.0, 4.0)):
        sim, n, log = _volar_lunar(mod, lon, ang0, LUNAR_LOI=True)
        if sim_cap is None:
            sim_cap = sim
        ev = sim.eventos
        d = ev.datos_mision
        et = f"LOI λ={lon:g} fase={ang0}"
        chk(ev.mision_cumplida is True and d.get("resultado") == "captura_lunar", f"{et}: sin captura ({d.get('motivo')})")
        lo = d.get("loi")
        if not lo:
            chk(False, f"{et}: sin datos de LOI")
            continue
        chk(not ev.impacto_lunar_registrado, f"{et}: impacto")
        chk(abs(lo["rp_min_verif"] - R - 100e3) < 15e3 and abs(lo["ra_max_verif"] - R - 100e3) < 15e3,
            f"{et}: órbita {(lo['rp_min_verif'] - R) / 1e3:.1f}x{(lo['ra_max_verif'] - R) / 1e3:.1f} km")
        chk(lo["ecc"] < 0.02 and lo["eps"] < 0.0, f"{et}: ecc={lo['ecc']:.4f}, eps={lo['eps']:.0f}")
        chk(abs(lo["dv_prop"] - lo["dv_requerido"]) < 0.03 * lo["dv_requerido"],
            f"{et}: Δv {lo['dv_prop']:.0f} vs requerido {lo['dv_requerido']:.0f}")
        chk(lo["comb_restante"] > 0.0 and lo["periodos_verificados"] >= 3.0, f"{et}: combustible/periodos")
        chk(lo["t_cut"] > lo["t_ign"] and d.get("t_encendido_loi") == lo["t_ign"], f"{et}: tiempos")
        # (12.6.2) corte con residuo registrado y convergido; captura provisional 2D
        chk(lo.get("corte_convergido") is True and abs(lo.get("eps_error_corte", 1e9)) <= 1e-8 * 1.5e6,
            f"{et}: corte {lo.get('corte_convergido')} err={lo.get('eps_error_corte')}")
        chk(1 <= lo.get("iter_corte", 99) <= 6, f"{et}: iteraciones del corte {lo.get('iter_corte')}")
        chk(d.get("captura_validada") is False and d.get("modelo_captura") == "2D", f"{et}: captura no marcada provisional 2D")
        chk(lo.get("captura_ligada") is True and lo.get("captura_segura") is True and lo.get("orbita_objetivo") is True
            and abs(lo.get("error_rp_objetivo", 1e9)) < 15e3 and abs(lo.get("error_ra_objetivo", 1e9)) < 15e3,
            f"{et}: niveles de captura {lo.get('captura_ligada')}/{lo.get('captura_segura')}/{lo.get('orbita_objetivo')}")
        chk(lo.get("eps_max_verif", 1) < 0.0 and (lo["T_max_verif"] - lo["T_min_verif"]) < 0.02 * lo["periodo"],
            f"{et}: deriva T {lo.get('T_min_verif')}..{lo.get('T_max_verif')}")
        chk(sim.estado.fase == "orbita_lunar" and sim.estado.simulacion_terminada, f"{et}: fase final {sim.estado.fase}")
        pl = d.get("plan_tli", {})
        vf = pl.get("variables_fijas", {})
        chk(vf.get("loi") is True and pl.get("dv_loi_requerido", 0) > 500.0 and
            abs(vf.get("dist_objetivo_m", 0) - (R + 100e3)) < 1.0, f"{et}: plan LOI")
        # el periselene real queda dentro de la quema (encendido centrado)
        chk(lo["t_cut"] - lo["t_ign"] < 60.0, f"{et}: quema larga {lo['t_cut'] - lo['t_ign']:.0f} s")
    # 1b) GUI (sin ventana): anuncio de la captura y fila del HUD
    if sim_cap is not None and hasattr(mod, "InterfazSimulacion"):
        from unittest.mock import MagicMock
        g = object.__new__(mod.InterfazSimulacion)
        g.sim, g.config = sim_cap, sim_cap.config
        g.root, g.etiqueta_error, g.boton_iniciar = MagicMock(), MagicMock(), MagicMock()
        g._mostrar_error = lambda t: None
        g._after_id = None
        g._refrescar_todo = lambda: None
        with contextlib.redirect_stdout(io.StringIO()):
            g._paso_simulacion()
        textos = [c.kwargs.get("text", "") for c in g.etiqueta_error.config.call_args_list]
        chk(any("CAPTURA LUNAR" in t for t in textos), f"GUI sin anuncio de captura: {textos}")
        filas = g._filas_lunar(sim_cap)
        chk(any(f[0] == "Órbita lunar:" and "CAPTURA" in f[1] for f in filas), f"HUD sin órbita lunar: {filas}")
    # 2) convergencia de la quema con el paso (k = 1, 2, 4)
    orig = mod.Simulacion._ajustar_dt_lunar
    K = {"k": 1.0}

    def aj(self):
        orig(self)
        if self.estado.fase == "quema_loi":
            self.estado.dt = self._dt_base / K["k"]
    mod.Simulacion._ajustar_dt_lunar = aj
    res = []
    try:
        for k in (1.0, 2.0, 4.0):
            K["k"] = k
            sim, n, log = _volar_lunar(mod, 90.0, 3.3, LUNAR_LOI=True)
            lo = sim.eventos.datos_mision.get("loi")
            res.append(None if not lo else (lo["dv_prop"], lo["rp"], lo["ra"], lo["t_cut"]))
    finally:
        mod.Simulacion._ajustar_dt_lunar = orig
    if all(res):
        d_dv = max(abs(r[0] - res[-1][0]) for r in res)
        d_r = max(max(abs(r[1] - res[-1][1]), abs(r[2] - res[-1][2])) for r in res)
        print(f"   [loi] convergencia con el paso: Δ(Δv)<{d_dv:.4f} m/s, Δ(rp,ra)<{d_r:.1f} m")
        chk(d_dv < 0.5 and d_r < 500.0, f"LOI no converge: Δv {d_dv:.3f} m/s, órbita {d_r:.0f} m")
    else:
        chk(False, "convergencia LOI: alguna corrida sin LOI")
    # 3) sin captura: T_MAX justo tras la SOI -> NO CUMPLIDA (el encuentro ya no cumple con LOI)
    sim, n, log = _volar_lunar_tmax_corto(mod, 90.0, 3.3, LUNAR_LOI=True, T_MAX_LUNAR=311_700.0)
    ev = sim.eventos
    chk(ev.entro_soi_luna and ev.mision_cumplida is False
        and "captura" in ev.datos_mision.get("motivo", ""), f"T_MAX sin captura: {ev.mision_cumplida} {ev.datos_mision.get('motivo')}")
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        mod._imprimir_resumen_lunar(sim)
    chk("NO CUMPLIDA" in buf.getvalue(), "resumen sin captura")
    # 3b) (12.6.2) fallos controlados: Δv insuficiente (margen exigido enorme) y
    #     combustible agotado en mitad de la quema
    sim, n, log = _volar_lunar(mod, 90.0, 3.3, LUNAR_LOI=True, LUNAR_MARGEN_DV=50.0)
    ev = sim.eventos
    chk(ev.mision_cumplida is False and ev.datos_mision.get("resultado") != "captura_lunar"
        and sim.estado.simulacion_terminada, f"Δv insuficiente: {ev.mision_cumplida} {ev.datos_mision.get('motivo')}")
    orig_enc = mod.Simulacion._loi_encender

    def enc_poco(self, ctx, info):
        orig_enc(self, ctx, info)
        if self._loi is not None:
            ctx.etapa.m_comb_restante = min(ctx.etapa.m_comb_restante, 400.0)
    mod.Simulacion._loi_encender = enc_poco
    try:
        sim, n, log = _volar_lunar(mod, 90.0, 3.3, LUNAR_LOI=True)
    finally:
        mod.Simulacion._loi_encender = orig_enc
    ev = sim.eventos
    chk(ev.mision_cumplida is False and "combustible" in ev.datos_mision.get("motivo", "")
        and ev.datos_mision.get("resultado") != "captura_lunar",
        f"combustible agotado en la LOI: {ev.mision_cumplida} {ev.datos_mision.get('motivo')}")
    # 3b2) (12.6.3) corte sin converger -> fallo controlado, nunca captura
    it0 = mod.LOI_MAX_ITER_CORTE
    mod.LOI_MAX_ITER_CORTE = 1
    try:
        sim, n, log = _volar_lunar(mod, 90.0, 3.3, LUNAR_LOI=True)
    finally:
        mod.LOI_MAX_ITER_CORTE = it0
    ev = sim.eventos
    chk(ev.mision_cumplida is False and "sin converger" in ev.datos_mision.get("motivo", "")
        and ev.datos_mision.get("resultado") != "captura_lunar",
        f"corte sin converger: {ev.mision_cumplida} {ev.datos_mision.get('motivo')}")
    # 3b3) (12.7.1) ligada y segura pero fuera de la órbita objetivo -> NO CUMPLIDA
    for kw in ({"LUNAR_ORBITA_TOL_ALT": 100.0}, {"LUNAR_ORBITA_ECC_MAX": 1e-6}):
        sim, n, log = _volar_lunar(mod, 90.0, 3.3, LUNAR_LOI=True, **kw)
        ev = sim.eventos
        lo = ev.datos_mision.get("loi", {})
        chk(ev.mision_cumplida is False and ev.datos_mision.get("resultado") == "captura_fuera_de_objetivo"
            and lo.get("captura_ligada") is True and lo.get("captura_segura") is True and lo.get("orbita_objetivo") is False
            and "fuera de la órbita objetivo" in ev.datos_mision.get("motivo", ""),
            f"fuera de objetivo {kw}: {ev.mision_cumplida} {ev.datos_mision.get('resultado')} {ev.datos_mision.get('motivo')}")
    # 3c) (12.6.2) cónica selenocéntrica: estados no válidos lanzan ValueError;
    #     alrededor de ε=0 no hay excepciones ni valores fuera de dominio
    mu = float(mod.CUERPOS["luna"].mu)
    for malo in ((0.0, 0.0, 1.0, 1.0), (float("nan"), 1e6, 1.0, 1.0), (1e6, 1e6, float("inf"), 0.0)):
        try:
            mod._conica_selenocentrica(*malo, mu)
            chk(False, f"conica sin ValueError para {malo}")
        except ValueError:
            pass
    r0 = 5.0e6
    v_esc = math.sqrt(2.0 * mu / r0)
    for fac in (1.0 - 1e-6, 1.0 - 1e-12, 1.0, 1.0 + 1e-12, 1.0 + 1e-6):
        e = mod._conica_selenocentrica(r0, 0.0, 0.0, v_esc * fac, mu)
        chk(all(math.isfinite(e[k]) for k in ("eps", "h", "ecc", "rp", "r", "vr")) and e["ecc"] >= 0.0
            and e["rp"] > 0.0 and (e["eps"] < 0.0) == math.isfinite(e["T"]),
            f"cónica casi parabólica fac={fac}: {e}")
    # 4) validación (matriz: tipos, NaN, inf, cero y negativos)
    for kw in ({"LUNAR_ORBITA_ALT": 5_000.0}, {"LUNAR_PERIODOS_VERIF": 0.0}, {"LUNAR_LOI": 1},
               {"LUNAR_ORBITA_ALT": float("nan")}, {"LUNAR_ORBITA_ALT": float("inf")},
               {"LUNAR_ORBITA_ALT": -1.0}, {"LUNAR_ORBITA_ALT": "100000"},
               {"LUNAR_PERIODOS_VERIF": float("nan")}, {"LUNAR_PERIODOS_VERIF": -3.0},
               {"LUNAR_PERIODOS_VERIF": "3"}, {"LUNAR_LOI": "si"},
               {"LUNAR_ORBITA_TOL_ALT": 0.0}, {"LUNAR_ORBITA_TOL_ALT": float("nan")},
               {"LUNAR_ORBITA_ECC_MAX": 1.0}, {"LUNAR_ORBITA_ECC_MAX": float("nan")},
               {"LUNAR_ORBITA_ECC_MAX": -0.1}):
        try:
            c = _cfg_lunar(mod, LUNAR_LOI=kw.get("LUNAR_LOI", True), **{k: v for k, v in kw.items() if k != "LUNAR_LOI"})
            c._validar_config()
            chk(False, f"la validación debía rechazar {kw}")
        except Exception:
            pass
    # 5) (12.6.2) GUI: LUNAR_LOI y la altitud de la órbita son configurables
    I = getattr(mod, "InterfazSimulacion", None)
    if I is not None and hasattr(I, "CONVERSORES_CAMPOS"):
        claves = {c for c, _ in I.CAMPOS_NUMERICOS}
        conv = dict(I.CONVERSORES_CAMPOS)
        chk(claves == set(conv), f"campos sin conversor: {claves ^ set(conv)}")
        chk("LUNAR_ORBITA_ALT" in conv and conv["LUNAR_ORBITA_ALT"]("100") == 100_000.0, "campo de altitud LOI")
        import inspect
        fuente = inspect.getsource(I)
        chk("var_loi" in fuente and "cfg.LUNAR_LOI" in fuente, "falta la casilla LOI en la GUI")
    if errores:
        print(f"[loi] FALLA ({len(errores)}):")
        for e in errores:
            print("   ", e)
        return False
    print("[loi] OK (captura ~100 km en 4 geometrías, Δv=requerido, corte convergido, convergencia con el paso, "
          "fallos controlados, sin captura = NO CUMPLIDA)")
    return True


def verificar_ventanas(archivo: str) -> bool:
    """(12.7.0) Planificador de ventanas: efemérides de Standish, Lambert contra la
    propagación de dos cuerpos y contra Hohmann, ventanas Tierra-Marte conocidas,
    Tierra-Venus, restricciones, parseo de campos y el diálogo de la GUI (con mocks)."""
    import math
    import numpy as np
    from scipy.integrate import solve_ivp
    mod = cargar_modulo(archivo)
    if not hasattr(mod, "PlanificadorVentanas"):
        print("[ventanas] omitido")
        return True
    errores = []

    def chk(c, m):
        if not c:
            errores.append(m)
    MU, UA, DIA, J0 = mod.MU_SOL, mod._UA, mod._DIA, mod._JD_J2000
    jd = mod._jd_de_fecha
    eph = mod.EfemeridesPlanetas()
    # 1) efemérides: Tierra en el perihelio; rechazos
    r, v = eph.estado("tierra", jd("2000-01-03"))
    chk(abs(np.linalg.norm(r) / UA - 0.9833) < 2e-3 and abs(np.linalg.norm(v) - 30.29e3) < 100.0, "Tierra al perihelio")
    for malo in ("1700-01-01", "2100-01-01"):
        try:
            eph.estado("marte", jd(malo))
            chk(False, f"debía rechazar {malo}")
        except ValueError:
            pass
    for o, d in (("tierra", "luna"), ("marte", "marte")):
        try:
            mod.PlanificadorVentanas(o, d)
            chk(False, f"{o}->{d} debía fallar")
        except ValueError:
            pass

    # 2) Lambert contra la propagación de dos cuerpos
    def propaga(r1, v1, dt):
        f = lambda t, y: np.concatenate([y[3:], -MU * y[:3] / np.linalg.norm(y[:3]) ** 3])
        sol = solve_ivp(f, (0, dt), np.concatenate([r1, v1]), method="DOP853", rtol=1e-12, atol=1e-3)
        return sol.y[:3, -1], sol.y[3:, -1]
    peor = 0.0
    for o, d, f0, tof in (("tierra", "marte", "2020-07-20", 200), ("tierra", "venus", "2023-06-06", 128),
                          ("tierra", "jupiter", "2023-01-01", 800), ("marte", "tierra", "2021-03-01", 250)):
        r1, _ = eph.estado(o, jd(f0) )
        r2, _ = eph.estado(d, jd(f0) + tof)
        v1, v2, ok = mod._lambert(r1, r2, tof * DIA, MU)
        chk(ok, f"Lambert {o}->{d}")
        if ok:
            rf, vf = propaga(r1, v1, tof * DIA)
            peor = max(peor, np.linalg.norm(rf - r2) / np.linalg.norm(r2), np.linalg.norm(vf - v2) / np.linalg.norm(v2))
    chk(peor < 1e-7, f"Lambert vs propagación: {peor:.1e}")
    # 3) Hohmann con planetas circulares coplanares
    a1, a2 = 1.0 * UA, 1.5 * UA
    vinf_h = math.sqrt(MU / a1) * (math.sqrt(2 * a2 / (a1 + a2)) - 1.0)
    t_h = math.pi * math.sqrt((0.5 * (a1 + a2)) ** 3 / MU)
    ang0 = (math.pi - math.radians(0.05)) - math.sqrt(MU / a2 ** 3) * t_h

    class Circ:
        def estado(self, c, jd_):
            t = (jd_ - J0) * DIA
            a = a1 if c == "tierra" else a2
            w = math.sqrt(MU / a ** 3)
            th = w * t + (ang0 if c == "marte" else 0.0)
            return (np.array([a * math.cos(th), a * math.sin(th), 0.0]),
                    np.array([-a * w * math.sin(th), a * w * math.cos(th), 0.0]))
    c = Circ()
    r1, v1o = c.estado("tierra", J0)
    r2, v2d = c.estado("marte", J0 + t_h / DIA)
    v1, v2, ok = mod._lambert(r1, r2, t_h, MU)
    chk(ok and abs(np.linalg.norm(v1 - v1o) - vinf_h) / vinf_h < 2e-3, "Hohmann")
    _, _, ok = mod._lambert(np.array([UA, 0, 0.0]), np.array([-1.5 * UA, 0, 0.0]), t_h, MU)
    chk(not ok, "180° exactos debía dar ok=False")
    # 4) Tierra-Marte 2014-2027: ventanas conocidas
    plan, vm = mod.calcular_ventanas("tierra", "marte", "2014-01-01", "2027-12-31")
    conocidas = ["2016-03-15", "2018-05-10", "2020-07-20", "2022-09-01", "2024-10-15", "2026-11-15"]
    chk(len(vm) == len(conocidas), f"Marte: {len(vm)} ventanas")
    for w, k in zip(vm, conocidas):
        chk(abs(w.salida_jd - jd(k)) < 75.0, f"Marte: {mod.VentanaLanzamiento.fecha(w.salida_jd)} lejos de {k}")
        chk(7.0 < w.c3_km2s2 < 18.0 and 150 < w.tof_dias < 400, f"Marte {k}: C3 {w.c3_km2s2:.1f}, TOF {w.tof_dias:.0f}")
    if len(vm) == len(conocidas):
        chk(abs(vm[1].c3_km2s2 - 8.0) < 2.0 and abs(vm[2].c3_km2s2 - 14.0) < 2.0, "C3 2018/2020")
        chk(abs(plan.periodo_sinodico() / DIA - 779.9) < 1.5, "periodo sinódico")
    # 5) restricciones
    _, vr = mod.calcular_ventanas("tierra", "marte", "2014-01-01", "2027-12-31", c3_max_km2s2=10.0)
    chk(0 < len(vr) < len(vm) and all(w.c3_km2s2 <= 10.0 + 1e-9 for w in vr), f"C3<=10: {len(vr)}")
    _, vd = mod.calcular_ventanas("tierra", "marte", "2014-01-01", "2027-12-31", dv_salida_max_kms=3.7)
    chk(0 < len(vd) < len(vm) and all(w.dv_salida <= 3700.0 + 1e-6 for w in vd), f"Δv sal<=3.7: {len(vd)}")
    _, ve = mod.calcular_ventanas("tierra", "marte", "2014-01-01", "2027-12-31", c3_max_km2s2=1.0)
    chk(ve == [], "restricción imposible -> lista vacía")
    try:
        mod.calcular_ventanas("tierra", "marte", "2027-01-01", "2026-01-01")
        chk(False, "fechas invertidas debían fallar")
    except ValueError:
        pass
    # 6) Tierra-Venus
    _, vv = mod.calcular_ventanas("tierra", "venus", "2016-01-01", "2025-12-31", criterio="c3")
    sep = np.diff([w.salida_jd for w in vv])
    chk(len(vv) >= 5 and abs(sep.mean() - 583.9) < 40.0 and all(5.0 < w.c3_km2s2 < 20.0 for w in vv), "Venus")
    # 7) tabla, figura y parseo de campos
    tabla = mod.texto_tabla_ventanas(vm)
    chk(tabla.count("\n") == len(vm) and "2026-1" in tabla, "tabla de texto")
    fig = mod.figura_porkchop(plan, vm[2])
    chk(len(fig.axes) >= 1, "figura de porkchop")
    base = dict(origen="tierra", destino="marte", inicio="2026-01-01", fin="2029-01-01", c3_max="", dv_salida_max="3,8",
                vinf_llegada_max="", criterio="c3", capturar=True)
    kw = mod.parsear_campos_ventanas(base)
    chk(math.isinf(kw["c3_max_km2s2"]) and kw["dv_salida_max_kms"] == 3.8 and kw["criterio"] == "c3", f"parseo: {kw}")
    for malo in ({"inicio": "26/01/2026"}, {"c3_max": "abc"}, {"c3_max": "-5"}, {"dv_salida_max": "nan"}, {"fin": ""}):
        try:
            mod.parsear_campos_ventanas({**base, **malo})
            chk(False, f"parseo debía rechazar {malo}")
        except ValueError:
            pass
    # 8) diálogo de la GUI con mocks: calcula en un hilo y lista las ventanas
    I = getattr(mod, "InterfazSimulacion", None)
    if I is not None and hasattr(I, "_abrir_ventanas"):
        from unittest.mock import MagicMock
        import time

        class Var:
            def __init__(self, value=""):
                self.v = value

            def get(self):
                return self.v

            def set(self, x):
                self.v = x
        viejo_tk, viejo_cv = mod.tk, mod.FigureCanvasTkAgg
        mod.tk = MagicMock()
        mod.tk.StringVar = Var
        mod.tk.BooleanVar = Var
        mod.FigureCanvasTkAgg = MagicMock()
        try:
            g = object.__new__(I)
            g.root = MagicMock()
            g.entradas = {"fecha_lanzamiento": Var("2026-01-01T00:00:00")}
            g._abrir_ventanas()
            top = mod.tk.Toplevel.return_value
            botones = {c.kwargs.get("text"): c.kwargs.get("command") for c in mod.tk.Button.call_args_list}
            chk({"CALCULAR", "PORKCHOP", "CERRAR"} <= set(botones), f"botones del diálogo: {list(botones)}")
            lista = mod.tk.Listbox.return_value
            botones["CALCULAR"]()
            for _ in range(600):
                if lista.insert.called:
                    break
                for c in list(top.after.call_args_list):
                    pass
                top.after.call_args_list[-1].args[1]()
                time.sleep(0.1)
            filas = [c.args[1] for c in lista.insert.call_args_list]
            chk(len(filas) >= 2 and filas[0].lstrip().startswith("#") and "2026-1" in "\n".join(filas), f"filas del diálogo: {filas[:3]}")
            lista.curselection.return_value = (1,)
            botones["PORKCHOP"]()
            chk(mod.FigureCanvasTkAgg.called, "el porkchop no se dibujó")
        finally:
            mod.tk, mod.FigureCanvasTkAgg = viejo_tk, viejo_cv
    # (12.7.1) Lambert contra un caso publicado (Curtis, ej. 5.2: r en km, 1 h, μ de la Tierra)
    v1c, v2c, okc = mod._lambert(np.array([5000e3, 10000e3, 2100e3]), np.array([-14600e3, 2500e3, 7000e3]),
                                 3600.0, 3.986004418e14)
    chk(okc and np.allclose(v1c, [-5992.5, 1925.4, 3245.6], atol=2.0)
        and np.allclose(v2c, [-3312.5, -4196.6, -385.29], atol=2.0), f"Lambert Curtis 5.2: {v1c}, {v2c}")
    # (12.7.1) fechas con zona horaria: se convierten a UTC, no se reinterpretan
    chk(abs(jd("2026-01-01T00:00:00-05:00") - jd("2026-01-01T05:00:00")) < 1e-9
        and abs(jd("2026-01-01T00:00:00+00:00") - jd("2026-01-01")) < 1e-9
        and abs(jd("2026-01-01T12:00:00Z") - (jd("2026-01-01") + 0.5)) < 1e-9, "fecha con zona horaria")
    # (12.7.1) la malla no manda: ventanas con pasos 4, 2 y 1 d (refinadas) coinciden;
    # sin refinar la fecha es múltiplo del paso y el refinado la mejora o la iguala
    res_paso = {}
    for paso in (4.0, 2.0, 1.0):
        pl_p = mod.PlanificadorVentanas("tierra", "marte")
        pl_p.calcular(jd("2020-01-01"), jd("2021-01-01"), paso)
        res_paso[paso] = pl_p.ventanas("dv_total")[0]
        if paso == 4.0:
            gruesa = pl_p.ventanas("dv_total", refinar=False)[0]
            chk(res_paso[paso].dv_total <= gruesa.dv_total + 1e-6, "el refinado empeoró el coste")
    dd = max(abs(a.salida_jd - res_paso[1.0].salida_jd) for a in res_paso.values())
    dc = max(abs(a.dv_total - res_paso[1.0].dv_total) for a in res_paso.values())
    chk(dd < 3.0 and dc < 20.0, f"ventana depende del paso de la malla: {dd:.2f} d, {dc:.1f} m/s")
    print(f"   [ventanas] pasos 4/2/1 d: fecha ±{dd:.2f} d, Δv ±{dc:.1f} m/s")
    # (12.7.1) límites de TOF, rango de efemérides y porkchop sin celdas válidas
    pl_t, vt = mod.calcular_ventanas("tierra", "marte", "2018-01-01", "2019-01-01", tof_min_dias=150, tof_max_dias=250)
    chk(len(vt) == 1 and 150 <= vt[0].tof_dias <= 250, f"TOF acotado: {[w.tof_dias for w in vt]}")
    pr = mod.parsear_campos_ventanas({"origen": "tierra", "destino": "marte", "inicio": "2026-01-01", "fin": "2027-01-01",
                                      "tof_min": "150", "tof_max": ""})
    chk(pr["tof_min_dias"] == 150.0 and pr["tof_max_dias"] is None, f"parseo TOF: {pr}")
    try:
        mod.calcular_ventanas("tierra", "marte", "2049-06-01", "2049-12-31")
        chk(False, "fuera del rango de efemérides debía fallar antes de calcular")
    except ValueError as e:
        chk("efemérides" in str(e), f"mensaje de rango: {e}")
    try:
        mod.calcular_ventanas("tierra", "marte", "2018-01-01", "2019-01-01", tof_min_dias=300, tof_max_dias=200)
        chk(False, "TOF mín > máx debía fallar")
    except ValueError:
        pass
    pl_v, vv0 = mod.calcular_ventanas("tierra", "marte", "2018-01-01", "2019-01-01")
    pl_v.malla["valido"][:] = False
    try:
        mod.figura_porkchop(pl_v, vv0[0])
        chk(False, "porkchop sin celdas válidas debía fallar")
    except ValueError:
        pass
    # (12.7.2) el refinamiento no sale del intervalo pedido (en 12.7.1 con un solo día de salida y el
    # óptimo fuera del intervalo la ventana salía hasta 2 d)
    for ini, fin in (("2018-05-01", "2018-05-20"), ("2018-05-09", "2018-05-10"), ("2018-05-13", "2018-05-14")):
        pl_e, ve = mod.calcular_ventanas("tierra", "marte", ini, fin)
        j0, j1 = mod._jd_de_fecha(ini), mod._jd_de_fecha(fin)
        chk(all(j0 - 1e-9 <= w.salida_jd <= j1 + 1e-9 for w in ve), f"refinado fuera de [{ini},{fin}]: {[w.fecha(w.salida_jd) for w in ve]}")
        chk(all(pl_e._lim_arr[0] <= w.llegada_jd <= pl_e._lim_arr[1] for w in ve), "llegada fuera de efemérides")
    # (12.8.1) `refinada` es True solo si de verdad se refinó
    # (12.8.3) la prueba anterior comparaba contra una lista posiblemente vacía y pasó con el defecto
    pl_g, vg = mod.calcular_ventanas("tierra", "marte", "2018-01-01", "2019-01-01")
    gruesas = pl_g.ventanas(refinar=False)
    chk(len(gruesas) >= 1 and all(not w.refinada for w in gruesas), f"ventanas(refinar=False) refinada={[w.refinada for w in gruesas]}")
    chk(len(vg) >= 1 and all(w.refinada for w in vg), "ventanas refinadas deben marcarse refinadas")
    chk(pl_g._ventana_de(pl_g.malla, 0, 0, 1, refinada=False).refinada is False
        and pl_g._ventana_de(pl_g.malla, 0, 0, 1).refinada is True, "_ventana_de ignora `refinada`")
    # (12.7.3) catálogo derivado de CUERPOS, refinamiento marcado, límites validados
    for n, c in mod._CAT_VENTANAS.items():
        chk(c["mu"] == mod.CUERPOS[n].mu and c["radio"] == mod.CUERPOS[n].radio, f"_CAT_VENTANAS[{n}] no sale de CUERPOS")
    chk(all(w.refinada for w in vv0) and "*" not in mod.texto_tabla_ventanas(vv0), "ventanas nominales sin marcar como refinadas")
    import dataclasses
    chk(mod.texto_tabla_ventanas([dataclasses.replace(vv0[0], refinada=False)]).count("*") == 2, "tabla no marca la malla gruesa")
    chk(abs(mod._periodo_orbital(mod._UA) / 86400 - 365.25) < 0.1, "_periodo_orbital de 1 UA")
    for kw in ({"c3_max_km2s2": float("nan")}, {"c3_max_km2s2": 0.0}, {"dv_salida_max_kms": -1.0},
               {"vinf_llegada_max_kms": float("nan")}):
        try:
            mod.calcular_ventanas("tierra", "marte", "2018-01-01", "2019-01-01", **kw)
            chk(False, f"debía rechazar {kw}")
        except ValueError:
            pass
    # (12.7.2) validación de argumentos públicos
    for kw, que in (({"paso_dias": 0.0}, "paso 0"), ({"paso_dias": float("nan")}, "paso nan"), ({"paso_dias": -1.0}, "paso <0"),
                    ({"criterio": "foo"}, "criterio"), ({"tof_min_dias": float("nan")}, "tof nan"),
                    ({"tof_max_dias": -5.0}, "tof<0"), ({"tof_min_dias": 0.0, "tof_max_dias": 100.0}, "tof 0")):
        try:
            mod.calcular_ventanas("tierra", "marte", "2018-01-01", "2019-01-01", **kw)
            chk(False, f"debía rechazar {que}")
        except ValueError:
            pass
    pl_c, vc = mod.calcular_ventanas("tierra", "marte", "2018-01-01", "2019-01-01", criterio="c3")
    fig_c = mod.figura_porkchop(pl_c, vc[0])
    chk("mín. C3" in fig_c.axes[0].get_legend().get_texts()[0].get_text(), "etiqueta del porkchop con criterio c3")
    fig_d = mod.figura_porkchop(pl_v if False else mod.calcular_ventanas("tierra", "marte", "2018-01-01", "2019-01-01")[0], vv0[0])
    chk("mín. Δv" in fig_d.axes[0].get_legend().get_texts()[0].get_text(), "etiqueta del porkchop con dv_total")
    # consola: --ventanas (códigos de salida 0 con ventanas, 1 sin ventanas, 2 error)
    import contextlib, io
    if hasattr(mod, "_main_ventanas"):
        for args, rc in ((["tierra", "marte", "2024-01-01", "2027-12-31", "--c3-max", "12"], 0),
                         (["tierra", "marte", "2026-01-01", "2027-06-01", "--c3-max", "1"], 1),
                         (["tierra", "marte", "2027-01-01", "2026-01-01"], 2)):
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                got = mod._main_ventanas(args)
            chk(got == rc, f"--ventanas {args}: código {got}, esperado {rc}")
            if rc == 0:
                chk("2026-10" in buf.getvalue() and ("2024-09" in buf.getvalue() or "2024-10" in buf.getvalue()),
                    f"tabla CLI: {buf.getvalue()[:200]}")
        try:
            with contextlib.redirect_stderr(io.StringIO()):
                mod._main_ventanas(["tierra", "luna", "2026-01-01", "2027-01-01"])
            chk(False, "destino luna debía rechazarse")
        except SystemExit as e:
            chk(e.code == 2, "destino luna: código de salida")
    if errores:
        print(f"[ventanas] FALLA ({len(errores)}):")
        for e in errores:
            print("   ", e)
        return False
    print(f"[ventanas] OK (Lambert {peor:.0e}, Hohmann, Marte {len(vm)} ventanas C3 "
          f"{min(w.c3_km2s2 for w in vm):.1f}-{max(w.c3_km2s2 for w in vm):.1f}, Venus {len(vv)}, restricciones, GUI, consola)")
    return True


def verificar_config_congelada(archivo: str) -> bool:
    """(12.7.1) La GUI inicia la simulación con una COPIA de la configuración y no
    cambia el modo en pleno vuelo (antes la simulación compartía el objeto con los
    widgets y `_cambiar_modo` aplicaba presets sobre la misión en curso)."""
    mod = cargar_modulo(archivo)
    I = getattr(mod, "InterfazSimulacion", None)
    if I is None or not hasattr(I, "_mision_activa"):
        print("[config_congelada] omitido")
        return True
    from unittest.mock import MagicMock
    import contextlib, io
    errores = []

    def chk(c, m):
        if not c:
            errores.append(m)
    mod._TERCEROS_CUERPOS_FUENTE = "sin_pymcel"
    g = object.__new__(I)
    g.root, g.etiqueta_error, g.boton_iniciar = MagicMock(), MagicMock(), MagicMock()
    g.config = mod.Config()
    g.config.modo_mision = "satelital"
    g.sim, g._after_id, g._planificador = None, None, None
    g._leer_config_desde_widgets = lambda: True
    g._cancelar_loop = g._limpiar_paneles_y_animacion = g._cancelar_planificador = lambda: None
    g._instalar_planificador = lambda cfg: None
    g._mostrar_error = lambda t: None
    g.var_modo = MagicMock()
    g._refrescar_campos_del_preset = g._actualizar_visibilidad_perfil = g._actualizar_visibilidad_cutoff_subor = lambda: None
    msgs = []
    g._mostrar_mensaje_temporal = lambda t, c, d=1500: msgs.append(t)
    with contextlib.redirect_stdout(io.StringIO()):
        g._iniciar()
    chk(g.sim is not None, "no se creó la simulación")
    if g.sim is not None:
        chk(g.sim.config is not g.config, "la simulación comparte la configuración con la GUI")
        chk(g.sim.config.modo_mision == "satelital" and g.sim.config.sim_speed == g.config.sim_speed, "la copia difiere")
        chk(g._mision_activa(), "misión activa")
        # el usuario toca el selector en pleno vuelo
        g.var_modo.get.return_value = "escape"
        g._cambiar_modo()
        chk(g.sim.config.modo_mision == "satelital" and g.config.modo_mision == "satelital",
            f"el modo cambió en vuelo: sim={g.sim.config.modo_mision} gui={g.config.modo_mision}")
        chk(g.var_modo.set.call_args is not None and g.var_modo.set.call_args.args == ("satelital",), "el selector no se revirtió")
        chk(msgs and "REINICIAR" in msgs[-1], f"sin aviso: {msgs}")
        # los cambios de la GUI no tocan la misión
        g.config.sim_speed = 99
        chk(g.sim.config.sim_speed != 99, "cambiar la config de la GUI cambió la misión")
        # terminada la misión, sí se puede cambiar de modo
        g.sim.estado.simulacion_terminada = True
        g._cambiar_modo()
        chk(g.config.modo_mision == "escape", "con la misión terminada el modo debía cambiar")
    if errores:
        print(f"[config_congelada] FALLA ({len(errores)}):")
        for e in errores:
            print("   ", e)
        return False
    print("[config_congelada] OK (copia de la configuración, selector de modo bloqueado en vuelo)")
    return True


def verificar_tarjetas_gui(archivo: str) -> bool:
    """(12.8.0) Barra superior (MET, fase, estado) y tarjetas de telemetría: funciones
    puras y lógica de actualización con un tkinter de mentira que registra los textos."""
    mod = cargar_modulo(archivo)
    I = getattr(mod, "InterfazSimulacion", None)
    if I is None or not hasattr(mod, "_agrupar_filas_en_tarjetas"):
        print("[tarjetas_gui] omitido")
        return True
    from types import SimpleNamespace as NS
    import contextlib, io
    errores = []

    def chk(c, m):
        if not c:
            errores.append(m)
    # MET
    for t, esp in ((0, "+00:00:00"), (59.9, "+00:00:59"), (3661, "+01:01:01"), (360000, "+100:00:00"),
                   (-5, "+00:00:00"), (float("nan"), "+00:00:00"), (float("inf"), "+00:00:00")):
        chk(mod._formatear_met(t) == esp, f"MET({t}) = {mod._formatear_met(t)} != {esp}")
    # estado global
    def sim_falso(cumplida=None, terminada=False, plan=False, motivo=None):
        dm = {} if motivo is None else {"motivo": motivo}
        return NS(eventos=NS(mision_cumplida=cumplida, datos_mision=dm),
                  estado=NS(simulacion_terminada=terminada), esperando_plan=plan)
    chk(mod._estado_global(sim_falso())[0] == "EN CURSO", "estado EN CURSO")
    chk(mod._estado_global(sim_falso(plan=True))[0] == "PLANIFICANDO", "estado PLANIFICANDO")
    chk(mod._estado_global(sim_falso(terminada=True))[0] == "FINALIZADA", "estado FINALIZADA")
    chk(mod._estado_global(sim_falso(cumplida=True))[0] == "MISIÓN CUMPLIDA · SEGUIMIENTO", "cumplida con la simulación en marcha")
    chk(mod._estado_global(sim_falso(cumplida=True, terminada=True))[0] == "MISIÓN CUMPLIDA", "estado CUMPLIDA")
    t, c, d = mod._estado_global(sim_falso(cumplida=False, terminada=True, motivo="x" * 100))
    chk(t == "NO CUMPLIDA" and c == mod.COLOR_TEXTO_ERROR and len(d) <= 60 and d.endswith("…"), f"estado NO CUMPLIDA {t} {len(d)}")
    # agrupación
    filas = [("Tiempo:", "1 s", "normal"), ("Fase:", "vertical", "fase"), ("Altitud:", "1 km", "normal"),
             ("Empuje:", "9 kN", "empuje_on"), ("Aparcam.:", "x", "normal"), ("Dist. Luna:", "y", "normal")]
    g = dict(mod._agrupar_filas_en_tarjetas(filas))
    chk(list(g) == ["TRAYECTORIA", "PROPULSIÓN", "MISIÓN"], f"tarjetas {list(g)}")
    chk([f[0] for f in g["TRAYECTORIA"]] == ["Altitud:"] and [f[0] for f in g["PROPULSIÓN"]] == ["Empuje:"]
        and [f[0] for f in g["MISIÓN"]] == ["Aparcam.:", "Dist. Luna:"], f"reparto {g}")
    chk(all(f[0] not in ("Tiempo:", "Fase:") for v in g.values() for f in v), "Tiempo/Fase deben ir solo en la barra")
    chk(all(len(v) == 0 for _, v in mod._agrupar_filas_en_tarjetas([])), "sin filas = tarjetas vacías")
    dup = dict(mod._agrupar_filas_en_tarjetas([("Dist. Luna:", "1", "normal"), ("Altitud:", "2", "normal"),
                                               ("Dist. Luna:", "3", "normal")]))
    chk(dup["MISIÓN"] == [("Dist. Luna:", "1", "normal")], f"fila repetida no unificada: {dup['MISIÓN']}")
    # cada fila que produce el HUD cae en alguna tarjeta (no se pierde ninguna)
    gui = object.__new__(I)
    with contextlib.redirect_stdout(io.StringIO()):
        sim = mod.Simulacion(mod.Config())
        for _ in range(200):
            sim.avanzar_paso()
        filas_reales = gui._filas_estado_general(sim) + gui._filas_carga_g_y_orbita(sim)
    ecc_sat = [f[1] for f in filas_reales if f[0] == "Excentric.:"]
    chk(ecc_sat and "≤" not in ecc_sat[0], f"satelital: umbral de excentricidad sin sentido: {ecc_sat}")
    cfg_t = mod.Config()
    cfg_t.modo_mision = "tripulada"
    cfg_t.aplicar_preset()
    with contextlib.redirect_stdout(io.StringIO()):
        sim_t = mod.Simulacion(cfg_t)
        for _ in range(200):
            sim_t.avanzar_paso()
        ecc_trip = [f[1] for f in gui._filas_carga_g_y_orbita(sim_t) if f[0] == "Excentric.:"]
    chk(ecc_trip and "≤" in ecc_trip[0], f"tripulada: debe mostrar el umbral: {ecc_trip}")
    en_tarjetas = [f for _, v in mod._agrupar_filas_en_tarjetas(filas_reales) for f in v]
    chk(len(en_tarjetas) == len([f for f in filas_reales if f[0] not in mod.FILAS_EN_BARRA]), "se perdieron filas del HUD")

    # lógica de widgets con un tk de mentira
    class W:
        def __init__(self, *a, **k):
            self.opts = dict(k); self.empaquetado = False
        def config(self, **k): self.opts.update(k)
        configure = config
        def pack(self, *a, **k): self.empaquetado = True
        def place(self, *a, **k): pass
        def pack_forget(self, *a, **k): self.empaquetado = False
        def __getattr__(self, n): return lambda *a, **k: None
    creados = []

    def fab(*a, **k):
        w = W(*a, **k); creados.append(w); return w
    viejo_tk = mod.tk
    mod.tk = NS(Frame=fab, Label=fab, Canvas=fab)
    try:
        gu = object.__new__(I)
        gu.root = W()
        gu._construir_barra_superior()
        gu._construir_columna_derecha(700)
        chk(set(gu._tarjetas) == {"TRAYECTORIA", "PROPULSIÓN", "MISIÓN"}, "faltan tarjetas")
        chk(all(t["orden"] for t in gu._tarjetas.values()), "tarjetas vacías al construir (deben mostrar «—»)")
        gu._actualizar_barra_superior(None)
        chk(gu.lbl_estado.opts["text"] == "LISTO" and gu.lbl_met.opts["text"] == "+00:00:00", "barra vacía")
        sim.estado.t = 3725.0
        gu._actualizar_barra_superior(sim)
        chk(gu.lbl_met.opts["text"] == "+01:02:05", f"MET en barra {gu.lbl_met.opts['text']}")
        chk(gu.lbl_estado.opts["text"] == "EN CURSO" and gu.lbl_modo.opts["text"], "estado/modo en barra")
        chk(gu.lbl_fase.opts["text"] == str(sim.estado.fase).replace("_", " ").upper(), "fase en barra")
        gu._actualizar_tarjetas(filas)
        tray = gu._tarjetas["TRAYECTORIA"]
        chk(tray["orden"] == ["Altitud:"] and tray["filas"]["Altitud:"][2].opts["text"] == "1 km", "valor de la tarjeta")
        n0 = len(creados)
        gu._actualizar_tarjetas(filas)          # mismas filas: no crea widgets ni reordena
        chk(len(creados) == n0, "recrea widgets con las mismas filas")
        filas2 = [("Altitud:", "2 km", "alerta"), ("Velocidad:", "7 m/s", "normal")]
        gu._actualizar_tarjetas(filas2)
        chk(tray["orden"] == ["Altitud:", "Velocidad:"] and tray["filas"]["Altitud:"][2].opts["text"] == "2 km"
            and tray["filas"]["Altitud:"][2].opts["fg"] == mod.COLOR_TEXTO_ERROR, "actualización de valor/color")
        mis = gu._tarjetas["MISIÓN"]
        chk(not any(f[0].empaquetado for f in mis["filas"].values()), "filas ausentes deben ocultarse")
        gu._limpiar_texto_sync()
        chk(gu.lbl_estado.opts["text"] == "LISTO" and gu.lbl_met.opts["text"] == "+00:00:00", "limpiar: barra")
        chk(all(t["ultimo"].get(n, ("",))[0] == "—" for t in gu._tarjetas.values() for n in t["orden"])
            and gu._tarjetas["TRAYECTORIA"]["orden"] == ["Altitud:", "Velocidad:", "Apoapsis:", "Periapsis:"],
            "limpiar: las tarjetas deben quedar en reposo («—»)")
        gu.config = mod.Config()
        gu.config.modo_mision = "lunar"
        gu._actualizar_barra_superior(None)
        chk(gu.lbl_modo.opts["text"] == "lunar", f"modo en reposo: {gu.lbl_modo.opts['text']}")
    finally:
        mod.tk = viejo_tk
    # construcción completa de la ventana (tk, matplotlib-tk y pyplot simulados): la geometría
    # suma el ancho de la ventana y el drawer se coloca bajo la barra superior
    from unittest.mock import MagicMock
    viejos = (mod.tk, mod.FigureCanvasTkAgg, mod.plt)
    try:
        def eje():
            a = MagicMock(); a.plot.side_effect = lambda *x, **k: [MagicMock()]; a.twinx.side_effect = eje; a.inset_axes.side_effect = lambda *x, **k: eje()   # (12.11.0) recuadro de la Tierra
            return a
        mod.tk, mod.FigureCanvasTkAgg, mod.plt = MagicMock(), MagicMock(), MagicMock()
        mod.plt.subplots.side_effect = lambda **k: (MagicMock(), eje())
        gc = object.__new__(I)
        gc.root, gc.config, gc.entradas, gc.sim = MagicMock(), mod.Config(), {}, None
        gc.camara_libre, gc.arrastre = {"activa": False}, {"activo": False, "x0": None, "y0": None}
        gc.drawer_estado = {"abierto": False}
        gc._construir_ui()
        chk(I.ANCHO_COLUMNA_IZQ + I.ANCHO_COLUMNA_DER + I.ANCHO_TAB < I.ANCHO_VENTANA, "sin espacio para la animación")
        gc._alternar_drawer()
        chk(gc.drawer_frame.place.call_args.kwargs.get("y") == I.ALTO_BARRA, "drawer no queda bajo la barra")
        chk(set(gc._tarjetas) == {"TRAYECTORIA", "PROPULSIÓN", "MISIÓN"}, "ventana construida sin tarjetas")
    finally:
        mod.tk, mod.FigureCanvasTkAgg, mod.plt = viejos
    if errores:
        print(f"[tarjetas_gui] FALLA ({len(errores)}):")
        for e in errores:
            print("   ", e)
        return False
    print("[tarjetas_gui] OK (MET, estado global, reparto en tarjetas, actualización sin recrear widgets)")
    return True


def verificar_corte_y_filas_lunar(archivo: str) -> bool:
    """(12.8.1) En un vuelo lunar nominal: (1) todos los cortes de `_k_quema` (TLI) convergen
    con residuo de energía ~1e-14 (el endurecimiento no cambia vuelos nominales); (2) el máximo
    de filas del HUD lunar cabe en las tarjetas de la interfaz."""
    mod = cargar_modulo(archivo)
    I = getattr(mod, "InterfazSimulacion", None)
    if I is None or not hasattr(mod, "_k_quema"):
        print("[corte_filas_lunar] omitido")
        return True
    import contextlib, io, math
    mod._TERCEROS_CUERPOS_FUENTE = "sin_pymcel"
    errores = []

    def chk(c, m):
        if not c:
            errores.append(m)
    orig = mod._k_quema
    res, oks, args0 = [], [], []

    def envuelta(*a):
        o = orig(*a)
        if not args0:
            args0.append(a)
        oks.append(o[6] == 0)
        if o[6] == 0:
            x, y, vx, vy = o[:4]
            res.append(abs(0.5 * (vx * vx + vy * vy) - a[-4] / math.hypot(x, y) - a[10]) / abs(a[10]))
        return o
    mod._k_quema = envuelta
    try:
        gui = object.__new__(I)
        cfg = _cfg_lunar(mod, 90.0, LUNAR_LOI=True)
        nombres = {}
        with contextlib.redirect_stdout(io.StringIO()):
            sim = mod.Simulacion(cfg, proveedor_efemerides=_proveedor_circular(3.3))
            n = 0
            while not sim.estado.simulacion_terminada and n < 400000:
                sim.avanzar_paso(); n += 1
                if n % 500 == 0 or sim.estado.simulacion_terminada:
                    filas = gui._filas_estado_general(sim) + gui._filas_carga_g_y_orbita(sim)
                    for tit, fs in mod._agrupar_filas_en_tarjetas(filas):
                        nombres.setdefault(tit, set()).update(f[0] for f in fs)
    finally:
        mod._k_quema = orig
    chk(len(res) > 20 and all(oks), f"cortes medidos: {len(res)}, ok={all(oks)}")
    # (12.8.4) códigos de estado de `_k_quema`: 0 ok, 1 no convergió, 2 sin combustible
    if args0:
        a = list(args0[0])
        chk(orig(*a)[6] == 0, "llamada nominal repetida debe dar estado 0")
        b = list(a); b[10] = float("nan")                    # energía objetivo no finita
        chk(orig(*b)[6] == 1, f"eps_obj NaN debe dar estado 1, da {orig(*b)[6]}")
        c = list(a); c[9] = a[5] - 1.0                       # masa seca = masa - 1 kg: sin combustible
        chk(orig(*c)[6] == 2, f"sin combustible debe dar estado 2, da {orig(*c)[6]}")
        # (12.9.1) el último paso, limitado por el propelente, se intenta antes de rendirse
        oka = orig(*a); m0, mf = a[5], oka[5]
        d = list(a); d[9] = mf - 1e-6 * (m0 - mf)           # combustible justo para el corte
        od = orig(*d)
        chk(od[6] == 0 and abs(od[5] - mf) < 1.0 and abs(od[4] - oka[4]) < 0.05,
            f"corte con el propelente justo: estado {od[6]}, m {od[5]:.2f} vs {mf:.2f}, t {od[4]:.3f} vs {oka[4]:.3f}")
        g_ = list(a); g_[8] = 1.0e12                          # (12.10.1) presión que anula el empuje neto
        og = orig(*g_)
        chk(og[6] in (1, 3) and og[5] == a[5], f"sin empuje neto: estado {og[6]}, masa {og[5]} (debe ser 3 desde 12.13.6, o 1 antes, y {a[5]})")
        e = list(a); e[9] = mf + 5.0                          # faltan 5 kg: no llega
        chk(orig(*e)[6] == 2, f"propelente insuficiente en el último paso debe dar 2, da {orig(*e)[6]}")
    else:
        errores.append("no se capturó ninguna llamada a _k_quema")
    chk(res and max(res) < 1e-12, f"residuo del corte TLI {max(res) if res else None}")
    filas_max = sum(len(v) for v in nombres.values())
    alto_est = 3 * 52 + filas_max * 22          # 3 tarjetas con título/márgenes + ~22 px por fila
    chk(alto_est <= I.ALTO_VENTANA - I.ALTO_BARRA, f"{filas_max} filas no caben en la columna ({alto_est} px)")
    if errores:
        print(f"[corte_filas_lunar] FALLA ({len(errores)}):")
        for e in errores:
            print("   ", e)
        return False
    print(f"[corte_filas_lunar] OK ({len(res)} cortes TLI, residuo máx {max(res):.1e}; {filas_max} filas máx en tarjetas, ~{alto_est} px)")
    return True


def verificar_robustez_1284(archivo: str) -> bool:
    """(12.8.4) Configuraciones imposibles rechazadas (y todos los presets aceptados), tope de
    sim_speed, y la GUI nombra el campo inválido (incluido OverflowError) sin dejar la
    configuración a medio actualizar."""
    mod = cargar_modulo(archivo)
    I = getattr(mod, "InterfazSimulacion", None)
    if I is None or not hasattr(mod.Config, "_validar_config"):
        print("[robustez_1284] omitido")
        return True
    from types import SimpleNamespace as NS
    from unittest.mock import MagicMock
    errores = []

    def chk(c, m):
        if not c:
            errores.append(m)

    def rechaza(cfg, texto, que):
        try:
            cfg._validar_config()
            errores.append(f"{que}: debía rechazarse")
        except ValueError as e:
            chk(texto in str(e), f"{que}: mensaje sin «{texto}»: {e}")
    # todos los presets siguen siendo válidos
    for modo, _ in I.MODOS_MISION:
        cfg = mod.Config(); cfg.modo_mision = modo; cfg.aplicar_preset()
        try:
            cfg._validar_config()
            if modo == "lunar":
                cfg.LUNAR_LOI = True
                cfg._validar_config()
        except ValueError as e:
            errores.append(f"preset {modo} rechazado: {e}")
    c = mod.Config(); c.modo_mision = "suborbital"; c.aplicar_preset(); c.APOGEO_OBJETIVO = 50_000.0
    rechaza(c, "KARMAN", "suborbital con apogeo < Kármán")
    c = mod.Config(); c.modo_mision = "satelital"; c.aplicar_preset(); c.ALTURA_SEGURIDAD = 2e6
    rechaza(c, "ALTURA_SEGURIDAD", "apogeo <= altura de seguridad")
    c = mod.Config(); c.modo_mision = "lunar"; c.aplicar_preset(); c.T_MAX_LUNAR = 5 * 86400.0
    rechaza(c, "T_MAX_LUNAR", "T_MAX_LUNAR <= horizonte")
    # (12.8.5) el tope del modo lunar es T_MAX_LUNAR; T_MAX_ALTA_ENERGIA es de TLI/escape y no lo afecta
    c = mod.Config(); c.modo_mision = "lunar"; c.aplicar_preset(); c.T_MAX_ALTA_ENERGIA = 86400.0
    try:
        c._validar_config()
    except ValueError as e:
        errores.append(f"T_MAX_ALTA_ENERGIA no debe afectar al modo lunar: {e}")
    c = mod.Config(); c.modo_mision = "lunar"; c.aplicar_preset(); c.T_MAX_LUNAR = float("inf")
    rechaza(c, "T_MAX_LUNAR", "T_MAX_LUNAR infinito")
    c = mod.Config(); c.modo_mision = "gto"; c.aplicar_preset(); c.GTO_RP_FINAL = 4.0e7
    rechaza(c, "GTO_RP_FINAL", "GTO con periapsis final sobre el apogeo")
    c = mod.Config(); c.sim_speed = True
    rechaza(c, "sim_speed", "sim_speed=True (bool)")
    c = mod.Config(); c.modo_mision = "lunar"; c.aplicar_preset(); c.LUNAR_LOI = True; c.LUNAR_ORBITA_ALT = 1e8
    rechaza(c, "SOI lunar", "órbita LOI fuera de la SOI")
    for v in (0, 10_001):
        c = mod.Config(); c.sim_speed = v
        rechaza(c, "sim_speed", f"sim_speed={v}")
    # la GUI nombra el campo
    textos = {"angulo_kick": "3", "alt_inicio_giro": "400", "APOGEO_OBJETIVO": "1.3e6", "ALTURA_SEGURIDAD": "120000",
              "VEL_MIN_SES": "2500", "G_MAX_TRIPULADA": "4", "ECC_MAX_TRIPULADA_BCB": "0.01",
              "ECC_MAX_TRIPULADA_DIRECTO": "0.08", "sim_speed": "1", "latitud_lanzamiento": "0",
              "azimut_lanzamiento": "90", "longitud_lanzamiento_ecliptica": "90", "LUNAR_DIST_OBJETIVO": "1537.4",
              "LUNAR_ORBITA_ALT": "100", "fecha_lanzamiento": "2026-01-01T00:00:00"}
    claves = [k for k, _ in I.CONVERSORES_CAMPOS]
    chk(set(claves) == set(textos), f"campos de la GUI sin texto de prueba: {set(claves) ^ set(textos)}")
    g = object.__new__(I)
    g.config = mod.Config()
    g.var_modo = MagicMock(); g.var_modo.get.return_value = "satelital"
    g.var_perfil = MagicMock(); g.var_perfil.get.return_value = "bcb"
    for nom in ("var_crtbp", "var_cutoff_subor", "var_escala_log", "var_grilla", "var_vista_bari", "var_rotacion"):
        setattr(g, nom, MagicMock())
    msgs = []
    g._mostrar_error = msgs.append

    def con(**cambios):
        t = dict(textos); t.update(cambios)
        g.entradas = {k: NS(get=(lambda v=v: v)) for k, v in t.items()}
    con()
    chk(g._leer_config_desde_widgets() is True, f"lectura nominal falló: {msgs}")
    for texto, esp in (("1e400", "Velocidad de simulación"), ("abc", "Velocidad de simulación"), ("nan", "Velocidad de simulación")):
        msgs.clear(); g.config.angulo_kick = 7.0
        con(sim_speed=texto, angulo_kick="9")
        chk(g._leer_config_desde_widgets() is False and msgs and esp in msgs[0] and repr(texto) in msgs[0],
            f"sim_speed={texto!r}: {msgs}")
        chk(g.config.angulo_kick == 7.0, f"sim_speed={texto!r}: la configuración quedó a medio actualizar")
    msgs.clear(); con(fecha_lanzamiento="")
    ok = g._leer_config_desde_widgets()
    chk(ok is False and msgs and "fecha_lanzamiento" in msgs[0], f"fecha vacía: {ok} {msgs}")   # la valida _validar_config
    msgs.clear(); con(angulo_kick="xx")
    chk(g._leer_config_desde_widgets() is False and "Ángulo de kick" in msgs[0], f"campo con texto: {msgs}")
    if errores:
        print(f"[robustez_1284] FALLA ({len(errores)}):")
        for e in errores:
            print("   ", e)
        return False
    print("[robustez_1284] OK (configs imposibles rechazadas, presets válidos, sim_speed acotado, GUI nombra el campo)")
    return True


def verificar_robustez_1291(archivo: str) -> bool:
    """(12.9.1) Verlet 3D con diagnóstico, lectura transaccional de la configuración de la GUI y
    sim_speed entero estricto."""
    mod = cargar_modulo(archivo)
    I = getattr(mod, "InterfazSimulacion", None)
    if I is None or not hasattr(mod, "EfemeridesNoValidas"):
        print("[robustez_1291] omitido")
        return True
    import numpy as np
    from types import SimpleNamespace as NS
    from unittest.mock import MagicMock
    errores = []

    def chk(c, m):
        if not c:
            errores.append(m)
    # -- Verlet: singularidad y estados no finitos --
    val = getattr(mod, "_tabla_verlet3d_validada", None)
    if val is None:
        errores.append("falta _tabla_verlet3d_validada")
    else:
        y_ok = np.array([1.5e11, 0, 0, 0, 29780.0, 0, 3.844e8, 0, 0, 0, 1022.0, 0])
        mus = (mod.MU_SOL + mod.MU_T, mod.MU_T + mod.MU_LUNA, mod.MU_SOL)
        t = val(y_ok, 20.0, 3, 5, *mus)
        chk(t.shape == (5, 12) and np.all(np.isfinite(t)), "caso nominal")
        y_sing = y_ok.copy(); y_sing[6:9] = 0.0              # la Luna sobre la Tierra: r_L = 0
        y_nan = y_ok.copy(); y_nan[0] = float("nan")
        y_inf = y_ok.copy(); y_inf[4] = float("inf")
        for nombre, y in (("r_L=0", y_sing), ("NaN", y_nan), ("inf", y_inf)):
            try:
                val(y, 20.0, 3, 5, *mus)
                errores.append(f"Verlet {nombre}: debía lanzar EfemeridesNoValidas")
            except mod.EfemeridesNoValidas as e:
                chk(len(str(e)) > 10, f"Verlet {nombre}: mensaje vacío")
    # -- sim_speed entero estricto --
    ee = getattr(mod, "_entero_estricto", None)
    if ee is None:
        errores.append("falta _entero_estricto")
    else:
        chk(ee("5") == 5 and ee("5.0") == 5 and ee("1e3") == 1000 and ee(" 7 ") == 7, "enteros válidos")
        for malo in ("1.9", "0.5", "nan", "inf", "-inf", "abc", "1e400"):
            try:
                ee(malo)
                errores.append(f"_entero_estricto({malo!r}) debía rechazarse")
            except (ValueError, OverflowError):
                pass
    # -- lectura transaccional --
    textos = {"angulo_kick": "3", "alt_inicio_giro": "400", "APOGEO_OBJETIVO": "1.3e6", "ALTURA_SEGURIDAD": "120000",
              "VEL_MIN_SES": "2500", "G_MAX_TRIPULADA": "4", "ECC_MAX_TRIPULADA_BCB": "0.01",
              "ECC_MAX_TRIPULADA_DIRECTO": "0.08", "sim_speed": "1", "latitud_lanzamiento": "0",
              "azimut_lanzamiento": "90", "longitud_lanzamiento_ecliptica": "90", "LUNAR_DIST_OBJETIVO": "1537.4",
              "LUNAR_ORBITA_ALT": "100", "fecha_lanzamiento": "2026-01-01T00:00:00"}
    g = object.__new__(I)
    g.config = mod.Config()
    g.var_modo = MagicMock(); g.var_modo.get.return_value = "suborbital"
    g.var_perfil = MagicMock(); g.var_perfil.get.return_value = "directo"
    for nom in ("var_crtbp", "var_cutoff_subor", "var_escala_log", "var_grilla", "var_vista_bari", "var_rotacion"):
        setattr(g, nom, MagicMock())
    msgs = []
    g._mostrar_error = msgs.append

    def con(**cambios):
        tt = dict(textos); tt.update(cambios)
        g.entradas = {k: NS(get=(lambda v=v: v)) for k, v in tt.items()}
    antes = (g.config.modo_mision, g.config.perfil_tripulada, g.config.angulo_kick)
    objeto_antes = g.config
    for que, cambios in (("conversión fallida", {"angulo_kick": "xx"}),
                         ("sim_speed fraccionario", {"sim_speed": "1.9"}),
                         ("config imposible (latitud)", {"latitud_lanzamiento": "95"}),
                         ("config imposible (validación)", {"ALTURA_SEGURIDAD": "-5"})):
        msgs.clear(); con(**cambios)
        chk(g._leer_config_desde_widgets() is False and msgs, f"{que}: debía rechazarse ({msgs})")
        chk(g.config is objeto_antes and (g.config.modo_mision, g.config.perfil_tripulada, g.config.angulo_kick) == antes,
            f"{que}: la configuración quedó a medio actualizar ({g.config.modo_mision}, {g.config.perfil_tripulada})")
    msgs.clear(); con()
    chk(g._leer_config_desde_widgets() is True, f"lectura nominal: {msgs}")
    chk(g.config.modo_mision == "suborbital" and g.config.perfil_tripulada == "directo" and g.config.angulo_kick == 3.0,
        "tras una lectura válida la configuración debe reflejar los widgets")
    if errores:
        print(f"[robustez_1291] FALLA ({len(errores)}):")
        for e in errores:
            print("   ", e)
        return False
    print("[robustez_1291] OK (Verlet con diagnóstico, sim_speed entero, lectura transaccional de la GUI)")
    return True


def verificar_canvas_glow(archivo: str) -> bool:
    """(12.10.0) Glow de la trayectoria (cola reciente, mismos segmentos y colores, debajo y más
    ancho) y halo de la atmósfera en capas. Usa matplotlib real sobre un Figure sin ventana."""
    mod = cargar_modulo(archivo)
    I = getattr(mod, "InterfazSimulacion", None)
    if I is None:
        print("[canvas_glow] omitido")
        return True
    import contextlib, io
    import numpy as np
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    errores = []

    def chk(c, m):
        if not c:
            errores.append(m)
    g = object.__new__(I)
    fig = Figure(figsize=(6, 6)); FigureCanvasAgg(fig)
    g.fig, g.ax = fig, fig.add_subplot(111)
    g.ax.set_xlim(-9e6, 9e6); g.ax.set_ylim(-9e6, 9e6)
    g._crear_artistas_tierra()
    g._crear_artistas_trayectoria()
    glow = getattr(g, "lc_tray_glow", None)
    chk(glow is not None, "falta lc_tray_glow")
    if glow is not None:
        chk(max(glow.get_linewidth()) > 2 * max(g.lc_tray.get_linewidth()), "el glow debe ser bastante más ancho")
        chk(glow.get_zorder() < g.lc_tray.get_zorder(), "el glow va debajo de la trayectoria")
        chk(0 < glow.get_alpha() < 0.5, f"el glow debe ser translúcido: {glow.get_alpha()}")
        with contextlib.redirect_stdout(io.StringIO()):
            sim = mod.Simulacion(mod.Config())
            for _ in range(1500):
                sim.avanzar_paso()
        g._dibujar_trayectoria(sim)
        def expandir(lc):
            """(12.13.0) polilíneas → segmentos (n, 2, 2) y color por segmento."""
            polis, cols = lc.get_segments(), lc.get_edgecolor()
            segs = [np.stack([p[:-1], p[1:]], axis=1) for p in polis]
            return (np.concatenate(segs), np.concatenate([np.repeat(c[None], len(sg), axis=0) for sg, c in zip(segs, cols)]))
        seg, col = expandir(g.lc_tray)
        segg, colg = expandir(glow)
        n = I.GLOW_MAX_SEGMENTOS
        chk(len(seg) > n, f"la prueba necesita más de {n} segmentos y hay {len(seg)}")
        chk(len(segg) == min(n, len(seg)) and np.array_equal(segg, seg[-n:]),
            "el glow debe ser la cola de la trayectoria")
        chk(np.allclose(colg[:, :3], col[-len(segg):, :3]),
            "el glow debe usar los colores de la trayectoria")
        chk(len(g.lc_tray.get_segments()) <= 30 and len(glow.get_segments()) <= 30,
            f"(12.13.0) pocas polilíneas: {len(g.lc_tray.get_segments())} y {len(glow.get_segments())}")
        # sin glow: no se dibuja
        g.lc_tray_glow.set_segments([])
        I.GLOW_TRAYECTORIA = False
        try:
            g._dibujar_trayectoria(sim)
            chk(len(glow.get_segments()) == 0, "con GLOW_TRAYECTORIA=False el glow debe quedar vacío")
        finally:
            I.GLOW_TRAYECTORIA = True
    # halo de la atmósfera: capas concéntricas, más tenues hacia afuera
    capas = [p for p in g.patches_tierra if p.get_facecolor()[3] > 0 and p.get_edgecolor()[3] == 0]
    radios = [p.radius for p in capas]
    alphas = [p.get_alpha() for p in capas]
    chk(len(capas) >= 3, f"el halo debe tener al menos 3 capas, tiene {len(capas)}")
    ordenadas = sorted(zip(radios, alphas))
    chk(all(a1 > a2 for (_, a1), (_, a2) in zip(ordenadas, ordenadas[1:])), f"el halo debe ser más tenue hacia afuera: {ordenadas}")
    chk(all(r > mod.R_T for r in radios), "las capas del halo deben quedar sobre la superficie")
    if errores:
        print(f"[canvas_glow] FALLA ({len(errores)}):")
        for e in errores:
            print("   ", e)
        return False
    print("[canvas_glow] OK (glow de la cola, colores compartidos, desactivable, halo en capas)")
    return True


def verificar_robustez_12101(archivo: str) -> bool:
    """(12.10.1) Tope de tiempo del suborbital, barrido de ventana TLI tolerante a fallos
    puntuales, guarda casi parabólica de la LOI, `_hermite_estado` con dt nulo y estados
    ABORTADA/IMPACTO de la barra superior."""
    mod = cargar_modulo(archivo)
    I = getattr(mod, "InterfazSimulacion", None)
    if I is None or not hasattr(mod.Config, "T_MAX_SUBORBITAL"):
        print("[robustez_12101] FALLA: falta Config.T_MAX_SUBORBITAL")
        return False
    import contextlib, io
    from types import SimpleNamespace as NS
    errores = []

    def chk(c, m):
        if not c:
            errores.append(m)
    # -- suborbital sin apogeo: termina en T_MAX_SUBORBITAL --
    def volar_sub(cutoff, tmax=None, max_pasos=100_000):
        cfg = mod.Config(); cfg.modo_mision = "suborbital"; cfg.aplicar_preset()
        cfg.suborbital_cutoff_por_apogeo = cutoff
        if tmax is not None:
            cfg.T_MAX_SUBORBITAL = tmax
        with contextlib.redirect_stdout(io.StringIO()):
            sim = mod.Simulacion(cfg)
            n = 0
            while not sim.estado.simulacion_terminada and n < max_pasos:
                sim.avanzar_paso(); n += 1
        return sim, n
    sim, n = volar_sub(False, 900.0)
    d = sim.eventos.datos_mision
    chk(sim.estado.simulacion_terminada and sim.eventos.mision_cumplida is False and d.get("fin_por_t_max")
        and "T_MAX_SUBORBITAL" in d.get("motivo", "") and 899.0 <= sim.estado.t < 910.0 and {"alt_max", "objetivo"} <= set(d),
        f"sin apogeo: terminada={sim.estado.simulacion_terminada} cumplida={sim.eventos.mision_cumplida} t={sim.estado.t:.0f} {d}")
    sim, n = volar_sub(True, 1000.0)                                  # nominal: impacto a ~437 s, el tope no estorba
    chk(sim.eventos.mision_cumplida is True and sim.estado.t < 600.0 and not sim.eventos.datos_mision.get("fin_por_t_max"),
        f"suborbital nominal alterado: {sim.eventos.mision_cumplida} t={sim.estado.t:.0f}")
    for malo in (0.0, -1.0, float("nan"), float("inf")):
        c = mod.Config(); c.T_MAX_SUBORBITAL = malo
        try:
            c._validar_config()
            errores.append(f"T_MAX_SUBORBITAL={malo} debía rechazarse")
        except ValueError as e:
            chk("T_MAX_SUBORBITAL" in str(e), f"mensaje sin el campo: {e}")
    # -- barrido de ventana TLI: un fallo puntual no aborta; todos fallando sí --
    orig = mod._k_quema
    cuenta = [0]

    def falla_una(*a):
        cuenta[0] += 1
        if cuenta[0] == 2:
            return (*a[:6], 1)
        return orig(*a)

    def falla_todas(*a):
        return (*a[:6], 1)
    try:
        mod._k_quema = falla_una
        sim, _, _ = _volar_lunar(mod, 90.0, 0.5)
        chk(sim.eventos.mision_cumplida is True, f"un fallo puntual abortó la búsqueda: {sim.eventos.datos_mision.get('motivo')}")
        mod._k_quema = falla_todas
        sim, _, _ = _volar_lunar(mod, 90.0, 0.5)
        chk(sim.eventos.mision_cumplida is False and "no convergió" in sim.eventos.datos_mision.get("motivo", ""),
            f"todos fallando: {sim.eventos.datos_mision.get('motivo')}")
    finally:
        mod._k_quema = orig
    # -- un fallo DENTRO del tramo con encuentro no lo trunca (12.10.2) --
    orig_k, orig_c, orig_p = mod._k_quema, mod._k_coast, mod._coast_hasta_periselene
    hits, tenc, cuenta2, objetivo = [], [], [0], [None]

    def espia_peri(*a):
        o = orig_p(*a)
        if a[6] == 60.0:
            hits.append(o[7] > 0.0)           # ¿entró a la SOI? (llamadas del barrido)
        return o

    def espia_coast(*a):
        if a[-1] is False and a[6] == 10.0:
            tenc.append(a[5])                 # t_enc de cada evaluación
        return orig_c(*a)

    def falla_en(*a):
        cuenta2[0] += 1
        if cuenta2[0] == objetivo[0]:
            return (*a[:6], 1)
        return orig_k(*a)

    def x_t_de(tiempos):
        """t_enc de la primera evaluación tras el barrido (la del centro de la ventana)."""
        for i in range(1, len(tiempos)):
            if abs(tiempos[i] - tiempos[i - 1] - mod.Config().LUNAR_PASO_VENTANA_S) > 1e-6:
                return tiempos[i]
        return None
    try:
        mod._coast_hasta_periselene, mod._k_coast = espia_peri, espia_coast
        sim0, _, log0 = _volar_lunar(mod, 90.0, 0.5)
        x_nominal = x_t_de(tenc)
        idx = [i for i, h in enumerate(hits) if h]
        chk(sim0.eventos.mision_cumplida is True and len(idx) >= 3 and x_nominal is not None,
            f"vuelo nominal de referencia: cumplida={sim0.eventos.mision_cumplida} aciertos={len(idx)} x_t={x_nominal}")
        if len(idx) >= 3 and x_nominal is not None:
            objetivo[0] = idx[1] + 1                     # falla el 2.º instante con encuentro
            tenc.clear(); hits.clear(); cuenta2[0] = 0
            mod._k_quema = falla_en
            sim1, _, log1 = _volar_lunar(mod, 90.0, 0.5)
            x_fallo = x_t_de(tenc)
            chk(sim1.eventos.mision_cumplida is True, f"fallo dentro del tramo: {sim1.eventos.datos_mision.get('motivo')}")
            chk(x_fallo is not None and abs(x_fallo - x_nominal) < 1e-6,
                f"el centro de la ventana debe conservarse: {x_fallo} vs {x_nominal}")
            # (12.13.8) Desde 12.13.6 el barrido ya descarta por diseño los instantes cuyo
            # periselene no se alcanza (mínimo provisional); el vuelo nominal los cuenta y
            # los avisa. Se comprueba entonces que el fallo inyectado suma exactamente uno.
            def _omitidos(log):
                m = re.search(r"Barrido TLI: (\d+)/\d+ instantes omitidos", log)
                return int(m.group(1)) if m else 0
            chk(log1.count("Barrido TLI: primer fallo") == 1 and "instantes omitidos" in log1,
                "debe imprimirse el primer fallo y un resumen")
            chk(_omitidos(log1) == _omitidos(log0) + 1,
                f"el fallo inyectado debe sumar un instante omitido: {_omitidos(log0)} -> {_omitidos(log1)}")
    finally:
        mod._k_quema, mod._k_coast, mod._coast_hasta_periselene = orig_k, orig_c, orig_p
    # -- LOI casi parabólica --
    g = object.__new__(mod.Simulacion)
    g.etapa_actual = lambda: NS()
    g._elementos_selenocentricos = lambda: {"eps": 1.0e-12, "ecc": 1.0, "r": 1.9e6, "rp": 1.9e6, "h": 1.0, "vr": -1.0}
    try:
        g._loi_estimar()
        errores.append("LOI casi parabólica debía rechazarse")
    except ValueError as e:
        chk("parabólica" in str(e), f"mensaje: {e}")
    # -- Hermite con dt nulo --
    y0, y1 = (1.0, 2.0, 3.0, 4.0), (5.0, 6.0, 7.0, 8.0)
    chk(mod.Simulacion._hermite_estado(y0, y1, 0.0, 0.5) == (5.0, 6.0, 7.0, 8.0), "Hermite con dt=0")
    e1 = mod.Simulacion._hermite_estado(y0, y1, 2.0, 1.0)
    chk(all(abs(a - b) < 1e-12 for a, b in zip(e1, y1)), f"Hermite en s=1 debe dar el estado final: {e1}")
    # -- estados terminales sin veredicto --
    def sim_falso(**ev):
        base = dict(mision_cumplida=None, datos_mision={}, fallo_integrador=None, impacto_registrado=False)
        base.update(ev)
        return NS(eventos=NS(**base), estado=NS(simulacion_terminada=True), esperando_plan=False)
    chk(mod._estado_global(sim_falso(fallo_integrador="RK45: x"))[0] == "ABORTADA", "estado ABORTADA")
    chk(mod._estado_global(sim_falso(impacto_registrado=True))[0] == "IMPACTO", "estado IMPACTO")
    chk(mod._estado_global(sim_falso())[0] == "FINALIZADA", "estado FINALIZADA")
    chk(mod._estado_global(sim_falso(mision_cumplida=False, impacto_registrado=True, datos_mision={"motivo": "m"}))[0] == "NO CUMPLIDA",
        "el veredicto tiene prioridad sobre el impacto")
    if errores:
        print(f"[robustez_12101] FALLA ({len(errores)}):")
        for e in errores:
            print("   ", e)
        return False
    print("[robustez_12101] OK (T_MAX suborbital, barrido tolerante, LOI casi parabólica, Hermite dt=0, ABORTADA/IMPACTO)")
    return True


def verificar_drawer_poda(archivo: str) -> bool:
    """(12.10.3) La gráfica de vuelo no une el ascenso con la última vuelta cuando el historial
    orbital se podó (NaN en el corte), los límites de los ejes siguen siendo finitos y el
    `.update()` de `_fin_suborbital_por_t_max` conserva claves previas."""
    mod = cargar_modulo(archivo)
    I = getattr(mod, "InterfazSimulacion", None)
    if I is None:
        print("[drawer_poda] omitido")
        return True
    import math
    from types import SimpleNamespace as NS
    from unittest.mock import MagicMock
    errores = []

    def chk(c, m):
        if not c:
            errores.append(m)
    chk(hasattr(mod, "_serie_con_corte"), "falta _serie_con_corte")
    if hasattr(mod, "_serie_con_corte"):
        r = mod._serie_con_corte([1.0, 2.0, 3.0], 1)
        chk(len(r) == 4 and r[0] == 1.0 and math.isnan(r[1]) and r[2:] == [2.0, 3.0], f"_serie_con_corte: {r}")
    g = object.__new__(I)
    g.drawer_estado = {"abierto": True}
    g.linea_alt, g.linea_vel = MagicMock(), MagicMock()
    g.ax2, g.ax2b, g.canvas2 = MagicMock(), MagicMock(), MagicMock()
    n_asc, n_orb = 6, 5
    hist_t = [float(i) for i in range(n_asc)] + [1000.0 + i for i in range(n_orb)]
    hist_alt = [1000.0 * (i + 1) for i in range(n_asc + n_orb)]
    hist_v = [100.0 * (i + 1) for i in range(n_asc + n_orb)]

    def sim(podada):
        return NS(hist_t=list(hist_t), hist_alt=list(hist_alt), hist_v=list(hist_v),
                  _orbita_podada=podada, _idx_inicio_orbita=n_asc)
    g._actualizar_drawer(sim(True))
    t_, a_ = g.linea_alt.set_data.call_args.args
    nan_pos = [i for i, v in enumerate(t_) if isinstance(v, float) and math.isnan(v)]
    chk(nan_pos == [n_asc] and len(t_) == n_asc + n_orb + 1 and t_[n_asc - 1] == 5.0 and t_[n_asc + 1] == 1000.0,
        f"con poda debe haber un NaN entre ascenso y órbita: {t_}")
    chk(len(a_) == len(t_) and math.isnan(a_[n_asc]), "la altitud debe tener el mismo corte")
    chk(len(g.linea_vel.set_data.call_args.args[0]) == len(t_), "la velocidad debe tener el mismo corte")
    chk(math.isfinite(g.ax2.set_xlim.call_args.args[1]) and math.isfinite(g.ax2.set_ylim.call_args.args[1])
        and math.isfinite(g.ax2b.set_ylim.call_args.args[1]), "límites de ejes no finitos")
    g._actualizar_drawer(sim(False))
    t_, _ = g.linea_alt.set_data.call_args.args
    chk(len(t_) == n_asc + n_orb and not any(math.isnan(v) for v in t_), "sin poda no debe haber NaN")
    # .update(): conserva claves previas
    with __import__("contextlib").redirect_stdout(__import__("io").StringIO()):
        cfg = mod.Config(); cfg.modo_mision = "suborbital"; cfg.aplicar_preset()
        s0 = mod.Simulacion(cfg)
        s0.eventos.datos_mision["clave_previa"] = "valor"
        s0._fin_suborbital_por_t_max()
    chk(s0.eventos.datos_mision.get("clave_previa") == "valor" and s0.eventos.datos_mision.get("fin_por_t_max") is True,
        f"datos_mision tras T_MAX: {s0.eventos.datos_mision}")
    if errores:
        print(f"[drawer_poda] FALLA ({len(errores)}):")
        for e in errores:
            print("   ", e)
        return False
    print("[drawer_poda] OK (corte NaN tras la poda, límites finitos, update conserva claves)")
    return True


def verificar_pestanas_gui(archivo: str) -> bool:
    """(12.9.0) Columna izquierda en pestañas, línea de tiempo de eventos, separador de miles
    y elección de fuente. Usa un tkinter de mentira con widgets distinguibles."""
    mod = cargar_modulo(archivo)
    I = getattr(mod, "InterfazSimulacion", None)
    if I is None:
        print("[pestanas_gui] omitido")
        return True
    from types import SimpleNamespace as NS
    errores = []

    def chk(c, m):
        if not c:
            errores.append(m)
    # -- funciones puras --
    for t, esp in (("12345 m", "12,345 m"), ("1234567.8 m/s", "1,234,567.8 m/s"), ("-12345", "-12,345"),
                   ("1234", "1,234"), ("999 m", "999 m"), ("384.4 km", "384.4 km"), ("2026-10-22", "2026-10-22"),
                   ("T+00:12:34", "T+00:12:34"), ("—", "—"), ("0123 s", "0123 s"), ("12345abc", "12345abc")):
        r = mod._con_miles(t) if hasattr(mod, "_con_miles") else None
        chk(r == esp, f"_con_miles({t!r}) = {r!r} != {esp!r}")
    ef = getattr(mod, "_elegir_fuente_mono", lambda x: None)
    chk(ef({"Consolas", "Courier New"}) == "Consolas", "prefiere Consolas")
    chk(ef({"Menlo", "DejaVu Sans Mono"}) == "DejaVu Sans Mono", "orden de preferencia")
    chk(ef({"Courier New"}) == "Courier New" and ef(set()) == "Courier New" and ef(None) == "Courier New",
        "sin monoespaciadas preferidas: Courier New")
    # -- cada campo numérico está en una sola pestaña válida --
    pc = getattr(I, "PESTANA_DE_CAMPO", {})
    claves = [c for c, _ in I.CAMPOS_NUMERICOS]
    chk(set(pc) == set(claves) and len(claves) == len(set(claves)), "PESTANA_DE_CAMPO no cubre exactamente los campos")
    claves_pest = [k for k, _ in getattr(I, "PESTANAS_IZQ", ())]
    chk(len(claves_pest) == 4 and "eventos" in claves_pest and all(v in claves_pest for v in pc.values()),
        f"pestañas {claves_pest}")

    ALZADOS = []

    class W:
        def __init__(self, *a, **k):
            self.opts = dict(k); self.empaquetado = False; self.hijos = []; self.items = []; self.rejilla = None
            self.llamadas = []; self.bindings = {}
            if a and hasattr(a[0], "hijos"):
                a[0].hijos.append(self)
            self.padre = a[0] if a and hasattr(a[0], "hijos") else None
        def config(self, **k): self.opts.update(k)
        configure = config
        def pack(self, *a, **k): self.empaquetado = True; self.llamadas.append(("pack", k))
        def pack_forget(self, *a, **k): self.empaquetado = False
        def place(self, *a, **k): pass
        def grid(self, *a, **k): self.rejilla = dict(k)
        def tkraise(self): ALZADOS.append(self)
        def bind(self, seq, f): self.bindings[seq] = f
        # Canvas
        def create_oval(self, *a, **k): self.items.append(("oval", a, k)); return len(self.items)
        def create_line(self, *a, **k): self.items.append(("line", a, k)); return len(self.items)
        def create_text(self, *a, **k): self.items.append(("text", a, k)); return len(self.items)
        def bbox(self, i):
            _, a, k = self.items[i - 1]
            lineas = 1 + str(k.get("text", "")).count("\n")
            return (0, a[1], 100, a[1] + 12 * lineas)
        def delete(self, *a): self.items.clear()
        def yview_moveto(self, f): self.opts["_yview"] = f
        def yview_scroll(self, n, u): self.opts["_scroll"] = (n, u)
        def __getattr__(self, n): return lambda *a, **k: None
    viejo_tk = mod.tk
    mod.tk = NS(Frame=W, Label=W, Canvas=W, Button=W, Entry=W, Radiobutton=W, Checkbutton=W, Scrollbar=W,
                StringVar=lambda **k: NS(get=lambda: k.get("value"), set=lambda v: None),
                BooleanVar=lambda **k: NS(get=lambda: k.get("value"), set=lambda v: None))
    try:
        g = object.__new__(I)
        g.root = W()
        g._crear_columna_izquierda(700)
        chk(set(g._pestanas) == set(claves_pest), "faltan pestañas")
        chk(ALZADOS and ALZADOS[-1] is g._pestanas["mision"], "al crear debe quedar arriba la primera pestaña")
        celdas = {(f.rejilla or {}).get("row", None) is not None and ((f.rejilla["row"], f.rejilla["column"]))
                  for f in g._pestanas.values()}
        chk(celdas == {(0, 0)}, f"las pestañas deben apilarse en la misma celda: {celdas}")
        chk(not any(f.empaquetado for f in g._pestanas.values()), "las pestañas no se empaquetan (grid + tkraise)")
        g._elegir_pestana("eventos")
        chk(ALZADOS[-1] is g._pestanas["eventos"] and g._pestana_activa == "eventos", "elegir pestaña")
        chk(g._botones_pestana["eventos"].opts["fg"] != g._botones_pestana["mision"].opts["fg"], "pestaña activa sin resaltar")
        g._elegir_pestana("no_existe")
        chk(g._pestana_activa == "eventos", "pestaña inexistente cambió el estado")
        # -- la configuración reparte los widgets por pestaña y deja el pie fijo --
        g.config = mod.Config()
        g.entradas = {}
        g.var_modo = NS(get=lambda: "orbital", set=lambda v: None)
        g._construir_panel_config()
        for clave, _ in I.CAMPOS_NUMERICOS:
            fila = [h for h in g._pestanas[pc[clave]].hijos if isinstance(h, W)]
            chk(clave in g.entradas and fila, f"campo {clave} no está en su pestaña {pc[clave]}")
        chk(g.boton_iniciar.padre is not None and g.boton_iniciar.padre.padre is g._pie_izq, "INICIAR debe estar en el pie")
        chk(g.etiqueta_error.padre is g._pie_izq, "el error debe estar en el pie")
        chk(g.boton_ventanas.padre is g._pestanas["lanz"], "VENTANAS debe estar en LANZ.")
        # -- línea de tiempo --
        g._construir_panel_eventos()
        lt = g._linea_tiempo["lienzo"]
        g._agregar_evento("BURNOUT ETAPA 0", "titulo")
        g._agregar_evento("  t=150.0s  alt=70.0km\n")
        g._agregar_evento("APOGEO", "titulo")
        g._agregar_evento("  t=500.0s\n")
        g._agregar_evento("  a 12 km de la base")
        g._agregar_evento("💥 IMPACTO", "fallo")
        g._agregar_evento("   \n")                                   # vacío: se ignora
        textos = [it for it in lt.items if it[0] == "text"]
        ovalos = [it for it in lt.items if it[0] == "oval"]
        lineas = [it for it in lt.items if it[0] == "line"]
        chk(len(textos) == 6 and len(ovalos) == 3 and len(lineas) == 2,
            f"eventos dibujados: {len(textos)} textos, {len(ovalos)} puntos, {len(lineas)} líneas")
        ys = [it[1][1] for it in textos]
        chk(ys == sorted(ys) and len(set(ys)) == len(ys), f"el texto debe ir bajando sin solaparse: {ys}")
        chk([o[2]["fill"] for o in ovalos] == [mod.COLOR_EVENTO["titulo"], mod.COLOR_EVENTO["titulo"], mod.COLOR_EVENTO["fallo"]],
            "colores de los puntos")
        chk(textos[1][2]["text"] == "t=150.0s  alt=70.0km" and textos[1][1][1] > textos[0][1][1], "detalle bajo su título")
        sr = lt.opts.get("scrollregion")
        chk(sr and sr[3] > ys[-1] and lt.opts.get("_yview") == 1.0, f"scroll al final: {sr} {lt.opts.get('_yview')}")
        lt.bindings["<MouseWheel>"](NS(num=None, delta=-120))
        chk(lt.opts.get("_scroll") == (1, "units"), "rueda sobre la línea de tiempo")
        g._limpiar_linea_tiempo()
        chk(not lt.items and g._linea_tiempo["y_punto"] is None, "limpiar la línea de tiempo")
        for i in range(300):                                        # muchos eventos: el scroll crece, sin errores
            g._agregar_evento(f"EVENTO {i}", "exito" if i % 2 else "fallo")
            g._agregar_evento(f"  t={i}.0s")
        chk(sum(1 for it in lt.items if it[0] == "oval") == 300 and lt.opts["scrollregion"][3] > 300 * 24,
            f"300 eventos: scrollregion {lt.opts.get('scrollregion')}")
        g._limpiar_linea_tiempo()
        g._agregar_evento("  detalle suelto")                         # sin evento previo: no revienta
        chk(len(lt.items) == 1, "detalle sin evento previo")
        # -- los _mostrar_evento_* usan la línea de tiempo --
        g._limpiar_linea_tiempo()
        g.eventos_ya_mostrados = set()
        ev = NS(entro_soi_luna=True, datos_soi_luna={"t": 7200.0})
        g._mostrar_evento_soi_luna(ev)
        g._mostrar_evento_soi_luna(ev)                                # no se repite
        chk(sum(1 for it in lt.items if it[0] == "oval") == 1 and len(lt.items) == 3, f"soi: {lt.items}")
        # -- tarjetas: miles al mostrar --
        g._marco_tarjetas = W(); g._tarjetas = {}
        g._crear_tarjeta("TRAYECTORIA"); g._crear_tarjeta("PROPULSIÓN"); g._crear_tarjeta("MISIÓN")
        g._actualizar_tarjetas([("Altitud:", "12345 m", "normal")])
        chk(g._tarjetas["TRAYECTORIA"]["filas"]["Altitud:"][2].opts["text"] == "12,345 m", "miles en la tarjeta")
        # -- iniciar/reiniciar cambian de pestaña --
        g._elegir_pestana("mision")
        g._elegir_pestana("eventos")
        chk(g._pestana_activa == "eventos", "ir a eventos")
    finally:
        mod.tk = viejo_tk
    if errores:
        print(f"[pestanas_gui] FALLA ({len(errores)}):")
        for e in errores:
            print("   ", e)
        return False
    print("[pestanas_gui] OK (pestañas, campos por pestaña, pie fijo, línea de tiempo, miles, fuentes)")
    return True


def verificar_inset_y_scroll(archivo: str) -> bool:
    """(12.11.0) Barra de desplazamiento de la línea de tiempo (aparece solo si hace falta, el
    autodesplazamiento respeta al usuario) y recuadro de la Tierra (aparece cuando la Tierra no
    cabe en el mapa, histéresis, corte NaN de la poda, sin que la Luna aplaste la escala)."""
    mod = cargar_modulo(archivo)
    I = getattr(mod, "InterfazSimulacion", None)
    if I is None:
        print("[inset_scroll] omitido")
        return True
    import math
    from types import SimpleNamespace as NS
    errores = []

    def chk(c, m):
        if not c:
            errores.append(m)
    # -- funciones puras --
    vf, te = getattr(mod, "_vista_al_final", None), getattr(mod, "_tierra_en_vista", None)
    chk(vf is not None and te is not None, "faltan _vista_al_final / _tierra_en_vista")
    if vf and te:
        chk(vf((0.5, 1.0)) and vf((0.0, 1.0)) and vf(None) and not vf((0.2, 0.6)), "_vista_al_final")
        R = 100.0
        chk(te((-200, 200), (-200, 200), R, 1.0) and not te((-50, 200), (-200, 200), R, 1.0)
            and not te((-110, 200), (-200, 200), R, 1.15) and te((-110, 200), (-200, 200), R, 1.0)
            and te((200, -200), (200, -200), R, 1.0), "_tierra_en_vista")
    # -- barra de la línea de tiempo (tkinter de mentira) --
    class W:
        def __init__(self, *a, **k):
            self.opts = dict(k); self.empaquetado = False; self.items = []; self.llamadas = []; self.bindings = {}
        def config(self, **k): self.opts.update(k)
        configure = config
        def pack(self, *a, **k): self.empaquetado = True; self.llamadas.append(k)
        def pack_forget(self, *a, **k): self.empaquetado = False
        def bind(self, seq, f): self.bindings[seq] = f
        def create_oval(self, *a, **k): self.items.append(("oval", a, k)); return len(self.items)
        def create_line(self, *a, **k): self.items.append(("line", a, k)); return len(self.items)
        def create_text(self, *a, **k): self.items.append(("text", a, k)); return len(self.items)
        def bbox(self, i):
            _, a, k = self.items[i - 1]
            return (0, a[1], 100, a[1] + 12 * (1 + str(k.get("text", "")).count("\n")))
        def delete(self, *a): self.items.clear()
        def yview_moveto(self, f): self.opts["_yview"] = f
        def set(self, a, b): self.opts["_set"] = (a, b)
        def __getattr__(self, n): return lambda *a, **k: None
    viejo_tk = mod.tk
    mod.tk = NS(Frame=W, Canvas=W, Scrollbar=W)
    try:
        g = object.__new__(I)
        g._pestanas = {"eventos": W()}
        g._construir_panel_eventos()
        lt = g._linea_tiempo
        barra, lienzo = lt.get("barra"), lt["lienzo"]
        chk(barra is not None, "falta la barra en _linea_tiempo")
        cmd = lienzo.opts.get("yscrollcommand")
        chk(callable(cmd), "el lienzo debe tener yscrollcommand")
        if barra is not None and callable(cmd):
            cmd("0.0", "1.0")
            chk(not barra.empaquetado, "cabe todo: la barra no debe verse")
            cmd("0.0", "0.4")
            chk(barra.empaquetado and barra.llamadas and barra.llamadas[-1].get("side") == "right"
                and barra.opts.get("_set") == ("0.0", "0.4"), "no cabe: la barra debe verse a la derecha y recibir set()")
            n_llamadas = len(barra.llamadas)
            cmd("0.1", "0.5")
            chk(len(barra.llamadas) == n_llamadas, "no debe re-empaquetarse en cada aviso")
            cmd("0.0", "1.0")
            chk(not barra.empaquetado, "vuelve a caber: la barra se oculta")
        # autodesplazamiento: solo si el usuario estaba abajo
        lienzo.yview = lambda: (0.2, 0.6)
        lienzo.opts.pop("_yview", None)
        g._agregar_evento("EVENTO LEJOS", "exito")
        chk("_yview" not in lienzo.opts, "el usuario estaba leyendo arriba: no se le debe mover la vista")
        lienzo.yview = lambda: (0.5, 1.0)
        g._agregar_evento("EVENTO ABAJO", "exito")
        chk(lienzo.opts.get("_yview") == 1.0, "el usuario estaba abajo: la vista debe seguir al final")
    finally:
        mod.tk = viejo_tk
    # -- recuadro de la Tierra (artistas reales sobre un Figure Agg) --
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    chk(hasattr(I, "_crear_artistas_inset") and hasattr(I, "_actualizar_inset"), "faltan los métodos del recuadro")
    if hasattr(I, "_crear_artistas_inset") and hasattr(I, "_actualizar_inset"):
        h = object.__new__(I)
        fig = Figure(figsize=(6, 6)); FigureCanvasAgg(fig)
        h.fig, h.ax = fig, fig.add_subplot(111)
        h.ax.set_aspect("equal")
        h._crear_artistas_tierra(); h._crear_artistas_trayectoria(); h._crear_artistas_luna_y_hud()
        h._crear_artistas_inset()
        chk(h.ax_inset.get_visible() is False and h._inset_visible is False, "el recuadro empieza oculto")
        with __import__("contextlib").redirect_stdout(__import__("io").StringIO()):
            sim = mod.Simulacion(mod.Config())
            for _ in range(3000):
                sim.avanzar_paso()
        px, py = sim.posicion_display_eci()
        luna = {"x": 3.84e8, "y": 0.0}
        R = mod.R_T

        def lim(x0, x1, y0, y1):
            h.ax.set_xlim(x0, x1); h.ax.set_ylim(y0, y1)
        lim(-1.5 * R, 1.5 * R, -1.5 * R, 1.5 * R)
        h._actualizar_inset(sim, (px, py), luna, (0.0, 0.0))
        chk(not h._inset_visible and not h.ax_inset.get_visible(), "Tierra entera en el mapa: sin recuadro")
        lim(px - 0.5 * R, px + 0.5 * R, py - 0.5 * R, py + 0.5 * R)
        h._actualizar_inset(sim, (px, py), luna, (0.0, 0.0))
        chk(h._inset_visible and h.ax_inset.get_visible(), "Tierra cortada: debe verse el recuadro")
        x0, x1 = h.ax_inset.get_xlim()
        chk(math.isfinite(x1) and x1 > R and x1 < 5 * R and h.ax_inset.get_ylim() == (x0, x1),
            f"la Luna lejana no debe aplastar la escala del recuadro: {x0, x1}")
        xn, yn = h.inset_nave.get_data()
        ex, ey = (float(v) for v in mod.vista_xy(px, py, sim.config.escala_log))
        chk(len(xn) == 1 and abs(float(xn[0]) - ex) < 1e-6 and abs(float(yn[0]) - ey) < 1e-6,
            f"la nave en el recuadro: {xn, yn} vs {ex, ey}")
        chk(len(h.inset_tray.get_xdata()) > 2 and len(h.inset_tray.get_xdata()) <= h.INSET_MAX_PUNTOS + 2,
            f"trayectoria del recuadro acotada: {len(h.inset_tray.get_xdata())}")
        fig.canvas.draw()                                            # dibuja sin errores con el recuadro
        # histéresis: Tierra entera pero justa (entre 1.0 y 1.15)
        lim(-1.08 * R, 1.08 * R, -1.08 * R, 1.08 * R)
        h._actualizar_inset(sim, (px, py), luna, (0.0, 0.0))
        chk(h._inset_visible, "histéresis: si ya se veía, en la franja intermedia sigue visible")
        h._ocultar_inset()
        h._actualizar_inset(sim, (px, py), luna, (0.0, 0.0))
        chk(not h._inset_visible, "histéresis: si estaba oculto, en la franja intermedia sigue oculto")
        # Tierra desplazada (vista baricéntrica): el recuadro compara contra la Tierra dibujada
        lim(5e6 - 1.5 * R, 5e6 + 1.5 * R, -1.5 * R, 1.5 * R)
        h._actualizar_inset(sim, (px, py), luna, (5e6, 0.0))
        chk(not h._inset_visible, "Tierra desplazada pero entera: sin recuadro")
        h._actualizar_inset(sim, (px, py), luna, (0.0, 0.0))
        chk(h._inset_visible, "la misma ventana sin desplazamiento deja la Tierra fuera: con recuadro")
        # poda: el recuadro no une el ascenso con la última vuelta
        n = len(sim.tray_wx)
        sim._orbita_podada, sim._idx_inicio_orbita = True, n // 2
        h._actualizar_inset(sim, (px, py), luna, (0.0, 0.0))
        xs = list(h.inset_tray.get_xdata())
        chk(sum(1 for v in xs if isinstance(v, float) and math.isnan(v)) == 1, "con poda: un NaN en el recuadro")
        # ocultar: limpia y esconde
        h._ocultar_inset()
        chk(not h.ax_inset.get_visible() and len(h.inset_tray.get_xdata()) == 0, "_ocultar_inset")
    if errores:
        print(f"[inset_scroll] FALLA ({len(errores)}):")
        for e in errores:
            print("   ", e)
        return False
    print("[inset_scroll] OK (barra de la línea de tiempo, autodesplazamiento, recuadro de la Tierra)")
    return True


def verificar_resumen_poda(archivo: str) -> bool:
    """(12.11.1) La gráfica de vuelo no tiene hueco tras la poda: los puntos borrados se resumen
    (~60 por periodo, con tope y diezmado) y el drawer los empalma entre ascenso y última vuelta."""
    mod = cargar_modulo(archivo)
    I = getattr(mod, "InterfazSimulacion", None)
    if I is None:
        print("[resumen_poda] omitido")
        return True
    import contextlib, io, math
    import numpy as np
    from types import SimpleNamespace as NS
    from unittest.mock import MagicMock
    errores = []

    def chk(c, m):
        if not c:
            errores.append(m)
    f = getattr(mod, "_serie_con_resumen", None)
    chk(f is not None, "falta _serie_con_resumen")
    if f:
        from array import array
        h = array("d", [0.0, 1.0, 2.0, 3.0, 10.0, 11.0])
        r = f(h, [4.0, 6.0, 8.0], 4, 1)
        chk(r == [0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0, 11.0], f"empalme sin submuestreo: {r}")
        r = f(h, [4.0, 6.0, 8.0], 4, 2, 1)
        chk(r[0] == 0.0 and 4.0 in r and 6.0 in r and 8.0 in r and r[-1] == 11.0 and r == sorted(r), f"empalme con paso: {r}")
        chk(f(h, [], 6, 1) == list(h) and f(array("d"), [1.0], 0, 3) == [1.0], "tramos vacíos")
    # -- drawer con un sim de mentira --
    g = object.__new__(I)
    g.drawer_estado = {"abierto": True}
    g.linea_alt, g.linea_vel = MagicMock(), MagicMock()
    g.ax2, g.ax2b, g.canvas2 = MagicMock(), MagicMock(), MagicMock()
    sim = NS(hist_t=[0.0, 1.0, 2.0, 100.0, 101.0], hist_alt=[0.0, 1e3, 2e3, 3e3, 4e3], hist_v=[0.0, 10.0, 20.0, 30.0, 40.0],
             hist_resumen_t=[10.0, 30.0, 60.0], hist_resumen_alt=[5e3, 6e3, 7e3], hist_resumen_v=[50.0, 60.0, 70.0],
             _idx_resumen=3, _orbita_podada=True, _idx_inicio_orbita=3)
    g._actualizar_drawer(sim)
    t_, a_ = g.linea_alt.set_data.call_args.args
    chk(t_ == [0.0, 1.0, 2.0, 10.0, 30.0, 60.0, 100.0, 101.0] and not any(math.isnan(v) for v in t_ + a_),
        f"drawer con resumen: {t_}")
    chk(a_ == [0.0, 1.0, 2.0, 5.0, 6.0, 7.0, 3.0, 4.0], f"altitud (km) con el mismo empalme: {a_}")
    chk(len(g.linea_vel.set_data.call_args.args[0]) == len(t_), "velocidad con el mismo empalme")
    chk(all(math.isfinite(c.args[1]) for c in (g.ax2.set_xlim.call_args, g.ax2.set_ylim.call_args, g.ax2b.set_ylim.call_args)),
        "límites finitos")
    # -- simulación real: poda, resumen acotado, línea continua --
    with contextlib.redirect_stdout(io.StringIO()):
        s0 = mod.Simulacion(mod.Config())
        n = 0
        while not s0._orbita_podada and n < 400000:
            s0.avanzar_paso(); n += 1
    chk(s0._orbita_podada, "la misión de prueba no llegó a podar")
    if s0._orbita_podada:
        chk(len(s0.hist_resumen_t) > 0 and s0._idx_resumen is not None, "tras la primera poda debe haber resumen")
        idx0 = s0._idx_resumen
        with contextlib.redirect_stdout(io.StringIO()):
            for _ in range(60000):
                s0.avanzar_paso()
        chk(s0._idx_resumen == idx0, "el punto de empalme debe ser estable")
        res = np.asarray(s0.hist_resumen_t)
        paso_t = s0._periodo_orbital / 60.0
        chk(np.all(np.diff(res) > 0) and 0.9 * paso_t <= float(np.mean(np.diff(res))) <= 1.2 * paso_t + 2 * s0.estado.dt,
            f"espaciado del resumen: media {float(np.mean(np.diff(res))):.1f} s, esperado ~{paso_t:.1f} s")
        chk(len(res) < 3 * (s0.estado.t / paso_t) + 10, f"el resumen no debe crecer más que ~t/espaciado: {len(res)}")
        total = np.concatenate([np.asarray(s0.hist_t)[:idx0], res, np.asarray(s0.hist_t)[idx0:]])
        chk(np.all(np.diff(total) >= 0), "el tiempo completo (ascenso + resumen + última vuelta) debe ser monótono")
        g._actualizar_drawer(s0)
        t_, _ = g.linea_alt.set_data.call_args.args
        chk(not any(math.isnan(v) for v in t_) and all(b >= a for a, b in zip(t_, t_[1:])), "línea continua y ordenada")
        salto = max(b - a for a, b in zip(t_, t_[1:]))
        chk(salto <= 3 * paso_t + 1.0, f"sin hueco: salto máximo {salto:.0f} s (espaciado {paso_t:.0f} s)")
        chk(len(t_) <= 2 * mod.GUI_MAX_PUNTOS, f"puntos dibujados acotados: {len(t_)}")
        # TLI: _idx_inicio_orbita se pone a None pero el resumen sigue empalmado
        inicio_orbita = s0._idx_inicio_orbita
        s0._idx_inicio_orbita = None
        g._actualizar_drawer(s0)
        t2, _ = g.linea_alt.set_data.call_args.args
        chk(t2 == t_, "con _idx_inicio_orbita=None el resumen debe seguir en la gráfica")
        s0._idx_inicio_orbita = inicio_orbita                  # se restituye: sin él no hay más poda
        # tope: diezmado a la mitad, espaciado doble, sigue ordenado
        s0.RESUMEN_MAX_PUNTOS = len(s0.hist_resumen_t) + 3
        p0 = s0._resumen_paso_t
        with contextlib.redirect_stdout(io.StringIO()):
            for _ in range(60000):
                s0.avanzar_paso()
        chk(s0._resumen_paso_t >= 2 * p0 and len(s0.hist_resumen_t) <= s0.RESUMEN_MAX_PUNTOS
            and all(b > a for a, b in zip(s0.hist_resumen_t, s0.hist_resumen_t[1:])),
            f"tope del resumen: paso {p0:.1f}->{s0._resumen_paso_t:.1f}, {len(s0.hist_resumen_t)} puntos")
    # el snapshot de las gráficas finales no incluye el resumen
    chk("hist_resumen_t" not in mod._SERIES_HISTORIAL_ARRAY, "el resumen no es una serie del snapshot/poda")
    if errores:
        print(f"[resumen_poda] FALLA ({len(errores)}):")
        for e in errores:
            print("   ", e)
        return False
    print("[resumen_poda] OK (empalme, resumen acotado, línea continua, tope con diezmado)")
    return True


def verificar_almacen_historial(archivo: str) -> bool:
    """(12.12.0) Lo podado se guarda en disco y no se pierde: unidad del almacén (segmentos, bloques,
    lectura por rangos, limpieza, fallo de disco) y comparación bit a bit contra una corrida sin poda;
    exportación a .npz/.csv y tecla E."""
    mod = cargar_modulo(archivo)
    Alm = getattr(mod, "AlmacenHistorial", None)
    if Alm is None:
        print("[almacen_historial] FALLA: falta AlmacenHistorial")
        return False
    import contextlib, gc, io, json, os, tempfile
    import numpy as np
    from unittest.mock import MagicMock, patch
    errores = []

    def chk(c, m):
        if not c:
            errores.append(m)
    with tempfile.TemporaryDirectory() as tmp:
        # -- unidad --
        a = Alm(directorio=tmp)
        a.BLOQUE = 10
        mk = lambda i0, k: [np.arange(i0, i0 + k, dtype=float) * m for m in (1, 2, 3, 4, 5, 6, 7)]
        t, alt, v, wx, wy, ex, ey = mk(0, 4)
        a.anexar(5, t, alt, v, wx, wy, ex, ey, ["a", "a", "b", "b"])
        a.anexar(5, *mk(4, 3), ["b", "c", "c"])                      # mismo pos, contiguo: un solo segmento
        a.anexar(9, *mk(7, 5), ["c"] * 5)                            # otro pos: segmento nuevo
        chk(a.segmentos == [[5, 0, 7], [9, 7, 12]] and a.n == 12, f"segmentos: {a.segmentos} n={a.n}")
        chk(a.ruta is not None and os.path.exists(a.ruta), "con BLOQUE superado debe existir el archivo")
        d = a.leer()
        chk(d.shape == (12, 8) and np.array_equal(d[:, 0], np.arange(12.0)) and np.array_equal(d[:, 6], np.arange(12.0) * 7),
            f"leer: {d.shape}")
        chk(a.nombres_fase(d[:, 7]) == ["a", "a", "b", "b", "b", "c", "c"] + ["c"] * 5, "códigos de fase")
        r = a.leer_rango(3, 7)
        chk(r.shape == (4, 8) and np.array_equal(r[:, 0], [3.0, 4.0, 5.0, 6.0]), "leer_rango")
        chk(a.leer_rango(5, 5).shape == (0, 8) and a.leer_rango(-3, 2).shape == (2, 8) and a.leer_rango(10, 99).shape == (2, 8),
            "leer_rango con límites fuera/vacío")
        ruta = a.ruta
        a.cerrar()
        chk(not os.path.exists(ruta) and a.n == 0 and a.segmentos == [], "cerrar borra el archivo y reinicia")
        a.cerrar()                                                  # idempotente
        b = Alm(directorio=tmp)
        b.BLOQUE = 2
        b.anexar(0, *mk(0, 3), ["x"] * 3)
        ruta_b = b.ruta
        chk(ruta_b and os.path.exists(ruta_b), "b escribió")
        del b
        gc.collect()
        chk(not os.path.exists(ruta_b), "al recolectar el almacén el archivo temporal debe borrarse")
        # -- fallo de disco --
        c = Alm(directorio=os.path.join(tmp, "no", "existe"))
        c.BLOQUE = 2
        with contextlib.redirect_stdout(io.StringIO()) as buf:
            c.anexar(0, *mk(0, 3), ["x"] * 3)
            c.anexar(3, *mk(3, 3), ["x"] * 3)
        chk(c.ok is False and buf.getvalue().count("No se pudo escribir") == 1, f"un solo aviso: {buf.getvalue()!r}")
        chk(c.leer().shape == (0, 8), "con el disco caído, leer devuelve vacío")
    # -- integración: lo podado + lo de RAM == corrida sin poda --
    N = 40000

    def correr(podar):
        with contextlib.redirect_stdout(io.StringIO()):
            sim = mod.Simulacion(mod.Config())
            if not podar:
                sim._podar_historial_orbital = lambda: None
            else:
                sim._almacen.BLOQUE = 2000                           # varios vaciados durante la corrida
            for _ in range(N):
                sim.avanzar_paso()
        return sim
    sp, ss = correr(True), correr(False)
    chk(sp._orbita_podada, "la corrida de prueba debe podar")
    chk(len(sp.hist_t) < len(ss.hist_t) and sp._almacen.n > 0, "la RAM debe quedar acotada y el resto estar en disco")
    fp, fs = sp.historial_completo(), ss.historial_completo()
    for k in ("hist_t", "hist_alt", "hist_v", "tray_wx", "tray_wy", "tray_ex", "tray_ey"):
        chk(np.array_equal(fp[k], fs[k]), f"{k}: el historial recompuesto no coincide con el de la corrida sin poda")
    chk(fp["hist_fase"] == fs["hist_fase"] and len(fp["hist_t"]) == len(ss.hist_t), "fases y longitud")
    chk(fp["incompleto"] is False, "no debe estar incompleto")
    # -- exportación --
    with tempfile.TemporaryDirectory() as tmp:
        npz = mod.exportar_historial(sp, os.path.join(tmp, "h.npz"))
        z = np.load(npz)
        chk(set(z.files) >= {"hist_t", "hist_alt", "hist_v", "tray_wx", "tray_wy", "tray_ex", "tray_ey", "hist_fase", "meta"},
            f"claves del npz: {z.files}")
        chk(np.array_equal(z["hist_t"], fs["hist_t"]) and len(z["hist_fase"]) == len(fs["hist_t"]), "npz completo")
        meta = json.loads(str(z["meta"]))
        chk(meta["modo_mision"] == sp.config.modo_mision and meta["incompleto"] is False and "config" in meta, f"meta: {list(meta)}")
        csv_ = mod.exportar_historial(sp, os.path.join(tmp, "h.csv"))
        with open(csv_, encoding="utf-8") as f:
            filas = f.read().splitlines()
        chk(filas[0].split(",") == list(mod.EXPORT_COLUMNAS_CSV) and len(filas) == len(fs["hist_t"]) + 1,
            f"csv: {len(filas)} filas")
        ult = filas[-1].split(",")
        chk(float(ult[0]) == float(fs["hist_t"][-1]) and ult[7] == fs["hist_fase"][-1], "última fila del csv")
        sin_ext = mod.exportar_historial(sp, os.path.join(tmp, "otro"))
        chk(sin_ext.endswith(".npz") and os.path.exists(sin_ext), "sin extensión se guarda como .npz")
        # -- ruta automática (12.12.1) --
        from datetime import datetime
        r1 = mod._ruta_exportacion("sat/éli te", ".npz", carpeta=tmp, ahora=datetime(2026, 10, 25, 17, 30, 5))
        chk(r1.name == "historial_sat_li_te_20261025_173005.npz", f"nombre: {r1.name}")
        chk(str(r1.parent) == str(tmp) or r1.parent == __import__("pathlib").Path(tmp), "en la carpeta indicada")
        chk(set(r1.name) <= set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-."), f"nombre seguro: {r1.name}")
        r1.write_bytes(b"x")
        r2 = mod._ruta_exportacion("sat/éli te", ".npz", carpeta=tmp, ahora=datetime(2026, 10, 25, 17, 30, 5))
        chk(r2 != r1 and r2.name.endswith("_2.npz"), f"no sobrescribe: {r2.name}")
        chk(mod._carpeta_proyecto() == __import__("pathlib").Path(archivo).resolve().parent, "carpeta del proyecto = la del .py")
        # -- tecla E / Shift+E (GUI de mentira, sin diálogo) --
        I = mod.InterfazSimulacion
        g = object.__new__(I)
        g.sim = sp
        g._mostrar_mensaje_temporal = MagicMock()
        with patch.object(mod, "_carpeta_proyecto", return_value=__import__("pathlib").Path(tmp)):
            g._on_key(MagicMock(key="e"))
            chk(g._mostrar_mensaje_temporal.call_args.args[1] == mod.COLOR_OK, "tecla e: aviso verde")
            g._on_key(MagicMock(key="E"))
            chk(g._mostrar_mensaje_temporal.call_args.args[1] == mod.COLOR_OK, "tecla E: aviso verde")
            g._on_key(MagicMock(key="e"))
        guardados = sorted(f for f in os.listdir(tmp) if f.startswith("historial_satelital_"))
        npz = [f for f in guardados if f.endswith(".npz")]
        csvs = [f for f in guardados if f.endswith(".csv")]
        chk(len(npz) == 2 and len(csvs) == 1, f"e,E,e deben dejar 2 .npz y 1 .csv sin pisarse: {guardados}")
        if npz and csvs:
            chk(np.array_equal(np.load(os.path.join(tmp, npz[0]))["hist_t"], fs["hist_t"]), "el .npz automático es completo")
            with open(os.path.join(tmp, csvs[0]), encoding="utf-8") as f:
                chk(len(f.read().splitlines()) == len(fs["hist_t"]) + 1, "el .csv automático es completo")
        with patch.object(mod, "_carpeta_proyecto", return_value=__import__("pathlib").Path(tmp) / "no" / "existe"):
            g._mostrar_mensaje_temporal.reset_mock()
            g._exportar_historial()
        chk(g._mostrar_mensaje_temporal.call_args.args[1] == mod.COLOR_TEXTO_ERROR, "carpeta inexistente: aviso en rojo")
        g.sim = None
        g._mostrar_mensaje_temporal.reset_mock()
        g._exportar_historial()
        chk(g._mostrar_mensaje_temporal.call_args.args[1] == mod.COLOR_AVISO, "sin simulación: aviso")
    ruta_sp = sp._almacen.ruta
    sp.cerrar_historial()
    chk(not (ruta_sp and os.path.exists(ruta_sp)), "cerrar_historial borra el archivo")
    if errores:
        print(f"[almacen_historial] FALLA ({len(errores)}):")
        for e in errores:
            print("   ", e)
        return False
    print("[almacen_historial] OK (segmentos, bloques, lectura por rangos, limpieza, fallo de disco, "
          "recomposición idéntica a la corrida sin poda, exportación .npz/.csv, ruta automática, teclas E/Shift+E)")
    return True


def verificar_deepcopy_almacen(archivo: str) -> bool:
    """(12.13.1) copy.deepcopy(Simulacion) funciona con el historial ya podado (archivo del almacén
    abierto): lo usa el planificador de desórbita. La copia tiene su propio almacén y no toca el original."""
    mod = cargar_modulo(archivo)
    import contextlib, copy, io, os
    errores = []

    def chk(c, m):
        if not c:
            errores.append(m)
    with contextlib.redirect_stdout(io.StringIO()):
        sim = mod.Simulacion(mod.Config())
        for _ in range(60000):
            sim.avanzar_paso()
    alm = sim._almacen
    chk(alm._estado["f"] is not None, "la prueba necesita el archivo del almacén abierto")
    n0, ruta0, seg0 = alm.n, alm.ruta, [list(x) for x in alm.segmentos]
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            clon = copy.deepcopy(sim)
    except Exception as exc:                              # noqa: BLE001
        print(f"[deepcopy_almacen] FALLA: deepcopy(Simulacion) con almacén abierto: {type(exc).__name__}: {exc}")
        with contextlib.suppress(Exception):
            sim.cerrar_historial()
        return False
    ca = clon._almacen
    chk(ca is not alm, "la copia comparte el almacén del original")
    chk(ca.n == 0 and ca.ruta is None, "el almacén de la copia debía empezar vacío y sin archivo")
    with contextlib.redirect_stdout(io.StringIO()):
        for _ in range(60000):                            # la copia sigue podando en SU almacén
            clon.avanzar_paso()
    chk(alm.n == n0 and alm.ruta == ruta0 and alm.segmentos == seg0,
        "simular la copia cambió el almacén del original")
    chk(ca.ruta is None or ca.ruta != ruta0, "la copia escribe en el archivo del original")
    chk(ruta0 is not None and os.path.exists(ruta0), "el archivo del original desapareció")
    ruta_c = ca.ruta
    ca.cerrar()
    chk(ruta_c is None or not os.path.exists(ruta_c), "cerrar la copia no borró su archivo")
    chk(os.path.exists(ruta0), "cerrar la copia borró el archivo del original")
    with contextlib.suppress(Exception):
        sim.cerrar_historial()
    if errores:
        print("[deepcopy_almacen] FALLA: " + "; ".join(errores))
        return False
    print("[deepcopy_almacen] OK")
    return True


def verificar_pytest(archivo: str) -> bool:
    """(12.13.4) Corre test_cohete.py (la suite estable, junto a este script) contra `archivo`:
    guardas de 12.13.2-12.13.8 y propiedad de equivalencia margen/corte. Se omite si falta pytest."""
    import subprocess
    suite = Path(__file__).with_name("test_cohete.py")
    if not suite.is_file():
        print("[pytest] omitido: no está test_cohete.py junto a este script")
        return True
    try:
        import pytest  # noqa: F401
    except ImportError:
        print("[pytest] omitido: pytest no está instalado")
        return True
    entorno = dict(os.environ, COHETE_ARCHIVO=str(Path(archivo).resolve()))
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", str(suite)], capture_output=True,
                       text=True, env=entorno, cwd=str(suite.parent))
    ultima = (r.stdout.strip().splitlines() or ["(sin salida)"])[-1]
    if r.returncode != 0:
        print("[pytest] FALLA: " + ultima + "\n" + r.stdout[-1500:])
        return False
    print("[pytest] OK: " + ultima)
    return True


def verificar_fluidez_1213(archivo: str) -> bool:
    """(12.13.0) Fluidez de la GUI: polilíneas en vez de miles de segmentos, gráfica de vuelo
    refrescada a ritmo limitado, lotes de física en rebanadas que ceden a Tk, menos marcas en el mapa
    y repintado al cambiar de pestaña."""
    mod = cargar_modulo(archivo)
    I = getattr(mod, "InterfazSimulacion", None)
    if I is None:
        print("[fluidez_1213] omitido")
        return True
    import contextlib, io
    import numpy as np
    from unittest.mock import MagicMock, patch
    errores = []

    def chk(c, m):
        if not c:
            errores.append(m)
    # -- agrupar_en_polilineas --
    f = getattr(mod, "agrupar_en_polilineas", None)
    chk(f is not None, "falta agrupar_en_polilineas")
    if f:
        rng = np.random.default_rng(7)
        pts = np.cumsum(rng.normal(size=(40, 2)), axis=0)
        segs = np.stack([pts[:-1], pts[1:]], axis=1)                       # 39 segmentos contiguos
        cols = np.zeros((39, 4)); cols[:, 3] = 1.0
        cols[10:25, 0] = 1.0                                               # 3 tramos de color
        segs2 = segs.copy(); segs2[30:] += 100.0                           # discontinuidad (la «cuerda» omitida) en el 30
        for sg, cl, esperadas in ((segs, cols, 3), (segs2, cols, 4), (segs[:1], cols[:1], 1)):
            polis, cp = f(sg, cl)
            chk(len(polis) == esperadas and len(cp) == esperadas, f"polilíneas esperadas {esperadas}, hay {len(polis)}")
            rec = np.concatenate([np.stack([p[:-1], p[1:]], axis=1) for p in polis])
            colr = np.concatenate([np.repeat(c[None], len(p) - 1, axis=0) for p, c in zip(polis, cp)])
            chk(np.array_equal(rec, sg) and np.array_equal(colr, cl), "las polilíneas deben reproducir exactamente los segmentos y colores")
        pv, cv = f(np.empty((0, 2, 2)), np.empty((0, 4)))
        chk(pv == [] and len(cv) == 0, "entrada vacía")
        # el glow nunca une la cola con lo omitido: la discontinuidad parte la polilínea
        polis, _ = f(segs2, cols)
        chk(all(np.max(np.abs(np.diff(p, axis=0))) < 50 for p in polis), "ninguna polilínea cruza la discontinuidad")
    # -- gráfica de vuelo a ritmo limitado --
    g = object.__new__(I)
    g._t_drawer = float("-inf")                      # (12.13.2) lo inicializa __init__
    g.drawer_estado = {"abierto": False}
    sim = type("S", (), {"estado": type("E", (), {"simulacion_terminada": False})()})()
    chk(g._drawer_toca(sim) is False, "cerrada: no se dibuja")
    g.drawer_estado["abierto"] = True
    relojes = iter([100.0, 100.1, 100.39, 100.41, 100.42])
    with patch.object(mod, "_monotonic", side_effect=lambda: next(relojes)):
        chk(g._drawer_toca(sim) is True, "primera vez: se dibuja")
        chk(g._drawer_toca(sim) is False, "a los 0,1 s no toca")
        chk(g._drawer_toca(sim) is False, "a los 0,39 s no toca")
        chk(g._drawer_toca(sim) is True, "a los 0,41 s toca")
    g._t_drawer = float("-inf")
    chk(g._drawer_toca(sim) is True, "al abrirla (-inf) se dibuja")
    sim.estado.simulacion_terminada = True
    chk(g._drawer_toca(sim) is True and g._drawer_toca(sim) is True, "terminada: siempre se dibuja el estado final")
    # -- rebanadas de física --
    from types import SimpleNamespace as NS
    pasos_hechos = [0]
    est = NS(simulacion_terminada=False, fase="ascenso")

    class SimF:
        def __init__(self):
            self.estado = est
            self.config = NS(sim_speed=5, LUNAR_PASOS_MIN_COAST=1, DEORBIT_VEL_MAX_PLAN=1,
                             DEORBIT_PASO_QUEMA_MS=10, DEORBIT_PASO_ESPERA_MS=100)
            self._lunar = False
            self._planificador_externo = None
            self.esperando_plan = False
            self._plan_gen = None
        def avanzar_paso(self):
            pasos_hechos[0] += 1
    h = object.__new__(I)
    h.sim = SimF()
    h.root = MagicMock()
    h._refrescar_todo = MagicMock()
    h._refrescar_terminal_sin_graficas = MagicMock()
    h.PRESUPUESTO_FISICA_S = 0.0                        # cada rebanada = 1 paso
    for _ in range(4):
        h._paso_simulacion()
    chk(pasos_hechos[0] == 4 and not h._refrescar_todo.called, f"rebanadas: {pasos_hechos[0]} pasos, refresco {h._refrescar_todo.called}")
    chk(h.root.after.call_args.args[0] == 1, "entre rebanadas se reprograma con 1 ms")
    h._paso_simulacion()
    chk(pasos_hechos[0] == 5 and h._refrescar_todo.call_count == 1 and h._pasos_restantes is None,
        f"al completar el lote se refresca una vez: {pasos_hechos[0]} pasos, {h._refrescar_todo.call_count} refrescos")
    chk(h.root.after.call_args.args[0] == 10, "tras el refresco, ritmo normal")
    pasos_hechos[0] = 0
    h.PRESUPUESTO_FISICA_S = 1e9
    h._paso_simulacion()
    chk(pasos_hechos[0] == 5 and h._refrescar_todo.call_count == 2, "con presupuesto holgado: un lote por fotograma (como antes)")
    # la misión termina a mitad de un lote partido: nada de rebanadas colgadas
    pasos_hechos[0] = 0
    h.PRESUPUESTO_FISICA_S = 0.0
    def termina():
        pasos_hechos[0] += 1
        est.simulacion_terminada = pasos_hechos[0] >= 2
    h.sim.avanzar_paso = termina
    h._paso_simulacion()
    h._paso_simulacion()
    chk(h._pasos_restantes is None and h._refrescar_terminal_sin_graficas.called, "fin de misión a mitad de lote")
    # -- marcas del mapa y repintado de pestañas --
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    fig = Figure(figsize=(6, 6)); FigureCanvasAgg(fig)
    viejos = (mod.plt, mod.FigureCanvasTkAgg)
    try:
        mod.plt = MagicMock(); mod.FigureCanvasTkAgg = MagicMock()
        mod.plt.subplots.side_effect = lambda **k: (fig, fig.add_subplot(111))
        c = object.__new__(I)
        c.config = mod.Config()
        c._conectar_eventos_canvas = MagicMock()
        c._construir_canvas_animacion(MagicMock(), 680, 720)
    finally:
        mod.plt, mod.FigureCanvasTkAgg = viejos
    c.ax.set_xlim(-9e6, 9e6); c.ax.set_ylim(-9e6, 9e6)
    fig.canvas.draw()
    chk(len(c.ax.get_xticks()) <= 6 and len(c.ax.get_yticks()) <= 6, f"marcas del mapa: {len(c.ax.get_xticks())}, {len(c.ax.get_yticks())}")
    t = object.__new__(I)
    t.root = MagicMock()
    t._pestanas = {"mision": MagicMock(), "eventos": MagicMock()}
    t._botones_pestana = {"mision": MagicMock(), "eventos": MagicMock()}
    t._elegir_pestana("eventos")
    chk(t.root.update_idletasks.called, "elegir pestaña debe repintar (update_idletasks)")
    if errores:
        print(f"[fluidez_1213] FALLA ({len(errores)}):")
        for e in errores:
            print("   ", e)
        return False
    print("[fluidez_1213] OK (polilíneas, gráfica de vuelo a ritmo limitado, rebanadas de física, marcas del mapa, pestañas)")
    return True


def comparar(ref: dict, nuevo: dict, ruta: str = "", rtol: float | None = None) -> list[str]:
    """Compara recursivamente con tolerancia; devuelve la lista de diferencias."""
    dif = []
    if isinstance(ref, dict):
        if set(ref) != set(nuevo):
            dif.append(f"{ruta}: claves distintas "
                       f"{sorted(set(ref) ^ set(nuevo))}")
        # (12.13.8) rp y ra de una órbita casi parabólica (escape) están mal condicionados:
        # un redondeo de 1e-16 en el estado los mueve ~1e-7 entre plataformas, con
        # posiciones y velocidades idénticas. Se comparan con 1e-5 solo ahí.
        ecc = ref.get("ecc")
        casi_parabolica = (isinstance(ecc, (int, float)) and not isinstance(ecc, bool)
                           and ecc >= 1.0 - 1e-5)
        for k in ref:
            if k in nuevo:
                rt = max(RTOL, 1e-5) if (casi_parabolica and k in ("rp", "ra")) else rtol
                dif += comparar(ref[k], nuevo[k], f"{ruta}/{k}", rt)
    elif isinstance(ref, list):
        if len(ref) != len(nuevo):
            dif.append(f"{ruta}: longitud {len(ref)} vs {len(nuevo)}")
        for i, (a, b) in enumerate(zip(ref, nuevo)):
            dif += comparar(a, b, f"{ruta}[{i}]", rtol)
    elif isinstance(ref, (int, float)) and not isinstance(ref, bool) \
            and isinstance(nuevo, (int, float)):
        if not np.isclose(ref, nuevo, rtol=RTOL if rtol is None else rtol, atol=ATOL):
            dif.append(f"{ruta}: {ref!r} vs {nuevo!r}")
    elif ref != nuevo:
        dif.append(f"{ruta}: {ref!r} vs {nuevo!r}")
    return dif


def verificar_datos_pymcel(archivo: str) -> bool:
    """(12.3.2) `_preparar_datos_pymcel` copia los datos de pymcel a
    `pymcel/data/` sin `cp` (Windows), solo lo que falta y sin pisar nada."""
    import tempfile
    mod = cargar_modulo(archivo)
    if not hasattr(mod, "_preparar_datos_pymcel"):     # (12.13.8) motores anteriores a 12.3.2
        print("[datos_pymcel] omitido (el archivo no tiene _preparar_datos_pymcel)")
        return True
    ok = True
    cwd0 = os.getcwd()
    with tempfile.TemporaryDirectory() as td:
        origen = Path(td) / "pkg" / "data"
        origen.mkdir(parents=True)
        (origen / "kernels.txt").write_text("K")
        (origen / "de430.tpc").write_text("T")
        trabajo = Path(td) / "trabajo"
        (trabajo / "pymcel" / "data").mkdir(parents=True)
        (trabajo / "pymcel" / "data" / "de430.tpc").write_text("PROPIO")
        viejo = getattr(mod.pc, "ROOTDIR", None)
        try:
            os.chdir(trabajo)
            mod.pc.ROOTDIR = str(Path(td) / "pkg")
            mod._preparar_datos_pymcel()
            mod._preparar_datos_pymcel()          # idempotente
            d = trabajo / "pymcel" / "data"
            if (d / "kernels.txt").read_text() != "K":
                print("[datos_pymcel] FALLA: kernels.txt no se copió"); ok = False
            if (d / "de430.tpc").read_text() != "PROPIO":
                print("[datos_pymcel] FALLA: se pisó un archivo existente"); ok = False
            # sin carpeta pymcel/data previa
            trabajo2 = Path(td) / "t2"; trabajo2.mkdir()
            os.chdir(trabajo2)
            mod._preparar_datos_pymcel()
            if not (trabajo2 / "pymcel" / "data" / "kernels.txt").is_file():
                print("[datos_pymcel] FALLA: no creó pymcel/data"); ok = False
        finally:
            os.chdir(cwd0)
            if viejo is not None:
                mod.pc.ROOTDIR = viejo
    print(f"[datos_pymcel] {'OK' if ok else 'FALLA'} (copia con shutil, idempotente, sin pisar)")
    return ok


def _estado3d_falso(inc_deg: float = 5.1):
    """Fetch 3D artificial (Luna a 384 400 km inclinada, Sol a 1 UA)."""
    incl = math.radians(inc_deg)

    def fetch(cuerpo, fecha_iso, timeout=0):
        if cuerpo == "luna":
            R, w, a = 384.4e6, 2 * math.pi / (27.321661 * 86400), 0.5
            return (R * math.cos(a), R * math.sin(a) * math.cos(incl), R * math.sin(a) * math.sin(incl),
                    -R * w * math.sin(a), R * w * math.cos(a) * math.cos(incl),
                    R * w * math.cos(a) * math.sin(incl))
        R, w, a = 1.496e11, 2 * math.pi / (365.25 * 86400), 1.0
        return (R * math.cos(a), R * math.sin(a), 0.0,
                -R * w * math.sin(a), R * w * math.cos(a), 0.0)
    return fetch


def verificar_horizons_verlet(archivo: str) -> bool:
    """(12.4.0) Fuente de efemérides lunar «horizons_verlet»: parser de
    Horizons, numérica del Verlet 3D contra DOP853 del mismo modelo, contrato
    de consulta arbitraria, política estricta y una misión completa."""
    mod = cargar_modulo(archivo)
    if not hasattr(mod, "_ProveedorHorizonsVerlet"):
        print("[horizons_verlet] omitido (el archivo no tiene la fuente)")
        return True
    errores: list[str] = []

    def chk(c, t):
        if not c:
            errores.append(t)

    mod._TERCEROS_CUERPOS_FUENTE = "sin_pymcel"
    if hasattr(mod, "_EPH_DISCO"):
        mod._EPH_DISCO = None
    fetch0 = mod._fetch_horizons_state_3d

    # 1) parser
    bueno = ("header\n$$SOE\n2461041.5, A.D. 2026-Jan-01 00:00:00.0000, "
             "-2.0E+05, 3.0E+05, 4.0E+04, -1.0, -0.5, 0.1, 1.28, 3.8E+05, 0.0,\n$$EOE\n")
    est = mod._parsear_vectores_horizons(bueno, "luna")
    chk(abs(est[0] + 2.0e8) < 1e-6 and abs(est[5] - 100.0) < 1e-9, f"parser: {est}")
    for nombre, txt, cuerpo in (
            ("sin SOE", "nada", "luna"),
            ("error de Horizons", "No ephemeris for target\n$$SOE\n$$EOE", "luna"),
            ("línea corta", "$$SOE\n1, 2, 3\n$$EOE", "luna"),
            ("distancia Luna", bueno.replace("-2.0E+05, 3.0E+05", "-2.0E+03, 3.0E+03"), "luna"),
            ("no numérico", bueno.replace("-1.0,", "abc,"), "luna")):
        try:
            mod._parsear_vectores_horizons(txt, cuerpo)
            errores.append(f"parser: «{nombre}» debía fallar")
        except RuntimeError:
            pass

    # 2) numérica: Verlet 3D vs DOP853 del mismo modelo (< 10 m en 10 d)
    mod._fetch_horizons_state_3d = _estado3d_falso()
    prov = mod._ProveedorHorizonsVerlet("2026-01-01T00:00:00", 12 * 86400.0)
    from scipy.integrate import solve_ivp

    def f(t, y):
        a = np.empty(6)
        mod._acc_sol_luna_3d.py_func(y, a, mod.MU_SOL + mod.MU_T, mod.MU_T + mod.MU_LUNA, mod.MU_SOL)
        return np.concatenate([y[3:6], a[0:3], y[9:12], a[3:6]])
    ts = [3.0e5, 5.0e5, 9.0e5]
    ref = solve_ivp(f, [0, ts[-1] + 1], prov.estado0, method="DOP853", rtol=1e-12, atol=1e-3, t_eval=ts)
    for k, t in enumerate(ts):
        for cuerpo, o in (("luna", 6), ("sol", 0)):
            (x, y), (vx, vy) = prov(cuerpo, t)
            e = math.hypot(x - ref.y[o, k], y - ref.y[o + 1, k])
            chk(e < 10.0, f"Verlet vs DOP853 {cuerpo} t={t:g}: {e:.3g} m")
    # el estado inicial se reproduce y la proyección descarta z
    (x0, y0), _ = prov("luna", 0.0)
    f0 = mod._fetch_horizons_state_3d("luna", "")
    chk(abs(x0 - f0[0]) < 1e-3 and abs(y0 - f0[1]) < 1e-3, "t=0 no reproduce el estado de Horizons")
    chk(abs(prov.estado0[8]) > 1e6, "la Luna de prueba debía tener z != 0")
    # 3) contrato: orden libre, repetible, fuera de tabla = error explícito
    a = prov("luna", 8e5); prov("luna", 1e3); prov("sol", 5e5)
    chk(a == prov("luna", 8e5), "consulta en desorden no determinista")
    for t in (-100.0, 12 * 86400.0 + 2e3):
        try:
            prov("luna", t)
            errores.append(f"t={t:g} fuera de la tabla debía lanzar EfemeridesNoValidas")
        except mod.EfemeridesNoValidas:
            pass
    # 4) Config
    chk(mod.Config().efemerides_lunar == "rebound", "el valor por defecto debe ser rebound (12.13.8)")
    for kw, ok in ((dict(efemerides_lunar="otra"), False),
                   (dict(efemerides_lunar="spice"), True),
                   (dict(efemerides_lunar="horizons_verlet"), True),
                   (dict(efemerides_lunar="rebound"), True)):
        c = _cfg_lunar(mod) if kw.get("modo_mision") is None else mod.Config()
        for k, v in kw.items():
            setattr(c, k, v)
        try:
            c._validar_config()
            chk(ok, f"Config {kw}: debía rechazarse")
        except ValueError:
            chk(not ok, f"Config {kw}: se rechazó siendo válida")
    # 4b) GUI: campo de fecha (texto) en la lista de campos y conversor que recorta
    gui = getattr(mod, "InterfazSimulacion", None)
    if gui is not None:
        chk(any(k == "fecha_lanzamiento" for k, _ in gui.CAMPOS_NUMERICOS), "GUI: falta el campo de fecha")
        conv = dict(gui.CONVERSORES_CAMPOS).get("fecha_lanzamiento")
        chk(conv is not None and conv(" 2026-03-05T12:30:00 ") == "2026-03-05T12:30:00", "GUI: conversor de fecha")
    # 5) política estricta: si Horizons falla, el modo lunar se detiene
    def roto(*a, **k):
        raise RuntimeError("Horizons: fallo de red/URL: sin red")
    mod._fetch_horizons_state_3d = roto
    try:
        mod.Simulacion(_cfg_lunar(mod, efemerides_lunar="horizons_verlet"))
        errores.append("Horizons caído: el modo lunar debía detenerse")
    except mod.EfemeridesNoValidas as e:
        chk("Kepler" in str(e) and "Horizons" in str(e), f"mensaje poco claro: {e}")
    # 6) misión completa con la fuente (etiqueta honesta, SOI exacta)
    import contextlib, io
    mod._fetch_horizons_state_3d = _estado3d_falso()
    cfg = _cfg_lunar(mod, efemerides_lunar="horizons_verlet")
    with contextlib.redirect_stdout(io.StringIO()):
        sim = mod.Simulacion(cfg)
        n = 0
        while not sim.estado.simulacion_terminada and n < 400_000:
            sim.avanzar_paso()
            n += 1
    ev = sim.eventos
    chk("Horizons + Verlet" in sim.efemerides_efectiva() and "no es SPICE" in sim.efemerides_efectiva(),
        f"etiqueta: {sim.efemerides_efectiva()}")
    chk(ev.mision_cumplida is True and ev.entro_soi_luna,
        f"misión Horizons+Verlet: no entró a la SOI ({ev.datos_mision.get('motivo')})")
    if ev.entro_soi_luna:
        chk(abs(ev.datos_soi_luna["d"] - mod.R_SOI_LUNA) < 1e-3, "SOI no exacta con la tabla")
    # 6b) (12.13.8) misma misión con la fuente REBOUND (sin kernels: Horizons simulado, sin planetas)
    if hasattr(mod, "_ProveedorRebound"):
        try:
            import rebound  # noqa: F401
        except ImportError:
            print("[horizons_verlet] REBOUND no instalado: se omite la misión con rebound")
        else:
            def sin_kernels():
                raise OSError("sin kernels en la prueba")
            mod._asegurar_kernels_spice = sin_kernels
            cfg = _cfg_lunar(mod, efemerides_lunar="rebound")
            with contextlib.redirect_stdout(io.StringIO()):
                sim_r = mod.Simulacion(cfg)
                n = 0
                while not sim_r.estado.simulacion_terminada and n < 400_000:
                    sim_r.avanzar_paso()
                    n += 1
            evr = sim_r.eventos
            chk("REBOUND" in sim_r.efemerides_efectiva() and "sin planetas" in sim_r.efemerides_efectiva(),
                f"etiqueta REBOUND: {sim_r.efemerides_efectiva()}")
            chk(evr.mision_cumplida is True and evr.entro_soi_luna,
                f"misión REBOUND: no entró a la SOI ({evr.datos_mision.get('motivo')})")
            if evr.entro_soi_luna:
                chk(abs(evr.datos_soi_luna["d"] - mod.R_SOI_LUNA) < 1e-3, "REBOUND: SOI no exacta con la tabla")
    mod._fetch_horizons_state_3d = fetch0
    if errores:
        for e in errores:
            print("[horizons_verlet] FALLA:", e)
        return False
    print("[horizons_verlet] OK (parser, Verlet 3D vs DOP853 < 10 m, contrato, política estricta, misión)")
    return True


def main() -> int:
    global RTOL
    ap = argparse.ArgumentParser()
    ap.add_argument("archivo", nargs="?",
                    default=str(Path(__file__).resolve().parents[1] / "src" / "simulador" / "cohete.py"),
                    help="motor a probar (por defecto src/simulador/cohete.py)")
    ap.add_argument("--guardar", action="store_true",
                    help="regenerar la referencia con este archivo")
    ap.add_argument("--escenarios", nargs="*", default=list(ESCENARIOS))
    ap.add_argument("--ref", default=str(REF),
                    help="archivo de referencia (por defecto regresion_ref.json)")
    ap.add_argument("--solo-ref", action="store_true",
                    help="(12.13.8) solo escenarios y referencia, sin las verificaciones "
                         "funcionales; para generar referencias con motores históricos")
    ap.add_argument("--rtol", type=float, default=RTOL,
                    help="tolerancia relativa (por defecto 1e-9: 'no cambió "
                         "la física'; usar ~1e-7 al refactorizar el integrador)")
    args = ap.parse_args()
    RTOL = args.rtol
    if args.solo_ref:
        for _nombre in [n for n in globals() if n.startswith("verificar_")]:
            globals()[_nombre] = lambda *a, **k: True

    arbol_ok = verificar_arbol(cargar_modulo(args.archivo))
    arbol_ok = verificar_pymcel_spice(cargar_modulo(args.archivo)) and arbol_ok
    arbol_ok = verificar_cache_disco(cargar_modulo(args.archivo)) and arbol_ok

    ref_path = Path(args.ref)
    resultados = {}
    for nombre in args.escenarios:
        mod = cargar_modulo(args.archivo)   # módulo limpio por escenario
        if (ESCENARIOS[nombre].get("_solo_v11")
                and not hasattr(mod.Simulacion, "es_alta_energia")):
            print(f"[{nombre}] omitido (el archivo no tiene este modo)")
            continue
        t0 = time.time()
        resultados[nombre] = correr(mod, nombre, ESCENARIOS[nombre])
        r = resultados[nombre]["resumen"]
        print(f"[{nombre}] {r['pasos']} pasos, terminada={r['terminada']}, "
              f"órbita={r['orbita_detectada']}, efem={r['efemerides']}  "
              f"({time.time() - t0:.1f} s)")

    arbol_ok = verificar_deorbit_sitio(mod, resultados) and arbol_ok
    arbol_ok = verificar_fisica_17(mod, resultados) and arbol_ok
    arbol_ok = verificar_propulsion_18(mod) and arbol_ok
    arbol_ok = verificar_dv_prop(mod) and arbol_ok
    arbol_ok = verificar_legacy(resultados) and arbol_ok
    arbol_ok = verificar_compat_v122(resultados) and arbol_ok
    arbol_ok = verificar_dinamica_v12(mod) and arbol_ok
    arbol_ok = verificar_eventos_v12(args.archivo) and arbol_ok
    arbol_ok = verificar_plan_por_tramos(args.archivo) and arbol_ok
    arbol_ok = verificar_transiciones_v12(args.archivo) and arbol_ok
    arbol_ok = verificar_lunar_v13(args.archivo) and arbol_ok
    arbol_ok = verificar_gui_lunar(args.archivo) and arbol_ok
    arbol_ok = verificar_datos_pymcel(args.archivo) and arbol_ok
    arbol_ok = verificar_horizons_verlet(args.archivo) and arbol_ok
    arbol_ok = verificar_gui_bucle(args.archivo) and arbol_ok
    arbol_ok = verificar_impacto_lunar(args.archivo) and arbol_ok
    arbol_ok = verificar_pestanas_gui(args.archivo) and arbol_ok
    arbol_ok = verificar_drawer_poda(args.archivo) and arbol_ok
    arbol_ok = verificar_inset_y_scroll(args.archivo) and arbol_ok
    arbol_ok = verificar_resumen_poda(args.archivo) and arbol_ok
    arbol_ok = verificar_almacen_historial(args.archivo) and arbol_ok
    arbol_ok = verificar_fluidez_1213(args.archivo) and arbol_ok
    arbol_ok = verificar_pytest(args.archivo) and arbol_ok
    arbol_ok = verificar_deepcopy_almacen(args.archivo) and arbol_ok
    arbol_ok = verificar_canvas_glow(args.archivo) and arbol_ok
    arbol_ok = verificar_tarjetas_gui(args.archivo) and arbol_ok
    arbol_ok = verificar_corte_y_filas_lunar(args.archivo) and arbol_ok
    arbol_ok = verificar_robustez_1284(args.archivo) and arbol_ok
    arbol_ok = verificar_robustez_1291(args.archivo) and arbol_ok
    arbol_ok = verificar_robustez_12101(args.archivo) and arbol_ok
    arbol_ok = verificar_config_congelada(args.archivo) and arbol_ok
    arbol_ok = verificar_auditoria_1259(args.archivo) and arbol_ok
    arbol_ok = verificar_convergencia_lunar(args.archivo) and arbol_ok
    arbol_ok = verificar_apuntado_vuelo(args.archivo) and arbol_ok
    arbol_ok = verificar_loi(args.archivo) and arbol_ok
    arbol_ok = verificar_ventanas(args.archivo) and arbol_ok
    arbol_ok = verificar_protecciones_19(mod) and arbol_ok
    if hasattr(mod, "_plan_deorbit_en_proceso"):
        arbol_ok = verificar_import_hijo(args.archivo) and arbol_ok

    if args.guardar:
        ref_path.write_text(json.dumps(resultados, indent=1), encoding="utf-8")
        print(f"Referencia guardada en {ref_path}")
        return 0 if arbol_ok else 1

    if not ref_path.exists():
        print("No hay referencia: corré primero con --guardar")
        return 2
    ref = json.loads(ref_path.read_text(encoding="utf-8"))
    ok = arbol_ok
    for nombre, res in resultados.items():
        if nombre not in ref:
            print(f"[{nombre}] sin referencia"); ok = False; continue
        # json no distingue tuplas/listas ni NaN: normalizamos vía json
        nuevo = json.loads(json.dumps(res))
        dif = comparar(ref[nombre], nuevo)
        if dif:
            ok = False
            print(f"[{nombre}] DIFERENCIAS ({len(dif)}):")
            for d in dif[:10]:
                print("   ", d)
        else:
            print(f"[{nombre}] OK")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
