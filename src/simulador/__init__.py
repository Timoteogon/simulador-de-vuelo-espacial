"""Simulador de vuelo espacial.

Simula cohetes de dos etapas en 2D (ascenso, órbita, GTO, TLI, escape, desórbita y
vuelo lunar) con integración numérica, efemérides de Sol y Luna (REBOUND, SPICE u
Horizons+Verlet) y una interfaz gráfica Tkinter. El motor está en `simulador.cohete`.
"""

__version__ = "12.14.0"
