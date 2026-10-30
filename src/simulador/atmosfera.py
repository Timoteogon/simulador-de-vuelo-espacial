"""Modelo simplificado de la atmósfera terrestre.

Proporciona densidad del aire y presión según una escala de altura
exponencial, usados para calcular el arrastre aerodinámico.
"""

import math

# Constantes del modelo
DENSIDAD_NIVEL_MAR = 1.225  # kg/m^3
ESCALA_ALTURA = 8500.0      # m
G0 = 9.80665                # m/s^2, gravedad a nivel del mar
R_TIERRA = 6371000.0        # m, radio medio terrestre
MASA_TIERRA = 5.972e24      # kg


def densidad(altitud: float) -> float:
    """Densidad del aire a una altitud dada.

    Usa un modelo exponencial: rho = rho0 * exp(-h / H).

    Args:
        altitud: Altitud sobre el nivel del mar, en metros.

    Returns:
        Densidad en kg/m^3 (cero por debajo del nivel del mar).
    """
    if altitud <= 0:
        return DENSIDAD_NIVEL_MAR
    return DENSIDAD_NIVEL_MAR * math.exp(-altitud / ESCALA_ALTURA)


def gravedad(altitud: float) -> float:
    """Aceleración de gravedad a una altitud dada.

    Usa la ley de Newton con simetría esférica:
    g = G*M / r^2, aproximado con g0*(R/(R+h))^2.

    Args:
        altitud: Altitud sobre el nivel del mar, en metros.

    Returns:
        Gravedad en m/s^2 (mínimo cero; constante a nivel del mar o bajo él).
    """
    r = R_TIERRA + max(altitud, 0.0)
    return G0 * (R_TIERRA / r) ** 2


def arrastre(altitud: float, velocidad: float, area: float,
             coef_arrastre: float) -> float:
    """Fuerza de arrastre aerodinámico.

    Modela un flujo estacionario sobre un cuerpo con coeficiente
    de arrastre constante: F = 0.5 * rho * v^2 * Cd * A.

    Args:
        altitud: Altitud en metros.
        velocidad: Rapidez respecto al aire, en m/s.
        area: Área de referencia, en m^2.
        coef_arrastre: Coeficiente de arrastre adimensional.

    Returns:
        Magnitud de la fuerza de arrastre, en newtons (siempre >= 0).
    """
    if velocidad <= 0:
        return 0.0
    return 0.5 * densidad(altitud) * velocidad ** 2 * coef_arrastre * area