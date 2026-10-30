"""Integradores numéricos para ecuaciones de movimiento.

Implementa métodos de paso fijo para integrar ecuaciones
diferenciales ordinarias de primer orden.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

Vector = list[float]
EstadoFuncion = Callable[[float, Vector], Vector]


class _Dinamica(Protocol):
    def __call__(self, t: float, y: Vector) -> Vector: ...


@dataclass
class ResultadoIntegracion:
    """Almacena los resultados de la integración.

    Attributes:
        tiempos: Lista de instantes de tiempo.
        estados: Lista de vectores de estado en cada instante.
    """

    tiempos: list[float]
    estados: list[Vector]


def _validar(dt: float, t_final: float) -> None:
    """Valida los parámetros de integración.

    Raises:
        ValueError: Si dt o t_final no son positivos.
    """
    if dt <= 0:
        raise ValueError("El paso de tiempo dt debe ser positivo.")
    if t_final <= 0:
        raise ValueError("El tiempo final debe ser positivo.")


def euler(f: _Dinamica, y0: Vector, t0: float, t_final: float,
          dt: float) -> ResultadoIntegracion:
    """Integra con el método de Euler explícito.

    Args:
        f: Función que define la dinámica, dy/dt = f(t, y).
        y0: Estado inicial.
        t0: Tiempo inicial.
        t_final: Tiempo final de la integración.
        dt: Paso de tiempo (positivo).

    Returns:
        Resultado con tiempos y estados en cada paso.

    Raises:
        ValueError: Si dt o t_final no son positivos.
    """
    _validar(dt, t_final)
    t = t0
    y = list(y0)
    tiempos = [t]
    estados = [list(y)]

    while t < t_final:
        dydt = f(t, y)
        y = [yi + dt * fi for yi, fi in zip(y, dydt)]
        t += dt
        tiempos.append(t)
        estados.append(list(y))

    return ResultadoIntegracion(tiempos, estados)


def runge_kutta4(f: _Dinamica, y0: Vector, t0: float, t_final: float,
                 dt: float) -> ResultadoIntegracion:
    """Integra con el método clásico de Runge-Kutta de 4º orden.

    Args:
        f: Función que define la dinámica, dy/dt = f(t, y).
        y0: Estado inicial.
        t0: Tiempo inicial.
        t_final: Tiempo final de la integración.
        dt: Paso de tiempo (positivo).

    Returns:
        Resultado con tiempos y estados en cada paso.

    Raises:
        ValueError: Si dt o t_final no son positivos.
    """
    _validar(dt, t_final)
    t = t0
    y = list(y0)
    tiempos = [t]
    estados = [list(y)]

    while t < t_final:
        k1 = f(t, y)
        y2 = [yi + 0.5 * dt * k for yi, k in zip(y, k1)]
        k2 = f(t + 0.5 * dt, y2)
        y3 = [yi + 0.5 * dt * k for yi, k in zip(y, k2)]
        k3 = f(t + 0.5 * dt, y3)
        y4 = [yi + dt * k for yi, k in zip(y, k3)]
        k4 = f(t + dt, y4)
        y = [yi + (dt / 6.0) * (a + 2 * b + 2 * c + d)
             for yi, a, b, c, d in zip(y, k1, k2, k3, k4)]
        t += dt
        tiempos.append(t)
        estados.append(list(y))

    return ResultadoIntegracion(tiempos, estados)