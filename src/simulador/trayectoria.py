"""Simulación de la trayectoria de vuelo vertical.

Combina el modelo del cohete, la atmósfera y los integradores
numéricos para producir un perfil completo de vuelo.
"""

from dataclasses import dataclass

from .atmosfera import arrastre, gravedad
from .cohete import Cohete
from .integradores import runge_kutta4


@dataclass
class PerfilVuelo:
    """Resultados clave de una simulación de vuelo.

    Attributes:
        altitud_max: Apogeo alcanzado, en metros.
        tiempo_apogeo: Tiempo hasta el apogeo, en segundos.
        velocidad_max: Rapidez máxima durante el vuelo, en m/s.
        aterrizo: True si el cohete regresa al suelo dentro de la simulación.
        tiempos: Instantes de tiempo de la simulación.
        altitudes: Altitud en cada instante, en metros.
        velocidades: Rapidez en cada instante, en m/s.
    """

    altitud_max: float
    tiempo_apogeo: float
    velocidad_max: float
    aterrizo: bool
    tiempos: list[float]
    altitudes: list[float]
    velocidades: list[float]


def simular_vuelo_vertical(cohete: Cohete, t_final: float = 300.0,
                           dt: float = 0.1) -> PerfilVuelo:
    """Simula el vuelo vertical completo de un cohete.

    El estado es [altitud, velocidad] y la dinámica incluye empuje,
    gravedad variable y arrastre atmosférico. La simulación termina
    cuando el cohete regresa al suelo o se alcanza t_final.

    Args:
        cohete: Cohete con sus parámetros físicos.
        t_final: Tiempo máximo de simulación, en segundos.
        dt: Paso de integración, en segundos.

    Returns:
        Perfil de vuelo con apogeo, velocidades y series de tiempo.

    Raises:
        ValueError: Si dt o t_final no son positivos.
    """
    if dt <= 0 or t_final <= 0:
        raise ValueError("dt y t_final deben ser positivos.")

    masa_seca = cohete.masa_seca
    estado = [0.0, 0.0]  # [altitud, velocidad]

    def f(t: float, y: list[float]) -> list[float]:
        altitud, velocidad = y
        masa_actual = masa_seca + max(
            cohete.masa_combustible - cohete.consumo * t, 0.0
        )
        empuje = cohete.empuje_actual(t)
        gravedad_local = gravedad(altitud)
        fuerza_arrastre = arrastre(altitud, abs(velocidad), cohete.area,
                                   cohete.coef_arrastre)
        # Empuje hacia arriba; gravedad y arrastre se oponen al movimiento
        aceleracion = (
            empuje
            - masa_actual * gravedad_local
            - (fuerza_arrastre if velocidad >= 0 else -fuerza_arrastre)
        ) / masa_actual
        return [velocidad, aceleracion]

    resultado = runge_kutta4(f, estado, 0.0, t_final, dt)

    tiempos = resultado.tiempos
    altitudes = [s[0] for s in resultado.estados]
    velocidades = [s[1] for s in resultado.estados]

    aterrizo = False
    for i in range(1, len(tiempos)):
        if altitudes[i] <= 0 and altitudes[i - 1] > 0:
            aterrizo = True
            break

    # Recortar series tras el aterrizaje para no mostrar vuelo "bajo tierra"
    if aterrizo:
        tiempos = tiempos[: i + 1]
        altitudes = altitudes[: i + 1]
        velocidades = velocidades[: i + 1]

    apogeo_idx = max(range(len(altitudes)), key=lambda k: altitudes[k])

    return PerfilVuelo(
        altitud_max=max(altitudes, default=0.0),
        tiempo_apogeo=tiempos[apogeo_idx] if tiempos else 0.0,
        velocidad_max=max((abs(v) for v in velocidades), default=0.0),
        aterrizo=aterrizo,
        tiempos=tiempos,
        altitudes=altitudes,
        velocidades=velocidades,
    )