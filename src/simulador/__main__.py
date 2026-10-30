"""Punto de entrada: ejecuta una simulación de ejemplo.

Uso:
    python -m src.simulador
"""

from .cohete import Cohete
from .trayectoria import simular_vuelo_vertical


def main() -> None:
    """Ejecuta una simulación de vuelo de ejemplo e imprime el resumen."""
    cohete = Cohete(
        masa_seca=1500.0,
        masa_combustible=25000.0,
        empuje=500000.0,
        consumo=250.0,
    )
    perfil = simular_vuelo_vertical(cohete, t_final=300.0, dt=0.1)

    print("=== Simulación de vuelo vertical ===")
    print(f"Apogeo:        {perfil.altitud_max / 1000:.2f} km")
    print(f"Tiempo apogeo: {perfil.tiempo_apogeo:.1f} s")
    print(f"Vel. máxima:   {perfil.velocidad_max:.1f} m/s "
          f"({perfil.velocidad_max * 3.6:.0f} km/h)")
    print(f"Aterrizó:      {'Sí' if perfil.aterrizo else 'No (dentro del tiempo simulado)'}")


if __name__ == "__main__":
    main()