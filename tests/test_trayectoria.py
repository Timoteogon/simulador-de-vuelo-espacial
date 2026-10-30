"""Pruebas de la simulación de trayectoria vertical."""

import pytest

from src.simulador.cohete import Cohete
from src.simulador.trayectoria import simular_vuelo_vertical


def cohete_pequeno() -> Cohete:
    """Cohete de prueba: motor pequeño y corta combustión."""
    return Cohete(
        masa_seca=100.0,
        masa_combustible=100.0,
        empuje=5000.0,
        consumo=20.0,
    )


class TestSimulacion:
    def test_parametros_invalidos_lanzan_error(self):
        with pytest.raises(ValueError):
            simular_vuelo_vertical(cohete_pequeno(), dt=0.0)

    def test_apogeo_positivo(self):
        perfil = simular_vuelo_vertical(cohete_pequeno(), t_final=200.0)
        assert perfil.altitud_max > 0.0

    def test_apogeo_mayor_que_estado_final(self):
        """Sin motor tras el apogeo, el cohete debe descender."""
        perfil = simular_vuelo_vertical(cohete_pequeno(), t_final=500.0)
        assert perfil.altitud_max > perfil.altitudes[-1]

    def test_series_coherentes(self):
        perfil = simular_vuelo_vertical(cohete_pequeno(), t_final=100.0)
        assert len(perfil.tiempos) == len(perfil.altitudes) == len(perfil.velocidades)
        assert perfil.tiempos[0] == 0.0

    def test_velocidad_maxima_positiva(self):
        perfil = simular_vuelo_vertical(cohete_pequeno(), t_final=100.0)
        assert perfil.velocidad_max > 0.0