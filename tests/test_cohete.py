"""Pruebas del modelo físico del cohete."""

import pytest

from src.simulador.cohete import Cohete


def crear_cohete() -> Cohete:
    """Fixture: cohete de prueba con parámetros simples."""
    return Cohete(
        masa_seca=1000.0,
        masa_combustible=500.0,
        empuje=100000.0,
        consumo=50.0,
    )


class TestMasa:
    def test_masa_total_inicial(self):
        cohete = crear_cohete()
        assert cohete.masa_total == 1500.0

    def test_masa_total_disminuye_al_quemar(self):
        cohete = crear_cohete()
        cohete.quemar(2.0)
        assert cohete.masa_total == 1400.0

    def test_masa_no_baja_de_cero(self):
        cohete = crear_cohete()
        cohete.quemar(100.0)  # mucho más que el combustible total
        assert cohete.masa_combustible == 0.0
        assert cohete.masa_total == 1000.0


class TestCombustion:
    def test_tiempo_quemado_correcto(self):
        cohete = crear_cohete()
        assert cohete.tiempo_quemado == pytest.approx(10.0)

    def test_consumo_cero_lanza_error(self):
        cohete = crear_cohete()
        cohete.consumo = 0.0
        with pytest.raises(ValueError):
            _ = cohete.tiempo_quemado

    def test_dt_negativo_lanza_error(self):
        cohete = crear_cohete()
        with pytest.raises(ValueError):
            cohete.quemar(-1.0)


class TestEmpuje:
    def test_empuje_durante_quemado(self):
        cohete = crear_cohete()
        assert cohete.empuje_actual(5.0) == 100000.0

    def test_empuje_cero_tras_agotar(self):
        cohete = crear_cohete()
        assert cohete.empuje_actual(20.0) == 0.0

    def test_empuje_cero_antes_del_encendido(self):
        cohete = crear_cohete()
        assert cohete.empuje_actual(-1.0) == 0.0