"""Pruebas de los integradores numéricos."""

import math

import pytest

from src.simulador.integradores import euler, runge_kutta4


def crecimiento(t, y):
    """Dinámica de prueba: y' = y (crecimiento exponencial)."""
    return [y[0]]


def caida_libre(t, y):
    """Dinámica de prueba: caída libre con g = 9.8 m/s^2.

    Estado: [posicion, velocidad].
    """
    return [y[1], -9.8]


class TestValidacion:
    def test_euler_dt_invalido(self):
        with pytest.raises(ValueError):
            euler(crecimiento, [1.0], 0.0, 1.0, -0.1)

    def test_rk4_dt_invalido(self):
        with pytest.raises(ValueError):
            runge_kutta4(crecimiento, [1.0], 0.0, 1.0, 0.0)

    def test_euler_t_final_invalido(self):
        with pytest.raises(ValueError):
            euler(crecimiento, [1.0], 0.0, -1.0, 0.1)


class TestExactitud:
    def test_rk4_exponencial_precision(self):
        """RK4 en y'=y con dt=0.1 debe ser muy preciso (error << 1%)."""
        r = runge_kutta4(crecimiento, [1.0], 0.0, 1.0, 0.1)
        esperado = math.e
        assert r.estados[-1][0] == pytest.approx(esperado, rel=1e-6)

    def test_euler_exponencial_error_acotado(self):
        """Euler en y'=y subestima; el error debe ser menor al 10%."""
        r = euler(crecimiento, [1.0], 0.0, 1.0, 0.1)
        esperado = math.e
        assert r.estados[-1][0] == pytest.approx(esperado, rel=0.1)

    def test_rk4_caida_libre_exacta(self):
        """Con aceleración constante RK4 es exacto (menos redondeo)."""
        dt = 0.1
        t_final = 2.0
        r = runge_kutta4(caida_libre, [100.0, 0.0], 0.0, t_final, dt)
        # Solución exacta: v = -g*t, y = 100 - g*t^2/2
        assert r.estados[-1][1] == pytest.approx(-9.8 * t_final, abs=1e-9)
        assert r.estados[-1][0] == pytest.approx(
            100.0 - 0.5 * 9.8 * t_final ** 2, abs=1e-9
        )


class TestEstructura:
    def test_longitud_de_series(self):
        r = euler(crecimiento, [1.0], 0.0, 1.0, 0.25)
        assert len(r.tiempos) == len(r.estados) == 5  # t = 0, 0.25, 0.5, 0.75, 1.0

    def test_estado_inicial_preservado(self):
        r = runge_kutta4(crecimiento, [1.0], 0.0, 1.0, 0.1)
        assert r.estados[0] == [1.0]
        assert r.tiempos[0] == 0.0