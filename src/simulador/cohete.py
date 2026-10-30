"""Modelo físico del cohete.

Define la clase Cohete con las propiedades relevantes para la
simulación de vuelo: masas, empuje, consumo y geometría.
"""

from dataclasses import dataclass, field


@dataclass
class Cohete:
    """Parámetros físicos de un cohete de una sola etapa.

    Attributes:
        masa_seca: Masa estructural sin combustible, en kg.
        masa_combustible: Masa de combustible inicial, en kg.
        empuje: Empuje constante del motor, en newtons.
        consumo: Flujo másico de combustible, en kg/s.
        area: Área transversal de referencia, en m^2.
        coef_arrastre: Coeficiente de arrastre adimensional.
    """

    masa_seca: float
    masa_combustible: float
    empuje: float
    consumo: float
    area: float = 10.0
    coef_arrastre: float = 0.4

    @property
    def masa_total(self) -> float:
        """Masa total actual del cohete (kg)."""
        return self.masa_seca + self.masa_combustible

    @property
    def tiempo_quemado(self) -> float:
        """Tiempo total de combustión del motor (s).

        Returns:
            Tiempo en el que el combustible se agota a la tasa
            de consumo constante.

        Raises:
            ValueError: Si el consumo no es positivo.
        """
        if self.consumo <= 0:
            raise ValueError("El consumo de combustible debe ser positivo.")
        return self.masa_combustible / self.consumo

    def quemar(self, dt: float) -> float:
        """Consume combustible durante un intervalo dt.

        Args:
            dt: Intervalo de tiempo en segundos (no negativo).

        Returns:
            Combustible consumido en ese intervalo (kg).

        Raises:
            ValueError: Si dt es negativo.
        """
        if dt < 0:
            raise ValueError("El intervalo de tiempo no puede ser negativo.")
        consumido = min(self.masa_combustible, self.consumo * dt)
        self.masa_combustible -= consumido
        return consumido

    def empuje_actual(self, t: float) -> float:
        """Empuje del motor en el tiempo t.

        Args:
            t: Tiempo transcurrido desde el encendido, en segundos.

        Returns:
            Empuje en newtons: cero cuando el combustible se agotó.
        """
        if t <= 0 or t > self.tiempo_quemado:
            return 0.0
        return self.empuje