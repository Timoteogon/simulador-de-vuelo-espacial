# 🚀 Simulador de Vuelo Espacial

Simulador educativo de trayectorias de cohetes desarrollado en Python, con integración numérica de ecuaciones de movimiento, análisis de fases de vuelo y visualización de resultados.

## ✨ Características

- Simulación de vuelo vertical y trayectorias 2D con gravedad variable
- Modelado de empuje, consumo de combustible y arrastre atmosférico básico
- Integración numérica robusta (Euler y Runge-Kutta 4)
- Registro y exportación de resultados por fases de vuelo
- Suite de pruebas con `pytest`

## 🧪 Requisitos

- Python 3.10 o superior
- Dependencias listadas en `requirements.txt`

## ⚙️ Instalación

```bash
git clone https://github.com/Timoteogon/simulador-de-vuelo-espacial.git
cd simulador-de-vuelo-espacial
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 🚀 Uso rápido

```bash
python -m src.simulador
```

## 🧪 Ejecutar las pruebas

```bash
pip install -r requirements-dev.txt
pytest
```

## 📁 Estructura del proyecto

```
simulador-de-vuelo-espacial/
├── src/
│   └── simulador/
│       ├── __init__.py
│       ├── __main__.py
│       ├── cohete.py
│       ├── atmosfera.py
│       ├── integradores.py
│       └── trayectoria.py
├── tests/
│   ├── test_cohete.py
│   └── test_integradores.py
├── .github/
│   └── workflows/
│       └── ci.yml
├── requirements.txt
├── requirements-dev.txt
├── pyproject.toml
├── LICENSE
└── README.md
```

## 🤝 Contribuir

Las contribuciones son bienvenidas. Abre un issue o un Pull Request.

## 📄 Licencia

Este proyecto está bajo la [Licencia MIT](LICENSE) — `Copyright (c) 2026 Mateo Chavarria Gonzalez`.
