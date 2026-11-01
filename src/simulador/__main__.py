"""Punto de entrada: abre la interfaz gráfica del simulador.

Uso:
    python -m simulador                  # interfaz gráfica (Tkinter)
    python -m simulador --ventanas ...   # ventanas de lanzamiento por consola
"""

import multiprocessing

from .cohete import main

if __name__ == "__main__":
    multiprocessing.freeze_support()    # Windows con ejecutables congelados
    main()
