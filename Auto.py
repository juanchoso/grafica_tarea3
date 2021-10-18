"""
| En este archivo hice una clase que se encargará de procesar las físicas del auto.
| Debe ser importado en la tarea principal para que funcione. 
| La idea de separarlo en otro script es para separar las responsabilidades de ambos códigos.

"""

__author__ = "Ignacio Pinto"
__license__ = "MIT"


from tarea3_v1 import CAR_SPEED
import numpy as np


class Auto:
    def __init__(self,X: float,Y: float,Z: float) -> None:
        self.X = X
        self.Y = Y
        self.Z = Z 
        self.direction = 0
        self.speed = 0

    def step(self,dt: float) -> None:
        """Ejecuta un paso físico no-discreto, donde dt es el tiempo que se tiene que procesar."""
        self.X += dt*CAR_SPEED*np.sin(self.direction)
        self.Z += dt*CAR_SPEED*np.cos(self.direction)

        





