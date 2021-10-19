"""
| En este archivo hice una clase que se encargará de procesar las físicas del auto.
| Debe ser importado en la tarea principal para que funcione. 
| La idea de separarlo en otro script es para separar las responsabilidades de ambos códigos.

"""

__author__ = "Ignacio Pinto"
__license__ = "MIT"


import numpy as np

# [Adicional]
class Auto:
    def __init__(self,X: float,Y: float,Z: float) -> None:
        self.MAX_SPEED = 6                  # Velocidad máxima que permitiremos alcanzar al auto
        self.CAR_ACCELERATION = 3           # Poder de aceleración del auto, mientras mayor el valor más rápido se acelerará
        self.CAR_ROTATION_SPEED = 2.5       # Poder de rotación del auto, mientras mayor el valor más rápido se rotará
        self.CAR_FRICTION = 1.5             # Fricción del auto, esto  controla que tan rápido decrece la velocidad al no acelerar.
        self.X = X
        self.Y = Y
        self.Z = Z 
        self.direction = 0
        self.speed = 0

        self.acceleration = 0               # Variable para recibir el input del acelerador
        self.steering = 0                   # Variable para recibir el input del manubrio

    def accelerate(self,direction):
        """Método para pisar el acelerador del vehículo"""
        assert (direction in [-1,0,1])
        self.acceleration = self.CAR_ACCELERATION*direction

    def steer(self,direction):
        """Método para mover el volante hacia los lados"""
        assert (direction in [-1,0,1])
        self.steering = direction


    def step(self,dt: float) -> None:
        """Ejecuta un paso físico no-discreto, donde dt es el tiempo que se tiene que procesar."""
        self.X += self.speed*np.sin(self.direction)*dt + 0.5*self.acceleration*(dt**2)
        self.Z += self.speed*np.cos(self.direction)*dt + 0.5*self.acceleration*(dt**2)

        self.speed += self.acceleration*dt
        self.direction += self.steering*self.CAR_ROTATION_SPEED*dt

        # Se le coloca un máximo a la velocidad.
        if abs(self.speed) > self.MAX_SPEED:
            self.speed = self.MAX_SPEED*np.sign(self.speed)

        # Se emula una fricción para que al no estar acelerando o frenando el auto se detenga eventualmente.
        if self.speed != 0: 
                self.speed = np.sign(self.speed)*max(0,abs(self.speed)-self.CAR_FRICTION*dt)







