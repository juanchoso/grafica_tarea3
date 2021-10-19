# coding=utf-8
"""Tarea 3"""

from Auto import Auto
import glfw
from OpenGL.GL import *
import OpenGL.GL.shaders
import numpy as np
import sys
import os.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import grafica.transformations as tr
import grafica.basic_shapes as bs
import grafica.scene_graph as sg
import grafica.easy_shaders as es
import grafica.lighting_shaders as ls
import grafica.performance_monitor as pm
import grafica.text_renderer as tx
from grafica.assets_path import getAssetPath
from operator import add

__author__ = "Ivan Sipiran"
__license__ = "MIT"

# A class to store the application control
class Controller:
    def __init__(self):
        self.fillPolygon = True
        self.showAxis = True
        self.viewPos = np.array([12,12,12])
        self.at = np.array([0,0,0])
        self.camUp = np.array([0, 1, 0])
        self.distance = 20


controller = Controller()

def setPlot(texPipeline, axisPipeline, lightPipeline):
    projection = tr.perspective(45, float(width)/float(height), 0.1, 100)

    glUseProgram(axisPipeline.shaderProgram)
    glUniformMatrix4fv(glGetUniformLocation(axisPipeline.shaderProgram, "projection"), 1, GL_TRUE, projection)

    glUseProgram(texPipeline.shaderProgram)
    glUniformMatrix4fv(glGetUniformLocation(texPipeline.shaderProgram, "projection"), 1, GL_TRUE, projection)

    glUseProgram(lightPipeline.shaderProgram)
    glUniformMatrix4fv(glGetUniformLocation(lightPipeline.shaderProgram, "projection"), 1, GL_TRUE, projection)
    
    glUniform3f(glGetUniformLocation(lightPipeline.shaderProgram, "La"), 1.0, 1.0, 1.0)
    glUniform3f(glGetUniformLocation(lightPipeline.shaderProgram, "Ld"), 1.0, 1.0, 1.0)
    glUniform3f(glGetUniformLocation(lightPipeline.shaderProgram, "Ls"), 1.0, 1.0, 1.0)

    glUniform3f(glGetUniformLocation(lightPipeline.shaderProgram, "Ka"), 0.2, 0.2, 0.2)
    glUniform3f(glGetUniformLocation(lightPipeline.shaderProgram, "Kd"), 0.9, 0.9, 0.9)
    glUniform3f(glGetUniformLocation(lightPipeline.shaderProgram, "Ks"), 1.0, 1.0, 1.0)

    glUniform3f(glGetUniformLocation(lightPipeline.shaderProgram, "lightPosition"), 5, 5, 5)
    
    glUniform1ui(glGetUniformLocation(lightPipeline.shaderProgram, "shininess"), 1000)
    glUniform1f(glGetUniformLocation(lightPipeline.shaderProgram, "constantAttenuation"), 0.1)
    glUniform1f(glGetUniformLocation(lightPipeline.shaderProgram, "linearAttenuation"), 0.1)
    glUniform1f(glGetUniformLocation(lightPipeline.shaderProgram, "quadraticAttenuation"), 0.01)

def setView(texPipeline, axisPipeline, lightPipeline):
    view = tr.lookAt(
            controller.viewPos,
            controller.at,
            controller.camUp
        )

    glUseProgram(axisPipeline.shaderProgram)
    glUniformMatrix4fv(glGetUniformLocation(axisPipeline.shaderProgram, "view"), 1, GL_TRUE, view)

    glUseProgram(texPipeline.shaderProgram)
    glUniformMatrix4fv(glGetUniformLocation(texPipeline.shaderProgram, "view"), 1, GL_TRUE, view)

    glUseProgram(lightPipeline.shaderProgram)
    glUniformMatrix4fv(glGetUniformLocation(lightPipeline.shaderProgram, "view"), 1, GL_TRUE, view)
    glUniform3f(glGetUniformLocation(lightPipeline.shaderProgram, "viewPosition"), controller.viewPos[0], controller.viewPos[1], controller.viewPos[2])
    

def on_key(window, key, scancode, action, mods):

    if action != glfw.PRESS:
        return
    
    global controller

    if key == glfw.KEY_SPACE:
        controller.fillPolygon = not controller.fillPolygon

    elif key == glfw.KEY_LEFT_CONTROL:
        controller.showAxis = not controller.showAxis

    elif key == glfw.KEY_ESCAPE:
        glfw.set_window_should_close(window, True)
    
    elif key == glfw.KEY_1:
        controller.viewPos = np.array([controller.distance,controller.distance,controller.distance]) #Vista diagonal 1
        controller.camUp = np.array([0,1,0])
    
    elif key == glfw.KEY_2:
        controller.viewPos = np.array([0,0,controller.distance]) #Vista frontal
        controller.camUp = np.array([0,1,0])

    elif key == glfw.KEY_3:
        controller.viewPos = np.array([controller.distance,0,controller.distance]) #Vista lateral
        controller.camUp = np.array([0,1,0])

    elif key == glfw.KEY_4:
        controller.viewPos = np.array([0,controller.distance,0]) #Vista superior
        controller.camUp = np.array([1,0,0])
    
    elif key == glfw.KEY_5:
        controller.viewPos = np.array([controller.distance,controller.distance,-controller.distance]) #Vista diagonal 2
        controller.camUp = np.array([0,1,0])
    
    elif key == glfw.KEY_6:
        controller.viewPos = np.array([-controller.distance,controller.distance,-controller.distance]) #Vista diagonal 2
        controller.camUp = np.array([0,1,0])
    
    elif key == glfw.KEY_7:
        controller.viewPos = np.array([-controller.distance,controller.distance,controller.distance]) #Vista diagonal 2
        controller.camUp = np.array([0,1,0])
    else:
        print('Unknown key')

def createOFFShape(pipeline, filename, r,g, b):
    shape = readOFF(getAssetPath(filename), (r, g, b))
    gpuShape = es.GPUShape().initBuffers()
    pipeline.setupVAO(gpuShape)
    gpuShape.fillBuffers(shape.vertices, shape.indices, GL_STATIC_DRAW)

    return gpuShape

def readOFF(filename, color):
    vertices = []
    normals= []
    faces = []

    with open(filename, 'r') as file:
        line = file.readline().strip()
        assert line=="OFF"

        line = file.readline().strip()
        aux = line.split(' ')

        numVertices = int(aux[0])
        numFaces = int(aux[1])

        for i in range(numVertices):
            aux = file.readline().strip().split(' ')
            vertices += [float(coord) for coord in aux[0:]]
        
        vertices = np.asarray(vertices)
        vertices = np.reshape(vertices, (numVertices, 3))
        print(f'Vertices shape: {vertices.shape}')

        normals = np.zeros((numVertices,3), dtype=np.float32)
        print(f'Normals shape: {normals.shape}')

        for i in range(numFaces):
            aux = file.readline().strip().split(' ')
            aux = [int(index) for index in aux[0:]]
            faces += [aux[1:]]
            
            vecA = [vertices[aux[2]][0] - vertices[aux[1]][0], vertices[aux[2]][1] - vertices[aux[1]][1], vertices[aux[2]][2] - vertices[aux[1]][2]]
            vecB = [vertices[aux[3]][0] - vertices[aux[2]][0], vertices[aux[3]][1] - vertices[aux[2]][1], vertices[aux[3]][2] - vertices[aux[2]][2]]

            res = np.cross(vecA, vecB)
            normals[aux[1]][0] += res[0]  
            normals[aux[1]][1] += res[1]  
            normals[aux[1]][2] += res[2]  

            normals[aux[2]][0] += res[0]  
            normals[aux[2]][1] += res[1]  
            normals[aux[2]][2] += res[2]  

            normals[aux[3]][0] += res[0]  
            normals[aux[3]][1] += res[1]  
            normals[aux[3]][2] += res[2]  
        #print(faces)
        norms = np.linalg.norm(normals,axis=1)
        normals = normals/norms[:,None]

        color = np.asarray(color)
        color = np.tile(color, (numVertices, 1))

        vertexData = np.concatenate((vertices, color), axis=1)
        vertexData = np.concatenate((vertexData, normals), axis=1)

        print(vertexData.shape)

        indices = []
        vertexDataF = []
        index = 0

        for face in faces:
            vertex = vertexData[face[0],:]
            vertexDataF += vertex.tolist()
            vertex = vertexData[face[1],:]
            vertexDataF += vertex.tolist()
            vertex = vertexData[face[2],:]
            vertexDataF += vertex.tolist()
            
            indices += [index, index + 1, index + 2]
            index += 3        



        return bs.Shape(vertexDataF, indices)

def createGPUShape(pipeline, shape):
    gpuShape = es.GPUShape().initBuffers()
    pipeline.setupVAO(gpuShape)
    gpuShape.fillBuffers(shape.vertices, shape.indices, GL_STATIC_DRAW)

    return gpuShape

def createTexturedArc(d):
    vertices = [d, 0.0, 0.0, 0.0, 0.0,
                d+1.0, 0.0, 0.0, 1.0, 0.0]
    
    currentIndex1 = 0
    currentIndex2 = 1

    indices = []

    cont = 1
    cont2 = 1

    for angle in range(4, 185, 5):
        angle = np.radians(angle)
        rot = tr.rotationY(angle)
        p1 = rot.dot(np.array([[d],[0],[0],[1]]))
        p2 = rot.dot(np.array([[d+1],[0],[0],[1]]))

        p1 = np.squeeze(p1)
        p2 = np.squeeze(p2)
        
        vertices.extend([p2[0], p2[1], p2[2], 1.0, cont/4])
        vertices.extend([p1[0], p1[1], p1[2], 0.0, cont/4])
        
        indices.extend([currentIndex1, currentIndex2, currentIndex2+1])
        indices.extend([currentIndex2+1, currentIndex2+2, currentIndex1])

        if cont > 4:
            cont = 0


        vertices.extend([p1[0], p1[1], p1[2], 0.0, cont/4])
        vertices.extend([p2[0], p2[1], p2[2], 1.0, cont/4])

        currentIndex1 = currentIndex1 + 4
        currentIndex2 = currentIndex2 + 4
        cont2 = cont2 + 1
        cont = cont + 1

    return bs.Shape(vertices, indices)

def createTiledFloor(dim):
    vert = np.array([[-0.5,0.5,0.5,-0.5],[-0.5,-0.5,0.5,0.5],[0.0,0.0,0.0,0.0],[1.0,1.0,1.0,1.0]], np.float32)
    rot = tr.rotationX(-np.pi/2)
    vert = rot.dot(vert)

    indices = [
         0, 1, 2,
         2, 3, 0]

    vertFinal = []
    indexFinal = []
    cont = 0

    for i in range(-dim,dim,1):
        for j in range(-dim,dim,1):
            tra = tr.translate(i,0.0,j)
            newVert = tra.dot(vert)

            v = newVert[:,0][:-1]
            vertFinal.extend([v[0], v[1], v[2], 0, 1])
            v = newVert[:,1][:-1]
            vertFinal.extend([v[0], v[1], v[2], 1, 1])
            v = newVert[:,2][:-1]
            vertFinal.extend([v[0], v[1], v[2], 1, 0])
            v = newVert[:,3][:-1]
            vertFinal.extend([v[0], v[1], v[2], 0, 0])
            
            ind = [elem + cont for elem in indices]
            indexFinal.extend(ind)
            cont = cont + 4

    return bs.Shape(vertFinal, indexFinal)

# TAREA3: Implementa la función "createHouse" que crea un objeto que representa una casa
# y devuelve un nodo de un grafo de escena (un objeto sg.SceneGraphNode) que representa toda la geometría y las texturas
# Esta función recibe como parámetro el pipeline que se usa para las texturas (texPipeline)

def createHouse(pipeline):
    gpuMuro = createGPUShape(pipeline, bs.createTextureQuad(1,1))
    gpuMuro.texture = es.textureSimpleSetup(getAssetPath('wall1.jpg'), GL_REPEAT, GL_REPEAT, GL_LINEAR, GL_LINEAR)
    gpuTecho = createGPUShape(pipeline, bs.createTextureQuad(1,1))
    gpuTecho.texture = es.textureSimpleSetup(getAssetPath('roof2.jpg'), GL_REPEAT, GL_REPEAT, GL_LINEAR, GL_LINEAR)
    
    muro1 = sg.SceneGraphNode('muro1')
    muro1.transform = tr.translate(0,0,0.5)
    muro1.childs += [gpuMuro]

    muro2 = sg.SceneGraphNode('muro2')
    muro2.transform = tr.translate(0,0,-0.5)
    muro2.childs += [gpuMuro]

    muro3 = sg.SceneGraphNode('muro3')
    muro3.transform = tr.matmul([tr.translate(0.5,0,0),tr.rotationY(np.pi/2)])
    muro3.childs += [gpuMuro]

    muro4 = sg.SceneGraphNode('muro4')
    muro4.transform = tr.matmul([tr.translate(-0.5,0,0),tr.rotationY(np.pi/2)])
    muro4.childs += [gpuMuro]

    techo = sg.SceneGraphNode('techo')
    techo.transform = tr.matmul([tr.translate(0,0.5,0),tr.rotationX(np.pi/2)])
    techo.childs += [gpuTecho]

    house = sg.SceneGraphNode('casa')
    # house.transform = tr.matmul([tr.translate(0,0.25,0),tr.uniformScale(0.5)])
    house.childs += [muro1,muro2,muro3,muro4,techo]
    

    casas = []
    # Tuplas:_ (X,Z,ROTACIÓN)
    posiciones = [(4,3,np.pi*0.25), (-4,3,np.pi*0.33), (0,-8,np.pi*0.66), (0,9,np.pi*0.76), (4.6,-13,np.pi*0.66),(-3.8,-11,np.pi*0.66),
    (-6.4,-9,np.pi*0.89),(6.8,-9,np.pi*0.13),(6.8,9,np.pi*0.45),(-7,-9,np.pi*0.89),(7,-3,2.94),(3.35,-6.93,3.8), (-4.63,-4.36,5.47),
    (-5.11,-0.34,6.43),(-4.64,6.06,6.77),(-2.89,11.08,6.21),(3.44,6.98,8.56),(6.46,-0.42,-2.75),(9.08,4.51,0.93),(-0.62,0.14,-2.58),
    (0.25,4.70,5.38),(0.05,-4.76,3.41)]

    for x,z,r in posiciones:
        casa = sg.SceneGraphNode('instancia')
        casa.transform = tr.matmul([tr.translate(x,0.35,z),tr.rotationY(r),tr.uniformScale(0.75)])
        casa.childs += [house]
        casas.append(casa)



    system = sg.SceneGraphNode('system')
    system.childs = casas

    return system

# TAREA3: Implementa la función "createWall" que crea un objeto que representa un muro
# y devuelve un nodo de un grafo de escena (un objeto sg.SceneGraphNode) que representa toda la geometría y las texturas
# Esta función recibe como parámetro el pipeline que se usa para las texturas (texPipeline)
def createWall(pipeline):
    gpuMuro = createGPUShape(pipeline, bs.createTextureQuad(10,0.5))
    gpuMuro.texture = es.textureSimpleSetup(getAssetPath('wall3.jpg'), GL_REPEAT, GL_REPEAT, GL_LINEAR, GL_LINEAR)
    
    gpuMuro_small = createGPUShape(pipeline, bs.createTextureQuad(0.5,0.5))
    gpuMuro_small.texture = es.textureSimpleSetup(getAssetPath('wall3.jpg'), GL_REPEAT, GL_REPEAT, GL_LINEAR, GL_LINEAR)

    gpuMuro_smallR = createGPUShape(pipeline, bs.createTextureQuad(2.5,0.1))
    gpuMuro_smallR.texture = es.textureSimpleSetup(getAssetPath('wall3.jpg'), GL_REPEAT, GL_REPEAT, GL_LINEAR, GL_LINEAR)

    muro1 = sg.SceneGraphNode('muro1')
    muro1.transform = tr.translate(0,0,0.1)
    muro1.childs += [gpuMuro]

    muro2 = sg.SceneGraphNode('muro2')
    muro2.transform = tr.translate(0,0,-0.1)
    muro2.childs += [gpuMuro]

    muro3 = sg.SceneGraphNode('muro3')
    muro3.transform = tr.matmul([tr.translate(0.5,0,0),tr.scale(1,1,0.2),tr.rotationY(np.pi/2)])
    muro3.childs += [gpuMuro_small]

    muro4 = sg.SceneGraphNode('muro3')
    muro4.transform = tr.matmul([tr.translate(-0.5,0,0),tr.scale(1,1,0.2),tr.rotationY(np.pi/2)])
    muro4.childs += [gpuMuro_small]

    techo = sg.SceneGraphNode('techo')
    techo.transform = tr.matmul([tr.translate(0,0.5,0),tr.scale(1,1,0.2),tr.rotationX(np.pi/2)])
    techo.childs += [gpuMuro_smallR]

    contencion = sg.SceneGraphNode('contencion')
    contencion.transform = tr.matmul([tr.translate(2.6,0.2,0.5), tr.rotationY(np.pi/2), tr.scale(10,0.4,1)])
    contencion.childs += [muro1,muro2,muro3,muro4,techo]
    
    copia1 = sg.SceneGraphNode('copia1')
    copia1.transform = tr.translate(-1.2,0,0)
    copia1.childs += [contencion]

    copia2 = sg.SceneGraphNode('copia2')
    copia2.transform = tr.translate(-4,0,0)
    copia2.childs += [copia1,contencion]

    system = sg.SceneGraphNode('system')
    system.childs += [contencion,copia1,copia2]

    return system

# TAREA3: Esta función crea un grafo de escena especial para el auto.
def createCarScene(pipeline):
    chasis = createOFFShape(pipeline, 'alfa2.off', 1.0, 0.0, 0.0)
    wheel = createOFFShape(pipeline, 'wheel.off', 0.0, 0.0, 0.0)

    scale = 2.0
    rotatingWheelNode = sg.SceneGraphNode('rotatingWheel')
    rotatingWheelNode.childs += [wheel]

    chasisNode = sg.SceneGraphNode('chasis')
    chasisNode.transform = tr.uniformScale(scale)
    chasisNode.childs += [chasis]

    wheel1Node = sg.SceneGraphNode('wheel1')
    wheel1Node.transform = tr.matmul([tr.uniformScale(scale),tr.translate(0.056390,0.037409,0.091705)])
    wheel1Node.childs += [rotatingWheelNode]

    wheel2Node = sg.SceneGraphNode('wheel2')
    wheel2Node.transform = tr.matmul([tr.uniformScale(scale),tr.translate(-0.060390,0.037409,-0.091705)])
    wheel2Node.childs += [rotatingWheelNode]

    wheel3Node = sg.SceneGraphNode('wheel3')
    wheel3Node.transform = tr.matmul([tr.uniformScale(scale),tr.translate(-0.056390,0.037409,0.091705)])
    wheel3Node.childs += [rotatingWheelNode]

    wheel4Node = sg.SceneGraphNode('wheel4')
    wheel4Node.transform = tr.matmul([tr.uniformScale(scale),tr.translate(0.066090,0.037409,-0.091705)])
    wheel4Node.childs += [rotatingWheelNode]

    car1 = sg.SceneGraphNode('car1')
    #tr.translate(2.0, -0.037409, 5.0)
    #, tr.rotationY(np.pi)
    car1.transform = tr.matmul([tr.translate(0, 0, 0)])
    car1.childs += [chasisNode]
    car1.childs += [wheel1Node]
    car1.childs += [wheel2Node]
    car1.childs += [wheel3Node]
    car1.childs += [wheel4Node]

    scene = sg.SceneGraphNode('system')
    scene.childs += [car1]

    return scene

# TAREA3: Esta función crea toda la escena estática y texturada de esta aplicación.
# Por ahora ya están implementadas: la pista y el terreno
# En esta función debes incorporar las casas y muros alrededor de la pista

def createStaticScene(pipeline):

    roadBaseShape = createGPUShape(pipeline, bs.createTextureQuad(1.0, 1.0))
    roadBaseShape.texture = es.textureSimpleSetup(
        getAssetPath("Road_001_basecolor.jpg"), GL_REPEAT, GL_REPEAT, GL_LINEAR_MIPMAP_LINEAR, GL_NEAREST)
    glGenerateMipmap(GL_TEXTURE_2D)

    sandBaseShape = createGPUShape(pipeline, createTiledFloor(50))
    sandBaseShape.texture = es.textureSimpleSetup(
        getAssetPath("Sand 002_COLOR.jpg"), GL_REPEAT, GL_REPEAT, GL_LINEAR_MIPMAP_LINEAR, GL_NEAREST)
    glGenerateMipmap(GL_TEXTURE_2D)

    arcShape = createGPUShape(pipeline, createTexturedArc(1.5))
    arcShape.texture = roadBaseShape.texture
    
    roadBaseNode = sg.SceneGraphNode('plane')
    roadBaseNode.transform = tr.rotationX(-np.pi/2)
    roadBaseNode.childs += [roadBaseShape]

    arcNode = sg.SceneGraphNode('arc')
    arcNode.childs += [arcShape]

    sandNode = sg.SceneGraphNode('sand')
    sandNode.transform = tr.translate(0.0,-0.01,0.0)
    sandNode.childs += [sandBaseShape]

    linearSector = sg.SceneGraphNode('linearSector')
        
    for i in range(10):
        node = sg.SceneGraphNode('road'+str(i)+'_ls')
        node.transform = tr.translate(0.0,0.0,-1.0*i)
        node.childs += [roadBaseNode]
        linearSector.childs += [node]

    linearSectorLeft = sg.SceneGraphNode('lsLeft')
    linearSectorLeft.transform = tr.translate(-2.0, 0.0, 5.0)
    linearSectorLeft.childs += [linearSector]

    linearSectorRight = sg.SceneGraphNode('lsRight')
    linearSectorRight.transform = tr.translate(2.0, 0.0, 5.0)
    linearSectorRight.childs += [linearSector]

    arcTop = sg.SceneGraphNode('arcTop')
    arcTop.transform = tr.translate(0.0,0.0,-4.5)
    arcTop.childs += [arcNode]

    arcBottom = sg.SceneGraphNode('arcBottom')
    arcBottom.transform = tr.matmul([tr.translate(0.0,0.0,5.5), tr.rotationY(np.pi)])
    arcBottom.childs += [arcNode]
    
    scene = sg.SceneGraphNode('system')
    scene.childs += [linearSectorLeft]
    scene.childs += [linearSectorRight]
    scene.childs += [arcTop]
    scene.childs += [arcBottom]
    scene.childs += [sandNode]
    
    return scene

if __name__ == "__main__":

    # Initialize glfw
    if not glfw.init():
        glfw.set_window_should_close(window, True)

    width = 800
    height = 800
    title = "Tarea 3"
    window = glfw.create_window(width, height, title, None, None)

    if not window:
        glfw.terminate()
        glfw.set_window_should_close(window, True)

    glfw.make_context_current(window)

    # Connecting the callback function 'on_key' to handle keyboard events
    glfw.set_key_callback(window, on_key)

    # Assembling the shader program (pipeline) with both shaders
    axisPipeline = es.SimpleModelViewProjectionShaderProgram()
    texPipeline = es.SimpleTextureModelViewProjectionShaderProgram()
    lightPipeline = ls.SimpleGouraudShaderProgram()
    textPipeline = tx.TextureTextRendererShaderProgram()
    
    # Telling OpenGL to use our shader program
    glUseProgram(axisPipeline.shaderProgram)

    # Setting up the clear screen color
    glClearColor(0.85, 0.85, 0.85, 1.0)

    # As we work in 3D, we need to check which part is in front,
    # and which one is at the back
    glEnable(GL_DEPTH_TEST)

    # Creating shapes on GPU memory
    cpuAxis = bs.createAxis(7)
    gpuAxis = es.GPUShape().initBuffers()
    axisPipeline.setupVAO(gpuAxis)
    gpuAxis.fillBuffers(cpuAxis.vertices, cpuAxis.indices, GL_STATIC_DRAW)

    #NOTA: Aqui creas un objeto con tu escena
    dibujo = createStaticScene(texPipeline)
    car =createCarScene(lightPipeline)
    casa = createHouse(texPipeline)
    
    
    # --- GRAFO DE ESCENA PARA LOS MUROS ---
    muros = createWall(texPipeline)

    

    setPlot(texPipeline, axisPipeline,lightPipeline)

    perfMonitor = pm.PerformanceMonitor(glfw.get_time(), 0.5)
    
    # ============ Velocímetro =============
    textTexture = tx.generateTextBitsTexture()
    gpuTexture = tx.toOpenGLTexture(textTexture)
    gpuSpeedometer = es.GPUShape().initBuffers()
    speedShape = tx.textToShape("0.0 mph",0.05,0.05)
    textPipeline.setupVAO(gpuSpeedometer)
    gpuSpeedometer.fillBuffers(speedShape.vertices, speedShape.indices, GL_STREAM_DRAW)
    gpuSpeedometer.texture = gpuTexture



    # glfw will swap buffers as soon as possible
    glfw.swap_interval(0)


    # ===========================================================
    # Variables globales
    # ===========================================================

    # Variables globales para almacenar la posición del auto y determinar las transformaciones a aplicar para la cámara
    auto = Auto(2.0, -0.037409, 5.0)

    auto_X = 2.0
    auto_Y = -0.037409
    auto_Z = 5.0
    car_theta = 0

    cam_X = 0
    cam_Y = 0
    cam_Z = 0

    # <======= Propiedades ajustables ========>
    camera_height = 0.75
    cam_radius = 2
    cam_angle = 0
    cam_fangle = 0

    # =========

    t0 = glfw.get_time()

    while not glfw.window_should_close(window):

        # Measuring performance
        perfMonitor.update(glfw.get_time())
        glfw.set_window_title(window, title + str(perfMonitor))

        # Using GLFW to check for input events
        glfw.poll_events()

        # <===== Controlador =======>
        t1 = glfw.get_time()
        dt = t1 - t0
        t0 = t1 

        # <============ Input: maniobrar (rotar) ===============>
        auto.steer(0)
        if (glfw.get_key(window, glfw.KEY_A) == glfw.PRESS):
            auto.steer(1)
        if (glfw.get_key(window, glfw.KEY_D) == glfw.PRESS):
            auto.steer(-1)
        
        #  <================= Input: acelerar ==================>
        pressed = False
        if (glfw.get_key(window, glfw.KEY_W) == glfw.PRESS):
            auto.accelerate(1)
            pressed = True
            
        if (glfw.get_key(window, glfw.KEY_S) == glfw.PRESS):
            auto.accelerate(-1)
            pressed = True
        
        if not pressed:
            auto.accelerate(0)

        auto.step(dt)
    
        car.transform = tr.matmul([tr.translate(auto.X,auto.Y,auto.Z), tr.rotationY(auto.direction)])
        
        # Efecto de suavizado de movimiento de cámara
        # [Adicional]
        cam_angle = auto.direction+np.pi

        # <===== Input: presionar botón F para ver en reversa =======>
        # [Adicional]
        if (glfw.get_key(window, glfw.KEY_F) == glfw.PRESS):
            cam_angle = auto.direction
        
        # <=========== Suavizado de cámara =============>
        # [Adicional]
        if cam_fangle != cam_angle:
            cam_fangle += dt*7.5*(cam_angle-cam_fangle)

        cam_X = auto.X + (cam_radius * np.sin(cam_fangle))
        cam_Z = auto.Z + (cam_radius * np.cos(cam_fangle))
        cam_Y = auto.Y + camera_height
        
        controller.viewPos = np.array([cam_X,cam_Y,cam_Z])
        controller.at = np.array([auto.X, auto.Y, auto.Z])
        up = np.array([0,1,0])

        # <===== Controlador =======>

        # Clearing the screen in both, color and depth
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # Filling or not the shapes depending on the controller state
        if (controller.fillPolygon):
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
        else:
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)

        setView(texPipeline, axisPipeline, lightPipeline)

        if controller.showAxis:
            glUseProgram(axisPipeline.shaderProgram)
            glUniformMatrix4fv(glGetUniformLocation(axisPipeline.shaderProgram, "model"), 1, GL_TRUE, tr.identity())
            axisPipeline.drawCall(gpuAxis, GL_LINES)

        #NOTA: Aquí dibujas tu objeto de escena
        glUseProgram(texPipeline.shaderProgram)
        sg.drawSceneGraphNode(dibujo, texPipeline, "model")
        sg.drawSceneGraphNode(casa, texPipeline, "model")
        sg.drawSceneGraphNode(muros, texPipeline, "model")


        glUseProgram(lightPipeline.shaderProgram)
        sg.drawSceneGraphNode(car, lightPipeline, "model")
        
        
        color = [1.0,1.0,1.0]
        speedometer_value = abs(round(auto.speed,1))

        # <=== Efecto de vibración en el velocímetro cuando la velocidad es mucha ===>
        # [Adicional]
        offsets = [(np.random.rand()*2) - 1,(np.random.rand()*2) - 1]
        shakeMag = 0
        if speedometer_value > 5:
            shakeMag = (speedometer_value-5)*0.02

        # [Adicional]
        # < ==== Velocímetro ==== >
        speedShape = tx.textToShape(f"{speedometer_value} mph",0.1,0.1)
        gpuSpeedometer.fillBuffers(speedShape.vertices, speedShape.indices, GL_STREAM_DRAW)
        gpuSpeedometer.texture = gpuTexture

        glUseProgram(textPipeline.shaderProgram)
        glUniform4f(glGetUniformLocation(textPipeline.shaderProgram, "fontColor"), color[0], color[1], color[2], 1.0)
        glUniform4f(glGetUniformLocation(textPipeline.shaderProgram, "backColor"), 1-color[0], 1-color[1], 1-color[2],0)
        glUniformMatrix4fv(glGetUniformLocation(textPipeline.shaderProgram, "transform"), 1, GL_TRUE,
            tr.translate(-0.9 + shakeMag*offsets[0], -0.9 + shakeMag*offsets[1], 0))
        textPipeline.drawCall(gpuSpeedometer)

        # Once the render is done, buffers are swapped, showing only the complete scene.
        glfw.swap_buffers(window)

    # freeing GPU memory
    gpuAxis.clear()
    gpuSpeedometer.clear()
    dibujo.clear()
    

    glfw.terminate()