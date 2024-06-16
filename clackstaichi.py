import numpy as np
import taichi as ti
import taichi.math as tm

ti.init(arch=ti.cpu)

digits = 6
mass0 = 1.0
mass1 = 100 ** (digits - 1)
size0 = 1.0
size1 = 2.0
v0 = 0.0
v1 = -1.0
x0 = 5.0
x1 = 9.0
time = 0.0
x, y = 800, 800

halfSize0 = size0 / 2
halfSize1 = size1 / 2

max_steps = 10 ** digits
states = ti.Vector.field(n=5, dtype=ti.f64, shape=(max_steps, 1))
collisions = ti.field(dtype=ti.u64, shape=())


@ti.kernel
def simulate():
    states[0, 0] = [time, x0, x1, v0, v1]
    ind = 0
    curr = states[ind, 0]
    obCollision = True
    while not ((obCollision or curr[3] > 0) and curr[4] > curr[3]):
        if obCollision:
            delta = (curr[2] - halfSize1 - curr[1] - halfSize0) / (curr[3] - curr[4])
            fv2 = (2 * mass0 * curr[3] + curr[4] * (mass1 - mass0)) / (mass0 + mass1)
            fv1 = fv2 + curr[4] - curr[3]
            newTime = curr[0] + delta
            states[ind + 1, 0] = [newTime, curr[1] + curr[3] * delta, curr[2] + curr[4] * delta, fv1, fv2]
        else:
            delta = (0 - curr[1] + halfSize0) / curr[3]
            newTime = curr[0] + delta
            states[ind + 1, 0] = [newTime, halfSize0, curr[2] + curr[4] * delta, -curr[3], curr[4]]
        obCollision = not obCollision
        ind += 1
        curr = states[ind, 0]
    print(ind)
    collisions[None] = ind


simulate()

gui = ti.GUI(f"Clack Test for {digits} digits", res=(x, y))
elapsedTime = 0.0
dt = 1.0 / 60.0
X = np.array([[0.0, 0.0], [0.0, 0.0]])
Y = np.array([[0.0, 0.1], [1.0, 0.0]])
Z = np.array([[1.0, 0.1], [1.0, 0.1]])

currInd = 0


def masterDraw():
    global currInd
    global dt
    gui.triangles(a=X, b=Y, c=Z)
    curr = states[currInd, 0]
    nextTime = states[currInd + 1, 0][0]
    delta = 0
    if nextTime - curr[0] < elapsedTime - curr[0]:
        delta = nextTime - curr[0]
    else:
        delta = elapsedTime - curr[0]

    pos0 = curr[1] + delta * curr[3]
    pos1 = curr[2] + delta * curr[4]

    a = np.array([
        [(pos0 - halfSize0) / 10.0, 0.1],
        [(pos0 - halfSize0) / 10.0, 0.1],
        [(pos1 - halfSize1) / 10.0, 0.1],
        [(pos1 - halfSize1) / 10.0, 0.1]
    ])
    b = np.array([
        [(pos0 - halfSize0) / 10.0, 0.1 + size0 / 10],
        [(pos0 + halfSize0) / 10.0, 0.1],
        [(pos1 - halfSize1) / 10.0, 0.1 + size1 / 10],
        [(pos1 + halfSize1) / 10.0, 0.1]
    ])
    c = np.array([
        [(pos0 + halfSize0) / 10.0, 0.1 + size0 / 10],
        [(pos0 + halfSize0) / 10.0, 0.1 + size0 / 10],
        [(pos1 + halfSize1) / 10.0, 0.1 + size1 / 10],
        [(pos1 + halfSize1) / 10.0, 0.1 + size1 / 10]
    ])
    gui.triangles(a=a, b=b, c=c)

    gui.text(f"Index: {currInd}", pos=[0.3, 0.5], font_size=20)

    while states[currInd + 1, 0][0] <= elapsedTime and currInd < collisions[None]:
        currInd += 1


while gui.running:
    elapsedTime += dt

    masterDraw()

    gui.show()
