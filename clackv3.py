import sys

import numpy as np
import taichi as ti
import taichi.math as tm

ti.init(arch=ti.cpu)

digits = 1
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
dt = 1.0 / 60.0
scale = 10

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
frames = 0
if states[collisions[None], 0][0] == float('-inf'):
    frames = 2147483638
else:
    frames = int(tm.ceil(states[collisions[None], 0][0] / dt)) + 1

print(frames)
frameIndexes = ti.field(ti.u32, shape=frames)


@ti.kernel
def findRenderStates():
    # ti.loop_config(serialize=True)
    # currentLow = ti.u32(0)
    for i in range(frames):
        currTime = dt * i
        low = ti.i32(0)
        high = ti.i32(collisions[None])

        while low <= high:
            mid = (low + high) // 2
            iTime = states[mid, 0][0]
            if iTime <= currTime < states[mid + 1, 0][0]:
                frameIndexes[i] = ti.u32(mid)
                break
            elif iTime > currTime:
                high = mid - 1
            else:
                low = mid + 1

            if low > high:
                if currTime >= states[collisions[None], 0][0]:
                    frameIndexes[i] = ti.u32(collisions[None])
                else:
                    frameIndexes[i] = ti.u32(low)
        # currentLow = frameIndexes[i]


findRenderStates()
print("done")


def toScreen(vec0, vec1):
    global x, y, scale
    return [vec0 * scale / x, vec1 * scale / y]


gui = ti.GUI(f"Clack Test for {digits} digits", res=(x, y))
elapsedTime = 0.0

X = np.array([toScreen(0.0, 0.0), toScreen(0.0, 0.0)])
Y = np.array([toScreen(0.0, 10), toScreen(x, 0.0)])
Z = np.array([toScreen(x, 10), toScreen(x, 10)])

currInd = 0
currFrame = 0

def masterDraw():
    global currInd, dt, currFrame
    gui.triangles(a=X, b=Y, c=Z)
    currInd = frameIndexes[currFrame]
    curr = states[currInd, 0]
    nextTime = states[currInd + 1, 0][0]
    delta = elapsedTime - curr[0]
    if nextTime - curr[0] < elapsedTime - curr[0] and currInd < collisions[None]:
        delta = nextTime - curr[0]

    pos0 = curr[1] + delta * curr[3]
    pos1 = curr[2] + delta * curr[4]

    a = np.array([
        toScreen(pos0 - halfSize0, 10),
        toScreen(pos0 - halfSize0, 10),
        toScreen(pos1 - halfSize1, 10),
        toScreen(pos1 - halfSize1, 10)
    ])
    b = np.array([
        toScreen(pos0 - halfSize0, 10 + size0),
        toScreen(pos0 + halfSize0, 10),
        toScreen(pos1 - halfSize1, 10 + size1),
        toScreen(pos1 + halfSize1, 10),
    ])
    c = np.array([
        toScreen(pos0 + halfSize0, 10 + size0),
        toScreen(pos0 + halfSize0, 10 + size0),
        toScreen(pos1 + halfSize1, 10 + size1),
        toScreen(pos1 + halfSize1, 10 + size1)
    ])
    gui.triangles(a=a, b=b, c=c)

    gui.text(f"Index: {currInd}", pos=[0.3, 0.5], font_size=20)
    if currFrame < frames - 1:
        currFrame += 1




while gui.running:
    elapsedTime += dt

    masterDraw()

    gui.show()
