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
x1 = 10.0
time = 0.0

halfSize0 = size0 / 2
halfSize1 = size1 / 2


max_steps = 10 ** digits
states = np.zeros((max_steps, 5), dtype=np.float64)
states[0] = [time, x0, x1, v0, v1]

ind = 0
curr = states[ind]
while curr[4] <= 0 or curr[4] <= abs(curr[3]) or curr[3] < 0:
    if curr[3] < 0:
        if curr[4] >= curr[3] or (0 - curr[1] + halfSize0) / curr[3] < (curr[1] + halfSize0 - curr[2] + halfSize1) / (
                curr[4] - curr[3]):
            delta = (0 - curr[1] + halfSize0) / curr[3]
            newTime = curr[0] + delta
            newState = np.array([newTime, halfSize0, curr[2] + curr[4] * delta, -curr[3], curr[4]])
        else:
            delta = (curr[2] - halfSize1 - curr[1] - halfSize0) / (curr[3] - curr[4])
            fv2 = (2 * mass0 * curr[3] + curr[4] * (mass1 - mass0)) / (mass0 + mass1)
            fv1 = fv2 + curr[4] - curr[3]
            newTime = curr[0] + delta
            newState = np.array([newTime, curr[1] + curr[3] * delta, curr[2] + curr[4] * delta, fv1, fv2])
    else:
        delta = (curr[2] - halfSize1 - curr[1] - halfSize0) / (curr[3] - curr[4])
        fv2 = (2 * mass0 * curr[3] + curr[4] * (mass1 - mass0)) / (mass0 + mass1)
        fv1 = fv2 + curr[4] - curr[3]
        newTime = curr[0] + delta
        newState = np.array([newTime, curr[1] + curr[3] * delta, curr[2] + curr[4] * delta, fv1, fv2])

    ind += 1
    states[ind] = newState
    curr = states[ind]
print(ind)
