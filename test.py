import taichi as ti

ti.init(arch=ti.cpu)


@ti.kernel
def testForLoop():
    a = 2147483638
    print(a)
    # max = 0

    # ti.loop_config(serialize=True)
    for i in range(a):
        if i == a - 1:
            print(i)

testForLoop()
