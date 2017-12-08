import matplotlib.pyplot as plt
import numpy
import MyMayaviPlots as me

# from https://stackoverflow.com/questions/32491646/how-to-plot-the-heat-map-for-a-given-function-in-python


def fun(mu, gamma, z):
    return float(me.f.subs(me.x, mu).subs(me.y, gamma).subs(me.z, z).evalf())


def create_fun_map(fun):
    mu = numpy.linspace(-me.side, me.side, me.ticks)
    gamma = numpy.linspace(-me.side, me.side, me.ticks)
    fun_map = numpy.empty((mu.size, gamma.size))
    for i in range(mu.size):
        for j in range(gamma.size):
            fun_map[i,j] = fun(mu[i], gamma[j])
    # print fun_map
    return fun_map


def add_subplots(plts, titles):
    n = len(plts)
    i = 1
    lowest = min(map(lambda x: numpy.amin(x), plts))
    highest = max(map(lambda x: numpy.amax(x), plts))
    for p in plts:
        s = fig.add_subplot(1, n, i, xlabel='x', ylabel='y')
        s.title.set_text("z = " + str(titles[i-1]))
        # https://stackoverflow.com/questions/3584805/in-matplotlib-what-does-the-argument-mean-in-fig-add-subplot111
        im = s.imshow(
                p,
                cmap=plt.cm.seismic, vmin=lowest, vmax=highest,
                # extent=(gamma[0], gamma[-1], mu[0], mu[-1]),
                origin='lower')
        i += 1


fun_upper = lambda x, y: fun(x, y, 1.5)
fun_middle = lambda x, y: fun(x, y, 1.)
fun_lower = lambda x, y: fun(x, y, 0.75)
fun_lowest = lambda x, y: fun(x, y, 0.1)

fig = plt.figure()

upper_map = create_fun_map(fun_upper)
lower_map = create_fun_map(fun_lower)
middle_map = create_fun_map(fun_middle)
lowest_map = create_fun_map(fun_lowest)

add_subplots([upper_map, middle_map, lower_map, lowest_map], [1.5, 1., 0.75, .1])

fig.show()
plt.show()
