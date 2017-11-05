import mayavi.mlab as plt
import numpy as np
import math as m
from sympy import *


def mexp(x):
    return m.exp(x)


def mcos(x):
    return m.cos(x)


def surface_fn(xv, yv, c, e):
    return c(xv) ** 2 + c(yv) ** 2  #c(yv ** 2 + xv ** 2) * e(- (xv ** 2 + yv ** 2) / 2)


# see https://stackoverflow.com/questions/952914/making-a-flat-list-out-of-list-of-lists-in-python
def flatten(xs):
    return [item for sublist in xs for item in sublist]


ticks = 20


def spaced():
    x_space = y_space = np.linspace(-side, side, ticks)
    z_space = np.linspace(0, 2 * side, ticks)
    return x_space, y_space, z_space


def meshed():
    # see https://stackoverflow.com/questions/36013063/what-is-purpose-of-meshgrid-in-python
    x_space, y_space, z_space = spaced()
    xv, yv, zv = np.meshgrid(x_space, y_space, z_space,
                             sparse=False, indexing='ij')
    return xv, yv, zv


def gradient(xpt, ypt, zpt):
    # print "xpt = ", xpt, "zpt = ", zpt, "ypt = ", ypt,
    gradx = div[0].subs(x, xpt).subs(y, ypt).subs(z, zpt).evalf()
    grady = div[1].subs(x, xpt).subs(y, ypt).subs(z, zpt).evalf()
    gradz = div[2].subs(x, xpt).subs(y, ypt).subs(z, zpt).evalf()
    return float(gradx), float(grady), float(gradz)


def hess(xpt, ypt, zpt):
    co_ords = [xpt, ypt, zpt]
    hessian_fn = [[f.diff(a).diff(b) for a in [x, y, z]] for b in [x, y, z]]
    values = [sub_vals(item, co_ords) for item in hessian_fn]
    hess_mat_eval = Matrix(values)
    curl = sub_vals(div, co_ords)
    if hess_mat_eval.det() == 0:
        print "uninvertible hessian at ", xpt, ypt, zpt
        return 0, 0, 0
    else:
        p = - (hess_mat_eval.inv() * Matrix(curl))
        # x =  0.5 y =  0.921052631579 z =  1.06590562042 p =  Matrix([[-0.778703862327451], [1.79747592958838], [-1.06590562041511]])
        print "x = ", xpt, "y = ", ypt , "z = ", zpt, "p = ", p
        return float(p[0]), float(p[1]), float(p[2])


def sub_vals(items, vals):
    return [i.subs(x, vals[0]).subs(y, vals[1]).subs(z, vals[2]).evalf() for i in items]


def line_on_surface(displacement = 0):
    x_space = np.zeros(ticks) + displacement
    y_space = np.linspace(-side, side, ticks)
    zs = map(lambda (xpos, ypos): z_fn(xpos, ypos), zip(x_space, y_space))
    return x_space, y_space, zs


def plot_hessian_line(displacement = 0):
    xv, yv, zv = line_on_surface(displacement)
    hessFn = np.vectorize(hess)
    ps = hessFn(xv, yv, zv)
    print "ps shape", np.shape(ps)
    plt.quiver3d(xv, yv, zv, ps[0], ps[1], ps[2], mode='arrow',
                 scale_mode='none', opacity=0.5, color=(1., 0., 0.))


def plot_gradient_line(grad_fn):
    xv, yv, zv = line_on_surface()
    zVal = grad_fn(xv, yv, zv)
    dx = zVal[0]
    dy = zVal[1]
    dz = zVal[2]
    plt.quiver3d(xv, yv, zv, dx, dy, dz, mode='arrow',
             scale_mode='none', opacity=0.5, color=(0., 0., 1.))


def z_fn(xpt, ypt, z_displacement = 0):
    z_n = multiplier * float(fn.subs(x, xpt).subs(y, ypt).evalf()) + z_displacement  # http://docs.sympy.org/latest/modules/evalf.html
    # z =  mexp(-(xpt ** 2 + ypt ** 2))  #mcos(z_n) # ** (1 / z_power) # z_power
    # print "xpt", xpt, "ypt", ypt, "z_n", z_n  #, "z", z
    return z_n ** (1/z_power)


side = 2.5
z_power = 2.
multiplier = 1.

x, y, z = symbols('x y z')

fn = surface_fn(x, y, cos, exp)
f = z ** z_power - multiplier * fn  # z_power
div = [f.diff(v) for v in [x, y, z]]

# http://hplgit.github.io/primer.html/doc/pub/plot/._plot-bootstrap007.html#plot:surf:mesh_surf
if __name__ == '__main__':
    plt.figure(fgcolor=(.0, .0, .0), bgcolor=(1.0, 1.0, 1.0))

    z3vFn = np.vectorize(z_fn)

    z_displaced_up = lambda x, y: z_fn(x, y, 2)
    z_displaced_down = lambda x, y: z_fn(x, y, 4)

    z3vFn_d_up = np.vectorize(z_displaced_up)
    z3vFn_d_down = np.vectorize(z_displaced_down)

    plot_hessian_line()
    plot_hessian_line(0.5)
    plot_hessian_line(1.)
    plot_hessian_line(1.5)
    plot_hessian_line(-0.5)
    plot_hessian_line(-1.)
    plot_hessian_line(-1.5)
    plot_gradient_line(np.vectorize(gradient))

    # plt.quiver3d(0.5, 0.921052631579, 1.06590562042, -0.778703862327451, 1.79747592958838, -1.06590562041511, mode='arrow',
    #              scale_mode='none', opacity=0.5, color=(1., 0., 0.))

    x_space = y_space = np.linspace(-side, side, ticks * 2)
    xv, yv = np.meshgrid(x_space, y_space)
    zv = z3vFn(xv, yv)
    plt.points3d(xv, yv, zv, scale_factor=.05)
    plt.axes( xlabel='x', ylabel='y',
              zlabel='z')
    # zv_up = z3vFn_d_up(xv, yv)
    # plt.points3d(xv, yv, zv_up, scale_factor=.05)
    # zv_down = z3vFn_d_down(xv, yv)
    # plt.points3d(xv, yv, zv_down, scale_factor=.05)

    print "df = ", div

    plt.show()


