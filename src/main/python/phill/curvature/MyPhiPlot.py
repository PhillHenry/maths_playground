import matplotlib.pyplot as plt
import MyMayaviPlots as me
import numpy as np

x_start = 0.5
y_start = 0.921052631579
z_start = 1.06590562042

p_x = -0.778703862327451
p_y = 1.79747592958838
p_z = -1.06590562041511

steps = 50.
d_x = p_x / steps
d_y = p_y / steps
d_z = p_z / steps


def phis():
    xs = []
    ys = []

    for alpha in np.arange(0.0, steps * 2, 1.):
        new_x = x_start + (alpha * d_x)
        new_y = y_start + (alpha * d_y)
        new_z = z_start + (alpha * d_z)
        f_new = me.f.subs(me.x, new_x).subs(me.y, new_y).subs(me.z, new_z).evalf()
        xs.append(float(alpha) / float(steps))
        ys.append(f_new)

    return xs, ys


def armijo_vals(c1):
    xs = []
    ys = []
    f0 = me.f.subs(me.x, x_start).subs(me.y, y_start).subs(me.z, z_start).evalf()
    p = np.array([x_start, y_start, z_start])
    div_f = me.gradient(x_start, y_start, z_start)

    for alpha in np.arange(1., steps * 2, 2.):
        new_x = x_start + (alpha * d_x)
        new_y = y_start + (alpha * d_y)
        new_z = z_start + (alpha * d_z)
        f_new = me.f.subs(me.x, new_x).subs(me.y, new_y).subs(me.z, new_z).evalf()
        rhs = f0 + alpha * c1 * np.dot(np.matrix(div_f), p.transpose())
        if f_new <= rhs:
            xs.append(float(alpha) / float(steps))
            ys.append(f_new)

    return xs, ys


def curvature_vals(c2):
    xs = []
    ys = []
    p = np.array([x_start, y_start, z_start])
    div_f_0 = me.gradient(x_start, y_start, z_start)
    rhs = c2 * np.dot(p.transpose(), div_f_0)

    # for alpha in range(steps * 2):
    for alpha in np.arange(0.0, steps * 2, 2.):
        new_x = x_start + (alpha * d_x)
        new_y = y_start + (alpha * d_y)
        new_z = z_start + (alpha * d_z)
        div_f = me.gradient(new_x, new_y, new_z)
        lhs = np.dot(p.transpose(), div_f)
        f_new = me.f.subs(me.x, new_x).subs(me.y, new_y).subs(me.z, new_z).evalf()
        if lhs <= rhs:
            xs.append(float(alpha) / float(steps))
            ys.append(f_new)

    return xs, ys


xs, ys = phis()
plt.plot(xs, ys)
plt.ylabel(r'$\phi$ (alpha)')
plt.xlabel('alpha')

c1 = 1e-4
xs, ys = armijo_vals(c1)
print xs
plt.plot(xs, ys, 'o')

c2 = 0.5
xs, ys = curvature_vals(c2)
print xs
plt.plot(xs, ys, 'o', c='r')

plt.title('Minimizing f given a position (x) and a vector (p) with c1=%s, c2=%s' % (c1, c2))
# zip(xs, ys).filter(lambda (x,y): me.f())

# plt.axis([0, 6, 0, 20])
plt.show()
