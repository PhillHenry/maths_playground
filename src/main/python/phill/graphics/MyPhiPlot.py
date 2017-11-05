import matplotlib.pyplot as plt
import MyMayaviPlots as me

x_start = 0.5
y_start = 0.921052631579
z_start = 1.06590562042

p_x = -0.778703862327451
p_y = 1.79747592958838
p_z = -1.06590562041511

steps = 50
d_x = p_x / steps
d_y = p_y / steps
d_z = p_z / steps

xs = []
ys = []

for i in range(steps * 2):
    new_x = x_start + (i * d_x)
    new_y = y_start + (i * d_y)
    new_z = z_start + (i * d_z)
    f_new = me.f.subs(me.x, new_x).subs(me.y, new_y).subs(me.z, new_z).evalf()
    xs.append(float(i) / float(steps))
    ys.append(f_new)

plt.plot(xs, ys)
plt.title('Minimizing f given a position (x) and a vector (p)')
plt.ylabel(r'$\phi$ (alpha)')
plt.xlabel('alpha')
# plt.axis([0, 6, 0, 20])
plt.show()
