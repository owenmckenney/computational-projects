import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import casadi as ca

# physical parameters
g = 9.81
l = 1.5
pivot_y = 0.0
t_sim = 0.0

x_min, x_max = -2.5, 2.5    

cart_drag = 3.0
theta_damp = 0.15

dt_cap = 0.03  # sim dt clamp

# nmpc parameters
theta, omega, x, v = 0, 1, 2, 3
dt_mpc = 0.1 # controller update @ 10 Hz (fixed)
N = 10 # horizon steps (10 steps * 10 Hz = 1 second)
u_max = 10.0 # accel limit
v_max = 6.0 # optional v bound

theta0 = np.deg2rad(0.1)
omega0 = 0.0
x0 = 0.0
v0 = 0.0

# state
y = ca.DM([theta0, omega0, x0, v0])

target_theta = np.pi  # upright target, theta = pi

def system_dynamics(state, u):
    th = state[0]
    om = state[1]
    xx = state[2]
    vv = state[3]

    a = u - cart_drag * vv

    th_dot = om
    om_dot = -(g / l) * ca.sin(th) - (a / l) * ca.cos(th) - theta_damp * om
    x_dot = vv
    v_dot = a

    return ca.vertcat(th_dot, om_dot, x_dot, v_dot)

def RK4(state, u, dt):
    k1 = system_dynamics(state, u) * dt
    k2 = system_dynamics(state + k1 / 2, u) * dt
    k3 = system_dynamics(state + k2 / 2, u) * dt
    k4 = system_dynamics(state + k3, u) * dt
    return state + (k1 + 2*k2 + 2*k3 + k4) / 6


class NMPCController:
    def __init__(self, g, l, cart_drag, theta_damp, x_min, x_max, dt_mpc, N, u_max, v_max):
        self.g = float(g)
        self.l = float(l)
        self.cart_drag = float(cart_drag)
        self.theta_damp = float(theta_damp)

        self.x_min = float(x_min)
        self.x_max = float(x_max)

        self.dt_mpc = float(dt_mpc)
        self.N = int(N)
        self.u_max = float(u_max)
        self.v_max = float(v_max)

        self.nx = 4
        self.nu = 1

        self.solver = None
        self.lbx = None
        self.ubx = None
        self.lbg = None
        self.ubg = None

        self.Uguess = None
        self.Xguess = None

        self.build_nmpc_solver()

    def reset(self, y0):
        y0 = np.asarray(y0, dtype=float).reshape((self.nx,))
        self.Uguess = np.zeros(self.N, dtype=float)
        self.Xguess = np.tile(y0, (self.N + 1, 1)).astype(float)

    def build_nmpc_solver(self):
        nx, nu = 4, 1

        X = ca.MX.sym("X", nx, self.N + 1)
        U = ca.MX.sym("U", nu, self.N)
        P = ca.MX.sym("P", nx + nu)

        x0_sym = P[0:nx]
        u_prev_sym = P[nx:nx+nu]

        # cost function weights
        w_upright = 80.0
        w_omega = 5.0
        w_x = 5.0
        w_v = 1.0
        w_u = 0.1
        w_du = 1.0
        terminal_mult = 120.0

        J = 0
        g_con = []

        # initial condition constraint
        g_con.append(X[:, 0] - x0_sym)

        for k in range(self.N):
            xk = X[:, k]
            uk = U[:, k]

            # dynamics constraints (multiple shooting)
            x_next = RK4(xk, uk[0], self.dt_mpc)
            g_con.append(X[:, k+1] - x_next)

            th = xk[0]
            om = xk[1]
            xx = xk[2]
            vv = xk[3]

            # upright error without angle wrap: sin(th)=0 and 1+cos(th)=0 at th=pi
            e_sin = ca.sin(th)
            e_cos = 1 + ca.cos(th)
            upright_e = e_sin*e_sin + e_cos*e_cos

            # smooth input
            duk = uk - (u_prev_sym if k == 0 else U[:, k-1])

            # stage cost
            J += (w_upright * upright_e +
                  w_omega * (om*om) +
                  w_x * (xx*xx) +
                  w_v * (vv*vv) +
                  w_u * (uk[0]*uk[0]) +
                  w_du * (duk[0]*duk[0]))

        # terminal cost
        xN = X[:, self.N]
        thN, omN, xNpos, vN = xN[0], xN[1], xN[2], xN[3]
        e_sinN = ca.sin(thN)
        e_cosN = 1 + ca.cos(thN)
        upright_eN = e_sinN*e_sinN + e_cosN*e_cosN

        J += terminal_mult * (w_upright * upright_eN +
                              w_omega * (omN*omN) +
                              w_x * (xNpos*xNpos) +
                              w_v * (vN*vN))

        # pack NLP
        OPT = ca.vertcat(ca.reshape(X, -1, 1), ca.reshape(U, -1, 1))
        G = ca.vertcat(*g_con)
        nlp = {"x": OPT, "f": J, "g": G, "p": P}

        opts = {"ipopt.print_level": 0, "print_time": 0, "ipopt.max_iter": 80, "ipopt.sb": "yes"}
        self.solver = ca.nlpsol("solver", "ipopt", nlp, opts)

        # bounds
        nX = nx*(self.N+1)
        nU = nu*self.N

        self.lbx = -np.inf*np.ones(nX + nU)
        self.ubx = np.inf*np.ones(nX + nU)

        # constrain x and v across horizon (optional but helps)
        for k in range(self.N+1):
            base = k*nx
            self.lbx[base + 2] = self.x_min
            self.ubx[base + 2] = self.x_max
            self.lbx[base + 3] = -self.v_max
            self.ubx[base + 3] = self.v_max

        # input bounds
        u_start = nX
        for k in range(self.N):
            self.lbx[u_start + k] = -self.u_max
            self.ubx[u_start + k] = self.u_max

        # equality constraints for dynamics
        ng = int(G.shape[0])
        self.lbg = np.zeros(ng)
        self.ubg = np.zeros(ng)

    def solve_nmpc(self, x0, u_prev):
        if self.Xguess is None or self.Uguess is None:
            self.reset(np.asarray(x0, dtype=float))

        x0 = np.asarray(x0, dtype=float).reshape((self.nx,))
        u_prev = float(u_prev)

        X0_flat = self.Xguess.reshape(-1, order="F")
        U0_flat = self.Uguess.reshape(-1)
        opt0 = np.concatenate([X0_flat, U0_flat])

        p = np.concatenate([x0, [u_prev]])

        sol = self.solver(x0=opt0, lbx=self.lbx, ubx=self.ubx, lbg=self.lbg, ubg=self.ubg, p=p)
        w = np.array(sol["x"]).squeeze()

        nX = self.nx*(self.N+1)
        X_sol = w[:nX].reshape((self.nx, self.N+1), order="F").T
        U_sol = w[nX:].reshape((self.N,))

        # shift for warm-start
        self.Xguess[:-1] = X_sol[1:]
        self.Xguess[-1] = X_sol[-1]
        self.Uguess[:-1] = U_sol[1:]
        self.Uguess[-1] = U_sol[-1]

        return float(U_sol[0])


# nmpc
nmpc = NMPCController(g, l, cart_drag, theta_damp, x_min, x_max, dt_mpc, N, u_max, v_max)
nmpc.reset(np.array(y).astype(float).flatten())

u_prev_applied = 0.0
u_hold = 0.0
mpc_accum = 0.0


# matplotlib
fig, ax = plt.subplots(figsize=(5,5))
ax.set_aspect("equal", adjustable="box")
ax.set_xlabel("x (m)")
ax.set_ylabel("y (m)")
ax.set_aspect("equal", adjustable="box")

info_text = ax.text(0.02, 0.98, "", transform=ax.transAxes, ha="left", va="top")

pad = 0.6
ax.set_xlim(x_min - pad, x_max + pad)
ax.set_ylim(-l - pad + pivot_y, l + pad + pivot_y)

ax.set_aspect("equal", adjustable="box")

# make x/y ranges identical
xmin, xmax = ax.get_xlim()
ymin, ymax = ax.get_ylim()
span = max(xmax - xmin, ymax - ymin)

xc = 0.5 * (xmin + xmax)
yc = 0.5 * (ymin + ymax)

ax.set_xlim(xc - span/2, xc + span/2)
ax.set_ylim(yc - span/2, yc + span/2)

rod_line, = ax.plot([], [], lw=2)
bob_pt, = ax.plot([], [], "o", markersize=6)
pivot_pt, = ax.plot([], [], "s", markersize=8, color="black")

last_wall = time.time()

def init():
    rod_line.set_data([], [])
    bob_pt.set_data([], [])
    pivot_pt.set_data([], [])
    info_text.set_text("")
    return rod_line, bob_pt, pivot_pt, info_text

def update(frame):
    global y, last_wall, u_hold, mpc_accum, u_prev_applied, t_sim

    # wall-clock dt for simulation
    now = time.time()
    dt = now - last_wall
    last_wall = now
    dt = min(dt, dt_cap)
    if dt <= 0:
        dt = 1e-6

    # update NMPC on fixed schedule dt_mpc
    mpc_accum += dt
    while mpc_accum >= dt_mpc:
        mpc_accum -= dt_mpc
        try:
            yNp = np.array(y).astype(float).flatten()
            u_hold = nmpc.solve_nmpc(yNp, u_prev_applied)
        except Exception:
            u_hold = 0.0  # fail-safe
        u_prev_applied = u_hold

    # integrate sim using held control (single RK4)
    y = RK4(y, u_hold, dt)

    # safety clamp (convert to floats for comparisons)
    th = float(y[0])
    om = float(y[1])
    xx = float(y[2])
    vv = float(y[3])

    if xx < x_min:
        xx = x_min
        vv = 0.0
    elif xx > x_max:
        xx = x_max
        vv = 0.0

    y = ca.DM([th, om, xx, vv])
    t_sim += dt

    # draw geometry
    x_joint, y_joint = xx, pivot_y
    x_bob = x_joint + l * np.sin(th)
    y_bob = y_joint - l * np.cos(th)

    rod_line.set_data([x_joint, x_bob], [y_joint, y_bob])
    bob_pt.set_data([x_bob], [y_bob])
    pivot_pt.set_data([x_joint], [y_joint])

    theta_deg = np.degrees(th)
    info_text.set_text(f"t = {t_sim:.2f} s\nθ = {theta_deg:.2f}°")

    return rod_line, bob_pt, pivot_pt, info_text

ani = FuncAnimation(fig, update, init_func=init, interval=10, blit=True)
plt.show()
