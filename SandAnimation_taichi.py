import taichi as ti
from math import pi

# Implementation of the Paper
# Klár, Gergely, et al. "Drucker-prager elastoplasticity for sand animation." ACM Transactions on Graphics (TOG) 35.4 (2016): 1-12.


arch = ti.vulkan if ti._lib.core.with_vulkan() else ti.cuda
# arch = ti.cuda
ti.init(arch=arch)

########## simulation parameter ##############
grid_res = 64
particle_num = (grid_res ** 3) // 4  # 512 * 16  ##python global variable : not updated in taichi kernel
print(particle_num)
scene_len = 1
grid_dx = scene_len / grid_res
grid_inv_dx = 1 / grid_dx
particle_initial_volume = (grid_dx * 0.5) ** 3
particle_rho = 100

particle_mass = particle_rho * particle_initial_volume
dt = 5e-4

# material property

E = 100000  # 1000  # Young's modulus
nu = 0.2  # Poisson's ratio
mu_0, lambda_0 = E / (2 * (1 + nu)), E * nu / (
        (1 + nu) * (1 - 2 * nu))  # Lame parameters
friction_angle = pi/6
alpha = ti.sqrt(2 / 3) * (2 * ti.sin(friction_angle) / (3 - ti.sin(friction_angle)))

gravity = 9.8
bound = 3
tolerance = 1e-8

# taichi data
# particle data
ti_particle_pos = ti.Vector.field(3, ti.f32, particle_num)
ti_particle_vel = ti.Vector.field(3, ti.f32, particle_num)
ti_particle_Fe = ti.Matrix.field(3, 3, ti.f32, particle_num)  # Deformation gradient elastic part
ti_particle_Fp = ti.Matrix.field(3, 3, ti.f32, particle_num)  # Deformation gradient elastic part
# ti_particle_Jp = ti.field(ti.f32, particle_num)  # Plastic part of J
# ti_particle_alpha = ti.field(ti.f32, particle_num)  # particle hardening
ti_particle_C = ti.Matrix.field(3, 3, ti.f32, particle_num)  # affine momentum

# grid data
ti_grid_vel = ti.Vector.field(3, ti.f32, shape=(grid_res, grid_res, grid_res))
ti_grid_mass = ti.field(ti.f32, shape=(grid_res, grid_res, grid_res))

##########################################

particle_color = (194 / 256, 178 / 256, 128 / 256)
particle_radius = 0.005

desired_frame_dt = 1 / 60
frame = ti.field(ti.i32, shape=())
window = ti.ui.Window('Window Title', (1280, 720))
scene = ti.ui.Scene()
camera = ti.ui.Camera()
canvas = window.get_canvas()
canvas.set_background_color((0, 0, 0))

rot_theta = 10

rot_Mat = ti.Matrix([[1, 0, 0],
                     [0, ti.cos(rot_theta), -ti.sin(rot_theta)],
                     [0, ti.sin(rot_theta), ti.cos(rot_theta)]])


@ti.kernel
def init():
    frame[None] = 0
    # particle initialize
    for p in range(particle_num):
        ti_particle_pos[p] = [
            (ti.random() - 0.5) * 0.3 + 0.5,
            (ti.random() - 0.5) * 0.8 + 0.5,
            (ti.random() - 0.5) * 0.3 + 0.5,
        ]
        # ti_particle_pos[p]= rot_Mat@ti_particle_pos[p]
        ti_particle_vel[p] = [0, 0, 0]
        ti_particle_C[p] = ti.Matrix.zero(ti.f32, 3, 3)
        ti_particle_Fe[p] = ti.Matrix.identity(ti.f32, 3)
        ti_particle_Fp[p] = ti.Matrix.identity(ti.f32, 3)
        # ti_particle_Jp[p] = 1  # Plastic part of J
    # grid initialize
    for i, j, k in ti_grid_mass:
        ti_grid_mass[i, j, k] = 0
        ti_grid_vel[i, j, k] = [0, 0, 0]


@ti.func
def Z(sigma, alpha0):
    eps_p = ti.Matrix.zero(ti.f32, 3, 3)
    for d in ti.static(range(3)):
        eps_p[d, d] = ti.log(sigma[d, d])

    eps_hat_p = eps_p - ti.Matrix.identity(ti.f32, 3) * eps_p.trace() / 3
    # print('eps_p: ', eps_p)
    eps_hat_p_frobenius_norm = eps_hat_p.norm()
    # ti.sqrt(
    #   eps_hat_p[0] * eps_hat_p[0] + eps_hat_p[1] * eps_hat_p[1] + eps_hat_p[2] * eps_hat_p[2])
    delta_gamma = eps_hat_p_frobenius_norm + (3 * lambda_0 + 2 * mu_0) / (2 * mu_0) * eps_p.trace() * alpha0

    ret = sigma
    # case 1
    if delta_gamma < 0:
        ret = sigma
    # case 2
    elif eps_p.trace() > 0 or eps_hat_p_frobenius_norm < tolerance:
        ret = ti.Matrix.identity(ti.f32, 3)
        # print('case2: ' , ret)

    # case 3
    else:
        Hp = eps_p - delta_gamma * eps_hat_p / eps_hat_p_frobenius_norm
        ret = ti.exp(Hp)

    # print('ret: ',ret)
    return ret


@ti.func
def isNaN(x):
    return not (x < 0 or 0 < x or x == 0)


@ti.kernel
def substep():
    # init grid
    # can be optimized
    for i, j, k in ti_grid_mass:
        ti_grid_mass[i, j, k] = 0
    for i, j, k in ti_grid_vel:
        ti_grid_vel[i, j, k] = [0, 0, 0]

    # p2g
    for p in ti_particle_pos:
        Xp = ti_particle_pos[p] * grid_inv_dx
        base = int(Xp - 0.5)
        fx = Xp - base
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]  # quadratic kernel

        # plastic hardening
        U, sig, V = ti.svd(ti_particle_Fe[p])

        # projected_Fp = U @ projected_sig @ V.transpose()

        inv_sig = ti.Matrix.zero(ti.f32, 3, 3)
        log_sig = ti.Matrix.zero(ti.f32, 3, 3)
        for d in ti.static(range(3)):
            inv_sig[d, d] = 1 / sig[d, d]
            log_sig[d, d] = ti.log(sig[d, d])

        stress = -dt * 4 * (particle_initial_volume *
                            (U @ (
                                    2 * mu_0 * inv_sig @ log_sig + lambda_0 * log_sig.trace() * inv_sig) @ V.transpose()) @
                            ti_particle_Fe[p].transpose()) \
                 / grid_dx ** 2

        # print('stress: ',stress)
        affine = stress + particle_mass * ti_particle_C[p]

        # loop unrolling
        # scattering
        for i, j, k in ti.static(ti.ndrange(3, 3, 3)):
            offset = ti.Vector([i, j, k])
            weight = w[i].x * w[j].y * w[k].z
            dpos = (offset - fx) * grid_dx
            ti_grid_vel[base + offset] += weight * (particle_mass * ti_particle_vel[p] + affine @ dpos)
            ti_grid_mass[base + offset] += weight * particle_mass

    # grid update

    for i, j, k in ti_grid_mass:
        if ti_grid_mass[i, j, k] > 0:
            ti_grid_vel[i, j, k] /= ti_grid_mass[i, j, k]
            ti_grid_vel[i, j, k].y -= dt * gravity

        # cond = (I < bound) & (ti_grid_vel[I] < 0) | (I > grid_res - bound) & (ti_grid_vel[I] > 0)
        # ti_grid_vel[I] = ti.select(cond, 0, ti_grid_vel[I])

        if i < bound and ti_grid_vel[i, j, k].x < 0:
            ti_grid_vel[i, j, k] = [0, 0, 0]
        if i > grid_res - bound and ti_grid_vel[i, j, k].x > 0:
            ti_grid_vel[i, j, k] = [0, 0, 0]
        if j < bound and ti_grid_vel[i, j, k].y < 0:
            ti_grid_vel[i, j, k] = [0, 0, 0]
        if j > grid_res - bound and ti_grid_vel[i, j, k].y > 0:
            ti_grid_vel[i, j, k] = [0, 0, 0]
        if k < bound and ti_grid_vel[i, j, k].z < 0:
            ti_grid_vel[i, j, k] = [0, 0, 0]
        if k > grid_res - bound and ti_grid_vel[i, j, k].z > 0:
            ti_grid_vel[i, j, k] = [0, 0, 0]
        # if i < bound and ti_grid_vel[i, j, k].x < 0:
        #     ti_grid_vel[i, j, k].x = 0
        # if i > grid_res - bound and ti_grid_vel[i, j, k].x > 0:
        #     ti_grid_vel[i, j, k].x = 0
        # if j < bound and ti_grid_vel[i, j, k].y < 0:
        #     ti_grid_vel[i, j, k].y = 0
        # if j > grid_res - bound and ti_grid_vel[i, j, k].y > 0:
        #     ti_grid_vel[i, j, k].y = 0
        # if k < bound and ti_grid_vel[i, j, k].z < 0:
        #     ti_grid_vel[i, j, k].z = 0
        # if k > grid_res - bound and ti_grid_vel[i, j, k].z > 0:
        #     ti_grid_vel[i, j, k].z = 0

    # particle update
    for p in ti_particle_pos:
        # gather particle velocity
        Xp = ti_particle_pos[p] * grid_inv_dx
        base = int(Xp - 0.5)
        fx = Xp - base
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]  # quadratic kernel

        new_v = ti.zero(ti_particle_vel[p])
        new_C = ti.zero(ti_particle_C[p])

        # gathering
        for i, j, k in ti.static(ti.ndrange(3, 3, 3)):
            offset = ti.Vector([i, j, k])
            dpos = (offset - fx) * grid_dx
            weight = w[i].x * w[j].y * w[k].z

            new_v += weight * ti_grid_vel[base + offset]
            new_C += 4 * weight * ti_grid_vel[base + offset].outer_product(dpos) / grid_dx ** 2

        # particle update

        # deformation gradient update
        # first update deformation gradient
        ti_particle_vel[p] = new_v
        ti_particle_C[p] = new_C
        ti_particle_pos[p] += dt * ti_particle_vel[p]

        ti_particle_Fe[p] = (ti.Matrix.identity(ti.f32, 3) + dt * ti_particle_C[p]) @ ti_particle_Fe[p]
        U, sig, V = ti.svd(ti_particle_Fe[p])

        sig_vec = ti.Vector([sig[0, 0], sig[1, 1], sig[2, 2]])
        eps_p = ti.Vector([ti.log(sig[0, 0]), ti.log(sig[1, 1]), ti.log(sig[2, 2])])
        eps_p_trace = ti.log(sig[0, 0]) + ti.log(sig[1, 1]) + ti.log(sig[2, 2])
        eps_hat_p = eps_p - ti.Vector([eps_p_trace, eps_p_trace, eps_p_trace]) / 3
        # eps_hat_p = eps_p - ti.Matrix.identity(ti.f32, 3) * eps_p.trace() / 3
        # print('eps_p: ', eps_p)
        eps_hat_p_frobenius_norm = eps_hat_p.norm()
        # ti.sqrt(
        #   eps_hat_p[0] * eps_hat_p[0] + eps_hat_p[1] * eps_hat_p[1] + eps_hat_p[2] * eps_hat_p[2])
        delta_gamma = eps_hat_p_frobenius_norm + (3 * lambda_0 + 2 * mu_0) / (2 * mu_0) * eps_p_trace * alpha

        # if isNaN(delta_gamma):
        #     print('norm', eps_hat_p_frobenius_norm)
        #     # print('lambda: ',lambda_0)
        #     # print('mu: ', mu_0)
        #     print('trace: ', eps_p.trace())

        # print(delta_gamma)
        projected_sig = sig_vec
        # case 1

        if delta_gamma < 0:
            projected_sig = sig_vec
            # print('case1')
        # case 2
        elif eps_p_trace > 0 or eps_hat_p_frobenius_norm < tolerance:
            projected_sig = ti.Vector([1, 1, 1])  # ti.Matrix.identity(ti.f32, 3)
            # print('case2')[]
        # case 3
        else:
            Hp = eps_p - delta_gamma * eps_hat_p / eps_hat_p_frobenius_norm
            projected_sig = ti.exp(Hp)
            # print('case3')

        # print('sig: ', sig)
        # print('projected sig: ', projected_sig)
        projected_sig_mat = ti.Matrix([
            [projected_sig[0], 0, 0],
            [0, projected_sig[1], 0],
            [0, 0, projected_sig[2]]
        ])
        ti_particle_Fe[p] = U @ projected_sig_mat @ V.transpose()


def render_gui():
    global particle_radius
    global particle_color

    # global E
    window.GUI.begin("Render setting", 0.02, 0.02, 0.4, 0.15)
    particle_color = window.GUI.color_edit_3("particle color", particle_color)
    particle_radius = window.GUI.slider_float("particle radius", particle_radius, 0.001, 0.1)
    if window.GUI.button("restart"):
        init()
    window.GUI.end()

    # window.GUI.begin("Simulation setting", 0.02, 0.19, 0.3, 0.1)
    #
    # window.GUI.end()


def render():
    camera.track_user_inputs(window, movement_speed=0.01, hold_key=ti.ui.RMB)
    scene.set_camera(camera)

    scene.point_light((0.5, 2, 0.5), (1, 1, 1))

    scene.particles(ti_particle_pos, particle_radius, particle_color)
    canvas.scene(scene)


if __name__ == '__main__':
    init()

    camera.position(2, 2, 2)
    camera.lookat(0, 0, 0)
    camera.up(0, 1, 0)
    camera.fov(55)
    camera.projection_mode(ti.ui.ProjectionMode.Perspective)

    # print(ti_particle_Fe)

    while window.running:
        for s in range(int(5)):
            substep()
        #     # print(ti_particle_pos)

        frame[None] += 1

        render()
        render_gui()
        window.show()

    print("hello")