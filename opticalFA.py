from math import sin, cos, pi, sqrt, ceil
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from scipy import integrate
from progress.bar import IncrementalBar


# === Configuration ===
STEP = 0  # start step (useful to resume long computations)

c = 299_792_458  # speed of light, m/s
N_EMITTERS = 3 * 15  # number of emitters
A_max = [1 / N_EMITTERS for _ in range(N_EMITTERS)]  # amplitude per emitter

# wavelengths (m)
wavelengths = [700e-9 for _ in range(N_EMITTERS)]  # 700 nm (red)

# frequencies (Hz)
freqs = [c / wl for wl in wavelengths]

# distance from emitters to screen (m)
dist_to_screen = [20.0 for _ in range(N_EMITTERS)]

# --- geometry: concentric arrangement with rotated polygons ---
N_ANGLE = 3  # number of vertices in the polygon (triangle, etc.)
K_ANGLE = 30  # rotation angle per ring (degrees)
RING_STEP = 0.001  # base radius for the smallest polygon (m)

radii = [
    RING_STEP * ((i // N_ANGLE) + 1) for i in range(N_EMITTERS)
]  # radii of emitters
angles = []
for i in range(int(N_EMITTERS / N_ANGLE)):
    for j in range(N_ANGLE):
        angles.append((j * 360 / N_ANGLE + i * K_ANGLE) * pi / 180)

# Reference point for maximum (assumed center)
x_max = 0.0
y_max = 0.0


def L_max(i: int) -> float:
    """
    Distance from emitter i to the current reference point (x_max, y_max).
    Used to set relative initial phases so that emitter 0 is reference.
    """
    dx = x_max + radii[i] * cos(angles[i])
    dy = y_max + radii[i] * sin(angles[i])
    return sqrt(dist_to_screen[i] ** 2 + dx ** 2 + dy ** 2)


# initial phases so that emitter 0 is reference
phi = [
    (2 * pi * (L_max(0) - L_max(i)) / wavelengths[i]) % (2 * pi)
    for i in range(N_EMITTERS)
]


def amplitude(x: float, y: float, t: float, i: int) -> float:
    """
    Instantaneous amplitude from emitter i at screen point (x, y) and time t.
    Uses sin(omega * t + k * distance + initial_phase).
    """
    dx = x + radii[i] * cos(angles[i])
    dy = y + radii[i] * sin(angles[i])
    distance = sqrt(dist_to_screen[i] ** 2 + dx ** 2 + dy ** 2)
    k = 2 * pi / wavelengths[i]
    omega = 2 * pi * freqs[i]
    return A_max[i] * sin(omega * t + k * distance + phi[i])
def intensity_instant(x: float, y: float, t: float) -> float:
    """
    Instantaneous intensity proportional to square of summed amplitudes.
    I(t, x, y) = (sum_i A_i(t, x, y))^2
    """
    s = 0.0
    for i in range(N_EMITTERS):
        s += amplitude(x, y, t, i)
    return s ** 2


# Integration time and step settings (time averaging)
max_period = max(1.0 / f for f in freqs)
min_period = min(1.0 / f for f in freqs)
T_INTEGR = max_period * 10  # integrate over 10 periods of the slowest wave
DT_SUGGESTED = min_period / 100  # suggested integration step (not directly used)


def draw_rect():
    """
    Compute and draw the interference pattern on a rectangular patch.
    Results are saved to 'graf.png'. A second figure is saved with emitter positions.
    """
    # compute maximum brightness at the reference point
    max_brightness = integrate.quad(
        lambda t: intensity_instant(x_max, y_max, t), 0, T_INTEGR
    )[0]

    # image size (meters)
    lx = 0.05
    ly = 0.05

    plt.figure(1)
    axes = plt.gca()
    axes.set_aspect("equal")
    plt.xlim(-lx / 2, lx / 2)
    plt.ylim(-ly / 2, ly / 2)

    dx = 0.0001
    dy = dx

    nx = ceil(lx / dx)
    ny = ceil(ly / dy)

    bar = IncrementalBar("Progress", max=nx)

    for ix in range(STEP, nx):
        x = ix * dx - lx / 2
        bar.next()
        for jy in range(ny):
            y = jy * dy - ly / 2
            temp_integral = integrate.quad(
                lambda t: intensity_instant(x, y, t), 0, T_INTEGR
            )[0]
            # normalize brightness
            norm = temp_integral / max_brightness if max_brightness > 0 else 0.0
            red = min(max(norm, 0.0), 1.0)
            green = 0.0
            # if intensity exceeds max at off-center and is not at center, mark green
            if (sqrt((x - x_max) ** 2 + (y - y_max) ** 2) > 0.003) and (
                temp_integral > max_brightness
            ):
                red = 0.0
                green = 1.0

            rect = mpatches.Rectangle((x, y), dx, dy, color=(red, green, 0.0))
            axes.add_patch(rect)

    plt.savefig("graf.png", dpi=300)
    bar.finish()

    # draw emitter positions (scaled to mm for better visualization)
    plt.figure(2)
    axes2 = plt.gca()
    axes2.set_aspect("equal")
    for i in range(N_EMITTERS):
        plt.plot(radii[i] * cos(angles[i]) * 1000, radii[i] * sin(angles[i]) * 1000, "ko")
    plt.savefig("emitters.png", dpi=300)


if __name__ == "__main__":

    draw_rect()
