import numpy as np
import matplotlib.pyplot as plt
import os

def compute_hand_aperture(grasp_box_width):
    # This function computes the motor command required to acheive an aperture of size grasp_box_width
    coef_path = 'aperture_mapping.txt'
    if os.path.exists(coef_path):
        coeffs = np.loadtxt(coef_path)
    else:
        motor_commands = np.arange(0, 1100, 100)
        aperture_size = np.array([90, 88, 75, 60, 40, 30, 18, 10, 8, 5.9, 5.4])
        coeffs = np.polynomial.polynomial.polyfit(aperture_size, motor_commands, 4)
        np.savetxt(coef_path, coeffs)
    return np.polynomial.polynomial.polyval(grasp_box_width, coeffs)



motor_commands = np.arange(0, 1100, 100)
# aperture_size = np.array([90,81,72,58,45,29,19,12.58,10,5.9,0])
aperture_size = np.array([90,88,75,60,40,30,18,10,8,5.9,5.4])
# m, b = np.polyfit(motor_commands, aperture_size, 1)
# coeffs = np.polynomial.polynomial.polyfit(motor_commands, aperture_size, 4)
# ffit = np.polynomial.polynomial.polyval(motor_commands, coeffs)
#
# fig, ax = plt.subplots()
# ax.plot(motor_commands, aperture_size, label='Measurement')
# plt.plot(motor_commands, m*motor_commands + b, label='Linear Regression')
# ax.plot(motor_commands, ffit, label='Polynomial (degree = 4) Regression')
# ax.set_title('Hand Aperture Mapping')
# ax.set_xlabel('Motor Commands')
# ax.set_ylabel('Aperture Size (mm)')
# plt.xticks(motor_commands)
# plt.grid()
# plt.legend(loc="upper right")
# plt.show()
# import code; code.interact(local=dict(globals(), **locals()))

m, b = np.polyfit(aperture_size, motor_commands, 1)
coeffs = np.polynomial.polynomial.polyfit(aperture_size, motor_commands, 4)
ffit = np.polynomial.polynomial.polyval(aperture_size, coeffs)

fig, ax = plt.subplots()
ax.plot(aperture_size, motor_commands, label='Measurement')
plt.plot(aperture_size, m*aperture_size + b, label='Linear Regression')
ax.plot(aperture_size, ffit, label='Polynomial (degree = 4) Regression')
ax.set_title('Hand Aperture Mapping')
ax.set_xlabel('Aperture Size (mm)')
ax.set_ylabel('Motor Commands')
plt.yticks(motor_commands)
# plt.xticks(np.arange(np.min(aperture_size), np.max(aperture_size), (np.max(aperture_size)-np.min(aperture_size))/11))
plt.grid()
plt.legend(loc="upper right")
plt.show()
import code; code.interact(local=dict(globals(), **locals()))