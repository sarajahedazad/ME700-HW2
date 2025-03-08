import numpy as np
import matplotlib.pyplot as plt
import sympy
from geometry import *
from stiffness_matrices import * 

def interpolate_two_points(p0, p1, n):
    points_lst = [tuple(p0 + (p1 - p0) * i / n) for i in range(n + 1)]
    return np.array(points_lst)

class ShapeFunctions:
    def __init__(self, eigenvector, frame_obj, n=20, scale=5):
        self.eigenvector = eigenvector
        self.frame = frame_obj
        self.n = n
        self.scale = scale
        self.x_local = sympy.symbols(f'x:{self.n}')

    def transformation_1212_matrix_3D(self, gamma):
        Gamma = np.zeros((12, 12))
        Gamma[0:3, 0:3] = gamma
        Gamma[3:6, 3:6] = gamma
        Gamma[6:9, 6:9] = gamma
        Gamma[9:12, 9:12] = gamma
        return Gamma

    def transformation_nn_matrix_3D(self, gamma):
        Gamma = np.zeros((3 * self.n, 3 * self.n))
        for i in range(self.n):
            Gamma[3 * i: 3 * i + 3, 3 * i: 3 * i + 3] = gamma
        return Gamma

    def evaluate(self, expr, x0):
        func = sympy.lambdify(self.x_local, expr, 'numpy')
        func_val = func(*x0)
        return func_val.reshape(-1)

    def linear_N1(self, length):
        expr = sympy.Matrix([1 - self.x_local[i] / length for i in range(self.n)])
        return expr

    def linear_N2(self, length):
        expr = sympy.Matrix([self.x_local[i] / length for i in range(self.n)])
        return expr

    def hermite_N1(self, length):
        expr = sympy.Matrix([1 - 3 * (self.x_local[i] / length) ** 2 + 2 * (self.x_local[i] / length) ** 3 for i in range(self.n)])
        return expr

    def hermite_N2(self, length):
        expr = sympy.Matrix([3 * (self.x_local[i] / length) ** 2 - 2 * (self.x_local[i] / length) ** 3 for i in range(self.n)])
        return expr 

    def hermite_N3(self, length):
        expr = sympy.Matrix([self.x_local[i] * (1 - self.x_local[i] / length) ** 2 for i in range(self.n)])
        return expr

    def hermite_N4(self, length):
        expr = sympy.Matrix([self.x_local[i] * ((self.x_local[i] / length) ** 2 - self.x_local[i] / length) for i in range(self.n)])
        return expr

    def get_element_info(self, element_idx):
        connection = self.frame.connectivities[element_idx]
        p0_idx, p1_idx = connection
        p0 = self.frame.points[p0_idx]
        p1 = self.frame.points[p1_idx]
        length = self.frame.L_array[element_idx]
        v_temp = self.frame.v_temp_array[element_idx]
        return connection, p0_idx, p0, p1_idx, p1, length, v_temp

    def get_eigenvector_element_global(self, p0_idx, p1_idx):
        return np.concatenate((self.eigenvector[6 * p0_idx: 6 * p0_idx + 6], self.eigenvector[6 * p1_idx: 6 * p1_idx + 6]))


    def calc_element_interpolation(self, element_idx):
        connection, p0_idx, p0, p1_idx, p1, length, v_temp = self.get_element_info(element_idx)
        gamma = rotation_matrix_3D(p0[0], p0[1], p0[2], p1[0], p1[1], p1[2], v_temp)
        Gamma = self.transformation_1212_matrix_3D(gamma)
        eigenvector_el_global = self.get_eigenvector_element_global(p0_idx, p1_idx)
        eigenvector_el_local = Gamma @ eigenvector_el_global

        u_p0, v_p0, w_p0, theta_p0_x, theta_p0_y, theta_p0_z, u_p1, v_p1, w_p1, theta_p1_x, theta_p1_y, theta_p1_z = eigenvector_el_local
        x_local_val = np.linspace(0, length, num=self.n)

        u_local = u_p0 * self.evaluate(self.linear_N1(length), x_local_val) + u_p1 * self.evaluate(self.linear_N2(length), x_local_val)
        v_local_part1 = v_p0 * self.evaluate(self.hermite_N1(length), x_local_val) + v_p1 * self.evaluate(self.hermite_N2(length), x_local_val)
        v_local_part2 = theta_p0_z * self.evaluate(self.hermite_N3(length), x_local_val) + theta_p1_z * self.evaluate(self.hermite_N4(length), x_local_val)
        v_local = v_local_part1 + v_local_part2
        w_local_part1 =  w_p0 * self.evaluate(self.hermite_N1(length), x_local_val) + w_p1 * self.evaluate(self.hermite_N2(length), x_local_val)
        w_local_part2 =   theta_p0_y * self.evaluate(self.hermite_N3(length), x_local_val) + theta_p1_y * self.evaluate(self.hermite_N4(length), x_local_val)
        w_local = w_local_part1 + w_local_part2

        stacked = np.stack((self.scale * u_local, self.scale * v_local, self.scale * w_local))
        uvw_local = stacked.flatten('F')

        Gamma_nn = self.transformation_nn_matrix_3D(gamma)
        uvw_global_element = Gamma_nn.T @ uvw_local

        interpolated_points = interpolate_two_points(p0, p1, self.n - 1)
        return interpolated_points + uvw_global_element.reshape(-1, 3)

    def plot_element_interpolation(self, saving_dir_with_name):
        points = self.frame.points
        connectivities = self.frame.connectivities
    
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
    
        # Plot nodes
        for i, point in enumerate( points ):
            if i == 0 :
                ax.scatter(*point, color='black', label='Nodes')
            else:
                ax.scatter(*point, color='black')
    
        # Plot elements
        for i, connection in enumerate( connectivities ):
            p0 = points[connection[0]]
            p1 = points[connection[1]]
            if i == 0:
                ax.plot([p0[0], p1[0]], [p0[1], p1[1]], [p0[2], p1[2]], 'k--', label='Elements')
            else:
               ax.plot([p0[0], p1[0]], [p0[1], p1[1]], [p0[2], p1[2]], 'k--' )
    
        # Plot shape functions
        for j in range(len(connectivities)):
            element_interpolated = self.calc_element_interpolation(j)
            for i in range(len(element_interpolated) - 1):
                p0 = element_interpolated[i]
                p1 = element_interpolated[i + 1]
                if i == 0 and j == 0:
                    ax.plot([p0[0], p1[0]], [p0[1], p1[1]], [p0[2], p1[2]], 'r', label='Interpolated Shape')
                else:
                    ax.plot([p0[0], p1[0]], [p0[1], p1[1]], [p0[2], p1[2]], 'r' )
    
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.legend()
        ax.grid(True)
        ax.set_aspect('equal')

        plt.show()

        plt.savefig( saving_dir_with_name )
    
        
