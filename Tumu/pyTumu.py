'''
Title: pyTumu, Develope an inelasticity FEM software in python from zero
Function: Inelatic FEM computing library
Author: Jianhao Gao, University of Florida, College of Design, Construction, and Planning
Initial:
      2023/3/26, a good start is half of success, added FEM linear solver
Edit: 2023/3/27
          -4/14, added 2D truss and frame, euler and timoshenko beam
Edit: 2023/4/15, added 2d solid elastic static simulation
Edit: 2023/4/18, added Newmark method for dynamic problem
Edit: 2023/4/19, added drawingMeshField for 2d problem plot
Edit: 2023/4/20, added von mises stress computing function, J2 plasticity criteria
Edit: 2023/4/21, added 2d inelasticity computing structure
Edit: 2023/4/25, changed stress2D to strainstress2D
Edit: 2023/4/26, added internal force computing in strainstress2D function
Edit: 2023/4/27
         -/4/29, finished J2 quasi-static simulation, see QuasiStatic_Plastic.py
'''

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import matplotlib.collections as collections


# %% Linear Solver
def solution(GDof, prescribed_dof, stiffness, force):
    '''
    :param GDof: int, degree of freedom
    :param prescribed_dof: (m, )
    :param stiffness: (n, n)
    :param force: (n, )
    :return: displacements: (n, )
    '''
    activeDof = np.setdiff1d(np.arange(GDof), prescribed_dof)
    displacements = np.zeros(GDof)
    displacements[activeDof] = np.linalg.solve(stiffness[np.ix_(activeDof, activeDof)], force[activeDof])
    return displacements


# %% Shape functions
def shape_function_l2(xi):
    '''
    :param xi: natural coordinates (-1 ... +1)
    :return: shape value and derivative values
    '''
    # shape function and derivatives for L2 elements
    # shape : Shape functions
    # natural_derivatives: derivatives w.r.t. xi
    # xi: natural coordinates (-1 ... +1)
    shape = np.array([(1 - xi) / 2, (1 + xi) / 2])
    natural_derivatives = np.array([-0.5, 0.5])

    return shape, natural_derivatives


def shape_function_Q4(xi, eta):
    '''
    :param xi: natural coordinates between (-1 ... +1)
    :param eta: natural coordinates between (-1 ... +1)
    :return: shape (4, ) , naturalDerivatives (4, 2)
    shape values at [xi, eta] =
    [N1(xi, eta) N2(xi, eta) N3(xi, eta) N4(xi, eta)]

    naturalDerivatives at [xi, eta] =
    [dN1/dxi dN1/deta]
    [dN2/dxi dN2/deta]
    [dN3/dxi dN3/deta]
    [dN4/dxi dN4/deta]
    '''

    shape = 1 / 4 * np.array([(1 - xi) * (1 - eta),
                              (1 + xi) * (1 - eta),
                              (1 + xi) * (1 + eta),
                              (1 - xi) * (1 + eta)])

    naturalDerivatives = 1 / 4 * np.array([[-(1 - eta), -(1 - xi)],
                                           [1 - eta, -(1 + xi)],
                                           [1 + eta, 1 + xi],
                                           [-(1 + eta), 1 - xi]])

    return shape, naturalDerivatives


# %% Truss problem
def finite_element_analysis_1D_truss_static(number_nodes, elemnt_stiffness, element_nodes, prescribed_dof, force):
    # Number of elements
    number_elements = element_nodes.shape[0]

    # Displacement vector and stiffness matrix
    displacements = np.zeros(number_nodes)
    stiffness = np.zeros((number_nodes, number_nodes))

    # Computation of the system stiffness matrix
    for e in range(number_elements):
        element_dof = element_nodes[e, :]
        stiffness[np.ix_(element_dof, element_dof)] += elemnt_stiffness[e] * np.array([[1, -1], [-1, 1]])

    # Active degrees of freedom
    active_dof = np.setdiff1d(np.arange(number_nodes), prescribed_dof)

    # Solution
    displacements[active_dof] = np.linalg.solve(stiffness[np.ix_(active_dof, active_dof)], force[active_dof])
    force = np.dot(stiffness, displacements)

    return {
        'elementNodes': element_nodes,
        'elem_Dof': number_nodes,
        'numberElements': number_elements,
        'numberNodes': number_nodes,
        'displacements': displacements,
        'force': force,
        'stiffness': stiffness,
        'prescribedDof': prescribed_dof
    }


def stresses_2D_truss(element_nodes, node_coordinates, node_coordinates_end, E):
    stress = []
    for element_node in element_nodes:
        strain = np.linalg.norm(
            node_coordinates_end[element_node[0]] - node_coordinates_end[element_node[1]]) / np.linalg.norm(
            node_coordinates[element_node[0]] - node_coordinates[element_node[1]]) - 1
        stress.append(strain * E)
    return stress


def form_stiffness_2D_truss(GDof, number_elements, element_nodes, node_coordinates, EA):
    '''
    :param GDof: int, the Global degree of freedom
    :param number_elements: int, the number of frame elements
    :param element_nodes: (number_element, 2), the first dimension is the element indice,
                        the second dimension is connected nodes indices
    :param node_coordinates: (number_nodes, 2), The coordinates of each node.
    :param EA: Compressive Stiffness
    :return: stiffness matrix
    '''

    stiffness = np.zeros((GDof, GDof))

    for e in range(number_elements):
        indice = element_nodes[e, :]
        element_dof = np.array([indice[0] * 2, indice[0] * 2 + 1, indice[1] * 2, indice[1] * 2 + 1])
        node_coordinate_1 = node_coordinates[indice[0]]
        node_coordinate_2 = node_coordinates[indice[1]]
        xa = node_coordinate_1[0] - node_coordinate_2[0]
        ya = node_coordinate_1[1] - node_coordinate_2[1]
        length_element = np.linalg.norm(node_coordinate_1 - node_coordinate_2)
        C = xa / length_element
        S = ya / length_element
        k1 = EA / length_element * np.array([
            [C * C, C * S, -C * C, -C * S],
            [C * S, S * S, -C * S, -S * S],
            [-C * C, -C * S, C * C, C * S],
            [-C * S, -S * S, C * S, S * S]
        ])
        stiffness[np.ix_(element_dof, element_dof)] += k1

    return stiffness


def plot_2D_truss_static(node_coordinates_end, reactions, stress, element_nodes):
    # Create a scatter plot of the node coordinates

    plt.scatter(node_coordinates_end[:, 0], node_coordinates_end[:, 1])

    # Plot reactions at corresponding nodes
    for i, reaction in enumerate(reactions):
        plt.annotate(
            f"R{i}: ({reaction[0]:.2f}, {reaction[1]:.2f})",
            (node_coordinates_end[i, 0], node_coordinates_end[i, 1]),
            textcoords="offset points",
            xytext=(0, 10),
            ha='center',
        )

    # Draw lines between nodes according to element_nodes and color them based on stress
    for i, (node1, node2) in enumerate(element_nodes):
        x_values = [node_coordinates_end[node1, 0], node_coordinates_end[node2, 0]]
        y_values = [node_coordinates_end[node1, 1], node_coordinates_end[node2, 1]]

        # Assign colors to stress values

        if stress[i] < 0:
            color = 'blue'
        else:
            color = 'red'

        plt.plot(x_values, y_values, color=color)

    maxlim = 1.4 * node_coordinates_end.max()
    minlim = 0.9 * node_coordinates_end.min()
    plt.xlim([minlim, maxlim])
    plt.ylim([minlim, maxlim])
    plt.axis('equal')
    plt.show()


# %% Frame problem


def form_stiffness_2D_frame(GDof, number_nodes, number_elements, element_nodes, node_coordinates, EI, EA):
    '''
    :param GDof: int, the Global degree of freedom
    :param number_nodes: int, the number of nodes
    :param number_elements: int, the number of frame elements
    :param element_nodes: (number_element, 2), the first dimension is the element indice,
                        the second dimension is connected nodes indices
    :param node_coordinates: (number_nodes, 2), The coordinates of each node.
    :param EI: Bending Stiffness
    :param EA: Compressive Stiffness
    :return: stiffness matrix (3*number_nodes, 3*number_nodes)
    [:number_nodes] rows:                global horizontal displacements
    [number_nodes: 2*numner_nodes] rows: global verticall displacements
    [2*number_nodes: ] rows:             global rotation displacements
    '''
    xx, yy = node_coordinates[:, 0], node_coordinates[:, 1]
    stiffness = np.zeros((GDof, GDof))

    # Computation of the system stiffness matrix
    for e in range(number_elements):
        # Element degrees of freedom (Dof)
        indice = element_nodes[e, :]
        # indice: (2, )
        element_dof = np.hstack((indice, indice + number_nodes, indice + 2 * number_nodes))
        '''element_dof: (2, 3)
        [compressive_left, compressive_right]
        [vertical_left,    vertical_right]
        [rotation_left,    rotation_right]
        '''

        xa = xx[indice[1]] - xx[indice[0]]
        ya = yy[indice[1]] - yy[indice[0]]
        length_element = np.sqrt(xa * xa + ya * ya)
        cosa = xa / length_element
        sena = ya / length_element
        ll = length_element

        L = np.block([
            [cosa * np.eye(2), sena * np.eye(2), np.zeros((2, 2))],
            [-sena * np.eye(2), cosa * np.eye(2), np.zeros((2, 2))],
            [np.zeros((2, 4)), np.eye(2)]
        ])

        oneu = np.array([[1, -1], [-1, 1]])
        oneu2 = np.array([[1, -1], [1, -1]])
        oneu3 = np.array([[1, 1], [-1, -1]])
        oneu4 = np.array([[4, 2], [2, 4]])

        k1 = np.block([
            [EA / ll * oneu, np.zeros((2, 4))],
            [np.zeros((2, 2)), 12 * EI / ll ** 3 * oneu, 6 * EI / ll ** 2 * oneu3],
            [np.zeros((2, 2)), 6 * EI / ll ** 2 * oneu2, EI / ll * oneu4]
        ])

        stiffness[np.ix_(element_dof, element_dof)] += L.T @ k1 @ L

    return stiffness


def form_lumped_mass_2D_frame(GDof, number_nodes, number_elements, element_nodes, node_coordinates, rho, Area):
    '''
    :param GDof: int, the Global degree of freedom
    :param number_nodes: int, the number of nodes
    :param number_elements: int, the number of frame elements
    :param element_nodes: (number_element, 2), the first dimension is the element indice,
                        the second dimension is connected nodes indices
    :param node_coordinates: (number_nodes, 2), The coordinates of each node.
    :param EI: Bending Stiffness
    :param EA: Compressive Stiffness
    :return: mass matrix (3*number_nodes, 3*number_nodes)
    [:number_nodes] rows:                global horizontal displacements
    [number_nodes: 2*numner_nodes] rows: global verticall displacements
    [2*number_nodes: ] rows:             global rotation displacements
    '''
    xx, yy = node_coordinates[:, 0], node_coordinates[:, 1]
    mass = np.zeros((GDof, GDof))

    # Computation of the system stiffness matrix
    for e in range(number_elements):
        # Element degrees of freedom (Dof)
        indice = element_nodes[e, :]
        # indice: (2, )
        element_dof = np.hstack((indice, indice + number_nodes, indice + 2 * number_nodes))
        '''element_dof: (2, 3)
        [compressive_left, compressive_right]
        [vertical_left,    vertical_right]
        [rotation_left,    rotation_right]
        '''

        xa = xx[indice[1]] - xx[indice[0]]
        ya = yy[indice[1]] - yy[indice[0]]
        length_element = np.sqrt(xa * xa + ya * ya)
        cosa = xa / length_element
        sena = ya / length_element
        ll = length_element

        L = np.block([
            [cosa * np.eye(2), sena * np.eye(2), np.zeros((2, 2))],
            [-sena * np.eye(2), cosa * np.eye(2), np.zeros((2, 2))],
            [np.zeros((2, 4)), np.eye(2)]
        ])

        oneu = np.array([[1, 0], [0, 1]])

        m_bending = rho * Area * ll / 420 * np.array([[156, 22 * ll, 54, -13 * ll],
                                                      [22 * ll, 4 * ll * ll, 13 * ll, -3 * ll * ll],
                                                      [54, 13 * ll, 156, -22 * ll],
                                                      [-13 * ll, -3 * ll * ll, -22 * ll, 4 * ll * ll]])

        m_total = np.block([
            [rho * Area * ll / 2 * np.array([[1, 0], [0, 1]]), np.zeros((2, 4))],
            [np.zeros((4, 2)), m_bending]
        ])

        mass[np.ix_(element_dof, element_dof)] += L.T @ m_total @ L

    return mass


def plot_2D_frame_static(node_coordinates, node_coordinates_end, reactions, element_nodes):
    plt.scatter(node_coordinates_end[:, 0], node_coordinates_end[:, 1])
    reactions = reactions.reshape([len(reactions) // 3, 3])
    for i, reaction in enumerate(reactions):
        plt.annotate(
            f"R{i}: ({reaction[0]:.2f}, \n{reaction[1]:.2f}, \n{reaction[2]:.2f}))",
            (node_coordinates_end[i, 0], node_coordinates_end[i, 1]),
            textcoords="offset points",
            xytext=(0, 10),
            ha='center',
        )

    for i, (node1, node2) in enumerate(element_nodes):
        x_values = [node_coordinates[node1, 0], node_coordinates[node2, 0]]
        y_values = [node_coordinates[node1, 1], node_coordinates[node2, 1]]
        plt.plot(x_values, y_values, color='k')

    # Draw lines between nodes according to element_nodes and color them based on stress
    for i, (node1, node2) in enumerate(element_nodes):
        x_values = [node_coordinates_end[node1, 0], node_coordinates_end[node2, 0]]
        y_values = [node_coordinates_end[node1, 1], node_coordinates_end[node2, 1]]
        plt.plot(x_values, y_values, color='red')

        # Assign colors to stress values

        # if stress[i] < 0:
        #     color = 'blue'
        # else:
        #     color = 'red'
        #
        # plt.plot(x_values, y_values, color=color)

    # maxlim = 1.4 * node_coordinates_end.max()
    # minlim = 0.9 * node_coordinates_end.min()
    maxlim = 9
    minlim = -1
    plt.xlim([minlim, maxlim])
    plt.ylim([minlim, maxlim])
    plt.axis('equal')


# %% Euler bernoulli and Timoshenko Beam

def form_stiffness_bernoulli_beam(GDof: int,
                                  number_elements: int,
                                  element_nodes: np.ndarray,
                                  node_coordinates: np.ndarray,
                                  EI: float,
                                  P: float) -> [float, float]:
    force = np.zeros(GDof)
    stiffness = np.zeros((GDof, GDof))

    for e in range(number_elements):
        indice = element_nodes[e]
        elementDof = np.array([2 * indice[0], 2 * indice[0] + 1, 2 * indice[1], 2 * indice[1] + 1])
        LElem = node_coordinates[:, 0][indice[1]] - node_coordinates[:, 0][indice[0]]

        k1 = EI / (LElem ** 3) * np.array([[12, 6 * LElem, -12, 6 * LElem],
                                           [6 * LElem, 4 * LElem ** 2, -6 * LElem, 2 * LElem ** 2],
                                           [-12, -6 * LElem, 12, -6 * LElem],
                                           [6 * LElem, 2 * LElem ** 2, -6 * LElem, 4 * LElem ** 2]])
        f1 = np.array([P * LElem / 2, P * LElem ** 2 / 12, P * LElem / 2, -P * LElem ** 2 / 12])

        force[elementDof] += f1
        stiffness[np.ix_(elementDof, elementDof)] += k1

    return stiffness, force


def form_stiffness_mass_timoshenko_beam(GDof, number_elements, element_nodes, number_nodes, xx, C, P, rho, I,
                                        thickness):
    # computation of stiffness matrix and force vector
    # for Timoshenko beam element
    stiffness = np.zeros((GDof, GDof))
    mass = np.zeros((GDof, GDof))
    force = np.zeros(GDof)

    # stiffness matrix
    gauss_locations = np.array([0.577350269189626, -0.577350269189626])
    gauss_weights = np.ones(2)

    # bending contribution for stiffness matrix
    for e in range(number_elements):
        indice = element_nodes[e, :]
        element_dof = np.hstack((indice, indice + number_nodes))

        length_element = xx[indice[1]] - xx[indice[0]]
        det_jacobian = length_element / 2
        inv_jacobian = 1 / det_jacobian

        for q in range(gauss_weights.shape[0]):
            pt = gauss_locations[q]
            shape, natural_derivatives = shape_function_l2(pt)
            x_derivatives = natural_derivatives * inv_jacobian

            # B matrix
            B = np.zeros((2, 2 * len(indice)))
            B[1, len(indice): 2 * len(indice)] = x_derivatives

            # K
            stiffness[np.ix_(element_dof, element_dof)] += (
                    B.T @ B * gauss_weights[q] * det_jacobian * C[0, 0]
            )
            force[indice] += shape * P * det_jacobian * gauss_weights[q]
            mass[np.ix_(indice + number_nodes, indice + number_nodes)] += (
                    np.outer(shape, shape) * gauss_weights[q] * I * rho * det_jacobian
            )
            mass[np.ix_(indice, indice)] += (
                    np.outer(shape, shape) * gauss_weights[q] * thickness * rho * det_jacobian
            )

    # shear contribution for stiffness matrix
    gauss_locations = np.array([0.0])
    gauss_weights = np.array([2.0])

    for e in range(number_elements):
        indice = element_nodes[e, :]
        element_dof = np.hstack((indice, indice + number_nodes))

        length_element = xx[indice[1]] - xx[indice[0]]
        det_j0 = length_element / 2
        inv_j0 = 1 / det_j0

        for q in range(gauss_weights.shape[0]):
            pt = gauss_locations[q]
            shape, natural_derivatives = shape_function_l2(pt)
            x_derivatives = natural_derivatives * inv_jacobian

            # B
            B = np.zeros((2, 2 * len(indice)))
            B[1, : len(indice)] = x_derivatives
            B[1, len(indice): 2 * len(indice)] = shape

            # K
            stiffness[np.ix_(element_dof, element_dof)] += (
                    B.T @ B * gauss_weights[q] * det_jacobian * C[1, 1]
            )

    return stiffness, force, mass


# %% Solid mechanics
def rectangularMesh(Lx, Ly, numberElementsX, numberElementsY) -> Tuple[float, int]:
    deltaX = Lx / numberElementsX
    deltaY = Ly / numberElementsY
    nodeCoordinates = np.zeros(((numberElementsX + 1) * (numberElementsY + 1), 2))
    elementNodes = np.zeros((numberElementsX * numberElementsY, 4))
    k = 1
    for j in range(1, numberElementsY + 2):
        for i in range(1, numberElementsX + 2):
            nodeCoordinates[k - 1, :] = [(i - 1) * deltaX, (j - 1) * deltaY]
            k += 1

    k = 1
    for j in range(1, numberElementsY + 1):
        for i in range(1, numberElementsX + 1):
            n1 = (j - 1) * (numberElementsX + 1) + i - 1
            n2 = n1 + 1
            n3 = n2 + numberElementsX + 1
            n4 = n3 - 1
            elementNodes[k - 1, :] = [n1, n2, n3, n4]
            k += 1
    return nodeCoordinates, elementNodes


def drawingField(nodeCoordinates, elementNodes, ScalarField, Field_name = 'Field'):
    # Create a list of element coordinates and get strain[:, 2] values for coloring
    element_coords, element_colors = [], ScalarField
    for elem in elementNodes:
        element_coords.append([nodeCoordinates[node] for node in elem])

    # Create a PolyCollection object to plot
    polygons = collections.PolyCollection(element_coords, array=element_colors, cmap=plt.cm.viridis)

    # Set the axis limits based on the node coordinates
    plt.xlim(nodeCoordinates[:, 0].min(), nodeCoordinates[:, 0].max())
    plt.ylim(nodeCoordinates[:, 1].min(), nodeCoordinates[:, 1].max())

    # Add a colorbar
    plt.colorbar(polygons, label=f'{Field_name}')

    # Set the title and axis labels
    plt.title('Mesh Plot with Field')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.axis('equal')

    # Plot the polygons
    plt.gca().add_collection(polygons)

    # Show the plot
    # plt.show()



def drawingMesh(nodeCoordinates, elementNodes, lineSpec):
    # plt.figure()
    for nodes in elementNodes:
        connect = np.array([nodeCoordinates[nodes[0]],
                            nodeCoordinates[nodes[1]],
                            nodeCoordinates[nodes[2]],
                            nodeCoordinates[nodes[3]],
                            nodeCoordinates[nodes[0]]])

        plt.plot(connect[:, 0], connect[:, 1], lineSpec)

    plt.axis('equal')
    # plt.show()


def drawingMeshField(nodeCoordinates, elementNodes, field):
    '''
    :param nodeCoordinates: (n, 2), xy for one nodes
    :param elementNodes: (n, 4), four nodes for one element
    :param field: (n, ), scalar field value for each node
    :return: plot result
    '''
    # Normalize the field values
    norm = Normalize(vmin=field.min(), vmax=field.max())

    # Create a colormap and scalar mappable object for color mapping
    cmap = plt.cm.cool
    sm = ScalarMappable(norm=norm, cmap=cmap)

    for i, nodes in enumerate(elementNodes):
        connect = np.array([nodeCoordinates[nodes[0]],
                            nodeCoordinates[nodes[1]],
                            nodeCoordinates[nodes[2]],
                            nodeCoordinates[nodes[3]],
                            nodeCoordinates[nodes[0]]])

        # Compute the average field value for the current element
        avg_field_value = field[nodes].mean()

        # Get the corresponding color from the colormap
        color = sm.to_rgba(avg_field_value)

        # Plot the element boundaries
        plt.plot(connect[:, 0], connect[:, 1], color=color)

        # Fill the element area with the interpolated color
        plt.fill(connect[:, 0], connect[:, 1], color=color, alpha=0.7)

    plt.axis('equal')
    plt.colorbar(sm)
    # plt.show()


def gaussQuadrature(option):
    '''
    :param option: str, 'complete'(2x2), 'reduced'(1x1)
    :return: gauss weights and locations
    '''
    if option == 'complete':
        locations = np.array([
            [-0.577350269189626, -0.577350269189626],
            [0.577350269189626, -0.577350269189626],
            [0.577350269189626, 0.577350269189626],
            [-0.577350269189626, 0.577350269189626]
        ])
        weights = np.array([1, 1, 1, 1])
    elif option == 'reduced':
        locations = np.array([[0, 0]])
        weights = np.array([4])
    else:
        raise ValueError("Invalid option")
    return weights, locations


def Jacobian(node_coordinate, natural_derivatives):
    '''
    # Use gauss points
    :param node_coordinate: coordinates of the an element's nodes
    nodeCoordinates[indice, :]: (4, 2), four points, two dimensions
    [x1 y1]
    [x2 y2]
    [x3 y3]
    [x4 y4]
    :param natural_derivatives: (4, 2)
    [dN1/dxi dN1/deta]
    [dN2/dxi dN2/deta]
    [dN3/dxi dN3/deta]
    [dN4/dxi dN4/deta]
    :return Jacob: (2, 2)
    [dx/dxi dx/deta]
    [dy/dxi dy/deta]
    :return invJacob: (2, 2)
    [dx/dxi dx/deta]^(-1)
    [dy/dxi dy/deta]
    :return XYderivatives: (4, 2)
    [dN1/dx dN1/dy]
    [dN2/dx dN2/dy]
    [dN3/dx dN3/dy]
    [dN4/dx dN4/dy]
    '''

    Jacob = node_coordinate.T @ natural_derivatives
    '''Jacob: (2, 4) @ (4, 2) = (2, 2)
    [x1 x2 x3 x4]   *   [dN1/dxi dN1/deta] = [dx/dxi dx/deta]
    [y1 y2 y3 y4]       [dN2/dxi dN2/deta]   [dy/dxi dy/deta]
                        [dN3/dxi dN3/deta]
                        [dN4/dxi dN4/deta]
    '''
    invJacobian = np.linalg.inv(Jacob)
    '''invjacobian: (2, 2)
    ([dx/dxi dx/deta]) ^(-1)
    ([dy/dxi dy/deta])
    '''
    # print(invJacobian)
    # print(natural_derivatives)
    XYderivatives = natural_derivatives @ invJacobian
    '''XYderivatives: (4, 2) @ (2, 2) = (4, 2)
     [dN1/dx dN1/y]   *   ([dx/dxi dx/deta])   =  [dN1/dxi dN1/deta]
     [dN2/dx dN2/y]       ([dy/dxi dy/deta])      [dN2/dxi dN2/deta]
     [dN3/dx dN3/y]                               [dN3/dxi dN3/deta]
     [dN4/dx dN4/y]                               [dN4/dxi dN4/deta]
    '''
    return Jacob, invJacobian, XYderivatives


def formStiffness2D(GDof, numberElements, elementNodes, numberNodes, nodeCoordinates, C, rho, thickness):
    '''
    # Plane stress Matrix C: (3, 3)
    C = E / (1 - poisson ** 2) * [1,       poisson, 0                ]
                                 [poisson, 1,       0                ]
                                 [0,       0,       (1 - poisson) / 2]

    '''
    stiffness = np.zeros((GDof, GDof))
    mass = np.zeros((GDof, GDof))

    gaussWeights, gaussLocations = gaussQuadrature('complete')
    '''
    locations =
        [-0.577350269189626, -0.577350269189626]
        [0.577350269189626, -0.577350269189626]
        [0.577350269189626, 0.577350269189626]
        [-0.577350269189626, 0.577350269189626]
    weights = [1, 1, 1, 1]
    '''

    for e in range(numberElements):
        indice = elementNodes[e, :]
        # print('indice: ', indice)
        elementDof = np.hstack((indice, indice + numberNodes))
        # print('elementDof:', elementDof)
        ndof = len(indice)  # 4

        # Circle gauss points to sum
        for q in range(gaussWeights.shape[0]):
            GaussPoint = gaussLocations[q, :]
            xi = GaussPoint[0]
            eta = GaussPoint[1]

            shapeFunction, naturalDerivatives = shape_function_Q4(xi,
                                                                  eta)  # Get the shape function values and derivatives
            # print('shapeFunction:', shapeFunction)
            # print('naturalDerivatives:', naturalDerivatives)
            '''nodeCoordinates[indice, :]: (4, 2), four points, two dimensions
            [x1 y1]
            [x2 y2]
            [x3 y3]
            [x4 y4]

            naturalDerivatives: (4, 2)
            [dN1/dxi dN1/deta]
            [dN2/dxi dN2/deta]
            [dN3/dxi dN3/deta]
            [dN4/dxi dN4/deta]
            '''
            Jacob, invJacobian, XYderivatives = Jacobian(nodeCoordinates[indice, :], naturalDerivatives)
            '''
            nodeCoordinates[indice, :]: (4, 2), four points, two dimensions
            [x1 y1]
            [x2 y2]
            [x3 y3]
            [x4 y4]

            Jacob: (2, 2)
            [dx/dxi dx/deta]
            [dy/dxi dy/deta]

            invJacob: (2, 2)
            ([dx/dxi dx/deta]) ^ (-1)
            ([dy/dxi dy/deta])

            XYderivatives: (4, 2)
            [dN1/dx dN1/y]
            [dN2/dx dN2/y]
            [dN3/dx dN3/y]
            [dN4/dx dN4/y]
            '''
            # print('nodeCoordinates[indice, :]', nodeCoordinates[indice, :])
            # print('Jacob:', Jacob)
            # print('invjacobian:', invJacobian)
            # print('XYderivatives:', XYderivatives)

            B = np.zeros((3, 2 * ndof))
            B[0, :ndof] = XYderivatives[:, 0]
            B[1, ndof:2 * ndof] = XYderivatives[:, 1]
            B[2, :ndof] = XYderivatives[:, 1]
            B[2, ndof:2 * ndof] = XYderivatives[:, 0]
            '''
            B: (3, 2* ndof) = (3, 8)
            B = [dN1/dx dN2/dx dN3/dx dN4/dx 0      0      0      0     ]
                [0      0      0      0      dN1/dy dN2/dy dN3/dy dN4/dy]
                [dN1/dy dN2/dy dN3/dy dN4/dy dN1/dx dN2/dx dN3/dx dN4/dx]
            '''

            stiffness[np.ix_(elementDof, elementDof)] = stiffness[np.ix_(elementDof, elementDof)] + \
                                                        thickness * B.T @ C @ B * gaussWeights[q] * np.linalg.det(Jacob)

            mass[np.ix_(indice, indice)] = mass[np.ix_(indice, indice)] + \
                                           shapeFunction @ shapeFunction.T * \
                                           rho * thickness * gaussWeights[q] * np.linalg.det(Jacob)

            mass[np.ix_(indice + numberNodes, indice + numberNodes)] = mass[np.ix_(indice + numberNodes,
                                                                                   indice + numberNodes)] + \
                                                                       shapeFunction @ shapeFunction.T * \
                                                                       rho * thickness * gaussWeights[
                                                                           q] * np.linalg.det(Jacob)

    return stiffness, mass


def formStiffness2D_plastic(GDof, numberElements, elementNodes, numberNodes, nodeCoordinates, C, thickness):
    '''
    # Plane strength Matrix C: (num_nodes, 3, 3)
    C[n, :, :] = E / (1 - poisson ** 2) * [1,       poisson, 0                ]
                                             [poisson, 1,       0                ]
                                             [0,       0,       (1 - poisson) / 2]

    '''
    stiffness = np.zeros((GDof, GDof))

    gaussWeights, gaussLocations = gaussQuadrature('complete')
    '''
    locations =
        [-0.577350269189626, -0.577350269189626]
        [0.577350269189626, -0.577350269189626]
        [0.577350269189626, 0.577350269189626]
        [-0.577350269189626, 0.577350269189626]
    weights = [1, 1, 1, 1]
    '''

    for e in range(numberElements):
        indice = elementNodes[e, :]
        # print('indice: ', indice)
        elementDof = np.hstack((indice, indice + numberNodes))
        # print('elementDof:', elementDof)
        ndof = len(indice)  # 4

        # Circle gauss points to sum
        for q in range(gaussWeights.shape[0]):
            GaussPoint = gaussLocations[q, :]
            xi = GaussPoint[0]
            eta = GaussPoint[1]

            shapeFunction, naturalDerivatives = shape_function_Q4(xi,
                                                                  eta)  # Get the shape function values and derivatives
            # print('shapeFunction:', shapeFunction)
            # print('naturalDerivatives:', naturalDerivatives)
            '''nodeCoordinates[indice, :]: (4, 2), four points, two dimensions
            [x1 y1]
            [x2 y2]
            [x3 y3]
            [x4 y4]

            naturalDerivatives: (4, 2)
            [dN1/dxi dN1/deta]
            [dN2/dxi dN2/deta]
            [dN3/dxi dN3/deta]
            [dN4/dxi dN4/deta]
            '''
            Jacob, invJacobian, XYderivatives = Jacobian(nodeCoordinates[indice, :], naturalDerivatives)
            '''
            nodeCoordinates[indice, :]: (4, 2), four points, two dimensions
            [x1 y1]
            [x2 y2]
            [x3 y3]
            [x4 y4]

            Jacob: (2, 2)
            [dx/dxi dx/deta]
            [dy/dxi dy/deta]

            invJacob: (2, 2)
            ([dx/dxi dx/deta]) ^ (-1)
            ([dy/dxi dy/deta])

            XYderivatives: (4, 2)
            [dN1/dx dN1/y]
            [dN2/dx dN2/y]
            [dN3/dx dN3/y]
            [dN4/dx dN4/y]
            '''

            B = np.zeros((3, 2 * ndof))
            B[0, :ndof] = XYderivatives[:, 0]
            B[1, ndof:2 * ndof] = XYderivatives[:, 1]
            B[2, :ndof] = XYderivatives[:, 1]
            B[2, ndof:2 * ndof] = XYderivatives[:, 0]
            '''
            B: (3, 2* ndof) = (3, 8)
            B = [dN1/dx dN2/dx dN3/dx dN4/dx 0      0      0      0     ]
                [0      0      0      0      dN1/dy dN2/dy dN3/dy dN4/dy]
                [dN1/dy dN2/dy dN3/dy dN4/dy dN1/dx dN2/dx dN3/dx dN4/dx]
            '''

            stiffness[np.ix_(elementDof, elementDof)] = stiffness[np.ix_(elementDof, elementDof)] + \
                                                        thickness * B.T @ C[e] @ B * gaussWeights[q] * np.linalg.det(
                Jacob)

    return stiffness


def strainstress2D(elementNodes, nodeCoordinates, displacements_flatten, C, numberElements, numberNodes):
    '''
    :param elementNodes:
    :param nodeCoordinates:
    :param displacements_flatten:
    :param C:
    :param numberElements:
    :param numberNodes:
    :return:
    '''
    strain = np.zeros((numberElements, 4, 3))
    stress = np.zeros((numberElements, 4, 3))
    InternalForce = np.zeros(numberNodes * 2)
    weights, locations = gaussQuadrature('complete')
    # weights (4, ), locations (4, 2)
    for e in range(numberElements):
        indice = elementNodes[e, :]  # indice (4, )
        elementDof = np.hstack((indice, indice + numberNodes))  # elementDof (8, )

        for q in range(len(weights)):
            pt = locations[q, :]  # pt (2, )
            wt = weights[q]  # wt float
            xi, eta = pt  # xi, eta float

            shapeFunction, naturalDerivatives = shape_function_Q4(xi, eta)
            Jacob, invJacobian, XYderivatives = Jacobian(nodeCoordinates[indice, :], naturalDerivatives)

            B = np.zeros((3, 2 * len(indice)))  # B (3, 8)
            # B[0, :len(indice)] = naturalDerivatives[:, 0]
            # B[1, len(indice):] = naturalDerivatives[:, 1]
            # B[2, :len(indice)] = naturalDerivatives[:, 1]
            # B[2, len(indice):] = naturalDerivatives[:, 0]
            B[0, :len(indice)] = XYderivatives[:, 0]
            B[1, len(indice):] = XYderivatives[:, 1]
            B[2, :len(indice)] = XYderivatives[:, 1]
            B[2, len(indice):] = XYderivatives[:, 0]
            ''' B
            [dN1/dx dN2/dx dN3/dx dN4/dx 0      0      0      0      0      0      0      0     ]
            [0      0      0      0      dN1/dy dN2/dy dN3/dy dN4/dy 0      0      0      0     ]
            [dN1/dy dN2/dy dN3/dy dN4/dy 0      0      0      0      dN1/dx dN2/dx dN3/dx dN4/dx]
            '''
            strain[e, q, :] = B @ displacements_flatten[elementDof]
            '''
            strain (3, ) = B (3, 8) @ displacements[elementDof] (3, 1)
            get strain for a rectangle area
            '''
            stress[e, q, :] = C @ strain[e, q, :]  # (numberElements, 4, 3) 4 means 4 gauss points' result

            stress_vector = stress[e, q, :]
            int_force_contribution = B.T @ stress_vector * wt * np.linalg.det(Jacob)

            # Assemble the internal force contribution to the global internal force vector
            InternalForce[elementDof] += int_force_contribution

    stress = np.sum(stress, axis=1)
    strain = np.sum(strain, axis=1)
    '''
    stress_point = np.zeros((numberNodes, 3))
    for i in range(len(elementNodes)):
        stress_indices = elementNodes[i]
        for index in stress_indices:
            stress_point[index] = stress[i]

    strain_point = np.zeros((numberNodes, 3))
    for i in range(len(elementNodes)):
        strain_indices = elementNodes[i]
        for index in strain_indices:
            strain_point[index] = strain[i]
    '''
    '''
    stress_point is (n, 3)
    the three values for each node is: sigma_11, sigma_22, sigma_12
    '''
    return strain, stress, InternalForce


def displace2strain2D_gauss(elementNodes, nodeCoordinates, displacements_flatten, numberElements, numberNodes):
    '''
    :param elementNodes:
    :param nodeCoordinates:
    :param displacements_flatten:
    :param numberElements:
    :param numberNodes:
    :return:
    '''
    strain = np.zeros((numberElements, 4, 3))
    stress = np.zeros((numberElements, 4, 3))
    InternalForce = np.zeros(numberNodes * 2)
    weights, locations = gaussQuadrature('complete')
    # weights (4, ), locations (4, 2)
    for e in range(numberElements):
        indice = elementNodes[e, :]  # indice (4, )
        elementDof = np.hstack((indice, indice + numberNodes))  # elementDof (8, )

        for q in range(len(weights)):
            pt = locations[q, :]  # pt (2, )
            wt = weights[q]  # wt float
            xi, eta = pt  # xi, eta float

            shapeFunction, naturalDerivatives = shape_function_Q4(xi, eta)
            Jacob, invJacobian, XYderivatives = Jacobian(nodeCoordinates[indice, :], naturalDerivatives)

            B = np.zeros((3, 2 * len(indice)))  # B (3, 8)
            # B[0, :len(indice)] = naturalDerivatives[:, 0]
            # B[1, len(indice):] = naturalDerivatives[:, 1]
            # B[2, :len(indice)] = naturalDerivatives[:, 1]
            # B[2, len(indice):] = naturalDerivatives[:, 0]
            B[0, :len(indice)] = XYderivatives[:, 0]
            B[1, len(indice):] = XYderivatives[:, 1]
            B[2, :len(indice)] = XYderivatives[:, 1]
            B[2, len(indice):] = XYderivatives[:, 0]
            ''' B
            [dN1/dx dN2/dx dN3/dx dN4/dx 0      0      0      0      0      0      0      0     ]
            [0      0      0      0      dN1/dy dN2/dy dN3/dy dN4/dy 0      0      0      0     ]
            [dN1/dy dN2/dy dN3/dy dN4/dy 0      0      0      0      dN1/dx dN2/dx dN3/dx dN4/dx]
            '''
            strain[e, q, :] = B @ displacements_flatten[elementDof]

    return strain


def Compute_VonMises_2D(arr):
    '''
    :param arr: stress of shape (n, 3), the three values are sigma_11, sigma_22, sigma_12
    :return: von_mises_stress (n, ), the value is von mises stress
    Example usage:
    input_array = np.random.rand(231, 3)
    von_mises_stress = Compute_VonMises_2D(stress)

    print(von_mises_stress.shape)  # Output: (231,)
    '''
    # Extract the stress components from the input array
    sigma_11 = arr[:, 0]
    sigma_22 = arr[:, 1]
    sigma_12 = arr[:, 2]

    # Calculate the Von Mises stress
    von_mises_stress = np.sqrt((sigma_11 - sigma_22) ** 2 + 3 * sigma_12 ** 2)

    return von_mises_stress


# %% J2 Plasticity

def combHardJ2_3D(material_properties, C, deps, stressN, alphaN, epN):
    ''' J2 combined isotropic/kinematic hardening model
    :param material_properties: [lambda, mu, beta, H, Y0]: shape of (5, )
    :param D: elastic stiffness matrix: shape of (6, 6)
    :param deps: strain increment, shape of (6, )
    :param stressN: current stress state, shape of (6, )
    :param alphaN: back stress, shape of (6, )
    :param epN: plastic strain, float
    :return: stress, alpha, ep of next incremental step
    :note: use engineering strain
    '''

    # Material properties
    mu = material_properties['mu']
    beta = material_properties['beta']
    H = material_properties['H']
    Y0 = material_properties['Y0']

    # Compute trial stress
    stresstr = stressN + np.dot(C, deps)
    '''
    [sigmatr_11] = [sigma_11] + E/(1 + nu)/(1 - 2*nu) *
    [sigmatr_22]   [sigma_22]
    [sigmatr_33]   [sigma_33]
    [sigmatr_12]   [sigma_12]
    [sigmatr_23]   [sigma_23]
    [sigmatr_13]   [sigma_13]

    [1-nu nu   nu   0     0     0    ] * [epsilon_11]
    [nu   1-nu nu   0     0     0    ]   [epsilon_22]
    [nu   nu   1-nu 0     0     0    ]   [epsilon_33]
    [0    0    0    1-2nu 0     0    ]   [epsilon_12]
    [0    0    0    0     1-2nu 0    ]   [epsilon_23]
    [0    0    0    0     0     1-2nu]   [epsilon_13]
    '''

    # Trace of trial stress
    I1 = np.sum(stresstr[:3])

    # Compute deviatoric stress
    Iden = np.array([1, 1, 1, 0, 0, 0])
    str = stresstr - I1 * Iden / 3

    # Compute shifted stress
    eta = str - alphaN

    # Norm of eta
    etat = np.sqrt(np.sum(eta[:3] ** 2) + 2 * np.sum(eta[3:] ** 2))
    J2 = 1 / 2 * (np.sum(eta[:3] ** 2) + 2 * np.sum(eta[3:] ** 2))
    vonmises = np.sqrt(3 * J2)
    '''
    S_ij:S_ij = 2J_2
    von mises = sqrt (3J_2) < Y0
    '''
    # Yield function
    # YieldCondition = vonmises - (Y0 + (1 - beta) * H * epN)
    YieldCondition = etat - np.sqrt(2 / 3) * (Y0 + (1 - beta) * H * epN)

    # print('etat:', etat)
    # print('J2:', J2)
    # print('vonmises:', vonmises)
    # print('YieldCondition:', YieldCondition)
    # Check for yield status
    if YieldCondition < 0:
        # Trial states are final
        stress = stresstr
        alpha = alphaN
        ep = epN
        C_consistent = C
        C_continuum = C
    else:
        # Plastic consistency parameter
        # gamma = np.sqrt(2 / 3) * YieldCondition / (2 * mu + 2 / 3 * H)
        gamma = YieldCondition / (2 * mu + 2 / 3 * H)
        # Updated plastic strain
        ep = epN + gamma * np.sqrt(2 / 3)

        # Unit vector normal to yield surface
        N = eta / etat  # (6, )

        # Updated stress
        stress = stresstr - 2 * mu * gamma * N

        # Updated back stress
        alpha = alphaN + 2 / 3 * beta * H * gamma * N

        var1 = 4 * mu * mu / (2 * mu + 2 / 3 * H)
        var2 = 4 * mu * mu * gamma / etat

        C_consistent = C - (var1 - var2) * np.outer(N, N) + var2 * np.outer(Iden, Iden) / 3
        C_continuum = C - (var1 - var2) * np.outer(N, N)
        C_consistent[0, 0] -= var2
    return stress, alpha, ep, C_consistent, C_continuum


def CombHardJ2_PlaneStrain(material_properties, C, deps, stressN, alphaN, epN):
    ''' J2 combined isotropic/kinematic hardening model
    :param material_properties: [lambda, mu, beta, H, Y0]: shape of (5, )
    :param D: elastic stiffness matrix: shape of (3, 3)
    :param deps: strain increment, shape of (3, )
    :param stressN: current stress state, shape of (3, )
    :param alphaN: back stress, shape of (3, )
    :param epN: plastic strain, float
    :return: stress, alpha, ep of next incremental step
    :note: use engineering strain
    '''
    # plane strain for engineering strain
    assert deps.shape == (3,), 'strain increment deps should be a numpy array of size (3,)'
    assert stressN.shape == (3,), 'current stress state stressN should be a numpy array of size (3,)'
    assert alphaN.shape == (3,), 'back stress alphaN should be a numpy array of size (3,)'
    # assert type(epN) == float, 'plastic strain epN should be a float'
    '''
    Use with engineering strain
    C = D = E/(1 - nu^2) *
    [1  nu 0       ]
    [nu 1  0       ]
    [0  0  (1-nu)/2]
    '''

    # Material properties
    mu = material_properties['mu']
    beta = material_properties['beta']
    H = material_properties['H']
    Y0 = material_properties['Y0']

    # Compute trial stress
    stresstr = stressN + np.dot(C, deps)
    '''
    [sigmatr_11] = [sigma_11] + E/(1 + nu)/(1 - 2*nu) *
    [sigmatr_22]   [sigma_22]
    [sigmatr_12]   [sigma_12]

    [1-nu nu   nu    ] * [epsilon_11]
    [nu   1-nu nu    ]   [epsilon_22]
    [0    0    1-2nu ]   [epsilon_12]
    '''
    # Trace of trial stress
    I1 = np.sum(stresstr[:2])

    # Compute deviatoric stress
    Iden = np.array([1, 1, 0])
    str = stresstr - I1 * Iden / 3

    # Compute shifted stress
    eta = str - alphaN

    # Norm of eta
    etat = np.sqrt(np.sum(eta[:2] ** 2) + 2 * eta[2] ** 2)
    J2 = 1 / 2 * (np.sum(eta[:2] ** 2) + 2 * eta[2] ** 2)
    vonmises = np.sqrt(3 * J2)
    '''
    S_ij:S_ij = 2J_2
    von mises = sqrt (3J_2) < Y0
    '''
    # Yield function
    # YieldCondition = vonmises - (Y0 + (1 - beta) * H * epN) # Alternative
    YieldCondition = etat - np.sqrt(2 / 3) * (Y0 + (1 - beta) * H * epN)
    # print('etat:', etat)
    # print('J2:', J2)
    # print('vonmises:', vonmises)
    # print('YieldCondition:', YieldCondition)
    # Check for yield status
    if YieldCondition < 0:
        # Trial states are final
        stress = stresstr
        alpha = alphaN
        ep = epN
        C_consistent = C
        C_continuum = C
    else:
        # Plastic consistency parameter
        # gamma = np.sqrt(2 / 3) * YieldCondition / (2 * mu + 2 / 3 * H)
        gamma = YieldCondition / (2 * mu + 2 / 3 * H)
        # Updated plastic strain
        ep = epN + gamma * np.sqrt(2 / 3)

        # Unit vector normal to yield surface
        N = eta / etat

        # Updated stress
        stress = stresstr - 2 * mu * gamma * N

        # Updated back stress
        alpha = alphaN + 2 / 3 * beta * H * gamma * N
        var1 = 4 * mu * mu / (2 * mu + 2 / 3 * H)
        var2 = 4 * mu * mu * gamma / etat

        C_consistent = C - (var1 - var2) * np.outer(N, N) + var2 * np.outer(Iden, Iden) / 3
        C_continuum = C - (var1 - var2) * np.outer(N, N)
        C_consistent[0, 0] -= var2
    return stress, alpha, ep, C_consistent, C_continuum


# %% Elastic Dynamics

def Newmark(mass, stiffness, force, dt, gamma=1 / 2, beta=1 / 4):
    '''Newmark method in FEM
    :mass: (n, n)
    :stiffness: (n, n)
    :force: (n, Ntime)
    :dt: int
    :gamma, beta: Newmark parameters, 1/2, 1/4 are central differential method
    :return: Displacements without prescipt dof
    '''
    M = mass
    K = stiffness
    F = force
    Nmax = F.shape[1]

    # Compute Masshead for once
    Mhead = M + beta * dt * dt * K

    # Preallocate arrays
    Q = np.zeros((len(K), Nmax + 1))
    dQ = np.zeros((len(K), Nmax + 1))
    ddQ = np.zeros((len(K), Nmax + 1))
    ddQ[:, 0] = np.linalg.inv(M) @ (F[:, 0] - K @ Q[:, 0])
    print('Q shape at preallocate:', Q.shape)
    for n in range(Nmax):
        # Predict dQpr(n+1), Qpr(n+1)
        dQpr_n1 = dQ[:, n] + (1 - gamma) * dt * ddQ[:, n]
        Qpr_n1 = Q[:, n] + dt * dQ[:, n] + (1 / 2 - beta) * dt * dt * ddQ[:, n]

        # Compute F head
        Fhead_n1 = F[:, n] - K @ Qpr_n1

        # Linear solver
        ddQ_n1 = np.linalg.solve(Mhead, Fhead_n1)

        # Correct vector and store
        dQ[:, n + 1] = dQpr_n1 + gamma * dt * ddQ_n1
        Q[:, n + 1] = Qpr_n1 + beta * dt * dt * ddQ_n1
        ddQ[:, n + 1] = ddQ_n1
    print('Q shape after Newmark:', Q.shape)
    return Q[:, 1:]


def Newmark_with_prescribed_dof(mass, stiffness, force, dt, GDof, prescribed_dof, gamma=1 / 2, beta=1 / 4):
    activeDof = np.setdiff1d(np.arange(0, GDof), prescribed_dof)
    M = mass[activeDof, :][:, activeDof]
    K = stiffness[activeDof, :][:, activeDof]
    F = force[activeDof, :]
    Nmax = F.shape[1]
    Q = Newmark(mass=M, stiffness=K, force=F, dt=dt, gamma=1 / 2, beta=1 / 4)
    print('Q shape:', Q.shape)
    pre_rows = np.array([0] * Q.shape[1])
    for row in prescribed_dof:
        Q = np.insert(Q, row, pre_rows, axis=0)
    print('Q shape:', Q.shape)
    return Q


# %% Plastic Dynamic, Under developing

def Newmark_J2(mass, stiffness, force, dt, material_properties, nodeCoordinates, elementNodes, numberElements,
               numberNodes, C, gamma=1 / 2, beta=1 / 4):
    '''Newmark method in FEM
    :mass: (n, n)
    :stiffness: (n, n)
    :force: (n, Ntime)
    :dt: int
    :gamma, beta: Newmark parameters, 1/2, 1/4 are central differential method
    :return: Displacements without prescipt dof
    '''
    M = mass
    K = stiffness
    F = force
    Nmax = F.shape[1]

    # First d is iteration, second is node number, last is direction 11, 22, 12
    strain = np.zeros((Nmax + 1, numberNodes, 3))
    stress = np.zeros((Nmax + 1, numberNodes, 3))
    alpha = np.zeros((Nmax + 1, numberNodes, 3))
    ep = np.zeros((Nmax + 1, numberNodes, 3))
    # Compute Masshead for once
    Mhead = M + beta * dt * dt * K

    # Preallocate arrays
    Q = np.zeros((len(K), Nmax + 1))
    dQ = np.zeros((len(K), Nmax + 1))
    ddQ = np.zeros((len(K), Nmax + 1))
    ddQ[:, 0] = np.linalg.inv(M) @ (F[:, 0] - K @ Q[:, 0])

    for n in range(Nmax):
        # Predict dQpr(n+1), Qpr(n+1)
        dQpr_n1 = dQ[:, n] + (1 - gamma) * dt * ddQ[:, n]
        Qpr_n1 = Q[:, n] + dt * dQ[:, n] + (1 / 2 - beta) * dt * dt * ddQ[:, n]

        # Compute F head
        # This is actually trail
        Fhead_n1 = F[:, n] - K @ Qpr_n1

        # Linear solver
        ddQ_n1 = np.linalg.solve(Mhead, Fhead_n1)

        # Correct vector and store
        dQ[:, n + 1] = dQpr_n1 + gamma * dt * ddQ_n1
        Q[:, n + 1] = Qpr_n1 + beta * dt * dt * ddQ_n1
        ddQ[:, n + 1] = ddQ_n1

        # Strain computed from Q
        strain[n + 1, :, :] = \
            strainstress2D(elementNodes=elementNodes, nodeCoordinates=nodeCoordinates,
                           displacements_flatten=Q[:, n + 1],
                           C=C, numberElements=numberElements, numberNodes=numberNodes)[0]
        # Strain incremental
        dstrain = strain[n + 1, :, :] - strain[n, :, :]  # shape of (numberNodes, 3)
        # print(dstrain)
        # Compute stress by current state and incremental strain

        for m in range(numberNodes):
            epN = np.sqrt(ep[n + 1, m, 0] ** 2 + ep[n + 1, m, 1] + 2 * ep[n + 1, m, 2])
            epN = float(epN)
            stress[n + 1, m, :], alpha[n + 1, m, :], ep[n + 1, m, :] = CombHardJ2_PlaneStrain(
                material_properties=material_properties, C=C, deps=dstrain[m, :], stressN=stress[n + 1, m, :],
                alphaN=alpha[n + 1, m, :], epN=epN)

    ##
    return Q[:, 1:]


def Newmark_J2_with_prescribed_dof(mass, stiffness, force, dt, GDof, prescribed_dof, material_properties,
                                   nodeCoordinates, elementNodes, numberElements, numberNodes, C, gamma=1 / 2,
                                   beta=1 / 4):
    activeDof = np.setdiff1d(np.arange(0, GDof), prescribed_dof)
    M = mass[activeDof, :][:, activeDof]
    K = stiffness[activeDof, :][:, activeDof]
    F = force[activeDof, :]
    Nmax = F.shape[1]
    Q = Newmark_J2(mass=M,
                   stiffness=K,
                   force=F,
                   dt=dt,
                   material_properties=material_properties,
                   nodeCoordinates=nodeCoordinates,
                   elementNodes=elementNodes,
                   numberElements=numberElements,
                   numberNodes=numberNodes,
                   C=C, gamma=gamma, beta=beta)
    print('Q shape:', Q.shape)
    pre_rows = np.array([0] * Q.shape[1])
    for row in prescribed_dof:
        Q = np.insert(Q, row, pre_rows, axis=0)
    print('Q shape:', Q.shape)
    return Q
