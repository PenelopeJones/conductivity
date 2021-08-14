import numpy as np
from scipy.stats import norm, invwishart, multivariate_normal

import pdb

def compute_overlap(r, lower_r_bound, bin_size, ion_size):

    # intermediate calculations
    p1 = norm.cdf(lower_r_bound + bin_size, loc=r, scale=ion_size) - \
         norm.cdf(lower_r_bound, loc=r, scale=ion_size)

    p2 = norm.cdf(lower_r_bound + bin_size, loc=-r, scale=ion_size) - \
         norm.cdf(lower_r_bound, loc=-r, scale=ion_size)

    a = (ion_size / r) * (0.5 / np.pi) ** 0.5

    q1 = np.exp(-0.5*(lower_r_bound + bin_size - r)**2 / ion_size**2)
    q2 = np.exp(-0.5*(lower_r_bound - r)**2 / ion_size**2)
    q3 = np.exp(-0.5*(lower_r_bound + bin_size + r)**2 / ion_size**2)
    q4 = np.exp(-0.5*(lower_r_bound + r)**2 / ion_size**2)

    tot = p1 + p2 + a*(q2 - q1 + q3 - q4)

    return tot


def tabulate_overlaps(min_r_value, max_r_value, bin_size, ion_size, resolution):
    number_of_bins = int((max_r_value - min_r_value) / bin_size)
    lower_r_bounds = np.arange(min_r_value, max_r_value, bin_size)
    r_values = np.arange(ion_size, max_r_value + 4*ion_size, resolution)
    tabulated = np.zeros((lower_r_bounds.shape[0], r_values.shape[0]))


    for i, lower_r_bound in enumerate(lower_r_bounds):
        tabulated[i, :] = compute_overlap(r_values, lower_r_bound, bin_size, ion_size)

    return tabulated


def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

# Compute inter-ion distance given the particle positions
def radial_distance(x_i, y_i, z_i, x_j, y_j, z_j, box_length):
    """
    Calculates the effective distance between two particles using periodic BCs and the
    minimum image convention.

    :param x_i: x_coordinate of ion i (float).
    :param y_i: y_coordinate of ion i (float).
    :param z_i: z_coordinate of ion i (float).
    :param x_j: x_coordinate of ion j (float).
    :param y_j: y_coordinate of ion j (float).
    :param z_j: z_coordinate of ion j (float).
    :param box_length: box size (float).
    :return: distance between the two points (float).
    """
    delta_x = min(((x_i - x_j) % box_length), ((x_j - x_i) % box_length))
    delta_y = min(((y_i - y_j) % box_length), ((y_j - y_i) % box_length))
    delta_z = min(((z_i - z_j) % box_length), ((z_j - z_i) % box_length))
    return np.sqrt(delta_x ** 2 + delta_y ** 2 + delta_z ** 2)

# Compute g(r)
def rdf(r, prefactor, bin_size, ion_size, min_r_value, max_r_value, smoothed=False):
    """
    For a given array of central ion A - ion B distances this calculates
    the A-B (smoothed or not smoothed) radial distribution function g(r) as an array.
    :param r: Vector of distances of ions of type A from the central ion of type B.
    :param prefactor: Scalar value equal to the reciprocal of average number density
                      in ideal gas with the same overall density (float).
    :param bin_size: Histogram bin size (float).
    :param ion_size: Size of each ion, used to calculate the smoothed RDF (float).
    :param min_r_value: The minimum x value to be considered in the histogram (float).
    :param max_r_value: The maximum x value to be considered in the histogram (float).
    :param smoothed: If true, the smoothed RDF is calculated; if false, the standard RDF
                     is calculated.
    :return:
    """
    number_of_bins = int((max_r_value - min_r_value) / bin_size)
    gs = []
    lower_r_bound = min_r_value
    upper_r_bound = min_r_value + bin_size
    for i in range(0, number_of_bins):
        r_mean = 0.5 * (lower_r_bound + upper_r_bound)
        V_shell = 4 * np.pi * r_mean ** 2 * bin_size

        if smoothed:
            x = norm.cdf(upper_r_bound, loc=r, scale=ion_size) - \
                norm.cdf(lower_r_bound, loc=r, scale=ion_size)
            number_in_bin = np.sum(x)
        else:
            number_in_bin = ((lower_r_bound < r) & (r < upper_r_bound)).sum()

        g = prefactor * number_in_bin / V_shell
        gs.append(g)
        lower_r_bound = upper_r_bound
        upper_r_bound = upper_r_bound + bin_size
    return gs

# Compute g(r) and vacf(r)
def vrdf(r, speeds, directions, prefactor, bin_size, ion_size, min_r_value, max_r_value, smoothed=False):
    """
    For a given array of central ion A - ion B distances this calculates
    the A-B (smoothed or not smoothed) radial distribution function g(r) as an array.
    :param r: Vector of distances of ions of type A from the central ion of type B.
    :param prefactor: Scalar value equal to the reciprocal of average number density
                      in ideal gas with the same overall density (float).
    :param bin_size: Histogram bin size (float).
    :param ion_size: Size of each ion, used to calculate the smoothed RDF (float).
    :param min_r_value: The minimum x value to be considered in the histogram (float).
    :param max_r_value: The maximum x value to be considered in the histogram (float).
    :param smoothed: If true, the smoothed RDF is calculated; if false, the standard RDF
                     is calculated.
    :return:
    """
    number_of_bins = int((max_r_value - min_r_value) / bin_size)
    gs = []
    vs = []
    thetas = []
    lower_r_bound = min_r_value
    upper_r_bound = min_r_value + bin_size
    for i in range(0, number_of_bins):
        r_mean = 0.5 * (lower_r_bound + upper_r_bound)
        V_shell = 4 * np.pi * r_mean ** 2 * bin_size

        idx = np.where(abs(r - lower_r_bound - 0.5*bin_size) <= 0.5*bin_size)

        if smoothed:
            x = compute_overlap(r, lower_r_bound, bin_size, ion_size)
        else:
            x = ((lower_r_bound < r) & (r < upper_r_bound))
        number_in_bin = np.sum(x)
        v_bin = np.dot(x, speeds) / number_in_bin
        theta_bin = np.dot(x, directions) / number_in_bin
        g_bin = number_in_bin * prefactor / V_shell
        vs.append(v_bin)
        thetas.append(theta_bin)
        gs.append(g_bin)

        lower_r_bound = upper_r_bound
        upper_r_bound = upper_r_bound + bin_size
    return gs, vs, thetas

def dynamic_feature_vector(typeAx, typeBx, typeAv, typeBv, typeA_id, box_length, bin_size=0.5, ion_size=0.5,
                          min_r_value=0.5, max_r_value=5.0, smoothed=True):

    prefactor_aa = box_length ** 3 / (typeAx.shape[0] - 1)
    prefactor_ab = box_length ** 3 / (typeBx.shape[0])

    # Step 1: Compute typeA-typeA feature vector
    distances_aa = []
    speeds_aa = []
    directions_aa = []
    for j in range(typeAx.shape[0]):
        if j == typeA_id:
            continue
        r = radial_distance(typeAx[typeA_id, 0], typeAx[typeA_id, 1], typeAx[typeA_id, 2],
                            typeAx[j, 0], typeAx[j, 1], typeAx[j, 2], box_length)

        rel_speed = np.linalg.norm(typeAv[typeA_id, :] - typeAv[j, :])
        rel_direction = angle_between(typeAv[typeA_id, :], typeAv[j, :])

        distances_aa.append(r)
        speeds_aa.append(rel_speed)
        directions_aa.append(rel_direction)
    distances_aa = np.asarray(distances_aa)
    speeds_aa = np.asarray(speeds_aa)
    directions_aa = np.asarray(directions_aa)
    x_aa, v_aa, theta_aa = vrdf(distances_aa, speeds_aa, directions_aa, prefactor_aa, bin_size, ion_size, min_r_value, max_r_value, smoothed)
    # Step 2: Compute typeA-typeB feature vector
    distances_ab = []
    speeds_ab = []
    directions_ab = []
    for j in range(typeBx.shape[0]):
        r = radial_distance(typeAx[typeA_id, 0], typeAx[typeA_id, 1], typeAx[typeA_id, 2],
                            typeBx[j, 0], typeBx[j, 1], typeBx[j, 2], box_length)
        rel_speed = np.linalg.norm(typeAv[typeA_id, :] - typeBv[j, :])
        rel_direction = angle_between(typeAv[typeA_id, :], typeBv[j, :])
        distances_ab.append(r)
        speeds_ab.append(rel_speed)
        directions_ab.append(rel_direction)
    distances_ab = np.asarray(distances_ab)
    speeds_ab = np.asarray(speeds_ab)
    directions_ab = np.asarray(directions_ab)
    x_ab, v_ab, theta_ab = vrdf(distances_ab, speeds_ab, directions_ab, prefactor_ab, bin_size, ion_size, min_r_value, max_r_value, smoothed)

    return np.hstack((x_aa, x_ab, v_aa, v_ab, theta_aa, theta_ab))



def static_feature_vector(typeA, typeB, typeC, typeA_id, box_length, bin_size=0.5, ion_size=0.5,
                          min_r_value=0.5, max_r_value=5.0, smoothed=True):
    """
    typeA: np array of positions of type A ions [nA, 3]
    typeB: np array of positions of type B ions [nB, 3]
    typeA_id: int giving the id of the type A ion whose feature vector we are computing
    """
    prefactor_aa = box_length ** 3 / (typeA.shape[0] - 1)
    prefactor_ab = box_length ** 3 / (typeB.shape[0])

    # Step 1: Compute typeA-typeA feature vector
    distances_aa = []
    for j in range(typeA.shape[0]):
        if j == typeA_id:
            continue
        r = radial_distance(typeA[typeA_id, 0], typeA[typeA_id, 1], typeA[typeA_id, 2],
                            typeA[j, 0], typeA[j, 1], typeA[j, 2], box_length)
        distances_aa.append(r)
    distances_aa = np.asarray(distances_aa)
    x_aa = rdf(distances_aa, prefactor_aa, bin_size, ion_size, min_r_value, max_r_value, smoothed)

    # Step 2: Compute typeA-typeB feature vector
    distances_ab = []
    for j in range(typeB.shape[0]):
        r = radial_distance(typeA[typeA_id, 0], typeA[typeA_id, 1], typeA[typeA_id, 2],
                            typeB[j, 0], typeB[j, 1], typeB[j, 2], box_length)
        distances_ab.append(r)
    distances_ab = np.asarray(distances_ab)
    x_ab = rdf(distances_ab, prefactor_ab, bin_size, ion_size, min_r_value, max_r_value, smoothed)

    if typeC is None:
        return np.concatenate((x_aa, x_ab))

    else:
        # Step 3: Compute typeA-typeC feature vector
        prefactor_ac = box_length ** 3 / (typeC.shape[0])
        distances_ac = []
        for j in range(typeC.shape[0]):
            r = radial_distance(typeA[typeA_id, 0], typeA[typeA_id, 1], typeA[typeA_id, 2],
                                typeC[j, 0], typeC[j, 1], typeC[j, 2], box_length)
            distances_ac.append(r)
        distances_ac = np.asarray(distances_ac)
        x_ac = rdf(distances_ac, prefactor_ac, bin_size, ion_size, min_r_value, max_r_value, smoothed)

        return np.concatenate((np.concatenate((x_aa, x_ab)), x_ac))
