# This file is part of MX2CHE
# See LICENCE.txt for details

import numpy as np
import re


def encodeRGB(x):
    r, g, b = list(map(lambda x: max(0, min(int(x), 255)), re.split('[^\d]+', x)[1: 4]))
    return "#{0:02x}{1:02x}{2:02x}".format(r, g, b)
    

def formation_energy_per_atom(compound, neat_elements):
    fe = compound['relaxed_energy']
    formula = compound['formula']
    for element in formula.keys():
        element_energy = neat_elements.loc[element]['relaxed_energy_pa'].min()
        fe -= formula[element] * element_energy
    return fe / compound['natoms']


def element_content(element, data_point, relative=True):
    formula = data_point['formula']
    natoms = data_point['natoms']
    if element in formula.keys():
        return formula[element] / (natoms if relative else 1)
    return 0

    
def formula2string(ordered_dict_formula, multiply=1):
    out_str=''
    ks = [k for k in ordered_dict_formula.keys() if (k != 'Np' and k != 'U')]
    for k in ks:
        out_str += f'{k}{ordered_dict_formula[k] * multiply}'
    return out_str

    
def simplexBasis(n):  # n == number of unique elements - 1 
    basis = np.vstack((np.eye(n), (1 + np.sqrt(n + 1)) / n * np.ones((1,n))))
    basis -= np.mean(basis, axis=0)
    return basis / np.sqrt(2)


def compositionTranform(composition, simplex_basis):
    composition = composition / np.sum(composition)  # norm composition to the element fractions
    return np.sum(composition.reshape(-1, 1) * simplex_basis, axis=0) 


def getEhull(point, hull, return_simplice=False):
    equations = hull.equations  
    simplices = hull.simplices
    # -2 index is for form.energies, -1 index contains biases, 
    mask = list(map(lambda x: all([hull.points[point_index][-1] < 1e-05 for point_index in x]), hull.simplices))
    ## the mask avoid consideration of simplices containing positive form.energy vertiсes
    equations = equations[mask] 
    simplices = simplices[mask]
    mask = (np.abs(equations[:,-2]) > 1e-05) & (np.abs(1 - equations[:, -2]) > 1e-05)
    ## all facets parallel to the energy axis or having zero slope are removed by the mask
    equations = equations[mask] 
    simplices = simplices[mask]
    planes = equations.T
    energy_coord = -(np.dot(planes[:-2].T, point) + planes[-1]) / planes[-2]
    ## among all possible intersection the correct one should possess the maximum energy
    maxen_index = np.argmax(energy_coord)
    if return_simplice:
        return energy_coord[maxen_index], simplices[maxen_index]
    return energy_coord[maxen_index]