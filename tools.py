import numpy as np
import re
import pandas as pd
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error as root_mse
from pymatgen.core import Element, Structure
from pymatgen.io.vasp.outputs import Vasprun
from pymatgen.transformations.site_transformations import TranslateSitesTransformation
from tqdm.notebook import tqdm
from collections import OrderedDict, Counter
from scipy.constants import Boltzmann, elementary_charge


elements = ['W', 'Mo', 'U', 'Se', 'S', 'Np']
unique_elements = ['W', 'Mo', 'Se', 'S']
dict_elements = OrderedDict({'Np': 0, 'S': 1, 'Se': 2})
kB = Boltzmann / elementary_charge
T_ = np.linspace(0, 2000, 41)


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
    out_str = ''
    ks = [k for k in ordered_dict_formula.keys() if (k != 'Np' and k != 'U')]
    for k in ks:
        out_str += f'{k}{ordered_dict_formula[k] * multiply}'
    return out_str

    
def simplexBasis(n):  # n == number of unique elements - 1 
    basis = np.vstack((np.eye(n), (1 + np.sqrt(n + 1)) / n * np.ones((1, n))))
    basis -= np.mean(basis, axis=0)
    return basis / np.sqrt(2)


def compositionTranform(composition, simplex_basis):
    composition = composition / np.sum(composition)  # norm composition to the element fractions
    return np.sum(composition.reshape(-1, 1) * simplex_basis, axis=0) 


def getEhull(point, hull, return_simplice=False):
    equations = hull.equations  
    simplices = hull.simplices
    # -2 index is for form.energies, -1 index contains biases
    mask = list(map(lambda x: all([hull.points[point_index][-1] < 1e-05 for point_index in x]), hull.simplices))
    # the mask avoid consideration of simplices containing positive form.energy vertiсes
    equations = equations[mask] 
    simplices = simplices[mask]
    mask = (np.abs(equations[:, -2]) > 1e-05) & (np.abs(1 - equations[:, -2]) > 1e-05)
    # all facets parallel to the energy axis or having zero slope are removed by the mask
    equations = equations[mask] 
    simplices = simplices[mask]
    planes = equations.T
    energy_coord = -(np.dot(planes[:-2].T, point) + planes[-1]) / planes[-2]
    # among all possible intersection the correct one should possess the maximum energy
    maxen_index = np.argmax(energy_coord)
    if return_simplice:
        return energy_coord[maxen_index], simplices[maxen_index]
    return energy_coord[maxen_index]


def orderedDictFormula(structure, from_cif_mode=False):
    if from_cif_mode:
        structure = Structure.from_str(input_string=structure, fmt='cif')
    atomic_symbols = np.array(list(map(lambda x: x.symbol, structure.species)))
    return OrderedDict(sorted(Counter(atomic_symbols).items()))


def getStructuresAsList(workdir):
    xml_list = [xml for xml in os.listdir(workdir) if xml.endswith('xml')]
    pkl_file_name = re.sub('3res_structures/', '3res_list.pkl', workdir)
    print('\ngetStructuresAsList out :', pkl_file_name)
    if os.path.exists(pkl_file_name):
        with open(pkl_file_name, 'rb') as f:
            result = pickle.load(f)
        if len(xml_list) == len(result):
            print(f'result file for {workdir} has been read from pkl')
            return result
    print(f'result file for {workdir} is being updated')
    result = []
    for xml_file in tqdm(xml_list):
        tag = re.sub('vasprun_', '', re.sub('.xml', '', xml_file))
        try:
            run = Vasprun(f'{workdir}/{xml_file}', parse_potcar_file=False)
        except:
            print(f'vasprun.xml is not OK for {tag}')
        if not run.converged:
            print(f'vasprun.xml IS NOT CONVERGED for {tag}')

        initial_structure = run.initial_structure
        final_structure = run.final_structure
        last_step_forces = np.array(run.ionic_steps[-1]['forces'])
        last_step_pressure = np.trace(run.ionic_steps[-1]['stress']) / 3
        energy = run.final_energy
        atomic_symbols = run.atomic_symbols
        atomic_numbers = np.array(list(map(lambda x: Element(x).number, atomic_symbols)))
        formula = OrderedDict(sorted(Counter(atomic_symbols).items()))
        natoms = len(atomic_symbols)
        nelements = len(formula)
        result += [[tag, formula, natoms, atomic_symbols, atomic_numbers, nelements,
                    initial_structure,  # initial_cell, initial_structure,
                    final_structure,  # final_cell, final_structure,
                    last_step_pressure, last_step_forces, energy]]
    with open(pkl_file_name, 'wb') as f:
        pickle.dump(result, f)
    return result


def readVASPresults(data_group, data_subgroup='hs', return_neat_elements=False):
    RESULT = []
    for workdir in [f'dft/neat_elements/3res_structures/',
                    f'dft/{data_group}_{data_subgroup}/3res_structures/']:
        result = getStructuresAsList(workdir)
        RESULT += result

    RESULT = pd.DataFrame(RESULT,
                          columns=['compound', 'formula', 'natoms', 'atomic_symbols', 'atomic_numbers', 'nelements',
                                   'initial_structure',  # 'cell', 'pos',
                                   'relaxed_structure',  # 'relaxed_cell', 'relaxed_pos',
                                   'relaxed_pressure', 'relaxed_forces', 'relaxed_energy']).sort_values(
        'nelements').reset_index(drop=True)
    RESULT['relaxed_energy_pa'] = RESULT['relaxed_energy'] / RESULT['natoms']
    neat_elements = RESULT[RESULT['nelements'] == 1].copy()
    neat_elements['element'] = neat_elements['formula'].apply(lambda x: list(x.keys())[0])
    neat_elements.set_index('element', inplace=True)
    RESULT = RESULT[RESULT['nelements'] > 1].copy()
    try:
        RESULT['index'] = RESULT['compound'].apply(lambda x: int(re.sub('.vasp', '', x)))
    except:
        RESULT['index'] = RESULT['compound'].apply(lambda x: re.sub('.vasp', '', x))
    RESULT.set_index('index', inplace=True, drop=True)
    RESULT['formation_energy_per_atom'] = [formation_energy_per_atom(RESULT.loc[i], neat_elements) for i in
                                           RESULT.index]

    print('datapoints of neat elements:', neat_elements.shape)
    print('datapoints w/o neat elements:', RESULT.shape)
    if return_neat_elements:
        return RESULT, neat_elements
    return RESULT


def DefineItemPart(item_index, train_index, val_index, test_index):
    if item_index in train_index:
        return 'train'
    elif item_index in val_index:
        return 'val'
    elif item_index in test_index:
        return 'test'
    return ''


def DefineParts(result, space_sym_limit=8, random_state=0):
    inWhichPart = np.array(['train/val' if result.loc[i]['Space_group_no'] > space_sym_limit else 'test'
                            for i in result.index])
    strat = result[inWhichPart == 'train/val']['Space_group_no'].values
    train, val = train_test_split(result[inWhichPart == 'train/val'],
                                  test_size=0.1,
                                  random_state=random_state,
                                  stratify=strat)
    test = result[inWhichPart == 'test']
    inWhichPart = [DefineItemPart(i, train.index, val.index, test.index) for i in result.index]
    return inWhichPart


def spaceGroupConventional(sg):
    sg = re.sub('-[\d]', lambda x: '\\bar{' + x.group()[1:] + '}', sg)
    return f'${sg}$'


def getDefectNumber(defect, def_type='substitution', at_from='S', at_to='Se'):
    defect_nsites = len(defect)
    subst = 0
    if def_type == 'substitution':
        for i in range(defect_nsites):
            if defect[i]['type'] == def_type and defect[i]['from'] == at_from and defect[i]['to'] == at_to:
                subst += 1
        return subst
    elif def_type == 'vacancy':
        for i in range(defect_nsites):
            if defect[i]['type'] == def_type and defect[i]['element'] == at_from:
                subst += 1
        return subst
    return -1


def getElementCount(item, element):
    if element in item['formula'].keys():
        return item['formula'][element]
    elif element == 'U':
        Me_vac = getDefectNumber(item['defects'], def_type='vacancy', at_from='Mo')
        Me_vac += getDefectNumber(item['defects'], def_type='vacancy', at_from='W')
        return Me_vac
    elif element == 'Np':
        nonMe_vac = getDefectNumber(item['defects'], def_type='vacancy', at_from='S')
        nonMe_vac += getDefectNumber(item['defects'], def_type='vacancy', at_from='Se')
        return nonMe_vac
    return 0


def getFormula2Compare2DMD(item, elements=elements, multiplier=1):
    formula_str = ''
    for el in elements:
        el_number = int(multiplier * getElementCount(item, el))
        formula_str += f'{el}{el_number}'
    return formula_str


def getFormula2CompareFFCCS(item, elements=elements, multiplier=1):
    formula_str = ''
    for el in elements:
        el_number = int(multiplier * item[f'nsites_{el}'])
        formula_str += f'{el}{el_number}'
    return formula_str


def Z(energies_eV_cell, weights, T):
    """
    partition function for group of the ccs entries (same composition and number of atoms)
    energies_eV_cell =  energies per cell (formation energies or relaxed energies)
    weights = weights of the structures from the ccs used (number of merged symmetrical realizations)
    T = temperature in Kelvin
    """
    if np.abs(T) < 1e-15:
        T = 1e-15
    z = np.sum(weights * np.exp(-energies_eV_cell / (kB * T)))
    return z


def T_delta_Sconf(energies_eV_cell, weights, T, natoms_for_per_atom=0):
    """
    configurational entropy contribution to free energies
    energies_eV_cell = energies per cell (formation energies or relaxed energies)
    weights = weights of the structures from the ccs used (number of merged symmetrical realizations)
    T = temperature in Kelvin
    natoms_for_per_atom = number of atoms in the model cell from the ccs used
    """
    min_energy = energies_eV_cell.min()
    energies_eV_cell = energies_eV_cell - min_energy
    if np.abs(T) < 1e-15:
        T = 1e-15
    TdSconf = - T * kB * np.log(Z(energies_eV_cell, weights, T))
    if natoms_for_per_atom > 0:
        return (TdSconf + min_energy) / natoms_for_per_atom
    return TdSconf + min_energy


def getRMSEforModelEhullLimit (test_set, model, limit):
    subset = test_set[test_set['E_above_hull'] < limit]
    try:
        rmse = root_mse(subset['formation_energy_pa'], subset[model])
    except:
        rmse = np.nan
    return rmse


def structShifter(struct):
    indices_to_move = range(len(struct))
    translation_vector = np.array([0, 0, 0.5])
    translation = TranslateSitesTransformation(indices_to_move, translation_vector, vector_in_frac_coords=True)
    struct = translation.apply_transformation(struct)
    return struct


def structureSitesMapper(struct, dict_elements=dict_elements):
    result = np.zeros(len(dict_elements.keys()))
    for ith_site in struct.sites:
        element_symbol = ith_site.specie.symbol
        if element_symbol in dict_elements.keys():
            if ith_site.coords[2] > 10.1:
                result[dict_elements[element_symbol]] += 1
    return result


def enumerateElementPairs(struct, X_element, elements_list, r_cut_outer=3.5, r_cut_inner=0):
    elements_list = np.array(elements_list)
    result = np.zeros(len(elements_list))
    X_pos = np.where(np.array([ith_site.specie.symbol == X_element
                               for ith_site in struct.sites]))[0][0]
    neighbors = struct.get_neighbors(struct.sites[X_pos], r=r_cut_outer, include_image=True)
    for neighbor in neighbors:
        result[np.where(elements_list == neighbor.specie.symbol)[0][0]] += 1

    neighbors = struct.get_neighbors(struct.sites[X_pos], r=r_cut_inner, include_image=True)
    for neighbor in neighbors:
        result[np.where(elements_list == neighbor.specie.symbol)[0][0]] -= 1

    return result


def getDescriptors(base_structures, elements_list_dict, elements_dict,
                   normalize_over_cs=False, shift_to_min=False, model='allegro_random_2dmd-ldc'):
    columns = []
    for rcut_tag, r_cut_inner, r_cut_outer in tqdm([('CS#1', 2.4, 2.6),
                                                    ('CS#2', 2.6, 3.5),
                                                    ('CS#3', 3.5, 4.3)]):
        for element_id, element_of_choise in enumerate(elements_list_dict.keys()):
            columns += [f'{rcut_tag}: {elements_dict[element_of_choise]}-{elements_dict[element]} '
                        for element in elements_list_dict[element_of_choise]]
            distrib_temp = np.array([enumerateElementPairs(struct, element_of_choise,
                                                           elements_list_dict[element_of_choise],
                                                           r_cut_outer=r_cut_outer,
                                                           r_cut_inner=r_cut_inner)
                                     for struct in base_structures['initial_structure_marked']])
            if (element_id == 0) and (rcut_tag == 'CS#1'):
                distrib = distrib_temp.copy()
            else:
                distrib = np.hstack((distrib, distrib_temp))

    distrib = pd.DataFrame(distrib.astype(int), columns=columns, index=base_structures.index)
    additional_columns = ['favorability', model, 'formation_energy_per_atom', ]
    for additional_col in additional_columns:
        distrib[additional_col] = base_structures[additional_col]

    distrib['FE ranges, eV/atom'] = pd.cut(distrib['formation_energy_per_atom'], bins=4)

    col2drop = []
    for icol, col in enumerate(distrib.columns[:-len(additional_columns) + 1]):
        if len(distrib[col].unique()) < 2:
            col2drop += [col]
        else:
            for jcol in range(icol):
                if all(distrib[distrib.columns[icol]] == distrib[distrib.columns[jcol]]):
                    col2drop += [col]
    distrib.drop(col2drop, axis=1, inplace=True)

    # normalization over coordination sphere
    if normalize_over_cs:
        for rcut_tag, _, _ in [('CS#1', 2.4, 2.6), ('CS#2', 2.6, 3.5), ('CS#3', 3.5, 4.3)]:
            subset = distrib[[col for col in distrib.columns if col.startswith(rcut_tag)]]
            assert len(set(subset.sum(axis=1))) == 1
            normalizer = list(set(subset.sum(axis=1)))[0]
            for col in subset.columns:
                distrib[col] /= normalizer
    if shift_to_min:
        distrib_mins = distrib.min(axis=0)
        for feature in distrib_mins.index[:13]:
            distrib[feature] -= distrib_mins.loc[feature]

    return distrib


def countArity(formula):
    key_values = list(formula.keys())
    key_values = [key for key in key_values if (formula[key] > 0)]
    key_values = [key for key in key_values if (key != 'U' and key != 'Np')]
    return len(key_values)


def countMeXVac(formula, vac_tag):
    try:
        return formula[vac_tag]
    except:
        return 0


def styler(as_frame):
    style_mask = [0, ]
    for i in range(as_frame.shape[0])[1:-1]:
        if (as_frame.iloc[i]['type'] != 'less60k') & (as_frame.iloc[i + 1]['type'] == 'less60k'):
            style_mask += [1]
        elif (as_frame.iloc[i]['type'] != 'less60k') & (as_frame.iloc[i - 1]['type'] == 'less60k'):
            style_mask += [1]
        elif as_frame.iloc[i]['type'] != 'less60k':
            style_mask += [1]
        else:
            style_mask += [0]
    style_mask += [0]

    style2 = [0, ]
    st_i = 0
    for i in np.diff(style_mask):
        if i != 0:
            st_i += 1
        style2 += [st_i]

    return style2


def recoverFormulaFromVac(formula, element_list):
    updated_formulas = []
    for element in element_list:
        formula_temp = formula.copy()
        formula_temp.update({element: formula[element] + 1})
        updated_formulas += [formula_temp]
    return list(map(formula2string, updated_formulas))
