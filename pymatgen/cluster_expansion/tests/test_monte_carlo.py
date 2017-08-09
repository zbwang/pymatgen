from __future__ import division
import unittest

from pymatgen import Lattice, Structure
from pymatgen.cluster_expansion.monte_carlo import MonteCarloRunner, make_canonical_flip_function
from pymatgen.cluster_expansion.ce import ClusterExpansion
from pymatgen.transformations.advanced_transformations import OrderDisorderedStructureTransformation

import random
import numpy as np

class MonteCarloTest(unittest.TestCase):
    
    def setUp(self):
        self.lattice = Lattice([[3, 3, 0],[0, 3, 3],[3, 0, 3]])
        species = [{'Li+': 0.3333333}] * 3 + ['Br-']
        coords = ((0.25, 0.25, 0.25), (0.75, 0.75, 0.75), 
                  (0.5, 0.5, 0.5),  (0, 0, 0))
        self.structure = Structure(self.lattice, species, coords)
        self.structure.make_supercell([[2, 1, 1]])
        self.ce = ClusterExpansion.from_radii(self.structure, {2: 2}, use_ewald=False)
        
    def test_monte_carlo_single_site(self):
        ecis = np.array([-1, 0.1, 0.2])
        energies = []
        for x in OrderDisorderedStructureTransformation().apply_transformation(self.structure, 1000):
            energies.append(self.ce.structure_energy(x['structure'], ecis))

        energies = np.array(energies)
        self.assertEqual(len(energies), 15)  # 6 choose 3

        def enthalpy(temperature=500):
            k = 8.6173303e-05
            probabilities = np.exp(-energies/(k * temperature) + np.min(energies))
            probabilities /= np.sum(probabilities)
            return np.sum(energies * probabilities)

        s = OrderDisorderedStructureTransformation().apply_transformation(self.structure)
        cs = self.ce.supercell_from_structure(s)
        occu = cs.occu_from_structure(s)

        indices = np.array([i for i, b in enumerate(cs.bits) if np.all(b == cs.bits[0])])

        def flip_function(occu):
            i = random.choice(indices)
            val = occu[i]
            j = random.choice([j for j in indices if val != occu[j]])
            return [(i, occu[j]), (j, occu[i])]

        # failing these should be ~10 sigma event
        mcr = MonteCarloRunner(occu, cs, ecis, make_canonical_flip_function(cs))
        mcr.run_mc(n_iterations=20000, start_t=500, end_t=500, n_samples=5000)
        self.assertAlmostEqual(enthalpy(500), np.mean(mcr.energies), places=2)

        mcr = MonteCarloRunner(occu, cs, ecis, flip_function)
        mcr.run_mc(n_iterations=20000, start_t=3000, end_t=3000, n_samples=5000)
        self.assertAlmostEqual(enthalpy(3000), np.mean(mcr.energies), places=2)
