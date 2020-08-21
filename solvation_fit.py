from scipy.optimize import curve_fit
import numpy as np


class SolvationFit(object):
    """Class to calculate and store solvation and anion-cation energies.

    Attributes
    ----------
    cation_IA : dict
        Dictionary that contains the ionization energies and electron affinities of the cations.
    anion_IA : dict
        Dictionary that contains the ionization energies and electron affinities of the anions.
    solvent_IA : dict
        Dictionary that contains the ionization energies and electron affinities of the solvents.
    data_file : str
        File containing the experimental data.
    calc_type : list
        Indicates which components are going to be constant and which are going to be varied
        in the calculation.
        calc_type[0] : 'solvent' or 'ion' (constant part of the calculation).
        calc_type[1] : if calc_type[0] == 'solvent' this is a list solvents.
        calc_type[1] : if calc_type[0] == 'ion' this is a list of 2-tuples,
                       fist cation and second anion.
    min_bound : float
        Lower bound on the gamma and zeta parameters. Default is no bound.
    max_bound : float
        Upper bound on the gamma and zeta parameters. Default is no bound.
    cations : {None, list}
        Cations that will be considered in the calculation.
        Default is the same cations included in cations_IA.
    anions : {None, list}
        Anions that will be considered in the calculation.
        Default is the same anions included in anions_IA.
    solvents : {None, list}
        Solvents that will be considered in the calculation.
        Default is the same solvents included in solvent_IA.
    solvation_data : dict
        Dictionary that contains all the experimental data.
    i_cat_solv : np.ndarray
        Numpy array that contains the available I values for the cations that will be used
        in the solvation energy calculations.
    a_cat_solv : np.ndarray
        Numpy array that contains the available A values for the cations that will be used
        in the solvation energy calculations.
    i_an_solv : np.ndarray
        Numpy array that contains the available I values for the anions that will be used
        in the solvation energy calculations.
    a_an_solv : np.ndarray
        Numpy array that contains the available A values for the anions that will be used
        in the solvation energy calculations.
    i_sol_solv : np.ndarray
        Numpy array that contains the available I values for the solvents that will be used
        in the solvation energy calculations.
    a_sol_solv : np.ndarray
        Numpy array that contains the available A values for the solvents that will be used
        in the solvation energy calculations.
    i_cat_diff : np.ndarray
        Numpy array that contains the available I values for the cations that will be used
        in the anion-cation energy calculations.
    a_cat_diff : np.ndarray
        Numpy array that contains the available A values for the cations that will be used
        in the anion-cation energy calculations.
    i_an_diff : np.ndarray
        Numpy array that contains the available I values for the anions that will be used
        in the anion-cation energy calculations.
    a_an_diff : np.ndarray
        Numpy array that contains the available A values for the anions that will be used
        in the anion-cation energy calculations.
    i_sol_diff : np.ndarray
        Numpy array that contains the available I values for the solvents that will be used
        in the anion-cation energy calculations.
    a_sol_diff : np.ndarray
        Numpy array that contains the available A values for the solvents that will be used
        in the anion-cation energy calculations.
    popt_sol_e_allparams_unbound : np.ndarray (if it can be calculated)
        Numpy array with the (unbound) parameters of the solvation energy calculation.
        All parameters are varied.
    popt_sol_e_allparams_bound : np.ndarray (if it can be calculated)
        Numpy array with the (bound) parameters of the solvation energy calculation.
        All parameters are varied.
    popt_sol_e_gamma_unbound : np.ndarray (if it can be calculated)
        Numpy array with the (unbound) parameters of the solvation energy calculation.
        Only the gamma and linear parameters are varied.
    popt_sol_e_gamma_bound : np.ndarray (if it can be calculated)
        Numpy array with the (bound) parameters of the solvation energy calculation.
        Only the gamma and linear parameters are varied.
    popt_sol_e_zeta_unbound : np.ndarray (if it can be calculated)
        Numpy array with the (unbound) parameters of the solvation energy calculation.
        Only the zeta and linear parameters are varied.
    popt_sol_e_zeta_bound : np.ndarray (if it can be calculated)
        Numpy array with the (bound) parameters of the solvation energy calculation.
        Only the zeta and linear parameters are varied.
    popt_sol_e_simple_unbound : np.ndarray (if it can be calculated)
        Numpy array with the (unbound) parameters of the solvation energy calculation.
        Only the linear parameters are varied.
    popt_an_cat_diff_allparams_unbound : np.ndarray (if it can be calculated)
        Numpy array with the (unbound) parameters of the anion-cation energy calculation.
        All parameters are varied.
    popt_an_cat_diff_allparams_bound : np.ndarray (if it can be calculated)
        Numpy array with the (bound) parameters of the anion-cation energy calculation.
        All parameters are varied.
    popt_an_cat_diff_gamma_unbound : np.ndarray (if it can be calculated)
        Numpy array with the (unbound) parameters of the anion-cation energy calculation.
        Only the gamma and linear parameters are varied.
    popt_an_cat_diff_gamma_bound : np.ndarray (if it can be calculated)
        Numpy array with the (bound) parameters of the anion-cation energy calculation.
        Only the gamma and linear parameters are varied.
    popt_an_cat_diff_zeta_unbound : np.ndarray (if it can be calculated)
        Numpy array with the (unbound) parameters of the anion-cation energy calculation.
        Only the zeta and linear parameters are varied.
    popt_an_cat_diff_zeta_bound : np.ndarray (if it can be calculated)
        Numpy array with the (bound) parameters of the anion-cation energy calculation.
        Only the zeta and linear parameters are varied.
    popt_an_cat_diff_simple_unbound : np.ndarray (if it can be calculated)
        Numpy array with the (unbound) parameters of the anion-cation energy calculation.
        Only the linear parameters are varied.
    sol_e_allparams_unbound_result : np.ndarray (if it can be calculated)
        Numpy array with the results of the solvation energy calculations.
        Unbound parameters. All parameters varied.
    sol_e_allparams_bound_result : np.ndarray (if it can be calculated)
        Numpy array with the results of the solvation energy calculations.
        Bound parameters. All parameters varied.
    sol_e_gamma_unbound_result : np.ndarray (if it can be calculated)
        Numpy array with the results of the solvation energy calculations.
        Unbound parameters. Gamma and linear parameters varied.
    sol_e_gamma_bound_result : np.ndarray (if it can be calculated)
        Numpy array with the results of the solvation energy calculations.
        Bound parameters. Gamma and linear parameters varied.
    sol_e_zeta_unbound_result : np.ndarray (if it can be calculated)
        Numpy array with the results of the solvation energy calculations.
        Unbound parameters. Zeta and linear parameters varied.
    sol_e_zeta_bound_result : np.ndarray (if it can be calculated)
        Numpy array with the results of the solvation energy calculations.
        Bound parameters. Zeta and linear parameters varied.
    sol_e_simple_unbound_result : np.ndarray (if it can be calculated)
        Numpy array with the results of the solvation energy calculations.
        Unbound parameters. Linear parameters varied.
    an_cat_diff_allparams_unbound_result : np.ndarray (if it can be calculated)
        Numpy array with the results of the anion-cation energy calculations.
        Unbound parameters. All parameters varied.
    an_cat_diff_allparams_bound_result : np.ndarray (if it can be calculated)
        Numpy array with the results of the anion-cation energy calculations.
        Bound parameters. All parameters varied.
    an_cat_diff_gamma_unbound_result : np.ndarray (if it can be calculated)
        Numpy array with the results of the anion-cation energy calculations.
        Unbound parameters. Gamma and linear parameters varied.
    an_cat_diff_gamma_bound_result : np.ndarray (if it can be calculated)
        Numpy array with the results of the anion-cation energy calculations.
        Bound parameters. Gamma and linear parameters varied.
    an_cat_diff_zeta_unbound_result : np.ndarray (if it can be calculated)
        Numpy array with the results of the anion-cation energy calculations.
        Unbound parameters. Zeta and linear parameters varied.
    an_cat_diff_zeta_bound_result : np.ndarray (if it can be calculated)
        Numpy array with the results of the anion-cation energy calculations.
        Bound parameters. Zeta and linear parameters varied.
    an_cat_diff_simple_unbound_result : np.ndarray (if it can be calculated)
        Numpy array with the results of the anion-cation energy calculations.
        Unbound parameters. Linear parameters varied.
    sol_e_allparams_unbound_error : np.ndarray (if it can be calculated)
        Numpy array with the errors of the solvation energy calculations.
        Unbound parameters. All parameters varied.
    sol_e_allparams_bound_error : np.ndarray (if it can be calculated)
        Numpy array with the errors of the solvation energy calculations.
        Bound parameters. All parameters varied.
    sol_e_gamma_unbound_error : np.ndarray (if it can be calculated)
        Numpy array with the errors of the solvation energy calculations.
        Unbound parameters. Gamma and linear parameters varied.
    sol_e_gamma_bound_error : np.ndarray (if it can be calculated)
        Numpy array with the errors of the solvation energy calculations.
        Bound parameters. Gamma and linear parameters varied.
    sol_e_zeta_unbound_error : np.ndarray (if it can be calculated)
        Numpy array with the errors of the solvation energy calculations.
        Unbound parameters. Zeta and linear parameters varied.
    sol_e_zeta_bound_error : np.ndarray (if it can be calculated)
        Numpy array with the errors of the solvation energy calculations.
        Bound parameters. Zeta and linear parameters varied.
    sol_e_simple_unbound_error : np.ndarray (if it can be calculated)
        Numpy array with the errors of the solvation energy calculations.
        Unbound parameters. Linear parameters varied.
    an_cat_diff_allparams_unbound_error : np.ndarray (if it can be calculated)
        Numpy array with the errors of the anion-cation energy calculations.
        Unbound parameters. All parameters varied.
    an_cat_diff_allparams_bound_error : np.ndarray (if it can be calculated)
        Numpy array with the errors of the anion-cation energy calculations.
        Bound parameters. All parameters varied.
    an_cat_diff_gamma_unbound_error : np.ndarray (if it can be calculated)
        Numpy array with the errors of the anion-cation energy calculations.
        Unbound parameters. Gamma and linear parameters varied.
    an_cat_diff_gamma_bound_error : np.ndarray (if it can be calculated)
        Numpy array with the errors of the anion-cation energy calculations.
        Bound parameters. Gamma and linear parameters varied.
    an_cat_diff_zeta_unbound_error : np.ndarray (if it can be calculated)
        Numpy array with the errors of the anion-cation energy calculations.
        Unbound parameters. Zeta and linear parameters varied.
    an_cat_diff_zeta_bound_error : np.ndarray (if it can be calculated)
        Numpy array with the errors of the anion-cation energy calculations.
        Bound parameters. Zeta and linear parameters varied.
    an_cat_diff_simple_unbound_error : np.ndarray (if it can be calculated)
        Numpy array with the errors of the anion-cation energy calculations.
        Unbound parameters. Linear parameters varied.
    exp_data_solv : np.ndarray
        Experimental solvation energies.
    exp_data_solv : np.ndarray
        Experimental anion-cation energies.

    Methods
    -------
    __init__(self, cation_IA, anion_IA, solvent_IA, data_file, calc_type,
                 min_bound=-np.inf, max_bound=np.inf, cations=None, anions=None, solvents=None)
        Initialize the object.
    gen_data_container()
        Generate the dictionary that will contain all the experimental data.
    read_data(self)
        Read all the experimental data from self.data_file.
    select_data(self)
        Select the I and A values that will be used for the current calculation.
    fit_data(self)
        Fit the data to the experimental solvation energies and anion-cation energies.
    get_results_errors(self)
        Calculate the CDFT solvation energies and anion-cation energies and errors.
        Errors calculated as: CDFT_value - Experimental_value.

    Static Methods
    --------------
    Eab(i_a, a_a, i_b, a_b, gamma=1.0, zeta=1.0)
        Calculate the charge transfer energy between two compounds.
    sol_e_allparams(data, gamma_cat, gamma_an, zeta_cat, zeta_an, m, b)
        Input of the solvation energy fit, all parameters varied.
    sol_e_gamma(data, gamma_cat, gamma_an, m, b)
        Input of the solvation energy fit, gamma and linear parameters varied.
    sol_e_zeta(data, zeta_cat, zeta_an, m, b)
        Input of the solvation energy fit, zeta and linear parameters varied.
    sol_e_simple(data, m, b)
        Input of the solvation energy fit, linear parameters varied.
    an_cat_dif_allparams(data, gamma_cat, gamma_an, zeta_cat, zeta_an, m, b)
        Input of the anion-cation energy fit, all parameters varied.
    an_cat_dif_gamma(data, gamma_cat, gamma_an, m, b)
        Input of the anion-cation energy fit, gamma and linear parameters varied.
    an_cat_dif_zeta(data, zeta_cat, zeta_an, m, b)
        Input of the anion-cation energy fit, zeta and linear parameters varied.
    an_cat_dif_simple(data, m, b)
        Input of the anion-cation energy fit, linear parameters varied.
    sol_e_allparams_expl(i_cat, a_cat, i_an, a_an, i_sol, a_sol,
                         gamma_cat, gamma_an, zeta_cat, zeta_an, m, b)
        Calculate the solvation energy, all parameters varied.
    sol_e_gamma_expl(i_cat, a_cat, i_an, a_an, i_sol, a_sol, gamma_cat, gamma_an, m, b)
        Calculate the solvation energy, gamma and linear parameters varied.
    sol_e_zeta_expl(i_cat, a_cat, i_an, a_an, i_sol, a_sol, zeta_cat, zeta_an, m, b)
        Calculate the solvation energy, zeta and linear parameters varied.
    sol_e_simple_expl(i_cat, a_cat, i_an, a_an, i_sol, a_sol, m, b)
        Calculate the solvation energy, linear parameters varied.
    an_cat_dif_allparams_expl(i_cat, a_cat, i_an, a_an, i_sol, a_sol,
                              gamma_cat, gamma_an, zeta_cat, zeta_an, m, b)
        Calculate the anion-cation energy, all parameters varied.
    an_cat_dif_gamma_expl(i_cat, a_cat, i_an, a_an, i_sol, a_sol, gamma_cat, gamma_an, m, b)
        Calculate the anion-cation energy, gamma and linear parameters varied.
    an_cat_dif_zeta_expl(i_cat, a_cat, i_an, a_an, i_sol, a_sol, zeta_cat, zeta_an, m, b)
        Calculate the anion-cation energy, zeta and linear parameters varied.
    an_cat_dif_simple_expl(i_cat, a_cat, i_an, a_an, i_sol, a_sol, m, b)
        Calculate the anion-cation energy, linear parameters varied.
    """

    def __init__(self, cation_IA, anion_IA, solvent_IA, data_file, calc_type,
                 min_bound=-np.inf, max_bound=np.inf, cations=None, anions=None, solvents=None):
        """Initialize the object.

        Parameters
        ----------
        cation_IA : dict
            Dictionary that contains the ionization energies and electron affinities of the cations.
        anion_IA : dict
            Dictionary that contains the ionization energies and electron affinities of the anions.
        solvent_IA : dict
            Dictionary that contains the ionization energies and electron affinities
            of the solvents.
        data_file : str
            File containing the experimental data.
        calc_type : list
            Indicates which components are going to be constant and which are going to be varied
            in the calculation.
            calc_type[0] : 'solvent' or 'ion' (constant part of the calculation).
            calc_type[1] : if calc_type[0] == 'solvent' this is a list solvents.
            calc_type[1] : if calc_type[0] == 'ion' this is a list of 2-tuples,
                           fist cation and second anion.
        min_bound : float
            Lower bound on the gamma and zeta parameters. Default is no bound.
        max_bound : float
            Upper bound on the gamma and zeta parameters. Default is no bound.
        cations : {None, list}
            Cations that will be considered in the calculation.
            Default is the same cations included in cations_IA.
        anions : {None, list}
            Anions that will be considered in the calculation.
            Default is the same anions included in anions_IA.
        solvents : {None, list}
            Solvents that will be considered in the calculation.
            Default is the same solvents included in solvent_IA.
        """
        self.cation_IA = cation_IA
        self.anion_IA = anion_IA
        self.solvent_IA = solvent_IA
        self.data_file = data_file
        self.calc_type = calc_type
        self.min_bound = min_bound
        self.max_bound = max_bound
        if cations:
            self.cations = cations
        else:
            self.cations = cation_IA
        if anions:
            self.anions = anions
        else:
            self.anions = anion_IA
        if solvents:
            self.solvents = solvents
        else:
            self.solvents = solvent_IA
        self.gen_data_container()
        self.read_data()
        self.select_data()
        self.fit_data()
        self.get_results_errors()

    def gen_data_container(self):
        """Generate the dictionary that will contain all the experimental data."""
        SolvationFit = {}
        for solvent in self.solvent_IA:
            SolvationFit[solvent] = {}
            for cat in self.cation_IA:
                for an in self.anion_IA:
                    SolvationFit[solvent][cat + an] = {}
        self.solvation_data = SolvationFit

    def read_data(self):
        """Read all the experimental data from self.data_file."""
        infile = open(self.data_file, "r")
        raw_lines = infile.readlines()
        infile.close()
        for line in raw_lines:
            l = line.strip().split(",")
            for solvent in self.solvent_IA:
                for cat in self.cation_IA:
                    for an in self.anion_IA:
                        if (l[0] == solvent) and (l[1] == cat) and (l[2] == an):
                            if l[3] != "":
                                self.solvation_data[solvent][cat + an]["dH-+"] = float(l[3])
                            else:
                                self.solvation_data[solvent][cat + an]["dH-+"] = l[3]
                            if l[4] != "":
                                self.solvation_data[solvent][cat + an]["dHs"] = float(l[4])
                            else:
                                self.solvation_data[solvent][cat + an]["dHs"] = line[4]
                            if l[5] != "":
                                self.solvation_data[solvent][cat + an]["dG-+"] = float(l[5])
                            else:
                                self.solvation_data[solvent][cat + an]["dG-+"] = line[5]
                            if l[6] != "":
                                self.solvation_data[solvent][cat + an]["dGs"] = float(l[6])
                            else:
                                self.solvation_data[solvent][cat + an]["dGs"] = line[6]

    def select_data(self):
        """Select the I and A values that will be used for the current calculation."""
        state_function = "d" + self.calc_type[2]
        i_cat_solv = []
        a_cat_solv = []
        i_an_solv = []
        a_an_solv = []
        i_sol_solv = []
        a_sol_solv = []
        exp_data_solv = []
        i_cat_diff = []
        a_cat_diff = []
        i_an_diff = []
        a_an_diff = []
        i_sol_diff = []
        a_sol_diff = []
        exp_data_diff = []
        if self.calc_type[0] == "solvent":
            for solvent in self.calc_type[1]:
                for cat in self.cations:
                    for an in self.anions:
                        if isinstance(self.solvation_data[solvent][cat + an]
                                      [state_function + "s"], float):
                            i_cat_solv.append(self.cation_IA[cat][0])
                            a_cat_solv.append(self.cation_IA[cat][1])
                            i_an_solv.append(self.anion_IA[an][0])
                            a_an_solv.append(self.anion_IA[an][1])
                            i_sol_solv.append(self.solvent_IA[solvent][0])
                            a_sol_solv.append(self.solvent_IA[solvent][1])
                            exp_data_solv.append(self.solvation_data[solvent][cat + an][
                                                     state_function + "s"])
                        if isinstance(self.solvation_data[solvent][cat + an]
                                      [state_function + "-+"], float):
                            i_cat_diff.append(self.cation_IA[cat][0])
                            a_cat_diff.append(self.cation_IA[cat][1])
                            i_an_diff.append(self.anion_IA[an][0])
                            a_an_diff.append(self.anion_IA[an][1])
                            i_sol_diff.append(self.solvent_IA[solvent][0])
                            a_sol_diff.append(self.solvent_IA[solvent][1])
                            exp_data_diff.append(self.solvation_data[solvent][cat + an][
                                                     state_function + "-+"])
                    else:
                        pass
        elif "ion" in self.calc_type[0]:
            for salt in self.calc_type[1]:
                cat = salt[0]
                an = salt[1]
                for solvent in self.solvents:
                    if isinstance(self.solvation_data[solvent][cat + an]
                                  [state_function + "s"], float):
                        i_cat_solv.append(self.cation_IA[cat][0])
                        a_cat_solv.append(self.cation_IA[cat][1])
                        i_an_solv.append(self.anion_IA[an][0])
                        a_an_solv.append(self.anion_IA[an][1])
                        i_sol_solv.append(self.solvent_IA[solvent][0])
                        a_sol_solv.append(self.solvent_IA[solvent][1])
                        exp_data_solv.append(self.solvation_data[solvent][cat + an][
                                                 state_function + "s"])
                    if isinstance(self.solvation_data[solvent][cat + an]
                                  [state_function + "-+"], float):
                        i_cat_diff.append(self.cation_IA[cat][0])
                        a_cat_diff.append(self.cation_IA[cat][1])
                        i_an_diff.append(self.anion_IA[an][0])
                        a_an_diff.append(self.anion_IA[an][1])
                        i_sol_diff.append(self.solvent_IA[solvent][0])
                        a_sol_diff.append(self.solvent_IA[solvent][1])
                        exp_data_diff.append(self.solvation_data[solvent][cat + an][
                                                 state_function + "-+"])
                else:
                    pass
        self.i_cat_solv = np.array(i_cat_solv)
        self.a_cat_solv = np.array(a_cat_solv)
        self.i_an_solv = np.array(i_an_solv)
        self.a_an_solv = np.array(a_an_solv)
        self.i_sol_solv = np.array(i_sol_solv)
        self.a_sol_solv = np.array(a_sol_solv)
        self.exp_data_solv = np.array(exp_data_solv) * 0.239006
        self.i_cat_diff = np.array(i_cat_diff)
        self.a_cat_diff = np.array(a_cat_diff)
        self.i_an_diff = np.array(i_an_diff)
        self.a_an_diff = np.array(a_an_diff)
        self.i_sol_diff = np.array(i_sol_diff)
        self.a_sol_diff = np.array(a_sol_diff)
        self.exp_data_diff = np.array(exp_data_diff) * 0.239006

    @staticmethod
    def Eab(i_a, a_a, i_b, a_b, gamma=1.0, zeta=1.0):
        """Calculate the charge transfer energy between two compounds.

        Parameters
        ----------
        i_a : float
            Ionization energy of compound A.
        a_a : float
            Electron affinity of compound A.
        i_b : float
            Ionization energy of compound B.
        a_b : float
            Electron affinity of compound B.
        gamma : float
            Perturbation on the chemical potential.
        zeta : float
            Perturbation on the hardness.

        Returns
        -------
        dE : float
            Charge transfer energy between two compounds.
        """
        mu_a = -1 * (gamma * i_a + a_a)/(1 + gamma)
        mu_b = -1 * (i_b + gamma * a_b)/(1 + gamma)
        eta_a = zeta * (i_a - a_a)
        eta_b = zeta * (i_b - a_b)
        dE = -0.5 * (mu_a - mu_b)**2 / (eta_a + eta_b)
        return dE

    @staticmethod
    def sol_e_allaparams(data, gamma_cat, gamma_an, zeta_cat, zeta_an, m, b):
        """Input of the solvation energy fit, all parameters varied.

        Parameters
        ----------
        data : {tuple, list}
            Tuple/list containing the I and A values to be unpacked.
        gamma_cat : float
            Perturbation on the chemical potential of the cation.
        gamma_an : float
            Perturbation on the chemical potential of the anion.
        zeta_cat : float
            Perturbation on the hardness of the cation.
        zeta_an : float
            Perturbation on the hardness of the anion.
        m : float
            Slope of the linear fit.
        b : float
            Intercept of the linear fit.

        Returns
        -------
        Solvation energy, all parameters varied.
        """
        i_cat, a_cat, i_an, a_an, i_sol, a_sol = data
        dEsolv_cdft = (SolvationFit.Eab(i_cat, a_cat, i_sol, a_sol, gamma_cat, zeta_cat) +
                       SolvationFit.Eab(i_an, a_an, i_sol, a_sol, gamma_an, zeta_an) -
                       SolvationFit.Eab(i_cat, a_cat, i_an, a_an, gamma=1.0, zeta=1.0))
        return m * dEsolv_cdft + b

    @staticmethod
    def sol_e_gamma(data, gamma_cat, gamma_an, m, b):
        """Input of the solvation energy fit, gamma and linear parameters varied.

        Parameters
        ----------
        data : {tuple, list}
            Tuple/list containing the I and A values to be unpacked.
        gamma_cat : float
            Perturbation on the chemical potential of the cation.
        gamma_an : float
            Perturbation on the chemical potential of the anion.
        m : float
            Slope of the linear fit.
        b : float
            Intercept of the linear fit.

        Returns
        -------
        Solvation energy, gamma and linear parameters varied.
        """
        i_cat, a_cat, i_an, a_an, i_sol, a_sol = data
        dEsolv_cdft = (SolvationFit.Eab(i_cat, a_cat, i_sol, a_sol, gamma_cat, zeta=1.0) +
                       SolvationFit.Eab(i_an, a_an, i_sol, a_sol, gamma_an, zeta=1.0) -
                       SolvationFit.Eab(i_cat, a_cat, i_an, a_an, gamma=1.0, zeta=1.0))
        return m * dEsolv_cdft + b

    @staticmethod
    def sol_e_zeta(data, zeta_cat, zeta_an, m, b):
        """Input of the solvation energy fit, zeta and linear parameters varied.

        Parameters
        ----------
        data : {tuple, list}
            Tuple/list containing the I and A values to be unpacked.
        zeta_cat : float
            Perturbation on the hardness of the cation.
        zeta_an : float
            Perturbation on the hardness of the anion.
        m : float
            Slope of the linear fit.
        b : float
            Intercept of the linear fit.

        Returns
        -------
        Solvation energy, zeta and linear parameters varied.
        """
        i_cat, a_cat, i_an, a_an, i_sol, a_sol = data
        dEsolv_cdft = (SolvationFit.Eab(i_cat, a_cat, i_sol, a_sol, gamma=1.0, zeta=zeta_cat) +
                       SolvationFit.Eab(i_an, a_an, i_sol, a_sol, gamma=1.0, zeta=zeta_an) -
                       SolvationFit.Eab(i_cat, a_cat, i_an, a_an, gamma=1.0, zeta=1.0))
        return m * dEsolv_cdft + b

    @staticmethod
    def sol_e_simple(data, m, b):
        """Input of the solvation energy fit, linear parameters varied.

        Parameters
        ----------
        data : {tuple, list}
            Tuple/list containing the I and A values to be unpacked.
        m : float
            Slope of the linear fit.
        b : float
            Intercept of the linear fit.

        Returns
        -------
        Solvation energy, linear parameters varied.
        """
        i_cat, a_cat, i_an, a_an, i_sol, a_sol = data
        dEsolv_cdft = (SolvationFit.Eab(i_cat, a_cat, i_sol, a_sol, gamma=1.0, zeta=1.0) +
                       SolvationFit.Eab(i_an, a_an, i_sol, a_sol, gamma=1.0, zeta=1.0) -
                       SolvationFit.Eab(i_cat, a_cat, i_an, a_an, gamma=1.0, zeta=1.0))
        return m * dEsolv_cdft + b

    @staticmethod
    def an_cat_dif_allparams(data, gamma_cat, gamma_an, zeta_cat, zeta_an, m, b):
        """Input of the anion-cation energy fit, all parameters varied.

        Parameters
        ----------
        data : {tuple, list}
            Tuple/list containing the I and A values to be unpacked.
        gamma_cat : float
            Perturbation on the chemical potential of the cation.
        gamma_an : float
            Perturbation on the chemical potential of the anion.
        zeta_cat : float
            Perturbation on the hardness of the cation.
        zeta_an : float
            Perturbation on the hardness of the anion.
        m : float
            Slope of the linear fit.
        b : float
            Intercept of the linear fit.

        Returns
        -------
        Anion-cation energy, all parameters varied.
        """
        i_cat, a_cat, i_an, a_an, i_sol, a_sol = data
        dE_an_cat_cdft = (SolvationFit.Eab(i_an, a_an, i_sol, a_sol, gamma_an, zeta_an) -
                          SolvationFit.Eab(i_cat, a_cat, i_sol, a_sol, gamma_cat, zeta_cat))
        return m * dE_an_cat_cdft + b

    @staticmethod
    def an_cat_dif_gamma(data, gamma_cat, gamma_an, m, b):
        """Input of the anion-cation energy fit, gamma and linear parameters varied.

        Parameters
        ----------
        data : {tuple, list}
            Tuple/list containing the I and A values to be unpacked.
        gamma_cat : float
            Perturbation on the chemical potential of the cation.
        gamma_an : float
            Perturbation on the chemical potential of the anion.
        m : float
            Slope of the linear fit.
        b : float
            Intercept of the linear fit.

        Returns
        -------
        Anion-cation energy, gamma and linear parameters varied.
        """
        i_cat, a_cat, i_an, a_an, i_sol, a_sol = data
        dE_an_cat_cdft = (SolvationFit.Eab(i_an, a_an, i_sol, a_sol, gamma_an, zeta=1.0) -
                          SolvationFit.Eab(i_cat, a_cat, i_sol, a_sol, gamma_cat, zeta=1.0))
        return m * dE_an_cat_cdft + b

    @staticmethod
    def an_cat_dif_zeta(data, zeta_cat, zeta_an, m, b):
        """Input of the anion-cation energy fit, zeta and linear parameters varied.

        Parameters
        ----------
        data : {tuple, list}
            Tuple/list containing the I and A values to be unpacked.
        zeta_cat : float
            Perturbation on the hardness of the cation.
        zeta_an : float
            Perturbation on the hardness of the anion.
        m : float
            Slope of the linear fit.
        b : float
            Intercept of the linear fit.

        Returns
        -------
        Anion-cation energy, zeta and linear parameters varied.
        """
        i_cat, a_cat, i_an, a_an, i_sol, a_sol = data
        dE_an_cat_cdft = (SolvationFit.Eab(i_an, a_an, i_sol, a_sol, gamma=1.0, zeta=zeta_an) -
                          SolvationFit.Eab(i_cat, a_cat, i_sol, a_sol, gamma=1.0, zeta=zeta_cat))
        return m * dE_an_cat_cdft + b

    @staticmethod
    def an_cat_dif_simple(data, m, b):
        """Input of the anion-cation energy fit, linear parameters varied.

        Parameters
        ----------
        data : {tuple, list}
            Tuple/list containing the I and A values to be unpacked.
        m : float
            Slope of the linear fit.
        b : float
            Intercept of the linear fit.

        Returns
        -------
        Anion-cation energy, linear parameters varied.
        """
        i_cat, a_cat, i_an, a_an, i_sol, a_sol = data
        dE_an_cat_cdft = (SolvationFit.Eab(i_an, a_an, i_sol, a_sol,
                                           gamma=1.0, zeta=1.0) -
                          SolvationFit.Eab(i_cat, a_cat, i_sol, a_sol,
                                           gamma=1.0, zeta=1.0))
        return m * dE_an_cat_cdft + b

    @staticmethod
    def sol_e_allaparams_expl(i_cat, a_cat, i_an, a_an, i_sol, a_sol, gamma_cat, gamma_an,
                              zeta_cat, zeta_an, m, b):
        """Calculate the solvation energy, all parameters varied.

        Parameters
        ----------
        i_cat : float
            Ionization energy of the cation.
        a_cat : float
            Electron affinity energy of the cation.
        i_an : float
            Ionization energy of the anion.
        a_an : float
            Electron affinity energy of the anion.
        i_sol : float
            Ionization energy of the solvent.
        a_sol : float
            Electron affinity energy of the solvent.
        gamma_cat : float
            Perturbation on the chemical potential of the cation.
        gamma_an : float
            Perturbation on the chemical potential of the anion.
        zeta_cat : float
            Perturbation on the hardness of the cation.
        zeta_an : float
            Perturbation on the hardness of the anion.
        m : float
            Slope of the linear fit.
        b : float
            Intercept of the linear fit.

        Returns
        -------
        dEsolv_cdft : float
            Solvation energy, all parameters varied.
        """
        dEsolv_cdft = (SolvationFit.Eab(i_cat, a_cat, i_sol, a_sol, gamma_cat, zeta_cat) +
                       SolvationFit.Eab(i_an, a_an, i_sol, a_sol, gamma_an, zeta_an) -
                       SolvationFit.Eab(i_cat, a_cat, i_an, a_an, gamma=1.0, zeta=1.0))
        return m * dEsolv_cdft + b

    @staticmethod
    def sol_e_gamma_expl(i_cat, a_cat, i_an, a_an, i_sol, a_sol, gamma_cat, gamma_an, m, b):
        """Calculate the solvation energy, gamma and linear parameters varied.

        Parameters
        ----------
        i_cat : float
            Ionization energy of the cation.
        a_cat : float
            Electron affinity energy of the cation.
        i_an : float
            Ionization energy of the anion.
        a_an : float
            Electron affinity energy of the anion.
        i_sol : float
            Ionization energy of the solvent.
        a_sol : float
            Electron affinity energy of the solvent.
        gamma_cat : float
            Perturbation on the chemical potential of the cation.
        gamma_an : float
            Perturbation on the chemical potential of the anion.
        m : float
            Slope of the linear fit.
        b : float
            Intercept of the linear fit.

        Returns
        -------
        dEsolv_cdft : float
            Solvation energy, gamma and linear parameters varied.
        """
        dEsolv_cdft = (SolvationFit.Eab(i_cat, a_cat, i_sol, a_sol, gamma_cat, zeta=1.0) +
                       SolvationFit.Eab(i_an, a_an, i_sol, a_sol, gamma_an, zeta=1.0) -
                       SolvationFit.Eab(i_cat, a_cat, i_an, a_an, gamma=1.0, zeta=1.0))
        return m * dEsolv_cdft + b

    @staticmethod
    def sol_e_zeta_expl(i_cat, a_cat, i_an, a_an, i_sol, a_sol, zeta_cat, zeta_an, m, b):
        """Calculate the solvation energy, zeta and linear parameters varied.

        Parameters
        ----------
        i_cat : float
            Ionization energy of the cation.
        a_cat : float
            Electron affinity energy of the cation.
        i_an : float
            Ionization energy of the anion.
        a_an : float
            Electron affinity energy of the anion.
        i_sol : float
            Ionization energy of the solvent.
        a_sol : float
            Electron affinity energy of the solvent.
        zeta_cat : float
            Perturbation on the hardness of the cation.
        zeta_an : float
            Perturbation on the hardness of the anion.
        m : float
            Slope of the linear fit.
        b : float
            Intercept of the linear fit.

        Returns
        -------
        dEsolv_cdft : float
            Solvation energy, zeta and linear parameters varied.
        """
        dEsolv_cdft = (SolvationFit.Eab(i_cat, a_cat, i_sol, a_sol, gamma=1.0, zeta=zeta_cat) +
                       SolvationFit.Eab(i_an, a_an, i_sol, a_sol, gamma=1.0, zeta=zeta_an) -
                       SolvationFit.Eab(i_cat, a_cat, i_an, a_an, gamma=1.0, zeta=1.0))
        return m * dEsolv_cdft + b

    @staticmethod
    def sol_e_simple_expl(i_cat, a_cat, i_an, a_an, i_sol, a_sol, m, b):
        """Calculate the solvation energy, linear parameters varied.

        Parameters
        ----------
        i_cat : float
            Ionization energy of the cation.
        a_cat : float
            Electron affinity energy of the cation.
        i_an : float
            Ionization energy of the anion.
        a_an : float
            Electron affinity energy of the anion.
        i_sol : float
            Ionization energy of the solvent.
        a_sol : float
            Electron affinity energy of the solvent.
        m : float
            Slope of the linear fit.
        b : float
            Intercept of the linear fit.

        Returns
        -------
        dEsolv_cdft : float
            Solvation energy, linear parameters varied.
        """
        dEsolv_cdft = (SolvationFit.Eab(i_cat, a_cat, i_sol, a_sol, gamma=1.0, zeta=1.0) +
                       SolvationFit.Eab(i_an, a_an, i_sol, a_sol, gamma=1.0, zeta=1.0) -
                       SolvationFit.Eab(i_cat, a_cat, i_an, a_an, gamma=1.0, zeta=1.0))
        return m * dEsolv_cdft + b

    @staticmethod
    def an_cat_dif_allparams_expl(i_cat, a_cat, i_an, a_an, i_sol, a_sol, gamma_cat, gamma_an,
                                  zeta_cat, zeta_an, m, b):
        """Calculate the anion-cation energy, all parameters varied.

        Parameters
        ----------
        i_cat : float
            Ionization energy of the cation.
        a_cat : float
            Electron affinity energy of the cation.
        i_an : float
            Ionization energy of the anion.
        a_an : float
            Electron affinity energy of the anion.
        i_sol : float
            Ionization energy of the solvent.
        a_sol : float
            Electron affinity energy of the solvent.
        gamma_cat : float
            Perturbation on the chemical potential of the cation.
        gamma_an : float
            Perturbation on the chemical potential of the anion.
        zeta_cat : float
            Perturbation on the hardness of the cation.
        zeta_an : float
            Perturbation on the hardness of the anion.
        m : float
            Slope of the linear fit.
        b : float
            Intercept of the linear fit.

        Returns
        -------
        dEsolv_cdft : float
            Anion-cation energy, all parameters varied.
        """
        dE_an_cat_cdft = (SolvationFit.Eab(i_an, a_an, i_sol, a_sol, gamma_an, zeta_an) -
                          SolvationFit.Eab(i_cat, a_cat, i_sol, a_sol, gamma_cat, zeta_cat))
        return m * dE_an_cat_cdft + b

    @staticmethod
    def an_cat_dif_gamma_expl(i_cat, a_cat, i_an, a_an, i_sol, a_sol, gamma_cat, gamma_an, m, b):
        """Calculate the anion-cation energy, gamma and linear parameters varied.

        Parameters
        ----------
        i_cat : float
            Ionization energy of the cation.
        a_cat : float
            Electron affinity energy of the cation.
        i_an : float
            Ionization energy of the anion.
        a_an : float
            Electron affinity energy of the anion.
        i_sol : float
            Ionization energy of the solvent.
        a_sol : float
            Electron affinity energy of the solvent.
        gamma_cat : float
            Perturbation on the chemical potential of the cation.
        gamma_an : float
            Perturbation on the chemical potential of the anion.
        m : float
            Slope of the linear fit.
        b : float
            Intercept of the linear fit.

        Returns
        -------
        dEsolv_cdft : float
            Anion-cation energy, gamma and linear parameters varied.
        """
        dE_an_cat_cdft = (SolvationFit.Eab(i_an, a_an, i_sol, a_sol, gamma_an, zeta=1.0) -
                          SolvationFit.Eab(i_cat, a_cat, i_sol, a_sol, gamma_cat, zeta=1.0))
        return m * dE_an_cat_cdft + b

    @staticmethod
    def an_cat_dif_zeta_expl(i_cat, a_cat, i_an, a_an, i_sol, a_sol, zeta_cat, zeta_an, m, b):
        """Calculate the anion-cation energy, zeta and linear parameters varied.

        Parameters
        ----------
        i_cat : float
            Ionization energy of the cation.
        a_cat : float
            Electron affinity energy of the cation.
        i_an : float
            Ionization energy of the anion.
        a_an : float
            Electron affinity energy of the anion.
        i_sol : float
            Ionization energy of the solvent.
        a_sol : float
            Electron affinity energy of the solvent.
        zeta_cat : float
            Perturbation on the hardness of the cation.
        zeta_an : float
            Perturbation on the hardness of the anion.
        m : float
            Slope of the linear fit.
        b : float
            Intercept of the linear fit.

        Returns
        -------
        dEsolv_cdft : float
            Anion-cation energy, zeta and linear parameters varied.
        """
        dE_an_cat_cdft = (SolvationFit.Eab(i_an, a_an, i_sol, a_sol, gamma=1.0, zeta=zeta_an) -
                          SolvationFit.Eab(i_cat, a_cat, i_sol, a_sol, gamma=1.0, zeta=zeta_cat))
        return m * dE_an_cat_cdft + b

    @staticmethod
    def an_cat_dif_simple_expl(i_cat, a_cat, i_an, a_an, i_sol, a_sol, m, b):
        """Calculate the anion-cation energy, linear parameters varied.

        Parameters
        ----------
        i_cat : float
            Ionization energy of the cation.
        a_cat : float
            Electron affinity energy of the cation.
        i_an : float
            Ionization energy of the anion.
        a_an : float
            Electron affinity energy of the anion.
        i_sol : float
            Ionization energy of the solvent.
        a_sol : float
            Electron affinity energy of the solvent.
        m : float
            Slope of the linear fit.
        b : float
            Intercept of the linear fit.

        Returns
        -------
        dEsolv_cdft : float
            Anion-cation energy, linear parameters varied.
        """
        dE_an_cat_cdft = (SolvationFit.Eab(i_an, a_an, i_sol, a_sol, gamma=1.0, zeta=1.0) -
                          SolvationFit.Eab(i_cat, a_cat, i_sol, a_sol, gamma=1.0, zeta=1.0))
        return m * dE_an_cat_cdft + b

    def fit_data(self):
        """Fit the data to the experimental solvation energies and anion-cation energies."""
        data = (self.i_cat_solv, self.a_cat_solv, self.i_an_solv, self.a_an_solv,
                self.i_sol_solv, self.a_sol_solv)

        try:
            popt_sol_e_allparams_unbound, _ = curve_fit(SolvationFit.sol_e_allaparams, data,
                                                        self.exp_data_solv,
                                                        bounds=(-np.inf, np.inf), maxfev=10000000)
        except TypeError:
            popt_sol_e_allparams_unbound = None

        try:
            popt_sol_e_allparams_bound, _ = curve_fit(SolvationFit.sol_e_allaparams, data,
                                                      self.exp_data_solv, bounds=((self.min_bound,
                                                                                   self.min_bound,
                                                                                   self.min_bound,
                                                                                   self.min_bound,
                                                                                   -np.inf,
                                                                                   -np.inf),
                                                                                  (self.max_bound,
                                                                                   self.max_bound,
                                                                                   self.max_bound,
                                                                                   self.max_bound,
                                                                                   np.inf, np.inf)),
                                                      maxfev=10000000)
        except TypeError:
            popt_sol_e_allparams_bound = None

        try:
            popt_sol_e_gamma_unbound, _ = curve_fit(SolvationFit.sol_e_gamma, data,
                                                    self.exp_data_solv, bounds=(-np.inf, np.inf),
                                                    maxfev=10000000)
        except TypeError:
            popt_sol_e_gamma_unbound = None

        try:
            popt_sol_e_gamma_bound, _ = curve_fit(SolvationFit.sol_e_gamma, data,
                                                  self.exp_data_solv, bounds=((self.min_bound,
                                                                               self.min_bound,
                                                                               -np.inf, -np.inf),
                                                                              (self.max_bound,
                                                                               self.max_bound,
                                                                               np.inf, np.inf)),
                                                  maxfev=10000000)
        except TypeError:
            popt_sol_e_gamma_bound = None

        try:
            popt_sol_e_zeta_unbound, _ = curve_fit(SolvationFit.sol_e_zeta, data,
                                                   self.exp_data_solv, bounds=(-np.inf, np.inf),
                                                   maxfev=10000000)
        except TypeError:
            popt_sol_e_zeta_unbound = None

        try:
            popt_sol_e_zeta_bound, _ = curve_fit(SolvationFit.sol_e_zeta, data,
                                                 self.exp_data_solv, bounds=((self.min_bound,
                                                                              self.min_bound,
                                                                              -np.inf, -np.inf),
                                                                             (self.max_bound,
                                                                              self.max_bound,
                                                                              np.inf, np.inf)),
                                                 maxfev=10000000)
        except TypeError:
            popt_sol_e_zeta_bound = None

        try:
            popt_sol_e_simple_unbound, _ = curve_fit(SolvationFit.sol_e_simple, data,
                                                     self.exp_data_solv, bounds=(-np.inf, np.inf),
                                                     maxfev=10000000)
        except TypeError:
            popt_sol_e_simple_unbound = None

        data = (self.i_cat_diff, self.a_cat_diff, self.i_an_diff, self.a_an_diff,
                self.i_sol_diff, self.a_sol_diff)

        try:
            popt_an_cat_diff_allparams_unbound, _ = curve_fit(SolvationFit.an_cat_dif_allparams,
                                                              data, self.exp_data_diff,
                                                              bounds=(-np.inf, np.inf),
                                                              maxfev=10000000)
        except TypeError:
            popt_an_cat_diff_allparams_unbound = None

        try:
            popt_an_cat_diff_allparams_bound, _ = curve_fit(SolvationFit.an_cat_dif_allparams, data,
                                                            self.exp_data_diff,
                                                            bounds=((self.min_bound,
                                                                     self.min_bound,
                                                                     self.min_bound,
                                                                     self.min_bound,
                                                                     -np.inf, -np.inf),
                                                                    (self.max_bound,
                                                                     self.max_bound,
                                                                     self.max_bound,
                                                                     self.max_bound,
                                                                     np.inf, np.inf)),
                                                            maxfev=10000000)
        except TypeError:
            popt_an_cat_diff_allparams_bound = None

        try:
            popt_an_cat_diff_gamma_unbound, _ = curve_fit(SolvationFit.an_cat_dif_gamma, data,
                                                          self.exp_data_diff, bounds=(-np.inf,
                                                                                      np.inf),
                                                          maxfev=10000000)
        except TypeError:
            popt_an_cat_diff_gamma_unbound = None

        try:
            popt_an_cat_diff_gamma_bound, _ = curve_fit(SolvationFit.an_cat_dif_gamma, data,
                                                        self.exp_data_diff,
                                                        bounds=((self.min_bound, self.min_bound,
                                                                 -np.inf, -np.inf),
                                                                (self.max_bound, self.max_bound,
                                                                 np.inf, np.inf)),
                                                        maxfev=10000000)
        except TypeError:
            popt_an_cat_diff_gamma_bound = None

        try:
            popt_an_cat_diff_zeta_unbound, _ = curve_fit(SolvationFit.an_cat_dif_zeta, data,
                                                         self.exp_data_diff, bounds=(-np.inf,
                                                                                     np.inf),
                                                         maxfev=10000000)
        except TypeError:
            popt_an_cat_diff_zeta_unbound = None

        try:
            popt_an_cat_diff_zeta_bound, _ = curve_fit(SolvationFit.an_cat_dif_zeta, data,
                                                       self.exp_data_diff,
                                                       bounds=((self.min_bound, self.min_bound,
                                                                -np.inf, -np.inf),
                                                               (self.max_bound, self.max_bound,
                                                                np.inf, np.inf)),
                                                       maxfev=10000000)
        except TypeError:
            popt_an_cat_diff_zeta_bound = None

        try:
            popt_an_cat_diff_simple_unbound, _ = curve_fit(SolvationFit.an_cat_dif_simple, data,
                                                           self.exp_data_diff, bounds=(-np.inf,
                                                                                       np.inf),
                                                           maxfev=10000000)
        except TypeError:
            popt_an_cat_diff_simple_unbound = None

        if popt_sol_e_allparams_unbound is not None:
            pass
        if popt_sol_e_allparams_bound is not None:
            self.popt_sol_e_allparams_bound = popt_sol_e_allparams_bound
        if popt_sol_e_gamma_unbound is not None:
            pass
        if popt_sol_e_gamma_bound is not None:
            self.popt_sol_e_gamma_bound = popt_sol_e_gamma_bound
        if popt_sol_e_zeta_unbound is not None:
            pass
        if popt_sol_e_zeta_bound is not None:
            self.popt_sol_e_zeta_bound = popt_sol_e_zeta_bound
        if popt_sol_e_simple_unbound is not None:
            self.popt_sol_e_simple_unbound = popt_sol_e_simple_unbound

        if popt_an_cat_diff_allparams_unbound is not None:
            pass
        if popt_an_cat_diff_allparams_bound is not None:
            self.popt_an_cat_diff_allparams_bound = popt_an_cat_diff_allparams_bound
        if popt_an_cat_diff_gamma_unbound is not None:
            pass
        if popt_an_cat_diff_gamma_bound is not None:
            self.popt_an_cat_diff_gamma_bound = popt_an_cat_diff_gamma_bound
        if popt_an_cat_diff_zeta_unbound is not None:
            pass
        if popt_an_cat_diff_zeta_bound is not None:
            self.popt_an_cat_diff_zeta_bound = popt_an_cat_diff_zeta_bound
        if popt_an_cat_diff_simple_unbound is not None:
            self.popt_an_cat_diff_simple_unbound = popt_an_cat_diff_simple_unbound

    def get_results_errors(self):
        """Calculate the CDFT solvation energies and anion-cation energies and errors.

        Notes
        -----
        Errors calculated as: CDFT_value - Experimental_value.
        """
        if hasattr(self, 'popt_sol_e_allparams_unbound'):
            sol_e_allparams_unbound_result = []
            sol_e_allparams_unbound_error = []
            gamma_cat, gamma_an, zeta_cat, zeta_an, m, b = self.popt_sol_e_allparams_unbound
            for index, (i_c, a_c, i_a, a_a, i_s, a_s) in enumerate(
                zip(self.i_cat_solv, self.a_cat_solv, self.i_an_solv, self.a_an_solv,
                    self.i_sol_solv, self.a_sol_solv)):
                dE = SolvationFit.sol_e_allaparams_expl(i_c, a_c, i_a, a_a, i_s, a_s, gamma_cat,
                                                        gamma_an, zeta_cat, zeta_an, m, b)
                sol_e_allparams_unbound_result.append(dE)
                sol_e_allparams_unbound_error.append(self.exp_data_solv[index] - dE)
            self.sol_e_allparams_unbound_result = np.array(sol_e_allparams_unbound_result)
            self.sol_e_allparams_unbound_error = np.array(sol_e_allparams_unbound_error)

        if hasattr(self, 'popt_sol_e_allparams_bound'):
            sol_e_allparams_bound_result = []
            sol_e_allparams_bound_error = []
            gamma_cat, gamma_an, zeta_cat, zeta_an, m, b = self.popt_sol_e_allparams_bound
            for index, (i_c, a_c, i_a, a_a, i_s, a_s) in enumerate(
                    zip(self.i_cat_solv, self.a_cat_solv, self.i_an_solv, self.a_an_solv,
                        self.i_sol_solv, self.a_sol_solv)):
                dE = SolvationFit.sol_e_allaparams_expl(i_c, a_c, i_a, a_a, i_s, a_s, gamma_cat,
                                                         gamma_an, zeta_cat, zeta_an, m, b)
                sol_e_allparams_bound_result.append(dE)
                sol_e_allparams_bound_error.append(self.exp_data_solv[index] - dE)
            self.sol_e_allparams_bound_result = np.array(sol_e_allparams_bound_result)
            self.sol_e_allparams_bound_error = np.array(sol_e_allparams_bound_error)

        if hasattr(self, 'popt_sol_e_gamma_unbound'):
            sol_e_gamma_unbound_result = []
            sol_e_gamma_unbound_error = []
            gamma_cat, gamma_an, m, b = self.popt_sol_e_gamma_unbound
            for index, (i_c, a_c, i_a, a_a, i_s, a_s) in enumerate(
                    zip(self.i_cat_solv, self.a_cat_solv, self.i_an_solv, self.a_an_solv,
                        self.i_sol_solv, self.a_sol_solv)):
                dE = SolvationFit.sol_e_gamma_expl(i_c, a_c, i_a, a_a, i_s, a_s, gamma_cat,
                                                   gamma_an, m, b)
                sol_e_gamma_unbound_result.append(dE)
                sol_e_gamma_unbound_error.append(self.exp_data_solv[index] - dE)
            self.sol_e_gamma_unbound_result = np.array(sol_e_gamma_unbound_result)
            self.sol_e_gamma_unbound_error = np.array(sol_e_gamma_unbound_error)

        if hasattr(self, 'popt_sol_e_gamma_bound'):
            sol_e_gamma_bound_result = []
            sol_e_gamma_bound_error = []
            gamma_cat, gamma_an, m, b = self.popt_sol_e_gamma_bound
            for index, (i_c, a_c, i_a, a_a, i_s, a_s) in enumerate(
                    zip(self.i_cat_solv, self.a_cat_solv, self.i_an_solv, self.a_an_solv,
                        self.i_sol_solv, self.a_sol_solv)):
                dE = SolvationFit.sol_e_gamma_expl(i_c, a_c, i_a, a_a, i_s, a_s, gamma_cat,
                                                   gamma_an, m, b)
                sol_e_gamma_bound_result.append(dE)
                sol_e_gamma_bound_error.append(self.exp_data_solv[index] - dE)
            self.sol_e_gamma_bound_result = np.array(sol_e_gamma_bound_result)
            self.sol_e_gamma_bound_error = np.array(sol_e_gamma_bound_error)

        if hasattr(self, 'popt_sol_e_zeta_unbound'):
            sol_e_zeta_unbound_result = []
            sol_e_zeta_unbound_error = []
            zeta_cat, zeta_an, m, b = self.popt_sol_e_zeta_unbound
            for index, (i_c, a_c, i_a, a_a, i_s, a_s) in enumerate(
                    zip(self.i_cat_solv, self.a_cat_solv, self.i_an_solv, self.a_an_solv,
                        self.i_sol_solv, self.a_sol_solv)):
                dE = SolvationFit.sol_e_zeta_expl(i_c, a_c, i_a, a_a, i_s, a_s, zeta_cat,
                                                  zeta_an, m, b)
                sol_e_zeta_unbound_result.append(dE)
                sol_e_zeta_unbound_error.append(self.exp_data_solv[index] - dE)
            self.sol_e_zeta_unbound_result = np.array(sol_e_zeta_unbound_result)
            self.sol_e_zeta_unbound_error = np.array(sol_e_zeta_unbound_error)

        if hasattr(self, 'popt_sol_e_zeta_bound'):
            sol_e_zeta_bound_result = []
            sol_e_zeta_bound_error = []
            zeta_cat, zeta_an, m, b = self.popt_sol_e_zeta_bound
            for index, (i_c, a_c, i_a, a_a, i_s, a_s) in enumerate(
                    zip(self.i_cat_solv, self.a_cat_solv, self.i_an_solv, self.a_an_solv,
                        self.i_sol_solv, self.a_sol_solv)):
                dE = SolvationFit.sol_e_zeta_expl(i_c, a_c, i_a, a_a, i_s, a_s, zeta_cat,
                                                  zeta_an, m, b)
                sol_e_zeta_bound_result.append(dE)
                sol_e_zeta_bound_error.append(self.exp_data_solv[index] - dE)
            self.sol_e_zeta_bound_result = np.array(sol_e_zeta_bound_result)
            self.sol_e_zeta_bound_error = np.array(sol_e_zeta_bound_error)

        if hasattr(self, 'popt_sol_e_simple_unbound'):
            sol_e_simple_unbound_result = []
            sol_e_simple_unbound_error = []
            m, b = self.popt_sol_e_simple_unbound
            for index, (i_c, a_c, i_a, a_a, i_s, a_s) in enumerate(
                    zip(self.i_cat_solv, self.a_cat_solv, self.i_an_solv, self.a_an_solv,
                        self.i_sol_solv, self.a_sol_solv)):
                dE = SolvationFit.sol_e_simple_expl(i_c, a_c, i_a, a_a, i_s, a_s, m, b)
                sol_e_simple_unbound_result.append(dE)
                sol_e_simple_unbound_error.append(self.exp_data_solv[index] - dE)
            self.sol_e_simple_unbound_result = np.array(sol_e_simple_unbound_result)
            self.sol_e_simple_unbound_error = np.array(sol_e_simple_unbound_error)

        if hasattr(self, 'popt_an_cat_diff_allparams_unbound'):
            an_cat_diff_allparams_unbound_result = []
            an_cat_diff_allparams_unbound_error = []
            gamma_cat, gamma_an, zeta_cat, zeta_an, m, b = self.popt_an_cat_diff_allparams_unbound
            for index, (i_c, a_c, i_a, a_a, i_s, a_s) in enumerate(
                    zip(self.i_cat_diff, self.a_cat_diff, self.i_an_diff, self.a_an_diff,
                        self.i_sol_diff, self.a_sol_diff)):
                dE = SolvationFit.an_cat_dif_allparams_expl(i_c, a_c, i_a, a_a, i_s, a_s, gamma_cat,
                                                            gamma_an, zeta_cat, zeta_an, m, b)
                an_cat_diff_allparams_unbound_result.append(dE)
                an_cat_diff_allparams_unbound_error.append(self.exp_data_diff[index] - dE)
            self.an_cat_diff_allparams_unbound_result = np.array(
                an_cat_diff_allparams_unbound_result)
            self.an_cat_diff_allparams_unbound_error = np.array(
                an_cat_diff_allparams_unbound_error)

        if hasattr(self, 'popt_an_cat_diff_allparams_bound'):
            an_cat_diff_allparams_bound_result = []
            an_cat_diff_allparams_bound_error = []
            gamma_cat, gamma_an, zeta_cat, zeta_an, m, b = self.popt_an_cat_diff_allparams_bound
            for index, (i_c, a_c, i_a, a_a, i_s, a_s) in enumerate(
                    zip(self.i_cat_diff, self.a_cat_diff, self.i_an_diff, self.a_an_diff,
                        self.i_sol_diff, self.a_sol_diff)):
                dE = SolvationFit.an_cat_dif_allparams_expl(i_c, a_c, i_a, a_a, i_s, a_s, gamma_cat,
                                                            gamma_an, zeta_cat, zeta_an, m, b)
                an_cat_diff_allparams_bound_result.append(dE)
                an_cat_diff_allparams_bound_error.append(self.exp_data_diff[index] - dE)
            self.an_cat_diff_allparams_bound_result = np.array(an_cat_diff_allparams_bound_result)
            self.an_cat_diff_allparams_bound_error = np.array(an_cat_diff_allparams_bound_error)

        if hasattr(self, 'popt_an_cat_diff_gamma_unbound'):
            an_cat_diff_gamma_unbound_result = []
            an_cat_diff_gamma_unbound_error = []
            gamma_cat, gamma_an, m, b = self.popt_an_cat_diff_gamma_unbound
            for index, (i_c, a_c, i_a, a_a, i_s, a_s) in enumerate(
                    zip(self.i_cat_diff, self.a_cat_diff, self.i_an_diff, self.a_an_diff,
                        self.i_sol_diff, self.a_sol_diff)):
                dE = SolvationFit.an_cat_dif_gamma_expl(i_c, a_c, i_a, a_a, i_s, a_s, gamma_cat,
                                                        gamma_an, m, b)
                an_cat_diff_gamma_unbound_result.append(dE)
                an_cat_diff_gamma_unbound_error.append(self.exp_data_diff[index] - dE)
            self.an_cat_diff_gamma_unbound_result = np.array(an_cat_diff_gamma_unbound_result)
            self.an_cat_diff_gamma_unbound_error = np.array(an_cat_diff_gamma_unbound_error)

        if hasattr(self, 'popt_an_cat_diff_gamma_bound'):
            an_cat_diff_gamma_bound_result = []
            an_cat_diff_gamma_bound_error = []
            gamma_cat, gamma_an, m, b = self.popt_an_cat_diff_gamma_bound
            for index, (i_c, a_c, i_a, a_a, i_s, a_s) in enumerate(
                    zip(self.i_cat_diff, self.a_cat_diff, self.i_an_diff, self.a_an_diff,
                        self.i_sol_diff, self.a_sol_diff)):
                dE = SolvationFit.an_cat_dif_gamma_expl(i_c, a_c, i_a, a_a, i_s, a_s, gamma_cat,
                                                        gamma_an, m, b)
                an_cat_diff_gamma_bound_result.append(dE)
                an_cat_diff_gamma_bound_error.append(self.exp_data_diff[index] - dE)
            self.an_cat_diff_gamma_bound_result = np.array(an_cat_diff_gamma_bound_result)
            self.an_cat_diff_gamma_bound_error = np.array(an_cat_diff_gamma_bound_error)

        if hasattr(self, 'popt_an_cat_diff_zeta_unbound'):
            an_cat_diff_zeta_unbound_result = []
            an_cat_diff_zeta_unbound_error = []
            zeta_cat, zeta_an, m, b = self.popt_an_cat_diff_zeta_unbound
            for index, (i_c, a_c, i_a, a_a, i_s, a_s) in enumerate(
                    zip(self.i_cat_diff, self.a_cat_diff, self.i_an_diff, self.a_an_diff,
                        self.i_sol_diff, self.a_sol_diff)):
                dE = SolvationFit.an_cat_dif_zeta_expl(i_c, a_c, i_a, a_a, i_s, a_s, zeta_cat,
                                                       zeta_an, m, b)
                an_cat_diff_zeta_unbound_result.append(dE)
                an_cat_diff_zeta_unbound_error.append(self.exp_data_diff[index] - dE)
            self.an_cat_diff_zeta_unbound_result = np.array(an_cat_diff_zeta_unbound_result)
            self.an_cat_diff_zeta_unbound_error = np.array(an_cat_diff_zeta_unbound_error)

        if hasattr(self, 'popt_an_cat_diff_zeta_bound'):
            an_cat_diff_zeta_bound_result = []
            an_cat_diff_zeta_bound_error = []
            zeta_cat, zeta_an, m, b = self.popt_an_cat_diff_zeta_bound
            for index, (i_c, a_c, i_a, a_a, i_s, a_s) in enumerate(
                    zip(self.i_cat_diff, self.a_cat_diff, self.i_an_diff, self.a_an_diff,
                         self.i_sol_diff, self.a_sol_diff)):
                dE = SolvationFit.an_cat_dif_zeta_expl(i_c, a_c, i_a, a_a, i_s, a_s, zeta_cat,
                                                       zeta_an, m, b)
                an_cat_diff_zeta_bound_result.append(dE)
                an_cat_diff_zeta_bound_error.append(self.exp_data_diff[index] - dE)
            self.an_cat_diff_zeta_bound_result = np.array(an_cat_diff_zeta_bound_result)
            self.an_cat_diff_zeta_bound_error = np.array(an_cat_diff_zeta_bound_error)

        if hasattr(self, 'popt_an_cat_diff_simple_unbound'):
            an_cat_diff_simple_unbound_result = []
            an_cat_diff_simple_unbound_error = []
            m, b = self.popt_an_cat_diff_simple_unbound
            for index, (i_c, a_c, i_a, a_a, i_s, a_s) in enumerate(
                    zip(self.i_cat_diff, self.a_cat_diff, self.i_an_diff, self.a_an_diff,
                        self.i_sol_diff, self.a_sol_diff)):
                dE = SolvationFit.an_cat_dif_simple_expl(i_c, a_c, i_a, a_a, i_s, a_s, m, b)
                an_cat_diff_simple_unbound_result.append(dE)
                an_cat_diff_simple_unbound_error.append(self.exp_data_diff[index] - dE)
            self.an_cat_diff_simple_unbound_result = np.array(an_cat_diff_simple_unbound_result)
            self.an_cat_diff_simple_unbound_error = np.array(an_cat_diff_simple_unbound_error)
