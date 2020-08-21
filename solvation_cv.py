import numpy as np
from solvation_fit import SolvationFit


class SolvationCV(SolvationFit):
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
        Numpy array that contains the training I values for the cations that will be used
        in the solvation energy calculations.
    a_cat_solv : np.ndarray
        Numpy array that contains the training A values for the cations that will be used
        in the solvation energy calculations.
    i_an_solv : np.ndarray
        Numpy array that contains the training I values for the anions that will be used
        in the solvation energy calculations.
    a_an_solv : np.ndarray
        Numpy array that contains the training A values for the anions that will be used
        in the solvation energy calculations.
    i_sol_solv : np.ndarray
        Numpy array that contains the training I values for the solvents that will be used
        in the solvation energy calculations.
    a_sol_solv : np.ndarray
        Numpy array that contains the training A values for the solvents that will be used
        in the solvation energy calculations.
    i_cat_diff : np.ndarray
        Numpy array that contains the training I values for the cations that will be used
        in the anion-cation energy calculations.
    a_cat_diff : np.ndarray
        Numpy array that contains the training A values for the cations that will be used
        in the anion-cation energy calculations.
    i_an_diff : np.ndarray
        Numpy array that contains the training I values for the anions that will be used
        in the anion-cation energy calculations.
    a_an_diff : np.ndarray
        Numpy array that contains the training A values for the anions that will be used
        in the anion-cation energy calculations.
    i_sol_diff : np.ndarray
        Numpy array that contains the training I values for the solvents that will be used
        in the anion-cation energy calculations.
    a_sol_diff : np.ndarray
        Numpy array that contains the training A values for the solvents that will be used
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
        Experimantal solvation energies (training set).
    exp_data_solv : np.ndarray
        Experimental anion-cation energies (training set).
    i_cat_solv_total : np.ndarray
        Numpy array that contains the available I values for the cations that will be used
        in the solvation energy calculations.
    a_cat_solv_total : np.ndarray
        Numpy array that contains the available A values for the cations that will be used
        in the solvation energy calculations.
    i_an_solv_total : np.ndarray
        Numpy array that contains the available I values for the anions that will be used
        in the solvation energy calculations.
    a_an_solv_total : np.ndarray
        Numpy array that contains the available A values for the anions that will be used
        in the solvation energy calculations.
    i_sol_solv_total : np.ndarray
        Numpy array that contains the available I values for the solvents that will be used
        in the solvation energy calculations.
    a_sol_solv_total : np.ndarray
        Numpy array that contains the available A values for the solvents that will be used
        in the solvation energy calculations.
    i_cat_diff_total : np.ndarray
        Numpy array that contains the available I values for the cations that will be used
        in the anion-cation energy calculations.
    a_cat_diff_total : np.ndarray
        Numpy array that contains the available A values for the cations that will be used
        in the anion-cation energy calculations.
    i_an_diff_total : np.ndarray
        Numpy array that contains the available I values for the anions that will be used
        in the anion-cation energy calculations.
    a_an_diff_total : np.ndarray
        Numpy array that contains the available A values for the anions that will be used
        in the anion-cation energy calculations.
    i_sol_diff_total : np.ndarray
        Numpy array that contains the available I values for the solvents that will be used
        in the anion-cation energy calculations.
    a_sol_diff_total : np.ndarray
        Numpy array that contains the available A values for the solvents that will be used
        in the anion-cation energy calculations.
    exp_data_solv_total : np.ndarray
        Experimental solvation energies.
    exp_data_solv_total : np.ndarray
        Experimental anion-cation energies.
    i_cat_solv_test : np.ndarray
        Numpy array that contains the test I values for the cations that will be used
        in the solvation energy calculations.
    a_cat_solv_test : np.ndarray
        Numpy array that contains the test A values for the cations that will be used
        in the solvation energy calculations.
    i_an_solv_test : np.ndarray
        Numpy array that contains the test I values for the anions that will be used
        in the solvation energy calculations.
    a_an_solv_test : np.ndarray
        Numpy array that contains the test A values for the anions that will be used
        in the solvation energy calculations.
    i_sol_solv_test : np.ndarray
        Numpy array that contains the test I values for the solvents that will be used
        in the solvation energy calculations.
    a_sol_solv_test : np.ndarray
        Numpy array that contains the test A values for the solvents that will be used
        in the solvation energy calculations.
    i_cat_diff_test : np.ndarray
        Numpy array that contains the test I values for the cations that will be used
        in the anion-cation energy calculations.
    a_cat_diff_test : np.ndarray
        Numpy array that contains the test A values for the cations that will be used
        in the anion-cation energy calculations.
    i_an_diff_test : np.ndarray
        Numpy array that contains the test I values for the anions that will be used
        in the anion-cation energy calculations.
    a_an_diff_test : np.ndarray
        Numpy array that contains the test A values for the anions that will be used
        in the anion-cation energy calculations.
    i_sol_diff_test : np.ndarray
        Numpy array that contains the test I values for the solvents that will be used
        in the anion-cation energy calculations.
    a_sol_diff_test : np.ndarray
        Numpy array that contains the test A values for the solvents that will be used
        in the anion-cation energy calculations.
    exp_data_solv_test : np.ndarray
        Experimental solvation energies (test set).
    exp_data_solv_test : np.ndarray
        Experimental anion-cation energies (test set).
    sol_e_allparams_bound_cv : np.ndarray (if it can be calculated)
        Numpy array with the cv errors of the solvation energy calculations.
        Bound parameters. All parameters varied.
    sol_e_gamma_unbound_cv : np.ndarray (if it can be calculated)
        Numpy array with the cv errors of the solvation energy calculations.
        Unbound parameters. Gamma and linear parameters varied.
    sol_e_gamma_bound_cv : np.ndarray (if it can be calculated)
        Numpy array with the cv errors of the solvation energy calculations.
        Bound parameters. Gamma and linear parameters varied.
    sol_e_zeta_unbound_cv : np.ndarray (if it can be calculated)
        Numpy array with the cv errors of the solvation energy calculations.
        Unbound parameters. Zeta and linear parameters varied.
    sol_e_zeta_bound_cv : np.ndarray (if it can be calculated)
        Numpy array with the cv errors of the solvation energy calculations.
        Bound parameters. Zeta and linear parameters varied.
    sol_e_simple_unbound_cv : np.ndarray (if it can be calculated)
        Numpy array with the cv errors of the solvation energy calculations.
        Unbound parameters. Linear parameters varied.
    an_cat_diff_allparams_unbound_cv : np.ndarray (if it can be calculated)
        Numpy array with the cv errors of the anion-cation energy calculations.
        Unbound parameters. All parameters varied.
    an_cat_diff_allparams_bound_cv : np.ndarray (if it can be calculated)
        Numpy array with the cv errors of the anion-cation energy calculations.
        Bound parameters. All parameters varied.
    an_cat_diff_gamma_unbound_cv : np.ndarray (if it can be calculated)
        Numpy array with the cv errors of the anion-cation energy calculations.
        Unbound parameters. Gamma and linear parameters varied.
    an_cat_diff_gamma_bound_cv : np.ndarray (if it can be calculated)
        Numpy array with the cv errors of the anion-cation energy calculations.
        Bound parameters. Gamma and linear parameters varied.
    an_cat_diff_zeta_unbound_cv : np.ndarray (if it can be calculated)
        Numpy array with the cv errors of the anion-cation energy calculations.
        Unbound parameters. Zeta and linear parameters varied.
    an_cat_diff_zeta_bound_cv : np.ndarray (if it can be calculated)
        Numpy array with the cv errors of the anion-cation energy calculations.
        Bound parameters. Zeta and linear parameters varied.
    an_cat_diff_simple_unbound_cv : np.ndarray (if it can be calculated)
        Numpy array with the cv errors of the anion-cation energy calculations.
        Unbound parameters. Linear parameters varied.

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

    def __init__(self, solv_object, solv_test_indices, diff_test_indices):
        """Initialize the object.

        Parameters
        ----------
        solv_object : SolvationFit
            SolvationFit instance.
        solv_test_indices : list
            List with indices that will be used as test set in the CV for the solvation energy.
        diff_test_indices : list
            List with indices that will be used as test set in the CV for the
            anion_cat calculations.
        """
        self.calc_type = solv_object.calc_type
        self.min_bound = solv_object.min_bound
        self.max_bound = solv_object.max_bound

        # Copy selected data
        self.a_cat_solv_total = solv_object.a_cat_solv
        self.i_an_solv_total = solv_object.i_an_solv
        self.i_cat_solv_total = solv_object.i_cat_solv
        self.a_an_solv_total = solv_object.a_an_solv
        self.i_sol_solv_total = solv_object.i_sol_solv
        self.a_sol_solv_total = solv_object.a_sol_solv
        self.exp_data_solv_total = solv_object.exp_data_solv
        self.i_cat_diff_total = solv_object.i_cat_diff 
        self.a_cat_diff_total = solv_object.a_cat_diff
        self.i_an_diff_total = solv_object.i_an_diff
        self.a_an_diff_total = solv_object.a_an_diff
        self.i_sol_diff_total = solv_object.i_sol_diff
        self.a_sol_diff_total = solv_object.a_sol_diff
        self.exp_data_diff_total = solv_object.exp_data_diff
        
        self.solv_test_indices = solv_test_indices
        self.diff_test_indices = diff_test_indices
        self.separate_data_solv()
        self.separate_data_diff()
        self.fit_data()
        self.get_cv_results()

    def separate_data_solv(self):
        """Separate the solvation data into training and test sets."""
        training_indices = [i for i in range(len(self.i_cat_solv_total))
                            if i not in self.solv_test_indices]
        self.i_cat_solv = self.i_cat_solv_total[training_indices]
        self.a_cat_solv = self.a_cat_solv_total[training_indices]
        self.i_an_solv = self.i_an_solv_total[training_indices]
        self.a_an_solv = self.a_an_solv_total[training_indices]
        self.i_sol_solv = self.i_sol_solv_total[training_indices]
        self.a_sol_solv = self.a_sol_solv_total[training_indices]
        self.exp_data_solv = self.exp_data_solv_total[training_indices]
        self.i_cat_solv_test = self.i_cat_solv_total[self.solv_test_indices]
        self.a_cat_solv_test = self.a_cat_solv_total[self.solv_test_indices]
        self.i_an_solv_test = self.i_an_solv_total[self.solv_test_indices]
        self.a_an_solv_test = self.a_an_solv_total[self.solv_test_indices]
        self.i_sol_solv_test = self.i_sol_solv_total[self.solv_test_indices]
        self.a_sol_solv_test = self.a_sol_solv_total[self.solv_test_indices]
        self.exp_data_solv_test = self.exp_data_solv_total[self.solv_test_indices]
    
    def separate_data_diff(self):
        """Separate the anion_cation data into training and test sets."""
        training_indices = [i for i in range(len(self.i_cat_diff_total))
                            if i not in self.diff_test_indices]
        self.i_cat_diff = self.i_cat_diff_total[training_indices]
        self.a_cat_diff = self.a_cat_diff_total[training_indices]
        self.i_an_diff = self.i_an_diff_total[training_indices]
        self.a_an_diff = self.a_an_diff_total[training_indices]
        self.i_sol_diff = self.i_sol_diff_total[training_indices]
        self.a_sol_diff = self.a_sol_diff_total[training_indices]
        self.exp_data_diff = self.exp_data_diff_total[training_indices]
        self.i_cat_diff_test = self.i_cat_diff_total[self.diff_test_indices]
        self.a_cat_diff_test = self.a_cat_diff_total[self.diff_test_indices]
        self.i_an_diff_test = self.i_an_diff_total[self.diff_test_indices]
        self.a_an_diff_test = self.a_an_diff_total[self.diff_test_indices]
        self.i_sol_diff_test = self.i_sol_diff_total[self.diff_test_indices]
        self.a_sol_diff_test = self.a_sol_diff_total[self.diff_test_indices]
        self.exp_data_diff_test = self.exp_data_diff_total[self.diff_test_indices]

    def get_cv_results(self):
        """Calculate the CDFT solvation energies and anion-cation energies and errors.

        Notes
        -----
        Errors calculated as: CDFT_value - Experimental_value.
        """
        if hasattr(self, 'popt_sol_e_allparams_unbound'):
            sol_e_allparams_unbound_error = []
            gamma_cat, gamma_an, zeta_cat, zeta_an, m, b = self.popt_sol_e_allparams_unbound
            for index, (i_c, a_c, i_a, a_a, i_s, a_s) in enumerate(
                zip(self.i_cat_solv_test, self.a_cat_solv_test,
                    self.i_an_solv_test, self.a_an_solv_test,
                    self.i_sol_solv_test, self.a_sol_solv_test)):
                dE = SolvationFit.sol_e_allaparams_expl(i_c, a_c, i_a, a_a, i_s, a_s, gamma_cat,
                                                        gamma_an, zeta_cat, zeta_an, m, b)
                sol_e_allparams_unbound_error.append(self.exp_data_solv_test[index] - dE)
            self.sol_e_allparams_unbound_error = np.array(sol_e_allparams_unbound_error)
            self.sol_e_allparams_unbound_cv = np.sqrt(np.mean(self.sol_e_allparams_unbound_error**2))

        if hasattr(self, 'popt_sol_e_allparams_bound'):
            sol_e_allparams_bound_error = []
            gamma_cat, gamma_an, zeta_cat, zeta_an, m, b = self.popt_sol_e_allparams_bound
            for index, (i_c, a_c, i_a, a_a, i_s, a_s) in enumerate(
                    zip(self.i_cat_solv_test, self.a_cat_solv_test,
                        self.i_an_solv_test, self.a_an_solv_test,
                        self.i_sol_solv_test, self.a_sol_solv_test)):
                dE = SolvationFit.sol_e_allaparams_expl(i_c, a_c, i_a, a_a, i_s, a_s, gamma_cat,
                                                         gamma_an, zeta_cat, zeta_an, m, b)
                sol_e_allparams_bound_error.append(self.exp_data_solv_test[index] - dE)
            self.sol_e_allparams_bound_error = np.array(sol_e_allparams_bound_error)
            self.sol_e_allparams_bound_cv = np.sqrt(np.mean(self.sol_e_allparams_bound_error**2))

        if hasattr(self, 'popt_sol_e_gamma_unbound'):
            sol_e_gamma_unbound_error = []
            gamma_cat, gamma_an, m, b = self.popt_sol_e_gamma_unbound
            for index, (i_c, a_c, i_a, a_a, i_s, a_s) in enumerate(
                    zip(self.i_cat_solv_test, self.a_cat_solv_test,
                        self.i_an_solv_test, self.a_an_solv_test,
                        self.i_sol_solv_test, self.a_sol_solv_test)):
                dE = SolvationFit.sol_e_gamma_expl(i_c, a_c, i_a, a_a, i_s, a_s, gamma_cat,
                                                   gamma_an, m, b)
                sol_e_gamma_unbound_error.append(self.exp_data_solv_test[index] - dE)
            self.sol_e_gamma_unbound_error = np.array(sol_e_gamma_unbound_error)
            self.sol_e_gamma_unbound_cv = np.sqrt(np.mean(self.sol_e_gamma_unbound_error**2))

        if hasattr(self, 'popt_sol_e_gamma_bound'):
            sol_e_gamma_bound_error = []
            gamma_cat, gamma_an, m, b = self.popt_sol_e_gamma_bound
            for index, (i_c, a_c, i_a, a_a, i_s, a_s) in enumerate(
                    zip(self.i_cat_solv_test, self.a_cat_solv_test,
                        self.i_an_solv_test, self.a_an_solv_test,
                        self.i_sol_solv_test, self.a_sol_solv_test)):
                dE = SolvationFit.sol_e_gamma_expl(i_c, a_c, i_a, a_a, i_s, a_s, gamma_cat,
                                                   gamma_an, m, b)
                sol_e_gamma_bound_error.append(self.exp_data_solv_test[index] - dE)
            self.sol_e_gamma_bound_error = np.array(sol_e_gamma_bound_error)
            self.sol_e_gamma_bound_cv = np.sqrt(np.mean(self.sol_e_gamma_bound_error**2))

        if hasattr(self, 'popt_sol_e_zeta_unbound'):
            sol_e_zeta_unbound_error = []
            zeta_cat, zeta_an, m, b = self.popt_sol_e_zeta_unbound
            for index, (i_c, a_c, i_a, a_a, i_s, a_s) in enumerate(
                    zip(self.i_cat_solv_test, self.a_cat_solv_test,
                        self.i_an_solv_test, self.a_an_solv_test,
                        self.i_sol_solv_test, self.a_sol_solv_test)):
                dE = SolvationFit.sol_e_zeta_expl(i_c, a_c, i_a, a_a, i_s, a_s, zeta_cat,
                                                  zeta_an, m, b)
                sol_e_zeta_unbound_error.append(self.exp_data_solv_test[index] - dE)
            self.sol_e_zeta_unbound_error = np.array(sol_e_zeta_unbound_error)
            self.sol_e_zeta_unbound_cv = np.sqrt(np.mean(self.sol_e_zeta_unbound_error**2))

        if hasattr(self, 'popt_sol_e_zeta_bound'):
            sol_e_zeta_bound_error = []
            zeta_cat, zeta_an, m, b = self.popt_sol_e_zeta_bound
            for index, (i_c, a_c, i_a, a_a, i_s, a_s) in enumerate(
                    zip(self.i_cat_solv_test, self.a_cat_solv_test,
                        self.i_an_solv_test, self.a_an_solv_test,
                        self.i_sol_solv_test, self.a_sol_solv_test)):
                dE = SolvationFit.sol_e_zeta_expl(i_c, a_c, i_a, a_a, i_s, a_s, zeta_cat,
                                                  zeta_an, m, b)
                sol_e_zeta_bound_error.append(self.exp_data_solv_test[index] - dE)
            self.sol_e_zeta_bound_error = np.array(sol_e_zeta_bound_error)
            self.sol_e_zeta_bound_cv = np.sqrt(np.mean(self.sol_e_zeta_bound_error**2))

        if hasattr(self, 'popt_sol_e_simple_unbound'):
            sol_e_simple_unbound_error = []
            m, b = self.popt_sol_e_simple_unbound
            for index, (i_c, a_c, i_a, a_a, i_s, a_s) in enumerate(
                    zip(self.i_cat_solv_test, self.a_cat_solv_test,
                        self.i_an_solv_test, self.a_an_solv_test,
                        self.i_sol_solv_test, self.a_sol_solv_test)):
                dE = SolvationFit.sol_e_simple_expl(i_c, a_c, i_a, a_a, i_s, a_s, m, b)
                sol_e_simple_unbound_error.append(self.exp_data_solv_test[index] - dE)
            self.sol_e_simple_unbound_error = np.array(sol_e_simple_unbound_error)
            self.sol_e_simple_unbound_cv = np.sqrt(np.mean(self.sol_e_simple_unbound_error**2))

        if hasattr(self, 'popt_an_cat_diff_allparams_unbound'):
            an_cat_diff_allparams_unbound_error = []
            gamma_cat, gamma_an, zeta_cat, zeta_an, m, b = self.popt_an_cat_diff_allparams_unbound
            for index, (i_c, a_c, i_a, a_a, i_s, a_s) in enumerate(
                    zip(self.i_cat_diff_test, self.a_cat_diff_test,
                        self.i_an_diff_test, self.a_an_diff_test,
                        self.i_sol_diff_test, self.a_sol_diff_test)):
                dE = SolvationFit.an_cat_dif_allparams_expl(i_c, a_c, i_a, a_a, i_s, a_s, gamma_cat,
                                                            gamma_an, zeta_cat, zeta_an, m, b)
                an_cat_diff_allparams_unbound_error.append(self.exp_data_diff_test[index] - dE)
            self.an_cat_diff_allparams_unbound_error = np.array(
                an_cat_diff_allparams_unbound_error)
            self.an_cat_diff_allparams_unbound_cv = np.sqrt(np.mean(self.an_cat_diff_allparams_unbound_error**2))

        if hasattr(self, 'popt_an_cat_diff_allparams_bound'):
            an_cat_diff_allparams_bound_error = []
            gamma_cat, gamma_an, zeta_cat, zeta_an, m, b = self.popt_an_cat_diff_allparams_bound
            for index, (i_c, a_c, i_a, a_a, i_s, a_s) in enumerate(
                    zip(self.i_cat_diff_test, self.a_cat_diff_test,
                        self.i_an_diff_test, self.a_an_diff_test,
                        self.i_sol_diff_test, self.a_sol_diff_test)):
                dE = SolvationFit.an_cat_dif_allparams_expl(i_c, a_c, i_a, a_a, i_s, a_s, gamma_cat,
                                                            gamma_an, zeta_cat, zeta_an, m, b)
                an_cat_diff_allparams_bound_error.append(self.exp_data_diff_test[index] - dE)
            self.an_cat_diff_allparams_bound_error = np.array(an_cat_diff_allparams_bound_error)
            self.an_cat_diff_allparams_bound_cv = np.sqrt(np.mean(self.an_cat_diff_allparams_bound_error**2))

        if hasattr(self, 'popt_an_cat_diff_gamma_unbound'):
            an_cat_diff_gamma_unbound_error = []
            gamma_cat, gamma_an, m, b = self.popt_an_cat_diff_gamma_unbound
            for index, (i_c, a_c, i_a, a_a, i_s, a_s) in enumerate(
                    zip(self.i_cat_diff_test, self.a_cat_diff_test,
                        self.i_an_diff_test, self.a_an_diff_test,
                        self.i_sol_diff_test, self.a_sol_diff_test)):
                dE = SolvationFit.an_cat_dif_gamma_expl(i_c, a_c, i_a, a_a, i_s, a_s, gamma_cat,
                                                        gamma_an, m, b)
                an_cat_diff_gamma_unbound_error.append(self.exp_data_diff_test[index] - dE)
            self.an_cat_diff_gamma_unbound_error = np.array(an_cat_diff_gamma_unbound_error)
            self.an_cat_diff_gamma_unbound_cv = np.sqrt(np.mean(self.an_cat_diff_gamma_unbound_error**2))

        if hasattr(self, 'popt_an_cat_diff_gamma_bound'):
            an_cat_diff_gamma_bound_error = []
            gamma_cat, gamma_an, m, b = self.popt_an_cat_diff_gamma_bound
            for index, (i_c, a_c, i_a, a_a, i_s, a_s) in enumerate(
                    zip(self.i_cat_diff_test, self.a_cat_diff_test,
                        self.i_an_diff_test, self.a_an_diff_test,
                        self.i_sol_diff_test, self.a_sol_diff_test)):
                dE = SolvationFit.an_cat_dif_gamma_expl(i_c, a_c, i_a, a_a, i_s, a_s, gamma_cat,
                                                        gamma_an, m, b)
                an_cat_diff_gamma_bound_error.append(self.exp_data_diff_test[index] - dE)
            self.an_cat_diff_gamma_bound_error = np.array(an_cat_diff_gamma_bound_error)
            self.an_cat_diff_gamma_bound_cv = np.sqrt(np.mean(self.an_cat_diff_gamma_bound_error**2))

        if hasattr(self, 'popt_an_cat_diff_zeta_unbound'):
            an_cat_diff_zeta_unbound_error = []
            zeta_cat, zeta_an, m, b = self.popt_an_cat_diff_zeta_unbound
            for index, (i_c, a_c, i_a, a_a, i_s, a_s) in enumerate(
                    zip(self.i_cat_diff_test, self.a_cat_diff_test,
                        self.i_an_diff_test, self.a_an_diff_test,
                        self.i_sol_diff_test, self.a_sol_diff_test)):
                dE = SolvationFit.an_cat_dif_zeta_expl(i_c, a_c, i_a, a_a, i_s, a_s, zeta_cat,
                                                       zeta_an, m, b)
                an_cat_diff_zeta_unbound_error.append(self.exp_data_diff_test[index] - dE)
            self.an_cat_diff_zeta_unbound_error = np.array(an_cat_diff_zeta_unbound_error)
            self.an_cat_diff_zeta_unbound_cv = np.sqrt(np.mean(self.an_cat_diff_zeta_unbound_error**2))

        if hasattr(self, 'popt_an_cat_diff_zeta_bound'):
            an_cat_diff_zeta_bound_error = []
            zeta_cat, zeta_an, m, b = self.popt_an_cat_diff_zeta_bound
            for index, (i_c, a_c, i_a, a_a, i_s, a_s) in enumerate(
                    zip(self.i_cat_diff_test, self.a_cat_diff_test,
                        self.i_an_diff_test, self.a_an_diff_test,
                         self.i_sol_diff_test, self.a_sol_diff_test)):
                dE = SolvationFit.an_cat_dif_zeta_expl(i_c, a_c, i_a, a_a, i_s, a_s, zeta_cat,
                                                       zeta_an, m, b)
                an_cat_diff_zeta_bound_error.append(self.exp_data_diff_test[index] - dE)
            self.an_cat_diff_zeta_bound_error = np.array(an_cat_diff_zeta_bound_error)
            self.an_cat_diff_zeta_bound_cv = np.sqrt(np.mean(self.an_cat_diff_zeta_bound_error**2))

        if hasattr(self, 'popt_an_cat_diff_simple_unbound'):
            an_cat_diff_simple_unbound_error = []
            m, b = self.popt_an_cat_diff_simple_unbound
            for index, (i_c, a_c, i_a, a_a, i_s, a_s) in enumerate(
                    zip(self.i_cat_diff_test, self.a_cat_diff_test,
                        self.i_an_diff_test, self.a_an_diff_test,
                        self.i_sol_diff_test, self.a_sol_diff_test)):
                dE = SolvationFit.an_cat_dif_simple_expl(i_c, a_c, i_a, a_a, i_s, a_s, m, b)
                an_cat_diff_simple_unbound_error.append(self.exp_data_diff_test[index] - dE)
            self.an_cat_diff_simple_unbound_error = np.array(an_cat_diff_simple_unbound_error)
            self.an_cat_diff_simple_unbound_cv = np.sqrt(np.mean(self.an_cat_diff_simple_unbound_error**2))
