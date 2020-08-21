import numpy as np
from matplotlib import pyplot as plt
from solvation_fit import SolvationFit
from solvation_cv import SolvationCV


# Sample file to generate and process the results of a solvation calculation

# Dictionary with the (I, A) pairs for each cation.
cation_IA = {"Li": (63.95, 6.95), "Na": (39.18, 6.93), "K": (26.44, 5.84), "Rb": (23.01, 5.67),
             "Cs": (19.79, 5.26)}

# Dictionary with the (chemical potential, hardness) pairs for each anion
anion_desc = {"F": (-1.91, 5.39), "Cl": (-1.82, 5.43), "Br": (-0.88, 5.57), "I": (-1.62, 6.84)}

# Generate the (I, A) dictionary for the anions given the (Mu, Eta) values.
anion_IA = {}

for key in anion_desc:
    anion_IA[key] = (0.5 * (anion_desc[key][1] - 2 * anion_desc[key][0]),
                     - 0.5 * (anion_desc[key][1] + 2 * anion_desc[key][0]))

# Dictionary with the (I, A) pairs for each solvent.
solvent_IA = {"water": (8.7129, 0.2283), "MeOH": (7.6533, 0.0138), "EtOH": (7.5420, 0.0112),
              "1PrOH": (7.5330, -0.0547), "DMSO": (6.44, 0.07), "DMA": (6.6477, 0.1440),
              "DMF": (6.7990, 0.1484), "PC": (8.2237, 0.2351), "ACE": (6.9296, 0.6759),
              "BuOH": (7.5219, -0.0487), "EG": (7.5610, 0.0544), "FA": (7.3446, 0.1072),
              "MeCN": (9.1495, 0.2285), "NMF": (7.1727, 0.0476)}

# Cations that will be considered in the analysis.
cations = ["Li", "Na", "K", "Rb", "Cs"]

# Anions that will be considered in the analysis.
anions = ["Cl", "Br", "I"]

# Solvents that will be considered in the analysis.
solvents = ["water", "MeOH", "EtOH", "1PrOH", "DMSO", "DMA", "DMF", "PC", "ACE", "BuOH",
            "EG", "FA", "MeCN"]


# Auxiliary functions
def gather_data(s_object):
    """Collect the data from the SolvationFit instance

    Parameters
    ----------
    s_object : SolvationFit instance
        SolvationFit object.

    Returns
    -------
    s_attrs: list of str
        List that with the name of the results and errors attributes of the SolvationFit object.
    s_data : dictionary
        Dictionary with the values of the results and errors attributes of the SolvationFit object.
    """
    s_data = {}
    s_attrs = [a for a in s_object.__dict__.keys() if ("result" in a) or ("error" in a)]
    for attr in s_attrs:
        s_data[attr] = eval("s_object." + attr)
    return s_attrs, s_data


def ref_data(s_object):
    """Collect the reference experimental data from the SolvationFit instance

    Parameters
    ----------
    s_object : SolvationFit instance
        SolvationFit object.

    Returns
    -------
    s_object.exp_data_solv : np.ndarray
        Numpy array with the experimental solvation energies.
    s_object.exp_data_diff
        Numpy array with the experimental anion-cation energies.
    """
    return s_object.exp_data_solv, s_object.exp_data_diff


def gen_cross_indices(total_data, fraction=5, type="random", repetitions=7):
    """Generate the cross validation indices."""
    ignore_size = total_data//fraction
    c_indices = []
    if type == "random":
        for i in range(repetitions):
            c_inds = []
            while len(c_inds) < ignore_size:
                index = random.randint(0, total_data - 1)
                if index in c_inds:
                    pass
                else:
                    c_inds.append(index)
            c_indices.append(c_inds)
    elif type == "sequential":
        for i in range(fraction + 1):
            c_indices.append(list(range(total_data)[i * ignore_size:(i + 1) * ignore_size]))
    return c_indices


def cv_results(s_object, s_attrs, sol_indices, diff_indices):
    """Generate the cross validation results."""
    unique_attrs = []
    for attr in s_attrs:
        if len(attr.split("_error")) == 2:
            unique_attrs.append(attr.split("_error")[0])
    individual_cvs = {}
    cv_errors = {}
    for attr in unique_attrs:
        individual_cvs[attr] = []
    
    for solv_test_indices, diff_test_indices in zip(sol_indices, diff_indices):
        s_cv = SolvationCV(s_object, solv_test_indices, diff_test_indices)
        for attr in unique_attrs:
            individual_cvs[attr].append(eval("s_cv." + attr + "_cv"))
    
    for attr in unique_attrs:
        individual_cvs[attr] = np.array(individual_cvs[attr])
        cv_errors[attr] = np.mean(individual_cvs[attr])
    
    return individual_cvs, cv_errors


def out_str(s_object, s_attrs, s_data, ref_solv, ref_diff, cv_errors):
    """Generate output files.

    Notes
    -----
    Obviously convoluted and cumbersome, prone to be cleaned/streamlined.
    """
    state_function = s_object.calc_type[2]
    def aic_variants(attr, errors):
        if "allparams" in attr:
            K = 6
        elif "simple" in attr:
            K = 2
        elif ("gamma" in attr) or ("zeta" in attr):
            K = 4
        n = len(errors)
        rss = np.sum(errors**2)
        aic = 2 * K + n*np.log(rss/n)
        aic_c = aic + 2 * K * (K + 1)/(n - K - 1)
        return aic, aic_c
        
    s = "RESULTS\n\n"
    if s_object.calc_type[0] == "solvent":
        s += "For the following solvents:\n"
        for solvent in s_object.calc_type[1]:
            s += "{}  ".format(solvent)
        s += "\n"
        if isinstance(s_object.cations, dict):
            cations = s_object.cations.keys()
        else:
            cations = s_object.cations
        if isinstance(s_object.anions, dict):
            anions = s_object.anions.keys()
        else:
            anions = s_object.anions
        s += "We considered all the salts that could be formed with the following" \
             "cations and anions:\n"
        s += "Cations: "
        for cation in cations:
            s += "{}  ".format(cation)
        s += "\nAnions: "
        for anion in anions:
            s += "{}  ".format(anion)
    elif s_object.calc_type[0] == "ion":
        s += "We considered all the salts that could be formed with the following" \
             "cations and anions:\n"
        s += "Cations: "
        for pair in s_object.calc_type[1]:
            s += "{}  ".format(pair[0])
        s += "\n"
        s += "Anions: "
        for pair in s_object.calc_type[1]:
            s += "{}  ".format(pair[1])
        s += "\n"
        s += "In the following solvents:\n"
        if isinstance(s_object.solvents, dict):
            solvents = s_object.solvents.keys()
        else:
            solvents = s_object.solvents
        for solvent in solvents:
            s += "{}  ".format(solvent)
    s += "\n\nd{}s\n             ".format(state_function)
    for attr in sorted(s_attrs[::2]):
        if "sol_e" in attr:
            parts = attr.split("_")
            header = parts[-3] + "_" + parts[-2]
            s += "{:^32}  ".format(header)
    s += "\nRef d{}s     ".format(state_function)
    for attr in sorted(s_attrs):
        if "sol_e" in attr:
            parts = attr.split("_")
            s += "{:^12}     ".format(parts[-1])
    s += "\n"
    for j in range(len(ref_solv)):
        s += "{:>8.3f} ".format(ref_solv[j])
        for attr in sorted(s_attrs):
            if "sol_e" in attr:
                s += "{:>12.3f}     ".format(s_data[attr][j])
        s += "\n"
    s += "\nStatistical Summary d{}s\n             ".format(state_function)
    for attr in sorted(s_attrs[::2]):
        if "sol_e" in attr:
            parts = attr.split("_")
            header = parts[-3] + "_" + parts[-2]
            s += "{:^32}  ".format(header)
    s += "\n"
    s += "Unsigned Error"
    for attr in sorted(s_attrs):
        if "sol_e" in attr and "error" in attr:
            s += "{:^32.3f}  ".format(np.sum(s_data[attr]))
    s += "\n"
    s += "RMSD          "
    for attr in sorted(s_attrs):
        if "sol_e" in attr and "error" in attr:
            s += "{:^32.3f}  ".format(np.sqrt(np.mean(s_data[attr]**2)))
    s += "\n"
    s += "CV            "
    for attr in sorted(s_attrs):
        if "sol_e" in attr and "error" in attr:
            s += "{:^32.3f}  ".format(cv_errors[attr.split("_error")[0]])
    s += "\n"
    aic_values = []
    aic_c_values = []
    for attr in sorted(s_attrs):
        if "sol_e" in attr and "error" in attr:
            aic, aic_c = aic_variants(attr, s_data[attr])
            aic_values.append(aic)
            aic_c_values.append(aic_c)
    aic_values = np.array(aic_values)
    aic_c_values = np.array(aic_c_values)
    d_aic_values = aic_values - np.min(aic_values)
    d_aic_c_values = aic_c_values - np.min(aic_c_values)
    indices = []
    for attr in sorted(s_attrs):
        if "sol_e" in attr and "error" in attr:
            indices.append(1)
    s += "AIC           "
    for i in range(len(indices)):
        s += "{:^32.3f}  ".format(aic_values[i])
    s += "\n"
    s += "dAIC          "
    for i in range(len(indices)):
        s += "{:^32.3f}  ".format(d_aic_values[i])
    s += "\n"
    s += "AICc         "
    for i in range(len(indices)):
        s += "{:^32.3f}  ".format(aic_c_values[i])
    s += "\n"
    s += "dAICc        "
    for i in range(len(indices)):
        s += "{:^32.3f}  ".format(d_aic_c_values[i])
    s += "\n"
    
    s += "\n\nd{}-+\n             ".format(state_function)
    for attr in sorted(s_attrs[::2]):
        if "an_cat" in attr:
            parts = attr.split("_")
            header = parts[-3] + "_" + parts[-2]
            s += "{:^32}  ".format(header)
    s += "\nRef d{}-+    ".format(state_function)
    for attr in sorted(s_attrs):
        if "an_cat" in attr:
            parts = attr.split("_")
            s += "{:^12}     ".format(parts[-1])
    s += "\n"
    for j in range(len(ref_diff)):
        s += "{:>8.3f} ".format(ref_diff[j])
        for attr in sorted(s_attrs):
            if "an_cat" in attr:
                s += "{:>12.3f}     ".format(s_data[attr][j])
        s += "\n"
    s += "\nStatistical Summary d{}-+\n             ".format(state_function)
    for attr in sorted(s_attrs[::2]):
        if "an_cat" in attr:
            parts = attr.split("_")
            header = parts[-3] + "_" + parts[-2]
            s += "{:^32}  ".format(header)
    s += "\n"
    s += "Unsigned Error"
    for attr in sorted(s_attrs):
        if "an_cat" in attr and "error" in attr:
            s += "{:^32.3f}  ".format(np.sum(s_data[attr]))
    s += "\n"
    s += "RMSD          "
    for attr in sorted(s_attrs):
        if "an_cat" in attr and "error" in attr:
            s += "{:^32.3f}  ".format(np.sqrt(np.mean(s_data[attr]**2)))
    s += "\n"
    s += "CV            "
    for attr in sorted(s_attrs):
        if "an_cat" in attr and "error" in attr:
            s += "{:^32.3f}  ".format(cv_errors[attr.split("_error")[0]])
    s += "\n"
    aic_values = []
    aic_c_values = []
    for attr in sorted(s_attrs):
        if "an_cat" in attr and "error" in attr:
            aic, aic_c = aic_variants(attr, s_data[attr])
            aic_values.append(aic)
            aic_c_values.append(aic_c)
    aic_values = np.array(aic_values)
    aic_c_values = np.array(aic_c_values)
    d_aic_values = aic_values - np.min(aic_values)
    d_aic_c_values = aic_c_values - np.min(aic_c_values)
    indices = []
    for attr in sorted(s_attrs):
        if "an_cat" in attr and "error" in attr:
            indices.append(1)
    s += "AIC           "
    for i in range(len(indices)):
        s += "{:^32.3f}  ".format(aic_values[i])
    s += "\n"
    s += "dAIC          "
    for i in range(len(indices)):
        s += "{:^32.3f}  ".format(d_aic_values[i])
    s += "\n"
    s += "AICc         "
    for i in range(len(indices)):
        s += "{:^32.3f}  ".format(aic_c_values[i])
    s += "\n"
    s += "dAICc        "
    for i in range(len(indices)):
        s += "{:^32.3f}  ".format(d_aic_c_values[i])
    s += "\n"
    
    with open("{}_SolvationResults.txt".format(state_function), "w") as outfile:
        outfile.write(s)


def gen_individual_pic(state_function, name_data, x_label, y_label, attr, ref_values, cdft_values):
    """Generate the figure for a single result."""
    fig, ax = plt.subplots()
    ax.scatter(cdft_values, ref_values, s=25, cmap=plt.cm.coolwarm, zorder=10)
    lims = [np.min([ax.get_xlim(), ax.get_ylim()]),
            np.max([ax.get_xlim(), ax.get_ylim()])]
    ax.plot(lims, lims, 'k-', alpha=0.75, zorder=0)
    ax.set_aspect('equal')
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.title.set_text(name_data)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    fig.savefig("{}_{}.png".format(state_function, attr[:-7]))
    plt.cla()


def gen_pics(s_object, s_attrs, s_data, ref_solv, ref_diff):
    """Generate the figures for all the results."""
    state_function = s_object.calc_type[2]
    for attr in sorted(s_attrs):
        if "error" in attr:
            pass
        else:
            plt.close()
            if "sol_e" in attr:
                ref_values = ref_solv
                x_label = "dEsol_CDFT"
                y_label = "d{}sol_experimental".format(state_function)
            else:
                ref_values = ref_diff
                x_label = "dE-+_CDFT"
                y_label = "d{}-+_experimental".format(state_function)
            cdft_values = s_data[attr]
            
            name_data = ""
            if s_object.calc_type[0] == "solvent":
                name_data += "Const. solvs.: "
                if len(s_object.calc_type[1]) == 13:
                    name_data += "all\n"
                else:
                    for solvent in s_object.calc_type[1]:
                        name_data += "{} ". format(solvent)
                    name_data += "\n"
                if isinstance(s_object.cations, dict):
                    cations = s_object.cations.keys()
                else:
                    cations = s_object.cations
                if isinstance(s_object.anions, dict):
                    anions = s_object.anions.keys()
                else:
                    anions = s_object.anions
                name_data += "{Cations: "
                for cation in cations:
                    name_data += "{} ". format(cation)
                name_data += "} {Anions: "
                for anion in anions:
                    name_data += "{} ". format(anion)
                name_data += "}"
            elif s_object.calc_type[0] == "ion":
                name_data += "{Cations: "
                for pair in s_object.calc_type[1]:
                    name_data += "{} ".format(pair[0])
                name_data += "} {Anions: "
                for pair in s_object.calc_type[1]:
                    name_data += "{} ".format(pair[1])
                name_data += "}\n"
                if isinstance(s_object.solvents, dict):
                    solvents = s_object.solvents.keys()
                else:
                    solvents = s_object.solvents
                if len(solvents) == 13:
                    name_data += "Solvents: all"
                else:
                    name_data += "Solvents: "
                    for solvent in solvents:
                        name_data += "{} ".format(solvent)
            gen_individual_pic(state_function, name_data, x_label, y_label, attr,
                               ref_values, cdft_values)
            

if __name__ == "__main__":
    # SolvationFit object that will be used to generate the results.
    s_object = SolvationFit(cation_IA=cation_IA, anion_IA=anion_IA, solvent_IA=solvent_IA,
                            data_file="extracted_thermodynamics.csv",
                            calc_type=("ion", [("Rb", "Cl")], "G"), min_bound=0, cations=cations,
                            anions=anions, solvents=solvents)
    
    sol_indices = gen_cross_indices(total_data=len(s_object.i_cat_solv),
                                    fraction=len(s_object.i_cat_solv) - 1,
                                    type="sequential", repetitions=7)
    diff_indices = gen_cross_indices(total_data=len(s_object.i_cat_diff),
                                     fraction=len(s_object.i_cat_diff) - 1,
                                     type="sequential", repetitions=7)
    s_attrs, s_data = gather_data(s_object)
    ref_solv, ref_diff = ref_data(s_object)
    individual_cvs, cv_errors = cv_results(s_object, s_attrs, sol_indices, diff_indices)
    out_str(s_object, s_attrs, s_data, ref_solv, ref_diff, cv_errors)
    gen_pics(s_object, s_attrs, s_data, ref_solv, ref_diff)
