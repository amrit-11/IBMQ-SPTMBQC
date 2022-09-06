"""
Date: 2022/08/01
Name: Rio Weil
Description: Functions for Running Rotation-Counterrotation SPT MBQC experiments (4-qubit rings) on IBMQ devices
"""


import numpy as np
from qiskit import *
from qiskit.providers.ibmq.managed import IBMQJobManager
from qiskit.tools.monitor import job_monitor
from qiskit.ignis.mitigation.measurement import (complete_meas_cal,CompleteMeasFitter)

# Functions for states/gates requried for the rotation-counterrotation circuits
def get_modified_state(beta, outcome, a, b):
    """
    Returns |phi_outcome> =  T^dagger |psi(+/- beta)_outcome> 
            where  |psi(+/- beta)_outcome> is the + or - eigenstate of the O(beta) operator.
    Inputs: beta - Rotation angle
            outcome - 0 for the + outcome, 1 for the - outcome
            a, b - Coefficients of the T operator (aI + bX)
    Outputs: Returns |phi_outcome> as 1d numpy array. NOTE: NOT NORMALIZED
    """
    if outcome == 0:
        state = np.array([np.exp(complex(0,beta)), 1])/np.sqrt(2)  # + Eigenstate of O(beta)
    else:  # Outcome == 1
        state = np.array([-np.exp(complex(0,beta)), 1])/np.sqrt(2) # - Eigenstate of O(beta)
    flipped_state = np.array([state[1], state[0]])
    return np.conjugate(a) * state + np.conjugate(b) * flipped_state


def get_orthogonal_vector(vector):
    """
    Returns 2d vector orthogonal to the given vector
    Inputs: vector - a 2-entry 1d numpy array
    Outputs: vector_perp - a 2-entry 1d numpy array, orthogonal to the input vector
    """
    vector_perp = np.array([-np.conjugate(vector[1]), np.conjugate(vector[0])])
    # For a complex vector [v1, v2], an orthogonal vector is given by [-v2^*, v1^*].
    return vector_perp
    

def shift_meas_basis(qc, beta, qubit, outcome, a, b):
    """
    Shifts measurement basis for the specified qubit by applying unitary on that qubit for measurements in nontrivial basis.
    Original unitary is R_z(+/- beta) H (corresponding to measurement in O(+/- beta) = cos(beta) X +/- sin(beta)Y). 
    New unitary is U = |0><phi_outcome| + |1><phi_outcome^perp| where |phi_outcome> = T^dagger |psi(+/- beta)_outcome>
        where |psi(+/- beta)_outcome> is the + or - eigenstate of the O(+/- beta) operator.
    Inputs: qc - QuantumCircuit object
            beta - Rotation angle
            qubit - Which qubit we are shifting the measurement basis for (by applying the unitary). Must be 0 or 2.
            outcome - 0 for measuring in the + outcome basis, 1 for the - outcome basis
            a, b - Coefficients of the T operator (aI + bX)
    Outputs: None
    """
    zero_state = np.array([1, 0])
    one_state = np.array([0, 1])
    phi = get_modified_state(beta, outcome, a, b)
    phi_perp = get_orthogonal_vector(phi)
    U = np.outer(zero_state, phi.conj())/np.sqrt(phi.conj() @ phi) + np.outer(one_state, phi_perp.conj())/np.sqrt(phi_perp.conj() @ phi_perp)  #  U = |0><phi|/sqrt(<phi|phi>) + |1><phi^perp|/sqrt(<phi^perp|phi^perp>)
    sign = beta/abs(beta)
    gate_label = "U(" + str(sign) + "beta)" + str(outcome)
    qc.unitary(U, qubit, label = gate_label)

    
# Functions for preparing + running rotation-counterrotation circuits


def generate_circuit(beta, a, b, qubit_0_outcome, qubit_2_outcome, err_scale=1):
    """
    Generates one circuit corresponding to one measurement outcome of experiment we are trying to simulate.
    Inputs: beta - Rotation angle
            a, b - Coefficients for the T tensor, which is a product of aI + bX.
            qubit_0_outcome - Whether the position 0 qubit was measured to be the + or - outcome
            qubit_2_outcome - Whether the position 2 qubit was measured to be the + or - outcome
    Outputs: The circuit
    """
    # Initializes circuit
    qr = QuantumRegister(4)
    qc = QuantumCircuit(qr)
    
    # Creates 4-qubit chain
    qc.h(qr)
    
    # MODIFIED BY AMRIT FOR ZNE
    for _ in range(err_scale):
        qc.cz([0, 1, 2], [1, 2, 3])

    qc.barrier(qr)
    # From here, apply local unitaries to convert from 4-qubit chain -> ring
    # Applies first Local unitary
    qc.sx(2)
    qc.sdg([1, 3])
    qc.barrier(qr)
    # Applies second Local unitary
    qc.sx(1)
    qc.sdg([0, 2, 3])
    qc.barrier(qr)
    # Applies third local unitary. After this, if we swap qubits 1 and 2 (which we will keep track of) we have a 4-ring
    qc.sx(2)
    qc.sdg([0, 1])
    qc.barrier(qr)
    
    # Setup to measure qubits 0 and 2 (actually 1 after the SWAP) in the rotated/nontrivial basis
    shift_meas_basis(qc, beta, 0, qubit_0_outcome, a, b)
    shift_meas_basis(qc, -beta, 1, qubit_2_outcome, a, b)
    
    # Setup to measure qubits 1 (actually 2 after the SWAP) and 3 in the trivial (X) basis
    qc.h(2)
    qc.h(3)

    qc.measure_all()
    
    return qc

def generate_circuit_list(beta, a, b, err_scale=1):
    """
    Generates 4 circuits to simulate the 4 experiments 
    (one circuit corresponding to each measurement outcome of experiment we are trying to simulate)
    Inputs: beta - Rotation angle
            a, b - Coefficients for the T tensor, which is a product of aI + bX.
    Outputs: List of 4 circuits.
    """ 
    qc_list = []
    
    # THIS SHOULD BE IMPROVED FOR ANY NUMBER OF QUBIT MEASUREMENTS
    # IF CHANGED, ALSO CHANGE 
    for i in range(4):
        qubit_0_outcome = (i // 2) % 2
        qubit_2_outcome = i % 2
        qc_list.append(generate_circuit(beta, a, b, qubit_0_outcome, qubit_2_outcome, err_scale))
    return qc_list

def run_circuit(qc_list, backend, best_qubits, shots_num, job_manager, meas_filter = 0):
    """
    Runs quantum circuit(s)
    Inputs: qc_list - List of QuantumCircuit Objects
           backend - Which backend (simulator or device) to use
           best_qubits - which qubits to use on the backend
           shots_num - how many shots per experiment'
           job_manager - IBMQ job manager
           meas_filter - Measurement error mitigation filter; by default is not applied.
    Outputs: List containing 4 dictionaries corresponding to counts/experimental results from supplied experiments.
    """
    qc_trans = transpile(qc_list, backend=backend, initial_layout=best_qubits)
    job_exp = job_manager.run(qc_trans, backend=backend, shots=shots_num, name='SPTexp_rotcounterrot')
    for i in range(len(job_exp.jobs())):
        job_monitor(job_exp.jobs()[i])
    print('All jobs have finished.')
    results = job_exp.results()
    
    if meas_filter != 0:
        counts_mit_list = []
        for i in range(4):
            r = results.get_counts(i)
            mit_r = meas_filter.apply(r)
            counts_mit_list.append(mit_r)
        return counts_mit_list   
    else:
        counts_list = []
        for i in range(4):
            counts_list.append(results.get_counts(i))
        return counts_list


# Functions for post-processing data
def get_norms(beta, a, b):
    """
    Calculate (squared) norms of all four |phi_outcome> =  T^dagger |psi(+/- beta)_outcome> states
    Inputs: beta - Rotation angle
            a, b - Coefficients for the T tensor, which is a product of aI + bX.
    Outputs: List containing four squared norms (0/1 outcome norms for first qubit (with +beta), then 0/1 outcome norms for third qubit (-beta))
    """
    phi_qubit0_plus = get_modified_state(beta, 0, a, b)
    norm_0_plus = phi_qubit0_plus.conj() @ phi_qubit0_plus
    phi_qubit0_minus = get_modified_state(beta, 1, a, b)
    norm_0_minus = phi_qubit0_minus.conj() @ phi_qubit0_minus
    phi_qubit2_plus = get_modified_state(-beta, 0, a, b)
    norm_2_plus = phi_qubit2_plus.conj() @ phi_qubit2_plus
    phi_qubit2_minus = get_modified_state(-beta, 1, a, b)
    norm_2_minus = phi_qubit2_minus.conj() @ phi_qubit2_minus
    
    norms = [norm_0_plus, norm_0_minus, norm_2_plus, norm_2_minus]
    return norms

def get_probability(counts):
    """
    Returns list of raw experimental probability for outcome with 0 on 0-qubit and 2-qubit.
    Outcomes will be post-selected on the 1-qubit (i.e. only second qubit = 0 are considered)
    Note that the second qubit and third qubit are switched as a result of the SWAP.
    Note that qiskit reads from right to left; i.e. the rightmost is the 0 qubit, the leftmost is 3.
    
    Inputs: counts - Dictionary containing measurement statistics
    Outputs: Probability for the 0-qubit = 0 and 2-qubit = 0 (+) outcome.
    """
    states = counts.keys()
    total_counts = 0
    good_counts = 0
    for state in states:
        if state[1] == "0":
            total_counts += counts.get(state)
            if state[2] == "0" and state[3] == "0":
                good_counts += counts.get(state)
    mean = good_counts/total_counts
    bad_counts = total_counts - good_counts
    variance = ((1 - mean)**2 * good_counts + (0 - mean)**2 * bad_counts)/total_counts
    stdev = np.sqrt(variance)
    errorofmean = stdev/np.sqrt(total_counts)
    return good_counts/total_counts, errorofmean
        

def get_probabilities(counts_list):
    """
    Returns list of raw experimental probabilities for outcome with 0 (+) on first and third qubits.
    Outcomes will be post-selected on the second qubit (i.e. only second qubit = 0 are considered)
    Note that the second qubit and third qubit are switched as a result of the SWAP.
    
    Inputs: counts_list - List of dictionaries containing measurement statistics
    Outputs: List of probabilities for the 0-qubit = 0 and 2-qubit = 0 (+) outcome.
    """
    raw_probabilities = []
    raw_uncertainties = []
    for i in range(4):
        prob, uncertainty = get_probability(counts_list[i])
        raw_probabilities.append(prob)
        raw_uncertainties.append(uncertainty)
    return raw_probabilities, raw_uncertainties


def post_process_results(counts_list, beta, a, b):
    """
    Post-process results from 4 experiments.
    First, we calculate the probability of measuring the 0 (+) outcome on the first and third qubits for each experiment.
    We then normalize by dividing by the (squared) norms of the T-modified states.
    We then normalize one last time across the 4 probability values.
    Finally, we compute <X> as p0actual + p3actual - p1actual - p2actual
    p0actual - first experiment - probability of measuring s1 = 0 s3 = 0 in simulated experiment
    p1actual - second experiment - probability of measuring s1 = 0 s3 = 1 in simulated experiment
    p2actual - third experiment - probability of measuring s1 = 1 s3 = 0 in simulated experiment
    p3actual - fourth experiment - probability of measuring s1 = 1 s3 = 1 in simulated experiment
    a, b - Coefficients for T tensor of aI + bX
    
    Inputs: counts_list - List of dictionaries containing measurement statistics
            beta - Rotation angle
            a, b - Coefficients for the T tensor, which is a product of aI + bX.
    Outputs: Xavg - <X> of the logical qubit
             Xerr - Uncertainty in <X> of the logical qubit.
    """
    raw_probs, raw_uncertainties = get_probabilities(counts_list)
    norms = get_norms(beta, a, b)
    normed_probs = []
    normed_uncertainties = []
    for i in range(4):
        qubit_0_index = (i // 2) % 2
        qubit_2_index = i % 2 + 2
        normed_probs.append(raw_probs[i] / norms[qubit_0_index] / norms[qubit_2_index])
        normed_uncertainties.append(raw_uncertainties[i] / norms[qubit_0_index] / norms[qubit_2_index])
    normed_probs = np.array(normed_probs)
    normed_uncertainties = np.array(normed_uncertainties)
    actual_probs = normed_probs/np.sum(normed_probs)
    actual_uncertainties = normed_uncertainties/np.sum(normed_probs)
    Xavg = actual_probs[0] + actual_probs[3] - actual_probs[1] - actual_probs[2]
    Xerr = np.sqrt(np.sum(np.square(actual_uncertainties)))
    return np.real(Xavg), np.real(Xerr)


# Measurement error calibrations
def run_meas_error_calibs(backend, best_qubits, shots_num):
    """
    Runs measurement error calibration circuits and returns a measurement error mitigation filter
    Inputs: backend - Which backend (simulator or device) to use
           best_qubits - which qubits to use on the backend
           shots_num - how many shots per experiment
    Outputs: meas_filter - Measurement error mitigation filter based on calibrations
    """
    qc_err = qiskit.QuantumRegister(len(best_qubits))
    meas_calibs, state_labels = complete_meas_cal(qr=qc_err, circlabel='mcal')
    meas_calibs = transpile(meas_calibs, backend=backend, initial_layout=best_qubits)
    job_err = qiskit.execute(meas_calibs, backend=backend, shots=shots_num)
    job_monitor(job_err)
    cal_result = job_err.result()
    meas_fitter = CompleteMeasFitter(cal_result, state_labels, circlabel='mcal')
    meas_filter = meas_fitter.filter
    return meas_filter


# Running the experiments
def one_exp_point(backend, best_qubits, shots_num, job_manager, beta, a, b, meas_filter = 0):
    """
    Generates a single datapoint of the rotation-counter rotation experiment
    Inputs: backend - Which backend (simulator or device) to use
           best_qubits - which (4) qubits to use on the backend
           shots_num - how many shots per experiment
           job_manager - IBMQ jobmanager
           beta - rotation angle
           a - T-operator coefficient for I
           b - T-operator coefficient for X
           meas_filter - Measurement error mitigation filter; by default is not applied.
    Outputs: logicXavg - <X> of the logical qubit
             logicXerr - Uncertainty in <X> of the logical qubit.
    """
    qc_list = generate_circuit_list(beta, a, b)
    counts_list = run_circuit(qc_list, backend, best_qubits, shots_num, job_manager, meas_filter)
    logicXavg, logicXerr = post_process_results(counts_list, beta, a, b)
    return logicXavg, logicXerr

def many_exp_point(backend, best_qubits, shots_num, job_manager, beta, acoeff_ar, bcoeff_ar, meas_filter = 0):
    """
    Generates multiple datapoints of the rotation-counter rotation experiment
    Inputs: backend - Which backend (simulator or device) to use
           best_qubits - which (4) qubits to use on the backend
           shots_num - how many shots per experiment
           job_manager - IBMQ jobmanager
           beta - rotation angle
           a_ar - A list of the T-operator coefficients for I
           b_ar - A list of the T-operator coefficient for X
           meas_filter - Measurement error mitigation filter; by default is not applied
    Outputs: logicXavg_ar - <X> of the logical qubit (one for each pair of coefficients)
            logicXuncertainty_ar - Uncertainty in <X> of the logical qubit (one for each pair of coefficients)
    """
    logicXavg_ar = []
    logicXerr_ar = []
    for i in range(len(acoeff_ar)):
        a = acoeff_ar[i]
        b = bcoeff_ar[i]
        logicXavg, logicXerr = one_exp_point(backend, best_qubits, shots_num, job_manager, beta, a, b, meas_filter)
        logicXavg_ar.append(logicXavg)
        logicXerr_ar.append(logicXerr)
    return np.array(logicXavg_ar), np.array(logicXerr_ar)
        
        
    
#ADDED BY AMRIT

def generate_circuits_zne(beta, acoeff_ar, bcoeff_ar, sf_list):
    '''
    Generates circuits for each a and b coefficients for every scale factor in sf_list
    '''
    circ_list = []
    for sf in sf_list:
        for i in range(len(acoeff_ar)):
            a = acoeff_ar[i]
            b = bcoeff_ar[i]
            circ_ab = generate_circuit_list(beta, a, b, sf)
            circ_list.extend(circ_ab)
    return circ_list