"""
Date: 2022/08/01
Name: Rio Weil
Description: Functions for running VQE algorithms for approximating 4-qubit (ring) SPTMBQC resource (ground) states
"""
import numpy as np
import random
from qiskit import *
from qiskit.providers.ibmq.managed import IBMQJobManager
from qiskit.tools.monitor import job_monitor
from qiskit.ignis.mitigation.measurement import (complete_meas_cal,CompleteMeasFitter)

# Helper functions for generating basic gates/circuits as part of the VQE algorithms
def apply_Utheta(qc, qubit, theta):
    """
    Applies U(theta) = cos(theta)|0><0| + sin(theta)|0><1| - sin(theta)|1><0| + cos(theta)|1><1| to specified qubit in quantum circuit
    Inputs: qc - quantum circuit
            qubit - qubit in quantum circuit to apply U(theta) to
            theta - variational parameter
    Outputs: None
    """
    U = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])
    gate_label = "Utheta"
    qc.unitary(U, qubit, label = gate_label)
    
def generate_onequbit_circuit(theta, s):
    """
    Generates the VQE circuit |+>-[Z^s]-[U]-||Z for <Xi> and <Ki> measurements.
    Inputs: theta - variational parameter
            s - whether pauli Z is applied before U or not (0 or 1).
    Outputs: The 1-qubit VQE circuit
    """
    # Initializes circuit
    qr = QuantumRegister(1)
    qc = QuantumCircuit(qr)
    
    # Set |0> to |+>
    qc.h(0)

    # Apply Z^s
    if s == 1:
        qc.z(0)

    # Apply U(theta) to qubit
    apply_Utheta(qc, 0, theta)

    # Measure in comp. basis
    qc.measure_all()

    return qc

# MODIFIED FOR ZNE BY AMRIT
def generate_threequbit_circuit(theta, s2, s2z, err_scale=1):
    """
    Generates the three-qubit VQE circuit for measurement of <Ki>
    Inputs: theta - variational parameter
            s2 - measurement outcome on second physical qubit in original circuit (0 or 1). Defines whether Pauli-Z comes up on second qubit.
            s2z - measurement outcome on second ancilla qubit in original circuit (0 or 1). Defines whether Pauli-X and Pauli-Z come up on first/second qubits.
    """
    s2prime = (s2 + s2z)%2
    
    qr = QuantumRegister(3)
    qc = QuantumCircuit(qr)

    # Make cluster chain
    qc.h(qr)
    for _ in range(err_scale):
        qc.cz(0, 1)
        qc.cz(1, 2)
    
    # Some tricks with hadamards to get the correct cluster state ``orientation''
    qc.h(0)
    qc.h(1)
    
    # Applying X, Z conditioned on the s2 outcomes
    if s2z == 1:
        qc.x(0)
    if s2prime == 1:
        qc.z(1)
        
    # Set up measurement bases (and also CZ the last 2 qubits)
    apply_Utheta(qc, 0, theta)
    qc.h(1)
    for _ in range(err_scale):
        qc.cz(1, 2)
    apply_Utheta(qc, 1, theta)
    qc.h(2)
    
    qc.measure_all()
    
    return qc

def convert_dict_to_list(dictionary):
    """
    Converts a dictionary into a list, where each dictionary entry gets one list element (respecting multiplicity).
    Input: dictionary - A dictonary with keys associated to values (numbers)
    Output: baselist - A list containing X keys where X is the valuye associated to the key in the original dictionary.
    """
    baselist = []
    states = dictionary.keys()
    for state in states:
        baselist = baselist + round(dictionary.get(state)) * [state]
    return baselist


# Functions for calculating <X>_i (local magnetic field term)

def perform_avgX(backend, best_qubits, shots_num, job_manager, theta, meas_filter_one = 0):
    """
    Performs circuits and calculates local magnetic field <X>i for VQE (function of theta)
    Input: backend - Which backend (simulator or device) to use
           best_qubits - which (1) qubits to use on the backend
           shots_num - how many shots per experiment
           job_manager - IBMQ jobmanager
           theta - VQE parameter.
           meas_filter_one - One-qubit measurement error mitigation filter; by default is not applied.
    Output: Xavg - Local magnetic field expectation value
            Xerr - Statistical error associated with above expectation
    """
    qc_trans = transpile([generate_onequbit_circuit(theta, 0), generate_onequbit_circuit(theta, 0)], backend=backend, initial_layout=best_qubits)
    job_exp = job_manager.run(qc_trans, backend=backend, shots=shots_num, name='VQEX')
    for i in range(len(job_exp.jobs())):
        job_monitor(job_exp.jobs()[i])
    print('All jobs have finished.')
    
    results = job_exp.results()
    counts = results.get_counts(0)
    counts2 = results.get_counts(1)
    
    if meas_filter_one != 0:
        counts = meas_filter_one.apply(counts)
        counts2 = meas_filter_one.apply(counts2)
        
    return calculate_avgX(counts, counts2)

def calculate_avgX(counts, counts2):
    """
    Calculates <X>_i based on circuit outcomes.
    Input: counts - Outcome data from first one-qubit circuit
           counts2 - Outcome data from second (identical) one-qubit circuit
    Output: Xavg - Local magnetic field expectation value
            Xerr - Statistical error associated with above expectation
    """
    fulllist = convert_dict_to_list(counts) + convert_dict_to_list(counts2)
    batches = []
    
    while fulllist:
        s1 = random.choice(fulllist)
        fulllist.remove(s1)
        s2 = random.choice(fulllist)
        fulllist.remove(s2)
        batches.append([s1, s2])

    goodbatches = []
    while batches:
        batch1 = random.choice(batches)
        batches.remove(batch1)
        batch2 = random.choice(batches)
        batches.remove(batch2)
        if batch1[0] == batch1[1] and batch2[0] == batch2[1]:
            goodbatches.append(batch1)
            goodbatches.append(batch2)
       
    N = len(goodbatches)
    survivingmeas = []
    for batch in goodbatches:
        for s in batch:
            if s == "0":
                survivingmeas.append(1) 
            else:
                survivingmeas.append(-1)
    survivingmeas = np.array(survivingmeas)
        
    Xavg = np.sum(survivingmeas)/2/N  # Note: Robert's analysis has I believe an extra (erroneous) factor of 1/2 here.
    Xerr = np.std(survivingmeas)/np.sqrt(2*N)
            
    return Xavg, Xerr

def multiple_avgX(backend, best_qubits, shots_num, job_manager, theta_ar, meas_filter_one = 0):    
    """
    Performs circuits and calculates local magnetic field <X>i)for VQE for multiple thetas.
    Input: backend - Which backend (simulator or device) to use
           best_qubits - which (1) qubits to use on the backend
           shots_num - how many shots per experiment
           job_manager - IBMQ jobmanager
           theta_ar - Array of VQE parameters (which will be iterated over).
           meas_filter_one - One-qubit measurement error mitigation filter; by default is not applied.
    Output: Xavg_ar - Array of local magnetic field expectation value (as a function of theta)
          
          Xerr_ar - Statistical errors associated with above expectation values.
    """
    Xavg_ar = []
    Xerr_ar = []
    for theta in theta_ar:
        Xavg, Xerr = perform_avgX(backend, best_qubits, shots_num, job_manager, theta, meas_filter_one)
        Xavg_ar.append(Xavg)
        Xerr_ar.append(Xerr)
    return np.array(Xavg_ar), np.array(Xerr_ar)

# Functions for calcualting <K>_i (cluster state stabilizer term)

# MODIFIED FOR ZNE BY AMRIT
def perform_avgK(backend, best_qubits_one, best_qubits_three, shots_num, job_manager, theta, err_scale=1, meas_filter_one = 0, meas_filter_three = 0):
    """
    Performs circuits and calculates cluster state stabilizer expectation <K>i for VQE (function of theta)
    Input: backend - Which backend (simulator or device) to use
           best_qubits_one - which (1) qubit to use on the backend for single qubit circuits
           best_qubits_three - which (3) qubits to use on the backend for three qubit circuits
           shots_num - how many shots per experiment
           job_manager - IBMQ jobmanager
           theta - VQE parameter.
           meas_filter_one - One-qubit measurement error mitigation filter; by default is not applied.
           meas_filter_three - Three-qubit measurement error mitigation filter; by default is not applied
    Output: Kavg - Cluster state expectation value
            Kerr - Statistical error associated with above expectation
    """
    
    step1circ1 = generate_onequbit_circuit(theta, 0)
    step1circ2 = generate_onequbit_circuit(theta, 1)
    step3circ1 = generate_onequbit_circuit(theta, 0)
    step3circ2 = generate_onequbit_circuit(theta, 1)
    onequbitcircs = transpile([step1circ1, step1circ2, step3circ1, step3circ2], backend = backend, initial_layout=best_qubits_one)
    job_exp = job_manager.run(onequbitcircs, backend=backend, shots=shots_num, name='VQEK-1qubit')
    for i in range(len(job_exp.jobs())):
        job_monitor(job_exp.jobs()[i])
    print('All jobs have finished.')
    onequbitresults = job_exp.results()
    onequbitcounts0 = onequbitresults.get_counts(0)
    onequbitcounts1 = onequbitresults.get_counts(1)
    onequbitcounts2 = onequbitresults.get_counts(2)
    onequbitcounts3 = onequbitresults.get_counts(3)
    if meas_filter_one != 0:
        onequbitcounts0 = meas_filter_one.apply(onequbitcounts0)
        onequbitcounts1 = meas_filter_one.apply(onequbitcounts1)
        onequbitcounts2 = meas_filter_one.apply(onequbitcounts2)
        onequbitcounts3 = meas_filter_one.apply(onequbitcounts3)
    onequbitcountlist = [onequbitcounts0, onequbitcounts1, onequbitcounts2, onequbitcounts3]
    
    
    step2circ1 = generate_threequbit_circuit(theta, 0, 0, err_scale)
    step2circ2 = generate_threequbit_circuit(theta, 0, 1, err_scale)
    step2circ3 = generate_threequbit_circuit(theta, 1, 0, err_scale)
    step2circ4 = generate_threequbit_circuit(theta, 1, 1, err_scale)
    
    threequbitcircs = transpile([step2circ1, step2circ2, step2circ3, step2circ4], backend=backend, initial_layout=best_qubits_three)
    job_exp = job_manager.run(threequbitcircs, backend=backend, shots=(shots_num+10), name='VQEK-3qubit')
    for i in range(len(job_exp.jobs())):
        job_monitor(job_exp.jobs()[i])
    print('All jobs have finished.')
    threequbitresults = job_exp.results()
    threequbitcounts0 = threequbitresults.get_counts(0)
    threequbitcounts1 = threequbitresults.get_counts(1)
    threequbitcounts2 = threequbitresults.get_counts(2)
    threequbitcounts3 = threequbitresults.get_counts(3)
    if meas_filter_three != 0:
        threequbitcounts0 = meas_filter_three.apply(threequbitcounts0)
        threequbitcounts1 = meas_filter_three.apply(threequbitcounts1)
        threequbitcounts2 = meas_filter_three.apply(threequbitcounts2)
        threequbitcounts3 = meas_filter_three.apply(threequbitcounts3)
    threequbitcountlist = [threequbitcounts0, threequbitcounts1, threequbitcounts2, threequbitcounts3]
    return calculate_avgK(onequbitcountlist, threequbitcountlist)

def survival_test(counts, sz2, step2list, step3list0, step3list1):
    """
    Performs a test to see if a batch of outcomes satisfies a survivability condition, and counts how many that do so.
    Input: counts - Number of a particular outcome from one-qubit circuits
           sz2 - Whether the sz2 outcome (of which the above counts correspond to) was 0 or 1
           step2list - List of outcomes from step 2 three qubit circuit
           step3list0 - List of outcomes from step 3 three qubit circuit (for 0)
           step3list1 - List of outcomes from step3 three qubit circuit (for 1)
    """
    Nsurv = 0
    while counts > 0:
        resstep2 = random.choice(step2list)
        step2list.remove(resstep2)
        if resstep2[2] == "0" and resstep2[1] == "0":
            s4prime = int(resstep2[0])
            s4 = (s4prime + sz2) % 2
            if s4 == 0:
                resstep3 = random.choice(step3list0)
                step3list0.remove(resstep3)
                if int(resstep3) == sz2:
                    Nsurv += 1
            else: # s4 == 1
                resstep3 = random.choice(step3list1)
                step3list1.remove(resstep3)
                if int(resstep3) == sz2:
                    Nsurv += 1
        counts -= 1
    return Nsurv
            
        
def calculate_avgK(onequbitcountlist, threequbitcountlist):
    """
    Calculates cluster state stabilizer expectation <K>i from VQE circuit outcomes
    Input: onequbitcountlist - Data from one qubit circuits
           threequbitcountlist - Data from three qubit circuits
    Output: Kavg - Cluster state expectation value
            Kerr - Statistical error associated with above expectation
    """
    step2list00 = convert_dict_to_list(threequbitcountlist[0])
    step2list01 = convert_dict_to_list(threequbitcountlist[1])
    step2list10 = convert_dict_to_list(threequbitcountlist[2])
    step2list11 = convert_dict_to_list(threequbitcountlist[3])

    step3list0 = convert_dict_to_list(onequbitcountlist[2])
    step3list1 = convert_dict_to_list(onequbitcountlist[3])

    s20counts = onequbitcountlist[0]
    s21counts = onequbitcountlist[1]
        
    counts20s2z0 = s20counts.get("0")
    counts20s2z1 = s20counts.get("1")
    counts21s2z0 = s21counts.get("0")
    counts21s2z1 = s21counts.get("1")
        
    Nplus = 0
    Nminus = 0
    
    if counts20s2z0 is not None:
        Nplus += survival_test(counts20s2z0, 0, step2list00, step3list0, step3list1)
    if counts20s2z1 is not None:
        Nplus += survival_test(counts20s2z1, 1, step2list01, step3list0, step3list1)
    if counts21s2z0 is not None:
        Nminus += survival_test(counts21s2z0, 0, step2list10, step3list0, step3list1)
    if counts21s2z1 is not None:
        Nminus += survival_test(counts21s2z1, 1, step2list11, step3list0, step3list1)
        
    Ntot = Nplus + Nminus
        
    Kavg = (Nplus - Nminus)/Ntot
    Kstdev = np.sqrt((Nplus*(Kavg - 1)**2 + Nminus * (Kavg + 1)**2)/Ntot)
    Kerr = Kstdev/np.sqrt(Ntot)
    
    return Kavg, Kerr

def multiple_avgK(backend, best_qubits_one, best_qubits_three, shots_num, job_manager, theta_ar, meas_filter_one = 0, meas_filter_three = 0):    
    """
    Performs circuits and calculates cluster state stabilizer expectation <K>i for VQE for multiple thetas
    Input: backend - Which backend (simulator or device) to use
           best_qubits_one - which (1) qubit to use on the backend for single qubit circuits
           best_qubits_three - which (3) qubits to use on the backend for three qubit circuits
           shots_num - how many shots per experiment
           job_manager - IBMQ jobmanager
           theta_ar - Array of VQE parameters (which will be iterated over).
           meas_filter_one - One-qubit measurement error mitigation filter; by default is not applied.
           meas_filter_three - Three-qubit measurement error mitigation filter; by default is not applied
    Output: Kavg - Cluster state expectation value
            Kerr - Statistical error associated with above expectation
    """
    Kavg_ar = []
    Kerr_ar = []
    for theta in theta_ar:
        Kavg, Kerr = perform_avgK(backend, best_qubits_one, best_qubits_three, shots_num, job_manager, theta, meas_filter_one, meas_filter_three)
        Kavg_ar.append(Kavg)
        Kerr_ar.append(Kerr)
    return np.array(Kavg_ar), np.array(Kerr_ar)


# Theory predictions for <Xi> and <Ki> for VQE

def predictedX(theta):
    """
    Calculates theory prediction for local magnetic field expectation value as a function of VQE parameter theta.
    Input: theta - VQE paramter
    Output: Xavgtheory - Theory prediction for <Xi> 
    """
    x = 2 * np.cos(theta) * np.sin(theta)
    Xavgtheory = 2 * x / (1 + x**2)
    return Xavgtheory

def predictedK(theta):
    """
    Calculates theory prediction for cluster stabilizer expectation value as a function of VQE parameter theta.
    Input: theta - VQE paramter
    Output: Kavgtheory - Theory prediction for <Ki> 
    """
    x = 2 * np.cos(theta) * np.sin(theta)
    Kavgtheory = (1 - x**2)/(1+x**2)
    return Kavgtheory


#  Extracting optimal parameters from VQE
def get_thetamin_VQE(alpha, theta_ar, Xavg_ar, Kavg_ar):
    """
    Find the VQE parameter theta that minimizes E(alpha) = -sin(alpha)X(theta) - cos(alpha)Ki(theta) for a given alpha.
    Input: alpha - Interpolation paramter for SPT ground states
           theta_ar - Array of VQE paramters
           Xavg_ar - Local magnetic field expectation values (as a function of theta) obtained via VQE.
           Kavg_ar - Cluster stabilizer expectation value (as a function of theta) obtained via VQE.
    Output: thetamin - theta that minimizes energy
    """
    E = -np.sin(alpha)*Xavg_ar - np.cos(alpha)*Kavg_ar
    thetamin = theta_ar[np.argmin(E)]
    return thetamin

def multiple_get_thetamin_VQE(alpha_ar,  theta_ar, Xavg_ar, Kavg_ar):
    """
    Find the VQE parameter theta that minimizes E(alpha) = -sin(alpha)X(theta) - cos(alpha)Ki(theta) for multiple alphas.
    Input: alpha_ar - Range of interpolation paramters for SPT ground states
           theta_ar - Array of VQE paramters
           Xavg_ar - Local magnetic field expectation values (as a function of theta) obtained via VQE.
           Kavg_ar - Cluster stabilizer expectation value (as a function of theta) obtained via VQE.
    Output: thetamin - theta that minimizes energy
    """
    thetamin_ar = []
    for alpha in alpha_ar:
        thetamin_ar.append(get_thetamin_VQE(alpha, theta_ar, Xavg_ar, Kavg_ar))
    return np.array(thetamin_ar)

def getT_VQE(theta):
    """
    Calculates coefficients a, b based on VQE paramter theta for the transformation T. 
    Input: theta - VQE paramter
    Output: acoeff - Coefficient in front of identity 
            bcoeff - Coefficient in front of identity.
    """
    acoeff = np.cos(theta)
    bcoeff = np.sin(theta)
    return acoeff, bcoeff

#  Fourier fitting of the VQE data

def fouriercoeff(theta_ar, T, n, ftheta_ar):
    """
    Obtain the nth fourier coefficient for a function f(theta).
    Input: theta_ar - Inputs to the function
           T - Period of the function. Should be the max of the thetas
           n - The nth coefficient
           ftheta_ar - The function f evaluated at each of the thetas
    """
    N = len(theta_ar)
    width = T / N
    scale = 2 * np.pi / T
    exp_ar = np.exp(-1j * n * scale * theta_ar)
    if n == 0:
        cn = scale /2/ np.pi * np.sum(width * ftheta_ar * exp_ar)
    else:
        cn = scale / np.pi * np.sum(width * ftheta_ar * exp_ar)
    return cn

def mult_fouriercoeff(theta_ar, T, maxn, ftheta_ar):
    """
    Produces up to the nth fourier coefficient of a function f(theta), starting at n = 0.
    Input: theta_ar - Inputs to the function
           T - Period of the function. Should be the max of the thetas
           maxn - Up to this n will the Fourier coefficients be calculated.
           ftheta_ar - The function f evaluated at each of the thetas    
    Output: cn_ar - Fourier coefficients ranging from the n = 0 coefficient to the maxn coefficient.
    """
    cn_ar = []
    for n in range(maxn+1):
        cn_ar.append(fouriercoeff(theta_ar, T, n, ftheta_ar))
    return np.array(cn_ar)

def fourierfit(theta_ar, n_ar, cn_ar):
    """
    Produces the fourier sum of a function (with fourier coefficients cns).
    Input: theta_ar - domain/inputs for the fourier fit function
           n_ar - the coefficients which are desired to be included in the fit
           cns - an array of fourier coefficients for the function
    Output: fhattheta_ar - Fourier fit approximation to f.
    """
    fhattheta_ar = np.zeros(len(theta_ar), dtype = 'complex128')
    for n in n_ar:
        fhattheta_ar += cn_ar[n] * np.exp(1j * n * 2 * theta_ar)
    return fhattheta_ar

# ADDED BY AMRIT FOR ZNE
def perform_avgK_1q(backend, best_qubits_one, best_qubits_three, shots_num, job_manager, theta, err_scale=1, meas_filter_one = 0, meas_filter_three = 0):
    """
    Performs circuits and calculates cluster state stabilizer expectation <K>i for VQE (function of theta)
    Input: backend - Which backend (simulator or device) to use
           best_qubits_one - which (1) qubit to use on the backend for single qubit circuits
           best_qubits_three - which (3) qubits to use on the backend for three qubit circuits
           shots_num - how many shots per experiment
           job_manager - IBMQ jobmanager
           theta - VQE parameter.
           meas_filter_one - One-qubit measurement error mitigation filter; by default is not applied.
           meas_filter_three - Three-qubit measurement error mitigation filter; by default is not applied
    Output: Kavg - Cluster state expectation value
            Kerr - Statistical error associated with above expectation
    """
    
    step1circ1 = generate_onequbit_circuit(theta, 0)
    step1circ2 = generate_onequbit_circuit(theta, 1)
    step3circ1 = generate_onequbit_circuit(theta, 0)
    step3circ2 = generate_onequbit_circuit(theta, 1)
    onequbitcircs = transpile([step1circ1, step1circ2, step3circ1, step3circ2], backend = backend, initial_layout=best_qubits_one)
    job_exp = job_manager.run(onequbitcircs, backend=backend, shots=shots_num, name='VQEK-1qubit')
    for i in range(len(job_exp.jobs())):
        job_monitor(job_exp.jobs()[i])
    print('All jobs have finished.')
    onequbitresults = job_exp.results()
    onequbitcounts0 = onequbitresults.get_counts(0)
    onequbitcounts1 = onequbitresults.get_counts(1)
    onequbitcounts2 = onequbitresults.get_counts(2)
    onequbitcounts3 = onequbitresults.get_counts(3)
    if meas_filter_one != 0:
        onequbitcounts0 = meas_filter_one.apply(onequbitcounts0)
        onequbitcounts1 = meas_filter_one.apply(onequbitcounts1)
        onequbitcounts2 = meas_filter_one.apply(onequbitcounts2)
        onequbitcounts3 = meas_filter_one.apply(onequbitcounts3)
    return onequbitcountlist = [onequbitcounts0, onequbitcounts1, onequbitcounts2, onequbitcounts3]
    

def circuits_avgK_3q(theta, sf_list):
    circ_list = []
    for err_scale in sf_list:
        step2circ1 = generate_threequbit_circuit(theta, 0, 0, err_scale)
        step2circ2 = generate_threequbit_circuit(theta, 0, 1, err_scale)
        step2circ3 = generate_threequbit_circuit(theta, 1, 0, err_scale)
        step2circ4 = generate_threequbit_circuit(theta, 1, 1, err_scale)
        circ_list.extend([step2circ1, step2circ2, step2circ3, step2circ4])
    return circ_list

def multiple_avgK_zne(backend, best_qubits_one, best_qubits_three, shots_num, job_manager, theta_ar, meas_filter_one = 0, meas_filter_three = 0):    
    """
    Performs circuits and calculates cluster state stabilizer expectation <K>i for VQE for multiple thetas
    Input: backend - Which backend (simulator or device) to use
           best_qubits_one - which (1) qubit to use on the backend for single qubit circuits
           best_qubits_three - which (3) qubits to use on the backend for three qubit circuits
           shots_num - how many shots per experiment
           job_manager - IBMQ jobmanager
           theta_ar - Array of VQE parameters (which will be iterated over).
           meas_filter_one - One-qubit measurement error mitigation filter; by default is not applied.
           meas_filter_three - Three-qubit measurement error mitigation filter; by default is not applied
    Output: Kavg - Cluster state expectation value
            Kerr - Statistical error associated with above expectation
    """
    Kavg_ar = []
    Kerr_ar = []
    for theta in theta_ar:
        onequbitcountlist = 
        Kavg, Kerr = calculate_avgK(onequbitcountlist, threequbitcountlist)
        # perform_avgK(backend, best_qubits_one, best_qubits_three, shots_num, job_manager, theta, meas_filter_one, meas_filter_three)
        
        Kavg_ar.append(Kavg)
        Kerr_ar.append(Kerr)
    return np.array(Kavg_ar), np.array(Kerr_ar)


'''
    
    threequbitcircs = transpile([step2circ1, step2circ2, step2circ3, step2circ4], backend=backend, initial_layout=best_qubits_three)
    job_exp = job_manager.run(threequbitcircs, backend=backend, shots=(shots_num+10), name='VQEK-3qubit')
    for i in range(len(job_exp.jobs())):
        job_monitor(job_exp.jobs()[i])
    print('All jobs have finished.')
    threequbitresults = job_exp.results()
    threequbitcounts0 = threequbitresults.get_counts(0)
    threequbitcounts1 = threequbitresults.get_counts(1)
    threequbitcounts2 = threequbitresults.get_counts(2)
    threequbitcounts3 = threequbitresults.get_counts(3)
    if meas_filter_three != 0:
        threequbitcounts0 = meas_filter_three.apply(threequbitcounts0)
        threequbitcounts1 = meas_filter_three.apply(threequbitcounts1)
        threequbitcounts2 = meas_filter_three.apply(threequbitcounts2)
        threequbitcounts3 = meas_filter_three.apply(threequbitcounts3)
    threequbitcountlist = [threequbitcounts0, threequbitcounts1, threequbitcounts2, threequbitcounts3]
    return calculate_avgK(onequbitcountlist, threequbitcountlist)
'''