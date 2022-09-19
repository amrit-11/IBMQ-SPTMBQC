import numpy as np
import rotcounterrot as rcr
from scipy.optimize import curve_fit
import pandas as pd

from qiskit import *
from qiskit.providers.ibmq.managed import IBMQJobManager
from qiskit.tools.monitor import job_monitor
from qiskit.ignis.mitigation.measurement import (complete_meas_cal, tensored_meas_cal,
                                                 CompleteMeasFitter, TensoredMeasFitter)


# caluclate the fit parameters and value of func at x=0
def extrapolate_res(func, xdata, ydata, p0=None, sigma=None):
    popt, pcov = curve_fit(func, xdata, ydata, p0, sigma)    
    X0 = func(0, *popt)
    return X0, popt, pcov


# run circuits on ibmq backend and return results list of dictionaries
def ibmq_executor_results(circuits, backend, job_manager, meas_cal=False, meas_fitter=None,
                          shots=8000, layout=None, opt_lvl=0, jobname='mbqc'):
    qc_trans = transpile(circuits, backend=backend, 
                         initial_layout=layout, optimization_level = opt_lvl)
    job_exp = job_manager.run(qc_trans, backend=backend, shots=shots, name=jobname)
    for j in np.arange(len(job_exp.jobs())):
        job_monitor(job_exp.jobs()[j])
    print('All jobs have finished.')
    exp_results = job_exp.results()
    
    res_list = []
    mit_res_list = []
    for k in range(len(circuits)):
        res_dict = exp_results.get_counts(k)
        res_list.append(res_dict)
        if meas_cal:
            meas_filter = meas_fitter.filter
            mit_res_dict = meas_filter.apply(res_dict)
            mit_res_list.append(mit_res_dict)
    return res_list, mit_res_list

# run circuits and measurement error cal at the same time
# dont use this if number of qubits exceeds 10
def ibmq_executor_results_meascal(n_qubits, circuits, backend, job_manager, cal_fitter=CompleteMeasFitter, shots=8000, layout=None, mit_pattern = None, opt_lvl=0, jobname='mbqc'):
    # create meascal circuits
    if cal_fitter == CompleteMeasFitter:
        meas_calib, state_labels = complete_meas_cal(qr=QuantumRegister(n_qubits), circlabel='mcal')
    elif cal_fitter == TensoredMeasFitter:
        meas_calib, state_labels = tensored_meas_cal(mit_pattern=mit_pattern, 
                                                      qr=QuantumRegister(n_qubits), circlabel='tcal')
    # transpile both meascal and experiment circuits
    meas_calibs_trans = transpile(meas_calib, backend=backend, initial_layout=layout)    
    qc_trans = transpile(circuits, backend=backend, 
                         initial_layout=layout, optimization_level = opt_lvl)
    
    # run all the jobs
    job_cal = job_manager.run(meas_calibs_trans, backend=backend, shots=shots, name='meascal')
    job_exp = job_manager.run(qc_trans, backend=backend, shots=shots, name=jobname)
    
    #track all jobs
    print("===== Measurement Error Calibration =====")
    for j in np.arange(len(job_cal.jobs())):
        job_monitor(job_cal.jobs()[j])
        print('Measurement Calibration jobs have finished.')
    
    print("========= Experiment Jobs ==========")
    for j in np.arange(len(job_exp.jobs())):
        job_monitor(job_exp.jobs()[j])
    print('All jobs have finished.')
    
    # extract and process results
    cal_res = job_cal.results()
    cal_res_comb = cal_res.combine_results()
    meas_fitter = cal_fitter(cal_res_comb, state_labels)
    exp_results = job_exp.results()
    
    res_list = []
    mit_res_list = []
    for k in range(len(circuits)):
        res_dict = exp_results.get_counts(k)
        res_list.append(res_dict)
        # these steps will take a long time for qubit nums > 10
        meas_filter = meas_fitter.filter
        mit_res_dict = meas_filter.apply(res_dict)
        mit_res_list.append(mit_res_dict)
    return res_list, mit_res_list, meas_fitter


# run circuits for measurement error calibration and return fitter object
def run_meas_calibration(n, backend, job_manager, cal_fitter = CompleteMeasFitter, 
                         shots=8000,layout=None,mit_pattern=None):
    if cal_fitter == CompleteMeasFitter:
        meas_calibs, state_labels = complete_meas_cal(qr=QuantumRegister(n), circlabel='mcal')
    elif cal_fitter == TensoredMeasFitter:
        meas_calibs, state_labels = tensored_meas_cal(mit_pattern=mit_pattern, 
                                                      qr=QuantumRegister(n), circlabel='tcal')
    meas_calibs = transpile(meas_calibs, backend=backend, initial_layout=layout)
    job_cal = job_manager.run(meas_calibs, backend=backend, shots=shots)
    print("===== Measurement Error Calibration =====")
    for j in np.arange(len(job_cal.jobs())):
        job_monitor(job_cal.jobs()[j])
    print('Measurement Calibration jobs have finished.')
    cal_res = job_cal.results()
    cal_res_comb = cal_res.combine_results()
    meas_fitter = cal_fitter(cal_res_comb, state_labels)
    return meas_fitter

def repeat_circuits(circ_list, num_reps):
    rep_list = []
    for circ in circ_list:
        for i in range(num_reps):
            rep_list.append(circ)
    return rep_list


def linear(x, a, b):
    return a*x + b

def quadratic(x, a, c):
    return -a**2*x**2 + c

def tanhyp(x, a, b):
    return 0.5*(np.tanh(a*x+b) + 1)

def expfit(x, a, b, c):
    return a*np.exp(-x/b) + c

def expfit2(x, a, c):
    return np.exp(-a**2 * x - c**2)

def expquadfit(x, a, c):
    return np.exp(- a**2 * x**2 - c**2)

def expquadlinfit(x, a, b, c):
    return np.exp(- a**2 * x**2 - b**2 * x - c**2)


def rcr_X_from_counts(res_list, beta, coeff_a, coeff_b, num_sf, num_reps=1):
    
    num_coeff = len(coeff_a)
    if num_reps==1:
        repeat_circs = False
    elif num_reps>1:
        repeat_circs = True
    
    res_dflist = []
    
    if repeat_circs:
        X_exp = np.zeros((num_sf,num_coeff,num_reps))
        X_err = np.zeros((num_sf,num_coeff,num_reps))
    else:
        X_exp = np.zeros((num_sf,num_coeff))
        X_err = np.zeros((num_sf,num_coeff))

    for i in range(len(res_list)//4):
        if repeat_circs:
            rep_num = i % num_reps 
            coeff_ind = (i // num_reps) % num_coeff
            sf_ind = i // (num_reps * num_coeff)
        else:
            coeff_ind = i % num_coeff
            sf_ind = i // num_coeff
        
        a = coeff_a[coeff_ind]
        b = coeff_b[coeff_ind]
        
        start_ind = i+((3*num_reps)*(i//num_reps))
        res_to_process = res_list[start_ind:start_ind+4*num_reps:num_reps]
        X_r, Xerr_r = rcr.post_process_results(res_to_process, beta, a, b)
        
        if repeat_circs:
            X_exp[sf_ind,coeff_ind,rep_num] = X_r
            X_err[sf_ind,coeff_ind,rep_num] = Xerr_r
            header = pd.MultiIndex.from_product([[str(sf_ind)],[str(coeff_ind)],[str(rep_num)]],
                                        names=['sf_ind','coeff_ind','rep_num'])
        else:
            X_exp[sf_ind,coeff_ind] = X_r
            X_err[sf_ind,coeff_ind] = Xerr_r
            header = pd.MultiIndex.from_product([[str(sf_ind)],[str(coeff_ind)]],
                                        names=['sf_ind','coeff_ind'])

        res_dflist.append(pd.DataFrame.from_dict(res_list[i],orient='index',columns=header))
    
    df_res = pd.concat(res_dflist, axis=1)
    return X_exp, X_err, df_res


