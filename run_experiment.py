from LQGSystem import LQGSystem
from dual_SDP import SDPProblem
from dual_frankwolfe import FrankWolfeOptimizer
import pickle
import numpy as np
import scipy as sp
import time

SDP_data = {}
fw_data = {}
for T in range(1, 101):
    print("--" * 20)
    print("Running experiment for T =", T)
    lqg = LQGSystem(n_x=1, n_u=1, n_y=1, T=T)
    if T <= 50:
        time_start = time.time()
        sdp = SDPProblem(lqg)
        optimal_value_SDP, mu_opt_SDP, M_opt_SDP, K_opt_SDP, N_opt_SDP, L_opt_SDP, Lambda_opt_SDP = sdp.solve()
        Sigma_opt_SDP = M_opt_SDP - mu_opt_SDP @ mu_opt_SDP.T
        time_SDP = time.time() - time_start


    time_start = time.time()
    fw = FrankWolfeOptimizer(lqg, max_iter=100, delta=0.9, tol=1e-2)
    optimal_value_fw, mu_opt_fw, Sigma_opt_fw = fw.optimize()
    time_fw = time.time() - time_start

    #print("mu diff:", np.max(np.abs(mu_opt_SDP - lqg.mu_hat)))
    #print("sigma diff:", np.max(np.abs(Sigma_opt_SDP - lqg.Sigma_hat)))# - 2 * sp.linalg.sqrtm(sp.linalg.sqrtm(sdp.Sigma_hat) @ Sigma_opt @ sp.linalg.sqrtm(sdp.Sigma_hat)))) )
    #print("squared gelbrich distance:", np.sum(np.linalg.norm(mu_opt_SDP - lqg.mu_hat)**2 + np.linalg.norm(np.trace(Sigma_opt_SDP + sdp.LQG.Sigma_hat - 2 * sp.linalg.sqrtm(sp.linalg.sqrtm(lqg.Sigma_hat) @ Sigma_opt_SDP @ sp.linalg.sqrtm(lqg.Sigma_hat))))) )

    #print(Sigma_opt)
    #print(mu_opt)
    #print("Optimal value:", optimal_value_SDP)
    if T <= 50:
        #assert np.all(np.isclose(optimal_value_SDP, optimal_value_fw, atol=1e-3)), f"Optimal values differ: {optimal_value_SDP} vs {optimal_value_fw}"
        #assert np.all(np.isclose(mu_opt_SDP, mu_opt_fw, atol=1e-2)), f"Optimal means differ: {np.max(np.abs(mu_opt_SDP - mu_opt_fw))}"
        #assert np.all(np.isclose(Sigma_opt_SDP, Sigma_opt_fw, atol=1e-2)), f"Optimal covariances differ: {np.max(np.abs(mu_opt_SDP - mu_opt_fw))}"
        SDP_data[T] = {
            "optimal_value": optimal_value_SDP,
            "mu_opt": mu_opt_SDP,
            "Sigma_opt": Sigma_opt_SDP,
            "time": time_SDP}
    fw_data[T] = {
        "optimal_value": optimal_value_fw,
        "mu_opt": mu_opt_fw,
        "Sigma_opt": Sigma_opt_fw,
        "time": time_fw}
    if T%10==0:
        pickle.dump(SDP_data, open("SDP_data.pkl", "wb"))
        pickle.dump(fw_data, open("fw_data.pkl", "wb"))
# Save the results in .pkl format
# Save the results in .npy format
#pickle.load(open("optimal_values_SDP.pkl", "rb"))
#pickle.load(open("optimal_means_SDP.pkl", "rb"))