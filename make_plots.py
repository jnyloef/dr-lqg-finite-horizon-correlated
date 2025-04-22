import pickle
import matplotlib.pyplot as plt
SDP_data = pickle.load(open("SDP_data.pkl", "rb"))
fw_data = pickle.load(open("fw_data.pkl", "rb"))
# Plot the results
T_values_SDP = list(SDP_data.keys())
optimal_values_SDP = [SDP_data[T]["optimal_value"] for T in T_values_SDP]
T_values_fw = list(fw_data.keys())
optimal_values_fw = [fw_data[T]["optimal_value"] for T in T_values_fw]
run_times_SDP = [SDP_data[T]["time"] for T in T_values_SDP]
run_times_fw = [fw_data[T]["time"] for T in T_values_fw]

plt.figure(figsize=(12, 6))

# Plot optimal values
plt.subplot(1, 2, 1)
plt.plot(T_values_SDP, optimal_values_SDP, label="SDP", marker='o')
plt.plot(T_values_fw, optimal_values_fw, label="Frank-Wolfe", marker='o')
plt.xlabel("Time Horizon T")
plt.ylabel("Optimal Value")
#plt.title("Optimal Value vs Time Horizon T")
plt.legend()
plt.grid(which='both', linestyle='--', linewidth=0.5)

# Plot run times
plt.subplot(1, 2, 2)
plt.plot(T_values_SDP, run_times_SDP, label="SDP", marker='o')
plt.plot(T_values_fw, run_times_fw, label="Frank-Wolfe", marker='o')
plt.xlabel("Time Horizon T")
plt.xscale('log')
plt.ylabel("Run Time (s)")
plt.yscale('log')
#plt.title("Run Time vs Time Horizon T")
plt.legend()
plt.tight_layout()
plt.grid(which='both', linestyle='--', linewidth=0.5)

#save as pdf figure
plt.savefig("results.pdf", format='pdf', bbox_inches='tight')
plt.show()
