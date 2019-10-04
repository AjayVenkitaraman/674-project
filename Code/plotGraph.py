import re
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

with open("softmax_results.txt",'r') as f:
	content = f.readlines()
y = []
x = []
cnt = 20
for line in content:
	if "tensorflow:step:" in line and "cost" in line:
		cost = re.findall(r"cost:\s[0-9]+\.[0-9]*", line)
		y.append(float(cost[0].split(":")[-1][1:]))
		x.append(cnt)
		cnt += 20

print ("Plotting the graphs\n")
plt.plot(x, y , 'b')
plt.xlabel("Epochs")
plt.ylabel("Loss function value")
plt.title("Plot of Loss function value vs Epochs")
plt.savefig("cost.png")
plt.close()
y = []
x = []
cnt = 20
for line in content:
	if "tensorflow:step:" in line and "train_time_taken" in line:
		cost = re.findall(r"train_time_taken:\s[0-9]+\.[0-9]*", line)
		y.append(float(cost[0].split(":")[-1][1:]))
		x.append(cnt)
		cnt += 20

plt.plot(x, y , 'b')
plt.xlabel("Epochs")
plt.ylabel("Time Taken")
plt.title("Plot of Time Taken vs Epochs")
plt.savefig("time.png")
