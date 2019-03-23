import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def plot(nelbo, kl, rec):
	epochs = np.arange(1, len(nelbo) + 1) * 50
	
	plt.plot(epochs, nelbo)
	plt.title("Negative ELBO vs. epoch")
	plt.xlabel("Epochs")
	plt.ylabel("NELBO")
	plt.savefig("NELBO")
	plt.close()

	plt.plot(epochs, kl)
	plt.title("KL-divergence vs. epoch")
	plt.xlabel("Epochs")
	plt.ylabel("KL")
	plt.savefig("KL")
	plt.close()

	plt.plot(epochs, rec)
	plt.title("Reconstruction loss vs. epoch")
	plt.xlabel("Epochs")
	plt.ylabel("Rec Loss")
	plt.savefig("Rec")



def main():
	nelbo = []
	kl = []
	rec = []
	
	for i in range(50, 20050, 50):
		summaries = pd.read_csv('summaries/summaries_epoch_' + str(i) + '.csv', header=None) 
		nelbo.append(float(summaries.iloc[0][1]))
		kl.append(float(summaries.iloc[1][1]))
		rec.append(float(summaries.iloc[2][1]))

	plot(nelbo, kl, rec)


if __name__ == '__main__':
	main()