import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
import matplotlib

# matplotlib.use('TkAgg')
import time
from sklearn.manifold import TSNE
from matplotlib.ticker import NullFormatter
font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 22}

matplotlib.rc('font', **font)

def plot_tsne_embeddings(data, labels, classes, title, saving_path):
	print("Plotting TSNE embeddings...")
	t0 = time.time()
	fig = plt.figure(figsize=(15, 8))
	tsne = TSNE(n_components=2, init='pca', verbose=1, random_state=0)  # print_interval=1e6) #, random_state=0)
	Y = tsne.fit_transform(data)
	t1 = time.time()
	
	print("t-SNE: %.2g sec" % (t1 - t0))
	ax = fig.add_subplot(1, 1, 1)
	for i, emotion in enumerate(classes):
		y_labels = np.where(labels == i)[0]
		y_data = Y[y_labels, :]
		plt.scatter(y_data[:, 0], y_data[:, 1], label=classes[i])
	# handles, _ = plt.scatter.legend_elements(prop='colors')
	plt.legend(loc='best', prop={'size': 16}) # fontsize = 'x-large')
	# plt.title(title)
	ax.xaxis.set_major_formatter(NullFormatter())
	ax.yaxis.set_major_formatter(NullFormatter())
	plt.axis('tight')
	plt.axis('off')
	plt.tight_layout()
	plt.gcf()
	print(saving_path)
	plt.savefig(saving_path, transparent=True)
	plt.show()
	plt.close()
	


def plot_confusion_matrix(y_true, y_pred, classes, configs,
                          normalize=True,
                          title=None,
                          cmap=plt.cm.GnBu):
	"""
	This function prints and plots the confusion matrix.
	Normalization can be applied by setting `normalize=True`.
	"""
	if not title:
		if normalize:
			title = 'Normalized confusion matrix'
		else:
			title = 'Confusion matrix, without normalization'
	
	# Compute confusion matrix
	cm = confusion_matrix(y_true, y_pred) * 100
	# Only use the labels that appear in the data
	classes = np.array(classes)
	classes = classes[unique_labels(y_true, y_pred)]
	if normalize:
		cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
		print("Normalized confusion matrix")
	else:
		print('Confusion matrix, without normalization')
	
	print(cm)
	
	fig, ax = plt.subplots()
	im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
	# ax.figure.colorbar(im, ax=ax)
	# We want to show all ticks...
	ax.set(xticks=np.arange(cm.shape[1]),
	       yticks=np.arange(cm.shape[0]),
	       # ... and label them with the respective list entries
	       xticklabels=classes, yticklabels=classes)
	# title=title,
	# ylabel='True label',
	# xlabel='Predicted label')
	
	# Rotate the tick labels and set their alignment.
	plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
	         rotation_mode="anchor")
	
	# Loop over data dimensions and create text annotations.
	fmt = '.2f' if normalize else 'd'
	cm = cm * 100
	thresh = cm.max() / 2.
	for i in range(cm.shape[0]):
		for j in range(cm.shape[1]):
			ax.text(j, i, format(cm[i, j], fmt),
			        ha="center", va="center",
			        color="white" if cm[i, j] > thresh else "black")
	fig.tight_layout()
	fig.savefig('results/' + '{}_{}_confusion_matrix_results.pdf'.format(configs['dataset'],
	                                                                     configs['branch']),
	            transparent=True)
	plt.show()
	plt.pause(interval=5)
	plt.close()
	return ax, fig
