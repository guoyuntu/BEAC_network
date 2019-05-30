import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import rc

with PdfPages('/users/appletgj/desktop/attribution/attribution_across_datasets.pdf') as pdf:
	fig = plt.figure()
	x1 = [1, 2, 3, 4, 5, 6, 7, 8, 9]
	y1 = [100, 100, 100, 99.71, 98.32, 95.33, 92.53, 88.52, 81.53]
	y2 = [62.16, 52.02, 45.95, 39.06, 34.40, 27.97, 25.97, 21.30, 18.31]
	y3 = [35.36, 26.65, 16.0, 9.86, 6.58, 3.91, 2.99, 2.23, 1.62]

	group_labels = ['0.2', '0.25', '0.3', '0.35', '0.4', '0.45', '0.5']  
	
	plt.rc('text', usetex=True)
	plt.rc('font', family='serif')
	
	plt.title('Emotion Attribution Across Two Datasets of BEAC-net')  
	plt.xlabel(r'\textit{tIou} threshold')  
	plt.ylabel(r'mAPs(\%)')

	plt.plot(x1, y1,'o-', color='darkred', label='Emotion6 Video', alpha=0.7, linewidth=3)  
	plt.plot(x1, y2,'o-', color='olive', label='Ekman6(two classes)', alpha=0.7, linewidth=3) 
	plt.plot(x1, y3,'o-', color='deepskyblue', label='Ekman6(full dataset)', alpha=0.7, linewidth=3)
	plt.xticks(x1, group_labels, rotation=0)

	plt.legend(loc='upper center', bbox_to_anchor=(0.75,0.75),ncol=1,fancybox=False,shadow=False)
	plt.show()
	pdf.savefig(fig)