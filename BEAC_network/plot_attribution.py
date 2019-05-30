import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import rc

with PdfPages('/users/appletgj/desktop/attribution_emotion6.pdf') as pdf:
	fig = plt.figure()
	x1 = [1, 2, 3, 4, 5]
	y1 = [100, 100, 95.00, 81.60, 62]
	y2 = [99, 91.10, 83.0, 63.0, 39.0]
	y3 = [87.0, 80.2, 74.6, 57.0, 35.6]
	y4 = [91, 91.5, 92.0, 91.6, 88.9]

	group_labels = ['0.3', '0.4', '0.5', '0.6', '0.7']  
	
	plt.rc('text', usetex=True)
	plt.rc('font', family='serif')
	
	plt.title('Emotion Attribution on Emotion6 Video Dataset')  
	plt.xlabel(r'\textit{tIou} threshold')  
	plt.ylabel(r'mAPs(\%)')  

	plt.plot(x1, y1,'o-', color='#99CC01', label='BEAC-Net', alpha=0.7, linewidth=3)  
	plt.plot(x1, y2,'o-', color='#0D81CF', label='E-Stream', alpha=0.7, linewidth=3) 
	plt.plot(x1, y3,'o-', color='#FA4D3D', label='ITE', alpha=0.7, linewidth=3)
	plt.plot(x1, y4,'o-', color='#34495E', label='SVM', alpha=0.7, linewidth=3)
	plt.xticks(x1, group_labels, rotation=0)  

	plt.legend(loc='upper center', bbox_to_anchor=(0.75,1),ncol=2,fancybox=False,shadow=False)
	plt.show()
	pdf.savefig(fig)

with PdfPages('/users/appletgj/desktop/attribution_ekman6.pdf') as pdf:
	fig = plt.figure()
	x1 = [1,2,3,4,5]
	y1 = [72.50, 60.40, 45.90, 34.40, 26.00]
	y2 = [80.00, 56.50, 40.30, 31.60, 23.60]
	y3 = [62.8, 33.4, 14.2, 5.8, 4.2]
	y4 = [23.1, 11.3, 4.7, 2.4, 1.1]

	group_labels = ['0.1', '0.2', '0.3', '0.4', '0.5']  
	plt.title('Emotion Attribution on Ekman6 Dataset(two classes)')  
	plt.xlabel(r'\textit{tIou} threshold')  
	plt.ylabel(r'mAPs(\%)') 

	plt.plot(x1, y1,'o-', color='#99CC01', label='BEAC-Net', alpha=0.7, linewidth=3)  
	plt.plot(x1, y2,'o-', color='#0D81CF', label='E-Stream', alpha=0.7, linewidth=3) 
	plt.plot(x1, y3,'o-', color='#FA4D3D', label='ITE', alpha=0.7, linewidth=3)
	plt.plot(x1, y4,'o-', color='#34495E', label='SVM', alpha=0.7, linewidth=3)
	plt.xticks(x1, group_labels, rotation=0)  
	plt.ylim(0,100)

	plt.legend(loc='upper center', bbox_to_anchor=(0.75,1),ncol=2,fancybox=False,shadow=False)
	plt.show()  
	pdf.savefig(fig)

with PdfPages('/users/appletgj/desktop/attribution_ekman2.pdf') as pdf:
	fig = plt.figure()
	x1 = [1, 2, 3, 4, 5]
	y1 = [60.0,35.4,12.5,6.6,2.6]
	y2 = [58, 27, 12, 6, 2.5]
	y3 = [44.3, 28.2, 11.7, 5.1, 2.1]
	y4 = [14.9, 8.1, 4.1, 2.0, 0.9]

	group_labels = ['0.1', '0.2', '0.3', '0.4', '0.5']  
	plt.title('Emotion Attribution on Ekman6 Dataset(full dataset)')  
	plt.xlabel(r'\textit{tIou} threshold') 
	plt.ylabel(r'mAPs(\%)')  

	plt.plot(x1, y1,'o-', color='#99CC01', label='BEAC-Net', alpha=0.7, linewidth=3)  
	plt.plot(x1, y2,'o-', color='#0D81CF', label='E-Stream', alpha=0.7, linewidth=3) 
	plt.plot(x1, y3,'o-', color='#FA4D3D', label='ITE', alpha=0.7, linewidth=3)
	plt.plot(x1, y4,'o-', color='#34495E', label='SVM', alpha=0.7, linewidth=3)
	plt.xticks(x1, group_labels, rotation=0)  

	plt.legend(loc='upper center', bbox_to_anchor=(0.75,1),ncol=2,fancybox=False,shadow=False)
	plt.show()  
	pdf.savefig(fig)

