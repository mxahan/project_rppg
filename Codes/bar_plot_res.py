#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 00:54:21 2020

@author: zahid
"""
import matplotlib.pyplot as plt
import matplotlib
# matplotlib.use('PS')

# plt.style.use('ggplot')


#%%

x = ['Nuclear', 'Hydro', 'Gas', 'Oil', 'Coal', 'Biofuel']
energy = [5, 6, 15, 22, 24, 8]

x_pos = [i for i, _ in enumerate(x)]

plt.bar(x_pos, energy, color='green')
plt.xlabel("Energy Source")
plt.ylabel("Energy Output (GJ)")
plt.title("Energy output from various fuel sources")

plt.xticks(x_pos, x)

plt.show()

#%% Real Deal Starts Here



import numpy as np

N = 3
tot_peak = [69, 69, 69]
TP = np.array([61, 62, 62])/69*100
FP = np.array([9, 9, 6])/69*100
FN = np.array([8, 7, 6])/69*100

fig = plt.figure(figsize=(19.20,10.80))


ax4 = fig.add_subplot(121)

ind = np.arange(N) 
width = 0.5      
plt.bar(ind, TP, width/2.5)
plt.bar(ind + width/2, FP, width/2.5)
plt.bar(ind + 2*width/2, FN, width/2.5)

plt.yticks(fontsize=50)
plt.ylabel('Percentage', fontweight="bold", fontsize=50)
plt.title('In house dataset',fontsize=40, fontweight="bold")

plt.xlabel("Collected Dataset", fontsize=40, fontweight="bold")

xlabels = ('Personal', 'Tx from\n person', 'Tx from \n MTL')

plt.xticks(ind + width/3, xlabels, fontsize=50)
ax4.set_xticklabels(xlabels, rotation=45, fontsize=35)

# plt.legend(loc='best', ncol=3, fontsize=25)
# plt.show()

N = 2
tot_peak = (72,72)
TP = np.array([63, 68])/72*100
FP = np.array([8, 6])/72*100
FN = np.array([9, 4])/72*100

ax2 = fig.add_subplot(122)

ind = np.arange(N)      
width = 0.5    
plt.bar(ind, TP, width/4, label='TP peaks')
plt.bar(ind + width/3, FP, width/4, label='FP')
plt.bar(ind + 2*width/3, FN, width/4, label='FN')

plt.yticks(fontsize=50)

# plt.ylabel('Scores')
plt.title('UBFC-RPPG dataset', fontsize=40, fontweight="bold")

plt.xlabel("UBFC-rPPG Dataset", fontsize=40, fontweight="bold")


xlabels = ( 'Tx from \n person', 'Tx from \n MTL')

plt.xticks(ind + width/3, xlabels)
ax2.set_xticklabels(xlabels, rotation=45, fontsize=35)

plt.legend(loc='best', ncol=3, fontsize=30)

plt.savefig('bar_chart_sample1.eps', format = 'eps', dpi= 500, bbox_inches="tight")

plt.show()



#%%





fig = plt.figure(figsize=(19.20,10.80))
ax1 = fig.add_subplot(121)

N = 3
tot_peak = [104, 104, 104]
TP = np.array([98, 75, 63])/104*100
FP = np.array([8, 19, 33])/104*100
FN = np.array([6, 29, 41])/104*100

ind = np.arange(N) 
width = 0.5      
plt.bar(ind, TP, width/2.5)
plt.bar(ind + width/2, FP, width/2.5)
plt.bar(ind + 2*width/2, FN, width/2.5)

plt.yticks(fontsize=30)
plt.ylabel('Percentage', fontweight="bold", fontsize=40)
plt.title('Ablation Study for MERL',fontsize=40, fontweight="bold")

plt.xlabel("MERL dataset", fontsize=40, fontweight="bold")

xlabels = ('tanh+MSE\n+ Sign loss', 'ReLU+MSE\n+ Sign loss', 'tanh+MSE\n')

plt.xticks(ind + width/3, xlabels, fontsize=40)
ax1.set_xticklabels(xlabels, rotation=45, fontsize=35)

ax1 = fig.add_subplot(122)

N = 3
tot_peak = [201, 201, 201]
TP = np.array([172, 172, 170])/201*100
FP = np.array([31, 31, 33])/201*100
FN = np.array([29, 29, 31])/201*100

ind = np.arange(N) 
width = 0.5      
plt.bar(ind, TP, width/2.5, label = "TP")
plt.bar(ind + width/2, FP, width/2.5, label = "FP")
plt.bar(ind + 2*width/2, FN, width/2.5, label = "FN")
plt.yticks(fontsize=30)

# plt.ylabel('Percentage', fontweight="bold", fontsize=30)
plt.title('MERL personalized Result',fontsize=40, fontweight="bold")

plt.xlabel("MERL dataset", fontsize=40, fontweight="bold")

xlabels = ('RGB', 'RGB \n demosai.', 'NIR')

plt.xticks(ind + width/3, xlabels, fontsize=40)
ax1.set_xticklabels(xlabels, rotation=45, fontsize=35)

plt.legend(loc='best', ncol=3, fontsize=30)

plt.savefig('bar_chart_sample2.eps', format = 'eps', dpi= 500, bbox_inches="tight")

plt.show()




#%%


N = 5
tot_peak = [103, 103, 103, 103, 103]
TP = np.array([98, 99, 97, 97, 97])/103*100
FP = np.array([5, 4, 9, 8, 3])/103*100
FN = np.array([5, 4, 6, 6, 6])/103*100

fig = plt.figure(figsize=(19.20,10.80))


ax1 = fig.add_subplot()

ind = np.arange(5) 
width = 0.5      
plt.bar(ind, TP, width/2.5, label = "TP")
plt.bar(ind + width/2, FP, width/2.5, label = "FP")
plt.bar(ind + 2*width/2, FN, width/2.5, label = "FN")

plt.yticks(fontsize=40)
plt.ylabel('Percentage', fontweight="bold", fontsize=40)
plt.title('Transfer result for MERL',fontsize=40, fontweight="bold")

plt.xlabel("MERL dataset", fontsize=30, fontweight="bold")

xlabels = ('RGB Tx', 'RGB to\n NIR', 'RGB to \n demos.', 'RGB MTL', 'NIR MTL')

plt.xticks(ind + width/3, xlabels, fontsize=40)
ax1.set_xticklabels(xlabels, rotation=45, fontsize=35)

plt.legend(loc='best', ncol=3, fontsize=30)

plt.savefig('bar_chart_sample3.eps', bbox_inches="tight", format = 'eps', dpi= 500)
# 
plt.show()

#%%  Horizontal plot totally sepeate here






import numpy as np

N = 3
tot_peak = [69, 69, 69]
TP = np.array([61, 62, 62])/69*100
FP = np.array([9, 9, 6])/69*100
FN = np.array([8, 7, 6])/69*100

fig = plt.figure(figsize=(19.20,10.80))


ax4 = fig.add_subplot(121)

ind = np.arange(N) 
width = 0.5      
plt.barh(ind, TP, width/2.5)
plt.barh(ind + width/2, FP, width/2.5)
plt.barh(ind + 2*width/2, FN, width/2.5)

plt.xticks(fontsize=30)
plt.xlabel('Percentage', fontweight="bold", fontsize=40)
plt.title('In house dataset',fontsize=40, fontweight="bold")

# plt.ylabel("Collected Dataset", fontsize=40, fontweight="bold")

xlabels = ('Personal', 'Tx from\n person', 'Tx from \n MTL')

plt.yticks(ind + width/3, xlabels, fontsize=40)
ax4.set_yticklabels(xlabels, rotation=90, fontsize=35)

# plt.legend(loc='best', ncol=3, fontsize=25)
# plt.show()

N = 2
tot_peak = (72,72)
TP = np.array([63, 68])/72*100
FP = np.array([8, 6])/72*100
FN = np.array([9, 4])/72*100

ax2 = fig.add_subplot(122)

ind = np.arange(N)      
width = 0.5    
plt.barh(ind, TP, width/4, label='TP peaks')
plt.barh(ind + width/3, FP, width/4, label='FP')
plt.barh(ind + 2*width/3, FN, width/4, label='FN')

plt.xticks(fontsize=30)
plt.xlabel('Percentage', fontweight="bold", fontsize=40)

# plt.ylabel('Scores')
plt.title('UBFC-RPPG dataset', fontsize=40, fontweight="bold")

# plt.ylabel("UBFC-rPPG Dataset", fontsize=40, fontweight="bold")


xlabels = ( 'Tx from \n person', 'Tx from \n MTL')

plt.yticks(ind + width/2, xlabels)
ax2.set_yticklabels(xlabels, rotation=90, fontsize=35)

plt.legend(loc='upper right', ncol=1, fontsize=30)

# plt.savefig('bar_chart_sample1.eps', format = 'eps', dpi= 500, bbox_inches="tight")

plt.show()



#%%





fig = plt.figure(figsize=(19.20,10.80))
ax1 = fig.add_subplot(121)

N = 3
tot_peak = [104, 104, 104]
TP = np.array([98, 75, 63])/104*100
FP = np.array([8, 19, 33])/104*100
FN = np.array([6, 29, 41])/104*100

ind = np.arange(N) 
width = 0.5      
plt.barh(ind, TP, width/2.5)
plt.barh(ind + width/2, FP, width/2.5)
plt.barh(ind + 2*width/2, FN, width/2.5)

plt.xticks(fontsize=30)
plt.xlabel('Percentage', fontweight="bold", fontsize=40)
plt.title('Ablation Study for MERL',fontsize=40, fontweight="bold")

plt.ylabel("MERL dataset", fontsize=40, fontweight="bold")

xlabels = ('tanh+MSE\n+ Sign loss', 'ReLU+MSE\n+ Sign loss', 'tanh+MSE\n')

plt.yticks(ind + width/3, xlabels, fontsize=40)
ax1.set_yticklabels(xlabels, rotation=90, fontsize=35)

ax1 = fig.add_subplot(122)

N = 3
tot_peak = [201, 201, 201]
TP = np.array([172, 172, 170])/201*100
FP = np.array([31, 31, 33])/201*100
FN = np.array([29, 29, 31])/201*100

ind = np.arange(N) 
width = 0.5      
plt.barh(ind, TP, width/2.5, label = "TP")
plt.barh(ind + width/2, FP, width/2.5, label = "FP")
plt.barh(ind + 2*width/2, FN, width/2.5, label = "FN")
plt.xticks(fontsize=30)

# plt.ylabel('Percentage', fontweight="bold", fontsize=30)
plt.title('MERL personalized Result',fontsize=40, fontweight="bold")

plt.xlabel("MERL dataset", fontsize=40, fontweight="bold")

xlabels = ('RGB', 'RGB \n demosai.', 'NIR')

plt.yticks(ind + width/3, xlabels, fontsize=40)
ax1.set_yticklabels(xlabels, rotation=90, fontsize=35)

plt.legend(loc='best', ncol=3, fontsize=30)

# plt.savefig('bar_chart_sample2.eps', format = 'eps', dpi= 500, bbox_inches="tight")

plt.show()


#%%
N = 5
tot_peak = [103, 103, 103, 103, 103]
TP = np.array([98, 99, 97, 97, 97])/103*100
FP = np.array([5, 4, 9, 8, 3])/103*100
FN = np.array([5, 4, 6, 6, 6])/103*100

fig = plt.figure(figsize=(19.20,10.80))


ax1 = fig.add_subplot()

ind = np.arange(5) 
width = 0.5      
plt.barh(ind, TP, width/2.5, label = "TP")
plt.barh(ind + width/2, FP, width/2.5, label = "FP")
plt.barh(ind + 2*width/2, FN, width/2.5, label = "FN")

plt.xticks(fontsize=30)
plt.xlabel('Percentage', fontweight="bold", fontsize=40)
plt.title('Transfer result for MERL',fontsize=40, fontweight="bold")

# plt.ylabel("MERL dataset", fontsize=30, fontweight="bold")

xlabels = ('RGB Tx', 'RGB to\n NIR', 'RGB to \n demos.', 'RGB MTL', 'NIR MTL')

plt.yticks(ind + width, xlabels, fontsize=40)
ax1.set_yticklabels(xlabels, rotation=0, fontsize=35)

plt.legend(loc='best', ncol=3, fontsize=30)

# plt.savefig('bar_chart_sample3.eps', bbox_inches="tight", format = 'eps', dpi= 500)
# 
# plt.show()
