#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style

#%%
# Read company metadata
c_dataframe = pd.read_csv(r'C:\Users\Public\Data\honestCompanyMetadata_6.csv')

#%%
axis_fontsize = 8
title_fontsize = 10
fig_title_fontsize = 16
bold_wt = 'bold'

#%%
style.use('ggplot')
fig1 = plt.figure(1)
fig1.suptitle('Distribution of number of employees', fontsize=fig_title_fontsize, fontweight=bold_wt)

#%%
size = 'VERY SMALL'

series = c_dataframe.loc[c_dataframe.loc[:, 'emp_base'] == size].loc[:, 'num_employees']

ax1 = fig1.add_subplot(321)
style.use('ggplot')
data = ax1.hist(series, bins=10)
ax1.set_title('Employee base category - VERY SMALL', fontsize=title_fontsize)
ax1.set_xlabel('Number of employees', fontsize=axis_fontsize)
ax1.set_ylabel('Frequency', fontsize=axis_fontsize)
text = 'Min: ' + str(np.min(series)) + ' Max: ' + str(np.max(series)) + ' Median: ' + str(np.median(series)) 
ax1.text(5,max(data[0]), text, fontsize=axis_fontsize)


#%%
size = 'SMALL'

series = c_dataframe.loc[c_dataframe.loc[:, 'emp_base'] == size].loc[:, 'num_employees']
style.use('ggplot')
ax2 = fig1.add_subplot(322)
data = ax2.hist(series, bins=11)
ax2.set_title('Employee base category - SMALL', fontsize=title_fontsize)
ax2.set_xlabel('Number of employees', fontsize=axis_fontsize)
ax2.set_ylabel('Frequency', fontsize=axis_fontsize)
text = 'Min: ' + str(np.min(series)) + ' Max: ' + str(np.max(series)) + ' Median: ' + str(np.median(series)) 
ax2.text(5,max(data[0]), text, fontsize=axis_fontsize)

#%%
size = 'MEDIUM'

series = c_dataframe.loc[c_dataframe.loc[:, 'emp_base'] == size].loc[:, 'num_employees']
style.use('ggplot')
ax3 = fig1.add_subplot(323)
data = ax3.hist(series, bins=11)
ax3.set_title('Employee base category - MEDIUM', fontsize=title_fontsize)
ax3.set_xlabel('Number of employees', fontsize=axis_fontsize)
ax3.set_ylabel('Frequency', fontsize=axis_fontsize)
text = 'Min: ' + str(np.min(series)) + ' Max: ' + str(np.max(series)) + ' Median: ' + str(np.median(series)) 
ax3.text(5,max(data[0]), text, fontsize=axis_fontsize)

#%%

size = 'LARGE'

series = c_dataframe.loc[c_dataframe.loc[:, 'emp_base'] == size].loc[:, 'num_employees']

style.use('ggplot')
ax4 = fig1.add_subplot(324)
data = ax4.hist(series, bins=11)
ax4.set_title('Employee base category - LARGE', fontsize=title_fontsize)
ax4.set_xlabel('Number of employees', fontsize=axis_fontsize)
ax4.set_ylabel('Frequency', fontsize=axis_fontsize)
text = 'Min: ' + str(np.min(series)) + ' Max: ' + str(np.max(series)) + ' Median: ' + str(np.median(series)) 
ax4.text(5,max(data[0]), text, fontsize=axis_fontsize)

#%%
size = 'VERY LARGE'

series = c_dataframe.loc[c_dataframe.loc[:, 'emp_base'] == size].loc[:, 'num_employees']
style.use('ggplot')
ax5 = fig1.add_subplot(325)
data= ax5.hist(series, bins=11)
ax5.set_title('Employee base category - VERY LARGE', fontsize=title_fontsize)
ax5.set_xlabel('Number of employees', fontsize=axis_fontsize)
ax5.set_ylabel('Frequency', fontsize=axis_fontsize)
text = 'Min: ' + str(np.min(series)) + ' Max: ' + str(np.max(series)) + ' Median: ' + str(np.median(series)) 
ax5.text(5,max(data[0]), text, fontsize=axis_fontsize)
#%%
size = 'EXTRA LARGE'

series = c_dataframe.loc[c_dataframe.loc[:, 'emp_base'] == size].loc[:, 'num_employees']
style.use('ggplot')
ax6 = fig1.add_subplot(326)
data = ax6.hist(series, bins=11)
ax6.set_title('Employee base category - EXTRA LARGE', fontsize=title_fontsize)
ax6.set_xlabel('Number of employees', fontsize=axis_fontsize)
ax6.set_ylabel('Frequency', fontsize=axis_fontsize)
text = 'Min: ' + str(np.min(series)) + ' Max: ' + str(np.max(series)) + ' Median: ' + str(np.median(series)) 
ax6.text(5,max(data[0]), text, fontsize=axis_fontsize)

fig1.savefig('NoOfEMployeesDistribution.png')
plt.tight_layout(pad=2)
plt.show()

#%%
fig2 = plt.figure(2)
#fig2.suptitle('Distributions of companies', fontsize=12, fontweight='bold')
ns = 10
industry_sectors = np.arange(ns)
industry_sector_names = ['Comp.s/w', 'Hotels-resorts', 'Clth manu.', 'Construc.', 'Elex comp. M','Chem.M.', 'Insurance A.', 'Jewelers','Medi.N.','Pharma']

cnt_per_sector = []
for sector in industry_sectors:
    cnt_per_sector.append(len(c_dataframe.loc[c_dataframe.loc[:, 'industry_sector'] == sector]))
    
ax1 = fig2.add_subplot(111)
ax1.bar(industry_sectors, cnt_per_sector, label='Number of companies per sector', facecolor='green', edgecolor='gray')
ax1.set_xticks(industry_sectors)
ax1.set_xticklabels(industry_sector_names, rotation=45)
ax1.set_title('No. of companies per sector', fontsize=10)
ax1.set_xlabel('Industry sector', fontsize=8)
ax1.set_ylabel('No. of companies', fontsize=8)
fig2.savefig('SectrwiseNoCompanies.png')
plt.tight_layout()
plt.show()

#%%
fig3 = plt.figure(3)
number_of_emp = c_dataframe.loc[:, 'num_employees']
bins = [0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e5, 1e7, 1e8]
ax2 = fig3.add_subplot(111)
ax2.hist(number_of_emp, bins, facecolor='lightblue', edgecolor='gray')
ax2.set_xlabel('Number of employees', fontsize=8)
ax2.set_ylabel ('Frequency', fontsize=8)
ax2.set_title('Distribution of number of employees', fontsize=10)
#ax2.set_yscale('log')
ax2.set_xscale('log')
fig3.savefig('NumberOfEmployees.png')
plt.tight_layout(pad=1.5)
plt.show()

#%%
fig4 = plt.figure(4)
annual_rev = c_dataframe.loc[:, 'annual_revenue']
bins = [1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10, 1e11, 1e12, 1e13, 1e14]
ax3 = fig4.add_subplot(111)
ax3.hist(annual_rev, bins, facecolor='lightblue', edgecolor='gray')
ax3.set_xlabel('Annual revenue', fontsize=8)
ax3.set_ylabel('Number of companies', fontsize=8)
ax3.set_title('Distribution of annual revenue', fontsize=10)
#ax3.set_yscale('log')
ax3.set_xscale('log')
fig4.savefig('AnnualRevenue.png')
plt.tight_layout(pad=1.5)
plt.show()

#%%
fig5 = plt.figure(5)
net_rev = c_dataframe.loc[:, 'net_pl']
#bins = [-1e14, -1e13, -1e12, -1e11, -1e10, -1e9, -1e8, -1e7, -1e6, -1e5, -1e4, -1e3, -1e2, -1e1, -1e0, 1e0,1e1,1e2,1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10, 1e11, 1e12, 1e13, 1e14]

ax4 = fig5.add_subplot(111)
ax4.hist(net_rev, 21, facecolor='red', alpha= 0.5, edgecolor='brown')
ax4.set_xlabel('Net profit/loss', fontsize=8)
ax4.set_ylabel('Number of companies', fontsize=8)
ax4.set_title('Distribution of net profit/loss', fontsize=10)
ax4.set_yscale('log')
#ax4.set_xscale('log')

fig5.savefig('ProfitLoss.png')
plt.tight_layout(pad=1.5)
plt.show()

#%%

fig3 = plt.figure(3)
fig3.suptitle('Distribution of annual revenue for various categories of Annual revenue', fontsize=12, fontweight='bold')
#%%
size = 'VERY LOW'

R = (c_dataframe.loc[c_dataframe.loc[:, 'annual_revenue_category'] == size]).loc[:, 'annual_revenue']
bins = [1e5, 1e6, 1e7, 1e8, 1e9, 1e10, 1e11, 1e12, 1e13, 1e14]
ax1 = fig3.add_subplot(321)
ax1.hist(R, bins)
ax1.set_xlabel('Annual revenue', fontsize=8)
ax1.set_ylabel('Number of companies', fontsize=8)
ax1.set_title('Annual revenue category - VERY LOW', fontsize=10)
ax1.set_xscale('log')
print('Median: ', np.median(R), 'Min: ', np.min(R), 'Max: ', np.max(R))

#%%
size = 'LOW'

R = (c_dataframe.loc[c_dataframe.loc[:, 'annual_revenue_category'] == size]).loc[:, 'annual_revenue']
bins = [1e5, 1e6, 1e7, 1e8, 1e9, 1e10, 1e11, 1e12, 1e13, 1e14]
ax2 = fig3.add_subplot(322)
ax2.hist(R, bins)
ax2.set_xlabel('Annual revenue', fontsize=8)
ax2.set_ylabel('Number of companies', fontsize=8)
ax2.set_title('Annual revenue category - LOW', fontsize=10)
ax2.set_xscale('log')

print('Median: ', np.median(R), 'Min: ', np.min(R), 'Max: ', np.max(R))

#%%
size = 'MEDIUM'
bins = [1e5, 1e6, 1e7, 1e8, 1e9, 1e10, 1e11, 1e12, 1e13, 1e14]
R = (c_dataframe.loc[c_dataframe.loc[:, 'annual_revenue_category'] == size]).loc[:, 'annual_revenue']
ax3 = fig3.add_subplot(323)
ax3.hist(R, bins)
ax3.set_xlabel('Annual revenue', fontsize=8)
ax3.set_ylabel('Number of companies', fontsize=8)
ax3.set_title('Annual revenue category - MEDIUM', fontsize=10)
ax3.set_xscale('log')

print('Median: ', np.median(R), 'Min: ', np.min(R), 'Max: ', np.max(R))

#%%
size = 'HIGH'
bins = [1e5, 1e6, 1e7, 1e8, 1e9, 1e10, 1e11, 1e12, 1e13, 1e14]
R = (c_dataframe.loc[c_dataframe.loc[:, 'annual_revenue_category'] == size]).loc[:, 'annual_revenue']
ax4 = fig3.add_subplot(324)
ax4.hist(R, bins)
ax4.set_xlabel('Annual revenue', fontsize=8)
ax4.set_ylabel('Number of companies', fontsize=8)
ax4.set_title('Annual revenue category - HIGH', fontsize=10)
ax4.set_xscale('log')

print('Median: ', np.median(R), 'Min: ', np.min(R), 'Max: ', np.max(R))
#%%
size = 'VERY HIGH'
bins = [1e5, 1e6, 1e7, 1e8, 1e9, 1e10, 1e11, 1e12, 1e13, 1e14]
R = (c_dataframe.loc[c_dataframe.loc[:, 'annual_revenue_category'] == size]).loc[:, 'annual_revenue']
ax5 = fig3.add_subplot(325)
ax5.hist(R, bins)
ax5.set_xlabel('Annual revenue', fontsize=8)
ax5.set_ylabel('Number of companies', fontsize=8)
ax5.set_title('Annual revenue category - VERY HIGH', fontsize=10)
ax5.set_xscale('log')

print('Median: ', np.median(R), 'Min: ', np.min(R), 'Max: ', np.max(R))

#%%
size = 'EXTRA HIGH'
bins = [1e5, 1e6, 1e7, 1e8, 1e9, 1e10, 1e11, 1e12, 1e13, 1e14]
R = (c_dataframe.loc[c_dataframe.loc[:, 'annual_revenue_category'] == size]).loc[:, 'annual_revenue']
ax6 = fig3.add_subplot(326)
ax6.hist(R, bins)
ax6.set_xlabel('Annual revenue', fontsize=8)
ax6.set_ylabel('Number of companies', fontsize=8)
ax6.set_title('Annual revenue category - EXTRA HIGH', fontsize=10)
ax6.set_xscale('log')

fig3.savefig('AnnualRevenue.png')
plt.tight_layout(pad=1.5)
plt.show()
print('Median: ', np.median(R), 'Min: ', np.min(R), 'Max: ', np.max(R))
#%%

fig4, ax = plt.subplots(nrows=2, ncols=1, sharex=True)
fig4.suptitle('Sector wise distribution of annual revenue', fontsize=12, fontweight='bold')
c_dataframe.boxplot(column='annual_revenue', by='industry_sector', ax=ax[0], showmeans=True)
c_dataframe.boxplot(column='annual_revenue', by='industry_sector', ax=ax[1], showmeans=True)
#ax[0].set_xticks(industry_sectors)
#ax[0].set_xticklabels(industry_sector_names, rotation=40)
ax[0].set_title('On linear scale', fontsize=10)
ax[1].set_title('On log scale', fontsize=10)
ax[1].set_xticks(industry_sectors)
ax[1].set_xticklabels(industry_sector_names, rotation=30)
ax[1].set_yscale('log')
fig4.savefig('SectorwiseAnnualRevenue.png')
plt.show()
#%%

fig5, ax = plt.subplots(nrows=2, ncols=1, sharex=True)
fig4.suptitle('Sector wise distribution of annual revenue', fontsize=12, fontweight='bold')
c_dataframe.boxplot(column='net_pl', by='industry_sector', ax=ax[0], showmeans=True)
c_dataframe.boxplot(column='net_pl', by='industry_sector', ax=ax[1], showmeans=True)
#ax[0].set_xticks(industry_sectors)
#ax[0].set_xticklabels(industry_sector_names, rotation=40)
ax[0].set_title('On linear scale', fontsize=10)
ax[1].set_title('On log scale', fontsize=10)
ax[1].set_xticks(industry_sectors)
ax[1].set_xticklabels(industry_sector_names, rotation=30)
ax[1].set_yscale('log')
fig4.savefig('SectorwiseNetRevenue.png')
plt.show()
#%%

fig6 = plt.figure();
fig6.suptitle('Distribution of annual revenue', fontsize=12, fontweight='bold')
annual_rev = c_dataframe.loc[:, 'annual_revenue']
bins = [0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e5, 1e7, 1e8, 1e9, 1e10, 1e11, 1e12, 1e13, 1e14, 1e15]
data = plt.hist(annual_rev, bins, facecolor = 'green', edgecolor='black')
plt.xticks(np.arange(len(bins)), [str(x) for x in bins], rotation=45)
for i in range(len(bins)-1):
    plt.text(data[1][i],data[0][i],str(int(data[0][i])))
#plt.yscale('log')
plt.xscale('log')
print(data[0])
plt.show()
#%%

################################################################################################################
# Read company metadata
c_dataframe = pd.read_csv(r'C:\Users\Public\Data\honestCompanyMetadata_3.csv')
#%%
axis_fontsize = 8
title_fontsize = 10
fig_title_fontsize = 16
bold_wt = 'bold'

#%%
style.use('ggplot')
fig1 = plt.figure(1)
fig1.suptitle('Distribution of number of employees', fontsize=fig_title_fontsize, fontweight=bold_wt)
#%%

size = 'SMALL'

series = c_dataframe.loc[c_dataframe.loc[:, 'emp_base'] == size].loc[:, 'num_employees']
style.use('ggplot')
ax2 = fig1.add_subplot(311)
data = ax2.hist(series, bins=11)
ax2.set_title('Employee base category - SMALL', fontsize=title_fontsize)
ax2.set_xlabel('Number of employees', fontsize=axis_fontsize)
ax2.set_ylabel('Frequency', fontsize=axis_fontsize)
text = 'Min: ' + str(np.min(series)) + ' Max: ' + str(np.max(series)) + ' Median: ' + str(np.median(series)) 
ax2.text(5,max(data[0]), text, fontsize=axis_fontsize)

#%%
size = 'MEDIUM'

series = c_dataframe.loc[c_dataframe.loc[:, 'emp_base'] == size].loc[:, 'num_employees']
style.use('ggplot')
ax3 = fig1.add_subplot(312)
data = ax3.hist(series, bins=11)
ax3.set_title('Employee base category - MEDIUM', fontsize=title_fontsize)
ax3.set_xlabel('Number of employees', fontsize=axis_fontsize)
ax3.set_ylabel('Frequency', fontsize=axis_fontsize)
text = 'Min: ' + str(np.min(series)) + ' Max: ' + str(np.max(series)) + ' Median: ' + str(np.median(series)) 
ax3.text(5,max(data[0]), text, fontsize=axis_fontsize)

#%%

size = 'LARGE'

series = c_dataframe.loc[c_dataframe.loc[:, 'emp_base'] == size].loc[:, 'num_employees']

style.use('ggplot')
ax4 = fig1.add_subplot(313)
data = ax4.hist(series, bins=11)
ax4.set_title('Employee base category - LARGE', fontsize=title_fontsize)
ax4.set_xlabel('Number of employees', fontsize=axis_fontsize)
ax4.set_ylabel('Frequency', fontsize=axis_fontsize)
text = 'Min: ' + str(np.min(series)) + ' Max: ' + str(np.max(series)) + ' Median: ' + str(np.median(series)) 
ax4.text(5,max(data[0]), text, fontsize=axis_fontsize)

#%%
fig1.savefig('NoOfEMployeesDistribution.png')
plt.tight_layout(pad=2)
plt.show()
#%%

fig2 = plt.figure(2)
#fig2.suptitle('Distributions of companies', fontsize=12, fontweight='bold')
ns = 10
industry_sectors = np.arange(ns)
industry_sector_names = ['Comp.s/w', 'Hotels-resorts', 'Clth manu.', 'Construc.', 'Elex comp. M','Chem.M.', 'Insurance A.', 'Jewelers','Medi.N.','Pharma']

cnt_per_sector = []
for sector in industry_sectors:
    cnt_per_sector.append(len(c_dataframe.loc[c_dataframe.loc[:, 'industry_sector'] == sector]))
    
ax1 = fig2.add_subplot(111)
ax1.bar(industry_sectors, cnt_per_sector, label='Number of companies per sector', facecolor='green', edgecolor='gray')
ax1.set_xticks(industry_sectors)
ax1.set_xticklabels(industry_sector_names, rotation=45)
ax1.set_title('No. of companies per sector', fontsize=10)
ax1.set_xlabel('Industry sector', fontsize=8)
ax1.set_ylabel('No. of companies', fontsize=8)
fig2.savefig('SectrwiseNoCompanies.png')
plt.tight_layout()
plt.show()
#%%
fig3 = plt.figure(3)
fig3.suptitle('Distribution of annual revenue for various categories of Annual revenue', fontsize=12, fontweight='bold')
#%%

size = 'LOW'

R = (c_dataframe.loc[c_dataframe.loc[:, 'annual_revenue_category'] == size]).loc[:, 'annual_revenue']
bins = [1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10, 1e11, 1e12, 1e13, 1e14]
ax2 = fig3.add_subplot(311)
ax2.hist(R, bins)
ax2.set_xlabel('Annual revenue', fontsize=8)
ax2.set_ylabel('Number of companies', fontsize=8)
ax2.set_title('Annual revenue category - LOW', fontsize=10)
ax2.set_xscale('log')

print('Median: ', np.median(R), 'Min: ', np.min(R), 'Max: ', np.max(R))

#%%
size = 'MEDIUM'
bins = [1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10, 1e11, 1e12, 1e13, 1e14, 1e15]
R = (c_dataframe.loc[c_dataframe.loc[:, 'annual_revenue_category'] == size]).loc[:, 'annual_revenue']
ax3 = fig3.add_subplot(312)
ax3.hist(R, bins)
ax3.set_xlabel('Annual revenue', fontsize=8)
ax3.set_ylabel('Number of companies', fontsize=8)
ax3.set_title('Annual revenue category - MEDIUM', fontsize=10)
ax3.set_xscale('log')

print('Median: ', np.median(R), 'Min: ', np.min(R), 'Max: ', np.max(R))

#%%
size = 'HIGH'
bins = [1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10, 1e11, 1e12, 1e13, 1e14, 1e15]
R = (c_dataframe.loc[c_dataframe.loc[:, 'annual_revenue_category'] == size]).loc[:, 'annual_revenue']
ax4 = fig3.add_subplot(313)
ax4.hist(R, bins)
ax4.set_xlabel('Annual revenue', fontsize=8)
ax4.set_ylabel('Number of companies', fontsize=8)
ax4.set_title('Annual revenue category - HIGH', fontsize=10)
ax4.set_xscale('log')

print('Median: ', np.median(R), 'Min: ', np.min(R), 'Max: ', np.max(R))
#%%
fig3.savefig('AnnualRevenue.png')
plt.tight_layout(pad=1.5)
plt.show()
#print('Median: ', np.median(R), 'Min: ', np.min(R), 'Max: ', np.max(R))
#%%