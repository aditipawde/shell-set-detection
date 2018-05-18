import numpy as np
import pandas as pd
import configRead as con
import sys
import ast
from matplotlib import pyplot as plt
from scipy.stats import entropy

def sanityCheck():

    # Read config files
    print('Reading config parameters')
    try:
        mainConfig = con.readMainConfigs ()
        con.setGlobals(mainConfig)
        configParams = con.readConfigFile(con.paramConfigFile)
        configDistributions = con.readConfigFile(con.distributionsFile)
        modeparams = mainConfig[con.mode]
        distributions = configDistributions['distributions']
        commonParams = configParams['common']
        home_location = commonParams['home_location']
    except:
        print(sys.exc_info()[0] , ' error occured!')
        print('Error in reading config files.. Make sure you have not made any mistake in editing parameters')
        sys.exit(1)

    # Read metadata
    try:
        metadata_df = pd.read_csv(con.metadataFile, converters={'in_amt_wt':ast.literal_eval, 'supply_amt_wt':ast.literal_eval, 'utility_proportions': ast.literal_eval, 'inside_customers': ast.literal_eval, 'inside_suppliers':ast.literal_eval, 'utility_accs':ast.literal_eval},index_col='index')
        # Read summary
        companyDataPath = modeparams['company_data_path']
        summaryDataPath = companyDataPath + 'SummaryData\\finalSummary.csv'
        summary_Df = pd.read_csv(summaryDataPath)

        tranInfoPath = companyDataPath + '\\TranInfo'
        edgeList = pd.read_csv(tranInfoPath + '\\edges.csv')
        vertexList = pd.read_csv(tranInfoPath + '\\vertices.csv')

        N_tran_total = np.sum(edgeList['n_tran'])
        group_id_list = list(metadata_df.groupby('group_id').groups.keys())
        n_accounts = len(vertexList)
        n_edges = len(edgeList)
        n_groups = len(group_id_list)
        
    except:
        print(sys.exc_info()[0] , ' error occured!')
        print('Error in reading metadata')
        sys.exit(1)

    metadata_df['industry_sector'] = metadata_df.industry_sector.astype('str')
    company_acc = metadata_df['acc_no']
    N = len(metadata_df)
    print('Metadata for ', N, ' companies is read successfully!')

    # Save to file
    metafile_path = con.additionalFiles + '\\metafile.txt'
    meta_file = open(metafile_path, 'w')
    meta_file.write('Data set type :' + con.mode)
    meta_file.write('\nNumber of companies :' + str(N))
    meta_file.write('\nTotal transactions generated : ' + str(N_tran_total))
    meta_file.write('\nNumber of vertices : ' + str(n_accounts))
    meta_file.write('\nNumber of edges (connections) : ' + str(n_edges))
    meta_file.write('\nNumber of groups : ' + str(n_groups))
    meta_file.close()

    # Read sanity check file
    sanityCheck_df = pd.read_csv('sanityCheck.csv', converters={'param_bins':ast.literal_eval, 'distribution':ast.literal_eval})
    
    n_checks = len(sanityCheck_df)
    sanityCheck_df['KL_divergence'] = [0.0] * n_checks
    sanityCheck_df['status'] = ['F'] * n_checks

    for i in range(n_checks):
        param_name = sanityCheck_df.loc[i,'parameter']
        print('Checking for parameter ', param_name)
        type = sanityCheck_df.loc[i,'type']

        if type == 'm':
            param_dist = commonParams[param_name]
            paramValue = {k:v for k,v in sorted(param_dist.items())}
            expected_dist = np.array(list(paramValue.values()))
            param_grp = metadata_df.groupby(by=param_name, as_index = False).comp_id.agg('count')
            param_grp = param_grp.sort_values(by=param_name)
            param_grp['comp_id'] = param_grp['comp_id'] / N
            param_grp = {str(k):v for k,v in param_grp.items()}
            actual_dist = param_grp['comp_id'].get_values()

            KLD = entropy(expected_dist, actual_dist)
           
        else:
            bins = sanityCheck_df.loc[i, 'param_bins']
            distribution = sanityCheck_df.loc[i,'distribution']

            if param_name in metadata_df.columns:
                slice = metadata_df.loc[:,param_name]
            else:
                slice = summary_Df.loc[:, param_name]
            bin_len = len(bins)
            
            fig1 = plt.figure(1)
            ax1 = fig1.add_subplot(111)

            arr = ax1.hist(slice, bins, facecolor='red', alpha= 0.5, edgecolor='brown')
            observed_distribution = list(arr[0]/N)

            KLD = entropy(distribution, observed_distribution)

        sanityCheck_df.loc[i, 'KL_divergence'] = KLD

        if abs(KLD *100) < 5.0:
            sanityCheck_df.loc[i, 'status'] = 'P'
            print('KL divergence is ', KLD, ' Status - Successful')
        else:
            print('KL divergence is ', KLD, ' Status - Failed')

        sanityCheck_df.to_csv(con.additionalFiles + '\\sanityCheckReport.csv')

        ## Visualizations
        # Industry sector
        fig1 = plt.figure(1)
        sector_grps = metadata_df.groupby(by='industry_sector',as_index=False).agg({'comp_id':'count'})
        industry_sector_names = list(sector_grps.loc[:,'industry_sector'])
        industry_sectors = np.arange(len(industry_sector_names))
        cnt_per_sector = list(sector_grps.loc[:,'comp_id'])
        ax1 = fig1.add_subplot(111)
        ax1.bar(industry_sectors, cnt_per_sector, label='Number of companies per sector', facecolor='green', edgecolor='gray')
        ax1.set_xticks(industry_sectors)
        ax1.set_xticklabels(industry_sector_names, rotation=45)
        ax1.set_title('No. of companies per sector', fontsize=10)
        ax1.set_xlabel('Industry sector', fontsize=8)
        ax1.set_ylabel('No. of companies', fontsize=8)
        fig1.savefig(con.additionalFiles + '\\SectrwiseNoCompanies.png')

        # Employee base
        fig2 = plt.figure(2)
        emp_base_grps = metadata_df.groupby(by='employee_base',as_index=False).agg({'comp_id':'count'})
        emp_base_names = list(emp_base_grps.loc[:,'employee_base'])
        base_ticks = np.arange(len(emp_base_names))
        cnt_per_size = list(emp_base_grps.loc[:,'comp_id'])
        ax1 = fig2.add_subplot(111)
        ax1.bar(base_ticks, cnt_per_size, label='Number of companies per base size', facecolor='green', edgecolor='gray')
        ax1.set_xticks(base_ticks)
        ax1.set_xticklabels(emp_base_names, rotation=45)
        ax1.set_title('No. of companies per base size', fontsize=10)
        ax1.set_xlabel('Employee base sizes', fontsize=8)
        ax1.set_ylabel('No. of companies', fontsize=8)
        fig2.savefig(con.additionalFiles + '\\EmployeeBaseDistribution.png')

        # Number of employees
        fig3 = plt.figure(3)
        num_employees = metadata_df.loc[:, 'num_employees']
        bins = [0, 50,100,200, 500, 2000]
        ax1 = fig3.add_subplot(111)
        arr = ax1.hist(num_employees, bins, facecolor='lightblue', edgecolor='gray')
        ax1.set_xlabel('Number of employees', fontsize=8)
        ax1.set_ylabel('Number of companies', fontsize=8)
        ax1.set_title('Distribution of number of employees', fontsize=10)
        for i in range(len(arr[0])):
            plt.text(arr[1][i], arr[0][i], str(arr[0][i]/N*100))
        #ax1.set_xscale('log')
        max_no = max(num_employees)
        min_no = min(num_employees)
        text = 'Max. = ' + str(max_no) + '\nMin. = ' + str(min_no)
        plt.text(0, 40, text, fontsize=10)
        fig3.savefig(con.additionalFiles + '\\NumEmployees.png')

        # Employee profitability
        fig4 = plt.figure(4)
        emp_profitability = metadata_df.loc[:, 'rev_per_emp_per_month']
        bins = [1e3, 1e4, 1e5, 1e6, 1e7, 1e8]
        ax1 = fig4.add_subplot(111)
        arr = ax1.hist(emp_profitability, bins, facecolor='lightblue', edgecolor='gray')
        ax1.set_xlabel('Employee profitability in Rs.', fontsize=8)
        ax1.set_ylabel('Number of companies', fontsize=8)
        ax1.set_title('Distribution of employee profitability', fontsize=10)
        ax1.set_xscale('log')
        for i in range(len(arr[0])):
            plt.text(arr[1][i], arr[0][i], str(arr[0][i] / N *100))
        max_no = max(emp_profitability)
        min_no = min(emp_profitability)
        text = 'Max. = ' + str(max_no) + '\nMin. = ' + str(min_no)
        plt.text(1e3, 100, text, fontsize=10)
        fig4.savefig(con.additionalFiles +'\\EmployeeProfitability.png')

        # Annual revenue
        fig5 = plt.figure(5)
        annual_rev = summary_Df.loc[:, 'generated_annual_revenue']
        bins = [1e5, 1e6, 1e7, 1e8, 1e9, 1e10, 1e11, 1e12]
        ax1 = fig5.add_subplot(111)
        arr = ax1.hist(annual_rev, bins, facecolor='lightblue', edgecolor='gray')
        ax1.set_xlabel('Genearted annual revenue by BTS in Rs.', fontsize=8)
        ax1.set_ylabel('Number of companies', fontsize=8)
        ax1.set_title('Distribution of annual revenue', fontsize=10)
        #ax3.set_yscale('log')
        ax1.set_xscale('log')
        for i in range(len(arr[0])):
            plt.text(arr[1][i], arr[0][i], str(arr[0][i]/N*100))
        max_revenue = max(annual_rev)
        min_revenue = min(annual_rev)
        text = 'Max. = ' + str(max_revenue) + '\nMin. = ' + str(min_revenue)
        plt.text(1e5, 100, text, fontsize=10)
        fig5.savefig(con.additionalFiles + '\\AnnualRevenue.png')

        # deviation in annual revenue
        fig6 = plt.figure(6)
        dev_annual_rev = summary_Df.loc[:, 'revenue_deviation_per']
        bins = 11
        ax1 = fig6.add_subplot(111)
        arr= ax1.hist(dev_annual_rev, bins, facecolor='lightblue', edgecolor='gray')
        ax1.set_xlabel('Deviation in generated annual revenue in Rs.', fontsize=8)
        ax1.set_ylabel('Number of companies', fontsize=8)
        ax1.set_title('Distribution of deviation in generated annual revenue', fontsize=10)
        for i in range(len(arr[0])):
            plt.text(arr[1][i], arr[0][i], str(arr[0][i] / N *100))
        max_deviation = min(dev_annual_rev)
        text = 'Max. = ' + str(max_deviation) 
        plt.text(arr[1][0], 100, text, fontsize=10)
        fig6.savefig(con.additionalFiles +  '\\AnnualRevenueDeviation.png')

        # Generated expenses
        fig7 = plt.figure(7)
        generated_expenses = summary_Df.loc[:, 'generated_expenses']
        bins = [1e5, 1e6, 1e7, 1e8, 1e9, 1e10, 1e11, 1e12, 1e13]
        ax1 = fig7.add_subplot(111)
        arr= ax1.hist(generated_expenses, bins, facecolor='lightblue', edgecolor='gray')
        ax1.set_xlabel('Generated expenses in Rs.', fontsize=8)
        ax1.set_ylabel('Number of companies', fontsize=8)
        ax1.set_title('Distribution of generation of expenses', fontsize=10)
        ax1.set_xscale('log')
        for i in range(len(arr[0])):
            plt.text(arr[1][i], arr[0][i], str(arr[0][i]))
        max_expenses = max(generated_expenses)
        max_expenses = min(generated_expenses)
        text = 'Max. = ' + str(max_expenses) + '\nMin. = ' + str(max_expenses)
        plt.text(1e5, 100, text, fontsize=10)
        fig7.savefig(con.additionalFiles + '\\GeneratedExpenses.png')
        
        # Profit loss
        fig8 = plt.figure(8)
        profit_loss_per = summary_Df.loc[:, 'profit_loss_per']
        bins = [-100, -75, -50, -25, 0, 10, 20, 30, 100]

        ax1 = fig8.add_subplot(111)
        arr = ax1.hist(profit_loss_per, bins, facecolor='red', alpha= 0.5, edgecolor='brown')
        #ax1.yticks([0, 10, 20, 30, 40, 50, 60, 70, 80,90])
        ax1.set_xlabel('Net profit/loss in %', fontsize=8)
        ax1.set_ylabel('Number of companies', fontsize=8)
        ax1.set_title('Distribution of net profit/loss', fontsize=10)
        #ax1.set_yscale('log')
        for i in range(len(arr[0])):
            ax1.text(arr[1][i], arr[0][i], str(arr[0][i]))
        max_profit = max(profit_loss_per)
        max_loss = abs(min(profit_loss_per))
        text = 'Maximum profit = ' + str(max_profit) + '%' + '\nMaximum loss = ' + str(max_loss) + '%' + '\nN=' + str(N)
        ax1.text(-100,40, text, fontsize=12)
        fig8.savefig(con.additionalFiles + '\\ProfitLoss.png')

        # Percentage of incoming transactions
        fig9 = plt.figure(9)
        nan_indices = np.argwhere(np.isnan(summary_Df['i_tran_within_percentage']))
        nan_indices = list(nan_indices[:,0])
        summary_Df.loc[nan_indices, 'i_tran_within_percentage'] = 0
        i_tran_within = summary_Df.loc[:, 'i_tran_within_percentage']
        bins = 11
        ax1 = fig9.add_subplot(111)
        arr = ax1.hist(i_tran_within, bins,facecolor='red', alpha= 0.5, edgecolor='brown')
        ax1.set_xlabel('Percentage of incoming transations from within set', fontsize=8)
        ax1.set_ylabel('Number of companies', fontsize=8)
        ax1.set_title('Distribution of percenatge of incoming transactions from within set', fontsize=10)
        #ax1.set_yscale('log')
        for i in range(len(arr[0])):
            ax1.text(arr[1][i], arr[0][i], str(arr[0][i]))
        max_per = max(i_tran_within)
        min_per = min(i_tran_within)
        text = 'Maximum = ' + str(max_per) + '%' + '\nMaximum loss = ' + str(min_per) + '%' + '\nN=' + str(N)
        ax1.text(0,40, text, fontsize=12)
        fig9.savefig(con.additionalFiles +'\\PercentageTranWithin.png')

        # Percentage of outgoing transactions
        fig10 = plt.figure(10)
        nan_indices = np.argwhere(np.isnan(summary_Df['o_tran_within_percentage']))
        nan_indices = list(nan_indices[:,0])
        summary_Df.loc[nan_indices, 'o_tran_within_percentage'] = 0
        o_tran_within = summary_Df.loc[:, 'o_tran_within_percentage']
        bins = [0,10,20,30,40,50,70,80,90,100]

        ax1 = fig10.add_subplot(111)
        arr = ax1.hist(o_tran_within,bins, facecolor='red', alpha= 0.5, edgecolor='brown')
        ax1.set_xlabel('Percentage of outgoing transations to within set', fontsize=8)
        ax1.set_ylabel('Number of companies', fontsize=8)
        ax1.set_title('Distribution of percenatge of outgoing transactions to companies within set', fontsize=10)
        #ax1.set_yscale('log')
        for i in range(len(arr[0])):
            ax1.text(arr[1][i], arr[0][i], str(arr[0][i]))
        max_per = max(o_tran_within)
        min_per = min(o_tran_within)
        text = 'Maximum = ' + str(max_per) + '%' + '\nMaximum loss = ' + str(min_per) + '%' + '\nN=' + str(N)
        ax1.text(0,40, text, fontsize=12)
        fig10.savefig(con.additionalFiles + '\\PercentageTranWithinOutgoing.png')


#sanityCheck ()   
