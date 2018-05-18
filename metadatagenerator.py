#%%
import numpy as np
import pandas as pd
import random
import configRead as con
import json
import os
#%%

def in_amt_wt_per_flag (params, flag_list):
    flag_specific_in_amt_wt = dict()
    flag_specific_supply_amt_wt = dict()
    for flag in flag_list:
        flagParams = params[flag]
        in_amt_wt = flagParams['in_amt_distribution_wt']
        supply_amt_wt = flagParams['supply_amt_distribution_wt']
        flag_specific_in_amt_wt[flag] = in_amt_wt
        flag_specific_supply_amt_wt[flag] = supply_amt_wt
    return flag_specific_in_amt_wt, flag_specific_supply_amt_wt

def generateMetadata(N, companyIDseed):
    '''This function generates metadata for N comapnies, employee accounts and utility accounts based on parameter values in config file and outputs data to specified file name'''
 
    # Read parameters from config file
    configParams = con.readConfigFile(con.paramConfigFile)
    configDistributions = con.readConfigFile(con.distributionsFile)
    configurations = configDistributions['distributions']

    params = configParams['common']

    # Extract parameter values with probability distributions
    industry_sector_list, p_industry_sector = con.getParameterDistribution (params['industry_sector'])
    employee_base_list, p_employee_base = con.getParameterDistribution (params['employee_base'])
    num_owners_list, p_num_owners = con.getParameterDistribution (params['num_owners'])
    
    emp_base_levels = len(employee_base_list)
    num_emp_distribution = configurations['num_employees_distributions'] 

    employee_profitability_list = list(params['employee_profitability'].keys())
    n_emp_profitability_levels = len(employee_profitability_list)
    emp_profitability_distribution = configurations['emp_profitability_distributions'] 

    # Generate lists with discrete probabilities obtained above
    l_id = np.arange(companyIDseed, N+companyIDseed)
    l_sector = np.random.choice(industry_sector_list, N, p=p_industry_sector)
    l_emp_base = np.random.choice(employee_base_list, N, p=p_employee_base)
    l_num_owners = np.random.choice(num_owners_list, N, p=p_num_owners)

    # Concatenate lists and form dataframe
    c_list = list(zip(l_id, l_sector, l_emp_base, l_num_owners))

    c_dataframe = pd.DataFrame(data = c_list, columns=['comp_id', 'industry_sector', 'employee_base', 'num_owners'], index=l_id)
    c_dataframe.set_index('comp_id')

    # Clear lists
    c_list.clear()
    print('Generating metadata...')
    # Number of employees calculations
    # num_employees are uniform random for base size 
    emp_base_dict = {k:v for k,v in zip(employee_base_list, num_emp_distribution)}
    c_dataframe['num_employees'] = list(map(lambda x : abs(int(np.random.normal(emp_base_dict[x][0], emp_base_dict[x][1]))), c_dataframe['employee_base']))
    
    c_dataframe['acc_no'] = ['C{0}' .format(x) for x in l_id]
    c_dataframe['acc_type'] = ['C'] * N

     # clientele_categorization and supply intensity
    c_dataframe['clientele_categorization'] = list(map(lambda x:[k for k,v in params['clientele_categorization'].items() if x in v][0], c_dataframe['industry_sector']))
    c_dataframe['supply_intensity'] = list(map(lambda x:[k for k,v in params['supply_intensity'].items() if x in v][0], c_dataframe['industry_sector']))

    # Employee profitability - sector-wise
    c_dataframe['employee_profitability'] = list(map(lambda x: [k for k,v in params['employee_profitability'].items() if x in v][0], c_dataframe['industry_sector']))

    # Annual revenue calculations
    # Stochastic mean and stdev for annual_revenue_category L, M, H
    # X = N(m, s); Y = N(0.1*X, 0.01*X); Z = N(X, Y)
    # annual_revenue = 12 * Z * num_employees

    emp_profit_distr_dict = dict(zip(employee_profitability_list, emp_profitability_distribution))

    X = np.array(list(map(lambda x: np.random.normal(emp_profit_distr_dict[x][0], emp_profit_distr_dict[x][1]), c_dataframe['employee_profitability'])))
    Y = np.random.normal (0.1 * X, abs(0.01 * X))
    Z = abs(np.random.normal (X, abs(Y)))

    c_dataframe['rev_per_emp_per_month'] = np.round(Z, 2)

    c_dataframe['annual_revenue'] = np.round(12*c_dataframe['num_employees']*c_dataframe['rev_per_emp_per_month'], 2)

    # Assign flag label to companies
    if (con.mode == 'S'):
        shell_flag_list, shell_flag_distribution = con.getParameterDistribution (params['shell_flag_distribution'])
        c_dataframe['flag'] = np.random.choice (shell_flag_list, N, p=shell_flag_distribution)
        flag_specific_in_amt_wt, flag_specific_supply_amt_wt = in_amt_wt_per_flag (configParams, shell_flag_list)
    else:
        c_dataframe['flag'] = ['H'] * N
        flag_specific_in_amt_wt, flag_specific_supply_amt_wt = in_amt_wt_per_flag (configParams, ['H'])

    c_dataframe['in_amt_wt'] = list(map(lambda x, y: flag_specific_in_amt_wt[x][y], c_dataframe['flag'], c_dataframe['clientele_categorization']))

    # Expected number of incoming supply transactions
    in_amt_distribution = np.array(params['amt_distribution'])
    c_dataframe['revenue_proportions'] = list(map(lambda x,y: np.array(x) * y, c_dataframe['in_amt_wt'], c_dataframe['annual_revenue']))
    c_dataframe['tran_per_rev_proportion'] = c_dataframe['revenue_proportions'] / in_amt_distribution[:,0]
    c_dataframe['i_tran_expected'] = list(map(lambda x:np.int(np.sum(x)), c_dataframe['tran_per_rev_proportion']))
    c_dataframe['i_inside_tran_expected'] = list(map(lambda x:np.int(params['p_inside_account'] * x), c_dataframe['i_tran_expected']))
    c_dataframe['i_inside_available'] = c_dataframe['i_inside_tran_expected']

     # Supply expenses info
    c_dataframe['supply_expense_per'] = list(map(lambda x: abs(round(np.random.normal(configurations['supply_expense_per'][x][0], configurations['supply_expense_per'][x][1]),2)), c_dataframe['supply_intensity'] ))

    c_dataframe['supply_expenses'] = round(c_dataframe['annual_revenue'] * c_dataframe['supply_expense_per'] / 100, 2)
    c_dataframe['supply_amt_wt'] = list(map(lambda x, y: flag_specific_supply_amt_wt[x][y], c_dataframe['flag'], c_dataframe['supply_intensity']))
    c_dataframe['supply_proportions'] = list(map(lambda x,y: np.array(x) * y, c_dataframe['supply_amt_wt'], c_dataframe['supply_expenses']))
    c_dataframe['tran_per_supply_proportion'] = c_dataframe['supply_proportions'] / in_amt_distribution[:,0]
    c_dataframe['s_tran_expected'] = list(map(lambda x:np.int(np.sum(x)), c_dataframe['tran_per_supply_proportion']))
    c_dataframe['s_inside_tran_expected'] = list(map(lambda x:np.int(params['p_inside_account'] * x), c_dataframe['s_tran_expected']))
    c_dataframe['s_inside_available'] = c_dataframe['s_inside_tran_expected']
    # Group formation
    # Put constraint on annual_rev_category
    n_division_pro = int(n_emp_profitability_levels * 0.5)
    emp_profitability_lower = [employee_profitability_list[i] for i in range(n_division_pro)]
    emp_profitablity_upper = [employee_profitability_list[n_emp_profitability_levels - i - 1] for i in range(n_division_pro+1)]
    emp_profitability_comp = [emp_profitability_lower, emp_profitablity_upper]

    n_division_emp = int(emp_base_levels * 0.5)
    emp_base_lower = [employee_base_list[i] for i in range (n_division_emp)]
    emp_base_upper = [employee_base_list[emp_base_levels - i - 1] for i in range(n_division_emp)]
    emp_base_comp = [emp_base_lower, emp_base_upper]

    
    c_dataframe.insert(loc=len(c_dataframe.columns), column = 'is_in_group', value = ['N'] * N)
    c_dataframe.insert(loc=len(c_dataframe.columns), column = 'group_id', value = [''] * N)
    c_dataframe.insert(loc=len(c_dataframe.columns), column = 'p_tran_within_grp', value = [0.0] * N)
    group_id_pref = 'GH' if con.mode == 'H' else 'GS'

    for i in range(params['n_groups']):

        group_found = False

        while not group_found:
            n_compatibility = random.choice([1,0])
            comp_emp_base = emp_base_comp[n_compatibility]
            comp_emp_prof = emp_profitability_comp[n_compatibility]
            comp_set = set(c_dataframe.loc[(c_dataframe.employee_base.isin(comp_emp_base)) & (c_dataframe.employee_profitability.isin(comp_emp_prof) & (c_dataframe.is_in_group == 'N')), 'comp_id'])
            size_comp_set = len(comp_set)
            
            size = max(5,abs(int(np.random.normal(params['avg_group_size'][0], params['avg_group_size'][1])))) # Group size should be at least 5
            if size < size_comp_set:
               group_found = True

        group_members = random.sample(comp_set,size)
      
        c_dataframe.loc[c_dataframe.comp_id.isin(group_members), 'group_id'] = group_id_pref + str(i)
        c_dataframe.loc[c_dataframe.comp_id.isin(group_members), 'is_in_group'] = 'Y'
        c_dataframe.loc[c_dataframe.comp_id.isin(group_members), 'p_tran_within_grp'] = params['p_tran_within_group']
        
    c_dataframe['n_i_grp_expected'] = list(map(lambda x,y: np.rint(x*y), c_dataframe['p_tran_within_grp'], c_dataframe['tran_per_rev_proportion']))
    

    c_dataframe ['n_suppliers'] = list(map(lambda x : int(np.random.normal(configurations['n_suppliers'][x][0], configurations['n_suppliers'][x][1])), c_dataframe['supply_intensity']))
    c_dataframe['is_inside_customer'] = np.random.choice(['Y', 'N'], N, p=[configurations['p_inside_customer'], 1-configurations['p_inside_customer']])
    c_dataframe['per_inside_suppliers'] = list(map(lambda x:np.random.normal(configurations['per_inside_suppliers'][0], configurations['per_inside_suppliers'][1]) if c_dataframe.loc[x, 'is_inside_customer'] == 'Y' else 0, c_dataframe['comp_id']))
    list_n_suppliers = np.rint(c_dataframe['n_suppliers'] * c_dataframe['per_inside_suppliers'] / 100)
    c_dataframe['n_inside_suppliers'] =  [int(x) if x>0 else 1 for x in list_n_suppliers]
    c_dataframe['n_outside_suppliers'] = c_dataframe['n_suppliers'] - c_dataframe['n_inside_suppliers']

    ## Inside supplier customer
    n_division_emp = int(emp_base_levels * 0.5)
    emp_base_lower = [employee_base_list[i] for i in range (n_division_emp)]
    emp_base_upper = [employee_base_list[emp_base_levels - i - 1] for i in range(n_division_emp)]
    emp_base_comp = [emp_base_lower, emp_base_upper]

    supplier_customer_compatibility = configurations['supplier_customer_compatibility']
    has_supplier_list = list(c_dataframe.loc[c_dataframe.is_inside_customer == 'Y', 'comp_id'])
   
    has_supplier_dict = dict()
    has_customer_dict = dict()

    for c in has_supplier_list:
        # Find compatible suppliers
        #c_emp_base = c_dataframe.loc[c, 'employee_base']
        #c_emp_profitability = c_dataframe.loc[c, 'employee_profitability']
        n_inside_supplier = c_dataframe.loc[c, 'n_inside_suppliers']
        group_id = c_dataframe.loc[c, 'group_id']
        cust_acc = c_dataframe.loc[c, 'acc_no']
        c_clientele = c_dataframe.loc[c, 'clientele_categorization']
        comp_supply_intensity = supplier_customer_compatibility[c_clientele]

        #comp_emp_base = emp_base_lower if c_emp_base in emp_base_lower else emp_base_upper
        #comp_emp_pro = emp_profitability_lower if c_emp_profitability in emp_profitability_lower else emp_profitablity_upper
        
        candidate_comp_set = set(c_dataframe.loc[((c_dataframe.comp_id != c) & (c_dataframe.supply_intensity.isin(comp_supply_intensity))), 'comp_id'])

        # Remove group members
        if group_id != '':
            candidate_comp_set = candidate_comp_set - set(c_dataframe.loc[c_dataframe.group_id == group_id, 'comp_id'])
        
        # To avoid customer supplier two way relationship
        if c in has_customer_dict.keys():
            candidate_comp_set = candidate_comp_set - set(has_customer_dict[c])

        n_candidate_suppliers = len(candidate_comp_set)
        if n_candidate_suppliers > 0:
            candidate_comp_list = list(candidate_comp_set)
            if n_candidate_suppliers >= n_inside_supplier:
                random.shuffle(candidate_comp_list)
                inside_suppliers = list(random.sample(candidate_comp_list, n_inside_supplier)) 
            else:
                inside_suppliers = candidate_comp_list
            supplier_acc = list(c_dataframe.loc[c_dataframe.comp_id.isin(inside_suppliers), 'acc_no'])
        else:
            inside_suppliers = list()
            supplier_acc = list()

        has_supplier_dict[c] = supplier_acc

        for s in inside_suppliers:
            if s not in has_customer_dict.keys(): 
                has_customer_dict[s] = [cust_acc]
            else:
                has_customer_dict[s].append(cust_acc)

    has_customer_list = list(has_customer_dict.keys())
    c_dataframe['inside_suppliers'] = list(map(lambda k: has_supplier_dict[k] if k in has_supplier_list else list(), c_dataframe['comp_id']))
    c_dataframe['inside_customers'] = list(map(lambda k: has_customer_dict[k] if k in has_customer_list else list(), c_dataframe['comp_id']))
    c_dataframe['n_inside_customers'] = list(map(lambda x: len(c_dataframe.loc[x, 'inside_customers']) if c_dataframe.loc[x, 'inside_customers'] is not np.NaN else 0, c_dataframe['comp_id']))
            
    # Repeate customer probability
    c_dataframe['p_repeat_client'] = list(map(lambda x: configurations['p_repeat_client'][x], c_dataframe['clientele_categorization']))
    
    # Repeat supplier probability
    c_dataframe['p_repeat_supplier'] = list(map(lambda x: configurations['p_repeat_supplier'][x], c_dataframe['supply_intensity']))

    # Salary expenses
    c_dataframe['emp_hierarchy_distributions'] = list(map(lambda x : [int(x*y) if x*y!=0 else 1 for y in configurations['emp_hierarchy_distributions']], c_dataframe['num_employees']))
    c_dataframe['employee_profitability_multiplier'] = list(map(lambda x: configurations['employee_profitability_multiplier'][x], c_dataframe['supply_intensity']))
    c_dataframe['emp_salary'] = c_dataframe['rev_per_emp_per_month'] * c_dataframe['employee_profitability_multiplier'] /100
    c_dataframe['salary_distribution'] = list(map(lambda x: list(np.random.normal(x, 0.1*x, 5)), c_dataframe['emp_salary']))
    #c_dataframe['salary_distribution'] = list(map(lambda x: x.sort(), c_dataframe['salary_distribution']))

    # utility_expenses
    n_utility_acc = configurations['n_utility_acc']
    l_utility = ['U{0}' .format(x) for x in range(n_utility_acc)]
    l_utility_acc_type= ['C'] * n_utility_acc

    c_dataframe['n_utilities'] = [int(x) if int(x) > 2 else 2 for x in abs(np.random.normal(configurations['n_utilities_distribution'][0], configurations['n_utilities_distribution'][1], N))]
    c_dataframe['utility_accs'] = list(map(lambda x: random.sample(l_utility, x), c_dataframe['n_utilities'])) 
    c_dataframe['utility_expense_per'] = list(map(lambda x,y: abs(np.random.normal(configurations['utility_expense_per'][x][0], configurations['utility_expense_per'][x][1])) if y == 'N' else abs(np.random.normal(configurations['utility_expense_per_group'][x][0], configurations['utility_expense_per_group'][x][1])), c_dataframe['supply_intensity'], c_dataframe['is_in_group']))
    c_dataframe['utility_expenses'] = round(c_dataframe['annual_revenue'] * c_dataframe['utility_expense_per'] / 100, 2)
    c_dataframe['utility_proportions'] = list(map(lambda x: [round(x * y, 2) for y in configurations['utility_fractions']], c_dataframe['utility_expenses']))
    
    utility_df = pd.DataFrame(list(zip(l_utility, l_utility_acc_type)), columns= ['acc_no', 'acc_type'])
    utility_df.to_csv (con.additionalFiles + '\\utilityAcc.csv', index = False)

    c_dataframe['accumulated_revenue'] = [0.0] * N
    c_dataframe['performed_expenses'] = [0.0] * N
    
    # Generate supplier and employee accounts
    l_emp_acc_no = list()
    l_emp_comp_id = list()
    l_emp_levels = list()
    l_emp_acc_type = list()

    emp_hierarchy_levels = configurations['emp_hierarchy_levels']
    acc_type, acc_type_distribution = con.getParameterDistribution (configurations['emp_acc_type'])

    for c_id in l_id:
        n_emp = c_dataframe.loc[c_id, 'num_employees']
        emp_acc_no = ['E{0}{1}' .format(c_id, x) for x in range(n_emp)]
        emp_comp_id = [c_id] * n_emp
        emp_acc_type = list(np.random.choice(acc_type, n_emp,p=acc_type_distribution))

        emp_levels  = list(np.random.choice(emp_hierarchy_levels, n_emp, p=configurations['emp_hierarchy_distributions']))
        l_emp_acc_no = l_emp_acc_no + emp_acc_no
        l_emp_acc_type = l_emp_acc_type + emp_acc_type
        l_emp_comp_id = l_emp_comp_id + emp_comp_id
        l_emp_levels = l_emp_levels + emp_levels

    l_emp_accounts = list(zip(l_emp_acc_no, l_emp_acc_type, l_emp_comp_id, l_emp_levels))
    e_dataframe = pd.DataFrame(data = l_emp_accounts, columns=['acc_no', 'acc_type', 'comp_id','emp_level'])
    e_dataframe.to_csv(con.additionalFiles + '\\employeeAccounts.csv', index=False)
    e_dataframe.iloc[0:0]

    # Convert all numpy.ndarray columns to list before saving 
    c_dataframe['revenue_proportions'] = list(map(lambda x: list(x), c_dataframe['revenue_proportions']))
    c_dataframe['tran_per_rev_proportion'] = list(map(lambda x: list(x), c_dataframe['tran_per_rev_proportion']))
    c_dataframe['supply_proportions'] = list(map(lambda x: list(x), c_dataframe['supply_proportions']))
    c_dataframe['tran_per_supply_proportion'] = list(map(lambda x: list(x), c_dataframe['tran_per_supply_proportion']))
    c_dataframe['n_i_grp_expected'] = list(map(lambda x: list(x), c_dataframe['n_i_grp_expected']))

    # Save data to user-specified file
    c_dataframe.to_csv(con.metadataFile, index_label='index')
    c_dataframe.iloc[0:0]