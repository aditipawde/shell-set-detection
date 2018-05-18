# import packages
import pandas as pd
import random
import numpy as np
import configRead as con
import json
#import os
from math import ceil
import ast
from decimal import *

n_files = 0
def writeToFile (df):
    '''This function writes dataframe to csv file and returns empty data frame'''
    global n_files
    file_name = con.tranData + 'tran' + str(n_files) + '.csv'
    df.to_csv (file_name, index=False) 
    n_files = n_files + 1
    df = df.iloc[0:0]
    return df

def array_converter (array_string):
    '''This function converts numpy array read from csv into numpy array'''
    array_string = ','.join(array_string.replace('[  ', '[').split())
    array_string = ast.literal_eval(array_string)
    return np.array(array_string)

def calculateFriendProbability (friend_revenue_dict):
    '''This function calculates probability for availabile group companies'''

    total_revenue = np.sum(np.array(list(friend_revenue_dict.values()))[:])
    prob_sum = Decimal(0)
    wt = Decimal(0)
    item_len = len(friend_revenue_dict)
    friend_probability = dict()
    index = 0
    
    for k,v in friend_revenue_dict.items():
        if index != (item_len-1):
            wt = Decimal(v)/Decimal(total_revenue)
            friend_probability[k] = round(wt, 2)
            prob_sum = Decimal(prob_sum) + Decimal(friend_probability[k])

            index = index + 1     
        else:
            delta_diff = Decimal(1) - Decimal(prob_sum)
            if delta_diff > 0:
                friend_probability[k] = Decimal(1) - Decimal(prob_sum)
            else:
                friend_probability[k] = Decimal(0)
                delta_diff = abs(delta_diff)
                k_list = [k for k,v in friend_probability.items() if v > delta_diff]
                k_selected = random.choice(k_list)
                friend_probability[k_selected] = friend_probability[k_selected] - delta_diff

    return friend_probability

def generateTransactionData(companyIDSeed):
    '''This function generates incoming transactions and outgoing transactions for each company'''
    e_dataframe = pd.read_csv(con.additionalFiles + '\\employeeAccounts.csv')
    print('Reading metadatafile...')

    # Read metadata files
    c_dataframe = pd.read_csv(con.metadataFile, converters={'in_amt_wt':ast.literal_eval, 'revenue_proportions':ast.literal_eval, 'tran_per_rev_proportion':ast.literal_eval, 'supply_amt_wt':ast.literal_eval, 'supply_proportions':ast.literal_eval, 'tran_per_supply_proportion':ast.literal_eval, 'n_i_grp_expected':ast.literal_eval, 'inside_suppliers':ast.literal_eval, 'inside_customers': ast.literal_eval, 'salary_distribution': ast.literal_eval, 'utility_accs':ast.literal_eval, 'utility_proportions': ast.literal_eval}, index_col='index'); 

    c_dataframe['revenue_proportions'] = list(map(lambda x: np.array(x), c_dataframe['revenue_proportions']))
    c_dataframe['tran_per_rev_proportion'] = list(map(lambda x: np.array(x), c_dataframe['tran_per_rev_proportion']))
    c_dataframe['supply_proportions'] = list(map(lambda x: np.array(x), c_dataframe['supply_proportions']))
    c_dataframe['tran_per_supply_proportion'] = list(map(lambda x: np.array(x), c_dataframe['tran_per_supply_proportion']))
    c_dataframe['n_i_grp_expected'] = list(map(lambda x: np.array(x), c_dataframe['n_i_grp_expected']))

    c_dataframe.set_index('comp_id')
    #c_dataframe['']
    

    print('Metadata file read successfully!')

    N = len(c_dataframe)
    relation_mat = np.zeros([N,N], dtype=int)
    comp_id_list = c_dataframe['comp_id']

    tran_col_list = ['src_acc', 'src_acc_type', 'dest_acc', 'dest_acc_type', 'currency', 'payment_mode', 'amount', 'date', 'location', 'is_tax_haven', 'tran_label']

    parent_df = pd.DataFrame(columns = tran_col_list)
    n_records = 500000

    # Read transaction parameters and distribution settings
    allConfigParams = con.readConfigFile(con.paramConfigFile)
    configParams = allConfigParams['common']
    configDistributions = con.readConfigFile(con.distributionsFile)
    distribution_params = configDistributions['distributions']

    daterange = configParams['date']
    date_list = range(daterange[0], daterange[1]+1)
    p_date = [1/len(date_list)]*len(date_list)
    home_location = configParams['home_location']
    home_currency = configParams['home_currency']
    home_is_tax_haven = configParams['home_is_tax_haven']
    amt_distributions = np.array(configParams['amt_distribution'])
    grp_amt_distribution = configParams['grp_amt_distribution']
    p_inside_account = configParams['p_inside_account']
    '''
    # Salary distributions and amounts
    sal_distribution_dict = dict()
    #sal_fractions = [0.3, 0.6, 1.0, 1.3, 1.6]
    rev_per_emp_factor = distribution_params['employee_profitability_multiplier'] # Adjusts how much an employee is paid based on employee profitability
    for k in distribution_params['sal_distribution'].keys():
        base_distribution_idx = distribution_params['sal_distribution'][k]
        base_distribution = distribution_params['emp_profitability_distributions'][base_distribution_idx]
        sal_dist_m = np.ceil(np.random.normal(base_distribution[0], abs(base_distribution[1]), 5) / rev_per_emp_factor) # Take 5 means
        sal_dist_m.sort()   #Sort them  
        sal_dist_d = sal_dist_m * 0.1       # Stdev. for them
        sal_distribution_dict[k] = list(zip(sal_dist_m, sal_dist_d)) # Prepare dictionary
    '''
    n_groups = configParams['n_groups']
    
    # Generate group transactions
    if n_groups > 0:
        print('Generatimg group transactions')

        grp_records = list()
        group_tran_payment_mode_list, group_tran_payment_mode_distribution = con.getParameterDistribution(configParams['group_tran_payment_mode'])

        c_dataframe['grp_revenue_generated'] = [0.0] * len(c_dataframe)
        c_dataframe['grp_expense_performed'] = [0.0] * len(c_dataframe)
        c_dataframe['remaining_amt'] = c_dataframe['annual_revenue']
        c_dataframe['remaining_revenue'] = [0.0] * len(c_dataframe)
        c_dataframe['remaining_expense'] = [0.0] * len(c_dataframe)

        # All inter-comapny transactions are considered as performed from home location
        location,is_tax_haven, currency = home_location, home_is_tax_haven, home_currency
        tran_label = 'G'

        group_id_pre = 'GS' if con.mode == 'S' else 'GH'
        #group_id_pre = 'G'
        
        for i in range(n_groups):
            data_to_be_written = 1
            group_members = set(c_dataframe.loc[c_dataframe.group_id == group_id_pre + str(i), 'comp_id'])
            print('Group ', i,' ' ,group_members)
            src_acc_type = "C"
            dest_acc_type = "C"

            for g in group_members:
                print('For ', g)
                revenue, dest_acc, n_i_group_arr, in_amt_wt = c_dataframe.loc[g, ['annual_revenue', 'acc_no', 'n_i_grp_expected', 'in_amt_wt']]

                accumulated_grp_rev = 0

                # Retrieve friends details
                friends = group_members-{g} 
                n_friends = len(friends)
                friend_acc_dict = dict(c_dataframe.loc[friends, 'acc_no'])
                friend_amt_dict = dict(c_dataframe.loc[friends, 'remaining_amt'])
                friend_revenue = dict(c_dataframe.loc[friends, 'annual_revenue'])
                friend_exp_dict = dict(c_dataframe.loc[friends, 'grp_expense_performed'])

                ## Change for companies going in severe loss 
                friend_probability = calculateFriendProbability (friend_revenue)           
                candidate_friends = list(friend_probability.keys())
                candidate_friends_prob = list(friend_probability.values())
                
                grp_amt_dist_id = np.arange(0,5)
                all_friends_done = False

                for i in range(5):
                    n_i_group = n_i_group_arr[i]
                    while (n_i_group > 0 and revenue - accumulated_grp_rev > 0 and not all_friends_done) :# Enter while loop only when n_i_group not exhausted, revenue not completely generated

                        # Find a friend who can perform transaction 
                        is_friend_found = False

                        # Initially randomly choose a distribution # p changed to in_amt_wt -- 15/2
                        dist_id = i
                        amount = abs(ceil(np.random.normal(grp_amt_distribution[dist_id][0], abs(grp_amt_distribution[dist_id][1]))))
                        candidate_friends_available = [k for k,v in friend_amt_dict.items() if v - amount > 0]    # Friends who have amt remaining for performing transactions
                        if len(candidate_friends_available) < len(friend_acc_dict): # Recalculate probability if available friends reduce

                            candidate_friend_revenue = {k:v for k,v in friend_revenue.items() if k in candidate_friends_available}
                            friend_probability = calculateFriendProbability(candidate_friend_revenue)
                            candidate_friends = list(friend_probability.keys())
                            candidate_friends_prob = list(friend_probability.values())

                        #candidate_friends_prob = [v for k,v in friend_probability.items() if k in candidate_friends]

                        if len(candidate_friends_available) > 0:
                            #random.shuffle(candidate_friends)
                            #chosen_friend = random.choice(candidate_friends)
                            chosen_friend = np.random.choice(candidate_friends,p=candidate_friends_prob)

                            src_acc = friend_acc_dict[chosen_friend]
                            is_friend_found = True
                    
                        else:
                            while dist_id >= 0 and not is_friend_found:     # Try transaction with lower distribution
                                dist_id = dist_id - 1
                                amount = abs(ceil(np.random.normal(grp_amt_distribution[dist_id][0], abs(grp_amt_distribution[dist_id][1]))))
                                candidate_friends_available = [k for k,v in friend_amt_dict.items() if v - amount > 0]
                                #candidate_friends_prob = [v for k,v in friend_probability.items() if k in candidate_friends]

                                if len(candidate_friends_available) < len(friend_acc_dict): # Recalculate probability if available friends reduce

                                    candidate_friend_revenue = {k:v for k,v in friend_revenue.items() if k in   candidate_friends_available}
                                    friend_probability = calculateFriendProbability(candidate_friend_revenue)
                                    candidate_friends = list(friend_probability.keys())
                                    candidate_friends_prob = list(friend_probability.values())

                                if len(candidate_friends_available) > 0:
                                    #random.shuffle(candidate_friends)
                                    #chosen_friend = random.choice(candidate_friends)
                                    chosen_friend = np.random.choice(candidate_friends,p=candidate_friends_prob)
                                    src_acc = friend_acc_dict[chosen_friend]
                                    is_friend_found = True


                        if is_friend_found:
                            payment_mode = np.random.choice(group_tran_payment_mode_list, p=group_tran_payment_mode_distribution)
                            date = np.random.choice(date_list, p=p_date)
                            record = [src_acc, src_acc_type, dest_acc, dest_acc_type, currency, payment_mode, amount, date, location, is_tax_haven, tran_label]
                            grp_records.append(record)

                            n_i_group = n_i_group - 1
                            accumulated_grp_rev = accumulated_grp_rev + amount
                 
                            friend_amt_dict[chosen_friend] = friend_amt_dict[chosen_friend] - amount
                            friend_exp_dict[chosen_friend] = friend_exp_dict[chosen_friend] + amount

                            
                        else:
                            all_friends_done = True
                            break

                # Update dataframe
                c_dataframe.loc[g, 'grp_revenue_generated'] = accumulated_grp_rev
                c_dataframe.loc[list(friend_amt_dict.keys()), 'remaining_amt'] = list(friend_amt_dict.values()) 
                c_dataframe.loc[list(friend_exp_dict.keys()), 'grp_expense_performed'] = list(friend_exp_dict.values())

            # Tran generation for one group ends here
            if len(grp_records) > n_records:
                group_df = pd.DataFrame(data=grp_records, columns=tran_col_list)
                group_df = writeToFile(group_df)
                grp_records.clear()

        if len(grp_records) > 0:
            group_df = pd.DataFrame(data=grp_records, columns=tran_col_list)
            group_df = writeToFile(group_df)
            grp_records.clear()

        # Update fractions after group transactions
        c_dataframe['remaining_revenue'] = c_dataframe['annual_revenue'] - c_dataframe['grp_revenue_generated']
        c_dataframe['remaining_expense'] = c_dataframe['supply_expenses'] - c_dataframe['grp_expense_performed']
        c_dataframe['revenue_proportions'] = list(map(lambda x,y: np.array(x) * y, c_dataframe['in_amt_wt'], c_dataframe['remaining_revenue']))
        c_dataframe['supply_proportions'] = list(map(lambda x,y: np.array(x) * y, c_dataframe['supply_amt_wt'], c_dataframe['remaining_expense']))

        # Adjust available inside trans
        c_dataframe['tran_per_rev_proportion'] = c_dataframe['revenue_proportions'] / amt_distributions[:,0]
        c_dataframe['i_tran_expected'] = list(map(lambda x:np.int(np.sum(x)), c_dataframe['tran_per_rev_proportion']))
        c_dataframe['i_inside_tran_expected'] = list(map(lambda x:np.int(p_inside_account * x), c_dataframe['i_tran_expected']))
        c_dataframe['i_inside_available'] = c_dataframe['i_inside_tran_expected']


        c_dataframe['tran_per_supply_proportion'] = c_dataframe['supply_proportions'] / amt_distributions[:,0]
        c_dataframe['s_tran_expected'] = list(map(lambda x:np.int(np.sum(x)), c_dataframe['tran_per_supply_proportion']))
        c_dataframe['s_inside_tran_expected'] = list(map(lambda x:np.int(p_inside_account * x), c_dataframe['s_tran_expected']))
        c_dataframe['s_inside_available'] = c_dataframe['s_inside_tran_expected']

    #else: ## Commented as it is already calculated in metadata
        #c_dataframe['revenue_proportions'] = list(map(lambda x, z: [round(x * y, 2) for y in z], c_dataframe['annual_revenue'], c_dataframe['in_amt_wt']))
        #c_dataframe['supply_proportions'] = list(map(lambda x, z: [round(x * y, 2) for y in z], c_dataframe['supply_expenses'], c_dataframe['supply_amt_wt']))

    # Update csv
    c_dataframe.to_csv(con.metadataFile, index_label='index')
 
    # Load distributions as per mode
    if (con.mode == 'H'):
        tran_distributions = allConfigParams['H']
        payment_mode_in_list, payment_mode_in_distribution = con.getParameterDistribution(tran_distributions['payment_mode_in'])
        payment_mode_out_list, payment_mode_out_distribution = con.getParameterDistribution(tran_distributions['payment_mode_out'])
        location_list, location_distribution = con.getParameterDistribution(tran_distributions['location'])
        tax_haven, tax_haven_distribution = con.getParameterDistribution(tran_distributions['is_tax_haven'])
        currency_list, curreny_distribution = con.getParameterDistribution(tran_distributions['currency'])
        in_acc_type_list, in_acc_type_distribution = con.getParameterDistribution(tran_distributions['in_acc_type'])
        supply_acc_type_list, supply_acc_type_distribution = con.getParameterDistribution(tran_distributions['supply_acc_type'])   
    
    for comp_id in comp_id_list:
        
        print('Company: ', comp_id)
        # get flag specific distributions
        
        # Fetch required details in one go
        c_acc_no, p_repeat_client, inside_customers, rev_proportions, supply_proportions, n_outside_suppliers, inside_suppliers, p_repeat_supplier, rev_per_emp, n_utilties, utility_accounts, utility_proportions, salary_distribution = c_dataframe.loc[comp_id, ['acc_no', 'p_repeat_client', 'inside_customers', 'revenue_proportions', 'supply_proportions', 'n_outside_suppliers', 'inside_suppliers', 'p_repeat_supplier', 'employee_profitability', 'n_utilties', 'utility_accs', 'utility_proportions', 'salary_distribution']]
        comp_id_idx = comp_id - companyIDSeed       # Index of comp_id in relation matrix
        if (con.mode == 'S'):
            flag = str(c_dataframe.loc[comp_id, 'flag'])
            tran_distributions = allConfigParams[flag]
            payment_mode_in_list, payment_mode_in_distribution = con.getParameterDistribution(tran_distributions['payment_mode_in'])
            payment_mode_out_list, payment_mode_out_distribution = con.getParameterDistribution(tran_distributions['payment_mode_out'])
            location_list, location_distribution = con.getParameterDistribution(tran_distributions['location'])
            tax_haven, tax_haven_distribution = con.getParameterDistribution(tran_distributions['is_tax_haven'])
            currency_list, curreny_distribution = con.getParameterDistribution(tran_distributions['currency'])
            in_acc_type_list, in_acc_type_distribution = con.getParameterDistribution(tran_distributions['in_acc_type'])
            supply_acc_type_list, supply_acc_type_distribution = con.getParameterDistribution(tran_distributions['supply_acc_type'])
            #p_inside_account = tran_distributions['p_inside_account']

        # Get candidate inside customers
        candidate_inside_customers = list()

        if len(inside_customers) > 0:
            
            for cust in inside_customers:
                cust_acc = int(cust[1:])
                if relation_mat[comp_id_idx][cust_acc-companyIDSeed] == 0 and c_dataframe.loc[cust_acc, 's_inside_available'] > 0:
                    candidate_inside_customers.append(cust)          # acc_no not comp_id
            if len(candidate_inside_customers) > 0 :
                candiate_customers_available = True 
                # Fetch supply proportions for candidate inside customers
                
                inside_cust_comp_id = [int(x[1:]) for x in candidate_inside_customers] # get their comp_id
                cust_sup_amt_dict = dict(c_dataframe.loc[inside_cust_comp_id, 'supply_proportions'])      # get their supply proportions
                cust_sup_tran_dict = dict(c_dataframe.loc[inside_cust_comp_id, 's_inside_available'])      # get their available supply trans, this helps in keeping track of available inside supply transactions

            else:
                candiate_customers_available = False
                cust_sup_amt_dict = dict()
                cust_sup_tran_dict = dict()
        else:       # As no candidate inside customers, no inside account probability
            candiate_customers_available = False
            cust_sup_amt_dict = dict()
            cust_sup_tran_dict = dict()

        performed_tran_with = list() # List of customers with whome actually transaction is performed, contains comp_id

        print('Generating revenue')

        rev_generated = [0.0] * len(rev_proportions)
        home_client_list = list()
        offshore_client_list = list()
        in_tran_records = list()
        
        dest_acc = c_acc_no
        dest_acc_type = 'C'
        tran_label = 'C'

        for i in range(4,-1,-1): # For each normal distribution
            delta = rev_proportions[i] - rev_generated[i]
            mean = amt_distributions[i][0]

            while delta > mean: # Generate transactions till proportion of those amounts is exhausted
               
                # Gnerate location, mode, amount, date 
                location = np.random.choice(location_list, p=location_distribution)
                payment_mode = np.random.choice(payment_mode_in_list, p=payment_mode_in_distribution)
                amount = abs(np.random.normal(amt_distributions[i][0], abs(amt_distributions[i][1])))
                date = np.random.choice(date_list, p=p_date)

                is_inside_account = False
                found_inside_customer = False

                if location == home_location:                           # For home location
                    currency = home_currency
                    is_tax_haven = home_is_tax_haven
                    
                    if payment_mode == "C":                             # Check if cash
                        src_acc = "CASHC"
                        src_acc_type = ""
                    else:                                               # Check inside account
                        found_inside_customer = False
                        is_inside_account = False
                        if candiate_customers_available: # Generate inside account probability if there are candidate customers
                            is_inside_account = np.random.choice([False, True], p = [1-p_inside_account, p_inside_account]) 
                            
                            if is_inside_account:                           # Choose from candidate accounts if inside account

                                available_cust = [k for k,v in cust_sup_amt_dict.items() if v[i] - amount > 0]

                                if len(available_cust) > 0:
                                    random.shuffle(available_cust)
                                    src_acc_comp_id = random.choice(available_cust)
                                    src_acc = 'C' + str(src_acc_comp_id)
                                    src_acc_type = 'C'
                                    found_inside_customer = True
                                    if src_acc_comp_id not in performed_tran_with:
                                        performed_tran_with.append(src_acc_comp_id)  
                                    

                        if not found_inside_customer:          # Check if repeat client 
                            is_repeat = np.random.choice([0, 1], p = [1-p_repeat_client, p_repeat_client])

                            if (is_repeat and len(home_client_list) > 0): # Generate new account if not is_repeat
                                src_acc_detail = random.choice(home_client_list)    # Choose existing client
                                src_acc = src_acc_detail[0]
                                src_acc_type = src_acc_detail[1]
                            else:
                                src_acc = 'a{0}' .format(comp_id) + str(random.randint(N, 10000*N))
                                src_acc_type = np.random.choice(in_acc_type_list, p=in_acc_type_distribution)
                                home_client_list.append ([src_acc, src_acc_type])
                        
                else:                                               # Offshore transaction
                    currency = np.random.choice(currency_list, p=curreny_distribution)
                    is_tax_haven = np.random.choice(tax_haven, p=tax_haven_distribution)
                    
                    if payment_mode == 'C':
                        src_acc = 'CASHC'
                        src_acc_type = ""
                    else:
                        is_repeat = np.random.choice([0, 1], p = [1-p_repeat_client, p_repeat_client]) # Check is_repeat
                        if is_repeat and len(offshore_client_list) > 0:
                            src_acc_details = random.choice(offshore_client_list)
                            src_acc = src_acc_details[0]
                            src_acc_type = src_acc_details[1]
                        else:
                            src_acc = 'a{0}' .format(comp_id) + str(random.randint(N, 10000*N))
                            src_acc_type = np.random.choice(in_acc_type_list, p=in_acc_type_distribution)
                            offshore_client_list.append([src_acc, src_acc_type])

                # Create record, add it to tran data and update revenue proportion, supply for inside account
                amount = ceil(amount) if payment_mode == 'C' else round(amount,2)
                record = [src_acc, src_acc_type, dest_acc, dest_acc_type, currency, payment_mode, amount, date, location, is_tax_haven, tran_label]
                in_tran_records.append(record)

                rev_generated[i] = rev_generated[i] + amount
                delta = rev_proportions[i] - rev_generated[i]

                if found_inside_customer:       # Update supply proportion dict and  available insisde supply trans for chosen inside customer
                    # Either update in dataframe or dictionary as the reference is same. Else, results in double deductions
                    c_dataframe.loc[src_acc_comp_id, 'supply_proportions'][i] = c_dataframe.loc[src_acc_comp_id, 'supply_proportions'][i] - amount
                    #c_dataframe.loc[src_acc_comp_id, 's_inside_available'] = c_dataframe.loc[src_acc_comp_id, 's_inside_available'] - 1
                    cust_sup_tran_dict[src_acc_comp_id] = cust_sup_tran_dict[src_acc_comp_id] - 1
                    if cust_sup_tran_dict[src_acc_comp_id] == 0:    # Remove customers who no longer are availbale
                        c_dataframe.loc[src_acc_comp_id, 's_inside_available'] = 0
                        cust_sup_tran_dict.pop(src_acc_comp_id)
                        cust_sup_amt_dict.pop(src_acc_comp_id)
                    if len(cust_sup_tran_dict) == 0:
                        candiate_customers_available = False

            # delta = generated - accumulated < mean, now make a mean tarnsaction or leave
            if delta > 0: # make last tran
                payment_mode = np.random.choice(payment_mode_in_list, p=payment_mode_in_distribution)
                location = np.random.choice(location_list, p=location_distribution)
                amount = ceil(delta)
                date = np.random.choice(date_list, p=p_date)
                is_repeat = np.random.choice([0, 1], p = [1-p_repeat_client, p_repeat_client]) # Check is_repeat

                if location == home_location:
                    currency = home_currency
                    is_tax_haven = home_is_tax_haven 
                else:
                    currency = np.random.choice(currency_list, p=curreny_distribution)
                    is_tax_haven = np.random.choice(tax_haven, p=tax_haven_distribution)

                if payment_mode == 'C':
                    src_acc = 'CASHC'
                    src_acc_type = ""
                elif is_repeat and location == home_location and len(home_client_list) > 0: # Repeat home client
                    src_acc_details = random.choice(home_client_list)
                    src_acc = src_acc_details[0]
                    src_acc_type = src_acc_details[1]
                elif is_repeat and location != home_location and len(offshore_client_list) > 0:    # Repeat offshore client
                    src_acc_details = random.choice(offshore_client_list)
                    src_acc = src_acc_details[0]
                    src_acc_type = src_acc_details[1]
                else:   # Generate new account
                    src_acc = 'a{0}' .format(comp_id) + str(random.randint(N, 10000*N))
                    src_acc_type = np.random.choice(in_acc_type_list, p=in_acc_type_distribution)
                    home_client_list.append([src_acc, src_acc_type]) if location == home_location else offshore_client_list.append([src_acc, src_acc_type])

                record = [src_acc, src_acc_type, dest_acc, dest_acc_type, currency, payment_mode, amount, date, location, is_tax_haven,tran_label]
                in_tran_records.append(record)

                rev_generated[i] = rev_generated[i] + amount

        rev_proportions = [a-b for a,b in zip(rev_proportions,rev_generated)] # Calculate remaining revenue and update

        if len(cust_sup_tran_dict.keys()) > 0:
            c_dataframe.loc[list(cust_sup_tran_dict.keys()), 's_inside_available'] = list(cust_sup_tran_dict.values())
        c_dataframe.set_value(comp_id,'revenue_proportions',rev_proportions) 

        # Write incoming records to df
        in_tran_df = pd.DataFrame(data = in_tran_records, columns=tran_col_list)
        in_tran_records.clear() # Trans transferred to df

        # If records exceed limit, write to file and truncate df
        if len(in_tran_df) > n_records:
            in_tran_df = writeToFile(in_tran_df)

        # Update relation matrix
        if len(performed_tran_with) > 0:
            performed_tran_with_idx = [x-companyIDSeed for x in performed_tran_with]
            relation_mat[performed_tran_with_idx, comp_id_idx] = 1      # Mark all tran as comp_id supplier are done

        ## Supply
      
        print ('Generating supply')

        supply_generated = [0.0] * len(supply_proportions) 
        # Get candidate suppliers
        candidate_inside_suppliers = list()
        if (len(inside_suppliers) > 0):
            for sup in inside_suppliers:
                sup_acc = int(sup[1:])
                if relation_mat[comp_id_idx][sup_acc-companyIDSeed] == 0 and c_dataframe.loc[sup_acc, 'i_inside_available'] > 0:
                    candidate_inside_suppliers.append(sup)      # acc_no not id
            if len(candidate_inside_suppliers) > 0:
                candidate_suppliers_available = True
                inside_sup_comp_id = [int(x[1:]) for x in candidate_inside_suppliers] # get their comp_id
                sup_amt_dict = dict(c_dataframe.loc[inside_sup_comp_id, 'revenue_proportions'])      # get their revenue proportions
                sup_i_tran_dict = dict(c_dataframe.loc[inside_sup_comp_id, 'i_inside_available'])      # get their available inside tran

            else:
                candidate_suppliers_available = False
                sup_amt_dict = dict()
                sup_i_tran_dict = dict()
        else:
            candidate_suppliers_available = False
            sup_amt_dict = dict()
            sup_i_tran_dict = dict()

        performed_tran_with = list() # List of suppliers with whome actually transaction is performed
        supply_records = list()

        # Generate supplier accounts
        outside_suppliers_acc_no = ['s' + str(random.randint(10000*N, 20000*N)) for x in range(n_outside_suppliers)]
        outside_suppliers_acc_type = np.random.choice(supply_acc_type_list, n_outside_suppliers, p=supply_acc_type_distribution)
        outside_suppliers = list(zip(outside_suppliers_acc_no, outside_suppliers_acc_type))
        offshore_suppliers = list()
        src_acc = c_acc_no
        src_acc_type = 'C'
        tran_label = 'S'

        for i in range(4,-1,-1):
            delta = supply_proportions[i] - supply_generated[i]
            mean = amt_distributions[i][0]
 
            while delta > mean:                              # For every proportion of supply

                # Get location, payment mode, date, amount
                location = np.random.choice(location_list, p=location_distribution)
                payment_mode = np.random.choice(payment_mode_out_list, p=payment_mode_out_distribution)
                date = np.random.choice(date_list, p=p_date)
                amount = abs(np.random.normal(amt_distributions[i][0], abs(amt_distributions[i][1])))

                is_inside_account = False
                found_inside_supplier = False

                if location == home_location:                               # For home location
                    currency = home_currency
                    is_tax_haven = home_is_tax_haven

                    if payment_mode == "C":                              # Check if cash transaction
                        dest_acc = "CASHS"
                        dest_acc_type = ""
                    else:                                                   # Check if inside account
                        is_inside_account = False
                        found_inside_supplier = False
                        # Logic modified a bit for supply transactions, first check is_repeat
                        
                        if candidate_suppliers_available :# Check inside account probability only if candiate suppliers available
                            is_inside_account = np.random.choice([False, True], p = [1-p_inside_account, p_inside_account]) 

                            if is_inside_account:
                                available_sup = [k for k,v in sup_amt_dict.items() if v[i] - amount > 0]

                                if len(available_sup) > 0:                           # Choose from candidate accounts if inside account
                                    random.shuffle(available_sup)
                                    dest_acc_comp_id = random.choice(available_sup)
                                    dest_acc = 'C' + str(dest_acc_comp_id)
                                    dest_acc_type = 'C'
                                    found_inside_supplier = True
                                    if dest_acc_comp_id not in performed_tran_with:
                                        performed_tran_with.append(dest_acc_comp_id)  

                        if not found_inside_supplier:          # Check if repeat client 
                            is_repeat = np.random.choice([0, 1], p = [1-p_repeat_supplier, p_repeat_supplier])

                            if is_repeat:
                                random.shuffle(outside_suppliers)
                                dest_acc_details = random.choice(outside_suppliers)
                                dest_acc = dest_acc_details[0]
                                dest_acc_type = dest_acc_details[1]
                            else:
                                dest_acc = 's{0}' .format(comp_id) + str(random.randint(N, 10000*N))
                                dest_acc_type = np.random.choice(supply_acc_type_list, p=supply_acc_type_distribution)
                                outside_suppliers.append ([dest_acc, dest_acc_type])
                        
                else:                                   # Offshore transaction

                    currency = np.random.choice(currency_list, p=curreny_distribution)
                    is_tax_haven = np.random.choice(tax_haven, p=tax_haven_distribution)
                    
                    if payment_mode == 'C':
                        dest_acc = 'CASHS'
                        dest_acc_type = ""
                    else:
                        is_repeat = np.random.choice([0, 1], p = [1-p_repeat_client, p_repeat_client]) # Check is_repeat
                        if is_repeat and len(offshore_suppliers) > 0:
                            dest_acc_details = random.choice(offshore_suppliers)
                            dest_acc = dest_acc_details[0]
                            dest_acc_type = dest_acc_details[1]
                        else:
                            dest_acc = 's{0}' .format(comp_id) + str(random.randint(N, 10000*N))
                            dest_acc_type = np.random.choice(supply_acc_type_list, p=supply_acc_type_distribution)
                            offshore_suppliers.append([dest_acc, dest_acc_type])

                # Create record, append, update supply and revenue for inside account
                amount = ceil(amount) if payment_mode == 'C' else round(amount,2)
                record = [src_acc, src_acc_type, dest_acc, dest_acc_type, currency, payment_mode, amount, date, location, is_tax_haven, tran_label]
                supply_records.append(record)

                supply_generated[i] = supply_generated[i] + amount
                delta = supply_proportions[i] - supply_generated[i]

                if found_inside_supplier: # Just check if friend found --9/2 added inside
                    # Either reduce form dictionary or dataframe. Else double deductions
                    c_dataframe.loc[dest_acc_comp_id, 'revenue_proportions'][i] = c_dataframe.loc[dest_acc_comp_id, 'revenue_proportions'][i] - amount 
                    #c_dataframe.loc[dest_acc_comp_id, 'i_inside_available'] = c_dataframe.loc[dest_acc_comp_id, 'i_inside_available'] - 1
                    sup_i_tran_dict[dest_acc_comp_id] = sup_i_tran_dict[dest_acc_comp_id] - 1  
                    if sup_i_tran_dict[dest_acc_comp_id] == 0:      # Remove suppliers who no longer are availbale
                        c_dataframe.loc[dest_acc_comp_id, 'i_inside_available'] = 0
                        sup_i_tran_dict.pop(dest_acc_comp_id)
                        sup_amt_dict.pop(dest_acc_comp_id)
                        if len(sup_i_tran_dict) == 0:
                            candidate_suppliers_available = False


            if delta > 0:       #Make last transaction
                
                payment_mode = np.random.choice(payment_mode_out_list, p=payment_mode_out_distribution)
                location = np.random.choice(location_list, p=location_distribution)
                amount = ceil(delta)
                date = np.random.choice(date_list, p=p_date)
                is_repeat = np.random.choice([0, 1], p = [1-p_repeat_client, p_repeat_client]) # Check is_repeat

                if location == home_location:
                    currency = home_currency
                    is_tax_haven = home_is_tax_haven 
                else:
                    currency = np.random.choice(currency_list, p=curreny_distribution)
                    is_tax_haven = np.random.choice(tax_haven, p=tax_haven_distribution)

                if payment_mode == 'C':
                    dest_acc = 'CASHS'
                    dest_acc_type = ""
                elif is_repeat and location == home_location and len(outside_suppliers) > 0: # Repeat home client
                    dest_acc_details = random.choice(outside_suppliers)
                    dest_acc = dest_acc_details[0]
                    dest_acc_type = dest_acc_details[1]
                elif is_repeat and location != home_location and len(offshore_suppliers) > 0:    # Repeat offshore client
                    dest_acc_details = random.choice(offshore_suppliers)
                    dest_acc = dest_acc_details[0]
                    dest_acc_type = dest_acc_details[1]
                else:   # Generate new account
                    dest_acc = 's{0}' .format(comp_id) + str(random.randint(N, 10000*N))
                    dest_acc_type = np.random.choice(supply_acc_type_list, p=supply_acc_type_distribution)
                    outside_suppliers.append([dest_acc, dest_acc_type]) if location == home_location else offshore_suppliers.append([dest_acc, dest_acc_type])

                record = [src_acc, src_acc_type, dest_acc, dest_acc_type, currency, payment_mode, amount, date, location, is_tax_haven, tran_label]
                in_tran_records.append(record)

                supply_generated[i] = supply_generated[i] + amount
        
        supply_proportions = [a-b for a,b in zip(supply_proportions,supply_generated)] # Calculate remaining supply and update
        c_dataframe.set_value(comp_id,'supply_proportions',supply_proportions) 

        if len(sup_i_tran_dict.keys()) > 0:
            c_dataframe.loc[list(sup_i_tran_dict.keys()), 'i_inside_available'] = list(sup_i_tran_dict.values())

        sup_df = pd.DataFrame(data = supply_records, columns = tran_col_list)
        
        # Clear structures
        supply_records.clear()
        outside_suppliers.clear()

        # Clear df if records exceed n_records
        if len(sup_df) > n_records:
            sup_df = writeToFile(sup_df)

        # Update relation matrix
        if len(performed_tran_with) > 0:
            performed_tran_with_idx = [x-companyIDSeed for x in performed_tran_with]
            relation_mat[comp_id_idx, performed_tran_with_idx] = 1      # Mark all tran as comp_id supplier are done
     
        if (con.mode == 'H' or con.mode == 'S'): # Generate salary and utility only if mode is H
            ## Salary
            print ('Generating salary')
            n_months = int(daterange[1] / 30)
            employees = e_dataframe.loc[e_dataframe.comp_id == comp_id]
            employees = employees.sort_values('emp_level')
            emp_list = [x for x in list(employees.acc_no)] * n_months
            emp_acc_type_list = [x for x in list(employees.acc_type)] * n_months
            emp_list_len = len(emp_list)
            payment_mode = ['T'] * emp_list_len
            currency = ['INR'] * emp_list_len
            location = [home_location] * emp_list_len
            is_tax_haven = [home_is_tax_haven] * emp_list_len
            src_acc = [c_acc_no] * emp_list_len
            src_acc_type = ['C'] * emp_list_len
            tran_label_list = ['E'] * emp_list_len
            sal_date_list = list()
            sal_list = list()

            if len(employees) != c_dataframe.loc[comp_id,'num_employees']:
                print('Invalid num_employees reported')
            #sal_distribution = sal_distribution_dict[rev_per_emp]
            salary_distribution.sort()

            for month in range(n_months):
                date = [random.randint((month+1)*25, (month+1)*30)] * len(employees.index)
                sal_date_list = sal_date_list + date
                for i in range(5):
                    level = i+1
                    sample = len(employees.loc[employees.emp_level == level])
                    sal = list(np.ceil(np.random.normal(salary_distribution[i], 0.1 * salary_distribution[i], sample)))
                    sal_list = sal_list + sal

            c_sal = list(zip (src_acc, src_acc_type, emp_list, emp_acc_type_list, currency, payment_mode, sal_list, sal_date_list,location, is_tax_haven,tran_label_list))
            sal_df = pd.DataFrame(data = c_sal, columns=tran_col_list)

            if len(sal_df) > n_records:
                sal_df = writeToFile(sal_df)
           
            c_sal.clear()
            emp_list.clear()
            emp_acc_type_list.clear()
            payment_mode.clear()
            currency.clear()
            location.clear()
            is_tax_haven.clear()
            src_acc.clear()
            src_acc_type.clear()
            tran_label_list.clear()
            sal_date_list.clear()
            sal_list.clear()
   
            ## Utility     
            print('Generating utility')

            utility_records = []
            utility_generated = [0.0] * len(utility_proportions)

            src_acc = c_acc_no
            src_acc_type = 'C'
            tran_label = 'U'

            for i in range(5):
                delta = utility_proportions[i] - utility_generated[i]
                mean = amt_distributions[i][0]

                currency = home_currency
                location = home_location
                is_tax_haven = home_is_tax_haven

                while delta > mean:
                    payment_mode = np.random.choice(payment_mode_out_list, p=payment_mode_out_distribution)
                    dest_acc = 'CASHU' if payment_mode == 'C' else random.choice(utility_accounts)
                    dest_acc_type = "" if payment_mode == 'C' else 'C'
                    date = np.random.choice(date_list, p=p_date)
                    selected_amt = abs(np.random.normal(amt_distributions[i][0], abs(amt_distributions[i][1])))
                    amount = ceil(selected_amt) if payment_mode == 'C' else round(selected_amt, 2)

                    record = [src_acc, src_acc_type, dest_acc, dest_acc_type, currency, payment_mode, amount, date, location, is_tax_haven,tran_label]
                    utility_records.append(record)
                    utility_generated[i] = utility_generated[i] + amount
                    delta = utility_proportions[i] - utility_generated[i]
                if delta > 0 :
                    dest_acc, dest_acc_type, currency, payment_mode, amount = 'CASHU', '',home_currency,'C',round(delta,2)
                    location = np.random.choice(location_list, p=location_distribution)
                    is_tax_haven = 'NO' if location == home_location else np.random.choice(tax_haven, p=tax_haven_distribution)
                    date = np.random.choice(date_list, p=p_date)
                    record = [src_acc, src_acc_type, dest_acc, dest_acc_type, currency, payment_mode, amount, date, location, is_tax_haven,tran_label]
                    utility_records.append(record)
                    utility_generated[i] = utility_generated[i] + amount
      
            utility_df = pd.DataFrame(data=utility_records, columns = tran_col_list)
            utility_records.clear()
            utility_accounts.clear()

            if len(utility_df) > n_records:
                utility_df = writeToFile(utility_df)
        else: # Empty df to avoid errors for Shell
            sal_df = pd.DataFrame(columns=tran_col_list)
            utility_df = pd.DataFrame(columns = tran_col_list)

        # Dump to parent clear earlier structures
        parent_df = pd.concat([parent_df, in_tran_df, sup_df, sal_df, utility_df])
        in_tran_df = in_tran_df.iloc[0:0]
        sal_df = sal_df.iloc[0:0]
        sup_df = sup_df.iloc[0:0]
        utility_df = utility_df.iloc[0:0]

        if len(parent_df) > n_records:
            parent_df = writeToFile(parent_df)

    print('Almost done.. writing the last file..')
    # Update csv
    c_dataframe.to_csv(con.metadataFile, index_label='index')

    if len(parent_df) > 0:
        parent_df = writeToFile(parent_df)

    