import numpy as np
import pandas as pd
import configRead as con
import glob as glob
import math
from scipy.stats import entropy
import ast
import sys
import matplotlib.pyplot as plt
import os

# Globals
tran_col_list = ['src_acc', 'src_acc_type', 'dest_acc', 'dest_acc_type', 'currency', 'payment_mode', 'amount', 'date', 'location', 'is_tax_haven', 'tran_label']

def readTransactionData (trandataPath, startId, endId):
    '''This function reads trandata files in chunks'''

    allFiles = [trandataPath + '\\tran{0}.csv' .format(x) for x in range(startId, endId+1)]
    df_list = list()
    tran_data_df = pd.DataFrame(columns=tran_col_list)
    for f in allFiles:
        df = pd.read_csv(f)
        tran_data_df = pd.concat([tran_data_df, df])
        df=df.iloc[0:0]
    return tran_data_df

def prepareSummary (parent_df, trandata_df, home_location):

    company_acc = list(parent_df['acc_no'])

    # Incoming transactions total
    in_total_df = trandata_df.loc[trandata_df.dest_acc.isin(company_acc)]
    in_total_grouped = in_total_df.groupby('dest_acc',as_index=False).agg({"amount": [np.sum, 'count']})
    in_total_grouped.columns = in_total_grouped.columns.droplevel(level=0)
    in_total_grouped.columns = ['acc_no', 'generated_annual_revenue', 'n_incoming_total']
    parent_df = pd.merge(parent_df, in_total_grouped, on='acc_no', how='left')

    # Incoming transctions within
    in_within_df = in_total_df.loc[in_total_df.src_acc.isin(company_acc)]
    in_within_grouped = in_within_df.groupby('dest_acc',as_index=False).agg({"amount": [np.sum, 'count']})
    in_within_grouped.columns = in_within_grouped.columns.droplevel(level=0)
    in_within_grouped.columns = ['acc_no', 'revenue_within','n_incoming_within']
    parent_df = pd.merge(parent_df, in_within_grouped, on='acc_no', how='left')

    # Incoming transctions group
    in_group_df = in_total_df.loc[in_total_df.tran_label == 'G']
    in_group_grouped = in_group_df.groupby('dest_acc',as_index=False).agg({"amount": [np.sum, 'count']})
    in_group_grouped.columns = in_group_grouped.columns.droplevel(level=0)
    in_group_grouped.columns = ['acc_no', 'revenue_group','n_incoming_group']
    parent_df = pd.merge(parent_df, in_group_grouped, on='acc_no', how='left')
 
    # Outgoing transactions total
    out_total_df = trandata_df.loc[(trandata_df.src_acc.isin(company_acc))]
    out_total_grouped = out_total_df.groupby('src_acc',as_index=False).agg({"amount":[np.sum, 'count']})
    out_total_grouped.columns = out_total_grouped.columns.droplevel(level=0)
    out_total_grouped.columns = ['acc_no', 'generated_expenses', 'n_outgoing_total']
    parent_df = pd.merge(parent_df, out_total_grouped, on='acc_no', how='left')

    # Outgoing transactions within
    out_within_df = out_total_df.loc[out_total_df.dest_acc.isin(company_acc)]
    out_within_grouped = out_within_df.groupby('src_acc',as_index=False).agg({"amount": [np.sum, 'count']})
    out_within_grouped.columns = out_within_grouped.columns.droplevel(level=0)
    out_within_grouped.columns = ['acc_no', 'expenses_within', 'n_outgoing_within']
    parent_df = pd.merge(parent_df, out_within_grouped, on='acc_no', how='left')

    # Outgoing transactions group
    out_group_df = out_total_df.loc[out_total_df.tran_label == 'G']
    out_group_grouped = out_group_df.groupby('src_acc',as_index=False).agg({"amount": [np.sum, 'count']})
    out_group_grouped.columns = out_group_grouped.columns.droplevel(level=0)
    out_group_grouped.columns = ['acc_no', 'expenses_group', 'n_outgoing_group']
    parent_df = pd.merge(parent_df, out_group_grouped, on='acc_no', how='left')

    # Outgoing transactions - supply
    out_supply_df = out_total_df.loc[out_total_df.tran_label =='S']
    out_supply_grouped = out_supply_df.groupby('src_acc',as_index=False).agg({"amount": [np.sum, 'count']})
    out_supply_grouped.columns = out_supply_grouped.columns.droplevel(level=0)
    out_supply_grouped.columns = ['acc_no', 'supply_expense', 'n_supply_tran']
    parent_df = pd.merge(parent_df, out_supply_grouped, on='acc_no', how='left')

    # Outgoing transactions - utility
    out_utility_df = out_total_df.loc[out_total_df.tran_label == 'U']
    out_utility_grouped = out_utility_df.groupby('src_acc',as_index=False).agg({"amount": [np.sum, 'count']})
    out_utility_grouped.columns = out_utility_grouped.columns.droplevel(level=0)
    out_utility_grouped.columns = ['acc_no', 'utility_expense', 'n_utility_tran']
    parent_df = pd.merge(parent_df, out_utility_grouped, on='acc_no', how='left')

    # Outgoing transactions - salary
    out_salary_df = out_total_df.loc[out_total_df.tran_label == 'E']
    out_salary_grouped = out_salary_df.groupby('src_acc',as_index=False).agg({"amount": [np.sum, 'count']})
    out_salary_grouped.columns = out_salary_grouped.columns.droplevel(level=0)
    out_salary_grouped.columns = ['acc_no', 'salary_expense', 'n_salary_tran']
    parent_df = pd.merge(parent_df, out_salary_grouped, on='acc_no', how='left')

    # Cash transactions incoming
    in_cash_df = in_total_df.loc[in_total_df.payment_mode == 'C']
    in_cash_grouped = in_cash_df.groupby('dest_acc',as_index=False).agg({"amount":[np.sum, 'count']})
    in_cash_grouped.columns = in_cash_grouped.columns.droplevel(level=0)
    in_cash_grouped.columns = ['acc_no', 'cash_amt_incoming', 'n_cash_incoming']
    parent_df = pd.merge(parent_df, in_cash_grouped, on='acc_no', how='left')

    # Cash transactions outgoing
    out_cash_df = out_total_df.loc[out_total_df.payment_mode == 'C']
    out_cash_grouped = out_cash_df.groupby('src_acc', as_index = False).agg({"amount":[np.sum, 'count']})
    out_cash_grouped.columns = out_cash_grouped.columns.droplevel(level=0)
    out_cash_grouped.columns = ['acc_no', 'cash_amt_outgoing', 'n_cash_outgoing']
    parent_df = pd.merge(parent_df, out_cash_grouped, on='acc_no', how='left')

    # Check transactions incoming
    in_check_df = in_total_df.loc[in_total_df.payment_mode == 'Q']
    in_check_grouped = in_check_df.groupby('dest_acc',as_index=False).agg({"amount":[np.sum, 'count']})
    in_check_grouped.columns = in_check_grouped.columns.droplevel(level=0)
    in_check_grouped.columns = ['acc_no', 'check_amt_incoming', 'n_check_incoming']
    parent_df = pd.merge(parent_df, in_check_grouped, on='acc_no', how='left')

    # Check transactions outgoing
    out_check_df = out_total_df.loc[out_total_df.payment_mode == 'Q']
    out_check_grouped = out_check_df.groupby('src_acc', as_index = False).agg({"amount":[np.sum, 'count']})
    out_check_grouped.columns = out_check_grouped.columns.droplevel(level=0)
    out_check_grouped.columns = ['acc_no', 'check_amt_outgoing', 'n_check_outgoing']
    parent_df = pd.merge(parent_df, out_check_grouped, on='acc_no', how='left')

    # Transfer transactions incoming
    in_transfer_df = in_total_df.loc[in_total_df.payment_mode == 'T']
    in_transfer_grouped = in_transfer_df.groupby('dest_acc',as_index=False).agg({"amount":[np.sum, 'count']})
    in_transfer_grouped.columns = in_transfer_grouped.columns.droplevel(level=0)
    in_transfer_grouped.columns = ['acc_no', 'transfer_amt_incoming', 'n_transfer_incoming']
    parent_df = pd.merge(parent_df, in_transfer_grouped, on='acc_no', how='left')

    # Transfer transactions outgoing
    out_transfer_df = out_total_df.loc[out_total_df.payment_mode == 'T']
    out_transfer_grouped = out_transfer_df.groupby('src_acc', as_index = False).agg({"amount":[np.sum, 'count']})
    out_transfer_grouped.columns = out_transfer_grouped.columns.droplevel(level=0)
    out_transfer_grouped.columns = ['acc_no', 'transfer_amt_outgoing', 'n_transfer_outgoing']
    parent_df = pd.merge(parent_df, out_transfer_grouped, on='acc_no', how='left')

    # Local incoming
    in_local_df = in_total_df.loc[in_total_df.location == home_location]
    in_local_grouped = in_local_df.groupby('dest_acc',as_index=False).agg({"amount":[np.sum, 'count']})
    in_local_grouped.columns = in_local_grouped.columns.droplevel(level=0)
    in_local_grouped.columns = ['acc_no', 'local_amt_incoming', 'n_local_incoming']
    parent_df = pd.merge(parent_df, in_local_grouped, on='acc_no', how='left')

    # Local outgoing
    out_transfer_df = out_total_df.loc[out_total_df.location == home_location]
    out_transfer_grouped = out_transfer_df.groupby('src_acc', as_index = False).agg({"amount":[np.sum, 'count']})
    out_transfer_grouped.columns = out_transfer_grouped.columns.droplevel(level=0)
    out_transfer_grouped.columns = ['acc_no', 'local_amt_outgoing', 'n_local_outgoing']
    parent_df = pd.merge(parent_df, out_transfer_grouped, on='acc_no', how='left')

    # Offshore incoming
    in_offshore_df = in_total_df.loc[in_total_df.location == 'OFFSHORE']
    in_offshore_grouped = in_offshore_df.groupby('dest_acc',as_index=False).agg({"amount":[np.sum, 'count']})
    in_offshore_grouped.columns = in_offshore_grouped.columns.droplevel(level=0)
    in_offshore_grouped.columns = ['acc_no', 'offshore_amt_incoming', 'n_offshore_incoming']
    parent_df = pd.merge(parent_df, in_offshore_grouped, on='acc_no', how='left')

    # Offshore outgoing
    out_offshore_df = out_total_df.loc[out_total_df.location == 'OFFSHORE']
    out_offshore_grouped = out_offshore_df.groupby('src_acc', as_index = False).agg({"amount":[np.sum, 'count']})
    out_offshore_grouped.columns = out_offshore_grouped.columns.droplevel(level=0)
    out_offshore_grouped.columns = ['acc_no', 'offshore_amt_outgoing', 'n_offshore_outgoing']
    parent_df = pd.merge(parent_df, out_offshore_grouped, on='acc_no', how='left')

    # Tax haven incoming
    in_tax_haven_df = in_total_df.loc[in_total_df.is_tax_haven == 'YES']
    in_tax_haven_grouped = in_tax_haven_df.groupby('dest_acc',as_index=False).agg({"amount":[np.sum, 'count']})
    in_tax_haven_grouped.columns = in_tax_haven_grouped.columns.droplevel(level=0)
    in_tax_haven_grouped.columns = ['acc_no', 'tax_haven_amt_incoming', 'n_tax_haven_incoming']
    parent_df = pd.merge(parent_df, in_tax_haven_grouped, on='acc_no', how='left')

    # tax haven outgoing
    out_tax_haven_df = out_total_df.loc[out_total_df.is_tax_haven == 'YES']
    out_tax_haven_grouped = out_tax_haven_df.groupby('src_acc', as_index = False).agg({"amount":[np.sum, 'count']})
    out_tax_haven_grouped.columns = out_tax_haven_grouped.columns.droplevel(level=0)
    out_tax_haven_grouped.columns = ['acc_no', 'tax_haven_amt_outgoing', 'n_tax_haven_outgoing']
    parent_df = pd.merge(parent_df, out_tax_haven_grouped, on='acc_no', how='left')

    # Current incoming
    in_current_df = in_total_df.loc[in_total_df.src_acc_type == 'C']
    in_current_grouped = in_current_df.groupby('dest_acc',as_index=False).agg({"amount":[np.sum, 'count']})
    in_current_grouped.columns = in_current_grouped.columns.droplevel(level=0)
    in_current_grouped.columns = ['acc_no', 'current_amt_incoming', 'n_current_incoming']
    parent_df = pd.merge(parent_df, in_current_grouped, on='acc_no', how='left')

    # Current outgoing
    out_current_df = out_total_df.loc[out_total_df.dest_acc_type == 'C']
    out_current_grouped = out_current_df.groupby('src_acc', as_index = False).agg({"amount":[np.sum, 'count']})
    out_current_grouped.columns = out_current_grouped.columns.droplevel(level=0)
    out_current_grouped.columns = ['acc_no', 'current_amt_outgoing', 'n_current_outgoing']
    parent_df = pd.merge(parent_df, out_current_grouped, on='acc_no', how='left')

    # Savings incoming
    in_savings_df = in_total_df.loc[in_total_df.src_acc_type == 'S']
    in_savings_grouped = in_savings_df.groupby('dest_acc',as_index=False).agg({"amount":[np.sum, 'count']})
    in_savings_grouped.columns = in_savings_grouped.columns.droplevel(level=0)
    in_savings_grouped.columns = ['acc_no', 'savings_amt_incoming', 'n_savings_incoming']
    parent_df = pd.merge(parent_df, in_savings_grouped, on='acc_no', how='left')

    # Savings outgoing
    out_savings_df = out_total_df.loc[out_total_df.dest_acc_type == 'S']
    out_savings_grouped = out_savings_df.groupby('src_acc', as_index = False).agg({"amount":[np.sum, 'count']})
    out_savings_grouped.columns = out_savings_grouped.columns.droplevel(level=0)
    out_savings_grouped.columns = ['acc_no', 'savings_amt_outgoing', 'n_savings_outgoing']
    parent_df = pd.merge(parent_df, out_savings_grouped, on='acc_no', how='left')

    # Tran info for graph
    trandata_df['month_no'] = np.floor(trandata_df['date'] / 30.5)
    trandata_df['month_no'] = trandata_df.month_no.astype(int)
    trandata_df['week_no'] = np.floor(trandata_df['date'] / 7)
    trandata_df['week_no'] = trandata_df.week_no.astype(int)

    tran_info_df = trandata_df.groupby(by=['src_acc', 'dest_acc', 'src_acc_type', 'dest_acc_type', 'date', 'week_no', 'month_no'], as_index = False).agg({'amount':[np.sum, 'count']})
    tran_info_df.columns = tran_info_df.columns.droplevel(level=0)
    tran_info_df.columns = ['src_acc', 'dest_acc','src_acc_type', 'dest_acc_type','date', 'week_no', 'month_no','amount', 'n_tran']

    return parent_df, tran_info_df

def combinePartialSummaries (summaryDataPath, parent_df):
    all_files = glob.glob(summaryDataPath + 'par*.csv')
    df_list = list()
    for file in all_files:
        df = pd.read_csv(file)
        df_list.append(df)
    combinedFiles_df = pd.concat(df_list)
    combinedFiles_df_grouped = combinedFiles_df.groupby(by='acc_no', as_index=False).agg({'generated_annual_revenue':np.sum, 'n_incoming_total':np.sum, 'revenue_within':np.sum,'n_incoming_within':np.sum, 'revenue_group':np.sum, 'n_incoming_group':np.sum, 'generated_expenses':np.sum, 'n_outgoing_total':np.sum, 'expenses_within':np.sum, 'n_outgoing_within':np.sum, 'expenses_group':np.sum, 'n_outgoing_group':np.sum, 'supply_expense':np.sum, 'n_supply_tran':np.sum, 'utility_expense':np.sum, 'n_utility_tran':np.sum, 'salary_expense':np.sum, 'n_salary_tran':np.sum, 'cash_amt_incoming':np.sum,'n_cash_incoming':np.sum, 'cash_amt_outgoing':np.sum,'n_cash_outgoing':np.sum, 'check_amt_incoming':np.sum, 'n_check_incoming':np.sum, 'check_amt_outgoing':np.sum, 'n_check_outgoing':np.sum, 'transfer_amt_incoming':np.sum, 'n_transfer_incoming':np.sum, 'transfer_amt_outgoing':np.sum, 'n_transfer_outgoing':np.sum, 'local_amt_incoming':np.sum, 'n_local_incoming':np.sum, 'local_amt_outgoing':np.sum, 'n_local_outgoing':np.sum, 'offshore_amt_incoming':np.sum, 'n_offshore_incoming':np.sum, 'offshore_amt_outgoing':np.sum, 'n_offshore_outgoing':np.sum, 'tax_haven_amt_incoming':np.sum, 'n_tax_haven_incoming':np.sum, 'tax_haven_amt_outgoing':np.sum, 'n_tax_haven_outgoing':np.sum, 'current_amt_incoming':np.sum, 'n_current_incoming':np.sum, 'current_amt_outgoing':np.sum, 'n_current_outgoing':np.sum, 'savings_amt_incoming':np.sum, 'n_savings_incoming':np.sum, 'savings_amt_outgoing':np.sum, 'n_savings_outgoing':np.sum})

    summarized_df = pd.merge(parent_df, combinedFiles_df_grouped, how='inner', on='acc_no') # Join annual_revenue

    # Add additional info
    #summarized_df['current_incoming_per'] = np.round(summarized_df['annual_revenue_deviation'] / summarized_df['annual_revenue'] * 100, 2)
    summarized_df['n_total_tran'] = summarized_df['n_incoming_total'] + summarized_df['n_outgoing_total']
    summarized_df['annual_revenue_deviation'] = np.round(summarized_df['annual_revenue'] - summarized_df['generated_annual_revenue'], 2)
    summarized_df['revenue_deviation_per'] = np.round(summarized_df['annual_revenue_deviation'] / summarized_df['annual_revenue'] * 100, 2)
    summarized_df['group_in_tran_per'] = np.round(summarized_df['n_incoming_group'] / summarized_df['n_incoming_total'] * 100,2)
    summarized_df['group_out_tran_per'] = np.round(summarized_df['n_outgoing_group'] / summarized_df['n_outgoing_total'] * 100,2)
    summarized_df['group_revenue_per'] = np.round(summarized_df['revenue_group'] / summarized_df['generated_annual_revenue'] * 100,2)
    summarized_df['group_expense_per'] = np.round(summarized_df['expenses_group'] / summarized_df['generated_annual_revenue'] * 100,2)
    summarized_df['supply_per'] = np.round(summarized_df['supply_expense'] / summarized_df['generated_annual_revenue'] * 100,2)
    summarized_df['salary_per'] = np.round(summarized_df['salary_expense'] / summarized_df['generated_annual_revenue'] * 100,2)
    summarized_df['utility_per'] = np.round(summarized_df['utility_expense'] / summarized_df['generated_annual_revenue'] * 100,2)
    summarized_df['total_expense_per'] = np.round(summarized_df['generated_expenses'] / summarized_df['generated_annual_revenue'] * 100,2)
    summarized_df['profit_loss'] = np.round(summarized_df['generated_annual_revenue'] - summarized_df['generated_expenses'], 2)
    summarized_df['profit_loss_per'] = np.round(summarized_df['profit_loss'] / summarized_df['generated_annual_revenue'] * 100,2)
    summarized_df['i_tran_within_percentage'] = summarized_df['n_incoming_within'] / summarized_df['n_incoming_total'] * 100
    summarized_df['o_tran_within_percentage'] = summarized_df['n_outgoing_within'] / summarized_df['n_outgoing_total'] * 100

    return summarized_df

def combineTranInfoFiles (tranInfoPath):
    '''this function combines partial tranInfo files'''
    # Combine partial Traninfo files
    all_files = glob.glob(tranInfoPath + 'par*.csv')
    df_list = list()
    for file in all_files:
        df = pd.read_csv(file)
        df_list.append(df)

    # Combined df
    combinedFiles_df = pd.concat(df_list)

    combinedFiles_df_grouped = combinedFiles_df.groupby(by=['src_acc','dest_acc', 'src_acc_type','dest_acc_type'], as_index=False).agg({'amount': [np.sum, np.mean, np.std], 'n_tran':[np.sum,np.mean, np.std]})
    combinedFiles_df_grouped.columns = combinedFiles_df_grouped.columns.droplevel(level=0)
    combinedFiles_df_grouped.columns = ['src_acc', 'dest_acc', 'src_acc_type','dest_acc_type', 'amount', 'mean_amount', 'std_amount','n_tran', 'mean_n_tran', 'std_n_tran'] # These are edges
    '''
     # group by month
    tran_info_month = combinedFiles_df.groupby(by=['src_acc', 'dest_acc', 'src_acc_type', 'dest_acc_type', 'month_no'], as_index = False).agg({'amount': [np.sum, np.sum]})
    tran_info_month.columns = tran_info_month.columns.droplevel(level=0)
    tran_info_month.columns = ['src_acc', 'dest_acc', 'src_acc_type', 'dest_acc_type', 'month_no', 'amount_per_month', 'n_tran_per_month']
    tran_info_month = tran_info_month.groupby(by=['src_acc', 'dest_acc', 'src_acc_type', 'dest_acc_type'], as_index = False).agg({'amount_per_month': [np.mean, np.std, np.max], 'n_tran_per_month':[np.mean, np.std, np.max]})
    tran_info_month.columns = tran_info_month.columns.droplevel(level=0)
    tran_info_month.columns = ['src_acc', 'dest_acc','src_acc_type', 'dest_acc_type','mean_amount_per_month', 'std_amount_per_month', 'max_amount_per_month', 'mean_n_tran_per_month', 'std_n_tran_per_month', 'max_n_tran_per_month']
    combinedFiles_df_grouped = pd.merge(combinedFiles_df_grouped, tran_info_month, how='inner', on=['src_acc', 'dest_acc', 'src_acc_type', 'dest_acc_type'])

     # group by week
    tran_info_week = combinedFiles_df.groupby(by=['src_acc', 'dest_acc', 'src_acc_type', 'dest_acc_type', 'week_no'], as_index = False).agg({'amount': [np.sum, np.sum]})
    tran_info_week.columns = tran_info_week.columns.droplevel(level=0)
    tran_info_week.columns = ['src_acc', 'dest_acc', 'src_acc_type', 'dest_acc_type', 'week_no', 'amount_per_week', 'n_tran_per_week']
    tran_info_week = tran_info_week.groupby(by=['src_acc', 'dest_acc', 'src_acc_type', 'dest_acc_type'], as_index = False).agg({'amount_per_week': [np.mean, np.std, np.max], 'n_tran_per_week':[np.mean, np.std, np.max]})
    tran_info_week.columns = tran_info_week.columns.droplevel(level=0)
    tran_info_week.columns = ['src_acc', 'dest_acc','src_acc_type', 'dest_acc_type','mean_amount_per_week', 'std_amount_per_week', 'max_amount_per_week', 'mean_n_tran_per_week', 'std_n_tran_per_week', 'max_n_tran_per_week']
    combinedFiles_df_grouped = pd.merge(combinedFiles_df_grouped, tran_info_week, how='inner', on=['src_acc', 'dest_acc', 'src_acc_type', 'dest_acc_type'])

    # group by date 
    tran_info_date = combinedFiles_df.groupby(by=['src_acc', 'dest_acc', 'src_acc_type', 'dest_acc_type', 'date'], as_index = False).agg({'amount': [np.sum, np.sum]})
    tran_info_date.columns = tran_info_date.columns.droplevel(level=0)
    tran_info_date.columns = ['src_acc', 'dest_acc', 'src_acc_type', 'dest_acc_type', 'date', 'amount_per_day', 'n_tran_per_day']
    tran_info_date = tran_info_date.groupby(by=['src_acc', 'dest_acc', 'src_acc_type', 'dest_acc_type'], as_index = False).agg({'amount_per_day': [np.mean, np.std, np.max], 'n_tran_per_day':[np.mean, np.std, np.max]})
    tran_info_date.columns = tran_info_date.columns.droplevel(level=0)
    tran_info_date.columns = ['src_acc', 'dest_acc','src_acc_type', 'dest_acc_type','mean_amount_per_day', 'std_amount_per_day',  'max_amount_per_day', 'mean_n_tran_per_day', 'std_n_tran_per_day', 'max_n_tran_per_day']
    combinedFiles_df_grouped = pd.merge(combinedFiles_df_grouped, tran_info_date, how='inner', on=['src_acc', 'dest_acc', 'src_acc_type', 'dest_acc_type'])
    '''

    return combinedFiles_df_grouped

def old_combineTranInfoFiles (tranInfoPath):
    '''this function combines partial tranInfo files'''
    # Combine partial Traninfo files
    all_files = glob.glob(tranInfoPath + 'par*.csv')
    df_list = list()
    for file in all_files:
        df = pd.read_csv(file)
        df_list.append(df)
    combinedFiles_df = pd.concat(df_list)

    combinedFiles_df_grouped = combinedFiles_df.groupby(by=['src_acc','dest_acc', 'src_acc_type','dest_acc_type'], as_index=False).agg({'amount': [np.sum, np.mean, np.std], 'n_tran':[np.sum,np.mean, np.std]})
    combinedFiles_df_grouped.columns = combinedFiles_df_grouped.columns.droplevel(level=0)
    combinedFiles_df_grouped.columns = ['src_acc', 'dest_acc', 'src_acc_type','dest_acc_type', 'amount', 'mean_amount', 'std_amount','n_tran', 'mean_n_tran', 'std_n_tran'] # These are edges

    print('Month summary')
     # group by month
    tran_info_month = combinedFiles_df.groupby(by=['src_acc', 'dest_acc', 'src_acc_type', 'dest_acc_type', 'month_no'], as_index = False).agg({'amount': np.sum, 'n_tran': np.sum})
    #tran_info_month.columns = tran_info_month.columns.droplevel(level=0)
    tran_info_month.columns = ['src_acc', 'dest_acc', 'src_acc_type', 'dest_acc_type', 'month_no', 'amount_in_month', 'n_tran_in_month']
    tran_info_month = tran_info_month.groupby(by=['src_acc', 'dest_acc', 'src_acc_type', 'dest_acc_type'], as_index = False).agg({'amount_in_month': np.max, 'n_tran_in_month': np.max})
    tran_info_month.columns = tran_info_month.columns.droplevel(level=0)
    tran_info_month.columns = ['src_acc', 'dest_acc','src_acc_type', 'dest_acc_type','max_amount_in_month','max_n_tran_in_month']
    combinedFiles_df_grouped = pd.merge(combinedFiles_df_grouped, tran_info_month, how='inner', on=['src_acc', 'dest_acc', 'src_acc_type', 'dest_acc_type']) # These are edges

    print('Week summary')
    # group by week
    tran_info_week = combinedFiles_df.groupby(by=['src_acc', 'dest_acc', 'src_acc_type', 'dest_acc_type', 'week_no'], as_index = False).agg({'amount': np.sum, 'n_tran': np.sum})
    #tran_info_month.columns = tran_info_month.columns.droplevel(level=0)
    tran_info_week.columns = ['src_acc', 'dest_acc', 'src_acc_type', 'dest_acc_type', 'week_no', 'amount_in_week', 'n_tran_in_week']
    tran_info_week = tran_info_week.groupby(by=['src_acc', 'dest_acc', 'src_acc_type', 'dest_acc_type'], as_index = False).agg({'amount_in_week': np.max, 'n_tran_in_week':np.max})
    #tran_info_week.columns = tran_info_week.columns.droplevel(level=0)
    tran_info_week.columns = ['src_acc', 'dest_acc','src_acc_type', 'dest_acc_type', 'max_amount_in_week', 'max_n_tran_in_week']
    combinedFiles_df_grouped = pd.merge(combinedFiles_df_grouped, tran_info_week, how='inner', on=['src_acc', 'dest_acc', 'src_acc_type', 'dest_acc_type']) # These are edges

    return combinedFiles_df_grouped

def probDistributionCheck (paramName, paramValue, metadata_df):
    '''This function calculates KL divergence of paramName parameter '''
    param_grp = metadata_df.groupby(paramName).comp_id.agg('count')
    param_grp = param_grp / np.sum(param_grp)
    actual_dist = param_grp.get_values()
    expected_dist =  list({k:v for k,v in sorted(paramValue.items())}.values())
    KLD = entropy(actual_dist, expected_dist)
    status = 'P' if abs(KLD) < 0.05 else 'F'
    return expected_dist, actual_dist, KLD, status

def summarizeData ():

    # Read config files
    print('Reading config parameters for data summary')
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
        print('Error in reading config files.. Make sure you have not made any mistake in editing parameters')

    # Read metadata
    metadata_df = pd.DataFrame()
    try:
        metadata_df = pd.read_csv(con.metadataFile, converters={'in_amt_wt':ast.literal_eval, 'supply_amt_wt':ast.literal_eval, 'utility_proportions': ast.literal_eval, 'inside_customers': ast.literal_eval, 'inside_suppliers':ast.literal_eval, 'utility_accs':ast.literal_eval},index_col='index')
    except:
        print('Error in reading metadata')
    
    metadata_df['industry_sector'] = metadata_df.industry_sector.astype('str')
    company_acc = metadata_df['acc_no']
    N = len(metadata_df)
    print('Metadata for ', N, ' companies is read successfully!')

    # Load tran data in chunks prepare summary for small data
    allFiles = glob.glob(con.tranData + '\*.csv')
    n_files = len(allFiles)
  
    if n_files == 0:
        print('No transaction data found!! Make sure trandatapath parameter is set appropriately.')
    else:
        chunk_size = 8
        n_iterations = math.ceil(n_files / chunk_size)
        # Prepare folder for summarized data
        companyDataPath = modeparams['company_data_path']
        summaryDataPath = companyDataPath + 'SummaryData\\'
        tranInfoPath = companyDataPath + 'TranInfo\\'
        if not os.path.exists(summaryDataPath):
            os.makedirs(summaryDataPath)
        if not os.path.exists(tranInfoPath):
            os.makedirs(tranInfoPath)
            
        i = 0

        for i in range(n_iterations): 

            # Read chunk
            start_index = i * chunk_size
            end_index = start_index + chunk_size - 1 if i < n_iterations - 1 else n_files -1
            print('Files getting read from ', start_index, end_index)
     
            tran_data_df = readTransactionData(con.tranData, start_index, end_index)

            # Summarize chunk
            partial_summary_df = metadata_df[['acc_no']]
            partial_summary_df, partial_traninfo_df = prepareSummary (partial_summary_df, tran_data_df, home_location)

            # Write summary to disk
            file_path = summaryDataPath + 'partialSummary' + str(i) + '.csv'
            partial_summary_df.to_csv (file_path)

            file_path = tranInfoPath + 'partialTranInfo' + str(i) + '.csv'
            partial_traninfo_df.to_csv (file_path)

        print('Partial summary files are written..')
        # Propare final summary
        summarized_df = metadata_df[['acc_no', 'annual_revenue']]
        
        summarized_df = combinePartialSummaries(summaryDataPath, summarized_df)
        combinedFiles_df_grouped = combineTranInfoFiles (tranInfoPath)

        # Find out clients and expense account from both

        # Get n_clients_total, n_clients_within, n_expense_accs_total, n_expense_acc_within
        acc_no_list = list(metadata_df.loc[:,'acc_no'])
        for_clients_total = combinedFiles_df_grouped.loc[(combinedFiles_df_grouped.src_acc != 'CASHC') & (combinedFiles_df_grouped.dest_acc.isin(acc_no_list))]
        for_clients_total_grouped = for_clients_total.groupby(by='dest_acc', as_index=False).agg({'src_acc':pd.Series.nunique})
        for_clients_total_grouped.columns = ['acc_no', 'n_clients_total']
        summarized_df = summarized_df.merge(for_clients_total_grouped, on='acc_no',how='left')

        for_clients_within = for_clients_total.loc[for_clients_total.src_acc.isin(acc_no_list)]
        for_clients_within = for_clients_within.groupby(by='dest_acc', as_index=False).agg({'src_acc':pd.Series.nunique})
        for_clients_within.columns = ['acc_no', 'n_clients_within']
        summarized_df = summarized_df.merge(for_clients_within, on='acc_no',how='left')

        for_exp_acc_total = combinedFiles_df_grouped.loc[(~combinedFiles_df_grouped.dest_acc.isin(['CASHS', 'CASHU'])) & (combinedFiles_df_grouped.src_acc.isin(acc_no_list))]
        for_exp_acc_total_grouped = for_exp_acc_total.groupby(by='src_acc', as_index=False).agg({'dest_acc':pd.Series.nunique})
        for_exp_acc_total_grouped.columns = ['acc_no', 'n_exp_acc_total']
        summarized_df = summarized_df.merge(for_exp_acc_total_grouped, on='acc_no',how='left')

        for_exp_acc_within = for_exp_acc_total.loc[for_exp_acc_total.dest_acc.isin(acc_no_list)]
        for_exp_acc_within = for_exp_acc_within.groupby(by='src_acc', as_index=False).agg({'dest_acc':pd.Series.nunique})
        for_exp_acc_within.columns = ['acc_no', 'n_exp_acc_within']
        summarized_df = summarized_df.merge(for_exp_acc_within, on='acc_no',how='left')

        summarized_df.to_csv (summaryDataPath + 'finalSummary.csv', index=False)

        #src_acc = set(combinedFiles_df_grouped.loc[:,'src_acc'])
        #dest_acc = set(combinedFiles_df_grouped.loc[:,'dest_acc'])
        #v = src_acc.union(dest_acc)
        #v_df = pd.DataFrame(data = list(v), columns=['vertex'])
        #v_df.to_csv(tranInfoPath + 'vertices.csv', index=False)

        #e = list(combinedFiles_df_grouped.itertuples(index=False,name=None))
        #combinedFiles_df_grouped.to_csv(tranInfoPath + 'edges.csv', index = False)

        # Print msg
        print('First level data summarization for sanity check is successful! Check out summary file at ', summaryDataPath)

#summarizeData()