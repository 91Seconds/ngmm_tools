#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 12 10:26:06 2021

@author: glavrent
"""
# Working directory and Packages
# ---------------------------
#load packages
import os
import sys
import pathlib
import glob
import re           #regular expression package
import pickle
#arithmetic libraries
import numpy as np
#statistics libraries
import pandas as pd
#plot libraries
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import  AutoLocator as plt_autotick
#user functions
sys.path.insert(0,'../../../Python_lib/regression/')
from pylib_stats import CalcRMS
from pylib_stats import CalcLKDivergece

# Define variables
# ---------------------------

# USER SETS DIRECTORIES AND FILE INFO OF SYNTHETIC DS AND REGRESSION RESULTS
# ++++++++++++++++++++++++++++++++++++++++
#processed dataset
# name_dataset = 'NGAWest2CANorth'
name_dataset  = 'NGAWest2CA'
# name_dataset  = 'NGAWest3CA'

#correlation info
# 1: Small Correlation Lengths
# 2: Large Correlation Lenghts
corr_id = 1

#kernel function 
# 1: Mattern kernel (alpha=2)
# 2: Negative Exp   (alpha=3/2)
ker_id = 1

#mesh type
# 1: Fine Mesh
# 2: Medium Mesh
# 3: Coarse Mesh
mesh_id = 3

#directories (synthetic dataset)
if corr_id == 1:
    dir_syndata = '../../../../Data/Verification/synthetic_datasets/ds2_small_corr_len'  
elif corr_id == 2:
    dir_syndata = '../../../../Data/Verification/synthetic_datasets/ds2_large_corr_len'    
#directories (regression results)    
if mesh_id == 1:
    dir_results = f'../../../../Data/Verification/regression/ds2/INLA_%s_uncorr_cells_fine'%name_dataset
elif mesh_id == 2:
    dir_results = f'../../../../Data/Verification/regression/ds2/INLA_%s_uncorr_cells_medium'%name_dataset
elif mesh_id == 3:
    dir_results = f'../../../../Data/Verification/regression/ds2/INLA_%s_uncorr_cells_coarse'%name_dataset
    
#cell info
fname_cellinfo = dir_syndata + '/' + 'CatalogNGAWest3CALite_cellinfo.csv'
fname_distmat  = dir_syndata + '/' + 'CatalogNGAWest3CALite_distancematrix.csv'

#prefix for synthetic data and results
prfx_syndata  = 'CatalogNGAWest3CALite_synthetic'

#regression results filename prefix
prfx_results  = f'%s_syndata'%name_dataset

# dataset info 
ds_id = np.arange(1,6)
# ++++++++++++++++++++++++++++++++++++++++

# USER NEEDS TO SPECIFY HYPERPARAMETERS OF SYNTHETIC DATASET
# ++++++++++++++++++++++++++++++++++++++++
# hyper-parameters
if corr_id == 1:
    # small correlation lengths
    hyp = {'omega_0': 0.1, 'omega_1e':0.1, 'omega_1as': 0.35, 'omega_1bs': 0.25,
           'ell_1e':60, 'ell_1as':30, 
           'c_cap_erg': -0.011,
           'omega_cap_mu': 0.005, 'omega_ca1p':0.004, 'omega_ca2p':0.002,
           'ell_ca1p': 75,
           'phi_0':0.4, 'tau_0':0.3 }
elif corr_id == 2:
    #large correlation lengths
    hyp = {'omega_0': 0.1, 'omega_1e':0.2, 'omega_1as': 0.4, 'omega_1bs': 0.3,
           'ell_1e':100, 'ell_1as':70, 
           'c_cap_erg': -0.02,
           'omega_cap_mu': 0.008, 'omega_ca1p':0.005, 'omega_ca2p':0.003,
           'ell_ca1p': 120,
           'phi_0':0.4, 'tau_0':0.3}
# ++++++++++++++++++++++++++++++++++++++++

# FILE INFO FOR REGRESSION RESULTS
# ++++++++++++++++++++++++++++++++++++++++
#output filename sufix
if corr_id == 1:
    synds_suffix = '_small_corr_len' 
elif corr_id == 2:
    synds_suffix = '_large_corr_len'
    
#kenel info
if ker_id == 1:
    ker_suffix = '' 
elif ker_id == 2:
    ker_suffix = '_nexp'
# ++++++++++++++++++++++++++++++++++++++++

#ploting options
flag_report = True

# Compare results 
# ---------------------------
#load cell data
df_cellinfo = pd.read_csv(fname_cellinfo).set_index('cellid')
df_distmat  = pd.read_csv(fname_distmat).set_index('rsn')

#initialize misfit metrics dataframe
df_misfit = pd.DataFrame(index=['Y%i'%d_id for d_id in ds_id])

#iterate over different datasets
for d_id in ds_id:
    # Load Data
    #---   ---   ---   ---   ---
    #file names
    #synthetic data
    fname_sdata_gmotion = '%s/%s_%s%s_Y%i'%(dir_syndata, prfx_syndata, 'data',  synds_suffix, d_id)  + '.csv'
    fname_sdata_atten   = '%s/%s_%s%s_Y%i'%(dir_syndata, prfx_syndata, 'atten', synds_suffix, d_id)  + '.csv'
    #regression results
    fname_reg_gmotion = '%s%s/Y%i/%s%s_Y%i_inla_%s'%(dir_results, ker_suffix+synds_suffix, d_id, prfx_results, synds_suffix, d_id, 'residuals')    + '.csv'
    fname_reg_coeff   = '%s%s/Y%i/%s%s_Y%i_inla_%s'%(dir_results, ker_suffix+synds_suffix, d_id, prfx_results, synds_suffix, d_id, 'coefficients') + '.csv'
    fname_reg_atten   = '%s%s/Y%i/%s%s_Y%i_inla_%s'%(dir_results, ker_suffix+synds_suffix, d_id, prfx_results, synds_suffix, d_id, 'catten') + '.csv'
   
    #load synthetic results
    df_sdata_gmotion = pd.read_csv(fname_sdata_gmotion).set_index('rsn')
    df_sdata_atten   = pd.read_csv(fname_sdata_atten).set_index('cellid')
    #load regression results
    df_reg_gmotion = pd.read_csv(fname_reg_gmotion).set_index('rsn')
    df_reg_coeff   = pd.read_csv(fname_reg_coeff).set_index('rsn')
    df_reg_atten   = pd.read_csv(fname_reg_atten).set_index('cellid')
    
    # Processing
    #---   ---   ---   ---   ---
    #keep only relevant columns from synthetic dataset
    df_sdata_gmotion = df_sdata_gmotion.reindex(df_reg_gmotion.index)
    df_sdata_atten   = df_sdata_atten.reindex(df_reg_atten.index)
    #distance matrix for records of interest
    df_dmat = df_distmat.reindex(df_sdata_gmotion.index)
    
    #find unique earthqakes and stations
    eq_id,  eq_idx, eq_nrec   = np.unique(df_sdata_gmotion.eqid, return_index=True, return_counts=True)
    sta_id, sta_idx, sta_nrec = np.unique(df_sdata_gmotion.ssn,  return_index=True, return_counts=True)
    
    #number of paths per cell
    cell_npath = np.sum(df_dmat.loc[:,df_reg_atten.cellname] > 0, axis=0)
    
    # Compute Root Mean Square Error
    #---   ---   ---   ---   ---    
    df_misfit.loc['Y%i'%d_id,'nerg_tot_rms'] = CalcRMS(df_sdata_gmotion.nerg_gm.values,            df_reg_gmotion.nerg_mu.values)
    df_misfit.loc['Y%i'%d_id,'dc_1e_rms']    = CalcRMS(df_sdata_gmotion['dc_1e'].values[eq_idx],   df_reg_coeff['dc_1e_mean'].values[eq_idx])
    df_misfit.loc['Y%i'%d_id,'dc_1as_rms']   = CalcRMS(df_sdata_gmotion['dc_1as'].values[sta_idx], df_reg_coeff['dc_1as_mean'].values[sta_idx])
    df_misfit.loc['Y%i'%d_id,'dc_1bs_rms']   = CalcRMS(df_sdata_gmotion['dc_1bs'].values[sta_idx], df_reg_coeff['dc_1bs_mean'].values[sta_idx])
    df_misfit.loc['Y%i'%d_id,'c_cap_rms']    = CalcRMS(df_sdata_atten['c_cap'].values,             df_reg_atten['c_cap_mean'].values)
    
    # Compute Divergence
    #---   ---   ---   ---   ---
    df_misfit.loc['Y%i'%d_id,'nerg_tot_KL'] = CalcLKDivergece(df_sdata_gmotion.nerg_gm.values,            df_reg_gmotion.nerg_mu.values)
    df_misfit.loc['Y%i'%d_id,'dc_1e_KL']    = CalcLKDivergece(df_sdata_gmotion['dc_1e'].values[eq_idx],   df_reg_coeff['dc_1e_mean'].values[eq_idx])
    df_misfit.loc['Y%i'%d_id,'dc_1as_KL']   = CalcLKDivergece(df_sdata_gmotion['dc_1as'].values[sta_idx], df_reg_coeff['dc_1as_mean'].values[sta_idx])
    df_misfit.loc['Y%i'%d_id,'dc_1bs_KL']   = CalcLKDivergece(df_sdata_gmotion['dc_1bs'].values[sta_idx], df_reg_coeff['dc_1bs_mean'].values[sta_idx])
    df_misfit.loc['Y%i'%d_id,'c_cap_KL']    = CalcLKDivergece(df_sdata_atten['c_cap'].values,             df_reg_atten['c_cap_mean'].values)
    
    # Output
    #---   ---   ---   ---   ---
    #figure directory
    dir_fig = '%s%s/Y%i/figures_cmp/'%(dir_results, ker_suffix+synds_suffix, d_id)
    pathlib.Path(dir_fig).mkdir(parents=True, exist_ok=True)
        
    #compare ground motion predictions
    #...   ...   ...   ...   ...   ...
    #figure title
    fname_fig = 'Y%i_scatter_tot_res'%d_id
    #create figure
    fig, ax = plt.subplots(figsize = (10,10))
    #median 
    ax.scatter(df_sdata_gmotion.nerg_gm.values, df_reg_gmotion.nerg_mu.values)
    ax.axline((0,0), slope=1, color="black", linestyle="--")
    #edit figure
    if not flag_report: ax.set_title('Comparison total residuals, Y: %i'%d_id, fontsize=30)
    ax.set_xlabel('Synthetic dataset', fontsize=35)
    ax.set_ylabel('Estimated',         fontsize=35)
    ax.grid(which='both')
    ax.tick_params(axis='x', labelsize=32)
    ax.tick_params(axis='y', labelsize=32)
    #plot limits
    # plt_lim = np.array([ax.get_xlim(), ax.get_ylim()])
    # plt_lim = (plt_lim[:,0].min(), plt_lim[:,1].max())
    # ax.set_xlim(plt_lim)
    # ax.set_ylim(plt_lim)
    ax.set_xlim([-10,2])
    ax.set_ylim([-10,2])
    fig.tight_layout()
    fig.savefig( dir_fig + fname_fig + '.png' )
 
    #compare dc_1e
    #...   ...   ...   ...   ...   ...
    #figure title
    fname_fig = 'Y%i_dc_1e_scatter'%d_id
    #create figure
    fig, ax = plt.subplots(figsize = (10,10))
    #coefficient scatter
    ax.scatter(df_sdata_gmotion['dc_1e'].values[eq_idx], df_reg_coeff['dc_1e_mean'].values[eq_idx])
    ax.axline((0,0), slope=1, color="black", linestyle="--")
    #edit figure
    if not flag_report: ax.set_title(r'Comparison $\delta c_{1,e}$, Y: %i'%d_id, fontsize=30)
    ax.set_xlabel('Synthetic dataset', fontsize=25)
    ax.set_ylabel('Estimated',         fontsize=25)
    ax.grid(which='both')
    ax.tick_params(axis='x', labelsize=22)
    ax.tick_params(axis='y', labelsize=22)
    #plot limits
    # plt_lim = np.array([ax.get_xlim(), ax.get_ylim()])
    # plt_lim = (plt_lim[:,0].min(), plt_lim[:,1].max())
    # ax.set_xlim(plt_lim)
    # ax.set_ylim(plt_lim)
    ax.set_xlim([-2,2])
    ax.set_ylim([-2,2])
    #save figure
    fig.tight_layout()
    fig.savefig( dir_fig + fname_fig + '.png' )
    
    #figure title
    fname_fig = 'Y%i_dc_1e_accuracy'%d_id
    #create figure
    fig, ax = plt.subplots(figsize = (10,10))
    #coefficient scatter
    ax.scatter(df_reg_coeff['dc_1e_sig'].values[eq_idx],
               df_sdata_gmotion['dc_1e'].values[eq_idx] - df_reg_coeff['dc_1e_mean'].values[eq_idx])
    ax.axline((0,0), slope=0, color="black", linestyle="--")
    #edit figure
    if not flag_report: ax.set_title(r'Comparison $\delta c_{1,e}$, Y: %i'%d_id, fontsize=30)
    ax.set_xlabel('Standard Deviation', fontsize=25)
    ax.set_ylabel('Actual - Estimated', fontsize=25)
    ax.grid(which='both')
    ax.tick_params(axis='x', labelsize=22)
    ax.tick_params(axis='y', labelsize=22)
    #plot limits
    # ax.set_ylim(np.abs(ax.get_ylim()).max()*np.array([-1,1]))
    ax.set_xlim([0,.5])
    ax.set_ylim([-2,2])
    #save figure
    fig.tight_layout()
    fig.savefig( dir_fig + fname_fig + '.png' )
    
    #figure title
    fname_fig = 'Y%i_dc_1e_nrec'%d_id
    #create figure
    fig, ax = plt.subplots(figsize = (10,10))
    #coefficient scatter
    ax.scatter(eq_nrec,
               df_sdata_gmotion['dc_1e'].values[eq_idx] - df_reg_coeff['dc_1e_mean'].values[eq_idx])
    ax.axline((0,0), slope=0, color="black", linestyle="--")
    #edit figure
    if not flag_report: ax.set_title(r'Comparison $\delta c_{1,e}$, Y: %i'%d_id, fontsize=30)
    ax.set_xlabel('Number of records',  fontsize=25)
    ax.set_ylabel('Actual - Estimated', fontsize=25)
    ax.grid(which='both')
    ax.set_xscale('log')
    ax.tick_params(axis='x', labelsize=22)
    ax.tick_params(axis='y', labelsize=22)
    #plot limits
    # ax.set_ylim(np.abs(ax.get_ylim()).max()*np.array([-1,1]))
    ax.set_xlim([0.9,1e3])
    ax.set_ylim([-2,2])
    #save figure
    fig.tight_layout()
    fig.savefig( dir_fig + fname_fig + '.png' )

    #compare dc_1as
    #...   ...   ...   ...   ...   ...
    #figure title
    fname_fig = 'Y%i_dc_1as_scatter'%d_id
    #create figure
    fig, ax = plt.subplots(figsize = (10,10))
    #coefficient scatter
    ax.scatter(df_sdata_gmotion['dc_1as'].values[sta_idx], df_reg_coeff['dc_1as_mean'].values[sta_idx])
    ax.axline((0,0), slope=1, color="black", linestyle="--")
    #edit figure
    if not flag_report: ax.set_title(r'Comparison $\delta c_{1a,s}$, Y: %i'%d_id, fontsize=30)
    ax.set_xlabel('Synthetic dataset', fontsize=25)
    ax.set_ylabel('Estimated',         fontsize=25)
    ax.grid(which='both')
    ax.tick_params(axis='x', labelsize=22)
    ax.tick_params(axis='y', labelsize=22)
    #plot limits
    # plt_lim = np.array([ax.get_xlim(), ax.get_ylim()])
    # plt_lim = (plt_lim[:,0].min(), plt_lim[:,1].max())
    # ax.set_xlim(plt_lim)
    # ax.set_ylim(plt_lim)
    ax.set_xlim([-2,2])
    ax.set_ylim([-2,2])
    #save figure
    fig.tight_layout()
    fig.savefig( dir_fig + fname_fig + '.png' )

    #figure title
    fname_fig = 'Y%i_dc_1as_accuracy'%d_id
    #create figure
    fig, ax = plt.subplots(figsize = (10,10))
    #accuray
    ax.scatter(df_reg_coeff['dc_1as_sig'].values[sta_idx],
               df_sdata_gmotion['dc_1as'].values[sta_idx] - df_reg_coeff['dc_1as_mean'].values[sta_idx])
    ax.axline((0,0), slope=0, color="black", linestyle="--")
    #edit figure
    if not flag_report: ax.set_title(r'Comparison $\delta c_{1a,s}$, Y: %i'%d_id, fontsize=30)
    ax.set_xlabel('Standard Deviation', fontsize=25)
    ax.set_ylabel('Actual - Estimated', fontsize=25)
    ax.grid(which='both')
    ax.tick_params(axis='x', labelsize=22)
    ax.tick_params(axis='y', labelsize=22)
    #plot limits
    # ax.set_ylim(np.abs(ax.get_ylim()).max()*np.array([-1,1]))
    ax.set_xlim([0,.5])
    ax.set_ylim([-2,2])
    #save figure
    fig.tight_layout()
    fig.savefig( dir_fig + fname_fig + '.png' )

    #figure title
    fname_fig = 'Y%i_dc_1as_nrec'%d_id
    #create figure
    fig, ax = plt.subplots(figsize = (10,10))
    #accuray
    ax.scatter(sta_nrec,
               df_sdata_gmotion['dc_1as'].values[sta_idx] - df_reg_coeff['dc_1as_mean'].values[sta_idx])
    ax.axline((0,0), slope=0, color="black", linestyle="--")
    #edit figure
    if not flag_report: ax.set_title(r'Comparison $\delta c_{1a,s}$, Y: %i'%d_id, fontsize=30)
    ax.set_xlabel('Number of records',  fontsize=25)
    ax.set_ylabel('Actual - Estimated', fontsize=25)
    ax.grid(which='both')
    ax.set_xscale('log')
    ax.tick_params(axis='x', labelsize=22)
    ax.tick_params(axis='y', labelsize=22)
    #plot limits
    # ax.set_ylim(np.abs(ax.get_ylim()).max()*np.array([-1,1]))
    ax.set_xlim([.9,1000])
    ax.set_ylim([-2,2])
    #save figure
    fig.tight_layout()
    fig.savefig( dir_fig + fname_fig + '.png' )

    #compare dc_1bs
    #...   ...   ...   ...   ...   ...
    #figure title
    fname_fig = 'Y%i_dc_1bs_scatter'%d_id
    #create figure
    fig, ax = plt.subplots(figsize = (10,10))
    #coefficient scatter
    ax.scatter(df_sdata_gmotion['dc_1bs'].values[sta_idx], df_reg_coeff['dc_1bs_mean'].values[sta_idx])
    ax.axline((0,0), slope=1, color="black", linestyle="--")
    #edit figure
    if not flag_report: ax.set_title(r'Comparison $\delta c_{1b,s}$, Y: %i'%d_id, fontsize=30)
    ax.set_xlabel('Synthetic dataset', fontsize=25)
    ax.set_ylabel('Estimated',         fontsize=25)
    ax.grid(which='both')
    ax.tick_params(axis='x', labelsize=22)
    ax.tick_params(axis='y', labelsize=22)
    #plot limits
    # plt_lim = np.array([ax.get_xlim(), ax.get_ylim()])
    # plt_lim = (plt_lim[:,0].min(), plt_lim[:,1].max())
    # ax.set_xlim(plt_lim)
    # ax.set_ylim(plt_lim)
    ax.set_xlim([-2,2])
    ax.set_ylim([-2,2])
    #save figure
    fig.tight_layout()
    fig.savefig( dir_fig + fname_fig + '.png' )
    
    #figure title
    fname_fig = 'Y%i_dc_1bs_accuracy'%d_id
    #create figure
    fig, ax = plt.subplots(figsize = (10,10))
    #accuray
    ax.scatter(df_reg_coeff['dc_1bs_sig'].values[sta_idx],
               df_sdata_gmotion['dc_1bs'].values[sta_idx] - df_reg_coeff['dc_1bs_mean'].values[sta_idx])
    ax.axline((0,0), slope=0, color="black", linestyle="--")
    #edit figure
    if not flag_report: ax.set_title(r'Comparison $\delta c_{1b,s}$, Y: %i'%d_id, fontsize=30)
    ax.set_xlabel('Standard Deviation', fontsize=25)
    ax.set_ylabel('Actual - Estimated', fontsize=25)
    ax.grid(which='both')
    ax.tick_params(axis='x', labelsize=22)
    ax.tick_params(axis='y', labelsize=22)
    #plot limits
    # ax.set_ylim(np.abs(ax.get_ylim()).max()*np.array([-1,1]))
    ax.set_xlim([0,.4])
    ax.set_ylim([-2,2])
    #save figure
    fig.tight_layout()
    fig.savefig( dir_fig + fname_fig + '.png' )
    
    #figure title
    fname_fig = 'Y%i_dc_1bs_nrec'%d_id
    #create figure
    fig, ax = plt.subplots(figsize = (10,10))
    #accuray
    ax.scatter(sta_nrec,
               df_sdata_gmotion['dc_1bs'].values[sta_idx] - df_reg_coeff['dc_1bs_mean'].values[sta_idx])
    ax.axline((0,0), slope=0, color="black", linestyle="--")
    #edit figure
    if not flag_report: ax.set_title(r'Comparison $\delta c_{1b,s}$, Y: %i'%d_id, fontsize=30)
    ax.set_xlabel('Number of records',  fontsize=25)
    ax.set_ylabel('Actual - Estimated', fontsize=25)
    ax.grid(which='both')
    ax.set_xscale('log')
    ax.tick_params(axis='x', labelsize=22)
    ax.tick_params(axis='y', labelsize=22)
    #plot limits
    # ax.set_ylim(np.abs(ax.get_ylim()).max()*np.array([-1,1]))
    ax.set_xlim([.9,1000])
    ax.set_ylim([-2,2])
    #save figure
    fig.tight_layout()
    fig.savefig( dir_fig + fname_fig + '.png' )

    #compare c_cap
    #...   ...   ...   ...   ...   ...
    #figure title
    fname_fig = 'Y%i_c_cap_scatter'%d_id
    #create figure
    fig, ax = plt.subplots(figsize = (10,10))
    #coefficient scatter
    ax.scatter(df_sdata_atten['c_cap'].values, df_reg_atten['c_cap_mean'].values)
    ax.axline((0,0), slope=1, color="black", linestyle="--")
    #edit figure
    if not flag_report: ax.set_title(r'Comparison $c_{ca,p}$, Y: %i'%d_id, fontsize=30)
    ax.set_xlabel('Synthetic dataset', fontsize=25)
    ax.set_ylabel('Estimated',         fontsize=25)
    ax.grid(which='both')
    ax.tick_params(axis='x', labelsize=22)
    ax.tick_params(axis='y', labelsize=22)
    #plot limits
    # plt_lim = np.array([ax.get_xlim(), ax.get_ylim()])
    # plt_lim = (plt_lim[:,0].min(), plt_lim[:,1].max())
    # ax.set_xlim(plt_lim)
    # ax.set_ylim(plt_lim)
    ax.set_xlim([-0.05,0.02])
    ax.set_ylim([-0.05,0.02])
    #save figure
    fig.tight_layout()
    fig.savefig( dir_fig + fname_fig + '.png' )
    
    #figure title
    fname_fig = 'Y%i_c_cap_accuracy'%d_id
    #create figure
    fig, ax = plt.subplots(figsize = (10,10))
    #coefficient scatter
    ax.scatter(df_reg_atten['c_cap_sig'],
               df_sdata_atten['c_cap'].values - df_reg_atten['c_cap_mean'].values)
    ax.axline((0,0), slope=0, color="black", linestyle="--")
    #edit figure
    if not flag_report: ax.set_title(r'Comparison $c_{ca,p}$, Y: %i'%d_id, fontsize=30)
    ax.set_xlabel('Standard Deviation', fontsize=25)
    ax.set_ylabel('Actual - Estimated', fontsize=25)
    ax.grid(which='both')
    ax.tick_params(axis='x', labelsize=22)
    ax.tick_params(axis='y', labelsize=22)
    #plot limits
    # ax.set_ylim(np.abs(ax.get_ylim()).max()*np.array([-1,1]))
    ax.set_xlim([0.00,0.03])
    ax.set_ylim([-0.04,0.04])
    #save figure
    fig.tight_layout()
    fig.savefig( dir_fig + fname_fig + '.png' )
    
    #figure title
    fname_fig = 'Y%i_c_cap_npath'%d_id
    #create figure
    fig, ax = plt.subplots(figsize = (10,10))
    #coefficient scatter
    ax.scatter(cell_npath,
               df_sdata_atten['c_cap'].values - df_reg_atten['c_cap_mean'].values)
    ax.axline((0,0), slope=0, color="black", linestyle="--")
    #edit figure
    if not flag_report: ax.set_title(r'Comparison $c_{ca,p}$, Y: %i'%d_id, fontsize=30)
    ax.set_xlabel('Number of paths',    fontsize=25)
    ax.set_ylabel('Actual - Estimated', fontsize=25)
    ax.grid(which='both')
    ax.set_xscale('log')
    ax.tick_params(axis='x', labelsize=22)
    ax.tick_params(axis='y', labelsize=22)
    #plot limits
    # ax.set_ylim(np.abs(ax.get_ylim()).max()*np.array([-1,1]))
    ax.set_xlim([.9,5e4])
    ax.set_ylim([-0.04,0.04])
    #save figure
    fig.tight_layout()
    fig.savefig( dir_fig + fname_fig + '.png' )


# Compare Misfit Metrics
# ---------------------------
#summary directory
dir_sum = '%s%s/summary/'%(dir_results,ker_suffix+synds_suffix)
pathlib.Path(dir_fig).mkdir(parents=True, exist_ok=True)
#figure directory
dir_fig = '%s/figures/'%(dir_sum)
pathlib.Path(dir_fig).mkdir(parents=True, exist_ok=True)

#save 
df_misfit.to_csv(dir_sum + 'misfit_summary.csv')

#RMS misfit
fname_fig = 'misfit_score' 
#plot KL divergence
fig, ax = plt.subplots(figsize = (10,10))   
ax.plot(ds_id, df_misfit.nerg_tot_rms, linestyle='-', marker='o', linewidth=2, markersize=10, label= 'tot nerg')
ax.plot(ds_id, df_misfit.dc_1e_rms,    linestyle='-', marker='o', linewidth=2, markersize=10, label=r'$\delta c_{1,e}$')
ax.plot(ds_id, df_misfit.dc_1as_rms,   linestyle='-', marker='o', linewidth=2, markersize=10, label=r'$\delta c_{1a,s}$')
ax.plot(ds_id, df_misfit.dc_1bs_rms,   linestyle='-', marker='o', linewidth=2, markersize=10, label=r'$\delta c_{1b,s}$')
ax.plot(ds_id, df_misfit.c_cap_rms,    linestyle='-', marker='o', linewidth=2, markersize=10, label=r'$c_{ca,p}$')
#figure properties
ax.set_ylim([0,0.50])
ax.set_xlabel('synthetic dataset', fontsize=25)
ax.set_ylabel('RSME', fontsize=25)
ax.grid(which='both')
ax.set_xticks(ds_id)
ax.set_xticklabels(labels=df_misfit.index)
ax.tick_params(axis='x', labelsize=22)
ax.tick_params(axis='y', labelsize=22)
#legend
ax.legend(loc='upper left', fontsize=25)
#save figure
fig.tight_layout()
fig.savefig( dir_fig + fname_fig + '.png' )

#KL divergence
fname_fig = 'KLdiv_score' 
#plot KL divergence
fig, ax = plt.subplots(figsize = (10,10))   
ax.plot(ds_id, df_misfit.nerg_tot_KL, linestyle='-', marker='o', linewidth=2, markersize=10, label= 'tot nerg')
ax.plot(ds_id, df_misfit.dc_1e_KL,    linestyle='-', marker='o', linewidth=2, markersize=10, label=r'$\delta c_{1,e}$')
ax.plot(ds_id, df_misfit.dc_1as_KL,   linestyle='-', marker='o', linewidth=2, markersize=10, label=r'$\delta c_{1a,s}$')
ax.plot(ds_id, df_misfit.dc_1bs_KL,   linestyle='-', marker='o', linewidth=2, markersize=10, label=r'$\delta c_{1b,s}$')
ax.plot(ds_id, df_misfit.c_cap_KL,    linestyle='-', marker='o', linewidth=2, markersize=10, label=r'$c_{ca,p}$')
#figure properties
ax.set_ylim([0,0.50])
ax.set_xlabel('synthetic dataset', fontsize=25)
ax.set_ylabel('KL divergence', fontsize=25)
ax.grid(which='both')
ax.set_xticks(ds_id)
ax.set_xticklabels(labels=df_misfit.index)
ax.tick_params(axis='x', labelsize=22)
ax.tick_params(axis='y', labelsize=22)
#legend
ax.legend(loc='upper left', fontsize=25)
#save figure
fig.tight_layout()
fig.savefig( dir_fig + fname_fig + '.png' )


# Compare hyper-paramters 
# ---------------------------
#iterate over different datasets
df_reg_hyp = list()
df_reg_hyp_post = list()
for d_id in ds_id:
    # Load Data
    #---   ---   ---   ---   ---
    #regression hyperparamters results
    fname_reg_hyp      = '%s%s/Y%i/%s%s_Y%i_inla_%s'%(dir_results, ker_suffix+synds_suffix, d_id,prfx_results, synds_suffix, d_id, 'hyperparameters') + '.csv'
    fname_reg_hyp_post = '%s%s/Y%i/%s%s_Y%i_inla_%s'%(dir_results, ker_suffix+synds_suffix, d_id,prfx_results, synds_suffix, d_id, 'hyperposterior')  + '.csv'  
    #load regression results
    df_reg_hyp.append( pd.read_csv(fname_reg_hyp, index_col=0) )
    df_reg_hyp_post.append( pd.read_csv(fname_reg_hyp_post, index_col=0) )

# Omega_1e    
#---   ---   ---   ---   ---
#hyper-paramter name
name_hyp = 'omega_1e'
#figure title
fname_fig = 'post_dist_' + name_hyp
#create figure
fig, ax = plt.subplots(figsize = (10,10))   
for d_id, df_r_h, df_r_h_p in zip(ds_id, df_reg_hyp, df_reg_hyp_post):
    #estimate vertical line height for mean and mode
    ymax_mode = df_r_h_p.loc[:,name_hyp+'_pdf'].max()
    # ymax_mean = 1.5*np.ceil(ymax_mode/10)*10
    ymax_mean = 40
    #plot posterior dist
    pl_pdf = ax.plot(df_r_h_p.loc[:,name_hyp], df_r_h_p.loc[:,name_hyp+'_pdf'])
    ax.vlines(df_r_h.loc[name_hyp,'mean'], ymin=0, ymax=ymax_mean, linestyle='-',  color=pl_pdf[0].get_color(), label='Mean')
    ax.vlines(df_r_h.loc[name_hyp,'mode'], ymin=0, ymax=ymax_mode, linestyle='--', color=pl_pdf[0].get_color(), label='Mode')
#plot true value
ymax_hyp = ymax_mean
ax.vlines(hyp[name_hyp], ymin=0, ymax=ymax_hyp, linestyle='-', linewidth=4, color='black', label='True value')
#edit figure
if not flag_report: ax.set_title(r'Comparison $\omega_{1,e}$',     fontsize=30)
ax.set_xlabel('$\omega_{1,e}$',                fontsize=25)
ax.set_ylabel('probability density function ', fontsize=25)
ax.grid(which='both')
ax.tick_params(axis='x', labelsize=22)
ax.tick_params(axis='y', labelsize=22)
#plot limits
ax.set_xlim([0,0.25])
ax.set_ylim([0,ymax_hyp])
#save figure
fig.tight_layout()
fig.savefig( dir_fig + fname_fig + '.png' )

# Omega_1as
#---   ---   ---   ---   ---
#hyper-paramter name
name_hyp = 'omega_1as'
#figure title
fname_fig = 'post_dist_' + name_hyp
#create figure
fig, ax = plt.subplots(figsize = (10,10))   
for d_id, df_r_h, df_r_h_p in zip(ds_id, df_reg_hyp, df_reg_hyp_post):
    #estimate vertical line height for mean and mode
    ymax_mode = df_r_h_p.loc[:,name_hyp+'_pdf'].max()
    # ymax_mean = 1.5*np.ceil(ymax_mode/10)*10
    ymax_mean = 30
    #plot posterior dist
    pl_pdf = ax.plot(df_r_h_p.loc[:,name_hyp], df_r_h_p.loc[:,name_hyp+'_pdf'])
    ax.vlines(df_r_h.loc[name_hyp,'mean'], ymin=0, ymax=ymax_mean, linestyle='-',  color=pl_pdf[0].get_color(), label='Mean')
    ax.vlines(df_r_h.loc[name_hyp,'mode'], ymin=0, ymax=ymax_mode, linestyle='--', color=pl_pdf[0].get_color(), label='Mode')
#plot true value
ymax_hyp = ymax_mean
ax.vlines(hyp[name_hyp], ymin=0, ymax=ymax_hyp, linestyle='-', linewidth=4, color='black', label='True value')
#edit figure
if not flag_report: ax.set_title(r'Comparison $\omega_{1a,s}$',     fontsize=30)
ax.set_xlabel('$\omega_{1a,s}$',                fontsize=25)
ax.set_ylabel('probability density function ', fontsize=25)
ax.grid(which='both')
ax.tick_params(axis='x', labelsize=22)
ax.tick_params(axis='y', labelsize=22)
#plot limits
ax.set_xlim([0,0.5])
ax.set_ylim([0,ymax_hyp])
#save figure
fig.tight_layout()
fig.savefig( dir_fig + fname_fig + '.png' )

# Omega_1bs
#---   ---   ---   ---   ---
#hyper-paramter name
name_hyp = 'omega_1bs'
#figure title
fname_fig = 'post_dist_' + name_hyp
#create figure
fig, ax = plt.subplots(figsize = (10,10))   
for d_id, df_r_h, df_r_h_p in zip(ds_id, df_reg_hyp, df_reg_hyp_post):
    #estimate vertical line height for mean and mode
    ymax_mode = df_r_h_p.loc[:,name_hyp+'_pdf'].max()
    # ymax_mean = 1.5*np.ceil(ymax_mode/10)*10
    ymax_mean = 60
    #plot posterior dist
    pl_pdf = ax.plot(df_r_h_p.loc[:,name_hyp], df_r_h_p.loc[:,name_hyp+'_pdf'])
    ax.vlines(df_r_h.loc[name_hyp,'mean'], ymin=0, ymax=ymax_mean, linestyle='-',  color=pl_pdf[0].get_color(), label='Mean')
    ax.vlines(df_r_h.loc[name_hyp,'mode'], ymin=0, ymax=ymax_mode, linestyle='--', color=pl_pdf[0].get_color(), label='Mode')
#plot true value
ymax_hyp = ymax_mean
ax.vlines(hyp[name_hyp], ymin=0, ymax=ymax_hyp, linestyle='-', linewidth=4, color='black', label='True value')
#edit figure
if not flag_report: ax.set_title(r'Comparison $\omega_{1b,s}$',     fontsize=30)
ax.set_xlabel('$\omega_{1b,s}$',                fontsize=25)
ax.set_ylabel('probability density function ', fontsize=25)
ax.grid(which='both')
ax.tick_params(axis='x', labelsize=22)
ax.tick_params(axis='y', labelsize=22)
#plot limits
ax.set_xlim([0,0.5])
ax.set_ylim([0,ymax_hyp])
#save figure
fig.tight_layout()
fig.savefig( dir_fig + fname_fig + '.png' )

# Ell_1e    
#---   ---   ---   ---   ---
#hyper-paramter name
name_hyp = 'ell_1e'
#figure title
fname_fig = 'post_dist_' + name_hyp
#create figure
fig, ax = plt.subplots(figsize = (10,10))   
for d_id, df_r_h, df_r_h_p in zip(ds_id, df_reg_hyp, df_reg_hyp_post):
    #estimate vertical line height for mean and mode
    ymax_mode = df_r_h_p.loc[:,name_hyp+'_pdf'].max()
    # ymax_mean = 1.5*np.ceil(ymax_mode/10)*10
    ymax_mean = 0.02
    #plot posterior dist
    pl_pdf = ax.plot(df_r_h_p.loc[:,name_hyp], df_r_h_p.loc[:,name_hyp+'_pdf'])
    ax.vlines(df_r_h.loc[name_hyp,'mean'], ymin=0, ymax=ymax_mean, linestyle='-',  color=pl_pdf[0].get_color(), label='Mean')
    ax.vlines(df_r_h.loc[name_hyp,'mode'], ymin=0, ymax=ymax_mode, linestyle='--', color=pl_pdf[0].get_color(), label='Mode')
#plot true value
ymax_hyp = ymax_mean
ax.vlines(hyp[name_hyp], ymin=0, ymax=ymax_hyp, linestyle='-', linewidth=4, color='black', label='True value')
#edit figure
if not flag_report: ax.set_title(r'Comparison $\ell_{1,e}$',     fontsize=30)
ax.set_xlabel('$\ell_{1,e}$',                fontsize=25)
ax.set_ylabel('probability density function ', fontsize=25)
ax.grid(which='both')
ax.tick_params(axis='x', labelsize=22)
ax.tick_params(axis='y', labelsize=22)
#plot limits
ax.set_xlim([0,500])
ax.set_ylim([0,ymax_hyp])
#save figure
fig.tight_layout()
fig.savefig( dir_fig + fname_fig + '.png' )

# Ell_1as
#---   ---   ---   ---   ---
#hyper-paramter name
name_hyp = 'ell_1as'
#figure title
fname_fig = 'post_dist_' + name_hyp
#create figure
fig, ax = plt.subplots(figsize = (10,10))   
for d_id, df_r_h, df_r_h_p in zip(ds_id, df_reg_hyp, df_reg_hyp_post):
    #estimate vertical line height for mean and mode
    ymax_mode = df_r_h_p.loc[:,name_hyp+'_pdf'].max()
    # ymax_mean = 1.5*np.ceil(ymax_mode/10)*10
    ymax_mean = 0.1
    #plot posterior dist
    pl_pdf = ax.plot(df_r_h_p.loc[:,name_hyp], df_r_h_p.loc[:,name_hyp+'_pdf'])
    ax.vlines(df_r_h.loc[name_hyp,'mean'], ymin=0, ymax=ymax_mean, linestyle='-',  color=pl_pdf[0].get_color(), label='Mean')
    ax.vlines(df_r_h.loc[name_hyp,'mode'], ymin=0, ymax=ymax_mode, linestyle='--', color=pl_pdf[0].get_color(), label='Mode')
#plot true value
ymax_hyp = ymax_mean
ax.vlines(hyp[name_hyp], ymin=0, ymax=ymax_hyp, linestyle='-', linewidth=4, color='black', label='True value')
#edit figure
if not flag_report: ax.set_title(r'Comparison $\ell_{1a,s}$',     fontsize=30)
ax.set_xlabel('$\ell_{1a,s}$',                fontsize=25)
ax.set_ylabel('probability density function ', fontsize=25)
ax.grid(which='both')
ax.tick_params(axis='x', labelsize=22)
ax.tick_params(axis='y', labelsize=22)
#plot limits
ax.set_xlim([0,150])
ax.set_ylim([0,ymax_hyp])
#save figure
fig.tight_layout()
fig.savefig( dir_fig + fname_fig + '.png' )

# Tau_0
#---   ---   ---   ---   ---
#hyper-paramter name
name_hyp = 'tau_0'
#figure title
fname_fig = 'post_dist_' + name_hyp
#create figure
fig, ax = plt.subplots(figsize = (10,10))   
for d_id, df_r_h, df_r_h_p in zip(ds_id, df_reg_hyp, df_reg_hyp_post):
    #estimate vertical line height for mean and mode
    ymax_mode = df_r_h_p.loc[:,name_hyp+'_pdf'].max()
    # ymax_mean = 1.5*np.ceil(ymax_mode/10)*10
    ymax_mean = 60
    #plot posterior dist
    pl_pdf = ax.plot(df_r_h_p.loc[:,name_hyp], df_r_h_p.loc[:,name_hyp+'_pdf'])
    ax.vlines(df_r_h.loc[name_hyp,'mean'], ymin=0, ymax=ymax_mean, linestyle='-',  color=pl_pdf[0].get_color(), label='Mean')
    ax.vlines(df_r_h.loc[name_hyp,'mode'], ymin=0, ymax=ymax_mode, linestyle='--', color=pl_pdf[0].get_color(), label='Mode')
#plot true value
ymax_hyp = ymax_mean
ax.vlines(hyp[name_hyp], ymin=0, ymax=ymax_hyp, linestyle='-', linewidth=4, color='black', label='True value')
#edit figure
if not flag_report: ax.set_title(r'Comparison $\tau_{0}$',     fontsize=30)
ax.set_xlabel(r'$\tau_{0}$',                fontsize=25)
ax.set_ylabel(r'probability density function ', fontsize=25)
ax.grid(which='both')
ax.tick_params(axis='x', labelsize=22)
ax.tick_params(axis='y', labelsize=22)
#plot limits
ax.set_xlim([0,0.5])
ax.set_ylim([0,ymax_hyp])
#save figure
fig.tight_layout()
fig.savefig( dir_fig + fname_fig + '.png' )


# Phi_0
#---   ---   ---   ---   ---
#hyper-paramter name
name_hyp = 'phi_0'
#figure title
fname_fig = 'post_dist_' + name_hyp
#create figure
fig, ax = plt.subplots(figsize = (10,10))   
for d_id, df_r_h, df_r_h_p in zip(ds_id, df_reg_hyp, df_reg_hyp_post):
    #estimate vertical line height for mean and mode
    ymax_mode = df_r_h_p.loc[:,name_hyp+'_pdf'].max()
    # ymax_mean = 1.5*np.ceil(ymax_mode/10)*10
    ymax_mean = 100
    #plot posterior dist
    pl_pdf = ax.plot(df_r_h_p.loc[:,name_hyp], df_r_h_p.loc[:,name_hyp+'_pdf'])
    ax.vlines(df_r_h.loc[name_hyp,'mean'], ymin=0, ymax=ymax_mean, linestyle='-',  color=pl_pdf[0].get_color(), label='Mean')
    ax.vlines(df_r_h.loc[name_hyp,'mode'], ymin=0, ymax=ymax_mode, linestyle='--', color=pl_pdf[0].get_color(), label='Mode')
#plot true value
ymax_hyp = ymax_mean
ax.vlines(hyp[name_hyp], ymin=0, ymax=ymax_hyp, linestyle='-', linewidth=4, color='black', label='True value')
#edit figure
if not flag_report: ax.set_title(r'Comparison $\phi_{0}$',     fontsize=30)
ax.set_xlabel('$\phi_{0}$',                fontsize=25)
ax.set_ylabel(r'probability density function ', fontsize=25)
ax.grid(which='both')
ax.tick_params(axis='x', labelsize=22)
ax.tick_params(axis='y', labelsize=22)
#plot limits
ax.set_xlim([0,0.6])
ax.set_ylim([0,ymax_hyp])
#save figure
fig.tight_layout()
fig.savefig( dir_fig + fname_fig + '.png' )

# Omega_ca
#---   ---   ---   ---   ---
#hyper-paramter name
name_hyp = 'omega_cap'
#figure title
fname_fig = 'post_dist_' + name_hyp
#create figure
fig, ax = plt.subplots(figsize = (10,10))   
for d_id, df_r_h, df_r_h_p in zip(ds_id, df_reg_hyp, df_reg_hyp_post):
    #estimate vertical line height for mean and mode
    ymax_mode = df_r_h_p.loc[:,name_hyp+'_pdf'].max()
    # ymax_mean = 1.5*np.ceil(ymax_mode/10)*10
    ymax_mean = 1500
    #plot posterior dist
    pl_pdf = ax.plot(df_r_h_p.loc[:,name_hyp], df_r_h_p.loc[:,name_hyp+'_pdf'])
    ax.vlines(df_r_h.loc[name_hyp,'mean'], ymin=0, ymax=ymax_mean, linestyle='-',  color=pl_pdf[0].get_color(), label='Mean')
    ax.vlines(df_r_h.loc[name_hyp,'mode'], ymin=0, ymax=ymax_mode, linestyle='--', color=pl_pdf[0].get_color(), label='Mode')
#plot true value
ymax_hyp = ymax_mean
ax.vlines(np.sqrt(hyp['omega_ca1p']**2+hyp['omega_ca2p']**2), ymin=0, ymax=ymax_hyp, linestyle='-', linewidth=4, color='black', label='True value')
#edit figure
if not flag_report: ax.set_title(r'Comparison $\omega_{ca,p}$',     fontsize=30)
ax.set_xlabel('$\omega_{ca,p}$',                fontsize=25)
ax.set_ylabel('probability density function ', fontsize=25)
ax.grid(which='both')
ax.tick_params(axis='x', labelsize=22)
ax.tick_params(axis='y', labelsize=22)
#plot limits
ax.set_xlim([0,0.05])
ax.set_ylim([0,ymax_hyp])
#save figure
fig.tight_layout()
fig.savefig( dir_fig + fname_fig + '.png' )

# # Delta c_0
# #---   ---   ---   ---   ---
# #hyper-paramter name
# name_hyp = 'dc_0'
# #figure title
# fname_fig = 'post_dist_' + name_hyp
# #create figure
# fig, ax = plt.subplots(figsize = (10,10))   
# for d_id, df_r_h, df_r_h_p in zip(ds_id, df_reg_hyp, df_reg_hyp_post):
#     #estimate vertical line height for mean and mode
#     ymax_mode = df_r_h_p.loc[:,name_hyp+'_pdf'].max()
#     ymax_mean = 1.5*np.ceil(ymax_mode/10)*10
#     ymax_mean = 15
#     #plot posterior dist
#     pl_pdf = ax.plot(df_r_h_p.loc[:,name_hyp], df_r_h_p.loc[:,name_hyp+'_pdf'])
#     ax.vlines(df_r_h.loc[name_hyp,'mean'], ymin=0, ymax=ymax_mean, linestyle='-',  color=pl_pdf[0].get_color(), label='Mean')
#     ax.vlines(df_r_h.loc[name_hyp,'mode'], ymin=0, ymax=ymax_mode, linestyle='--', color=pl_pdf[0].get_color(), label='Mode')
# #plot true value
# ymax_hyp = ymax_mean
# # ax.vlines(hyp[name_hyp], ymin=0, ymax=ymax_hyp, linestyle='-', linewidth=4, color='black', label='True value')
# #edit figure
# ax.set_title(r'Comparison $\delta c_{0}$',     fontsize=30)
# ax.set_xlabel('$\delta c_{0}$',                fontsize=25)
# ax.set_ylabel('probability density function ', fontsize=25)
# ax.grid(which='both')
# ax.tick_params(axis='x', labelsize=22)
# ax.tick_params(axis='y', labelsize=22)
# #plot limits
# ax.set_xlim([-1,1])
# ax.set_ylim([0,ymax_hyp])
# #save figure
# fig.tight_layout()
# # fig.savefig( dir_fig + fname_fig + '.png' )

