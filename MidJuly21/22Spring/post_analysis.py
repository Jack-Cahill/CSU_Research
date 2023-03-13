import geopandas as gp
import pandas as pd
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.colors as colors
import matplotlib.patheffects as pe
from scipy.stats import pearsonr
from shapely.geometry.polygon import LinearRing
import matplotlib.patheffects as pe

# Analysis functions for post model (nueral network) run

# TV_hmap_mean: comes from after running the neural network 
## provides predictions, confidences, indices and more for the 20%, 30% most confident and all samples

# Acc_map_data: previously calculated accuracies of predicted samples
## based on location, seed, season, and class

# idx_all: input samples  w/ date, ENSO/MJO phase and more information

# Acc_map_dict: simple dictionary that simplifies looping through 20%, 30% most confident and all samples

# plots accuracy map (region or CONUS, or CONUS+) based on season and class (3 types of confidences)
def AccMap_szn_cls(TV_hmap_mean, Acc_main_dict, idx_all, Reg):

    # Set map variables
    vmin = .34  # specific season - specific class
    vmax = .6
    fsize = 24
    SC_ord = [11, 15, 19, 12, 16, 20, 13, 17, 21, 14, 18, 22]  # set plot order for szn-cls
    if Reg == 'colo' or Reg == 'PNW9':
        counter = 4
    Seas_Name = [' - Summer', ' - Fall', ' - Winter', ' - Spring', '']
    Seas_Name2 = ['Summer', 'Fall', 'Winter', 'Spring']
    Cls_Name = ['(UFS Underestimates)', '(UFS Precise Forecast)', '(UFS Overestimates)', '(All Samples)']
    Cls_Name2 = ['UFS Underestimates', 'UFS Precise Forecast', 'UFS Overestimates', 'All Samples']
    
    # Only look at plots that are szn-class
    SC_idx = np.arange(11, 23, 1)
    
    # Map projection
    proj = ccrs.PlateCarree(central_longitude=180)
    
    # Loop through 20, 30, All
    for acm_mid, key in enumerate(Acc_main_dict):
    
        # Map set-up
        if Reg == 'fullplus':
            fig, axs = plt.subplots(4, 3, subplot_kw={'projection': proj}, figsize=(9, 7))
            alpha=0.25
        else:
            fig, axs = plt.subplots(4, 3, subplot_kw={'projection': proj}, figsize=(8.5, 8))
            alpha=1
    
        # axs is a 2 dimensional array of `GeoAxes`.  We will flatten it into a 1-D array
        axs = axs.flatten()
    
        # Get just columns of specific accuracy level
        pop20_pd = TV_hmap_mean[['Index_{}'.format(key), 'CorrOrNo_{}'.format(key), 'Class_{}'.format(key)]].copy()
    
        # Drop 20/30/All aspect of column names
        pop_drop = pop20_pd.rename(columns={'Index_{}'.format(key): 'Index', 'CorrOrNo_{}'.format(key): 'CorrOrNo',
                                            'Class_{}'.format(key): 'Class'})
    
        # include season info
        pop_szns = pd.merge(pop_drop, idx_all, on='Index')
    
        # count total cases
        total_pop = pop_szns['Index'].count()
    
        per_o_per = []
        for pop_class in range(3):
            for pop_season in Seas_Name2:
    
                # Select season & Class
                pop_szn1 = pop_szns[pop_szns["Season"] == pop_season].reset_index(drop=True)
                pop_szn_cls = pop_szn1[pop_szn1['Class'] == pop_class].reset_index(drop=True)
    
                # Count cases & divide by total
                per_o_per += [(pop_szn_cls.count()[0] / total_pop) * 100]
    
        for acm, SC in enumerate(SC_ord):
    
            # lat, lon extents
            if Reg == 'PNW9':
                axs[acm].set_extent([51, 66, 43.5, 50.5], ccrs.PlateCarree(central_longitude=180))
            elif Reg == 'fullplus':
                axs[acm].set_extent([-20, 115, 21, 54], ccrs.PlateCarree(central_longitude=180))
            elif Reg == 'colo':
                axs[acm].set_extent([66, 81, 34.5, 41.5], ccrs.PlateCarree(central_longitude=180))
            else:
                axs[acm].set_extent([53.5, 113.5, 22, 50], ccrs.PlateCarree(central_longitude=180))
    
            # Set savefig names
            seas = (acm + 1) % 4
            clss = (acm - 11) // 4
            Pr = 0
    
            # Add features
            axs[acm].add_feature(cfeature.COASTLINE, alpha=alpha)
            axs[acm].add_feature(cfeature.STATES.with_scale('10m'), edgecolor=ecolor, alpha=alpha)
            axs[acm].add_feature(cfeature.BORDERS, edgecolor=ecolor, alpha=alpha)
            axs[acm].add_feature(cfeature.LAKES, color=ecolor, alpha=alpha/2)
    
            # plot (founf by plotting their conus lons and lats and matching to reg_lat_lon values via indexing)
            if Reg == 'PNW9' or Reg == 'colo':
                cf = axs[acm].pcolor(lons_p - 180, lats_p, Acc_Map_Data[acm_mid][SC][5:11:4, 2:7:4].T,
                                     vmin=vmin, vmax=vmax,
                                     cmap=plt.cm.get_cmap('Reds'), transform=ccrs.PlateCarree(central_longitude=180))
            elif Reg == 'fullplus':
                cf = axs[acm].pcolor(lons_p - 180, lats_p, Acc_Map_Data[acm_mid][SC][:260:10, 2::10].T,
                                     vmin=vmin, vmax=vmax,
                                     cmap=plt.cm.get_cmap('Reds'), transform=ccrs.PlateCarree(central_longitude=180))
            else:
                cf = axs[acm].pcolor(lons_p - 180, lats_p, Acc_Map_Data[acm_mid][SC].T,
                                     vmin=vmin, vmax=vmax,
                                     cmap=plt.cm.get_cmap('Reds'), transform=ccrs.PlateCarree(central_longitude=180))
    
            # sub titles
            if acm < 3:
                axs[acm].set_title('{}'.format(Cls_Name2[acm]), fontsize=16)
    
            '''if acm % 3 == 0:
                axs[acm].set_ylabel('\n\n{}'.format(Seas_Name2[int(acm/3)]), rotation=0, fontsize=16)'''
    
            '''# plot percent of percent in bottom right
            axs[acm].text(.99, .18, '{}%'.format(round(per_o_per[acm], 1)), ha='right', va='bottom',
                          transform=axs[acm].transAxes, fontsize=11, weight='bold', color='darkred')'''
    
            '''# plot overall percent in bottom left
            axs[acm].text(.01, .01, '{}%'.format(round(ovr_pct[acm], 1)), ha='left', va='bottom',
                          transform=axs[acm].transAxes, fontsize=11, weight='bold')'''
    
        # plot info
        #cbaxes = fig.add_axes([0.1, 0.1, 0.03, 0.8])
        fig.colorbar(cf, ax=axs.ravel().tolist(), location='bottom', pad=0.01, aspect=50, extendrect=True)
        plt.subplots_adjust(left=0.1, bottom=0.25, right=0.9, top=0.785, wspace=0.1, hspace=0.1)
    
        # y labels not working so this a buffer
        if Reg == 'fullplus':
            fig.text(0.072, 0.5, ' Spring    Winter     Fall    Summer', va='center', rotation='vertical',
                 fontsize=16)
        else:
            fig.text(0.072, 0.5, '      Spring     Winter       Fall       Summer', va='center', rotation='vertical',
                 fontsize=16)
    
        # out plot
        fig.suptitle('\nAccuracy of Predicting {} Errors in the UFS\n{}'.format(out_dict[out][3], 
                                                                                Acc_main_dict[key][6]),
                     fontsize=fsize)
        #plt.savefig('AccMap_SZNCLS_{}_Pr{}k{}_{}_ac'.format(Reg, Pr, key, Pred), dpi=300)
        plt.show()


# plots accuracy map (region or CONUS, or CONUS+) based on season and class (3 types of confidences)-- COMP to ALL
def AccMap_szn_cls_comp(TV_hmap_mean, Acc_main_dictX, idx_all, Reg):
    
    # map variables
    vmin = .0  # specific season - specific class
    vmax = 0.25
    fsize = 24
    SC_ord = [11, 15, 19, 12, 16, 20, 13, 17, 21, 14, 18, 22]  # set plot order for szn-cls
    proj = ccrs.PlateCarree(central_longitude=180)
    Seas_Name = [' - Summer', ' - Fall', ' - Winter', ' - Spring', '']
    Seas_Name2 = ['Summer', 'Fall', 'Winter', 'Spring']
    Cls_Name = ['(UFS Underestimates)', '(UFS Precise Forecast)', '(UFS Overestimates)', '(All Samples)']
    Cls_Name2 = ['UFS Underestimates', 'UFS Precise Forecast', 'UFS Overestimates', 'All Samples']
    
    # Loop through 20, 30, All
    for acm_mid, key in enumerate(Acc_main_dictX):
    
        # Map set-up
        if Reg == 'fullplus':
            fig, axs = plt.subplots(4, 3, subplot_kw={'projection': proj}, figsize=(9, 7))
            alpha=0.25
        else:
            fig, axs = plt.subplots(4, 3, subplot_kw={'projection': proj}, figsize=(8.5, 8))
            alpha=1
    
        # axs is a 2 dimensional array of `GeoAxes`.  We will flatten it into a 1-D array
        axs = axs.flatten()
    
        for acm, SC in enumerate(SC_ord):
    
            # lat, lon extents
            if Reg == 'PNW9':
                axs[acm].set_extent([51, 66, 43.5, 50.5], ccrs.PlateCarree(central_longitude=180))
            elif Reg == 'fullplus':
                axs[acm].set_extent([-20, 115, 21, 54], ccrs.PlateCarree(central_longitude=180))
            elif Reg == 'colo':
                axs[acm].set_extent([66, 81, 34.5, 41.5], ccrs.PlateCarree(central_longitude=180))
            else:
                axs[acm].set_extent([53.5, 113.5, 22, 50], ccrs.PlateCarree(central_longitude=180))
    
            # Set savefig names
            seas = (acm + 1) % 4
            clss = (acm - 11) // 4
            Pr = 0
    
            # Add features
            axs[acm].add_feature(cfeature.COASTLINE, alpha=alpha)
            axs[acm].add_feature(cfeature.STATES.with_scale('10m'), edgecolor=ecolor, alpha=alpha)
            axs[acm].add_feature(cfeature.BORDERS, edgecolor=ecolor, alpha=alpha)
            axs[acm].add_feature(cfeature.LAKES, color=ecolor, alpha=alpha/2)
    
            # plot
            if Reg == 'PNW9' or Reg == 'colo':
                cf = axs[acm].pcolor(lons_p - 180, lats_p, 
                                     (Acc_Map_Data[acm_mid][SC][5:11:4, 2:7:4].T - Acc_Map_Data[2][SC][5:11:4, 2:7:4].T),
                                     vmin=vmin, vmax=vmax,
                                     cmap=plt.cm.get_cmap('Reds'), transform=ccrs.PlateCarree(central_longitude=180))
            elif Reg == 'fullplus':
                cf = axs[acm].pcolor(lons_p - 180, lats_p, 
                                     (Acc_Map_Data[acm_mid][SC][:260:10, 2::10].T - Acc_Map_Data[2][SC][:260:10, 2::10].T),
                                     vmin=0, vmax=.25,
                                     cmap=plt.cm.get_cmap('Reds'), transform=ccrs.PlateCarree(central_longitude=180))
            else:
                cf = axs[acm].pcolor(lons_p - 180, lats_p, (Acc_Map_Data[acm_mid][SC].T - Acc_Map_Data[2][SC].T),
                                     vmin=0, vmax=.25,
                                     cmap=plt.cm.get_cmap('Reds'), transform=ccrs.PlateCarree(central_longitude=180))
    
            # sub titles
            if acm < 3:
                axs[acm].set_title('{}'.format(Cls_Name2[acm]), fontsize=16)
    
            # plot overall percent in bottom left
            '''axs[acm].text(.01, .01, '{}%'.format(round(ovr_pct[acm], 1)), ha='left', va='bottom',
                          transform=axs[acm].transAxes, fontsize=11, weight='bold')'''
    
        # plot info
        fig.colorbar(cf, ax=axs.ravel().tolist(), location='bottom', pad=0.01, aspect=50, extendrect=True)
        plt.subplots_adjust(left=0.1, bottom=0.25, right=0.9, top=0.785, wspace=0.1, hspace=0.1)
    
        # y labels not working so this a buffer
        if Reg == 'fullplus':
            fig.text(0.072, 0.5, ' Spring    Winter     Fall    Summer', va='center', rotation='vertical',
                 fontsize=16)
        else:
            fig.text(0.072, 0.5, '      Spring     Winter       Fall       Summer', va='center', rotation='vertical',
                 fontsize=16)
    
        # out plot
        fig.suptitle('{}: Accuracy Improvements of the {}%\nMost Confident Samples Compared to All Samples\nLead Time of 10-14 days'.format(
            out_dict[out][3], key), fontsize=fsize)
    
        #plt.savefig('AccMap_SZNCLS_comp_{}_Pr{}k{}_{}_ac'.format(Reg, Pr, key, Pred), dpi=300)
        plt.show()

# plots input maps (composite) at various confideneces for specific class and season
def InMapFn(TV_hmap_mean, Acc_main_dictX, idx_all, Reg, sel_szn, sel_cls):

    # latitude lines
    lonA = np.linspace(-120, 120)
    latA = np.linspace(-25, -25)
    lonB = np.linspace(-120, 120)
    latB = np.linspace(25, 25)
    A1, A2 = (lonA, latA)
    B1, B2 = (lonB, latB)
    
    # HMAP loop
    for cc, key in enumerate(Acc_main_dictY):
    
        # Get just columns of specific accuracy level
        #imap_key = TV_hmap_mean[['Index_{}'.format(key), 'CorrOrNo_{}'.format(key), 'Class_{}'.format(key)]].copy()
        imap_key = TV_hmap_mean[['Index_{}'.format(key), 'CorrOrNo_{}'.format(key), 'ActCls_{}'.format(key)]].copy()
        
        # szn
        for szn_val in sel_szn:
        
            # class
            for class_num in sel_cls:
        
                # Drop key aspect of column names
                #imap_key = imap_key.rename(columns={'Index_{}'.format(key): 'Index', 'CorrOrNo_{}'.format(key): 'CorrOrNo',
                #                                    'Class_{}'.format(key): 'Class'})
                imap_key = imap_key.rename(columns={'Index_{}'.format(key): 'Index', 'CorrOrNo_{}'.format(key): 'CorrOrNo',
                                                    'ActCls_{}'.format(key): 'Class'})
        
                # Select season
                imap_key['Index'] = pd.to_numeric(imap_key['Index'])
                imap_szn = pd.merge(idx_all, imap_key, on='Index')
                i_all = imap_szn[imap_szn['Season'] == szn_val].reset_index(drop=True)
                
                # Look at correct samples for specific class
                class_i_all = i_all[i_all['Class'] == class_num].reset_index(drop=True)
                class_i_corr = class_i_all[class_i_all['CorrOrNo'] == 1].reset_index(drop=True)
                
                if key == '30':
                    save30 = class_i_all['Index'].value_counts()
                    print(save30)
                    print(type(save30))
                    #save30.to_csv('Save30_{}_{}_{}{}_ac.csv'.format(Reg, Pred, szn_val, class_num))
                   
                # Put them in a list, so we can loop through them
                if key == 'All':
                    info_list = [i_all, class_i_all]
                    out_names = ['all_all', '{}_corr'.format(class_num)]
                    out_names2 = ['All Classes', 'Class {}'.format(class_num)]
                else:
                    info_list = [class_i_all]
                    out_names = ['{}_corr'.format(class_num)]
                    out_names2 = ['Class {}'.format(class_num)]
        
                # Organize and plot each hmap
                for inf in range(len(info_list)):
         
                    # OLR
                    avged = info_list[inf]['obs'].mean()
                    
                    # u and v barbs
                    u_avg = info_list[inf]['u raw obs'].mean()
                    v_avg = info_list[inf]['v raw obs'].mean()
                    
                    # Plot normal
                    fig = plt.figure(figsize=(10, 8))
                    ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=180))
                    ax.set_extent([-120, 120, -25, 25], ccrs.PlateCarree(central_longitude=180))  # lat, lon extents
                    clevs = np.arange(-12, 12.5, .5)
                    cf = ax.contourf(lons, lats, avged, clevs, cmap=plt.cm.bwr, 
                                     transform=ccrs.PlateCarree(central_longitude=180), extend='both')
                    cbar = plt.colorbar(cf, orientation='horizontal', pad=0.04, aspect=50, extendrect=True)  # aspect=50 flattens cbar
                    ax.add_feature(cfeature.COASTLINE.with_scale('50m'))
                    ax.add_feature(cfeature.LAKES.with_scale('50m'), color='black', linewidths=0.5)
                    plt.title('{}, {} - Input Maps\n{}'.format(szn_val.capitalize(), out_names2[inf],  
                                                                    Acc_main_dictY[key][5]), fontsize=20)
                    #plt.savefig('Inmap_{}_{}_{}_k{}_{}_norm'.format(szn_val, Reg, out_names[inf], key, Pred), dpi=300)
                    plt.show()
                    
                    # Plot with winds
                    fig = plt.figure(figsize=(10, 8))
                    ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=180))
                    ax.set_extent([-120, 120, -30, 60], ccrs.PlateCarree(central_longitude=180))  # lat, lon extents
                    clevs = np.arange(-12, 12.5, .5)
                    cf = ax.contourf(lons, lats, avged, clevs, cmap=plt.cm.bwr, 
                                     transform=ccrs.PlateCarree(central_longitude=180), extend='both')
                    print(lons[::10].shape, lats[::10].shape)
                    ax.quiver(lons[::10], lats[::10], u_avg[::10, ::10], v_avg[::10, ::10])
                    cbar = plt.colorbar(cf, orientation='horizontal', pad=0.04, aspect=50, extendrect=True)  # aspect=50 flattens cbar
                    ax.add_feature(cfeature.COASTLINE.with_scale('50m'))
                    ax.add_feature(cfeature.LAKES.with_scale('50m'), color='black', linewidths=0.5)
                    ax.plot(A1, A2, linewidth=3, color='lime', linestyle='--')
                    ax.plot(B1, B2, linewidth=3, color='lime', linestyle='--')
                    plt.title('{}, {} - Input Maps\n{}'.format(szn_val.capitalize(), out_names2[inf],  
                                                                    Acc_main_dictY[key][5]), fontsize=20)
                    #plt.savefig('Inmap_{}_{}_{}_k{}_{}_winds'.format(szn_val, Reg, out_names[inf], key, Pred), dpi=300)
                    plt.show()

# plots input maps (composite) at various confideneces for specific class and season
def InMapFn_solos(idx_all, Reg, i_idxs, sel_szn, sel_cls):

    # latitude lines
    lonA = np.linspace(-120, 120)
    latA = np.linspace(-25, -25)
    lonB = np.linspace(-120, 120)
    latB = np.linspace(25, 25)
    A1, A2 = (lonA, latA)
    B1, B2 = (lonB, latB)
    
    # Solo plots
    
    for ii in i_idxs:
        
        # Get value
        i_solo = idx_all[idx_all['Index'] == ii].reset_index(drop=True)
         
        # Plot solos
        fig = plt.figure(figsize=(10, 8))
        ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=180))
        ax.set_extent([-120, 120, -30, 60], ccrs.PlateCarree(central_longitude=180))  # lat, lon extents
        clevs = np.arange(-100, 105, 5)
        cf = ax.contourf(lons, lats, i_solo['obs'].mean(), clevs, cmap=plt.cm.bwr, 
                         transform=ccrs.PlateCarree(central_longitude=180), extend='both')
        ax.quiver(lons[::10], lats[::10], i_solo['u raw obs'].mean()[::10, ::10], i_solo['v raw obs'].mean()[::10, ::10])
        cbar = plt.colorbar(cf, orientation='horizontal', pad=0.04, aspect=50, extendrect=True)  # aspect=50 flattens cbar
        ax.add_feature(cfeature.COASTLINE.with_scale('50m'))
        ax.add_feature(cfeature.LAKES.with_scale('50m'), color='black', linewidths=0.5)
        ax.plot(A1, A2, linewidth=3, color='lime', linestyle='--')
        ax.plot(B1, B2, linewidth=3, color='lime', linestyle='--')
        plt.title('Sample {} - Input Map'.format(ii), fontsize=20)
        plt.tight_layout()
        #plt.savefig('Inmap_{}{}_{}_{}_Samp{}'.format(sel_szn, sel_cls, Reg, Pred, ii), dpi=300)
        plt.show()

# plots ENSO-MJO heatmpas for various confidences at specific season and class
def hmap_relszn(TV_hmap_mean, Acc_main_dictY, idx_all, Reg, sel_szn, sel_cls):

    # Make a "total" hmap that can we can divide by to normalize the hmaps
    hmap_main = np.zeros((3, 9))
    for x in np.arange(-1, 2, 1):
        ENSOs = idx_all.loc[idx_all['E_Phase'] == x]
        for y in np.arange(0, 9, 1):
            MJO_ENSO = ENSOs.loc[ENSOs['M_Phase'] == y]
            hmap_main[x + 1, y] = len(MJO_ENSO['Index'])
    mout = hmap_main*6*counter
    
    # Plot hmap
    plt.imshow(mout, cmap=plt.cm.Reds, vmin=0, vmax=10000)
    plt.xlabel('MJO Phase')
    plt.ylabel('ENSO Phase')
    plt.title('Phase Counts\nAmong All Samples', fontsize=16)
    plt.xticks([0, 1, 2, 3, 4, 5, 6, 7, 8], ['None', 1, 2, 3, 4, 5, 6, 7, 8])
    plt.yticks([0, 1, 2], ['La Niña', 'Neutral', 'El Niño'])
    
    # plot values inside hmap
    for (m, n), label in np.ndenumerate(mout.astype(int)):
        plt.text(n, m, label, fontsize=7, ha='center', va='center', color='black')
    
    plt.colorbar(shrink=0.75, label='Count')
    plt.tight_layout()
    #plt.savefig('HeatmapOVRALL_{}_main_{}'.format(Reg, Pred), dpi=300)
    plt.show()
    
    # HMAP loop
    for cc, key in enumerate(Acc_main_dictY):
    
        # Get just columns of specific accuracy level
        hmap_key = TV_hmap_mean[['Index_{}'.format(key), 'CorrOrNo_{}'.format(key), 'ActCls_{}'.format(key)]].copy()
        
        # szn
        for szn_val in sel_szn:
        
            # class
            for class_num in sel_cls:
        
                # Drop key aspect of column names
                hmap_key = hmap_key.rename(columns={'Index_{}'.format(key): 'Index', 'CorrOrNo_{}'.format(key): 'CorrOrNo',
                                                    'ActCls_{}'.format(key): 'Class'})
        
                # Select season
                hmap_key['Index'] = pd.to_numeric(hmap_key['Index'])
                hmap_szn = pd.merge(idx_all, hmap_key, on='Index')
                print(hmap_szn.shape)
                print('\n')
                all_all = hmap_szn[hmap_szn['Season'] == szn_val].reset_index(drop=True)
        
                # Separate based on class and correct samples
                class_all = all_all[all_all['Class'] == class_num].reset_index(drop=True)
                class_corr = class_all[class_all['CorrOrNo'] == 1].reset_index(drop=True)
        
                # Put them in a list, so we can loop through them
                if key == 'All':
                    info_list = [all_all, class_all, class_corr]
                    out_names = ['all_all', '{}_all'.format(class_num), '{}_corr'.format(class_num)]
                    out_names2 = ['All Classes', 'Class {}'.format(class_num), 'Class {}'.format(class_num)]
                else:
                    info_list = [class_all, class_corr]
                    out_names = ['{}_all'.format(class_num), '{}_corr'.format(class_num)]
                    out_names2 = ['Class {}'.format(class_num), 'Class {}'.format(class_num)]
        
                # Organize and plot each hmap
                for inf in range(len(info_list)):
        
                    if inf == 0:
                        XX = 5
                    elif inf == 1 and key == 'All':
                        XX = 5
                    else:
                        XX = 0
        
                    # Count indices
                    info_count = info_list[inf].groupby(['Index'])['Index'].count()
                    info_count_df = info_count.to_frame()
                    info_idx = info_count_df.index  # Indices
                    info_counts = info_count_df['Index'].reset_index(drop=True)  # Count
        
                    # Convert index and counts into pandas array
                    countpd = pd.DataFrame({'Index': info_idx, 'count': info_counts})
        
                    # Merge with ENSO and MJO info
                    EMJO = pd.merge(countpd, idx_all, on='Index')
                        
                    # Create hmap to show frequency of ENSO and MJO
                    hmap = np.zeros((3, 9))
                    for x in np.arange(-1, 2, 1):
                        ENSOs = EMJO.loc[EMJO['E_Phase'] == x]
                        for y in np.arange(0, 9, 1):
                            MJO_ENSO = ENSOs.loc[ENSOs['M_Phase'] == y]
                            hmap[x + 1, y] = np.sum(MJO_ENSO['count'])
         
                    if key == 'All':
                        vmin = 0
                        if inf == 0:
                            ttlabel = 'Count'
                            clabel = 'Count'
                            vmax = 2500
                            fsize = 7
                            hmap_out = hmap * 1
                        elif inf == 1:
                            ttlabel = 'Count'
                            clabel = 'Count'
                            vmax = 1000
                            fsize = 7
                            hmap_special = hmap  # Set up special for all-spec_szn-spec_cls
                            hmap_out = hmap_special * 1
                        else:
                            ttlabel = 'Freqeuncy'
                            clabel = 'Freq Rel to All {} Class {} Samples (%)'.format(szn_val.capitalize(), class_num)
                            vmax = 100
                            fsize = 9
                            # round to 4 decimals
                            hmap_out = hmap / hmap_special
                            hmap_out = hmap_out * 100
                            hmap_out = np.around(hmap_out, decimals=2)    
                    
                    else:
                        hmap_out = hmap / hmap_special
                        
                        ttlabel = 'Freqeuncy'
                        clabel = 'Freq Rel to All {} Class {} Samples (%)'.format(szn_val.capitalize(), class_num)
                        vmin = 0
                        vmax = 100
                        fsize = 9
                        
                        # round to 4 decimals
                        hmap_out = hmap_out * 100
                        hmap_out = np.around(hmap_out, decimals=2) 
        
                    # Plot hmap
                    plt.imshow(hmap_out, cmap=plt.cm.Reds, vmin=vmin, vmax=vmax)
                    plt.xlabel('MJO Phase')
                    plt.ylabel('ENSO Phase')
                    plt.title('Phase {} - {} - {}\n{}'.format(ttlabel, out_names2[inf], szn_val.capitalize(), 
                                                                    Acc_main_dictY[key][XX]), fontsize=16)
                    plt.xticks([0, 1, 2, 3, 4, 5, 6, 7, 8], ['None', 1, 2, 3, 4, 5, 6, 7, 8])
                    plt.yticks([0, 1, 2], ['La Niña', 'Neutral', 'El Niño'])
                    for (m, n), label in np.ndenumerate(hmap_out):  # plot values inside hmap
                        plt.text(n, m, label, fontsize=fsize, ha='center', va='center', color='black')
                    plt.colorbar(shrink=0.75, label=clabel)
                    plt.tight_layout()
                    #plt.savefig('Heatmap_{}_{}_{}_k{}_{}_relszn'.format(szn_val, Reg, out_names[inf], key, Pred), dpi=300)
                    plt.show()

# Plot multiple LT progressions (composites) of 30% most confident samples
def LT_progression(TV_hmap_mean, idx_all, Reg, sel_szn, sel_cls, typo):
    
    # set base variables
    sel_rg = ['30']
    p_tits = ['UFS Forecast', 'Observations', 'Error (UFS - Obs']
    clevs = np.arange(-50, 51, 1)
    
    # set variables based on 3x3 or 6x3
    if typo == '3x3':
        time_rgs = [0, 5, 5, 10, 10, 15]
        num_rows = 3
    elif typo == '6x3':
        time_rgs = [0, 3, 3, 6, 6, 9, 9, 12, 12, 15, 15, 18]
        num_rows = 6
        
    for key in sel_rg:
    
        # Get just columns of specific accuracy level
        imap_key = TV_hmap_mean[['Index_{}'.format(key), 'CorrOrNo_{}'.format(key), 'ActCls_{}'.format(key)]].copy()
        
        # Drop key aspect of column names
        imap_key = imap_key.rename(columns={'Index_{}'.format(key): 'Index', 'CorrOrNo_{}'.format(key): 'CorrOrNo',
                                            'ActCls_{}'.format(key): 'Class'})
        imap_key['Index'] = pd.to_numeric(imap_key['Index'])
        
        # szn & class
        for szn_val in sel_szn:
            
            for cls_val in sel_cls:
    
                # Select season & class
                imap_szn = pd.merge(idx_all, imap_key, on='Index')
                i_szn = imap_szn[imap_szn['Season'] == szn_val].reset_index(drop=True)
                i_cls = i_szn[i_szn['Class'] == cls_val].reset_index(drop=True)
                
                top30 = i_cls['Index'].value_counts()
                top30_df = pd.DataFrame({'Index':top30.index, 'count':top30.values})
                
                # sum samples
                comp_list = []
                for ts in range(len(top30_df['Index'])):
                    
                    t_samp = top30_df['Index'][ts]
                    
                    # Get ufs values (need 15 LTs for 3x3 & 18 LTs for 6x3)
                    ufs_hold = ds_u[t_samp * LT_tot:t_samp * LT_tot + num_rows + 12]
                    df_u_l = list(zip(ufs_hold.time.values, ufs_hold.values))
                    df_u = pd.DataFrame(df_u_l)
                    df_u.columns = ['time', 'ufs']
                    df_u['time'] = pd.to_datetime(df_u['time']).dt.date
                    
                    # merge with obs & calc error
                    uoe = pd.merge(df_u, df_o, on='time')
                    uoe['error'] = uoe['ufs'] - uoe['obs']
                    
                    # sample 1041 only has LTs up to 14 (repeat 14th day - 1time for 3x3 & 4times for 6x3)
                    if t_samp == 1041:
                        for tsa in range(num_rows - 2):
                            uoe.loc[len(uoe['error'])] = uoe.loc[len(uoe['error'])-1]
                    
                    # change column name to match winds
                    uoe = uoe.rename({'time': 'dates'}, axis='columns')
                    
                    # merge with winds
                    dfu_raw1['dates'] = pd.to_datetime(dfu_raw1['dates']).dt.date
                    dfv_raw1['dates'] = pd.to_datetime(dfv_raw1['dates']).dt.date
                    uoe = pd.merge(uoe, dfu_raw1, on='dates')
                    uoe = pd.merge(uoe, dfv_raw1, on='dates')
                    
                    # drop date
                    uoe = uoe.drop(['dates'], axis=1)
                    
                    # put in list
                    uoe_l = uoe * int(top30_df['count'][ts])
                    comp_list += [uoe_l]
                    
                # Composite
                comp_out = sum(comp_list)/top30_df['count'].sum()
             
                # Plot setup
                fig, axs = plt.subplots(num_rows, 3, subplot_kw={'projection': ccrs.PlateCarree(central_longitude=180)}, figsize=(8, num_rows+2))
                axs = axs.flatten()
                
                # plot
                p_count = 0
                for lt in range(num_rows):  # 3 lead time ranges
                    for ty in range(3): # 3 plot types
                    
                        # map features
                        axs[p_count].set_extent([-120, 120, -30, 70], ccrs.PlateCarree(central_longitude=180))
                        axs[p_count].add_feature(cfeature.COASTLINE, edgecolor='black')
                        
                        # get map (format: gets coulmun name(ufs, obs, error)) - (time_rgs: gets LT range)
                        map1 = comp_out['{}'.format(comp_out.columns.values.tolist()[ty])][time_rgs[lt*2]: time_rgs[lt*2 + 1]]
                        map_avg = map1.mean()
                        
                        # plot
                        cf = axs[p_count].contourf(lons15, lats15, map_avg, clevs, extend='both',
                                                   cmap=plt.cm.bwr, transform=ccrs.PlateCarree(central_longitude=180))
                        
                        # plot winds if obs
                        if typo == '3x3':
                            if p_count == 1 or p_count == 4 or p_count == 7:
                                axs[p_count].quiver(lons15[::15], lats15[::15], 
                                                    comp_out['u raw obs'][time_rgs[lt*2]: time_rgs[lt*2 + 1]].mean()[::15, ::15], 
                                                    comp_out['v raw obs'][time_rgs[lt*2]: time_rgs[lt*2 + 1]].mean()[::15, ::15])
                        elif typo == '6x3':
                            if p_count == 1 or p_count == 4 or p_count == 7 or p_count == 10 or p_count == 13 or p_count == 16:
                                axs[p_count].quiver(lons15[::15], lats15[::15], 
                                                    comp_out['u raw obs'][time_rgs[lt*2]: time_rgs[lt*2 + 1]].mean()[::15, ::15], 
                                                    comp_out['v raw obs'][time_rgs[lt*2]: time_rgs[lt*2 + 1]].mean()[::15, ::15])
                         
                        # show study region
                        axs[p_count].plot([reg_lon_lat['longitude'][3]+180], [reg_lon_lat['latitude'][1]], 
                                          marker='s', markersize=6, markeredgecolor='lime', markerfacecolor='none')
                        
                        # sub titles
                        if p_count < 3:
                            axs[p_count].set_title('{}'.format(p_tits[p_count]), fontsize=16)
        
                        p_count += 1
            
                # y-label
                if typo == '3x3':
                    fig.text(0.08, 0.65, 'LT\n0-4', ha='center', fontsize=16)
                    fig.text(0.08, 0.475, 'LT\n5-10', ha='center', fontsize=16)
                    fig.text(0.08, 0.3, 'LT\n10-14', ha='center', fontsize=16)
                else:
                    fig.text(0.08, 0.805, 'LT\n0-2', ha='center', fontsize=16)
                    fig.text(0.08, 0.7, 'LT\n3-5', ha='center', fontsize=16)
                    fig.text(0.08, 0.595, 'LT\n6-8', ha='center', fontsize=16)
                    fig.text(0.08, 0.49, 'LT\n9-12', ha='center', fontsize=16)
                    fig.text(0.08, 0.385, 'LT\n12-14', ha='center', fontsize=16)
                    fig.text(0.08, 0.28, 'LT\n15-17', ha='center', fontsize=16)
                fig.colorbar(cf, ax=axs.ravel().tolist(), location='bottom', pad=0.01, aspect=50, extendrect=True) 
                plt.subplots_adjust(left=0.125, bottom=0.25, right=0.9, top=num_rows/30 + .675, wspace=0.1, hspace=0.025)           
                fig.suptitle('{} Progression - {}% Most Confident Samples\n{} {}'.format(Pred, key, szn_val, Cls_Name2[cls_val]), fontsize=20) # 
                #plt.savefig('{}x3_{}_{}_k{}_{}_{}_'.format(num_rows, szn_val, cls_val, key, Reg, Pred), dpi=300)
                plt.show()

# Plot multiple LT progressions (SOLO PLOTS) of 30% most confident samples
def LT_prog_solos(TV_hmap_mean, idx_all, Reg, sel_szn, sel_cls):
    
    # set variables
    sel_rg = ['30']
    time_rgs = [0, 3, 3, 6, 6, 9, 9, 12, 12, 15, 15, 18]
    p_tits = ['UFS Forecast', 'Observations', 'Error (UFS - Obs']
    clevs = np.arange(-300, 310, 10)

    # 30% only
    for key in sel_rg:
    
        # Get just columns of specific accuracy level
        imap_key = TV_hmap_mean[['Index_{}'.format(key), 'CorrOrNo_{}'.format(key), 'ActCls_{}'.format(key)]].copy()
        
        # Drop key aspect of column names
        imap_key = imap_key.rename(columns={'Index_{}'.format(key): 'Index', 'CorrOrNo_{}'.format(key): 'CorrOrNo',
                                            'ActCls_{}'.format(key): 'Class'})
        imap_key['Index'] = pd.to_numeric(imap_key['Index'])
        
        # szn & class
        for szn_val in sel_szn:
            
            for cls_val in sel_cls:
    
                # Select season & class
                imap_szn = pd.merge(idx_all, imap_key, on='Index')
                i_szn = imap_szn[imap_szn['Season'] == szn_val].reset_index(drop=True)
                i_cls = i_szn[i_szn['Class'] == cls_val].reset_index(drop=True)
                
                top30 = i_cls['Index'].value_counts()
                top30_df = pd.DataFrame({'Index':top30.index, 'count':top30.values})
                
                # Look at 3 most confident samples (conf across seed and location)
                for ts in range(4):
    
                    t_samp = top30_df['Index'][ts]
                    
                    # Get ufs values
                    ufs_hold = ds_u[t_samp * LT_tot:t_samp * LT_tot + 18]
                    df_u_l = list(zip(ufs_hold.time.values, ufs_hold.values))
                    df_u = pd.DataFrame(df_u_l)
                    df_u.columns = ['time', 'ufs']
                    df_u['time'] = pd.to_datetime(df_u['time']).dt.date
                    
                    # merge with obs & calc error
                    uoe = pd.merge(df_u, df_o, on='time')
                    uoe['error'] = uoe['ufs'] - uoe['obs']
                    
                    # fix 1041 missing day but repeating 13th day as 14th+
                    if t_samp == 1041:
                        for tsa in range(4):
                            uoe.loc[len(uoe['error'])] = uoe.loc[len(uoe['error'])-1]
                    
                    # change column name to match winds
                    uoe = uoe.rename({'time': 'dates'}, axis='columns')
                    
                    # merge with winds
                    dfu_raw1['dates'] = pd.to_datetime(dfu_raw1['dates']).dt.date
                    dfv_raw1['dates'] = pd.to_datetime(dfv_raw1['dates']).dt.date
                    uoe = pd.merge(uoe, dfu_raw1, on='dates')
                    uoe = pd.merge(uoe, dfv_raw1, on='dates')
                    
                    # drop date
                    uoe = uoe.drop(['dates'], axis=1)
             
                    # Plot setup
                    fig, axs = plt.subplots(6, 3, subplot_kw={'projection': ccrs.PlateCarree(central_longitude=180)}, figsize=(8, 8))
                    axs = axs.flatten()
                    
                    # plot
                    p_count = 0
                    for lt in range(6):  # 5 lead time ranges
                        for ty in range(3): # 3 plot types
                        
                            # map features
                            axs[p_count].set_extent([-120, 120, -30, 70], ccrs.PlateCarree(central_longitude=180))
                            axs[p_count].add_feature(cfeature.COASTLINE, edgecolor='black')
                            
                            # get map (format: gets coulmun name(ufs, obs, error)) - (time_rgs: gets LT range)
                            map1 = uoe['{}'.format(uoe.columns.values.tolist()[ty])][time_rgs[lt*2]: time_rgs[lt*2 + 1]]
                            map_avg = map1.mean()
                            
                            # plot
                            cf = axs[p_count].contourf(lons15, lats15, map_avg, clevs, extend='both',
                                                       cmap=plt.cm.bwr, transform=ccrs.PlateCarree(central_longitude=180))
                            
                            # plot winds if obs
                            if p_count == 1 or p_count == 4 or p_count == 7 or p_count == 10 or p_count == 13 or p_count == 16:
                                axs[p_count].quiver(lons15[::15], lats15[::15], 
                                                    uoe['u raw obs'][time_rgs[lt*2]: time_rgs[lt*2 + 1]].mean()[::15, ::15], 
                                                    uoe['v raw obs'][time_rgs[lt*2]: time_rgs[lt*2 + 1]].mean()[::15, ::15])
                             
                            # show study region
                            axs[p_count].plot([reg_lon_lat['longitude'][3]+180], [reg_lon_lat['latitude'][1]], 
                                              marker='s', markersize=6, markeredgecolor='lime', markerfacecolor='none')
                            
                            # sub titles
                            if p_count < 3:
                                axs[p_count].set_title('{}'.format(p_tits[p_count]), fontsize=16)
            
                            p_count += 1
                
                    # y-label
                    fig.text(0.08, 0.805, 'LT\n0-2', ha='center', fontsize=16)
                    fig.text(0.08, 0.7, 'LT\n3-5', ha='center', fontsize=16)
                    fig.text(0.08, 0.595, 'LT\n6-8', ha='center', fontsize=16)
                    fig.text(0.08, 0.49, 'LT\n9-12', ha='center', fontsize=16)
                    fig.text(0.08, 0.385, 'LT\n12-14', ha='center', fontsize=16)
                    fig.text(0.08, 0.28, 'LT\n15-17', ha='center', fontsize=16)
                    fig.colorbar(cf, ax=axs.ravel().tolist(), location='bottom', pad=0.01, aspect=50, extendrect=True) 
                    plt.subplots_adjust(left=0.125, bottom=0.25, right=0.9, top=0.875, wspace=0.1, hspace=0.025)           
                    fig.suptitle('{} Progression - Confident Sample {}\n{} {}'.format(Pred, int(t_samp), szn_val, Cls_Name2[cls_val]), fontsize=20) # 
                    #plt.savefig('6x3_{}_{}_k{}_{}_{}_solo{}'.format(szn_val, cls_val, key, Reg, Pred, int(t_samp)), dpi=300)
                    plt.show()

# Based on season and class, create two MJO lineplots (rel szn and szn/cls)
def MJO_lplot(TV_hmap_mean, idx_all, Reg, sel_szn, sel_cls, typo):
    
    # how many in lineplot
    if typo == 'M_Phase':
        lpn = 9
    elif typo == 'E_Phase':
        lpn = 3
    else:
        print('oops, typo doesn\'t exist')
    
    keys = ['All', '30']
    for key in keys:
    
        # Get just columns of specific accuracy level
        hmap_key = TV_hmap_mean[['Index_{}'.format(key), 'CorrOrNo_{}'.format(key), 'ActCls_{}'.format(key)]].copy()   
        
        
        # Drop key aspect of column names
        hmap_key = hmap_key.rename(columns={'Index_{}'.format(key): 'Index', 'CorrOrNo_{}'.format(key): 'CorrOrNo',
                                            'ActCls_{}'.format(key): 'Class'})
        
        # Select season
        hmap_key['Index'] = pd.to_numeric(hmap_key['Index'])
        hmap_szn = pd.merge(idx_all, hmap_key, on='Index')
        all_all = hmap_szn[hmap_szn['Season'] == sel_szn].reset_index(drop=True)
        
        # Get all season and all seaosn-x-class
        if key == 'All':
            
            # Just season
            M_cts_all = all_all.groupby([typo])[typo].count()
            
            # season and class
            class_all = all_all[all_all['Class'] == sel_cls].reset_index(drop=True)
            M_cts_all_x = class_all.groupby([typo])[typo].count()
        
        # Get MJO counts
        else:
            class_all = all_all[all_all['Class'] == sel_cls].reset_index(drop=True)
            M_cts = class_all.groupby([typo])[typo].count()
    
    # Divide    
    M_out = M_cts / M_cts_all
    M_out_x = M_cts / M_cts_all_x
    
    plt.plot(np.arange(int(lpn/9), lpn), M_out_x[int(lpn/9):], linewidth=4)
    plt.title('MJO Frequency of 30% Most Confident\n{} Class {} Samples'.format(sel_szn, sel_cls), fontsize=18)
    plt.xlabel('MJO Phase', fontsize=14)
    plt.ylabel('Frequency\nRel to all {} Class {} Samples'.format(sel_szn, sel_cls), fontsize=14)
    #plt.savefig('{}_{}{}_{}_k{}_{}_relszncls'.format(typo, sel_szn, sel_cls, Reg, key, Pred), dpi=300)
    plt.show()
    
    plt.plot(np.arange(int(lpn/9), lpn), M_out[int(lpn/9):], linewidth=4)
    plt.title('MJO Frequency of 30% Most Confident\n{} Class {} Samples'.format(sel_szn, sel_cls), fontsize=18)
    plt.xlabel('MJO Phase', fontsize=14)
    plt.ylabel('Frequency\nRel to all {} Samples'.format(sel_szn), fontsize=14)
    #plt.savefig('{}_{}{}_{}_k{}_{}_relszn'.format(typo, sel_szn, sel_cls, Reg, key, Pred), dpi=300)
    plt.show()
    
# Based on season and class, create two MJO lineplots (rel szn and szn/cls)
def E_hist(TV_hmap_mean, idx_all, Reg, sel_szn, sel_cls, typo):
    
    # 3 different histograms
    hist_list = []
    hlabels = ['{}'.format(sel_szn), '{} Class {}'.format(sel_szn, sel_cls), 
               '30% Most Conf\n{} Class {}'.format(sel_szn, sel_cls)]
    hwids = [1., .8, .6]
    
    # how many in lineplot
    if typo == 'M_Phase':
        lpn = 9
    elif typo == 'E_Phase':
        lpn = 3
    else:
        print('oops, typo doesn\'t exist')
    
    keys = ['All', '30']
    for key in keys:
    
        # Get just columns of specific accuracy level
        hmap_key = TV_hmap_mean[['Index_{}'.format(key), 'CorrOrNo_{}'.format(key), 'ActCls_{}'.format(key)]].copy()   
        
        
        # Drop key aspect of column names
        hmap_key = hmap_key.rename(columns={'Index_{}'.format(key): 'Index', 'CorrOrNo_{}'.format(key): 'CorrOrNo',
                                            'ActCls_{}'.format(key): 'Class'})
        
        # Select season
        hmap_key['Index'] = pd.to_numeric(hmap_key['Index'])
        hmap_szn = pd.merge(idx_all, hmap_key, on='Index')
        all_all = hmap_szn[hmap_szn['Season'] == sel_szn].reset_index(drop=True)
        
        # Get season and season/class
        if key == 'All':
            
            # season
            M_cts_szn = all_all.groupby([typo])[typo].count()
            hist_list += [M_cts_szn]
            
            # season and class
            class_all = all_all[all_all['Class'] == sel_cls].reset_index(drop=True)
            M_cts_cls = class_all.groupby([typo])[typo].count()
            hist_list += [M_cts_cls]
        
        # Get 30% season and class
        else:
            class_all = all_all[all_all['Class'] == sel_cls].reset_index(drop=True)
            M_cts_30 = class_all.groupby([typo])[typo].count()
            hist_list += [M_cts_30]
    
    # Plot
    h0 = []
    h1 = []
    for hh in range(len(hist_list)):
        bar = plt.bar(np.arange(1, 4), hist_list[hh], width=hwids[hh], edgecolor='white', label=hlabels[hh])
        
        # Text above bars
        if hh == 0:
            for bb in bar:
                h0 += [bb.get_height()]
                
        elif hh == 1:
            for bc, bb in enumerate(bar):
                height = bb.get_height()
                h1 += [height]
                plt.text(x=bb.get_x() + bb.get_width() / 5, y=height+.6, s='{}'.format(round(height/h0[bc], 2)), ha='center',
                         color='paleturquoise', fontsize=10, path_effects=[pe.withStroke(linewidth=3, foreground='black')])
        else:
            for bc, bb in enumerate(bar):
                height = bb.get_height()
                plt.text(x=bb.get_x() + bb.get_width() / 1.6, y=height+.6, s='{}'.format(round(height/h0[bc], 2)), ha='center',
                         color='paleturquoise', fontsize=9, path_effects=[pe.withStroke(linewidth=2, foreground='black')])
                plt.text(x=bb.get_x() + bb.get_width() / 1, y=height+.6, s='{}'.format(round(height/h1[bc], 2)), ha='center',
                         color='yellow', fontsize=9, path_effects=[pe.withStroke(linewidth=2, foreground='black')])
                
    
    plt.title('ENSO Count and Frequency\n30% Most Conf Samples - {} Class {}'.format(sel_szn, sel_cls), fontsize=18)
    plt.xlabel('ENSO Phase', fontsize=14)
    plt.ylabel('Count', fontsize=14)
    plt.xticks([1, 2, 3], ['La Niña', 'Neutral', 'El Niño'])
    #plt.savefig('{}_hist_{}{}_{}_k{}_{}'.format(typo, sel_szn, sel_cls, Reg, key, Pred), dpi=300)
    plt.legend()
    plt.show()
