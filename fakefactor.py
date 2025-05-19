#!/usr/bin/env python3
import os
import uproot
import numpy as np
import awkward as ak
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.colors as mcolors
import mplhep as hep

from pathlib import Path

plt.style.use(hep.style.ROOT)

def output_names(path_to_files):
    path_to_script          = Path(__file__).parent.resolve()
    absolute_path_to_files  = Path(path_to_script / path_to_files).resolve()
    data_dir = Path(absolute_path_to_files).name
    output_name = f"{data_dir}"
    relative_path_to_plot_dir  = (f"plots/{output_name}")
    plot_dir = Path(path_to_script / relative_path_to_plot_dir).resolve()
    plot_dir.mkdir(parents=True, exist_ok=True)
    results_txt_file_name = (f"{output_name}_fakefactor.txt")
    results_txt_file = path_to_script / results_txt_file_name
    return (plot_dir, results_txt_file)

def file_listing(absolute_path_to_files, prefix):
    root_files = [file for file in os.listdir(absolute_path_to_files)
                  if os.path.isfile(os.path.join(absolute_path_to_files, file))
                  and file.startswith(prefix)]
    return root_files

def load_root_files(path_to_file, root_filenames, treename="mini"):
    data_list = []
    for filename in root_filenames:
        fullpath = Path(path_to_file) / filename
        with uproot.open(fullpath) as file:
            tree = file[treename]
            branches = tree.arrays()
            data_list.append(branches)
    merged_data = ak.concatenate(data_list, axis=0)
    return merged_data

def kinematic_variables(data):
    branches = {}
    branches["channelNumber"]               =            data["channelNumber"]   
    branches["mcWeight"]                    =            data["mcWeight"]
    branches["XSection"]                    =            data["XSection"]
    branches["SumWeights"]                  =            data["SumWeights"]
    branches["scaleFactor_PILEUP"]          =            data["scaleFactor_PILEUP"]
    branches["scaleFactor_ELE"]             =            data["scaleFactor_ELE"]
    branches["scaleFactor_MUON"]            =            data["scaleFactor_MUON"]
    branches["scaleFactor_PHOTON"]          =            data["scaleFactor_PHOTON"]
    branches["scaleFactor_TAU"]             =            data["scaleFactor_TAU"]           
    branches["scaleFactor_BTAG"]            =            data["scaleFactor_BTAG"]
    branches["scaleFactor_LepTRIGGER"]      =            data["scaleFactor_LepTRIGGER"]    
    branches["scaleFactor_PhotonTRIGGER"]   =            data["scaleFactor_PhotonTRIGGER"]
    branches["lep_truthMatched"]            = ak.flatten(data["lep_truthMatched"])
    branches["lep_trigMatched"]             = ak.flatten(data["lep_trigMatched"])
    branches["lep_pt"]                      = ak.flatten(data["lep_pt"] / 1000)
    branches["lep_eta"]                     = ak.flatten(data["lep_eta"])
    branches["lep_phi"]                     = ak.flatten(data["lep_phi"])
    branches["lep_z0"]                      = ak.flatten(data["lep_z0"])
    branches["lep_type"]                    = ak.flatten(data["lep_type"])
    branches["lep_isTightID"]               = ak.flatten(data["lep_isTightID"])
    branches["lep_ptcone30"]                = ak.flatten(data["lep_ptcone30"] / 1000)
    branches["lep_etcone20"]                = ak.flatten(data["lep_etcone20"] / 1000)
    branches["lep_trackd0pvunbiased"]       = ak.flatten(data["lep_trackd0pvunbiased"])
    branches["lep_tracksigd0pvunbiased"]    = ak.flatten(data["lep_tracksigd0pvunbiased"])
    branches["met_et"]                      =            data["met_et"] / 1000
    branches["met_phi"]                     =            data["met_phi"]
    branches["jet_n"]                       =            data["jet_n"]
    branches["jet_pt"]                      =            data["jet_pt"]
    branches["jet_eta"]                     =            data["jet_eta"]
    branches["jet_phi"]                     =            data["jet_phi"]
    return branches

def extract_branches(path_to_files, prefix):
    path_to_script          = Path(__file__).parent.resolve()
    absolute_path_to_files  = Path(path_to_script / path_to_files).resolve()
    root_files = file_listing(absolute_path_to_files, prefix)
    data = load_root_files(absolute_path_to_files, root_files)
    print(f"{prefix} files in {absolute_path_to_files} loaded")
    data_branches = kinematic_variables(data)
    return data_branches

def normalization(data, luminosity_fb=10.06, luminosity_error_fb=0.37):
    luminosity_pb               = luminosity_fb * 1000
    luminosity_error_pb         = luminosity_error_fb * 1000
    mcWeight                    = data["mcWeight"]
    XSection                    = data["XSection"]
    SumWeights                  = data["SumWeights"]
    scaleFactor_PILEUP          = data["scaleFactor_PILEUP"]
    scaleFactor_ELE             = data["scaleFactor_ELE"]
    scaleFactor_MUON            = data["scaleFactor_MUON"]
    scaleFactor_PHOTON          = data["scaleFactor_PHOTON"]
    scaleFactor_TAU             = data["scaleFactor_TAU"]           
    scaleFactor_BTAG            = data["scaleFactor_BTAG"]
    scaleFactor_LepTRIGGER      = data["scaleFactor_LepTRIGGER"]    
    scaleFactor_PhotonTRIGGER   = data["scaleFactor_PhotonTRIGGER"]
    scaleFactor = scaleFactor_PILEUP * scaleFactor_ELE * scaleFactor_MUON * scaleFactor_PHOTON * scaleFactor_TAU * scaleFactor_BTAG * scaleFactor_LepTRIGGER * scaleFactor_PhotonTRIGGER
    norm_weight = (mcWeight * XSection * luminosity_pb*scaleFactor)/SumWeights
    norm_weight_error = ((mcWeight * XSection * scaleFactor)/SumWeights)*luminosity_error_pb
    return norm_weight, norm_weight_error 

def electron_selections(data):
    lep_theta   = 2 * np.arctan(np.exp(-(data["lep_eta"])))
    lep_type    = (data["lep_type"] == 11)
    lep_pt      = (data["lep_pt"] > 25)
    lep_eta     = ((abs(data["lep_eta"]) < 1.37) | ((abs(data["lep_eta"]) > 1.52) & (abs(data["lep_eta"]) < 2.47)))
    lep_sigd0   = (data["lep_tracksigd0pvunbiased"] < 5)
    lep_z0sinth = (abs(data["lep_z0"] * np.sin(lep_theta)) < 0.5)
    mask = lep_type & lep_pt & lep_eta & lep_sigd0 & lep_z0sinth
    return mask

def muon_selections(data):
    lep_theta   = 2 * np.arctan(np.exp(-(data["lep_eta"])))
    lep_type    = (data["lep_type"] == 13)
    lep_pt      = (data["lep_pt"] > 25)
    lep_eta     = (abs(data["lep_eta"]) < 2.5)
    lep_sigd0   = (data["lep_tracksigd0pvunbiased"] < 3)
    lep_z0sinth = (abs(data["lep_z0"] * np.sin(lep_theta)) < 0.5)
    mask = lep_type & lep_pt & lep_eta & lep_sigd0 & lep_z0sinth
    return mask

def mc_bkg_channels(mc, mc_branch, mc_weight, mc_mask):
    mc_truth = (mc["lep_truthMatched"] == 1)
    selections = [(mc["channelNumber"] >= 363356) & (mc["channelNumber"] <= 363493),
                  (mc["channelNumber"] >= 410011) & (mc["channelNumber"] <= 410026),
                  (mc["channelNumber"] == 410000),
                  (mc["channelNumber"] >= 364100) & (mc["channelNumber"] <= 364141),
                  (mc["channelNumber"] >= 364156) & (mc["channelNumber"] <= 364197)]
    mc_labels  = ["Diboson production", "Single t production", r"$t\bar{t}$ production", "Z production + jets", "W production + jets"]
    mc_colors  = ["mediumslateblue", "lightsteelblue", "lightgreen", "lightsalmon", "khaki"]
    mc_values  = []
    mc_weights = []
    for selection in selections:
        mc_values.append(mc_branch[mc_mask & mc_truth & selection])
        mc_weights.append(mc_weight[mc_mask & mc_truth & selection])
    mc_channels = [mc_values, mc_weights, mc_labels, mc_colors]
    return mc_channels

def branch_hist(data, mc, branch, lep_type, lep_type_mask, mc_lep_type_mask, plot_dir) -> None:
    data_branch = data[branch[0]]
    mc_branch = mc[branch[0]]
    mc_weight, mc_error = normalization(mc)
    if (branch[0] == "lep_eta"):
        min_bin = -2.5
        max_bin = 2.5        
    else:
        min_bin = min(ak.min(data_branch), ak.min(mc_branch))
        max_bin = 1000
    if branch[3] == "log":
        bins = np.logspace(np.log10(min_bin), np.log10(max_bin), 50+1)
    else:
        bins = np.linspace(min_bin, max_bin, 50 + 1)
    mc_channels = mc_bkg_channels(mc, mc_branch, mc_weight, mc_lep_type_mask)
    fig, ax = plt.subplots()
    ax.hist(data_branch[lep_type_mask], bins= bins, histtype= "step", color= "black", label= "Data")
    ax.hist(mc_channels[0], bins= bins, weights= mc_channels[1], histtype="stepfilled", label=mc_channels[2], color=mc_channels[3], stacked=True)
    ax.set_ylabel("Number of events")
    ax.set_xlabel(branch[1])
    ax.set_yscale(branch[2])
    ax.set_xscale(branch[3])
    bottom, top = ax.get_ylim()
    if branch[2] == "log":
        mergin = 100
    else:
        mergin = 1.5
        bottom = 0
        formatter = ticker.ScalarFormatter(useMathText=True)
        formatter.set_powerlimits((-3, 3))
        ax.yaxis.set_major_formatter(formatter)
        ax.ticklabel_format(axis='y', style='sci', scilimits=(-3, 3))
    ax.set_ylim(bottom= bottom, top=top * mergin)
    ax.legend(fancybox=False, facecolor="white", loc="upper right", handlelength=0.75, handletextpad=0.4)
    hep.atlas.label(loc=2, ax=ax, data=True, label="Work in Progress\n", rlabel=" ", fontname="Latin Modern sans", fontsize=20)
    ax.text(0.05, 0.86, r"$\sqrt{s}=13 TeV, 10.06 fb^{-1}$", transform=ax.transAxes, fontsize=21, fontname="Latin Modern Sans", ha="left", va="top")
    fig.tight_layout()
    fig.savefig(f"{plot_dir}/{branch[0]}_{lep_type}.png")
    plt.close()

def lep_iso_hist(data, mc, lep_type, lep_type_mask, mc_lep_type_mask, plot_dir) -> None:
    data_lep_iso  = (data["lep_ptcone30"] / data["lep_pt"])
    mc_lep_iso    = (mc["lep_ptcone30"] / mc["lep_pt"])
    mc_weight, mc_error = normalization(mc)
    min_bin = min(ak.min(data_lep_iso), ak.min(mc_lep_iso))
    bins = np.linspace(min_bin, 1, 50 + 1)
    mc_channels = mc_bkg_channels(mc, mc_lep_iso, mc_weight, mc_lep_type_mask)
    fig, ax = plt.subplots()
    ax.hist(data_lep_iso[lep_type_mask], bins= bins, histtype= "step", color= "black", label= "Data")
    ax.hist(mc_channels[0], bins= bins, weights= mc_channels[1], histtype="stepfilled", label=mc_channels[2], color=mc_channels[3], stacked=True)
    ax.set_ylabel("Number of events")
    ax.set_xlabel(r"$p_{\mathrm{T}}^{cone30}/p_{\mathrm{T}}$")
    ax.set_yscale("log")
    bottom, top = ax.get_ylim()
    ax.set_ylim(top=top * 100)
    ax.legend(fancybox=False, facecolor="white", loc="upper right", handlelength=0.75, handletextpad=0.4)
    hep.atlas.label(loc=2, ax=ax, data=True, label="Work in Progress\n", rlabel=" ", fontname="Latin Modern sans", fontsize=20)
    ax.text(0.05, 0.86, r"$\sqrt{s}=13 TeV, 10.06 fb^{-1}$", transform=ax.transAxes, fontsize=21, fontname="Latin Modern Sans", ha="left", va="top")
    fig.tight_layout()
    fig.savefig(f"{plot_dir}/lep_iso_{lep_type}.png")
    plt.close()

def tight_loose_regions(data, lep_iso_tight_cut, lep_iso_loose_cut):
    lep_iso = (data["lep_ptcone30"] / data["lep_pt"])
    tight_iso = (lep_iso < lep_iso_tight_cut)
    loose_iso = (lep_iso > lep_iso_tight_cut) & (lep_iso < lep_iso_loose_cut)
    tight_ID = ((data["lep_isTightID"]) == 1)
    loose_ID = ((data["lep_isTightID"]) == 0)
    tight = tight_iso & tight_ID
    loose = loose_iso | loose_ID
    return (tight, loose)

def fake_enriched_electrons(data):
    jet_n   = (data["jet_n"] >= 2)
    mask    = jet_n
    return mask

def fake_enriched_muons(data):
    jet_n   = (data["jet_n"] == 1)
    jet_pt  = ak.any(data["jet_pt"] > 35)
    jet_phi = ak.any(np.abs(np.arctan2(np.sin(data["jet_phi"]-data["lep_phi"]), np.cos(data["jet_phi"]-data["lep_phi"])))>2.5, axis=1)
    met_et  = (data["met_et"] < 40) # NO
    mask    = jet_n & jet_pt & jet_phi
    return mask

def lep_region_def(mask, tight, loose, fake_enriched):
    region_A = mask & tight & ~fake_enriched
    region_B = mask & loose & ~fake_enriched
    region_C = mask & tight & fake_enriched
    region_D = mask & loose & fake_enriched
    regions  = [region_A, region_B, region_C, region_D]
    return regions

def lep_region_counts(regions, weight):
    counts = []
    for region in regions:
        N = ak.sum(weight[region])
        counts.append(N)
    return counts

def lep_region_counts_with_uncertainties(regions, weight, error):
    counts = []
    count_errors = []
    for region in regions:
        N = ak.sum(weight[region])
        S = np.sqrt(ak.sum((error[region])**2))
        counts.append(N)
        count_errors.append(S)
    return counts, count_errors

def lep_selections(data, mask, lep_iso_tight_cut, lep_iso_loose_cut, fake_enriched):
    tight, loose = tight_loose_regions(data, lep_iso_tight_cut, lep_iso_loose_cut)
    regions = lep_region_def(mask, tight, loose, fake_enriched)
    return regions

def lep_selection_counts(data, weight, mask, lep_iso_tight_cut, lep_iso_loose_cut, fake_enriched):
    regions = lep_selections(data, mask, lep_iso_tight_cut, lep_iso_loose_cut, fake_enriched)
    counts  = lep_region_counts(regions, weight)
    return counts

def lep_selection_counts_with_uncertainties(mc, weight, error, mask, lep_iso_tight_cut, lep_iso_loose_cut, fake_enriched):
    regions = lep_selections(mc, mask, lep_iso_tight_cut, lep_iso_loose_cut, fake_enriched)
    counts, count_errors = lep_region_counts_with_uncertainties(regions, weight, error)
    return counts, count_errors

def branch_region_hist(data, mc, branch, lep_type, lep_type_mask, mc_lep_type_mask, lep_iso_tight_cut, lep_iso_loose_cut, fake_enriched, fake_enriched_mc, plot_dir) -> None:
    data_branch = data[branch[0]]
    mc_branch = mc[branch[0]]
    mc_weight, mc_error = normalization(mc)
    if (branch[0] == "lep_eta"):
        min_bin = -2.5
        max_bin = 2.5        
    else:
        min_bin = min(ak.min(data_branch), ak.min(mc_branch))
        max_bin = 1000
    if branch[3] == "log":
        bins = np.logspace(np.log10(min_bin), np.log10(max_bin), 50+1)
    else:
        bins = np.linspace(min_bin, max_bin, 50 + 1)
    regions = lep_selections(data, lep_type_mask, lep_iso_tight_cut, lep_iso_loose_cut, fake_enriched)
    mc_regions = lep_selections(mc, mc_lep_type_mask, lep_iso_tight_cut, lep_iso_loose_cut, fake_enriched_mc)
    for i in range(4):
        mc_channels = mc_bkg_channels(mc, mc_branch, mc_weight, mc_lep_type_mask & mc_regions[i])
        fig, ax = plt.subplots()
        ax.hist(data_branch[lep_type_mask & regions[i]], bins= bins, histtype= "step", color= "black", label= "Data")
        ax.hist(mc_channels[0], bins= bins, weights= mc_channels[1], histtype="stepfilled", label=mc_channels[2], color=mc_channels[3], stacked=True)
        ax.set_ylabel("Number of events")
        ax.set_xlabel(branch[1])
        ax.set_yscale(branch[2])
        ax.set_xscale(branch[3])
        bottom, top = ax.get_ylim()
        if branch[2] == "log":
            mergin = 100
        else:
            mergin = 1.5
            bottom = 0
            formatter = ticker.ScalarFormatter(useMathText=True)
            formatter.set_powerlimits((-3, 3))
            ax.yaxis.set_major_formatter(formatter)
            ax.ticklabel_format(axis='y', style='sci', scilimits=(-3, 3))
        ax.set_ylim(bottom= bottom, top=top * mergin)
        ax.legend(fancybox=False, facecolor="white", loc="upper right", handlelength=0.75, handletextpad=0.4)
        hep.atlas.label(loc=2, ax=ax, data=True, label="Work in Progress\n", rlabel=" ", fontname="Latin Modern sans", fontsize=20)
        ax.text(0.05, 0.86, r"$\sqrt{s}=13 TeV, 10.06 fb^{-1}$", transform=ax.transAxes, fontsize=21, fontname="Latin Modern Sans", ha="left", va="top")
        fig.tight_layout()
        fig.savefig(f"{plot_dir}/{branch[0]}_region_{i}_{lep_type}.png")
        plt.close()

def frac_err(num, den, num_err, den_err):
    with np.errstate(divide="ignore", invalid="ignore"):
        err = np.sqrt((1.0 / den**2) * (num_err**2) + ((-num) / (den**2))**2 * (den_err**2))
        err = np.nan_to_num(err, nan=0.0, posinf=0.0, neginf=0.0)
    return err

def mult_err(var1, var2, var1_err, var2_err):
    with np.errstate(divide="ignore", invalid="ignore"):
        err = np.sqrt((var2**2) * (var1_err**2) + (var1**2) * (var2_err**2))
        err = np.nan_to_num(err, nan=0.0, posinf=0.0, neginf=0.0)
    return err

def ff_uncertainty(N_tight, N_tight_MC, N_loose, N_loose_MC, N_tight_error, N_loose_error, N_tight_MC_error, N_loose_MC_error):
    with np.errstate(divide="ignore", invalid="ignore"):
        FF_error = np.sqrt((((N_tight_error**2)+(N_tight_MC_error**2))/((N_loose-N_loose_MC)**2))+(((N_tight-N_tight_MC)**2)*((N_loose_error**2)+(N_loose_MC_error)**2)/((N_loose-N_loose_MC)**4)))
        FF_error = np.nan_to_num(FF_error, nan=0.0, posinf=0.0, neginf=0.0)
    return FF_error

def abcd_method(data, mc, lep_type_mask, mc_lep_type_mask, lep_iso_tight_cut, lep_iso_loose_cut, fake_enriched, fake_enriched_mc):
    mc_truth = (mc["lep_truthMatched"] == 1)
    mc_weight, mc_error = normalization(mc)#!
    counts = lep_selection_counts(data, data["SumWeights"], lep_type_mask, lep_iso_tight_cut, lep_iso_loose_cut, fake_enriched)
    mc_counts, mc_count_errors = lep_selection_counts_with_uncertainties(mc, mc_weight, mc_error, (mc_lep_type_mask & mc_truth), lep_iso_tight_cut, lep_iso_loose_cut, fake_enriched_mc)
    FF = (counts[2]-mc_counts[2])/(counts[3]-mc_counts[3]) if (counts[3]-mc_counts[3]) != 0 else 0
    FF_error = ff_uncertainty(counts[2], mc_counts[2], counts[3], mc_counts[3], np.sqrt(counts[2]), np.sqrt(counts[3]), mc_count_errors[2], mc_count_errors[3]) #!
    fakes_in_signal = FF * (counts[1]-mc_counts[1])
    fakes_error = mult_err(FF, counts[1]-mc_counts[1], FF_error, ((np.sqrt(counts[1])**2)+(mc_count_errors[1]**2))) #!
    reals_in_signal = counts[0] - mc_counts[0] - fakes_in_signal
    return (FF, FF_error, fakes_in_signal, fakes_error, reals_in_signal)

def counts_per_pt_bin(data, weight, pt_bin_min, pt_bin_max, mask, lep_iso_tight_cut, lep_iso_loose_cut, fake_enriched):
    pt_bin = (data["lep_pt"] > pt_bin_min) & (data["lep_pt"] < pt_bin_max)
    counts = lep_selection_counts(data, weight, mask & pt_bin, lep_iso_tight_cut, lep_iso_loose_cut, fake_enriched)
    return counts

def counts_per_pt_bin_with_unceratinties(mc, mc_weight, mc_error, pt_bin_min, pt_bin_max, mask, lep_iso_tight_cut, lep_iso_loose_cut, fake_enriched):
    pt_bin = (mc["lep_pt"] > pt_bin_min) & (mc["lep_pt"] < pt_bin_max)
    mc_counts, mc_count_errors = lep_selection_counts_with_uncertainties(mc, mc_weight, mc_error, mask & pt_bin, lep_iso_tight_cut, lep_iso_loose_cut, fake_enriched)
    return mc_counts, mc_count_errors

def fake_factor(data, mc, lep_type, pt_max, lep_type_mask, mc_lep_type_mask, lep_iso_tight_cut, lep_iso_loose_cut, fake_enriched, fake_enriched_mc, plot_dir, n_bins):
    mc_truth = (mc["lep_truthMatched"] == 1)
    mc_weight, mc_error = normalization(mc)#!
    pt_bins     = np.logspace(np.log10(40), np.log10(pt_max), n_bins + 1)
    bin_centers = np.sqrt(pt_bins[:-1] * pt_bins[1:])
    eta_regions =  [[(abs(data["lep_eta"]) > 0   ) & (abs(data["lep_eta"]) < 0.7 ), (abs(mc["lep_eta"]) > 0   ) & (abs(mc["lep_eta"]) < 0.7 ), r"$|\eta|$ < 0.7"        ], 
                    [(abs(data["lep_eta"]) > 0.7 ) & (abs(data["lep_eta"]) < 1.37), (abs(mc["lep_eta"]) > 0.7 ) & (abs(mc["lep_eta"]) < 1.37), r"0.7 < $|\eta|$ < 1.37" ], 
                    [(abs(data["lep_eta"]) > 1.37) & (abs(data["lep_eta"]) < 1.52), (abs(mc["lep_eta"]) > 1.37) & (abs(mc["lep_eta"]) < 1.52), r"1.37 < $|\eta|$ < 1.52"], 
                    [(abs(data["lep_eta"]) > 1.52) & (abs(data["lep_eta"]) < 2.01), (abs(mc["lep_eta"]) > 1.52) & (abs(mc["lep_eta"]) < 2.01), r"1.52 < $|\eta|$ < 2.01"], 
                    [(abs(data["lep_eta"]) > 2.01) & (abs(data["lep_eta"]) < 2.47), (abs(mc["lep_eta"]) > 2.01) & (abs(mc["lep_eta"]) < 2.47), r"2.01 < $|\eta|$ < 2.47"]]
    if lep_type == "el":
        eta_regions.pop(2)
    j = 0
    fake_factors = []
    fake_factor_errors = []
    data_points = []
    mc_predictions = []
    mc_prediction_errors = []
    estimated_fakes = []
    estimated_fake_errors = []
    for eta_region in eta_regions:
        sub_fake_factors = []
        sub_fake_factor_errors = []
        sub_data_points = []
        sub_mc_predictions = []
        sub_mc_prediction_errors = []
        sub_estimated_fakes = []
        sub_estimated_fake_errors = []
        for i in range(n_bins):
            counts = counts_per_pt_bin(data, data["SumWeights"], pt_bins[i], pt_bins[i+1], lep_type_mask & eta_region[0], lep_iso_tight_cut, lep_iso_loose_cut, fake_enriched)
            mc_counts, mc_count_errors = counts_per_pt_bin_with_unceratinties(mc, mc_weight, mc_error, pt_bins[i], pt_bins[i+1], mc_lep_type_mask & eta_region[1] & mc_truth, lep_iso_tight_cut, lep_iso_loose_cut, fake_enriched_mc)
            FF = (counts[2]-mc_counts[2])/(counts[3]-mc_counts[3]) if (counts[3]-mc_counts[3]) != 0 else 0
            if FF < 0:
                FF = 0
            FF_error = ff_uncertainty(counts[2], mc_counts[2], counts[3], mc_counts[3], np.sqrt(counts[2]), np.sqrt(counts[3]), mc_count_errors[2], mc_count_errors[3]) #!
            est_fakes = FF * (counts[1] - mc_counts[1])
            est_fake_errors = mult_err(FF, counts[1]-mc_counts[1], FF_error, ((np.sqrt(counts[1])**2)+(mc_count_errors[1]**2))) #!
            sub_fake_factors.append(FF)
            sub_fake_factor_errors.append(FF_error)
            sub_data_points.append(counts[0])
            sub_mc_predictions.append(mc_counts[0])
            sub_mc_prediction_errors.append(mc_count_errors[0])
            sub_estimated_fakes.append(est_fakes)
            sub_estimated_fake_errors.append(est_fake_errors)
        fake_factors.append(sub_fake_factors)
        fake_factor_errors.append(sub_fake_factor_errors)
        data_points.append(sub_data_points)
        mc_predictions.append(sub_mc_predictions)
        mc_prediction_errors.append(sub_mc_prediction_errors)
        estimated_fakes.append(sub_estimated_fakes)
        estimated_fake_errors.append(sub_estimated_fake_errors)
        sub_fake_factors = np.array(sub_fake_factors)
        sub_fake_factor_errors = np.array(sub_fake_factor_errors)
        fig, ax = plt.subplots()
        ax.errorbar(bin_centers[sub_fake_factors != 0], sub_fake_factors[sub_fake_factors != 0], xerr=((pt_bins[1:] - pt_bins[:-1]) / 2.0)[sub_fake_factors != 0], yerr=sub_fake_factor_errors[sub_fake_factors != 0], fmt= "o", color="black")
        ax.errorbar(bin_centers[sub_fake_factors == 0], sub_fake_factors[sub_fake_factors == 0], xerr=((pt_bins[1:] - pt_bins[:-1]) / 2.0)[sub_fake_factors == 0], yerr=sub_fake_factor_errors[sub_fake_factors == 0], fmt= "rx")
        ax.set_xlabel(r"$p_{\mathrm{T}}$ [GeV]")
        ax.set_ylabel(r"Fake Factor")
        ax.set_xscale("log")
        bottom, top = ax.get_ylim()
        if top > 30:
            top = 30
        ax.set_ylim(bottom=0, top=top * 1.5)
        hep.atlas.label(loc=2, ax=ax, data=True, label="Work in Progress\n", rlabel=" ", fontname="Latin Modern sans", fontsize=20)
        ax.text(0.05, 0.86, r"$\sqrt{s}=13 TeV, 10.06 fb^{-1}$", transform=ax.transAxes, fontsize=21, fontname="Latin Modern Sans", ha="left", va="top")
        ax.text(0.05, 0.8, eta_region[2], transform=ax.transAxes, fontsize=21, fontname="Latin Modern Sans", ha="left", va="top")
        fig.tight_layout()
        fig.savefig(f"{plot_dir}/ff_pt_{lep_type}_no_lim_bin{j}_.png")
        plt.close()
        j += 1
    data_tgh = np.array([sum(column) for column in zip(*data_points)])
    real_bkg = np.array([sum(column) for column in zip(*mc_predictions)])
    fake_bkg = np.array([sum(column) for column in zip(*estimated_fakes)])
    real_bkg_errors = np.array([np.sqrt(sum(x**2 for x in column)) for column in zip(*mc_prediction_errors)])
    fake_bkg_errors = np.array([np.sqrt(sum(x**2 for x in column)) for column in zip(*estimated_fake_errors)])
    bkg = real_bkg + fake_bkg
    bkg_errors = np.sqrt((real_bkg_errors**2)+(fake_bkg_errors**2))
    return (fake_factors, fake_factor_errors, data_tgh, bkg, bkg_errors)

def closure_test(mc, data_points, bkg, bkg_errors, lep_type, pt_max, mc_lep_type_mask, lep_iso_tight_cut, lep_iso_loose_cut, fake_enriched_mc, plot_dir, n_bins) -> None:
    pt_bins     = np.logspace(np.log10(40), np.log10(pt_max), n_bins + 1)
    bin_centers = np.sqrt(pt_bins[:-1] * pt_bins[1:])
    branch = ["lep_pt",r"$p_{\mathrm{T}} [GeV]$"]
    mc_branch = mc[branch[0]]
    mc_weight, mc_errors = normalization(mc)
    mc_regions = lep_selections(mc, mc_lep_type_mask, lep_iso_tight_cut, lep_iso_loose_cut, fake_enriched_mc)
    mc_channels = mc_bkg_channels(mc, mc_branch, mc_weight, mc_lep_type_mask & mc_regions[0])
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = np.divide(data_points, bkg)
        ratio[bkg == 0] = np.nan
    ratio_error = frac_err(data_points, bkg, np.sqrt(data_points), bkg_errors)
    fig, (ax_main, ax_ratio) = plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [3, 1]}, figsize=(10, 12))
    ax_main.fill_between(bin_centers, bkg, step="mid", alpha=1, color="plum", label="Fakes")
    ax_main.hist(mc_channels[0], bins= pt_bins, weights= mc_channels[1], histtype="stepfilled", label=mc_channels[2], color=mc_channels[3], stacked=True)
    ax_main.scatter(bin_centers, data_points, label = "Data", color="black")
    ax_main.set_xlim(40, pt_max)
    ax_main.set_yscale("log")
    ax_main.set_xscale("log")
    ax_main.set_ylabel("Number of events")
    bottom, top = ax_main.get_ylim()
    ax_main.set_ylim(top=top * 100)
    ax_main.legend(fancybox=False, facecolor="white", loc="upper right", handlelength=0.75, handletextpad=0.4)
    hep.atlas.label(loc=2, ax=ax_main, data=True, label="Work in Progress\n", rlabel=" ", fontname="Latin Modern sans", fontsize=20)
    ax_main.text(0.05, 0.86, r"$\sqrt{s}=13 TeV, 10.06 fb^{-1}$", transform=ax_main.transAxes, fontsize=21, fontname="Latin Modern Sans", ha="left", va="top")
    ax_ratio.errorbar(bin_centers, ratio, xerr=(pt_bins[1:] - pt_bins[:-1]) / 2.0, yerr=ratio_error, color= "black", fmt= 'o')
    ax_ratio.set_ylabel("Data / Pred.")
    ax_ratio.set_xlabel(r"$p_{\mathrm{T}}$ [GeV]")
    ax_ratio.axhline(1.0, color='red', linestyle='--', linewidth=1)
    ax_ratio.grid(True, linestyle=':', linewidth=0.5)
    fig.tight_layout()
    plt.subplots_adjust(hspace=0.03)
    fig.savefig(f"{plot_dir}/closure_test_{lep_type}.png")
    plt.close()

def run(data, mc, lep_type, pt_max, n_bins, lep_iso_tight_cut, lep_iso_loose_cut, plot_dir):
    if lep_type == "el":
        lep_type_mask = electron_selections(data)
        mc_lep_type_mask = electron_selections(mc)
        fake_enriched = fake_enriched_electrons(data)
        fake_enriched_mc = fake_enriched_electrons(mc)
    elif lep_type == "mu":
        lep_type_mask = muon_selections(data)
        mc_lep_type_mask = muon_selections(mc)
        fake_enriched = fake_enriched_muons(data)
        fake_enriched_mc = fake_enriched_muons(mc)
    branches_to_plot = [["lep_pt",r"$p_{\mathrm{T}} [GeV]$", "log", "log"], ["lep_eta",r"$\eta$", "linear", "linear"], ["met_et",r"$E_{\mathrm{T}}^{miss}$ [GeV]", "log", "linear"]]
    for branch in branches_to_plot:
        branch_hist(data, mc, branch, lep_type, lep_type_mask, mc_lep_type_mask, plot_dir)
        branch_region_hist(data, mc, branch, lep_type, lep_type_mask, mc_lep_type_mask, lep_iso_tight_cut, lep_iso_loose_cut, fake_enriched, fake_enriched_mc, plot_dir)
    lep_iso_hist(data, mc, lep_type, lep_type_mask, mc_lep_type_mask, plot_dir)
    FF, FF_error, fakes_in_signal, fakes_error, reals_in_signal = abcd_method(data, mc, lep_type_mask, mc_lep_type_mask, lep_iso_tight_cut, lep_iso_loose_cut, fake_enriched, fake_enriched_mc)
    fake_factors, fake_factor_errors, data_points, real_bkg, fake_bkg = fake_factor(data, mc, lep_type, pt_max, lep_type_mask, mc_lep_type_mask, lep_iso_tight_cut, lep_iso_loose_cut, fake_enriched, fake_enriched_mc, plot_dir, n_bins)
    closure_test(mc, data_points, real_bkg, fake_bkg, lep_type, pt_max, mc_lep_type_mask, lep_iso_tight_cut, lep_iso_loose_cut, fake_enriched_mc, plot_dir, n_bins)
    return (FF, FF_error, fakes_in_signal, fakes_error, reals_in_signal)

def main(path_to_files, el_iso_tight_cut, el_iso_loose_cut, mu_iso_tight_cut, mu_iso_loose_cut) -> None:
    plot_dir, results_txt_file = output_names(path_to_files)
    data = extract_branches(path_to_files, "data")
    mc   = extract_branches(path_to_files, "mc")
    FF_el, FF_el_error, fake_el_in_signal, fake_el_error, real_el_in_signal = run(data, mc, "el", 1000, 20, el_iso_tight_cut, el_iso_loose_cut, plot_dir)
    FF_mu, FF_mu_error, fake_mu_in_signal, fake_mu_error, real_mu_in_signal = run(data, mc, "mu", 200, 12, mu_iso_tight_cut, mu_iso_loose_cut, plot_dir)
    with open(results_txt_file, "w") as f:
        f.write(f"Electron Fake Factor = {FF_el} +/- {FF_el_error}\n"
                f"Predicted Fake Electron Events in Region A = {round(fake_el_in_signal)} +/- {fake_el_error}\n"
                f"Predicted Real Electron Events in Region A = {round(real_el_in_signal)}\n\n"
                f"Muon Fake Factor = {FF_mu} +/- {FF_mu_error}\n"
                f"Predicted Fake Muon Events in Region A = {round(fake_mu_in_signal)} +/- {fake_mu_error}\n"
                f"Predicted Real Muon Events in Region A = {round(real_mu_in_signal)}\n")
    print(f"ABCD Method's results saved to {results_txt_file}")
    print(f"Plots saved to {plot_dir}")

if __name__ == "__main__":
    path_to_files = "samples-1lep"    # relative path to the root file(s) directory
    el_iso_tight_cut = 0.06
    el_iso_loose_cut = 0.15
    mu_iso_tight_cut = 0.04
    mu_iso_loose_cut = 0.15
    main(path_to_files, el_iso_tight_cut, el_iso_loose_cut, mu_iso_tight_cut, mu_iso_loose_cut)