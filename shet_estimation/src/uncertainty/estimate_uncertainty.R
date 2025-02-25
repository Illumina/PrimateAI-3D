
# Copyright 2024 Illumina Inc, San Diego, CA                                                                                                                            \

#
# This program is licensed under the terms of the Polyform strict license
#
# ***As far as the law allows, the software comes as is, without
# any warranty or condition, and the licensor will not be liable
# to you for any damages arising out of these terms or the use
# or nature of the software, under any kind of legal claim.***
#
# You should have received a copy of the PolyForm Strict License 1.0.0
# along with this program.  If not, see <https://polyformproject.org/licenses/strict/1.0.0>.
#
#
#

#########################
#USAGE
#The command to run the script for estimating depletion metrics and shet scores as well as uncertainty in depletion metrics and shet scores is :
#
#Rscript  /path/to/source/uncertainty/estimate_uncertainty.R  \
#     [in_dir] [in_file] [sel_file] [obsmis_file] [ranknorm_file] \
#     [output_file1] [output_file2]
#
#The input paramters are explained below.
#in_dir: working directory
#in_file : input file
#sel_file : the selection depletion mapping file
#AC_file  : the	rare variant allele count file for every gene
#obsmis_file : the observed missense variant file
#ranknorm_file : the recalibration probability file for PrimateAI-3D scores
#output_file1 : the output file for uncertainty of depletion metrics
#output_file2 :	the output file for uncertainty of selection coefficients
#########################


# Libraries
library(tidyverse)
library(Matrix)
library(matrixStats)
library(collapse)

# Define functions
`%ni%` <- Negate(`%in%`)

# Setup theme
my_theme <-
  BuenColors::pretty_plot(fontsize = 8) +
  BuenColors::L_border() +
  theme(
    plot.background = element_blank(),
    plot.margin = margin(0, 0.1, 0, 0.1, unit = "cm"),
    plot.title = element_text(hjust = 4e-3, margin = margin(b = -12)),
    #legend.position = "none",
    legend.position.inside = c(0.05, 0.98),
    legend.justification = c(1, 1),
    legend.title = element_text(margin = margin(0, 0, 0, 0)),
    legend.background = element_blank(),
    legend.key.size = unit(0.2, "cm")
  )

# Rstudio directory

args <- commandArgs(trailingOnly = TRUE)
in_dir=args[1]
in_file=args[2]
sel_file=args[3]
AC_file=args[4]
obsmis_file=args[5]
ranknorm_file=args[6]
output_file1=args[7]
output_file2=args[8]


# Number of genes to test with
bin_size <- 10
boot_num <- 1000

# Read in pAI-3D smoothed depletion bins
shet_df <- vroom::vroom(in_file, delim = ",")
shet_df <- shet_df %>%
  dplyr::select(genename, transcript, ENST, "bin_11" = cntlof, "exp_mis" = expmis, "exp_lof" = explof, selbin1:selbin10, "selbin11" = selPTV)

# Read in selection depletion map
sdmap_df <- vroom::vroom(sel_file, delim = ",")
sdmap_df <- sdmap_df %>%
  dplyr::filter(AC == 0, selcoeff > 0) %>%
  dplyr::mutate(selection = log10(selcoeff), nvar = 100000 - counts) %>%
  dplyr::mutate(depletion = 1 - nvar / max(nvar)) %>%
  dplyr::select(selection, depletion) %>%
  bind_rows(bind_cols("selection" = 0, "depletion" = 1))

# Read in rare variant ACs
AC_df <- vroom::vroom(AC_file, delim = "\t")

# Read in observed missense variants
obs_df <- vroom::vroom(obsmis_file, delim = ",", col_names = T) %>%
  dplyr::mutate(rankbin = cut(rankpct, seq(0, 1, 1 / bin_size), include.lowest = T))

# Transform raw counts
obs_sum_df <- obs_df %>%
  dplyr::group_by(ENST, rankbin) %>%
  dplyr::count()
obs_sum_wide_df <- obs_sum_df %>%
  pivot_wider(id_cols = "ENST", names_from = "rankbin", values_from = n, values_fill = 0) %>%
  dplyr::select(ENST, levels(obs_sum_df$rankbin)) %>%
  setNames(c('ENST', paste0("bin_", seq(1, bin_size, 1))))
shet_df <- shet_df %>% 
  inner_join(obs_sum_wide_df)

# Read in rank score normalization table
ranknorm_df <- vroom::vroom(ranknorm_file, delim = ",") %>%
  dplyr::rename("bincentile" = pAIbin) %>%
  dplyr::mutate("bindecile" = paste0("bin_", ceiling(bincentile / 10)))
names(ranknorm_df) <- str_replace(names(ranknorm_df), "bin", "bin_")
ranknorm_t_mat <- t(as.matrix(ranknorm_df[,paste0("bin_", seq(1:bin_size))]))
colnames(ranknorm_t_mat) <- seq(1:100)

# Convert 
obs_sum_filt_df <- shet_df %>%
  dplyr::select(genename, transcript, ENST, exp_mis, exp_lof, bin_1:bin_10, bin_11) %>%
  pivot_longer(!c("genename", "transcript", "ENST", "exp_mis", "exp_lof")) %>%
  dplyr::rename("n" = value, "rankbin" = name) %>%
  inner_join(shet_df %>%
               dplyr::select(genename, transcript, ENST, exp_mis, exp_lof, selbin1:selbin11) %>%
               pivot_longer(!c("genename", "transcript", "ENST", "exp_mis", "exp_lof")) %>%
               dplyr::rename("shet" = value, "rankbin" = name) %>%
               dplyr::mutate(rankbin = gsub("selbin", "bin_", rankbin))) %>%
  dplyr::mutate(rankbin = ordered(rankbin, levels = paste0("bin_", seq(1, bin_size + 1, 1))))

# Set up blocks to bootstrap across
indx <- 1:dim(obs_sum_filt_df)[1]
blocks <- ceiling(seq_along(indx) / ((bin_size + 1) * 1000))
indxl <- split(indx, blocks)

# Fast bootstrap number of variants in each bin
obs_bootsum_df <- lapply(1:length(indxl), function(i) {
  
  # Get input data frames for sampling
  message(paste0("Batch ", i))
  obs_boot_df <- obs_sum_filt_df[indxl[[i]],] %>%
    group_by(transcript) %>%
    dplyr::mutate(size = sum(n), weight = (n + 1) / size) %>%
    ungroup() %>%
    arrange(transcript, rankbin)
  rankbin_levels <- levels(obs_sum_filt_df$rankbin)
  obs_boot_list <- split(obs_boot_df, f = obs_boot_df$transcript)
  
  # Bootstrap observed counts
  lapply(1:length(obs_boot_list), function(x) {
    
    # Bootstrap across variant categories
    df <-  obs_boot_list[[x]]
    transcript <- df$transcript[1]
    exp_mis <- df$exp_mis[1]
    size <- df$size[1]
    shet <- df$shet[1]
    weights <- df$weight
    vec <- as.numeric(df$rankbin)
    set.seed(12345)
    tmp <- qtab(a = sample(vec, size * (boot_num), replace = T, prob = weights), b = rep(1:(boot_num), each = size)) %>%
      as.matrix() %>%
      t()
    colnames(tmp) <- paste0("bin_", colnames(tmp))
    
    # Sample and smooth centile ranks
    centile_smoot_mat <- tmp[,1:bin_size] %*% ranknorm_t_mat %>%
      as(., "TsparseMatrix")
    tmp_mat <- (unclass(tmp[,1:bin_size]) + 1) %>% 
      as(., "TsparseMatrix")
    i <- tmp_mat@i + 1
    j <- tmp_mat@j + 1
    x <-  tmp_mat@x - 1
    i_long <- rep(i, x)
    j_long <- rep(j, x)
    set.seed(12345)
    centile_samp <- (j_long - 1) * 10 + sample.int(10, length(j_long), replace = T)
    x_adj <- centile_smoot_mat[cbind(i_long, centile_samp)] / rep(x, x)
    tmp_smooth <- sparseMatrix(i = i_long, j = j_long, x = x_adj) %>%
      as.matrix()
    tmp_smooth <- cbind(tmp_smooth, tmp[,bin_size + 1])
    colnames(tmp_smooth) <- colnames(tmp) 
    tmp_smooth <- tmp_smooth %>%
      as_tibble()
    
    # Return bootstrap results
    bind_cols("transcript" = transcript, "exp_mis" = exp_mis, bootstrap = 1:boot_num, tmp_smooth)
    
  }) %>%
    bind_rows()
  
}) %>%
  bind_rows()

obs_bootsum_df %>%
  vroom::vroom_write(output_file1, delim = "\t")


# Compute depletion / shet bootstrapped estimates
obs_boot_df <- obs_sum_filt_df %>%
  group_by(transcript) %>%
  ungroup() %>%
  arrange(transcript, rankbin)
transcript_ids <- obs_bootsum_df %>% pull(transcript) %>% unique()
shet_boot_tmp_df <- lapply(1:(bin_size + 1), function(x) {
  in_bin <- paste0("bin_", x)
  depl <- 1 - obs_bootsum_df %>% pull(in_bin) / (obs_bootsum_df %>% .$exp_mis / (bin_size + 1))
  depl[depl > 1] <- 1
  depl[depl < 0] <- 0
  shet <- approx(x = sdmap_df$depletion, y = sdmap_df$selection, xout = depl, rule = 2, method = "linear")$y
  shet_limits <- log10(c(10^-4, 1))
  depl <- matrix(depl, nrow = length(depl) / boot_num, ncol = boot_num, byrow = T)
  shet <- matrix(shet, nrow = length(shet) / boot_num, ncol = boot_num, byrow = T)
  depl_mean <- rowMeans2(depl)
  depl_se <- rowSds(depl)
  depl_lower <- depl_mean - 1.96 * depl_se
  depl_upper <- depl_mean + 1.96 * depl_se
  shet_mean <- rowMeans2(shet)
  shet_se <- rowSds(shet)
  shet_mean <-  ifelse(shet_mean < shet_limits[1], shet_limits[1], shet_mean)
  shet_lower <- shet_mean - 1.96 * shet_se
  shet_lower <- ifelse(shet_lower < shet_limits[1], shet_limits[1], shet_lower)
  shet_upper <- shet_mean + 1.96 * shet_se
  shet_upper <- ifelse(shet_upper > shet_limits[2], shet_limits[2], shet_upper)
  bind_rows("transcript" = transcript_ids, "rankbin" = in_bin,
            "depl_boot_mean" = depl_mean, "depl_boot_lower" = depl_lower, "depl_boot_upper" = depl_upper, 
            "shet_boot_mean" = shet_mean, "shet_boot_lower" = shet_lower, "shet_boot_upper" = shet_upper,
            "depl_boot_se" = depl_se, "shet_boot_se" = shet_se)
}) %>%
  bind_rows() %>%
  as_tibble() 
shet_limits <- log10(c(10^-4, 1))
shet_boot_df <- obs_boot_df %>%
  dplyr::select(genename, transcript, ENST, rankbin, depl, shet) %>%
  inner_join(shet_boot_tmp_df) %>%
  dplyr::mutate(rankbin = as.character(rankbin)) %>%
  dplyr::mutate(shet = log10(shet)) %>%
  dplyr::mutate(shet_lower = shet - 1.96 * shet_boot_se, shet_upper = shet + 1.96 * shet_boot_se) %>%
  dplyr::mutate(shet_upper = ifelse(shet_boot_se == 0 & shet == -4, approx(x = sdmap_df$depletion, y = sdmap_df$selection, xout = 0 + 1.96 * depl_boot_se, rule = 2, method = "linear")$y, shet_upper)) %>%
  dplyr::mutate(shet =  ifelse(shet < shet_limits[1], shet_limits[1], shet),
                shet_lower = ifelse(shet_lower < shet_limits[1], shet_limits[1], shet_lower),
                shet_upper = ifelse(shet_upper > shet_limits[2], shet_limits[2], shet_upper))
  
shet_boot_stable_df <- shet_boot_df %>%
  dplyr::select(genename, transcript, ENST, rankbin, shet, shet_boot_mean, shet_lower, shet_upper) %>%
  dplyr::filter(rankbin %in% c("bin_10", "bin_11")) %>%
  pivot_wider(id_cols = c("genename", "transcript", "ENST"), names_from = "rankbin", values_from = c("shet", "shet_boot_mean", "shet_lower", "shet_upper")) %>%
  dplyr::select(genename, transcript, ENST, 
                "shet_boot_missense_top10perc" = shet_boot_mean_bin_10,
                "shet_missense_top10perc" = shet_bin_10, "shet_missense_top10perc_lowerCI" = shet_lower_bin_10, "shet_missense_top10perc_upperCI" = shet_upper_bin_10,
                "shet_boot_LoF" = shet_boot_mean_bin_11,
                "shet_LoF" = shet_bin_11, "shet_LoF_lowerCI" = shet_lower_bin_11, "shet_LoF_upperCI" = shet_upper_bin_11)
  
shet_boot_stable_df %>%
  vroom::vroom_write(output_file2, delim = "\t")


