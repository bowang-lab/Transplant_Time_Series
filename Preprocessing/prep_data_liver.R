library(dplyr)
library(ggplot2)
library(foreach)
library(tidyverse)
library(data.table)

#set this to your own working directory
setwd("/home/osvald/Projects/Diagnostics/github/preprocessing")

# configure these variables to load and save proper files
# organ_list = c("li", "in", "kp", "ki", "pa", "hl", "hr", "lu")
organ = "li"
linked_organ = "liin"
ori_file_txf = paste("data/txf_", organ, ".txt", sep="")
ori_file_tx = paste("data/tx_", organ, ".txt", sep="")
ori_file_cand = paste("data/cand_", linked_organ, ".txt", sep="")
ori_file_immuno = paste("data/immuno.txt", sep="")
ori_file_fol_immuno = paste("data/fol_immuno.txt", sep="")


# load the original plain text files
txf_li_prev = read.csv(ori_file_txf, header = T, sep = "|", stringsAsFactors = F, na.strings = "None",
                       quote = "")
tx_li_prev = read.csv(ori_file_tx, header = T, sep = "|", stringsAsFactors = F, na.strings = "None",
                      quote = "")
cand = read.csv(ori_file_cand, header = T, sep = "|", stringsAsFactors = F, na.strings = "None",
                     quote = "")
immuno = read.csv(ori_file_immuno, header = T, sep = "|", stringsAsFactors = F, na.strings = "None",
                quote = "")
fol_immuno = read.csv(ori_file_fol_immuno, header = T, sep = "|", stringsAsFactors = F, na.strings = "None",
                quote = "")

# filtering out multi+transplant patients
print("number of patients before filtering out multi-transplant patients:")
print(length(unique(tx_li_prev$TRR_ID)))

txf_li_prev = filter(txf_li_prev, REC_TX_ORG_TY == "LI")
tx_li_prev = filter(tx_li_prev, REC_TX_ORG_TY == "LI")
print("number of patients after filtering out multi-transplant patients:")
print(length(unique(tx_li_prev$TRR_ID)))

# filter patients with cause of death: cardio/cardiac related

cardio_death = c(4246, 4620, 4623, 4624, 4625, 4626, 4630)
malig_death = c(4850, 4851, 4855, 4856)
gf_death = c(4600, 4601, 4602, 4603, 4604, 4605, 4606, 4610)
inf_death = c(4800, 4801, 4802, 4803, 4804, 4805, 4806, 4810, 4811, 4660, 4645)
# 4660: multi organ system failure
# 4645: respiratory failure

txf_li_prev = txf_li_prev %>% arrange(TRR_ID, TFL_PX_STAT_DT)
fol_immuno = fol_immuno %>% arrange(TRR_ID)

# fill all CODs
txf_li_prev = merge(txf_li_prev, cbind(tx_li_prev[, c("TRR_ID", "PERS_OPTN_DEATH_DT")]), all.x = TRUE, by = "TRR_ID")
txf_li_prev = txf_li_prev %>% group_by(TRR_ID) %>% mutate(TFL_COD = rep(tail(TFL_COD, n=1), n()),
                                                          TFL_COD2 = rep(tail(TFL_COD2, n=1), n()),
                                                          TFL_COD3 = rep(tail(TFL_COD3, n=1), n()))
txf_li_prev$TX_DT = substring(txf_li_prev$REC_TX_DT, 1, 4)
txf_li_prev = filter(txf_li_prev, 2002 <= TX_DT)
print("number of patients after filtering pre 2002 post 2014")
print(length(unique(txf_li_prev$TRR_ID)))
txf_li_prev = filter(txf_li_prev, TX_DT <= 2014)
print("number of patients after filtering post 2014")
print(length(unique(txf_li_prev$TRR_ID)))

## remove patients died before the first follow-up
#txf_li_prev = merge(txf_li_prev, cbind(tx_li_prev[, c("TRR_ID", "PERS_OPTN_DEATH_DT")]), by = "TRR_ID")
#txf_li_prev = txf_li_prev %>% group_by(TRR_ID) %>% slice(
# which(difftime(PERS_OPTN_DEATH_DT, REC_TX_DT, unit="days") > 0)
#)
#print("number of patients after filtering immediate deaths / no followup data:")
#print(length(unique(txf_li_prev$TRR_ID)))
                                                        
# cardio_patients = unique(txf_li_prev$TRR_ID[txf_li_prev$TFL_COD %in% cardio_death ])
# cancer_patients = unique(txf_li_prev$TRR_ID[txf_li_prev$TFL_COD %in% malig_death])
# gf_patients = unique(txf_li_prev$TRR_ID[txf_li_prev$TFL_COD %in% gf_death])
# inf_patients = unique(txf_li_prev$TRR_ID[txf_li_prev$TFL_COD %in% inf_death])
# lived_patients = unique(txf_li_prev$TRR_ID[is.na(txf_li_prev$TFL_COD) & 
#                                              !(txf_li_prev$TRR_ID %in% cardio_patients) & 
#                                              !(txf_li_prev$TRR_ID %in% gf_patients) & 
#                                              !(txf_li_prev$TRR_ID %in% cancer_patients) &
#                                              !(txf_li_prev$TRR_ID %in% inf_patients)
#                                            ])

# filter out patients with multiple CODs
cardio_patients = unique(txf_li_prev$TRR_ID[txf_li_prev$TFL_COD %in% cardio_death & is.na(txf_li_prev$TFL_COD2) & is.na(txf_li_prev$TFL_COD3)])
cancer_patients = unique(txf_li_prev$TRR_ID[txf_li_prev$TFL_COD %in% malig_death & is.na(txf_li_prev$TFL_COD2) & is.na(txf_li_prev$TFL_COD3)])
gf_patients = unique(txf_li_prev$TRR_ID[txf_li_prev$TFL_COD %in% gf_death & is.na(txf_li_prev$TFL_COD2) & is.na(txf_li_prev$TFL_COD3)])
inf_patients = unique(txf_li_prev$TRR_ID[txf_li_prev$TFL_COD %in% inf_death & is.na(txf_li_prev$TFL_COD2) & is.na(txf_li_prev$TFL_COD3)])
lived_patients = unique(txf_li_prev$TRR_ID[is.na(txf_li_prev$TFL_COD) & 
                                             !(txf_li_prev$TRR_ID %in% cardio_patients) & 
                                             !(txf_li_prev$TRR_ID %in% gf_patients) & 
                                             !(txf_li_prev$TRR_ID %in% cancer_patients) &
                                             !(txf_li_prev$TRR_ID %in% inf_patients)
                                           ])

# clear up a 5-year outlook for lived-patients
txf_li_prev$follow_year = as.numeric(substring(txf_li_prev$TFL_PX_STAT_DT, 1, 4))
txf_li_prev = txf_li_prev %>% group_by(TRR_ID) %>% 
  mutate(out_year = if (TRR_ID[1] %in% lived_patients) rep(tail(follow_year, n=1) - 5, length(TRR_ID))
         else rep(tail(follow_year, n=1), length(TRR_ID)))

txf_li_prev = txf_li_prev %>% group_by(TRR_ID) %>% slice(which(follow_year <= out_year))



# upgrade
all_patients = c(cardio_patients, cancer_patients, gf_patients, inf_patients, lived_patients)
txf_li = filter(txf_li_prev, 
                TRR_ID %in% all_patients)
tx_li = filter(tx_li_prev, TRR_ID %in% txf_li$TRR_ID)

txf_li = dplyr::distinct(txf_li)
print("number of patients after filtering multi-COD")
print(length(unique(txf_li$TRR_ID)))

### Group together drugs of interest
immuno$INIT_CYCLO <- (immuno$REC_DRUG_CD == -2 | immuno$REC_DRUG_CD == 44 | immuno$REC_DRUG_CD == 46 | 
                        immuno$REC_DRUG_CD == 48 | immuno$REC_DRUG_CD == 3 | immuno$REC_DRUG_CD == 4)
immuno$INIT_TACRO <- (immuno$REC_DRUG_CD == 5 | immuno$REC_DRUG_CD == 54 | immuno$REC_DRUG_CD == 59)
immuno$INIT_SIRO <- (immuno$REC_DRUG_CD == 6)
immuno$INIT_STEROIDS <- (immuno$REC_DRUG_CD == 49)
immuno$INIT_OTHER <- (!immuno$INIT_TACRO & !immuno$INIT_CYCLO & !immuno$INIT_SIRO  & !immuno$INIT_STEROIDS)

fol_immuno$CYCLO <- (fol_immuno$TFL_IMMUNO_DRUG_CD == -2 | fol_immuno$TFL_IMMUNO_DRUG_CD == 44 | fol_immuno$TFL_IMMUNO_DRUG_CD == 46 | 
                       fol_immuno$TFL_IMMUNO_DRUG_CD == 48 | fol_immuno$TFL_IMMUNO_DRUG_CD== 3 | fol_immuno$TFL_IMMUNO_DRUG_CD== 4)
fol_immuno$TACRO <- (fol_immuno$TFL_IMMUNO_DRUG_CD == 5 | fol_immuno$TFL_IMMUNO_DRUG_CD == 54 | fol_immuno$TFL_IMMUNO_DRUG_CD == 59)
fol_immuno$SIRO <- (fol_immuno$TFL_IMMUNO_DRUG_CD == 6)
fol_immuno$STEROIDS <- (fol_immuno$TFL_IMMUNO_DRUG_CD == 49)
fol_immuno$OTHER <- (!fol_immuno$TACRO & !fol_immuno$CYCLO & !fol_immuno$SIRO  & !fol_immuno$STEROIDS)


#### Functions ####
to_binary = function(vector, yes_values, no_values, na_values) {
  result = rep(0, length(vector))
  result[!is.na(vector) & vector %in% yes_values] = 1
  result[!is.na(vector) & vector %in% no_values] = 0
  result[is.na(vector) | vector %in% na_values] = -1
  return(result)
}

is_event = function(x) {
  # Figure out who has diabetes at any point in time
  # is_event(c(NA,NA,NA,NA)) == NA
  # is_event(c(NA, 0, 1, NA, 0)) == 1
  # is_event(c(NA, 0, 0, NA)) == 0
  if (all(is.na(x))) NA
  else as.numeric(any(x == 1, na.rm = T))
}

categorize_diagnosis = function(df) {
  df$HEPC = as.numeric(df$REC_DGN_4216) #removed second DGN since it never showed up
  #df$HEPB = as.numeric(df$REC_DGN_4592) #no hepatitis B?
  df$NAFLD = as.numeric(df$REC_DGN_4270)
  df$alcohol = as.numeric(df$REC_DGN_4215 | df$REC_DGN_4216 | df$REC_DGN_4217)
  df$PBC = as.numeric(df$REC_DGN_4220)
  df$PSC = as.numeric(df$REC_DGN_4240 | df$REC_DGN_4241 | df$REC_DGN_4242 | df$REC_DGN_4245)
  df$AUTOIMMUNE_HEPATITIS = as.numeric(df$REC_DGN_4212)
  return(df)
}

cleanup_bmi_hgt_wgt = function(df, prefix = "") {
  hgt_col = paste0(prefix, "HGT_CM")
  wgt_col = paste0(prefix, "WGT_KG")
  bmi_col = paste0(prefix, "BMI")
  
  bmi_cond = sapply(df[, bmi_col], function(x) {if (!is.na(x)) return(x < 12 | x > 50) else F})
  df[bmi_cond, hgt_col] = NA
  df[bmi_cond, wgt_col] = NA
  df[bmi_cond, bmi_col] = NA
  
  df[!is.na(df[, hgt_col]) & df[, hgt_col] < 90, hgt_col] = NA
  df[!is.na(df[, wgt_col]) & df[, wgt_col] < 10, wgt_col] = NA
  
  return(df)
}


tx_len = nrow(tx_li)
txf_len = nrow(txf_li)

tx_study = data.frame(matrix(NA, nrow = tx_len, ncol = 0))
txf_study = data.frame(matrix(NA, nrow = txf_len, ncol = 0))

# variables of our interests
tx_cols = c("TRR_ID", "REC_LIFE_SUPPORT_OTHER", "REC_DGN", "REC_DGN2",
            "CAN_AGE_AT_LISTING", "CAN_GENDER", "DON_GENDER")

for (name in tx_cols) {
  #print(name)
  tx_study = cbind(tx_study, tx_li[name])
}

txf_study$follow_year = txf_li$follow_year
txf_study$TRR_ID = txf_li$TRR_ID
txf_study$TRR_FOL_ID = txf_li$TRR_FOL_ID
txf_study$TFL_COD = txf_li$TFL_COD
txf_study$TFL_COD2 = txf_li$TFL_COD2
txf_study$TFL_COD3 = txf_li$TFL_COD3


#### variables in the list but not included in this file ###
tx_vars  = c("CAN_AGE_AT_LISTING",
              "CAN_LAST_SERUM_SODIUM",
              "CAN_LAST_SERUM_CREAT",
              "CAN_LAST_SRTR_LAB_MELD",
              "CAN_PREV_HL", "CAN_PREV_HR", "CAN_PREV_IN", "CAN_PREV_KI", "CAN_PREV_KP", "CAN_PREV_LI",
              "CAN_PREV_LU", "CAN_PREV_PA", "CAN_PREV_TX", 
              "DON_AGE", "DON_GENDER", "REC_AGE_AT_TX", 
              "REC_POSTX_LOS",
              "REC_PREV_HL", "REC_PREV_HR", "REC_PREV_IN", "REC_PREV_KI", "REC_PREV_KP", "REC_PREV_LI",
              "REC_PREV_LU", "REC_PREV_PA",
              "REC_VENTILATOR", "PERS_OPTN_DEATH_DT",
              "REC_TX_DT", "REC_FAIL_DT",
              "CAN_LAST_DIAL_PRIOR_WEEK") # Added this

txf_vars = c("TFL_ALBUMIN", "TFL_ALKPHOS", "TFL_ANTIVRL_THERAPY_TY",
             "TFL_BMI", "TFL_CAD", "TFL_CREAT", "TFL_HOSP_NUM",
             "TFL_INR", "TFL_REJ_EVENT_NUM", "TFL_SGOT", "TFL_SGPT", "TFL_TOT_BILI",
             "TFL_WGT_KG",
             "REC_TX_DT", "TFL_PX_STAT_DT", "TFL_FAIL_DT", "TFL_TXFER_DT",
             "TFL_CMV_IGG", "TFL_CMV_IGM") # New CMV vars

for (name in tx_vars) {
  print(name)
  tx_study = cbind(tx_study, tx_li[name])
}

for (name in txf_vars) {
  print(name)
  txf_study = cbind(txf_study, txf_li[name])
}

#### Process tx ####
tx_study$TFL_COD = tx_li$TFL_COD


# gender
tx_study[!is.na(tx_study$CAN_GENDER) & (tx_study$CAN_GENDER == "M"), "CAN_GENDER"] = 0
tx_study[!is.na(tx_study$CAN_GENDER) & (tx_study$CAN_GENDER == "F"), "CAN_GENDER"] = 1
tx_study[!is.na(tx_study$DON_GENDER) & (tx_study$DON_GENDER == "M"), "DON_GENDER"] = 0
tx_study[!is.na(tx_study$DON_GENDER) & (tx_study$DON_GENDER == "F"), "DON_GENDER"] = 1


# Format education info
tx_study$CAN_EDUCATION = tx_li$CAN_EDUCATION
tx_study[!is.na(tx_study$CAN_EDUCATION) &
              (tx_study$CAN_EDUCATION == 996 |
                 tx_study$CAN_EDUCATION == 998), "CAN_EDUCATION"] = NA

# Format historical info info
tx_study$DON_HIST_DIAB = tx_li$DON_HIST_DIAB
tx_study[!is.na(tx_study$DON_HIST_DIAB) &
           (tx_study$DON_HIST_DIAB == 998), "DON_HIST_DIAB"] = NA
tx_study$DON_HIST_HYPERTEN = tx_li$DON_HIST_HYPERTEN
tx_study[!is.na(tx_study$DON_HIST_HYPERTEN) &
           (tx_study$DON_HIST_HYPERTEN == 998), "DON_HIST_HYPERTEN"] = NA



# # nlatin: not latin or unknown is 0
tx_study$DIAB = transmute(tx_li, as.numeric(!((CAN_DIAB == 1 & CAN_DIAB_TY == 1) | 
                                   (CAN_DIAB == 998 & CAN_DIAB_TY == 1) |
                                   (CAN_DIAB == 1 & CAN_DIAB_TY == 998) | 
                                   (is.na(CAN_DIAB) & CAN_DIAB_TY == 1) |
                                   (CAN_DIAB == 1 & is.na(CAN_DIAB_TY)))))


# tx_study$CAN_LAST_SRTR_LAB_MELD_TY = transmute(tx_li, as.numeric(CAN_LAST_SRTR_LAB_MELD_TY == "M"))
tx_study$DON_TY = transmute(tx_li, as.numeric(DON_TY == "L"))

# ANGINA - missing 19.47%
na_rows = tx_li %>% filter((is.na(CAN_ANGINA) |
                                    CAN_ANGINA == 998) &
                                   (is.na(CAN_ANGINA_CAD) | CAN_ANGINA_CAD == 998)) %>% .$TRR_ID
no_angina = tx_li %>% filter((
  CAN_ANGINA == 1 & (is.na(CAN_ANGINA_CAD) | CAN_ANGINA_CAD == 998 | CAN_ANGINA_CAD == 1)) |
    (
      CAN_ANGINA_CAD == 1 & (is.na(CAN_ANGINA) | CAN_ANGINA == 998 | CAN_ANGINA == 1)
    )) %>% .$TRR_ID
tx_study$ANGINA = 1
tx_study[tx_study$TRR_ID %in% no_angina, "ANGINA"] = 0
tx_study[tx_study$TRR_ID %in% na_rows, "ANGINA"] = NA

for (col in c("CAN_CEREB_VASC", "CAN_DRUG_TREAT_COPD", "CAN_DRUG_TREAT_HYPERTEN", 
              "CAN_PERIPH_VASC", "CAN_PULM_EMBOL", "CAN_MALIG", "CAN_TIPSS",
              "REC_IMMUNO_MAINT_MEDS", "REC_LIFE_SUPPORT",
              "REC_GRAFT_STAT", "REC_INOTROP_BP_SUPPORT", "REC_ON_VENTILATOR",
              "REC_TIPSS", "REC_VARICEAL_BLEEDING",
              "REC_FAIL_BILIARY", "REC_FAIL_HEP_DENOVO", "REC_FAIL_HEP_RECUR", "REC_FAIL_INFECT",
              "REC_FAIL_PRIME_GRAFT_FAIL", "REC_FAIL_RECUR_DISEASE", "REC_FAIL_REJ_ACUTE", 
              "REC_FAIL_VASC_THROMB", "REC_IMMUNO_MAINT_MEDS", "REC_TOLERANCE_INDUCTION_TECH",
              "REC_MALIG")) {
  tx_study[, col] =  to_binary(tx_li[, col], "Y", "N", c("U", ""))
}

for (col in c("REC_HIV_STAT")) {
  tx_study[, col] =  to_binary(tx_li[, col], "P", "N", c("", "U", "I", "C", "ND"))
}

# medical condition
tx_study$REC_MED_COND_HOSP = to_binary(tx_li$REC_MED_COND, c(1, 2), c(3), NA)
tx_study$REC_MED_COND_ICU = to_binary(tx_li$REC_MED_COND, c(1), c(2, 3), NA)

# ACUTE_REJ_EPISODE - missing 36.25%
tx_study$ACUTE_REJ_EPISODE = to_binary(tx_li$REC_ACUTE_REJ_EPISODE, c(1, 2), c(3), NA)
tx_study$ACUTE_REJ_EPISODE_TREATED_ADDITIONAL = to_binary(tx_li$REC_ACUTE_REJ_EPISODE, c(1), c(2, 3), NA)

# DGN - missing 2.86%
tx_study[!is.na(tx_li$REC_DGN) &
              tx_li$REC_DGN == 999, "REC_DGN"] = NA
tx_study[!is.na(tx_li$REC_DGN2) &
              tx_li$REC_DGN2 == 999, "REC_DGN2"] = NA

levels = union(unique(tx_li$REC_DGN),
               unique(tx_li$REC_DGN2))
levels = levels[order(levels)][1:(length(levels) - 1)]

cfactor = factor(tx_study$REC_DGN, levels = levels)
cfactor2 = factor(tx_study$REC_DGN2, levels = levels)
options(na.action = "na.pass")
dummies = data.frame(model.matrix( ~ cfactor))
colnames(dummies) = paste("REC_DGN", levels, sep = "_") %>% gsub(" ", "_", .)
dummies[, paste0("REC_DGN_", levels[1])] = as.numeric(!is.na(cfactor) &
                                                        cfactor == levels[1])
dummies[is.na(cfactor), paste0("REC_DGN_", levels[1])] = NA

dummies2 = data.frame(model.matrix( ~ cfactor2))
colnames(dummies2) = paste("REC_DGN", levels, sep = "_") %>% gsub(" ", "_", .)
dummies2[, paste0("REC_DGN_", levels[1])] = as.numeric(!is.na(cfactor2) &
                                                         cfactor2 == levels[1])
dummies2[is.na(cfactor2), paste0("REC_DGN_", levels[1])] = NA

dummies12 <-
  data.frame(apply(dummies | dummies2, 2, function(x)
    as.numeric(x)))
for (col in colnames(dummies12)) {
  rows = which(is.na(dummies12[, col]))
  dummies12[rows, col] = apply(cbind(dummies[rows, col], dummies2[rows, col]), 1,
                               function (x) {
                                 if (is.na(x[1]) & is.na(x[2])) NA
                                 else 0
                               })
}

tx_study = cbind(tx_study, dummies12)

tx_study = categorize_diagnosis(tx_study)

# FUNCTN_STAT - missing 18.19%
tx_li$REC_FUNCTN_STAT = tx_li$REC_FUNCTN_STAT
tx_li$REC_FUNCTN_STAT[is.na(tx_li$REC_FUNCTN_STAT) |
                          tx_li$REC_FUNCTN_STAT == 996 |
                          tx_li$REC_FUNCTN_STAT == 998] = NA
tx_li$REC_FUNCTN_STAT[!is.na(tx_li$REC_FUNCTN_STAT) &
                          tx_li$REC_FUNCTN_STAT %in% c(2100, 2090, 2080)] = 1
tx_li$REC_FUNCTN_STAT[!is.na(tx_li$REC_FUNCTN_STAT) &
                          tx_li$REC_FUNCTN_STAT %in% c(2070, 2060, 2050)] = 2
tx_li$REC_FUNCTN_STAT[!is.na(tx_li$REC_FUNCTN_STAT) &
                          tx_li$REC_FUNCTN_STAT %in% c(2040, 2030, 2020, 2010)] = 3

#REC_PHYSC_CAPACITY #not recorded frequently
#tx_li$REC_PHYSC_CAPACITY[is.na(tx_li$REC_PHYSC_CAPACITY) |
#                                 tx_li$REC_PHYSC_CAPACITY == 996 |
#                                 tx_li$REC_PHYSC_CAPACITY == 998] = NA

# REC_PRIMARY_PAY
tx_li$REC_PRIMARY_PAY[is.na(tx_li$REC_PRIMARY_PAY) | tx_li$REC_PRIMARY_PAY == 15] = NA

# TFL_WORK_NO_STAT
txf_li$TFL_WORK_NO_STAT[is.na(txf_li$TFL_WORK_NO_STAT) | 
                         txf_li$TFL_WORK_NO_STAT == 996 |
                         txf_li$TFL_WORK_NO_STAT == 998] = NA
# TFL_WORK_YES_STAT
txf_li$TFL_WORK_YES_STAT[is.na(txf_li$TFL_WORK_YES_STAT) | 
                         txf_li$TFL_WORK_YES_STAT == 998] = NA

# TFL_PERM_STATE
txf_li$TFL_PERM_STATE[is.na(txf_li$TFL_PERM_STATE) | 
                          txf_li$TFL_PERM_STATE == "ZZ"] = NA

# factorizes columns into one hot encoding # removed Physc_capacity
for (col in c("CAN_RACE_SRTR", "REC_FUNCTN_STAT", "REC_PRIMARY_PAY",
              "CAN_INIT_STAT")) {
  for (val in (unique(tx_li[, col][!is.na(tx_li[, col])]))) {
    tx_study[, paste(col, val, sep = "_")] = to_binary(tx_li[, col], val, c(), "")
  }
}

for (col in c("TFL_WORK_NO_STAT", "TFL_WORK_YES_STAT", "TFL_PERM_STATE")) {
  for (val in (unique(txf_li[, col]))) {
    txf_study[, paste(col, val, sep = "_")] = to_binary(txf_li[, col], val, c(), "")
  }
}

tx_study$CAN_ETHNICITY_SRTR_LATINO = rep(0, nrow(tx_study))
tx_study$CAN_ETHNICITY_SRTR_LATINO[tx_li$CAN_ETHNICITY_SRTR == "LATINO"] = 1

tx_li$PORTAL_VEIN = tx_li$REC_PORTAL_VEIN

for (col in c("PORTAL_VEIN")) {
  tx_study[, col] = to_binary(tx_li[, paste0("REC_", col)], "Y", "N", c("U", ""))
  na_rows = which(is.na(tx_li[, col]))
  tx_study[na_rows, col] = to_binary(tx_li[na_rows, paste0("CAN_", col)], "Y", "N", c("U", ""))
}

# Transplant details REC_TX_ORG_TY
other_transplant_types = c("HL", "HR", "IN", "KI", "KP", "LU", "PA")
for (transplant_type in other_transplant_types) {
  tx_study[, paste0("TX_ORG_TY_", transplant_type)] = as.numeric(grepl(transplant_type, tx_li$REC_TX_ORG_TY))
}

#transplant_year
# tx_study$transplant_year = format(tx_study$REC_TX_DT, "%Y")

# Calculate time since transplant
time_since_transplant = txf_li$TFL_FOL_CD/10
time_since_transplant[txf_li$TFL_FOL_CD == 6] = 0.5

rows = which(txf_li$TFL_FOL_CD == 800) # Graft failure
time_since_transplant[rows] = (as.Date(txf_li$TFL_FAIL_DT[rows]) - as.Date(txf_li$REC_TX_DT[rows]))/365

rows = which(txf_li$TFL_FOL_CD == 998) # Lost to followup
time_since_transplant[rows] = (as.Date(txf_li$TFL_PX_STAT_DT[rows]) - as.Date(txf_li$REC_TX_DT[rows]))/365

rows = which(txf_li$TFL_FOL_CD == 999) # Death
time_since_transplant[rows] = (as.Date(txf_li$TFL_PX_STAT_DT[rows]) - as.Date(txf_li$REC_TX_DT[rows]))/365

txf_study$time_since_transplant = time_since_transplant

## Status of diabetes
txf_study$TFL_DIAB_DURING_FOL = to_binary(txf_li$TFL_DIAB_DURING_FOL, "Y", "N", c("", "U"))

# Find diabetes time for those who have diabetes during followup
diab_time_since_tx = txf_study %>% 
  filter(TFL_DIAB_DURING_FOL == 1) %>% 
  select(TRR_ID, time_since_transplant) %>%
  group_by(TRR_ID) %>% 
  summarize(diab_time=min(time_since_transplant))

is_diab = txf_study %>% 
  select(TRR_ID, TFL_DIAB_DURING_FOL) %>% 
  group_by(TRR_ID) %>%
  summarize(is_diab = is_event(TFL_DIAB_DURING_FOL))

no_diab_ids = is_diab %>% 
  filter(is.na(is_diab) | is_diab==0) %>% .$TRR_ID


# Figure out last time observed for no diabetes and where diabetes is NA
# diab_time_since_tx = rbind(diab_time_since_tx, txf_study %>%
#                              filter(TRR_ID %in% no_diab_ids, TFL_DIAB_DURING_FOL == 0 | is.na(TFL_DIAB_DURING_FOL)) %>%
#                              select(TRR_ID, time_since_transplant) %>%
#                              group_by(TRR_ID) %>% 
#                              summarize(diab_time=max(time_since_transplant)))

# # Adding TRR_ID for those individuals in tx but not in txf
# all_ids = tx_study$TRR_ID
# ids_not_in_txf = all_ids[which(!(all_ids %in% txf_study$TRR_ID))]
# 
# is_diab = rbind(is_diab, cbind(TRR_ID = ids_not_in_txf, is_diab = NA))
# diab_time_since_tx = rbind(diab_time_since_tx, cbind(TRR_ID = ids_not_in_txf, diab_time = NA))

# # Merge diabetes information
# diab_info = merge(is_diab, diab_time_since_tx)
# tx_study = merge(tx_study, diab_info)

### cand information ###
tx_study = cbind(tx_study, tx_li["PERS_ID"])
cand_filter = filter(cand, PERS_ID %in% tx_study$PERS_ID)
tx_study2 = merge(tx_study, cand_filter[, c("CAN_DIAL", "PERS_ID")], all.x=T, by="PERS_ID")
tx_study = tx_study2

tx_study$CAN_DIAL[tx_study$CAN_DIAL == 998 | tx_study$CAN_DIAL == 999] = NA

#### Process txf ####
rownames(txf_study) = txf_study$TRR_FOL_ID

for (col in c("TFL_ANTIVRL_THERAPY", "TFL_INSULIN_DEPND")) {
  txf_study[, col] = to_binary(txf_li[, col], "Y", "N", c("", "U"))
  no_rows = which(txf_li[, col] == "N")
  
  for (col2 in colnames(txf_li)[grepl(paste0(col, "_TY_"), colnames(txf_li))]) {
    txf_study[no_rows, col2] = 0
  }
}

txf_study$ACUTE_REJ_EPISODE = to_binary(txf_li$TFL_ACUTE_REJ_EPISODE, c(1,2), c(3), 998)

for (col in c("TFL_FAIL_BILIARY", "TFL_GRAFT_STAT", "TFL_HOSP", "TFL_IMMUNO_DISCONT", 
              "TFL_MALIG", "TFL_MALIG_LYMPH","TFL_MALIG_RECUR_TUMOR", "TFL_MALIG_TUMOR",
              "TFL_PX_NONCOMP", "TFL_REJ_TREAT", "TFL_WORK_INCOME", "TFL_CAD")) {
  txf_study[col] =  to_binary(txf_li[,col], "Y", "N", "U")
}


txf_study$FUNCTN_STAT = to_binary(txf_li$TFL_FUNCTN_STAT, c(1, 2100), 2080, 996)
txf_study$IMMUNO_MAINT_MEDS = to_binary(txf_li$TFL_IMMUNO_MAINT_MEDS, c(1,2,4,5), 3, NA)
#txf_study$TFL_PHYSC_CAPACITY[txf_study$TFL_PHYSC_CAPACITY == 998 | txf_study$TFL_PHYSC_CAPACITY == 996] = NA

rows = which(is.na(txf_study$time_since_transplant))
txf_study$time_since_transplant[rows] = (as.Date(txf_li$TFL_PX_STAT_DT[rows]) - as.Date(txf_li$REC_TX_DT[rows]))/365

# txf_study$TFL_PX_STAT[txf_study$TFL_PX_STAT == ""] = NA
# txf_study$TFL_WORK_NO_STAT[txf_study$TFL_WORK_NO_STAT == 996 | txf_study$TFL_WORK_NO_STAT == 998] = NA
# txf_study$TFL_WORK_YES_STAT[txf_study$TFL_WORK_YES_STAT == 996 | txf_study$TFL_WORK_YES_STAT == 998] = NA

txf_li$TFL_EMPL_STAT_PRE04[txf_li$TFL_EMPL_STAT_PRE04 == 996 | txf_li$TFL_EMPL_STAT_PRE04 == 998] = NA
# factorizes columns into one hot encoding
for (col in c("TFL_CARE_PROV_BY", "TFL_EMPL_STAT_PRE04", 
               "TFL_PRIMARY_PAY")) {      #"TFL_PHYSC_CAPACITY", removed due to missingness
  for (val in (unique(txf_li[, col]))) {
    txf_study[, paste(col, val, sep = "_")] = to_binary(txf_li[, col], val, c(), "")
  }
}



# merge tx and txf
require("dplyr")
dup = intersect(colnames(tx_study), colnames(txf_study))
combined_data = merge(tx_study[, c(colnames(tx_study))], txf_study[, c(colnames(txf_study))], by = dup)

# arrange the trr id and dates
combined_data = combined_data %>% arrange(TRR_ID, TFL_PX_STAT_DT)
txf_study = txf_study %>% arrange(TRR_ID, TFL_PX_STAT_DT)
tx_study = tx_study %>% arrange(TRR_ID)
combined_data$DIAB = unlist(combined_data$DIAB, use.names=FALSE) # DIAB is list type for some reason
combined_data$DON_TY = unlist(combined_data$DON_TY, use.names=FALSE) # DON_TY is list type for some reason
combined_data <- dplyr::distinct(combined_data)

length(unique(tx_li$TRR_ID[combined_data$TFL_COD == 4626]))

######### add labels ########
# combined_data = read_csv("liverall.csv")
combined_data$DeathLabel <- rep(0, nrow(combined_data))
combined_data$Label1 <- rep(0, nrow(combined_data))
combined_data$Label5 <- rep(0, nrow(combined_data))
# combined_data[combined_data$TRR_ID %in% cardio_patients & !duplicated(combined_data$TRR_ID,fromLast=TRUE),]$DeathLabel <- 1
# combined_data[combined_data$TRR_ID %in% gf_patients & !duplicated(combined_data$TRR_ID,fromLast=TRUE),]$DeathLabel <- 2
# combined_data[combined_data$TRR_ID %in% cancer_patients & !duplicated(combined_data$TRR_ID,fromLast=TRUE),]$DeathLabel <- 3
# combined_data[combined_data$TRR_ID %in% inf_patients & !duplicated(combined_data$TRR_ID,fromLast=TRUE),]$DeathLabel <- 4

## time ##
combined_data$time_since_transplant = difftime(combined_data$TFL_PX_STAT_DT, combined_data$REC_TX_DT, units = "days") / 365
combined_data$time_to_death = difftime(combined_data$PERS_OPTN_DEATH_DT, combined_data$TFL_PX_STAT_DT, unit="days") / 365

#### split years
#combined_data$transfer_year =as.numeric(substring(combined_data$REC_TX_DT, 1, 4))


combined_data[combined_data$TRR_ID %in% cardio_patients,]$DeathLabel <- 1
combined_data[combined_data$TRR_ID %in% gf_patients,]$DeathLabel <- 2
combined_data[combined_data$TRR_ID %in% cancer_patients,]$DeathLabel <- 3
combined_data[combined_data$TRR_ID %in% inf_patients,]$DeathLabel <- 4

combined_data[combined_data$TRR_ID %in% cardio_patients & combined_data$time_to_death < 1, ]$Label1 <- 1
combined_data[combined_data$TRR_ID %in% gf_patients & combined_data$time_to_death < 1, ]$Label1 <- 2
combined_data[combined_data$TRR_ID %in% cancer_patients & combined_data$time_to_death < 1, ]$Label1 <- 3
combined_data[combined_data$TRR_ID %in% inf_patients & combined_data$time_to_death < 1, ]$Label1 <- 4

combined_data[combined_data$TRR_ID %in% cardio_patients & combined_data$time_to_death < 5, ]$Label5 <- 1
combined_data[combined_data$TRR_ID %in% gf_patients & combined_data$time_to_death < 5, ]$Label5 <- 2
combined_data[combined_data$TRR_ID %in% cancer_patients & combined_data$time_to_death < 5, ]$Label5 <- 3
combined_data[combined_data$TRR_ID %in% inf_patients & combined_data$time_to_death < 5, ]$Label5 <- 4


# filter out dead patients who only have one time point
combined_data = combined_data %>% group_by(TRR_ID) %>% filter((!(TRR_ID %in% lived_patients) & n() > 1) | 
                                                                TRR_ID %in% lived_patients)
# Add immunosuppression data
### cut down on immuno and fol_immuno based on TRR/TRR_FOL
#immuno <- immuno[which(immuno$TRR_ID %in% tx_li$TRR_ID),]
fol_immuno <- fol_immuno[which(fol_immuno$TRR_FOL_ID %in% combined_data$TRR_FOL_ID),]
#### sum immuno perscriptions in followups & start
#immuno = aggregate(cbind(INIT_CYCLO=immuno$INIT_CYCLO, INIT_TACRO=immuno$INIT_TACRO, 
#                         INIT_SIRO=immuno$INIT_SIRO, INIT_STEROIDS=immuno$INIT_STEROIDS, 
#                         INIT_OTHER=immuno$INIT_OTHER), by=list(TRR_ID = immuno$TRR_ID), FUN=sum)
fol_immuno = aggregate(cbind(CYCLO=fol_immuno$CYCLO, TACRO=fol_immuno$TACRO, 
                             SIRO=fol_immuno$SIRO, STEROIDS=fol_immuno$STEROIDS, 
                             OTHER=fol_immuno$OTHER), by=list(TRR_FOL_ID = fol_immuno$TRR_FOL_ID), FUN=sum)
combined_data = merge(combined_data, fol_immuno, by.x = "TRR_FOL_ID", all.x = TRUE)

print("number of patients in each class:")
print(length(unique(combined_data$TRR_ID[combined_data$DeathLabel == 0])))
print(length(unique(combined_data$TRR_ID[combined_data$DeathLabel == 1])))
print(length(unique(combined_data$TRR_ID[combined_data$DeathLabel == 2])))
print(length(unique(combined_data$TRR_ID[combined_data$DeathLabel == 3])))
print(length(unique(combined_data$TRR_ID[combined_data$DeathLabel == 4])))

print("total number of follow-ups:")
print(nrow(combined_data))
print("number of follow-ups in each class:")
print(nrow(filter(combined_data, DeathLabel == 0)))
print(nrow(filter(combined_data, DeathLabel == 1)))
print(nrow(filter(combined_data, DeathLabel == 2)))
print(nrow(filter(combined_data, DeathLabel == 3)))
print(nrow(filter(combined_data, DeathLabel == 4)))

print("[1 year] number of positive labels in each class:")
print(nrow(filter(combined_data, Label1 == 1)))
print(nrow(filter(combined_data, Label1 == 2)))
print(nrow(filter(combined_data, Label1 == 3)))
print(nrow(filter(combined_data, Label1 == 4)))

print("[5 year] number of positive labels in each class:")
print(nrow(filter(combined_data, Label5 == 1)))
print(nrow(filter(combined_data, Label5 == 2)))
print(nrow(filter(combined_data, Label5 == 3)))
print(nrow(filter(combined_data, Label5 == 4)))


# drop unneeded columns
droppings = c("TRR_FOL_ID", "PERS_ID", "DON_GENDER.1", 
              "CAN_AGE_AT_LISTING.1", "REC_DGN", "REC_DGN2",
              "TFL_PHYSC_CAPACITY_998", "TFL_PHYSC_CAPACITY_996",
              "REC_TX_DT", "PERS_OPTN_DEATH_DT", "TFL_PX_STAT_DT", "TFL_FAIL_DT", "TFL_TXFER_DT",
              "REC_FAIL_DT", "REC_DGN_999", "TFL_PRIMARY_PAY_NA", "TFL_PHYSC_CAPACITY_NA", 
              "TFL_EMPL_STAT_PRE04_NA", "TFL_CARE_PROV_BY_NA", "TFL_WORK_YES_STAT_NA", "TFL_PERM_STATE_NA", "TFL_WORK_NO_STAT_NA",
              "Label1", "Label5"
)
combined_data = combined_data[, !(colnames(combined_data) %in% droppings)]
# replace NA by -1
combined_data[is.na(combined_data)] <- -1


######### save file #########
## split and save to csv per patient
write <- function(df, dir) {
  write.csv(df, paste0(dir, unique(df$group_id), ".csv"), na="-1")
  return (df)
}

dir = './mar4/mar4_csv/'
combined_data$group_id = group_indices(combined_data, TRR_ID)
combined_data %>% group_by(group_id) %>% do(write(., dir))


