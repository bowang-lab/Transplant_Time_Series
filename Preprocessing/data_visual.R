library(dplyr)
library(ggplot2)
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

# load the original plain text files
txf_prev = read.csv(ori_file_txf, header = T, sep = "|", stringsAsFactors = F, na.strings = "None",
                       quote = "")
tx_prev = read.csv(ori_file_tx, header = T, sep = "|", stringsAsFactors = F, na.strings = "None",
                      quote = "")
# filter patients with cause of death: cardio/cardiac related
cardio_death = c(4246, 4620, 4623, 4624, 4625, 4626, 4630)

not_dead = unique(filter(tx_prev, is.na(REC_COD))$TRR_ID)
# 4584
cardio_patients = unique(filter(txf_prev, (TFL_COD %in% cardio_death | TFL_COD2 %in% cardio_death | 
                                                TFL_COD3 %in% cardio_death) & TRR_ID %in% not_dead)$TRR_ID)
died_patients = unique(filter(txf_prev, !(is.na(TFL_COD)& is.na(TFL_COD2) & is.na(TFL_COD3)) & 
                                TRR_ID %in% not_dead & !(TRR_ID %in% cardio_patients))$TRR_ID)
lived_patients = unique(filter(txf_prev, (is.na(TFL_COD) & is.na(TFL_COD2) & is.na(TFL_COD3)) & 
                                 TRR_ID %in% not_dead & 
                                 !(TRR_ID %in% died_patients) & 
                                 !(TRR_ID %in% cardio_patients))$TRR_ID)


txf = filter(txf_prev, 
                TRR_ID %in% cardio_patients | TRR_ID %in% lived_patients | TRR_ID %in% died_patients)
tx = filter(tx_prev, TRR_ID %in% txf$TRR_ID)

txf = dplyr::distinct(txf)
tx = dplyr::distinct(tx)

######## data visualization ########
# transplant year frequency
a1 = filter(tx, tx$TRR_ID %in% cardio_patients)$REC_TX_DT
b1 = as.numeric(substring(a1, 1, 4))
hist1 = hist(main='Transplant year frequency plot for cardiac death', b1,breaks=30, xlab='transplant year',
             col="lightblue")

a2 = filter(tx_prev, tx_prev$TRR_ID %in% died_patients)$REC_TX_DT
b2 = as.numeric(substring(a2, 1, 4))
hist2 = hist(main='Transplant year frequency plot for other death', b2, xlab='transplant year',
             col="lightblue")

a3 = tx_prev$REC_TX_DT
b3 = as.numeric(substring(a3, 1, 4))
hist2 = hist(main='Transplant year frequency plot', b3, xlab='transplant year',
             col="lightblue")

test =  table(b3)
df = as.data.frame.table(test)
theme_set(theme_classic())
head(df)
g <- ggplot(df, aes(b3, Freq))
g + geom_bar(stat="identity", width = 0.5, fill="#FFDB6D") +
  theme(axis.text.x=element_text(angle=65, vjust=0.6)) +
  labs(title="Histogram of transplant years")

a1 = filter(tx, tx$TRR_ID %in% cardio_patients)$REC_TX_DT
b1 = as.numeric(substring(a1, 1, 4))
test =  table(b1)
df = as.data.frame.table(test)
theme_set(theme_classic())
head(df)
g <- ggplot(df, aes(b1, Freq))
g + geom_bar(stat="identity", width = 0.5, fill="#FFDB6D") +
  theme(axis.text.x=element_text(angle=65, vjust=0.6)) +
  labs(title="Histogram of transplant years for cardiac death")

a2 = filter(tx_prev, tx_prev$TRR_ID %in% died_patients)$REC_TX_DT
b2 = as.numeric(substring(a2, 1, 4))
test =  table(b2)
df = as.data.frame.table(test)
theme_set(theme_classic())
head(df)
g <- ggplot(df, aes(b2, Freq))
g + geom_bar(stat="identity", width = 0.5, fill="#FFDB6D") +
  theme(axis.text.x=element_text(angle=65, vjust=0.6)) +
  labs(title="Histogram of transplant years for other death")


a3 = tx_prev$REC_TX_DT
b3 = as.numeric(substring(a3, 1, 4))
test =  table(b3)
df = as.data.frame.table(test)
theme_set(theme_classic())
head(df)
g <- ggplot(df, aes(b3, Freq))
g + geom_bar(stat="identity", width = 0.5, fill="#FFDB6D") +
  theme(axis.text.x=element_text(angle=65, vjust=0.6)) +
  labs(xlab="transplant year", title="Histogram of transplant years for other death")


a0 = filter(tx, tx$TRR_ID %in% cardio_patients)
survival = difftime(a0$PERS_OPTN_DEATH_DT, a0$REC_TX_DT, unit="days") / 365
survival = floor(survival)
test =  table(survival)
df = as.data.frame.table(test)
theme_set(theme_classic())
head(df)
g <- ggplot(df, aes(survival, Freq))
g + geom_bar(stat="identity", width = 0.5, fill="#FFABCD") +
  theme(axis.text.x=element_text(angle=65, vjust=0.6)) +
  labs(title="Histogram of survial time for cardiac death") + 
  scale_x_discrete(name="Survival time (year)")

# time to death by transplant years
a0 = filter(tx, tx$TRR_ID %in% cardio_patients & !is.na(PERS_OPTN_DEATH_DT) & !is.na(REC_TX_DT))
a0$survival_time = difftime(a0$PERS_OPTN_DEATH_DT, a0$REC_TX_DT, unit="days") / 365
a0$transplant_year = as.numeric(substring(a0$REC_TX_DT, 1, 4))
a0$time_factor = cut(x=as.numeric(a0$survival_time), breaks=c(0, 1, 2, 5, 10, 30), right=FALSE)
a0$count = rep(0, nrow(a0))
for (i in unique(a0$transplant_year)){
  for (j in unique(a0$time_factor)) {
    a0$count[a0$transplant_year == i & a0$time_factor == j] = nrow(filter(a0, transplant_year == i & time_factor == j))
  }
}

ggplot(a0, aes(transplant_year, time_factor))  + geom_tile(aes(fill=count)) + geom_text(aes(label=count), colour = "white") + 
  theme(axis.text=element_text(size=18), axis.title=element_text(size=18,face="bold"), 
        axis.text.x=element_text(angle=90, hjust=1), legend.text=element_text(size=18), legend.title=element_text(size=18)) + 
  xlab("Transplant year") + ylab("Survival (year)") + labs(fill = "count", title="Statistics - Survial time for cardiac death") +
  scale_x_continuous(breaks=seq(1987, 2018, by=1))




trans_year = as.numeric(substring(a0$REC_TX_DT, 1, 4))
test =  data.frame(cbind(survival_time,trans_year))
head(test)

theme_set(theme_classic())
g <- ggplot(test, aes(x = trans_year, y=survival_time, group = trans_year))
g + geom_boxplot(fill="#ffccb0") + scale_y_continuous(breaks = seq(0, 30, by=10)) + 
  scale_x_continuous(breaks=seq(1987, 2018, by=1)) +
  theme(axis.text.x = element_text(angle = 90)) +
  labs(title="Box plot of survival time for cardiac death by years")


### class statistics ###
a0 = filter(tx_study, tx_study$TRR_ID %in% lived_patients)
trans_date0 = substring(a0$REC_TX_DT, 1, 4)

a1 = filter(tx_study, tx_study$TRR_ID %in% cardio_patients)
trans_date1 = substring(a1$REC_TX_DT, 1, 4)

a2 = filter(tx_study, tx_study$TRR_ID %in% gf_patients)
trans_date2 = substring(a2$REC_TX_DT, 1, 4)

a3 = filter(tx_study, tx_study$TRR_ID %in% cancer_patients)
trans_date3 = substring(a3$REC_TX_DT, 1, 4)

a4 = filter(tx_study, tx_study$TRR_ID %in% inf_patients)
trans_date4 = substring(a4$REC_TX_DT, 1, 4)

a0$class = "Survival"
a1$class = "Cardiac Death"
a2$class = "Graft Failure"
a3$class = "Cancer"
a4$class = "Infection"
t = data.frame(cbind(c(trans_date0, trans_date1, trans_date2, trans_date3, trans_date4), 
                     c(a0$class, a1$class, a2$class, a3$class, a4$class)))
colnames(t) <- c("year", "class")

theme_set(theme_classic())
g <- ggplot(t, aes(year))
g + geom_bar(aes(fill=class), width = 0.5) + 
  theme(axis.text.x = element_text(angle=65, vjust=0.6)) + 
  labs(title="Transplant Year Across Patient Classes (Liver)")




##### plot ####
plot_cat = function(col){
  par(mfrow=c(1,2))
  a = data[, col][data$TRR_ID %in% cardio_patients 
                  & data$REC_TX_DT <= 2004
                  & data[, col] != -1]
  b = data[, col][data$TRR_ID %in% cardio_patients
                  & data$REC_TX_DT > 2004
                  & data[, col] != -1]
  barplot(main="before 2004", prop.table(table(a)))
  barplot(main="after 2004", prop.table(table(b)))
  mtext(col, side = 3, line = -21, outer = TRUE)
}

plot_box = function(col){
  par(mfrow=c(1,1))
  boxplot(main=col, data[, col][data$TRR_ID %in% cardio_patients & data$REC_TX_DT <= 2004 & data[, col] != -1], 
          data[, col][data$TRR_ID %in% cardio_patients & data$REC_TX_DT > 2004 & data[, col] != -1], names=c('before 2004', 'after 2004'))
  
}
plot_cat("TFL_DIAB_DURING_FOL")
plot_cat("TFL_GRAFT_STAT")
plot_cat("REC_HIV_STAT")
plot_cat("TFL_FAIL_BILIARY")
plot_cat("REC_MALIG")
plot_cat("TFL_INSULIN_DEPND")
plot_cat("CAN_DIAL")
plot_box("DON_AGE")
plot_box("DON_BMI")
plot_box("CAN_LAST_SERUM_CREAT")
plot_box("CAN_AGE_AT_LISTING")
plot_box("TFL_BMI")
plot_cat("DON_GENDER")
plot_cat("CAN_GENDER")

data2 = data[!duplicated(data$TRR_ID,fromFirst=TRUE),]

ctrans_date = filter(data2, data2$TRR_ID %in% cardio_patients)$transfer_year

dtrans_date = filter(data2, data2$TRR_ID %in% died_patients)$transfer_year

ltrans_date = filter(data2, data2$TRR_ID %in% lived_patients)$transfer_year

class1 = rep("Cardiac Death", length(ctrans_date))
class2 = rep("Other Death", length(dtrans_date))
class3 = rep("Survival", length(ltrans_date))
t = data.frame(cbind(c(ctrans_date, dtrans_date, ltrans_date), 
                     c(class1, class2, class3)))
colnames(t) <- c("year", "class")

theme_set(theme_classic())
g <- ggplot(t, aes(year))
g + geom_bar(aes(fill=class), width = 0.5) + 
  theme(axis.text.x = element_text(angle=65, vjust=0.6)) + 
  labs(title="Transplant Year Across Patient Classes (Selected)")


