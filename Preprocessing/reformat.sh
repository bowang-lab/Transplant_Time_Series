sas7bdat_dir=$1
#python sas7bdat_to_txt.py $sas7bdat_dir/"tx_li.sas7bdat" > data/tx_li.txt
##python3.6 sas7bdat_to_txt.py $sas7bdat_dir/"immuno.sas7bdat" > data/immuno.txt
#python sas7bdat_to_txt.py $sas7bdat_dir/"txf_li.sas7bdat" > data/txf_li.txt
##python3.6 sas7bdat_to_txt.py $sas7bdat_dir/"fol_immuno.sas7bdat" > data/fol_immuno.txt
#python sas7bdat_to_txt.py $sas7bdat_dir/"cand_liin.sas7bdat" > data/cand_liin.txt
##Rscript --vanilla prep_data_yuchen.R
source /pkgs/scripts/use-anaconda3.sh
conda activate sepsis
python sas7bdat_to_txt.py "/home/osvald/Projects/Diagnostics/github/preprocessing/data/tx_li.sas7bdat" > /home/osvald/Projects/Diagnostics/github/preprocessing/data/tx_li.txt
python sas7bdat_to_txt.py "/home/osvald/Projects/Diagnostics/github/preprocessing/data/txf_li.sas7bdat" > /home/osvald/Projects/Diagnostics/github/preprocessing/data/txf_li.txt
python sas7bdat_to_txt.py "/home/osvald/Projects/Diagnostics/github/preprocessing/data/cand_liin.sas7bdat" > /home/osvald/Projects/Diagnostics/github/preprocessing/data/cand_liin.txt
python sas7bdat_to_txt.py "/home/osvald/Projects/Diagnostics/github/preprocessing/data/fol_immuno.sas7bdat" > /home/osvald/Projects/Diagnostics/github/preprocessing/data/fol_immuno.txt
python sas7bdat_to_txt.py "/home/osvald/Projects/Diagnostics/github/preprocessing/data/immuno.sas7bdat" > /home/osvald/Projects/Diagnostics/github/preprocessing/data/immuno.txt