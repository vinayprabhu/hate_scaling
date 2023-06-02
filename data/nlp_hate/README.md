Data assets curated with the Pysentimiento experiments

1: The $(N_{parquet} \times 4)+1 = 641$ meta-dataset files generated for the 400m and 2B-en datasets are shared 
[here](https://hal.cse.msu.edu/assets/data/papers/hate_detect_laion_400m_2B-en.zip)


a) ```index_random_{ind_i}.npy```: $N_{parquet}$ random-index files of the naming-format: ```index_random_{ind_i}.npy```. Each of these contain 0.1 million random indices pertaining to the rows of the $ind\_i^{th}$ parquet file (in ```parquet_list```). Shape: ```(100000,)```

b) ```prob_hate_{ind_i}.npy```: $N_{parquet}$ _hate-probability-matrix_ files of shape ```(100000, 3)``` in the naming-format of ```prob_hate_{ind_i}.npy``` pertaining to the 0.1 million random-indexed rows of the $ind\_i^{th}$ parquet file.

c) ```qfr_file_{ind_i}.npy```: $N_{parquet}$ _quality-failure-rate_ files of shape ```(3,)``` containing the  percentage of the 0.1 million random-indexed alt-text text samples in the $ind\_i^{th}$ parquet file that triggered a P_hateful/P_targeted/P_aggressive value of > 0.5 by the pysentimento detector (See ```np.mean(res_mat_i>0.5,axis=0)*100``` in the cells above)

d) ```alt_text_{ind_i}.npy``` : $N_{parquet}$ _alt-text_ files of shape ```(100000, 1)``` in the naming-format of ```alt_text_{ind_i}.npy``` pertaining to the 0.1 million random-indexed textual row-contents of the $ind\_i^{th}$ parquet file (in the TEXT field)

e) ```qfr_400m_2Ben.npy```: A ($N_{parquet}$, 3) shaped numpy file that contains the parquet-file level mean-hate content.
```
# Code reference:

  ind_i=file_i.split('/')[-2]+'_'+file_i.split('/')[-1].split('-')[1]
  np.save(f'./{RESULT_DIR}/index_random_{ind_i}.npy',ind_random_i)
  np.save(f'./{RESULT_DIR}/prob_hate_{ind_i}.npy',prob_hate_i)
  np.save(f'./{RESULT_DIR}/qfr_file_{ind_i}.npy',qfr_file_i)
  np.save(f'./{RESULT_DIR}/alt_text_{ind_i}.npy',texts_np_i)
  
np.save(f'./{RESULT_DIR}/qfr_400m_2Ben.npy',qfr_all)
```
2: ```df_parquet_400m_2b.csv```: A 160 x 4 shaped input file-name dataframe whose ```file_loc``` column has the precise order of the 160 (=32 + 128) parquet files that will be used to index the results.

3: ```df_qfr_filewise_400M_2B.csv```: A 160 x 3 output QFR dataframe whose rows map to the 160 parquet files listed in the ```file_loc``` column of ```df_parquet_400m_2b.csv``` and the columns map to the QFR at 0.5 values mapping to ```[P_hateful,P_targeted,P_aggressive]```

4: ```df_summary_filewise_400M_2B.csv``` : A summary (160, 7) shaped dataframe that is a concatenation of ```df_parquet_400m_2b.csv``` & ```df_qfr_filewise_400M_2B.csv``` created by:

```
url_parquet_list='https://raw.githubusercontent.com/vinayprabhu/hate_scaling/main/data/nlp_hate/df_parquet_400m_2b.csv'
url_qfr_filewise='https://raw.githubusercontent.com/vinayprabhu/hate_scaling/main/data/nlp_hate/df_qfr_filewise_400M_2B.csv'
df_parquet_list=pd.read_csv(url_parquet_list)
df_qfr=pd.read_csv(url_qfr_filewise)
df_comb=pd.concat([df_parquet_list,df_qfr],axis=1)
```
