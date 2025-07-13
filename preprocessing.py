import pandas as pd
import json
import numpy as np
from rapidfuzz import fuzz
import phonetics
from tqdm import tqdm
import time
import re
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch

class Preprocessing:
    def __init__(self,df:pd.DataFrame):
        self.df=df
        self.honorifics = [
            'sri', 'shree', 'shri', 'sree', 'shre', 'sh', 'srii', 'siri', 'sr', 'sire', 'shi', 'shiri',
            'srimati', 'shrimati', 'shreemati', 'sreemati', 'smt', 'smtji', 'smti', 'smati', 'srmt', 'sirmati',
            'shreemti', 'shirmati', 'shrimti', 'sreemti',
            'late', 'lt', 'lte', 'lat', 'lete', 'laet', 'let', 'late shri', 'late smt', 'lt shri', 'late mr',
            'mr', 'mister', 'misr', 'mtr', 'mstr', 'mrr', 'mear',
            'mrs', 'missus', 'missis', 'mrss', 'mres', 'mers', 'mearrs'
        ]
        self.honorific_pattern = re.compile(rf"\b({'|'.join(self.honorifics)})\b", flags=re.IGNORECASE)
        # Register tqdm with pandas
        tqdm.pandas()
        self.model_name = "ai4bharat/indictrans2-en-indic-1B"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name, trust_remote_code=True)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(self.device)

        # Language tags
        self.src_lang = "eng_Latn"
        self.tgt_lang = "hin_Deva"
    

    def apply_with_progress(self,obj, func, func_name=None, axis=None):
        """
        Applies a function with tqdm progress bar and timing.
        Supports both Series (e.g., a column) and DataFrame.

        Parameters:
            obj (pd.DataFrame or pd.Series): DataFrame or Series.
            func (callable): The function to apply.
            func_name (str): Optional name to display.
            axis (int or None): 1=row-wise, 0=column-wise for DataFrames; None for Series.

        Returns:
            Transformed DataFrame or Series.
        """
        name = func_name or func.__name__
        print(f"\n🟢 Running function: {name}")
        start = time.time()

        if isinstance(obj, pd.Series):
            result = obj.progress_apply(func)
        else:
            result = obj.progress_apply(func, axis=axis)

        end = time.time()
        print(f"✅ Completed {name} in {end - start:.2f} seconds")
        return result

    def normalize_name(self,name):
        return ' '.join(str(name).lower().strip().split())
    
    def put_nan(self,row):
        if row['LASTNAME_EN']=='nan':
            row['LASTNAME_EN']=np.nan
        if row['RLN_L_NM_EN']=='nan':
            row['RLN_L_NM_EN']=np.nan
    
    def fix_voter_lastname(self,row):
        fm_name = str(row['FM_NAME_EN']) if pd.notna(row['FM_NAME_EN']) else ''
        lastname = row['LASTNAME_EN']
        rln_fname = row['RLN_FM_NM_EN']
        rln_lastname = row['RLN_L_NM_EN']

        # If LASTNAME_EN is missing, FM_NAME_EN is a single word, and RLN_L_NM_EN is present
        if pd.isna(lastname) and len(fm_name.strip().split()) == 1 and pd.notna(rln_lastname):
            row['LASTNAME_EN'] = rln_lastname
            return row
        
        if pd.isna(lastname) and len(fm_name.strip().split()) == 1 and pd.isna(rln_lastname) and len(rln_fname.strip().split()) > 1:
            row['LASTNAME_EN'] = rln_fname.split()[-1]
            row['RLN_L_NM_EN'] = rln_fname.split()[-1]
            row['RLN_FM_NM_EN'] = ' '.join(rln_fname.split()[:-1])
            return row
        
        if pd.isna(lastname) and len(fm_name.strip().split()) > 1:
            row['LASTNAME_EN']=fm_name.split()[-1]
            row['FM_NAME_EN']=' '.join(fm_name.split()[:-1])
            return row

        return row

    
    def fix_relative_lastname(self,row):
        rln_fm_name = str(row['RLN_FM_NM_EN']) if pd.notna(row['RLN_FM_NM_EN']) else ''
        rln_lastname = row['RLN_L_NM_EN']
        firstname = row["FM_NAME_EN"]
        lastname = row['LASTNAME_EN']

        # If RLN_L_NM_EN is missing, RLN_FM_NM_EN is a single word, and LASTNAME_EN is present
        if pd.isna(rln_lastname) and len(rln_fm_name.strip().split()) == 1 and pd.notna(lastname):
            row['RLN_L_NM_EN'] = lastname
            return row 
        
        if pd.isna(rln_lastname) and len(rln_fm_name.strip().split()) > 1:
            row['RLN_L_NM_EN'] = rln_fm_name.split()[-1]
            row['RLN_FM_NM_EN'] = ' '.join(rln_fm_name.split()[:-1])
            return row
        
        if pd.isna(rln_lastname) and len(rln_fm_name.strip().split()) == 1 and pd.isna(lastname) and len(firstname.strip().split())>1:
            row['LASTNAME_EN'] = firstname.split()[-1]
            row['FM_NAME_EN'] = ' '.join(firstname.split()[:-1])
            row['RLN_L_NM_EN']=row['LASTNAME_EN']
            return row

        return row
    
    def infer_missing_rln_lastname(self,row):
        lastname = str(row['LASTNAME_EN']).strip().lower() if pd.notna(row['LASTNAME_EN']) else ''
        rln_lastname = row['RLN_L_NM_EN']
        rln_type = row['RLN_TYPE']  # 'F' for father, 'M' for mother
        gender = row['GENDER']      # 'M' for male, 'F' for female

        # Handle RLN_L_NM_EN based on voter's LASTNAME_EN
        if pd.isna(rln_lastname):
            # Case 1: LASTNAME_EN has 'kumar' or 'kumari' and RLN_TYPE is F (father)
            if (rln_type == 'F' or rln_type == 'H') and lastname in ['kumar', 'kumari']:
                row['RLN_L_NM_EN'] = 'kumar'

            # Case 2: LASTNAME_EN has 'kumar' or 'kumari' and RLN_TYPE is M (mother)
            elif rln_type == 'M' and lastname in ['kumar', 'kumari']:
                row['RLN_L_NM_EN'] = 'kumari'

        # Handle LASTNAME_EN based on relative's RLN_L_NM_EN
        if pd.isna(row['LASTNAME_EN']) and pd.notna(rln_lastname):
            rln_lastname_lower = str(rln_lastname).strip().lower()

            # Case 3: RLN_L_NM_EN has 'kumar' or 'kumari' and voter is male
            if gender == 'M' and rln_lastname_lower in ['kumar', 'kumari']:
                row['LASTNAME_EN'] = 'kumar'

            # Case 4: RLN_L_NM_EN has 'kumar' or 'kumari' and voter is female
            elif gender == 'F' and rln_lastname_lower in ['kumar', 'kumari']:
                row['LASTNAME_EN'] = 'kumari'

        return row
    




    # Function to fix names repeated in lastname and surname like baby kumari|baby kumari
    def fix_full_name_voter(self,row):
        if row['FM_NAME_EN'] == row['LASTNAME_EN']:
            parts = row['FM_NAME_EN'].split()
            if len(parts) > 1:
                row['FM_NAME_EN'] = ' '.join(parts[:-1])
                row['LASTNAME_EN'] = parts[-1]
        return row

    def fix_full_name_relative(self,row):
        if row['RLN_FM_NM_EN'] == row['RLN_L_NM_EN']:
                parts = row['RLN_FM_NM_EN'].split()
                if len(parts) > 1:
                    row['RLN_FM_NM_EN'] = ' '.join(parts[:-1])
                    row['RLN_L_NM_EN'] = parts[-1]
        return row

    # Clearing redundant names like Dhaval Kumar| Kumar to Dhaval|kumar
    def clean_redundant_names_voter(self,row):
        first_clean = self.normalize_name(row['FM_NAME_EN'])
        last_clean = self.normalize_name(row['LASTNAME_EN'])

        first_parts = first_clean.split()
        last_parts = last_clean.split()

        # Case 1: full match with same length
        if first_clean == last_clean and len(row['FM_NAME_EN']) == len(row['LASTNAME_EN']):
            return row

        # Case 2: last_name is suffix of first_name
        if len(last_parts) == 1 and first_parts and first_parts[-1] == last_parts[0]:
            first_parts = first_parts[:-1]  # remove last word from first_name

        # Case 3: overlapping name part
        elif first_parts and last_parts and first_parts[-1] == last_parts[0]:
            last_parts = last_parts[1:]

        row['FM_NAME_EN'] = ' '.join(first_parts)
        row['LASTNAME_EN'] = ' '.join(last_parts)

        return row

    def clean_redundant_names_relative(self,row):
        first_clean = self.normalize_name(row['RLN_FM_NM_EN'])
        last_clean = self.normalize_name(row['RLN_L_NM_EN'])

        first_parts = first_clean.split()
        last_parts = last_clean.split()

        # Case 1: full match with same length
        if first_clean == last_clean and len(row['RLN_FM_NM_EN']) == len(row['RLN_L_NM_EN']):
            return row

        # Case 2: last_name is suffix of first_name
        if len(last_parts) == 1 and first_parts and first_parts[-1] == last_parts[0]:
            first_parts = first_parts[:-1]  # remove last word from first_name

        # Case 3: overlapping name part
        elif first_parts and last_parts and first_parts[-1] == last_parts[0]:
            last_parts = last_parts[1:]

        row['RLN_FM_NM_EN'] = ' '.join(first_parts)
        row['RLN_L_NM_EN'] = ' '.join(last_parts)

        return row

    # Correcting if part of first name in last name shashi | bhushan kumar sinh -> shashi bhushan kumar | sinh
    def clean_last_names_voter(self,row):
        first_parts=row['FM_NAME_EN'].split()
        last_parts=row['LASTNAME_EN'].split()

        if len(last_parts)>1 and first_parts!=last_parts:

            row['FM_NAME_EN'] = ' '.join(first_parts)+' '+' '.join(last_parts[:-1])
            row['LASTNAME_EN'] = last_parts[-1]

        return row
    
    def clean_last_names_relative(self,row):
        first_parts=row['RLN_FM_NM_EN'].split()
        last_parts=row['RLN_L_NM_EN'].split()

        if len(last_parts)>1 and first_parts!=last_parts:

            row['RLN_FM_NM_EN'] = ' '.join(first_parts)+' '+' '.join(last_parts[:-1])
            row['RLN_L_NM_EN'] = last_parts[-1]

        return row

    # correcting if parts of last name in first name: amresh kumar rajak|amresh->amresh kumar|rajak
    def clean_firstname_voter(self,row):
        first_name_parts=row["FM_NAME_EN"].split()
        last_name_parts=row['LASTNAME_EN'].split()
        if first_name_parts==last_name_parts:
            return row
        if len(last_name_parts)==1 and len(first_name_parts)>1:
            if last_name_parts[0] not in first_name_parts:
                return row
            else:
                row['LASTNAME_EN']=first_name_parts[-1]
                row['FM_NAME_EN']=" ".join(first_name_parts[:-1])
        return row

    def clean_firstname_relative(self,row):
        first_name_parts=row["RLN_FM_NM_EN"].split()
        last_name_parts=row['RLN_L_NM_EN'].split()
        if first_name_parts==last_name_parts:
            return row
        if len(last_name_parts)==1 and len(first_name_parts)>1:
            if last_name_parts[0] not in first_name_parts:
                return row
            else:
                row['RLN_L_NM_EN']=first_name_parts[-1]
                row['RLN_FM_NM_EN']=" ".join(first_name_parts[:-1])
        return row

    # Removing one letter surnames
    def correct_last_name_one_letter_voter(self,row):
        lastname_parts = row['LASTNAME_EN'].strip()
        firstname_parts = row['FM_NAME_EN'].strip().split()

        if len(lastname_parts) == 1:
            if len(firstname_parts)==1:
                row['LASTNAME_EN']=lastname_parts
            elif len(firstname_parts)>1:
                row['LASTNAME_EN']= firstname_parts[-1]
                row['FM_NAME_EN']=' '.join(firstname_parts[:-1])
        
        return row

    def correct_last_name_one_letter_relative(self,row):
        lastname_parts = row['RLN_FM_NM_EN'].strip()
        firstname_parts = row['RLN_L_NM_EN'].strip().split()

        if len(lastname_parts) == 1:
            if len(firstname_parts)==1:
                row['RLN_L_NM_EN']=lastname_parts
            elif len(firstname_parts)>1:
                row['RLN_L_NM_EN']= firstname_parts[-1]
                row['RLN_FM_NM_EN']=' '.join(firstname_parts[:-1])
        return row

    # Correcting if voter's name in relative or vice versa
    def check_voter_in_relative(self,row):
        voter_fullname=row['FM_NAME_EN']+' '+row['LASTNAME_EN']
        relative_fullname=row['RLN_FM_NM_EN']+' '+row['RLN_L_NM_EN']
        voter_parts=voter_fullname.split()
        relative_parts=relative_fullname.split()
        if relative_fullname==voter_fullname:
            return row

        if relative_parts[0] in voter_parts[0] or voter_parts[0] in relative_parts[0]:
            if len(relative_parts[0])!=len(voter_parts[0]):
                return row

        if voter_fullname==row['RLN_FM_NM_EN'] or relative_fullname==row['FM_NAME_EN']:
            return row

        if relative_parts[0]==voter_parts[0] and (relative_parts[1]=="kumar" and voter_parts[1]=="kumari") or (relative_parts[1]=="kumari" and voter_parts[1]=="kumar"):
            return row

        if row["RLN_TYPE"]=='F' or row["RLN_TYPE"]=='H':
            
            if voter_parts[1]==relative_parts[0]:
                return row

        if voter_fullname in relative_fullname:
            relative_fullname=relative_fullname.replace(voter_fullname,"").strip()
            rlt_name=relative_fullname.split()
            row['RLN_FM_NM_EN']=' '.join(rlt_name[:-1])
            row['RLN_L_NM_EN']=rlt_name[-1]

        return row

    def check_relative_in_voter(self,row):
        voter_fullname=row['FM_NAME_EN']+' '+row['LASTNAME_EN']
        relative_fullname=row['RLN_FM_NM_EN']+' '+row['RLN_L_NM_EN']
        voter_parts=voter_fullname.split()
        relative_parts=relative_fullname.split()
        if relative_fullname==voter_fullname:
            return row

        if relative_parts[0] in voter_parts[0] or voter_parts[0] in relative_parts[0]:
            if len(relative_parts[0])!=len(voter_parts[0]):
                return row

        if relative_parts[0]==voter_parts[0] and (relative_parts[1]=="kumar" and voter_parts[1]=="kumari") or (relative_parts[1]=="kumari" and voter_parts[1]=="kumar"):
            return row

        if relative_parts[0]==voter_fullname or voter_parts[0]==relative_fullname:
            return row

        if row["RLN_TYPE"]=='F' or row["RLN_TYPE"]=='H' or row['RLN_TYPE']=='M':
            if voter_parts[1]==relative_parts[0]:
                return row

        if relative_fullname in voter_fullname:
            voter_fullname=voter_fullname.replace(relative_fullname,"").strip()
            voter_name=voter_fullname.split()
            row['FM_NAME_EN']=' '.join(voter_name[:-1])
            row['LASTNAME_EN']=voter_name[-1]
        return row

    # join splitted surnames

    def join_surnames_voter(self,row):
        with open('Data/my_lst.json') as f:
            lst = json.load(f)
        first_parts=row['FM_NAME_EN'].split()
        last_parts=row['LASTNAME_EN'].strip()
        dict_fuzz={}
        joined_word=first_parts[-1]+""+last_parts
        if len(first_parts)==1:
            return row
        if joined_word.lower() in lst:
            row['LASTNAME_EN']=joined_word
            row['FM_NAME_EN']=' '.join(first_parts[:-1])
            return row
        for i in lst:
            ratio=fuzz.token_set_ratio(i,joined_word)
            if ratio>=80:
                if i=="kumari" and ratio<85:
                    ratio-=10
            if ratio>=80:
                dict_fuzz[i]=ratio

        
        if dict_fuzz:
            sorted_dict = dict(sorted(dict_fuzz.items(), key=lambda item: item[1]))
            psurname= next(iter(sorted_dict))
            
            row['LASTNAME_EN']=psurname
            row['FM_NAME_EN']=' '.join(first_parts[:-1])
        else:
            for i in lst:
                phonetic_match=phonetics.dmetaphone(i)[0] == phonetics.dmetaphone(joined_word)[0]
                if phonetic_match:
                    row['LASTNAME_EN']=i
                    row['FM_NAME_EN']=' '.join(first_parts[:-1])
                    break
        return row

    def join_surnames_relative(self,row):
        with open('Data/my_lst.json') as f:
            lst = json.load(f)
        first_parts=row['RLN_FM_NM_EN'].split()
        last_parts=row['RLN_L_NM_EN'].strip()
        dict_fuzz={}
        joined_word=first_parts[-1]+""+last_parts
        if len(first_parts)==1:
            return row
        if joined_word.lower() in lst:
            row['RLN_L_NM_EN']=joined_word
            row['RLN_FM_NM_EN']=' '.join(first_parts[:-1])
            return row
        for i in lst:
            ratio=fuzz.token_set_ratio(i,joined_word)
            if ratio>=80:
                if i=="kumari" and ratio<85:
                    ratio-=10
                if ratio>=80:
                    dict_fuzz[i]=ratio


        if dict_fuzz:
            sorted_dict = dict(sorted(dict_fuzz.items(), key=lambda item: item[1]))
            psurname= next(iter(sorted_dict))
            
            row['RLN_L_NM_EN']=psurname
            row['RLN_FM_NM_EN']=' '.join(first_parts[:-1])
        else:
            for i in lst:
                phonetic_match=phonetics.dmetaphone(i)[0] == phonetics.dmetaphone(joined_word)[0]
                if phonetic_match:
                    row['RLN_L_NM_EN']=i
                    row['RLN_FM_NM_EN']=' '.join(first_parts[:-1])
                    break
        return row

    # Correcting no space between surnames and names

    

    def split_on_kumari_voter(self,row):
        if row['FM_NAME_EN']==row['LASTNAME_EN'] and row['GENDER']=='F':
            if "kumari" in row['LASTNAME_EN']:
                first,middle,last=row['LASTNAME_EN'].partition('kumari')
                if last:
                    row['LASTNAME_EN']=last
                    row['FM_NAME_EN']=''+first+' '+middle
                else:
                    row['FM_NAME_EN']=first
                    row['LASTNAME_EN']=middle
        return row

    def split_on_kumar_voter(self,row):
        if row['FM_NAME_EN']==row['LASTNAME_EN'] and row['GENDER']=='M':
            if "kumar" in row['LASTNAME_EN']:
                first,middle,last=row['LASTNAME_EN'].partition('kumar')
                if last:
                    row['LASTNAME_EN']=last
                    row['FM_NAME_EN']=first+' '+middle
                else:
                    row['FM_NAME_EN']=first
                    row['LASTNAME_EN']=middle
        return row

    def split_on_kumari_relative(self,row):
        if row['RLN_FM_NM_EN']==row['RLN_L_NM_EN'] and (row['RLN_TYPE']=="M" or row['RLN_TYPE']=="W"):
            if "kumari" in row['RLN_L_NM_EN']:
                first,middle,last=row['RLN_L_NM_EN'].partition('kumari')
                if last:
                    row['RLN_L_NM_EN']=last
                    row['RLN_FM_NM_EN']=first+' '+middle
                else:
                    row['RLN_FM_NM_EN']=first
                    row['RLN_L_NM_EN']=middle
        return row

    def split_on_kumar_relative(self,row):
        if row['RLN_FM_NM_EN']==row['RLN_L_NM_EN'] and (row['RLN_TYPE']=="F" or row['RLN_TYPE']=="H"):
            if "kumar" in row['RLN_L_NM_EN']:
                first,middle,last=row['RLN_L_NM_EN'].partition('kumar')
                if last:
                    row['RLN_L_NM_EN']=last
                    row['RLN_FM_NM_EN']=''+first+' '+middle
                else:
                    row['RLN_FM_NM_EN']=first
                    row['RLN_L_NM_EN']=middle
        return row

    def clean_kumari_voter(self,row):
        if 'kumari' in row['LASTNAME_EN'].lower() and row['GENDER'] == 'F':
            if row['LASTNAME_EN'].lower() == 'kumari':
                return row  # already clean

            before, sep, after = row['LASTNAME_EN'].lower().partition('kumari')

            if before == '':
                # name starts with "kumari" (like "kumarif", "kumarii")
                if len(after) <= 1:
                    row['LASTNAME_EN'] = 'kumari'
                else:
                    row['LASTNAME_EN'] = after
                    row['FM_NAME_EN'] = row['FM_NAME_EN'] + ' kumari'
            elif after == '':
                # name ends with "kumari" (like "uekumari")
                row['LASTNAME_EN'] = 'kumari'
                row['FM_NAME_EN'] = before + row['FM_NAME_EN']
            else:
                # kumari in the middle (rare case)
                row['LASTNAME_EN'] = after
                row['FM_NAME_EN'] = row['FM_NAME_EN'] + ' kumari'
        return row

    def clean_kumar_voter(self,row):
        if 'kumar' in row['LASTNAME_EN'].lower() and row['GENDER'] == 'M':
                if row['LASTNAME_EN'].lower() == 'kumar':
                    return row  # already clean

                before, sep, after = row['LASTNAME_EN'].lower().partition('kumar')

                if before == '':
                    # name starts with "kumari" (like "kumarif", "kumarii")
                    if len(after) <= 1:
                        row['LASTNAME_EN'] = 'kumar'
                    else:
                        row['LASTNAME_EN'] = after
                        row['FM_NAME_EN'] = row['FM_NAME_EN'] + ' kumar'
                elif after == '':
                    # name ends with "kumari" (like "uekumari")
                    row['LASTNAME_EN'] = 'kumar'
                    row['FM_NAME_EN'] = before + row['FM_NAME_EN']
                else:
                    # kumari in the middle (rare case)
                    row['LASTNAME_EN'] = after
                    row['FM_NAME_EN'] = row['FM_NAME_EN'] + ' kumar'
        return row

    def clean_kumari_relative(self,row):
        if 'kumari' in row['RLN_L_NM_EN'].lower() and (row['RLN_TYPE'] == 'M' or row['RLN_TYPE']=='W'):
                if row['RLN_L_NM_EN'].lower() == 'kumari':
                    return row  # already clean

                before, sep, after = row['RLN_L_NM_EN'].lower().partition('kumari')

                if before == '':
                    # name starts with "kumari" (like "kumarif", "kumarii")
                    if len(after) <= 1:
                        row['RLN_L_NM_EN'] = 'kumari'
                    else:
                        row['RLN_L_NM_EN'] = after
                        row['RLN_FM_NM_EN'] = row['RLN_FM_NM_EN'] + ' kumari'
                elif after == '':
                    # name ends with "kumari" (like "uekumari")
                    row['RLN_L_NM_EN'] = 'kumari'
                    row['RLN_FM_NM_EN'] = before + row['RLN_FM_NM_EN']
                else:
                    # kumari in the middle (rare case)
                    row['RLN_L_NM_EN'] = after
                    row['RLN_FM_NM_EN'] = row['RLN_FM_NM_EN'] + ' kumari'
        return row

    def clean_kumar_relative(self,row):
        if 'kumar' in row['RLN_L_NM_EN'].lower() and (row['RLN_TYPE'] == 'F' or row['RLN_TYPE']=='H'):
                if row['RLN_L_NM_EN'].lower() == 'kumar':
                    return row  # already clean

                before, sep, after = row['RLN_L_NM_EN'].lower().partition('kumar')

                if before == '':
                    # name starts with "kumari" (like "kumarif", "kumarii")
                    if len(after) <= 1:
                        row['RLN_L_NM_EN'] = 'kumar'
                    else:
                        row['RLN_L_NM_EN'] = after
                        row['RLN_FM_NM_EN'] = row['RLN_FM_NM_EN'] + ' kumar'
                elif after == '':
                    # name ends with "kumari" (like "uekumari")
                    row['RLN_L_NM_EN'] = 'kumar'
                    row['RLN_FM_NM_EN'] = before + row['RLN_FM_NM_EN']
                else:
                    # kumari in the middle (rare case)
                    row['RLN_L_NM_EN'] = after
                    row['RLN_FM_NM_EN'] = row['RLN_FM_NM_EN'] + ' kumar'
        return row

    # Replacing wrong spelling of kumar and kumari

    def replace_wrong_spelling_kumar_voter(self,row):
        with open('Data/result_kumar.json') as f:
            result_kumar = json.load(f)
        l_name = row["LASTNAME_EN"]
        if row["GENDER"] == 'M' and l_name in result_kumar['kumar']:
            row["LASTNAME_EN"] = "kumar"
        return row

    def replace_wrong_spelling_kumar_relative(self,row):
        with open('Data/result_kumar.json') as f:
            result_kumar = json.load(f)
        l_name = row["RLN_L_NM_EN"]
        if (row["RLN_TYPE"] == 'H' or row["RLN_TYPE"] == 'F') and l_name in result_kumar['kumar']:
            row["RLN_L_NM_EN"] = "kumar"
        return row

    def replace_wrong_spelling_kumari_voter(self,row):
        with open('Data/result_kumari.json') as f:
            result_kumari = json.load(f)
        l_name = row["LASTNAME_EN"]
        if row["GENDER"] == 'F' and l_name in result_kumari['kumari']:
            row["LASTNAME_EN"] = "kumari"
        return row

    def replace_wrong_spelling_kumari_relative(self,row):
        with open('Data/result_kumari.json') as f:
            result_kumari = json.load(f)
        l_name = row["RLN_L_NM_EN"]
        if (row["RLN_TYPE"] == 'W' or row["RLN_TYPE"] == 'M') and l_name in result_kumari['kumari']:
            row["RLN_L_NM_EN"] = "kumari"
        return row

    def replace_wrong_spelling_devi_voter(self,row):
        with open('Data/result_devi.json') as f:
            result_devi = json.load(f)
        l_name = row["LASTNAME_EN"]
        if row["GENDER"] == 'F' and l_name in result_devi['devi']:
            row["LASTNAME_EN"] = "devi"
        return row

    def replace_wrong_spelling_devi_relative(self,row):
        with open('Data/result_devi.json') as f:
            result_devi = json.load(f)
        l_name = row["RLN_L_NM_EN"]
        if (row["RLN_TYPE"] == 'M' or row["RLN_TYPE"] == 'W') and l_name in result_devi['devi']:
            row["RLN_L_NM_EN"] = "devi"
        return row

    def correct_lastnames_rowwise_voter(self,row,typo_dict):
        voter_lastname = str(row['LASTNAME_EN']).strip().lower()
        rln_lastname = str(row['RLN_L_NM_EN']).strip().lower()

        for key,values in typo_dict.items():
            if voter_lastname in values:
                if row['FM_NAME_EN']==row['LASTNAME_EN']:
                    return row

                # if row['FINAL SURNAME']=="MUSLIM":
                #     return row

                # CASE 1: spelling mistake of KEY A | KEY A
                if key==rln_lastname.upper():
                    row['LASTNAME_EN']=key.lower()
                    return row

                # CASE 2: spelling mistake of KEY A | spelling mistake of KEY A
                elif rln_lastname in values:
                    row['LASTNAME_EN']=key.lower()
                    row['RLN_L_NM_EN']=key.lower()
                    return row

                #CASE 3: spelling mistake of KEY A also spelling mistake of Key B| Key B
                elif rln_lastname.upper() in typo_dict.keys() and rln_lastname.upper()!=key and voter_lastname in typo_dict[rln_lastname.upper()]:
                    row['LASTNAME_EN']=rln_lastname.lower()
                    return row
                
                # CASE 4: spelling mistake of KEY B | spelling mistake of KEY C and CASE 4: spelling mistake of KEY A | KEY B
                elif rln_lastname not in values:
                    row['LASTNAME_EN']=key.lower()
                    return row

        return row

    def correct_lastnames_rowwise_relative(self,row,typo_dict):
        voter_lastname = str(row['LASTNAME_EN']).strip().lower()
        rln_lastname = str(row['RLN_L_NM_EN']).strip().lower()

        for key,values in typo_dict.items():
            if rln_lastname in values:
                if row['RLN_FM_NM_EN']==row['RLN_L_NM_EN']:
                    return row

                # if row['FINAL SURNAME']=="MUSLIM":
                #     return row

                # CASE 1: KEY A | spelling mistake of KEY A
                if key==voter_lastname.upper():
                    row['RLN_L_NM_EN']=key.lower()
                    return row

                #CASE 3: spelling mistake of KEY A also spelling mistake of Key B| Key B
                elif voter_lastname.upper() in typo_dict.keys() and voter_lastname.upper()!=key and rln_lastname in typo_dict[voter_lastname.upper()]:
                    row['RLN_L_NM_EN']=voter_lastname.lower()
                    return row
                
                # CASE 2: any key | spelling mistake of KEY A
                elif rln_lastname in values:
                    # row['LASTNAME_EN']=key
                    row['RLN_L_NM_EN']=key.lower()
                    return row

        return row
    
    
    
    def remove_honorifics_voter(self,row):
        def remove_honorifics(text):
            """Remove standalone honorifics only (not substrings in names)."""
            if isinstance(text, str):
                # Replace multiple spaces with single after honorific removal
                cleaned = self.honorific_pattern.sub('', text)
                return re.sub(r'\s+', ' ', cleaned).strip()
            return text
        row["FM_NAME_EN"]=remove_honorifics(row["FM_NAME_EN"])
        return row

        
        

    def remove_honorifics_relative(self,row):
        def remove_honorifics(text):
            """Remove standalone honorifics only (not substrings in names)."""
            if isinstance(text, str):
                # Replace multiple spaces with single after honorific removal
                cleaned = self.honorific_pattern.sub('', text)
                return re.sub(r'\s+', ' ', cleaned).strip()
            return text
        row["RLN_FM_NM_EN"]=remove_honorifics(row["RLN_FM_NM_EN"])
        return row


    def translate_batch(self,batch_sentences):
        inputs = self.tokenizer(batch_sentences, return_tensors="pt", padding=True, truncation=True).to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(**inputs)
        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
    
    # following two functions to be used for creating data for model training, no need in family clustering

    def correct_hindi_names_voter(self,data):
        batch_size = 64
        translated_first_names = []
        translated_surnames = []
        for i in tqdm(range(0, len(data), batch_size)):
            batch_df = data.iloc[i:i+batch_size]

            # Prepare batch input in required format
            first_batch = [f"{self.src_lang} {self.tgt_lang} {name}" for name in batch_df["FM_NAME_EN"].fillna("")]
            surname_batch = [f"{self.src_lang} {self.tgt_lang} {name}" for name in batch_df["LASTNAME_EN"].fillna("")]

            translated_first_names.extend(self.translate_batch(first_batch))
            translated_surnames.extend(self.translate_batch(surname_batch))
        data["FM_NAME_V1"] = translated_first_names
        data["LASTNAME_V1"] = translated_surnames

    def correct_hindi_names_relative(self,data):
        batch_size = 64
        translated_first_names = []
        translated_surnames = []
        for i in tqdm(range(0, len(data), batch_size)):
            batch_df = data.iloc[i:i+batch_size]

            # Prepare batch input in required format
            first_batch = [f"{self.src_lang} {self.tgt_lang} {name}" for name in batch_df["RLN_FM_NM_EN"].fillna("")]
            surname_batch = [f"{self.src_lang} {self.tgt_lang} {name}" for name in batch_df["RLN_L_NM_EN"].fillna("")]

            translated_first_names.extend(self.translate_batch(first_batch))
            translated_surnames.extend(self.translate_batch(surname_batch))
        data["RLN_FM_NM_V1"] = translated_first_names
        data["RLN_L_NM_V1"] = translated_surnames




    

    
    def start_preprocessing(self):
        data=self.df.copy()
        with open('Data/result_wrong_surnames_unique.json') as f:
            self.result_wrong_surnames_unique = json.load(f)
        
        
        data = data[~data['LASTNAME_EN'].str.contains('AC_NO', case=False, na=False)]
        data = data[~data['RLN_L_NM_EN'].str.contains('AC_NO', case=False, na=False)]

        data = self.apply_with_progress(data, self.fix_voter_lastname,axis=1)
        
        data = self.apply_with_progress(data, self.fix_relative_lastname,axis=1)
        
        data = self.apply_with_progress(data, self.infer_missing_rln_lastname, axis=1)
        

        data['FM_NAME_EN'] = self.apply_with_progress(data['FM_NAME_EN'], self.normalize_name)
        data['LASTNAME_EN'] = self.apply_with_progress(data['LASTNAME_EN'], self.normalize_name)
        data['RLN_FM_NM_EN'] = self.apply_with_progress(data['RLN_FM_NM_EN'], self.normalize_name)
        data['RLN_L_NM_EN'] = self.apply_with_progress(data['RLN_L_NM_EN'], self.normalize_name)

        #data = self.apply_with_progress(data,self.put_nan,axis=1)
        
        data = self.apply_with_progress(data, self.fix_full_name_voter,axis=1)
        data = self.apply_with_progress(data, self.fix_full_name_relative,axis=1)
        data = self.apply_with_progress(data, self.clean_redundant_names_voter,axis=1)
        data = self.apply_with_progress(data, self.clean_redundant_names_relative,axis=1)
        data = self.apply_with_progress(data, self.clean_last_names_voter,axis=1)
        data = self.apply_with_progress(data, self.clean_last_names_relative,axis=1)
        data = self.apply_with_progress(data, self.clean_firstname_voter,axis=1)
        data = self.apply_with_progress(data, self.clean_firstname_relative,axis=1)
        data = self.apply_with_progress(data, self.correct_last_name_one_letter_voter,axis=1)
        data = self.apply_with_progress(data, self.correct_last_name_one_letter_relative,axis=1)
        data = self.apply_with_progress(data, self.check_voter_in_relative,axis=1)
        data = self.apply_with_progress(data, self.check_relative_in_voter,axis=1)
        data = self.apply_with_progress(data, self.join_surnames_voter,axis=1)
        data = self.apply_with_progress(data, self.join_surnames_relative,axis=1)
        data = self.apply_with_progress(data, self.split_on_kumari_voter,axis=1)
        data = self.apply_with_progress(data, self.split_on_kumar_voter,axis=1)
        data = self.apply_with_progress(data, self.split_on_kumari_relative,axis=1)
        data = self.apply_with_progress(data, self.split_on_kumar_relative,axis=1)
        data = self.apply_with_progress(data, self.clean_kumari_voter,axis=1)
        data = self.apply_with_progress(data, self.clean_kumar_voter,axis=1)
        data = self.apply_with_progress(data, self.clean_kumari_relative,axis=1)
        data = self.apply_with_progress(data, self.clean_kumar_relative,axis=1)
        data = self.apply_with_progress(data, self.replace_wrong_spelling_kumar_voter, axis=1)
        data = self.apply_with_progress(data, self.replace_wrong_spelling_kumar_relative, axis=1)
        data = self.apply_with_progress(data, self.replace_wrong_spelling_kumari_voter, axis=1)
        data = self.apply_with_progress(data, self.replace_wrong_spelling_kumari_relative, axis=1)
        data = self.apply_with_progress(data, self.replace_wrong_spelling_devi_voter, axis=1)
        data = self.apply_with_progress(data, self.replace_wrong_spelling_devi_relative, axis=1)
        data = self.apply_with_progress(data, self.remove_honorifics_voter,axis=1)
        data = self.apply_with_progress(data, self.remove_honorifics_relative,axis=1)

        data = self.apply_with_progress(
            data,
            lambda row: self.correct_lastnames_rowwise_voter(row, self.result_wrong_surnames_unique),
            func_name="correct_lastnames_rowwise_voter",
            axis=1
        )

        data = self.apply_with_progress(
            data,
            lambda row: self.correct_lastnames_rowwise_relative(row, self.result_wrong_surnames_unique),
            func_name="correct_lastnames_rowwise_relative",
            axis=1
        )
        self.correct_hindi_names_voter(data)
        self.correct_hindi_names_relative(data)

        # data['C_HOUSE_NO'] = pd.to_numeric(data['C_HOUSE_NO'], errors='coerce')
        data = data.dropna(subset=['FM_NAME_EN','LASTNAME_EN','RLN_FM_NM_EN',"RLN_L_NM_EN",'RLN_TYPE'])
        data = data[~data['FM_NAME_EN'].astype(str).str.isnumeric()]
        data = data[~data['FM_NAME_EN'].apply(lambda x: isinstance(x, (int, float)))]
        data = data[~data['LASTNAME_EN'].astype(str).str.isnumeric()]
        data = data[~data['LASTNAME_EN'].apply(lambda x: isinstance(x, (int, float)))]
        data = data[~data['RLN_FM_NM_EN'].astype(str).str.isnumeric()]
        data = data[~data['RLN_FM_NM_EN'].apply(lambda x: isinstance(x, (int, float)))]
        data = data[~data['RLN_L_NM_EN'].astype(str).str.isnumeric()]
        data = data[~data['RLN_L_NM_EN'].apply(lambda x: isinstance(x, (int, float)))]
        # print(data.head(1))
        # data = data[~(data['FM_NAME_EN'] == "nan")]
        # data = data[~(data['LASTNAME_EN']=="nan")]
        # data = data[~(data['RLN_FM_NM_EN']=="nan")]
        # data = data[~(data['RLN_L_NM_EN']=="nan")]
        # data['full_name_elector']=data['FM_NAME_EN']+' '+data['LASTNAME_EN']
        # data['full_name_relative']=data['RLN_FM_NM_EN']+' '+data['RLN_L_NM_EN']
        # data['full_name_elector'] = data['full_name_elector'].str.replace(r'\s+', '', regex=True)
        # data['full_name_relative'] = data['full_name_relative'].str.replace(r'\s+', '', regex=True)
        # data["family_id"] = -1
        return data
    
if __name__=="__main__":
    df=pd.read_excel("Preprocessed_mydata3.xlsx",engine="openpyxl")
    pre_obj=Preprocessing(df)
    res=pre_obj.start_preprocessing()
    res.to_excel("preprocessed_data.xlsx",index=False)