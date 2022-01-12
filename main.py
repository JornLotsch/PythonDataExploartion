#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 12 11:20:34 2022

@author: joern
"""

import pandas as pd
import numpy as np

pfad_o = "/home/joern/Aktuell/RiechenVerduennungAbstand/"
pfad_u1 = "09Originale/"
filename = "data_anne_huster_dec_2021.xlsx"
sheetname = "data"

dfRiechenVerduennungAbstand = pd.read_excel(pfad_o + pfad_u1 + filename, sheet_name = sheetname)
dfRiechenVerduennungAbstand.columns
dfRiechenVerduennungAbstand.index 
xx = dfRiechenVerduennungAbstand.set_index(["Sample ID","sex_0f", "YOB"])
mxx = xx["BMI"].mean(level = ["sex_0f", "YOB"])
xx.index.is_unique
print(xx.index)


data = dfRiechenVerduennungAbstand.BMI * 1
#data[12]= -11

explore_tukey_lop(data=data)
