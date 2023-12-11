# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 14:28:40 2023

@author: HP
"""
# IMPORT ALL THE REQUIRED LIBRARIES
import pandas as pd
import matplotlib.pyplot as plt
# IMPORT THE DATA YOU ARE GOING TO DO THE VISUALIZATION ON
ins=pd.read_csv("insurance.csv")
# PERFORMING SOME EDA ON THE DATA
ins.head(5)
print(ins.shape)
print(ins.columns)
print(ins.dtypes)
ins.describe()
ins.isnull().sum()
# plotting various kinds of graphs
def histogram(hist):
    '''THIS FUNCTION PLOTS THE DISTRIBUTION OF AGES OF THE CUSTOMERS OF THE 
       INSURANCE COMPANY'''
    plt.hist(hist, bins=10, range=(10,70), color='Purple')
    plt.xlabel("AGE")
    plt.ylabel("DISTRIBUTION")
    plt.title("Age Distribution")
    plt.figure(figsize=(4,6))
    plt.show()
def line():
    '''THIS FUNCTION PLOTS THE DISTRIBUTION OF AGES OF THE CUSTOMERS OF THE 
       INSURANCE COMPANY'''
    plt.plot(ins['age'][:10], ins['charges'][:10], label='AGE',color='purple')
    plt.plot(ins['bmi'][:10], ins['charges'][:10], label='BMI',color='orange')
    plt.xlabel("AGE AND BMI")
    plt.ylabel('CHARGES')
    plt.title('Fluctuation of charges according to different factors')
    plt.legend(loc=4)
    plt.show()
def barplot():
    '''THIS FUNCTION PLOTS A BAR PLOT SHOWING THE COUNT OF MALE AND 
    FEMALE CUSTOMERS'''
    ins['sex'].value_counts().plot(kind='bar', color='red')
    plt.ylabel("COUNT")
    plt.title("NUMBER OF MALES AND FEMALES")
    plt.show()
def piechart():
    '''THIS FUNCTION SHOWS THE DISTRIBUTION OF THE CUSTOMERS IN DIFFERENT 
    REGIONS'''
    ins['region'].value_counts().plot(kind='pie',autopct='%1.1f%%')
    plt.title("Distribution over regions")
    plt.show()
def barsmoker():
    '''THIS FUNCTION SHOWS THE NUMBER OF SMOKERS IN THE CUSTOMERS'''
    ins['smoker'].value_counts().plot(kind='bar',color='cyan')
    plt.xlabel("Smoker_or_not")
    plt.ylabel("Count")
    plt.show()
histogram(ins['age'])
line()
barplot()
piechart()
barsmoker()
