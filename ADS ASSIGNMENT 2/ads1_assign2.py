'''import all necessary libraries'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import stats

def read_file(filename):
    """
  Read all the csv  
  Parameters:
  - filename : csv selected
  
  Returns:
  - df: dataset
  """
    # reading the files needed
    data = pd.read_csv(filename,sep=',',skiprows=4)
    return data
elec = read_file("API_EG.USE.ELEC.KH.PC_DS2_en_csv_v2_5995551.csv")
agri = read_file("API_AG.LND.AGRI.ZS_DS2_en_csv_v2_5995314.csv")
renew = read_file("API_EG.FEC.RNEW.ZS_DS2_en_csv_v2_5995541.csv")
land = read_file("API_AG.LND.FRST.K2_DS2_en_csv_v2_5995336.csv")
def drop_unwanted(data):
    """
 Drop unwanted columns
  Parameters:
  - df:the dataset
  Returns:
  - new_df : the dataset with dropped unnecessary columns
  """
    new_df=data.drop(['Country Code','Indicator Code','Unnamed: 67'],axis=1)
    new_df=data.loc[:,['Country Name','Indicator Name','2013','2014','2015',
                     '2016','2017','2018','2019','2020','2021']]
    return new_df
new_elec = drop_unwanted(elec)
new_agri = drop_unwanted(agri)
new_renew = drop_unwanted(renew)
new_land = drop_unwanted(land)
print(new_elec)
new_renew.iloc[1,1]
def world_bank_data(file,sel_countries=None):
    """
  Create a transposed dataframe
  Parameters:
  - file : file selected
  - sel_countries : the countries we want the data for
  Returns:
  - df_years : the original dataframe
  - df_countries : the transposed dataframe
  """
    dataframe = file.copy()
    # Extract unique years and countries
    # Create a dataframe with years as columns
    df_years = dataframe.copy()
    # Create a dataframe with countries as columns
    df_countries = dataframe.transpose()
    df_countries=df_countries.rename_axis('Year')
    if sel_countries is not None:
        df_years = df_years[df_years['Country Name'].isin(sel_countries)]
        df_countries.columns=df_countries.iloc[0]
        df_countries=df_countries.drop('Country Name')
        df_countries = df_countries.loc[:, sel_countries]
    return df_years,df_countries
sel_countries=['Africa Eastern and Southern','Africa Western and Central',
               'Yemen, Rep.','Afghanistan','Pakistan','India','South Africa',
               'Zimbabwe','Zambia']
years_df_elec,countries_df_elec = world_bank_data(new_elec,sel_countries)
years_df_agri,countries_df_agri = world_bank_data(new_agri,sel_countries)
years_df_renew,countries_df_renew = world_bank_data(new_renew,sel_countries)
years_df_land,countries_df_land= world_bank_data(new_land,sel_countries)
years_df_agri.set_index('Country Name',inplace=True)
years_df_renew.set_index('Country Name',inplace=True)
years_df_land.set_index('Country Name',inplace=True)
years_df_elec.set_index('Country Name',inplace=True)
print(years_df_agri)
years_df_land.describe()
years_df_agri.describe()
def lineplot():
    """
  Create a lineplot for indicator forest area as indicator
  Parameters:
  - None

  Returns:
  - None
  shows a plot
  """
    global years_df_land
    years_df_land = years_df_land.apply(pd.to_numeric, errors='coerce')
    years_df_land = years_df_land.round(2)
    for country in years_df_land.index:
        plt.plot(years_df_land.columns, years_df_land.loc[country],
                     marker='*',label=country)
    plt.title('Line Plot of Countries Over Years for indicator Forest area (sq. km)')
    plt.xlabel('Year')
    plt.ylabel('Value in percentages')
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))  # Display legend with country names
    plt.grid(True)
    plt.show()
def barplot():
    """
    Create a barplot for showing the consumption of renewable energy between 
    countries
    Parameters:
    - None
    Returns:
    - None
    shows a plot
    """
    bar_new_renew=new_renew[new_renew['Country Name'].isin(['India','Pakistan',
                'Zimbabwe','Zambia','South Africa'])]
    bar_width = 0.35
    index = np.arange(len(bar_new_renew['Country Name']))
    plt.bar(index - bar_width/2, bar_new_renew['2013'], width=bar_width,
            label='2013', color='skyblue')
    plt.bar(index + bar_width/2, bar_new_renew['2020'], width=bar_width,
            label='2020',  color='orange')
    plt.xlabel('Countries')
    plt.ylabel('% Renewable energy consumption')
    plt.title('RENEWABLE ENERGY CONSUMPTION IN DIFFERENT COUNTRIES IN YEAR 2013 and 2020')
    plt.xticks(index, bar_new_renew['Country Name'])
    plt.legend()
    plt.show()
def pieplot():
    """
    Create a pieplot for distrbution
    Parameters:
    - None
    Returns:
    - None
    shows a plot
    """
    global years_df_elec
    # Assuming '2013' column contains numeric or convertible values
    years_df_elec['2013'] = pd.to_numeric(years_df_elec['2013'], errors='coerce')
    # Drop rows with NaN values in '2013' column
    years_df_elec = years_df_elec.dropna(subset=['2013'])
    years_df_elec.reset_index(inplace=True)
    labels=years_df_elec['Country Name']
    plt.pie(years_df_elec[str('2013')],labels=labels,autopct='%1.1f%%',
            startangle=90)
    plt.title('Electric power consumption (kWh per capita) by various countries')
    plt.show()
def merge():
    """
    Merges two datasets
    Parameters:
    - None
    Returns:
    - Merged datasets
    shows a heatmap as well
    """
    lc_land=new_land[['Country Name','2020']]
    lc_renew=new_renew[['Country Name','2020']]
    lc_agri=new_agri[['Country Name','2020']]
    all_df = pd.merge(lc_land,lc_renew,on='Country Name',suffixes=('land','renew'))
    all_merged = pd.merge(all_df,lc_agri,on='Country Name',suffixes=('s','agri'))
    corr = all_merged.corr()
    sns.heatmap(corr,annot= True)
def histo():
    """
    Create a histogram
    Parameters:
    - None
    Returns:
    - None
    shows a plot
    """
    skew_agri = stats.skew(years_df_agri['2021'])
    kurtosis_agri = stats.kurtosis(years_df_agri['2021'])
    print(f"Skewness: {skew_agri}")
    print(f"Kurtosis: {kurtosis_agri}")
    plt.hist(years_df_agri['2021'], bins='auto', alpha=0.7, color='teal', edgecolor='black')
    plt.title(f'''Histogram of year {2021} and Agricultural Land with 
              Skewness={skew_agri:.2f} and Kurtosis={kurtosis_agri:.2f}''')
    plt.xlabel('Agri Land in Percent for year 2021')
    plt.ylabel('Frequency')
    plt.show()
def scatter():
    """
    Create a scatterplot comparing two indicators
    Parameters:
    - None
    Returns:
    - None
    shows a plot
    """
    lc_land=new_land[['2020']]
    lc_agri=new_agri[['2020']]
    plt.scatter(lc_land,lc_agri,color='green',marker='*')
    plt.xlabel('Forest Area(sq. km)')
    plt.ylabel('Agricultural Land(% of land area)')
    plt.title(''''Comparing two indicators wrt to each other for year 2020
              (Forest Area and agricultural land)''')
    plt.show()
    
histo()
lineplot()
scatter()
barplot()
pieplot()
merge()

