import pandas as pd

def build_df(data_path):
    data = pd.read_csv(data_path,sep="\s+")
    dframe=pd.DataFrame(data)
    #Step# is the the new row name
    dframe.index=dframe.iloc[:,0]
    #remove column od Step# and Nan column
    dframe=dframe.drop(["*",'0'],axis=1)
    #concatenate 2 columns in 1
    dframe["ID.1"]+dframe["=.2"]
    #rename columns with the good labels
    dframe.columns=["X(mm)","Y(mm)","Z(mm)","DX","DY","DZ","KinE(MeV)","dE(MeV)","StepLeng","TrackLeng","NextVolume","ProcName"]
    #dframe.head(20)
    
    init_E = dframe.loc[dframe["StepLeng"] == 'initStep', 'DX'][1]
    init_E
    dframe.loc[dframe["StepLeng"] == 'initStep', ['DX', 'DY', 'DZ', 'KinE(MeV)','dE(MeV)', 'StepLeng', 'TrackLeng',\
                                      "NextVolume","ProcName"]]\
    =['0', '0', '0', init_E, '0', '0', '0', 'phantom', 'msc']
    return dframe


def data_file(i):
    return "../Data/E_" + str(i) + ".data"







def clean_df(df):
    #Select interesting values
    df_new = df[['DX', 'DY', 'DZ', 'KinE(MeV)', 'dE(MeV)',\
                 'StepLeng', 'NextVolume', 'ProcName']]

    #Remove unnecessary lines
    df_new = df_new.loc[(df.index != 'Step#') &\
                        (df.index != '*')]

    #Get proper indexing
    df_new=df_new.reset_index()
    df_new.rename(columns={"*": "index"}, inplace=True)

    #Remove unnecessary lines again
    df_new = df_new[~df_new['index'].str.startswith(':-')]

    #Localize particle creation
    df_new2 = df_new.loc[df_new['index']==':']

    #Store the lines with new columns to get the info about this new particle
    df_new3 = pd.DataFrame(columns=['DX_s', 'DY_s', 'DZ_s','Kin(MeV)_s', 'name_s'])
    df_new3[['DX_s', 'DY_s', 'DZ_s','Kin(MeV)_s', 'name_s']]\
    =df_new2[['DX', 'DY', 'DZ', 'KinE(MeV)', 'dE(MeV)']]

    #Store the information on the index that created the particle (-2 because we also removed a line before)
    df_new3.set_index(df_new3.index - 2, inplace=True)

    #Now merge to match indices of new paticle occurrences
    df_new = df_new.merge(df_new3, how='outer', left_index=True, right_index=True)

    #Now we can remove the indicator lines and focus on the inside material (phantom)
    df_new = df_new[~df_new['index'].str.startswith(':')]
    df_new = df_new[(df_new['NextVolume'] == 'phantom')]
    df_new.drop('NextVolume', axis=1, inplace=True)
    
    #Finally get new indexing for clarity
    df_new=df_new.reset_index(drop=True)
    
    return df_new
