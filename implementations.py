import pandas as pd
import numpy as np


def load_txt_data(data_path):
    """Loads data and returns the G4track information in track_info and the resulst in input_data"""
    """ It returns a serie of DataFrame """
    
    df = pd.read_fwf(data_path, header = None)
    df_array = pd.DataFrame()
    old_idx = 0
    
    for idx in range(len(df)) :
        if (df.iloc[idx,0].startswith("*")) :
            new_df = df.iloc[old_idx:idx-1]
            df_array.append(new_df)
            old_idx = idx + 1
    
    return df_array
    
#df_array.loc[0]
"""
    x = np.genfromtxt(data_path)
    
    #Si le premier élement du tableau commence par un *, alors on commence une nouvelle dataframe
    
    for c in x[0,:] :
        while (c !=  '*') :
            i++
        
        input_data.append(x[j:i].DataFrame())
        j = i
        i = 0
    
        

    return track_info, input_data
    
"""