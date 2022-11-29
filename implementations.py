import pandas as pd
import numpy as np

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


def input_output(df) :
    """ This function create two dataframe X of input and the Y of output, it would be neccessary to train 
    our data and creating a model for future prediciton 
    
    Return : 
        X : The input data composed of the following feature size (nb_inputs, 5)
            - cos_teta
            - Kin_E  <-- Arreter le programme quand Kin_E = 0 ! --> gérer les simulations dans ce sens
            - dE  <-- Se calcule juste comme la différence entre les 2 énergies qu'on a !
            - Step Lenght
            - Proc Name
            
        Y : The output data, of size (nb_inputs, 4)
            - distance
            - delta_e
            - cos_teta
            - Q
        
    """
    #conversion of the dataframe into a numpy array :
    np_data = df_clean.to_numpy()
    X = []
    Y = [0, 0, 0, 0]
    
    for idx, data_x in enumerate(data) :
    # Until we are at the end of the simulation (We don't iterate on the last element as we access in this loof the i+1 element) 
        if (idx != data.shape[0]-1) :
            if (data[idx+1] != 0) :
                line_input = []
                line_output = []
                #Creation of the line idx of the matrix X
                
                #1st element:cos_teta = (dx).(dx') scalar product of the postions vectors (dx, dy, dz) at the postions i and i+1
                # data_x[1] = Dx, data_x[2] = Dy, data_x[3] = Dz
                cos_teta_before = data[idx-1, 1]*data_x[1] + data[idx-1, 2]*data_x[2] + data[idx-1, 3]*data_x[3]
                cos_teta_after = data[idx+1, 1]*data_x[1] + data[idx+1, 2]*data_x[2] + data[idx+1, 3]*data_x[3]
                line_input.append(cos_teta_before)
                line_output.append(cos_teta_after)
                
                #2,3th elements : energy steps
                KinE_before = data_x[4]
                KinE_after = data[idx + 1, 4]
                line_input.append(KinE_before)
                line_output.append(KinE_after)
                
                dE_before = data_x[5]
                dE_after = data[idx + 1, 5]
                line_input.append(dE_before)
                line_output.append(dE_after)
                
                #4th element : step Length
                step_before = data_x[6]
                step_after = data[idx + 1, 6]
                line_input.append(step_before)
                line_output.append(step_after)
                
                #5th element : type of interactions (We have to use this parameter to create predict the type of interaction!
                # no change in the type of interaction : 0
                # change from msc to eIoni : 1 and the inverse -1
                # change from msc to eBrem : 2 and the inverse -2
                # change from eIoni to eBrem : 3 and the inverse -3
                
                name_nb = 0
                name_before = data_x[7]
                name_after = data[idx + 1, 7]
                
                if (name_before == "msc") : name_nb = 1
                if (name_before == "eIoni") : name_nb = 2
                if (name_before == "eBrem") : name_nb = 3
                    
                line_input.append(name_nb)
                name_nb = 0
                
                if (name_after == "msc") : name_nb = 1
                if (name_after == "eIoni") : name_nb = 2
                if (name_after == "eBrem") : name_nb = 3
                    
                line_output.append(name_nb)
                
                
                """
                name_before = data_x[7]
                name_after = data[idx + 1, 7]
                if (name_before == "msc") :
                    if (name_after == "eIoni") : 
                        name_nb = 1
                    else if (name_after == "eBrem") :
                        name_nb = 2
                        
                else if (name_after == "msc") :
                    if (name_before == "eIoni") : 
                        name_nb = -1
                    else if (name_before == "eBrem") :
                        name_nb = -2
                
                else if (name_before == "eIoni") :
                    if (name_after == "eBrem" ) :
                        name_nb = 3
                else if (name_after == "eIoni") :
                    if (name_before == "eBrem" ) :
                        name_nb = -3
                
                if (name_nb==0) : 
                    raise Exception("The type of interaction was not read correctly!!")
                """
                
               
                X.append(line_input)
                Y.append(line_output)

    
    
    return X, Y

"""DEUXIÈME VERSION !!!! """
def creation_array(df) :
    """ This function create two dataframe X of input and the Y of output, it would be neccessary to train 
    our data and creating a model for future prediciton 
    
    Return : 
        X : The input data composed of the following feature size (nb_inputs, 5)
            - cos_teta
            - Kin_E  <-- Arreter le programme quand Kin_E = 0 ! --> gérer les simulations dans ce sens
            - dE  <-- Se calcule juste comme la différence entre les 2 énergies qu'on a !
            - Step Lenght
            - Proc Name
            
        Y : The output data, of size (nb_inputs, 4)
            - distance [0]
            - delta_e [1]
            - cos_teta [2]
            - Q [3]
        
    """
    #conversion of the dataframe into a numpy array :
    data_np = df.to_numpy()
    data = data_np[:,0:7].astype(float)
    ProcName = data_np[:, 7]
    X = []
    Y = []
    track_len = 0
    
    for idx, data_x in enumerate(data) :
    # Until we are at the end of the simulation (We don't iterate on the last element as we access in this loof the i+1 element) 
        if (idx != data.shape[0]-1) :
            if (data[idx+1, 0] != 0) :
                line_input = []
                
                line_output = [0,0,0,0]
                
                #Creation of the line idx of the matrix X and Y
                
                #1st element:cos_teta = (dx).(dx') scalar product of the postions vectors (dx, dy, dz) at the postions i and i+1
                # data_x[1] = Dx, data_x[2] = Dy, data_x[3] = Dz
                cos_teta_before = data[idx-1, 1]*data_x[1] + data[idx-1, 2]*data_x[2] + data[idx-1, 3]*data_x[3]
                cos_teta_after = data[idx+1, 1]*data_x[1] + data[idx+1, 2]*data_x[2] + data[idx+1, 3]*data_x[3]
                line_input.append(cos_teta_before)
                line_output[2] = (cos_teta_after)
                
                #2,3th elements : energy steps
                KinE_before = data_x[4]
                line_input.append(KinE_before)
                
                dE_before = data_x[5]
                line_input.append(dE_before)
                
                line_output[1] = KinE_before - dE_before
                
                #4th element : step Length
                step = data_x[6]
                track_len += step

                line_input.append(step)
                line_output[0] = (track_len)
                
                #5th element : type of interactions (We have to use this parameter to create predict the type of interaction!
                # no change in the type of interaction : 0
                # change from msc to eIoni : 1 and the inverse -1
                # change from msc to eBrem : 2 and the inverse -2
                # change from eIoni to eBrem : 3 and the inverse -3
                
                name_nb = 0
                name_before = ProcName[idx]
                name_after = ProcName[idx + 1]
                
                if (name_before == "msc") : name_nb = 1
                if (name_before == "eIoni") : name_nb = 2
                if (name_before == "eBrem") : name_nb = 3
                    
                line_input.append(name_nb)
                
                Q = 0
                if (name_before != name_after) :
                    Q = 1
                    
                line_output[3] = Q
                
               
                X.append(line_input)
                Y.append(line_output)
                
            else : 
                track_len = 0 
                # We put back the track_len to 0 as we start from a new electron

    
    
    return np.asarray(X), np.asarray(Y)


def compute_loss(y, tx, w):
    """Calculate the loss using either MSE or MAE.

    Args:
        y: shape=(N, )
        tx: shape=(N,2)
        w: shape=(2,). The vector of model parameters.

    Returns:
        the value of the loss (a scalar), corresponding to the input parameters w.
    """
    e = y - tx.dot(w)
    return 1/2*np.mean(e**2)


def least_squares(y, tx):
    """Calculate the least square solution

    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.

    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.
        mse: scalar.
    """

    a = np.dot(tx.T, tx)
    b = np.dot(tx.T, y)
    w = np.linalg.solve(a, b)
    mse = compute_loss(y, tx, w)

    return w, mse


