import pandas as pd
import numpy as np
import numpy as np, queue, pylab as plt, random, sys, enum, math

def build_df(data_path):
    data = pd.read_csv(data_path,sep="\s+")
    dframe=pd.DataFrame(data)
    #Step# is the the new row name
    dframe.index=dframe.iloc[:,0]
    #remove column of Step# and Nan column
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
    =['1', '0', '0', init_E, '0', '0', '0', 'phantom', 'msc']
    return dframe


def data_file(i):
    return "../Data/E_" + str(i) + ".data"







def clean_df(df):
    #Select interesting values
    df_new = df[["X(mm)","Y(mm)","Z(mm)",'DX', 'DY', 'DZ', 'KinE(MeV)', 'dE(MeV)',\
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
    df_new3 = pd.DataFrame(columns=["X(mm)_s","Y(mm)_s","Z(mm)_s",'DX_s', 'DY_s', 'DZ_s','Kin(MeV)_s', 'name_s'])
    df_new3[["X(mm)_s","Y(mm)_s","Z(mm)_s",'DX_s', 'DY_s', 'DZ_s','Kin(MeV)_s', 'name_s']]\
    =df_new2[["X(mm)","Y(mm)","Z(mm)",'DX', 'DY', 'DZ', 'KinE(MeV)', 'dE(MeV)']]

    #Store the information on the index that created the particle (-2 because we also removed a line before)
    df_new3.set_index(df_new3.index - 2, inplace=True)

    #Now merge to match indices of new paticle occurrences
    df_new = df_new.merge(df_new3, how='outer', left_index=True, right_index=True)

    #Now we can remove the indicator lines and focus on the inside material (phantom)
    df_new = df_new[~df_new['index'].str.startswith(':')]
    df_new = df_new[(df_new['NextVolume'] == 'phantom')]
    df_new.drop(['ProcName','NextVolume'], axis=1, inplace=True)
    
    #Finally get new indexing for clarity
    df_new=df_new.reset_index(drop=True)
    
    #Remove NAN values
    df_new[["X(mm)_s","Y(mm)_s","Z(mm)_s",'DX_s', 'DY_s', 'DZ_s','Kin(MeV)_s', 'name_s']]\
    = df_new[["X(mm)_s","Y(mm)_s","Z(mm)_s",'DX_s', 'DY_s', 'DZ_s','Kin(MeV)_s', 'name_s']].fillna(0)
    
    #Get numeric types
    df_new[['index', "X(mm)","Y(mm)","Z(mm)",'DX', 'DY', 'DZ', 'KinE(MeV)', 'dE(MeV)','StepLeng',\
          "X(mm)_s","Y(mm)_s","Z(mm)_s",'DX_s', 'DY_s', 'DZ_s','Kin(MeV)_s']] = \
    df_new[['index', "X(mm)","Y(mm)","Z(mm)",'DX', 'DY', 'DZ', 'KinE(MeV)', 'dE(MeV)','StepLeng',\
          "X(mm)_s","Y(mm)_s","Z(mm)_s",'DX_s', 'DY_s', 'DZ_s','Kin(MeV)_s']]\
    .apply(pd.to_numeric)
    
    return df_new



def get_cos(df):
    df_new = df.copy(deep=True)
    data = df_new.to_numpy()
    cos_theta = []
    list_ind = []
    for idx, data_x in enumerate(data[:-1]) :
        if (data[idx+1, 0] != 0) :
            list_ind.append(idx)
            cos_theta.append(np.clip(data[idx+1, 4]*data_x[4] + data[idx+1, 5]*data_x[5] + data[idx+1, 6]*data_x[6],-1,1))
    df_new = df_new.iloc[list_ind]
    df_new['cos_theta'] = cos_theta
    return df_new
    
    
    
    
def type_to_num(ptcl):
    #Each particle is associated to an integer number 
    if(ptcl == 'gamma'):
        return 2
    elif(ptcl == "e-"):
        return 1
    elif(ptcl == 0):
        return 0
    else:
        raise Exception('Not a good type of particle')

def creation_array(df) :
    """ This function create two dataframe X of input and the Y of output, it would be neccessary to train 
    our data and creating a model for future prediciton 
    
    Return : 
        X : The input data composed of the following feature size (nb_inputs, 7)
            - pos (x,y,z) ---> [0,1,2]
            - dir (dx,dy,dz) ---> [3,4,5]
            - KinE(MeV) ---> [6]
            
        Y : The output data, of size (nb_inputs, 11)
            - distance ( StepLength ) [0]
            - delta_e ( dE(MeV) )[1]
            - cos_teta [2]
            - Q  <--- (name_s [3], x_s [4], y_s [5], z_s [6], dx_s [7], dy_s [8], dz_s [9], KinE(MeV)_s [10])
        
    """
    df = df.copy(deep = True)
    
    DONE = 0
    size_df = df.shape[0] - 1
    
    #conversion of the dataframe into a numpy array :
    df['name_s'] = df['name_s'].apply(type_to_num)
    data = df.to_numpy()
    #data[:,17] = data[:,17].apply(type_to_num)
    X = []
    Y = []
    
    
    for idx, data_x in enumerate(data[:-1]) :
    # Until we are at the end of the simulation (We don't iterate on the last element as we access in this loof the i+1 element) 
        DONE += 1
        if (data[idx+1, 0] != 0) :
            
            line_input = []

            #Creation of the line idx of the matrix X and Y
            #The 3 first elements are the positions : 
            line_input += [data_x[1], data_x[2], data_x[3]]
            
            # The 3,4,5th elements of X are the directions : 
            
            line_input += [data_x[4], data_x[5], data_x[6]]
            
            
            #The 6th element of X is KinE
            line_input.append(data_x[7])
            
            #Now we create the output array Y :
            line_output = []
            
            #1st element is the distance corresponding to the StepLenght of data[9]
            line_output.append(data_x[9])
            
            #2nd element is the change in energy corresponding to data[8]
            line_output.append(data_x[8])
            
            #3rd element is the new angle :
            #cos_teta = (dx).(dx') scalar product of the postions vectors (dx, dy, dz) at the postions i and i+1
            # data_x[4] = Dx, data_x[5] = Dy, data_x[6] = Dz
            
            cos_teta = np.clip(data[idx+1, 4]*data_x[4] + data[idx+1, 5]*data_x[5] + data[idx+1, 6]*data_x[6],-1,1)
            
            line_output.append(cos_teta)

            #The last element of the output Y will correspond to the emitted particule : 
            if (data_x[17] == 0) : 
                #Case where no particule is emitted, we set all the values to 0 
                
                line_output.append(data_x[17])
                line_output += [0,0,0,0,0,0,0]
                
            elif (data_x[17] == 1 or data_x[17] == 2) : 
                line_output.append(data_x[17])
                line_output += [data_x[10], data_x[11], data_x[12], data_x[13], data_x[14], data_x[15], data_x[16]]
                
        
        X.append(line_input)
        Y.append(line_output)
        if (DONE%1000 == 0) :
            sys.stdout.write(f"Finished {DONE:8} out of {size_df:8} {(100.0*DONE)/size_df:.2f} %\r"); sys.stdout.flush()



        
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


def evaluate_model(model, x_test, y_test):
    from sklearn import metrics

    # Predict Test Data 
    y_pred = model.predict(x_test)

    # Calculate accuracy, precision, recall, f1-score, and kappa score
    acc = metrics.accuracy_score(y_test, y_pred)
    
    # Display confussion matrix
    cm = metrics.confusion_matrix(y_test, y_pred)
    
    """
    prec = metrics.precision_score(y_test, y_pred)
    rec = metrics.recall_score(y_test, y_pred)
    f1 = metrics.f1_score(y_test, y_pred)
    kappa = metrics.cohen_kappa_score(y_test, y_pred)

    # Calculate area under curve (AUC)
    y_pred_proba = model.predict_proba(x_test)[::,1]
    fpr, tpr, _ = metrics.roc_curve(y_test, y_pred_proba)
    auc = metrics.roc_auc_score(y_test, y_pred_proba)

    # Display confussion matrix
    cm = metrics.confusion_matrix(y_test, y_pred)

    return {'acc': acc, 'prec': prec, 'rec': rec, 'f1': f1, 'kappa': kappa, 
            'fpr': fpr, 'tpr': tpr, 'auc': auc, 'cm': cm}
            """
    return {'acc': acc, 'cm': cm}
    
    
    
def map_energy_ranges(data, energy_ranges):
    data_emission = data.copy(deep=True)
    data_emission['E_range'] = 0
    for i, E in enumerate(energy_ranges):
        if(i==0):
            data_emission.loc[(data_emission['KinE(MeV)'] <= E), 'E_range'] = '0 _ ' + '%.1f' % E
        if(i==len(energy_ranges)-1):
            data_emission.loc[(data_emission['KinE(MeV)'] > energy_ranges[i-1]) & (data_emission['KinE(MeV)'] <= E), 'E_range']\
            = '%.1f' % energy_ranges[i-1] + ' _ ' + '%.1f' % E
            data_emission.loc[(data_emission['KinE(MeV)'] > E), 'E_range'] = '%.1f' % E + ' _ 20'
        else:
            data_emission.loc[(data_emission['KinE(MeV)'] > energy_ranges[i-1]) & (data_emission['KinE(MeV)'] <= E), 'E_range']\
            = '%.1f' % energy_ranges[i-1] + ' _ ' + '%.1f' % E
    return data_emission


    
    
    
    
    
    
    
    
    
    
    
    