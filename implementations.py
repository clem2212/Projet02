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



def evaluate_model(model, x_test, y_test):
    from sklearn import metrics

    # Predict Test Data 
    y_pred = model.predict(x_test)

    # Calculate accuracy, precision, recall, f1-score, and kappa score
    acc = metrics.accuracy_score(y_test, y_pred)
    
    # Display confussion matrix
    cm = metrics.confusion_matrix(y_test, y_pred)
    
    #Individual accuracy of the type of emission 
    acc_0 = get_accuracy(cm, 0)
    acc_1 = get_accuracy(cm, 1)
    acc_2 = get_accuracy(cm, 2)
    
            
    return {'acc': acc, 'cm': cm, 'acc0' : acc_0, 'acc1' : acc_1, 'acc2' : acc_2}

def get_percentage(cm, y_test) : 
    
    p0 = cm[0,0]/(y_test == 0).sum()*100
    p1 = cm[1,1]/(y_test == 1).sum()*100
    p2 = cm[2,2]/(y_test == 2).sum()*100
    
    return {'p0' : p0, 'p1' : p1, 'p2' : p2} 
    

def get_accuracy(cm, n) : 
    #cm is the confussion matrix and n = 0,1 or 2:
    
    if n == 0 : 
        #true positive
        TP = cm[0,0]
        TN = cm[1,1] + cm[1,2] + cm[2,1] + cm[2,2]
        FN = cm[0,1] + cm[0,2]
        FP = cm[1,0] + cm[2,0]
        
    if n == 1 : 
        TP = cm[1,1]
        TN = cm[0,0] + cm[0,2] + cm[2,0] + cm[2,2]
        FN = cm[1,0] + cm[1,2]
        FP = cm[0,1] + cm[2,1]
        
    if n == 2 : 
        TP = cm[2,2]
        TN = cm[0,0] + cm[0,1] + cm[1,0] + cm[1,1]
        FN = cm[2,0] + cm[2,1]
        FP = cm[0,2] + cm[1,2]
        
    return (TP + TN)/(TP + TN + FP + FN)

    
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

            
    
    
def compute_proba(df) : 
    
    """ Compute the probability of the classes in the energy range of the data_frame
    n is the list with the numbe of 0, 1 and 2 of the initial data_frame"""
    if (df.empty) : 
        return [0,0,0]
    
    n_range = df['name_s'].value_counts().tolist()
    if (len(n_range) != 3) : 
        n_range = [0,0,0]
        n_range[0] = (df['name_s']==0).sum()
        n_range[1] = (df['name_s']=='e-').sum()
        n_range[2] = (df['name_s']=='gamma').sum()
    
    
    prob = n_range/np.sum(n_range)

    return prob
    
    
def proba_table(data, diff = 0.1) : 
    
    
    table = pd.DataFrame(columns=['Energy_min', 'Energy_max' ,'proba_0','proba_1','proba_2'])
   
    E = 0
    E_next = E + diff
    range_energy = data[(data['KinE(MeV)']<=E_next) & (data['KinE(MeV)']>E)]
    prob = compute_proba(range_energy)
    table = table.append({'Energy_min' : E, 'Energy_max' : E_next, 'proba_0' : prob[0], 'proba_1' : prob[1] ,
                                                                      'proba_2' : prob[2]} , ignore_index=True)
    
    """ Treatment of the case where we don't have data to compute the probability so we don't accept 0 as
    proba (we extend the range of consideration )"""
    
    while (E_next<=20) :
        
        E = E_next
        E_next = E_next + diff
        range_energy = data[(data['KinE(MeV)']<=E_next) & (data['KinE(MeV)']>E)]
        prob = compute_proba(range_energy)
        

        while ((np.asarray(prob) == 0).any() and  E_next <= 20.0):
            E_next = E_next + diff
            range_energy = data[(data['KinE(MeV)']<=E_next) & (data['KinE(MeV)']>E)]
            prob = compute_proba(range_energy)          
            
        table = table.append({'Energy_min' : E, 'Energy_max' : E_next, 'proba_0' : prob[0], 'proba_1' : prob[1] ,
                                                                          'proba_2' : prob[2]} , ignore_index=True)
       
        sys.stdout.write(f"Finished {E_next:2} out of {20.0:2} {(100.0*E_next)/20:.2f} %\r"); sys.stdout.flush()


         
    return table  
    
    
    
    