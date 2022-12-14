import pandas as pd
import numpy as np
import queue, pylab as plt, random, sys, enum, math

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



def get_cos_theta(df):
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


def get_angles(df_old):
    df = df_old.copy(deep=True)
    data = df.to_numpy()
    cos_theta = []
    cos_psi = []
    list_ind = []
    df['cos_phi'] = df['DX']*df['DX_s'] + df['DY']*df['DY_s'] + df['DZ']*df['DZ_s']
    df['cos_phi'].clip(-1,1)
    # DX: data_x[13], DY: data_x[14], DZ: data_x[15]
    for idx, data_x in enumerate(data[:-1]) :
        if (data[idx+1, 0] != 0) :
            list_ind.append(idx)
            cos_theta.append(np.clip(data[idx+1, 4]*data_x[4] + data[idx+1, 5]*data_x[5] + data[idx+1, 6]*data_x[6],-1,1))
            cos_psi.append(np.clip(data[idx+1, 4]*data_x[13] + data[idx+1, 5]*data_x[14] + data[idx+1, 6]*data_x[15],-1,1))
    df = df.iloc[list_ind]
    df['cos_theta'], df['cos_psi'] = cos_theta, cos_psi
    return df
    
    
""" Set of functions to evaluate the accuracy of the different classification model (used in classification.ipynb) :
    - evaluate_model
    - get_percentage
    - compute_proba
"""



def evaluate_model(model, x_test, y_test):
    from sklearn import metrics

    # Predict Test Data 
    y_pred = model.predict(x_test)

    # Calculate accuracy, precision, recall, f1-score, and kappa score
    acc = metrics.accuracy_score(y_test, y_pred)
    
    # Display confussion matrix
    cm = metrics.confusion_matrix(y_test, y_pred)

    
            
    return {'acc': acc, 'cm': cm }

def get_percentage(cm, y_test) : 
    
    p0 = cm[0,0]/(y_test == 0).sum()*100
    p1 = cm[1,1]/(y_test == 1).sum()*100
    p2 = cm[2,2]/(y_test == 2).sum()*100
    
    return {'p0' : p0, 'p1' : p1, 'p2' : p2} 
    
    

""" Creatio of the table of probabilities depending on the energy range :
    - proba_table
    - compute_proba
"""    


    
def proba_table(data, diff = 0.1) : 
    
    
    table = pd.DataFrame(columns=['Energy_min', 'Energy_max' ,'proba_0','proba_1','proba_2'])
   
    E = 0
    E_next = E + diff
    range_energy = data[(data['KinE(MeV)']<=E_next) & (data['KinE(MeV)']>E)]
    prob = compute_proba(range_energy)
    
    table.loc[len(table.index)] = [E, E_next, prob[0], prob[1], prob[2]]
    
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
            
        table.loc[len(table.index)] = [E, E_next, prob[0], prob[1], prob[2]]
       
        sys.stdout.write(f"Finished {E_next:.2f} out of {20.0:2} {(100.0*E_next)/20:.2f} %\r"); sys.stdout.flush()


         
    return table  
    
def compute_proba(df) : 
    
    """ Compute the probability of the classes in the energy range of the data_frame """
    if (df.empty) : 
        return [0,0,0]
    
    n_range = df['name_s'].value_counts().tolist()
    if (len(n_range) != 3) : 
        n_range = [0,0,0]
        n_range[0] = (df['name_s']==0).sum()
        n_range[1] = (df['name_s']==1).sum()
        n_range[2] = (df['name_s']==2).sum()
    
    
    prob = n_range/np.sum(n_range)

    return prob    


    
""" 
Set of functions that will be used for the GAN :
    - map_energy_ranges
    - get_model
"""



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

            


    
def get_model(KinE=1.0, name_s=0):
    PATH = 'saved_model/model'

    if((KinE <= 7.8) & (name_s==0)):
        PATH=PATH+str(1)
        
    elif((KinE > 7.8) & (name_s==0)):
        PATH=PATH+str(2)
        
    elif((KinE <= 1.0) & (name_s==1)):
        PATH=PATH+str(3)
        
    elif((KinE > 1.0) & (name_s==1)):
        PATH=PATH+str(4)
        
    elif((KinE <= 1.0) & (name_s==2)):
        PATH=PATH+str(5)
        
    elif((KinE > 1.0) & (name_s==2)):
        PATH=PATH+str(6)
    
    gmodel = GeneratorMLP(dim_hidden=128, dim_out=dim_out, noise_dim=noise_dim)
    gmodel.load_state_dict(torch.load(PATH))
    
    return gmodel
         
        
        
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
        
        
        
## TEST

class Type(enum.Enum):
    photon = 0; electron = 1; positron = 2; proton = 4; nuetron = 5

class Arena :
    def __init__(self, wx:float, wy:float, wz:float, nx:int, ny:int, nz:int) :
        self.M = np.zeros((nx, ny, nz))
        self.voxel = np.array((wx/nx, wy/ny, wz/nz))
    def Deposit(self, pos, ene:float) :
        #wx = self.M.shape[0]*self.voxel[0]
        #wy = self.M.shape[1]*self.voxel[1]
        #wz = self.M.shape[2]*self.voxel[2]
        
        if((pos[0] > 0) & (pos[1] > 0) & (pos[2] > 0)):
            idx = (pos/self.voxel).astype(int)
            try :
                self.M[idx[0],idx[1],idx[2]] += ene
            except :
                return False # if a deposit fails, it means we're out of the arena
            return True
        else:
            return False



class Particle :
    def __init__(self, x:float, y:float, z:float, dx:float, dy:float, dz:float, e:float, t:Type, is_primary:bool=False) :
        self.pos = np.array((x, y, z), dtype=float)
        self.dir = np.array((dx, dy, dz), dtype=float)
        self.ene = e
        self.type = t
        self.is_primary = is_primary

    def Lose(self, energy:float, phantom:Arena) :
        energy = min(energy, self.ene) # lose this much energy and deposit it in the arena
        self.ene -= energy
        if not phantom.Deposit(self.pos, energy) :
            self.ene = 0.0 # if a deposit fails, it means we're out of the arena, so kill the particle

    def Move(self, distance:float) :
        self.pos += distance*self.dir

    def Rotate(self, cos_angle:float) :
        s1, s2 = np.random.randint(0,3,size=2)
        sin_theta = np.sqrt(1-np.clip(cos_theta,-1,1)**2)
        x_bis = self.dir[s1]
        y_bis = self.dir[s2]
        self.dir[s1] = x_bis*cos_theta - y_bis*sin_theta
        self.dir[s2] = x_bis*sin_theta + y_bis*cos_theta
        self.dir /= np.linalg.norm(self.dir)

def CoreEvent(max_dist:float, max_dele:float, max_cos:float, prob:float) :
    s = random.random() # simple event generator for testing
    distance = max_dist*s
    delta_e = max_dele*s
    cos_theta = max_cos*(random.random() - 0.5)
    if s > prob :
        delta_e *= 0.5
        Q = Particle(P.pos[0], P.pos[1], P.pos[2], P.dir[0], P.dir[1], P.dir[2], delta_e, Type.electron)
        Q.Rotate(0.5)
        return distance, delta_e, cos_theta, Q
    else :        
        return distance, delta_e, cos_theta, None



def GetEvent(P:Particle, table_of_p) :
    """CoreEvent by default if we need to simulate a photon, GetEvent_Model for electron event generation from the GAN models:
    distance:	distance the particle travels before having this event
    delta_e:	amount of energy that the particle loses during this event
    cos_theta:	cosine of the angle that the particle rotates by
    Q:		a particle generated during this event, or None
    """
    if P.type == Type.photon :
        return CoreEvent(5.0, 0.5, 0.05, 0.99)
    elif P.type == Type.electron :
        distance, delta_e, cos_theta, generated_particle = GetEvent_Model(P, table_of_p)
        return distance, delta_e, cos_theta, generated_particle


        
        
def GetEvent_Model(P:Particle, table_of_p):
    # Compute probability of particles
    table = table_of_p.copy(deep=True)
    KinE = P.ene
    
    #Get probability of emission type (or None) 
    prob_all = table.loc[(table['Energy_min'] <  KinE) & (table['Energy_max'] >= KinE)]
    prob0, prob1 = prob_all[['proba_0', 'proba_1']].values[0,:]
    
    #From uniform random variable get the emission type
    s = random.random() 
    if s <= prob0:
        name_s = 0
    elif ((s > prob0) & (s < prob0 + prob1)):
        name_s = 1
    else:
        name_s = 2
    
    # Now we have the type of the particle
    # Let's find the model necessary for the particle
    model = get_model(KinE=KinE, name_s=name_s)
    
    # And generate a random event from this event
    event = Get_GAN_event(g_model=model)[0]
    
    # We store the data generated
    if name_s!=0:
        #{'cos_theta': 0, 'dE': 1, 'cos_phi': 2, 'cos_psi': 3, 'E_s': 4}
        cos_theta=event[0]
        dE=event[1]
        cos_phi=event[2]
        EQ=np.minimum(event[4],KinE)
    else:
        cos_theta=event[0]
        dE=event[1]
     
    
    # We select the return needed
    if(dE < 0):
        dE = abs(dE)
        
    # Get the distance that we found to be proportional to dE variable in data analysis
    distance=6.170464696*dE
    

    # Return depending if there is emission or not
    if name_s!=0:
        if (name_s == 1):
            name_s = Type.electron
        else:
            name_s = Type.photon
        Q = Particle(P.pos[0], P.pos[1], P.pos[2], P.dir[0], P.dir[1], P.dir[2], EQ, name_s)
        Q.Rotate(cos_phi)
        return distance, dE, cos_theta, Q
    else:
        return distance, dE, cos_theta, None        
        
        
        
        
        
    
    
