import numpy as np, queue, pylab as plt, random, sys, enum, math

class Type(enum.Enum):
    photon = 0; electron = 1; positron = 2; proton = 4; nuetron = 5

class Arena :
    def __init__(self, wx:float, wy:float, wz:float, nx:int, ny:int, nz:int) :
        self.M = np.zeros((nx, ny, nz))
        self.voxel = np.array((wx/nx, wy/ny, wz/nz))
    def Deposit(self, pos, ene:float) :
        idx = (pos/self.voxel).astype(int)
        try :
            self.M[idx[0],idx[1],idx[2]] += ene
        except :
            return False # if a deposit fails, it means we're out of the arena
        return True

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
        s = cos_angle * random.random() # approximate version
        self.dir[1] += s
        self.dir[2] += (cos_angle - s)
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

def GetEvent(P:Particle) :
    """this is the function you are responsible for creating, given a particle it returns:
    distance:	distance the particle travels before having this event
    delta_e:	amount of energy that the particle loses during this event
    cos_theta:	cosine of the angle that the particle rotates by
    Q:		a particle generated during this event, or None
    """
    if P.type == Type.photon :
        return CoreEvent(5.0, 0.5, 0.05, 0.99)
    elif P.type == Type.electron :
        return CoreEvent(1.0, 0.2, 0.1, 0.75)

WX,WY,WZ = 300,200,200 # 300x200x200 mm
NX,NY,NZ = 150,100,100 # for 2x2x2 mm voxels
A = Arena(WX, WY, WZ, NX, NY, NZ)
NMC = 100000 # number of particles we want to simulate
EMAX = 20.0 # maximum energy of particles

ToSimulate = queue.Queue() # the queue of particles
for i in range(NMC) :
    # create NMC particles at x=0, y=WY/2, z=WZ/2, going in the X direction (1,0,0) and with a gamma distributed energy
    s = random.gauss(0.0, 0.1)
    ypos = (1+s)*WY/2
    ydir = 0.2*s
    s = random.gauss(0.0, 0.1)
    zpos = (1+s)*WZ/2
    zdir = 0.2*s
    xdir = math.sqrt(1 - ydir**2 - zdir**2)
    e = EMAX*random.gammavariate(2.0, 0.667)
    P = Particle(0.0, ypos, zpos, xdir, ydir, zdir, e, Type.photon, True)
    ToSimulate.put(P)

DONE = 0 # count number of primary particles
NALL = 0 # count total number of particles
while not ToSimulate.empty() > 0 :
    P = ToSimulate.get()
    if P.is_primary :
        DONE += 1
    NALL += 1
    while P.ene > 0.0 :
        distance, delta_e, cos_theta, generated_particle = GetEvent(P)
        P.Move(distance)
        P.Lose(delta_e, A)
        P.Rotate(cos_theta)
        if generated_particle is not None :
            ToSimulate.put(generated_particle)
    if DONE % 1000 :
        sys.stdout.write(f"Finished {DONE:8} out of {NMC:8} {(100.0*DONE)/NMC:.2f} %\r"); sys.stdout.flush()
sys.stdout.write(f"\nFinished with {NMC:8} primaries and a total of {NALL:8} particles\n"); sys.stdout.flush()

fig,axes = plt.subplots(nrows=1, ncols=3, figsize=(18,6))
CNTR = axes[0].contourf( np.log10(A.M.sum(axis=1)+0.001), cmap="inferno" )
axes[0].axvline(NY/2, ls=":", color="white")
axes[0].axhline(5, ls=":", color="green")
plt.colorbar(CNTR, aspect=60)
axes[0].set_title("log(Dose) in X/Z plane")
axes[1].plot( A.M[:,int(NY/2),int(NZ/2)] )
axes[1].set_title("dose along white line")
axes[2].plot( A.M[5,:,int(NZ/2)] )
axes[2].set_title("dose across green line")
plt.show()
