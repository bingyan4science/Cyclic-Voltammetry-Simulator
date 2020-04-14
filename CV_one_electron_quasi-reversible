import torch
import math
import matplotlib.pyplot as plt
torch.set_printoptions(profile="full")

#physical Constants
F = 96485.332 #C/mol - Faraday constant
T = 298 #K - Temperature
R = 8.314 #J/mol-K - Gas const.
Beta = 0.25 #grid expansion factor 0 < BT < 1

#reaction Constants
DR = 1e-9 #m^2/s - Diffusion coeff of reactant R
DP = 1e-9 #m^2/s - Diffusion coeff of reactant P
kf = 1e-4 #m^2/s - Surface rate constant (max value ~ 1 m^2/s)
kb = 1e-4 #m^2/s - Surface rate constant (max value ~ 1 m^2/s)
E0 = 2.5 #V - True reversible potential
Alpha = 0.5 #Transfer coefficient
n = 1 # # of electrons transferred
VT = Alpha*n*F/(R*T) #V -mod. thermal voltage for n = 1
Ncycle = 3 #number of CV cycles


#experiment constants
nu = 0.1 #V/s- Sweep rate
Ei = 2.0 #V - initial voltage
Emax = 4.0 #V - Max voltage (end for forwards sweep
Emin = 1.0 #V - Min voltage (end for backwards sweep)
dE = 0.001 #V - potential step
CR0 = 0.01 #M - bulk conc. of reactant R - MUST have a decimal!
CP0 = 0 #M - bulk conc. of product P

tf = (Emax-Emin) / nu #s, timescale of experiment operation
dt = dE / nu #s, timestep

kValues = torch.tensor([kf * math.exp(VT * (Emin-E0)), kb * math.exp(VT * (Emax-E0))])

Ds = R * T * max(kValues) / (F * nu) #Target dimensionless diffusion coeff

rat = (dE/0.001)*Ds/max([DR,DP]) #ratio of dt/(dx)^2
dx = math.sqrt(dt/rat) #Necesary 'base' x-spacing for accuracy

xmax = 6.0*math.sqrt(max([DR,DP])*tf) #maximum diffusion distance
N = 1 + math.ceil(math.log(1 + (xmax*(math.exp(Beta) - 1)/dx) )/Beta)  #+1 for ghost pt


#create the array of space
x = torch.empty(2+2*N, 1, dtype = torch.float)
x[0] = 0
for k in range(1, 1+N):
  x[k] = dx * (math.exp(Beta*(k-0.5))-1) / (math.exp(Beta)-1)
for k in range(1+N, 2*N+2):
  x[k] = x[k-N-1]


#calculate the normalized rate constant
dx1 = dx * (math.exp(Beta/2)-1)/(math.exp(Beta)-1)

kef0 = kf * math.exp((Ei-E0)*VT) * dx1 / DR
keb0 = kb * math.exp(-(Ei-E0)*VT) * dx1 / DP


#create tensors for the diffusion coefficients
D1 = torch.empty(N, 1, dtype = torch.float)
D2 = torch.empty(N, 1, dtype = torch.float)
D3 = torch.empty(N, 1, dtype = torch.float)

#set values for the diffusion coefficients
D1[0] = Ds * (math.exp(Beta)-1) / (math.exp(Beta/2) - 1)
D2[0] = Ds * math.exp(2*Beta*(3/4 - 1))
D3[0] = D1[0] + D2[0] + 1

for k in range(1, N):
  D1[k] = Ds * math.exp(2 * Beta * (5/4 - k - 1))
  D2[k] = Ds * math.exp(2 * Beta * (3/4 - k - 1))
  D3[k] = D1[k] + D2[k] + 1


#create a tensor size (2N+2)*(2N+2) with uninitialized memory
Dopt = torch.empty(2*N+2, 2*N+2, dtype=torch.float).fill_(0)
Drat = DR/DP

#set values for the operator Dopt

#set values for row 0
Dopt[0, 0] = 1 + kef0
Dopt[0, 1] = -1
Dopt[0, N+1] = - keb0
for j in range(2, N+1):
  Dopt[0, j] = 0
for j in range(N+2, 2*N+2):
  Dopt[0, j] = 0

#set values for row N+1
Dopt[N+1, 0] = -Drat
Dopt[N+1, 1] = Drat
Dopt[N+1, N+1] = -1
Dopt[N+1, N+2] = 1
for j in range(2, N+1):
  Dopt[N+1, j] = 0
for j in range(N+3, 2*N+2):
  Dopt[N+1, j] = 0

#set values for row N
Dopt[N, N] = 1
for j in range(0, N):
  Dopt[N, j] = 0
for j in range(N+1, 2*N+2):
  Dopt[N, j] = 0

#set values for row 2N+1
Dopt[2*N+1, 2*N+1] = 1
for j in range(N+1, 2*N+1):
  Dopt[2*N+1, j] = 0


#set values for other rows for Dopt
for i in range(1, N):
  Dopt[i, i] = D3[i-1]
  Dopt[i, i-1] = -D1[i-1]
  Dopt[i, i+1] = -D2[i-1]
  for j in range(0, i-1):
    Dopt[i, j] = 0
  for j in range(i+2, 2*N+2):
    Dopt[i, j] = 0

for i in range(N+2, 2*N+1):
  Dopt[i, i] = D3[i-N-2]
  Dopt[i, i-1] = -D1[i-N-2]
  Dopt[i, i+1] = -D2[i-N-2]
  for j in range(0, i-1):
    Dopt[i, j] = 0
  for j in range(i+2, 2*N+2):
    Dopt[i, j] = 0


#create tensors for the new concentration
Cnew = torch.empty(2*N+2, 1, dtype = torch.float)
for i in range(1, N+1):
  Cnew[i] = CR0
for i in range(N+1, 2*N+2):
  Cnew[i] = CP0

#create tensors for the old concentration
Cold = torch.empty(2*N+2, 1, dtype = torch.float)
for i in range(1, N+1):
  Cold[i] = CR0
for i in range(N+1, 2*N+2):
  Cold[i] = CP0

Cold[0] = 0
Cold[N+1] = 0

#Initialize potential vector

Evt = torch.cat([torch.arange((Ei+dE),(Emax+dE),dE), torch.arange(Emax,(Emin-dE),-dE), torch.arange(Emin, (Ei+2*dE), dE)])

Evt = Evt.reshape(len(Evt),1)
print(Evt.shape)
Istor = 0.0*Evt
count = 0

for E in Evt:
  kef = kf * math.exp((E-E0)*VT) * dx1 / DR
  keb = kb * math.exp(-(E-E0)*VT) * dx1 / DP
  
  #set values for row 0
  Dopt[0, 0] = 1 + kef
  Dopt[0, 1] = -1
  Dopt[0, N+1] = - keb
  for j in range(2, N+1):
    Dopt[0, j] = 0
  for j in range(N+2, 2*N+2):
    Dopt[0, j] = 0

  Cold[0] = 0
  Cold[N] = CR0
  Cold[N+1] = 0
  Cold[2*N+1] = CP0

  Cnew, LU = torch.solve(Cold, Dopt)

  Istor[count,:] = DR*(Cnew[1] - Cnew[0])/dx1
  print(E,kf,kb,Cnew[0],Istor[count,:])
  
  #plot the current profile every 100 mV increment of potential
  if count % 100 == 0:
         plt.figure(count)
         plt.plot(x[0:(N+1)],Cnew[0:(N+1)],'-b')
         plt.plot(x[(N+1)::],Cnew[(N+1)::],'-r')
         plt.title(['Voltage = ',E,' V'])
         plt.ylabel('Conc. (mM)')
         plt.xlabel('Distance from electrode (m)')
         plt.legend((1,2),('Reactant','Product'),loc=5)
         plt.show()
         print(count,E)
  
  count = count + 1
  Cold = Cnew

plt.figure(2)    
plt.plot(Evt,Istor)
plt.xlabel('Voltage (V)')
plt.ylabel('Current (A)')
plt.show()

#get the peak potential and peak current
dex = (Istor.tolist()).index(max(Istor.tolist()))
print(dex)
print(Evt[dex])
print(Istor[dex])
