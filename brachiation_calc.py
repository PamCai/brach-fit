import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

def rheo_calc(c,x,ku,kb,N,M,P,omega,T,plot_calc=False,plot_calc_all=False,print_on=True):
	# Rouse version of brachiation (dimensionalized)
	# Inputs
	# c - concentration of polymer (mols polymer chains / liter)
	# x - drag coefficient (in kg/s, on the order of 10^-3)
	# M - number of stickers along chain (integer)
	# ku - unbinding rate constant (s^-1)
	# kb - binding rate constant (1 / M^-1 s^-1)
	# N - number of total monomers on chain
	# P - number of modes to compute
	# omega - frequency range (dimensional)
	# T - temperature (Celsius)
	# plot_calc - turns on plots generated during the self-consistent calculation
	# plot_calc_all - turns on final plots at the end
	# Outputs
	# omega_min - arbitrary value right now
	# omega - re-dimensionalized frequency range 
	# G - dimensionalized modulus by c/N
	
	# Change units of concentration and kb
	c = c*6.022e23/1.e-3 # convert to  # polymer chains / cubic meter
	kb = kb*1.e-3/6.022e23 # convert to m^3 s^-1 / # polymer chains
	# Define Rouse time (assume b = 1.e-9, a = 1.e-9)
	kBT = 1.38e-23*(273.15+T) # Joules
	M = int(M)
	tR = x*np.float_power(N,2.)*1.e-18/(3.*np.float_power(np.pi,2.)*kBT) # sec 
	print('tR is ',tR)
	pb = kb*c*M/(ku+kb*c*M)
	print('pb is ',pb)
	ku = ku*tR # non-dimensionalize
	c = 2.*c*1.e-27/np.pi # non-dimensionalize concentration (assume b = a = 1 nm)
	zp = np.identity(P)    

	# define omega range
	omega = omega*tR # non-dimensionalize
	w_n = len(omega)
	w_hi = np.max(omega)
	# set max n to start from and normal mode vectors
	#n_max = np.maximum(np.ceil(0.00000001*w_hi/ku),50.)
	n_max = 50.
	p_vec = np.linspace(1.,P,P)
	kp = np.diag(np.float_power(p_vec,2.)) # also multiplied by (3kBT pi^2/(bN)^2)

	# set up initial Cpp matrix at high n
	E1 = np.outer(np.float_power(np.float_power(p_vec,2.)+n_max*ku,-1.),np.ones((1,w_n),dtype=np.complex))
	F1 = np.outer(np.float_power(n_max*ku+np.float_power(p_vec,2.),-2.),(1j*omega))
	Cpp = E1 - F1
	K = c*np.sum(Cpp,axis=0)*np.ones(w_n,dtype=np.complex)
	G0 = np.multiply((1j*omega+n_max*ku),K/c)

	# set up Phi_pp'
	def phip(p,M):
		phip = np.arange(M)
		phip = np.sqrt(2.)*np.cos(np.pi*p*phip/(M-1)) 
		return phip

	Phi_pp = np.zeros((P,P))
	p_mat = np.zeros((P,M))
	for l in range(P):
		p_mat[l,:] = phip((l+1),M)
	Phi_pp = np.diag(np.sum(np.float_power(p_mat,2.),axis=1))
	for l in range(P):
		for m in np.arange(l+1,P,1):
			Phi_pp[l,m] = np.sum(np.multiply(p_mat[l,:],p_mat[m,:]))
			Phi_pp[m,l] = Phi_pp[l,m]
	Phi_pp = Phi_pp/M

	# start at n_max and step back to 0
	count = np.linspace(0,n_max,int(n_max)+1)
	count_back = np.flip(count.astype(int))

	G = np.zeros_like(K)
	for n in count_back:
		for w in range(w_n):
			den = (1j*omega[w]+n*ku)*zp + (1j*omega[w]+n*ku)*pb*Phi_pp*M*K[w] + kp
			num = zp + pb*Phi_pp*M*K[w]
			try:
				C_pp = np.linalg.solve(den,num)
			except np.linalg.LinAlgError as err:
				if 'Singular matrix' in str(err):
					C_pp = np.diag(np.ones(len(omega)))
				else:
					raise
			K[w] = c*np.sum(np.diag(C_pp))
			G[w] = (1j*omega[w]+n*ku)*K[w]/c
		if print_on:            
			print(n)
		if plot_calc:
			plt.plot(omega,np.real(G),'r')
			plt.plot(omega,np.imag(G),'b')
			plt.plot(omega,np.real(G0),'r--')
			plt.plot(omega,np.imag(G0),'b--')
			plt.yscale('log')
			plt.xscale('log')
			plt.show(block=False)
			plt.pause(0.1)
			plt.close()

	# calculate Rouse model limit
	if plot_calc_all:
		GR = np.zeros(w_n)
		for p in range(P):
			GR = GR + 1j*omega/(np.float_power((p+1),2.)+1j*omega)
		plt.plot(omega,np.real(G),'r')
		plt.plot(omega,np.imag(G),'b')
		plt.plot(omega,np.real(G0),'r--')
		plt.plot(omega,np.imag(G0),'b--')
		#plt.plot(omega,np.real(GR),'ro',markersize=1)
		#plt.plot(omega,np.imag(GR),'bo',markersize=1)
		#plt.plot(np.array([ku,ku]),np.array([1.e-6,1.e2]),'k:')
		plt.ylabel(r'$NG/(ck_{B}T)$')
		plt.xlabel(r'$\tau_{R} \omega$')
		plt.yscale('log')
		plt.xscale('log')
		plt.show()

	# Re-dimensionalize variables
	c = c*np.pi/2.e-27
	G = G*c*kBT/N
	omega = omega/tR

	return omega,G