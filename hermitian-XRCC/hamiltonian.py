#    (C) Copyright 2018, 2019 Anthony D. Dutoi and Yuhong Liu
# 
#    This file is part of Qode.
# 
#    Qode is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
# 
#    Qode is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
# 
#    You should have received a copy of the GNU General Public License
#    along with Qode.  If not, see <http://www.gnu.org/licenses/>.
#



def _ket_loop(Hmat, bra_states, I, ket_states_now, J, N_frag, states_per_frag, H):
	n = len(ket_states_now) + 1
	ket_states_next = list(ket_states_now) + [None]
	if n==N_frag:
		monomer_Hamiltonians, dimer_Couplings, trimer_Couplings = H
		ket_states = ket_states_next
		for i in range(states_per_frag):
			ket_states[-1] = i
			transitions = []
			for m,(bra,ket) in enumerate(zip(bra_states,ket_states)):
				if bra!=ket:  transitions += [(m,bra,ket)]
			spf = states_per_frag	# just to make lines below shorter and easier to read
			if len(transitions)==0:
				for M in range(0,N_frag):
					Hmat[I,J] += monomer_Hamiltonians[M][bra_states[M], ket_states[M]]
					for N in range(M+1,N_frag):
						Hmat[I,J] += dimer_Couplings[M][N][bra_states[M]*spf+bra_states[N], ket_states[M]*spf+ket_states[N]]
			if len(transitions)==1:
				(M, Mb, Mk), = transitions
				Hmat[I,J] += monomer_Hamiltonians[M][Mb,Mk]
				for N in range(0,M):
					Hmat[I,J] += dimer_Couplings[N][M][bra_states[N]*spf+Mb, ket_states[N]*spf+Mk]		# Implicitly, we know that ...
				for N in range(M+1,N_frag):
					Hmat[I,J] += dimer_Couplings[M][N][Mb*spf+bra_states[N], Mk*spf+ket_states[N]]		# ... bra_states[N]=ket_states[N]
			if len(transitions)==2:
				(M, Mb, Mk),(N, Nb, Nk) = transitions
				Hmat[I,J] += dimer_Couplings[M][N][Mb*spf+Nb, Mk*spf+Nk]
				for O in range(0,M):
					Hmat[I,J] += trimer_Couplings[O][M][N][(bra_states[O]*spf+Mb)*spf+Nb, (ket_states[O]*spf+Mk)*spf+Nk]	# Implicitly, ...
				for O in range(M+1,N):
					Hmat[I,J] += trimer_Couplings[M][O][N][(Mb*spf+bra_states[O])*spf+Nb, (Mk*spf+ket_states[O])*spf+Nk]	# ... we know that ...
				for O in range(N+1,N_frag):
					Hmat[I,J] += trimer_Couplings[M][N][O][(Mb*spf+Nb)*spf+bra_states[O], (Mk*spf+Nk)*spf+ket_states[O]]	# ... bra_states[O]=ket_states[O]
			if len(transitions)==3:
				(M, Mb, Mk),(N, Nb, Nk),(O, Ob, Ok) = transitions
				Hmat[I,J] += trimer_Couplings[M][N][O][(Mb*spf+Nb)*spf+Ob, (Mk*spf+Nk)*spf+Ok]
			J += 1
	else:
		stride = states_per_frag**(N_frag-n)
		for i in range(states_per_frag):
			ket_states_next[-1] = i
			_ket_loop(Hmat, bra_states, I, ket_states_next, J, N_frag, states_per_frag, H)
			J += stride

def _bra_loop(Hmat, bra_states_now, I, N_frag, states_per_frag, H):
	n = len(bra_states_now) + 1
	bra_states_next = list(bra_states_now) + [None]
	if n==N_frag:
		bra_states = bra_states_next
		for i in range(states_per_frag):
			bra_states[-1] = i
			_ket_loop(Hmat, bra_states, I, [], 0, N_frag, states_per_frag, H)
			I += 1
	else:
		stride = states_per_frag**(N_frag-n)
		for i in range(states_per_frag):
			bra_states_next[-1] = i
			_bra_loop(Hmat, bra_states_next, I, N_frag, states_per_frag, H)
			I += stride

# Built on the assumption that states_per_frag is the same for all fragments
def braket_loops(Hmat, N_frag, states_per_frag, H):
	_bra_loop(Hmat, [], 0, N_frag, states_per_frag, H)
