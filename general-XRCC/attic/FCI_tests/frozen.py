
# 16 Eh*a0 / 4.5 A   =   1.881518963875555
#
# ATOMIC
#
# biorthogonal semi-MO frozen-core FCI energy =  -14.569073392630342
#                   MO frozen-core FCI energy =  -14.569073392630335
#
# DIMER 9A
#
# biorthogonal semi-MO frozen-core FCI energy =  -30.078906267422624
#                   MO frozen-core FCI energy =  -30.078906267422695
#
# DIMER 4.5 A
#
# biorthogonal semi-MO frozen-core FCI energy =  -31.01922876942379
#                   MO frozen-core FCI energy =  -31.0192331315643
#
# TRIMER 4.5 A
#
# biorthogonal semi-MO frozen-core FCI energy =  -48.41014512985866
#                   MO frozen-core FCI energy =  -48.41015405487584



atomic  = -14.569073392630335
dimer90 = -30.078906267422695
dimer45 = -31.0192331315643
trimer  = -48.41015405487584

dimer90 += 1.881518963875555/2
dimer45 += 1.881518963875555
trimer  += 1.881518963875555 + 1.881518963875555 + 1.881518963875555/2

Ddimer90 = dimer90 - 2*atomic
Ddimer45 = dimer45 - 2*atomic
Dtrimer  = trimer  - 3*atomic

print()
print(Dtrimer,    Ddimer45,  Ddimer90)
print(Dtrimer - 2*Ddimer45 - Ddimer90)



atomic  = -14.569073392630342
dimer90 = -30.078906267422624
dimer45 = -31.01922876942379
trimer  = -48.41014512985866

dimer90 += 1.881518963875555/2
dimer45 += 1.881518963875555
trimer  += 1.881518963875555 + 1.881518963875555 + 1.881518963875555/2

Ddimer90 = dimer90 - 2*atomic
Ddimer45 = dimer45 - 2*atomic
Dtrimer  = trimer  - 3*atomic

print()
print(Dtrimer,    Ddimer45,  Ddimer90)
print(Dtrimer - 2*Ddimer45 - Ddimer90)

print()
