# qode/atoms/integrals/fragments.py
#        for atom in atoms:
#            element, (x,y,z) = atom  #("element position")    # atom is a struct
# strategy for now is just to make structs in order, so it is compatible with a tuple
# someday when the tuple-passing code is deprecated, we can implement the change
# above and it will not break struct-passing code, but then structs also will not be
# order-dependent



#n_threads = 1

basis = "6-31G"
dist  = 4.5
frags = [
    struct(
        atoms=[struct(element="Be", position=[0, 0, 0])],
        core=[0],         # needs ...
        charge=0,         # ... defaulting ...
        multiplicity=1    # ... mechanism
    ),
    struct(
        atoms=[struct(element="Be", position=[0, 0, dist])],
        core=[0],
        charge=0,
        multiplicity=1
    )
]

thresh    = 1e-6
#nstates   = 30
#compress  = struct(method="SVD", divide="cc-aa")
#nat_orbs  = False
#abs_anti  = False


# def some_workhorse_function(<application-specific arguments>, archive=None, printout=print):
#     if archive is None:  archive = struct()
#     archive.textlog    = logger(printout)
#     ...
#     archive.member = <some data>
#     archive.textlog("some timely message")
