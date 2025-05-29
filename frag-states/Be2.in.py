# qode/atoms/integrals/fragments.py
#        for atom in atoms:
#            element, (x,y,z) = atom  #("element position")    # atom is a struct
# strategy for now is just to make structs in order, so it is compatible with a tuple
# someday when the tuple-passing code is deprecated, we can implement the change
# above and it will not break struct-passing code, but then structs also will not be
# order-dependent
#
# def some_workhorse_function(<application-specific arguments>, archive=None, printout=print):
#     if archive is None:  archive = struct()
#     archive.textlog    = logger(printout)
#     ...
#     archive.member = <some data>
#     archive.textlog("some timely message")



n_threads = 4

basis = "6-31G"
frags = [
    struct(
        atoms=[struct(element="Be", position=[0, 0, 0])],    # promote to struct internally
        core=[0],    # needs ...
        charge=0     # ... defaulting mechanism
    ),
    struct(
        atoms=[struct(element="Be", position=[0, 0, 4.5])],
        core=[0],
        charge=0
    )
]

thresh    = 1e-6
#nstates   = 30

