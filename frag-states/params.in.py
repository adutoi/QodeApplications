#n_threads = 1

basis = "6-31G"
dist  = 4.5
frags = [
    struct(
        atoms=[struct(element="Be", position=[0, 0, 0])],
        core=[0]
    ),
    struct(
        atoms=[struct(element="Be", position=[0, 0, dist])],
        core=[0]
    )
]

thresh    = 1e-6
#nstates   = 30
compress  = struct(method="SVD", divide="cc-aa")
#nat_orbs  = False
#abs_anti  = False

