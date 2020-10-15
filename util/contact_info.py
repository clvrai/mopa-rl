def str_mj_arr(arr):
    return " ".join(["%0.3f" % arr[i] for i in range(arr.shape[0])])


def print_contact_info(sim):
    if sim.data.ncon == 0:
        print("No contacts/collisions")
        return

    # Print contact metadata
    for coni in range(sim.data.ncon):
        print("  Contact %d:" % (coni,))
        con = sim.data.contact[coni]
        print("    dist     = %0.3f" % (con.dist,))
        print("    pos      = %s" % (str_mj_arr(con.pos),))
        print("    frame    = %s" % (str_mj_arr(con.frame),))
        print("    friction = %s" % (str_mj_arr(con.friction),))
        print("    dim      = %d" % (con.dim,))
        print("    geom1    = %d, %s" % (con.geom1, sim.model.geom_id2name(con.geom1)))
        print("    geom2    = %d, %s" % (con.geom2, sim.model.geom_id2name(con.geom2)))
