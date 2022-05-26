from ftl_guest import FTLGuest
from ftl_param import FTLParam

g_p = {
    "partner_addr": ("127.0.0.1", 1235),
    "role": "guest",
    "data_path": "data/mini_nus_wide_train_guest.csv"
}
guest_param = FTLParam(**g_p)

guest_ftl = FTLGuest(guest_param)
guest_ftl.compute_guest_components()

# rcv = guest_ftl.rcv()
# print(rcv)
# guest_ftl.send(b"hello host")
