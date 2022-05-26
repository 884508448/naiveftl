from ftl_host import FTLHost
from ftl_param import FTLParam

h_p = {
    "partner_addr": ("127.0.0.1", 1234),
    "role": "host",
    "data_path": "data/mini_nus_wide_train_host.csv"
}
host_param = FTLParam(**h_p)

host_ftl = FTLHost(host_param)
host_ftl.compute_host_components()

# host_ftl.send(b"hello guest")
# rcv = host_ftl.rcv()
# print(rcv)
