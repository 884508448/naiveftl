from ftl_host import FTLHost
from ftl_param import FTLParam

h_p = {"partner_addr": ("127.0.0.1", 1234), "role": "host", "data_path": "data/mini_nus_wide_train_host.csv",
       "nn_define": {
           "m_input_size": 9,
           "m_layer_params": [
               {
                   "m_layer_size": 8,
                   "m_activation": "ReLU"
               },
               {
                   "m_layer_size": 8,
                   "m_activation": "ReLU"
               },
           ],

           "m_optimizer_params": {
               "m_type": "Adam"
           },
           "m_loss_func_params": {
               "m_type": None,
           },
       }
       }
host_param = FTLParam(**h_p)

host_ftl = FTLHost(host_param)
host_ftl.compute_host_components()

# host_ftl.send(b"hello guest")
# rcv = host_ftl.rcv()
# print(rcv)
