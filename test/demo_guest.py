from ftl_guest import FTLGuest
from ftl_param import FTLParam

g_p = {"partner_addr": ("127.0.0.1", 1235), "role": "guest", "data_path": "data/mini_nus_wide_train_guest.csv",
       "nn_define": {
           "m_input_size": 9,
           "m_output_size": 9,
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
guest_param = FTLParam(**g_p)

guest_ftl = FTLGuest(guest_param)
guest_ftl.compute_guest_components()

# rcv = guest_ftl.rcv()
# print(rcv)
# guest_ftl.send(b"hello host")
